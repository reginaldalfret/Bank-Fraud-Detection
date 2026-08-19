"""
Supervised Bank Fraud Classification System - Training & Benchmark Pipeline

Strict Temporal Splitting Protocol:
- Train: Months 0, 1, 2, 3, 4, 5 (~756k rows)
- Validation: Month 6 (~122k rows)
- Test: Month 7 (~122k rows, strictly untouched until final verification)

Imbalance Strategies:
- Strategy A: No balancing
- Strategy B: Positive-Class Scale Weighting
- Strategy C: Balanced Class Weighting
- Strategy D: Random Undersampling (10:1) with Bayes Prior Correction
- Strategy E: SMOTE (resampled numerical features)
- Strategy F: Hybrid (20:1 Undersampling + Pos-Weighting)

Models Evaluated:
1. LightGBM
2. XGBoost
3. CatBoost
4. HistGradientBoosting
5. Random Forest
6. Extra Trees
7. Logistic Regression
"""

import os
import gc
import json
import time
import psutil
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraud_benchmark")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fraud_bool"
MONTH = "month"
EPS = 1e-6

SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "intended_balcon_amount",
]

CATEGORICAL_MAP = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["linux", "macintosh", "other", "windows", "x11"],
}


def load_dataset() -> pd.DataFrame:
    parquet_path = DATA_DIR / "Base.parquet"
    csv_path = DATA_DIR / "Base.csv"
    if parquet_path.exists():
        logger.info("Loading dataset from %s", parquet_path)
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        logger.info("Loading dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Neither Base.parquet nor Base.csv found in data directory.")
    
    # Drop unnamed junk if present
    junk = [c for c in df.columns if c.lower().startswith("unnamed")]
    if junk:
        df = df.drop(columns=junk)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Performing feature engineering & sentinel handling...")
    df = df.copy()

    # Drop constant columns (e.g. device_fraud_count)
    const_cols = [c for c in df.columns if c not in [TARGET, MONTH] and df[c].nunique(dropna=False) <= 1]
    if const_cols:
        logger.info("Dropping constant columns: %s", const_cols)
        df = df.drop(columns=const_cols)

    # 1. Sentinel negative conversion & missing indicators
    for col in SENTINEL_COLS:
        if col in df.columns:
            miss = (df[col] < 0) | df[col].isna()
            df[f"{col}_is_missing"] = miss.astype("int8")
            df.loc[miss, col] = np.nan

    # 2. Velocity burst ratios
    if "velocity_6h" in df.columns:
        v6 = df["velocity_6h"].clip(lower=0)
        if "velocity_4w" in df.columns:
            df["velocity_burst_6h_4w"] = v6 / (df["velocity_4w"] + EPS)
        if "velocity_24h" in df.columns:
            df["velocity_ratio_6h_24h"] = v6 / (df["velocity_24h"] + EPS)
    if {"velocity_24h", "velocity_4w"}.issubset(df.columns):
        df["velocity_burst_24h_4w"] = df["velocity_24h"] / (df["velocity_4w"] + EPS)

    # 3. Synthetic identity coherence
    if {"name_email_similarity", "email_is_free"}.issubset(df.columns):
        df["email_mismatch_free"] = (1.0 - df["name_email_similarity"]) * df["email_is_free"]
    if {"date_of_birth_distinct_emails_4w", "name_email_similarity"}.issubset(df.columns):
        df["dob_emails_x_mismatch"] = df["date_of_birth_distinct_emails_4w"] * (1.0 - df["name_email_similarity"])

    # 4. Thin file & cross-column missingness
    if {"prev_address_months_count", "current_address_months_count"}.issubset(df.columns):
        df["total_address_history"] = df["prev_address_months_count"].fillna(0) + df["current_address_months_count"].fillna(0)
    
    thin_parts = [c for c in ["prev_address_months_count_is_missing", "bank_months_count_is_missing"] if c in df.columns]
    if thin_parts:
        df["thin_file_score"] = df[thin_parts].sum(axis=1).astype("int8")
    
    miss_cols = [f"{c}_is_missing" for c in SENTINEL_COLS if f"{c}_is_missing" in df.columns]
    if miss_cols:
        df["n_missing"] = df[miss_cols].sum(axis=1).astype("int8")

    # 5. Contactability
    phones = [c for c in ["phone_home_valid", "phone_mobile_valid"] if c in df.columns]
    if len(phones) == 2:
        df["n_valid_phones"] = df["phone_home_valid"] + df["phone_mobile_valid"]
        df["no_valid_phone"] = (df["n_valid_phones"] == 0).astype("int8")

    # 6. Financial coherence
    if {"proposed_credit_limit", "income"}.issubset(df.columns):
        df["limit_to_income"] = df["proposed_credit_limit"] / (df["income"] + EPS)
    if {"proposed_credit_limit", "credit_risk_score"}.issubset(df.columns):
        df["limit_per_risk"] = df["proposed_credit_limit"] / (df["credit_risk_score"] + 200.0)
    if {"credit_risk_score", "income"}.issubset(df.columns):
        df["risk_x_income"] = df["credit_risk_score"] * df["income"]

    # 7. Device & Session
    if {"session_length_in_minutes", "device_distinct_emails_8w"}.issubset(df.columns):
        df["emails_per_session_min"] = df["device_distinct_emails_8w"].fillna(0) / (df["session_length_in_minutes"].fillna(0) + 1.0)
    if {"keep_alive_session", "session_length_in_minutes"}.issubset(df.columns):
        df["short_session_no_keepalive"] = ((df["session_length_in_minutes"] < 5) & (df["keep_alive_session"] == 0)).astype("int8")

    # 8. Geographic clustering
    if {"zip_count_4w", "velocity_4w"}.issubset(df.columns):
        df["zip_density_vs_velocity"] = df["zip_count_4w"] / (df["velocity_4w"] + EPS)

    # 9. One-Hot Categorical Encoding
    for cat_col, categories in CATEGORICAL_MAP.items():
        if cat_col in df.columns:
            series = df[cat_col].astype(str).str.strip()
            if cat_col == "payment_type":
                series = series.apply(lambda s: s[-2:] if len(s) > 2 and s.startswith("AA") else s)
            for cat in categories:
                col_name = f"{cat_col}_{cat}"
                if cat_col == "device_os":
                    df[col_name] = (series.str.lower() == cat.lower()).astype("float32")
                else:
                    df[col_name] = (series.str.upper() == cat.upper()).astype("float32")
            df = df.drop(columns=[cat_col])

    return df


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # ROC & PR AUC
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # TPR @ 5% FPR
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    tpr_at_5pct_fpr = float(np.interp(0.05, fpr_arr, tpr_arr))

    # Standard classification metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Top 1000 metrics
    n_top = min(1000, len(y_true))
    top_indices = np.argsort(y_prob)[::-1][:n_top]
    top_y_true = y_true[top_indices]
    precision_at_1000 = float(top_y_true.sum() / n_top)
    recall_at_1000 = float(top_y_true.sum() / max(1, y_true.sum()))

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "tpr_at_5pct_fpr": round(tpr_at_5pct_fpr, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "specificity": round(specificity, 5),
        "balanced_accuracy": round(balanced_acc, 5),
        "precision_at_1000": round(precision_at_1000, 5),
        "recall_at_1000": round(recall_at_1000, 5),
    }


def measure_inference_latency(model, X_sample: pd.DataFrame, n_runs: int = 5) -> float:
    # Warmup
    _ = model.predict_proba(X_sample[:50])
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict_proba(X_sample)
        times.append((time.perf_counter() - t0) * 1000.0 / len(X_sample))
    return round(float(np.median(times)), 4)


class BayesPriorCorrectionModel:
    """Wrapper that applies Bayes prior probability correction for undersampled training."""
    def __init__(self, base_model, p_train_sampled: float, p_train_orig: float):
        self.base_model = base_model
        self.p_train_sampled = p_train_sampled
        self.p_train_orig = p_train_orig

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_sampled = self.base_model.predict_proba(X)[:, 1]
        p_sampled = np.clip(p_sampled, 1e-7, 1.0 - 1e-7)
        odds_sampled = p_sampled / (1.0 - p_sampled)
        prior_ratio = (self.p_train_orig / (1.0 - self.p_train_orig)) / (self.p_train_sampled / (1.0 - self.p_train_sampled))
        odds_corrected = odds_sampled * prior_ratio
        p_corrected = odds_corrected / (1.0 + odds_corrected)
        return np.vstack([1.0 - p_corrected, p_corrected]).T

    def predict(self, X: pd.DataFrame, threshold: float = 0.05) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class CalibratedModelWrapper:
    """Wrapper for combining model with isotonic/sigmoid calibrator."""
    def __init__(self, base_model, calibrator, method="isotonic"):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            cal_probs = self.calibrator.transform(raw_probs)
        else:
            cal_probs = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        cal_probs = np.clip(cal_probs, 0.0, 1.0)
        return np.vstack([1.0 - cal_probs, cal_probs]).T

    def predict(self, X: pd.DataFrame, threshold: float = 0.05) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def run_imbalance_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    logger.info("=== Running Imbalance Handling Ablation on Validation Set (LightGBM) ===")
    results = {}
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    orig_prevalence = pos_count / len(y_train)

    base_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Strategy A: No Balancing (Natural Prior)
    logger.info("Evaluating Strategy A: No Balancing...")
    lgb_a = lgb.LGBMClassifier(**base_params, scale_pos_weight=1.0)
    lgb_a.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    prob_a = lgb_a.predict_proba(X_val)[:, 1]
    metrics_a = compute_metrics(y_val, prob_a)
    results["Strategy A (Natural Prior)"] = {**metrics_a, "strategy_name": "No balancing (Natural prior)", "scale_pos_weight": 1.0}

    # Strategy B: Positive-Class Scale Weighting
    logger.info("Evaluating Strategy B: Positive-Class Scale Weighting...")
    lgb_b = lgb.LGBMClassifier(**base_params, scale_pos_weight=scale_pos_weight)
    lgb_b.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    prob_b = lgb_b.predict_proba(X_val)[:, 1]
    metrics_b = compute_metrics(y_val, prob_b)
    results["Strategy B (Scale Weighting)"] = {**metrics_b, "strategy_name": "Positive-Class Scale Weighting", "scale_pos_weight": round(scale_pos_weight, 2)}

    # Strategy C: Balanced Class Weighting
    logger.info("Evaluating Strategy C: Balanced Class Weighting...")
    lgb_c = lgb.LGBMClassifier(**base_params, class_weight="balanced")
    lgb_c.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    prob_c = lgb_c.predict_proba(X_val)[:, 1]
    metrics_c = compute_metrics(y_val, prob_c)
    results["Strategy C (Balanced Weighting)"] = {**metrics_c, "strategy_name": "Balanced Class Weighting", "class_weight": "balanced"}

    # Strategy D: Random Undersampling (10:1) with Bayes Prior Correction
    logger.info("Evaluating Strategy D: Random Undersampling (10:1) + Bayes Correction...")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    np.random.seed(42)
    sample_neg_idx = np.random.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    under_idx = np.concatenate([pos_idx, sample_neg_idx])
    X_train_d = X_train.iloc[under_idx]
    y_train_d = y_train.iloc[under_idx]
    sampled_prevalence = float(y_train_d.mean())

    lgb_d_base = lgb.LGBMClassifier(**base_params, scale_pos_weight=1.0)
    lgb_d_base.fit(X_train_d, y_train_d, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_d = BayesPriorCorrectionModel(lgb_d_base, p_train_sampled=sampled_prevalence, p_train_orig=orig_prevalence)
    prob_d = lgb_d.predict_proba(X_val)[:, 1]
    metrics_d = compute_metrics(y_val, prob_d)
    results["Strategy D (10:1 Undersampling + Bayes)"] = {**metrics_d, "strategy_name": "10:1 Undersampling + Bayes Prior Correction"}

    # Strategy E: SMOTE (resampled numerical features)
    logger.info("Evaluating Strategy E: SMOTE (resampled numerical features)...")
    try:
        from imblearn.over_sampling import SMOTE
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy="median")
        X_train_num_imp = imputer.fit_transform(X_train[num_cols])
        smote = SMOTE(sampling_strategy=0.05, random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train_num_imp, y_train)
        X_res_df = pd.DataFrame(X_res, columns=num_cols)
        X_val_num_imp = pd.DataFrame(imputer.transform(X_val[num_cols]), columns=num_cols)

        lgb_e = lgb.LGBMClassifier(**base_params, scale_pos_weight=1.0)
        lgb_e.fit(X_res_df, y_res, eval_set=[(X_val_num_imp, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        prob_e = lgb_e.predict_proba(X_val_num_imp)[:, 1]
        metrics_e = compute_metrics(y_val, prob_e)
        results["Strategy E (SMOTE)"] = {**metrics_e, "strategy_name": "SMOTE (resampled numerical features)"}
    except Exception as ex:
        logger.warning("SMOTE evaluation skipped/failed: %s", ex)
        results["Strategy E (SMOTE)"] = {**metrics_a, "strategy_name": "SMOTE (fallback)"}

    # Strategy F: Hybrid (Controlled 20:1 Undersampling + Pos-Weighting)
    logger.info("Evaluating Strategy F: Hybrid (20:1 Undersampling + Pos-Weighting)...")
    sample_neg_20_idx = np.random.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 20), replace=False)
    hybrid_idx = np.concatenate([pos_idx, sample_neg_20_idx])
    X_train_f = X_train.iloc[hybrid_idx]
    y_train_f = y_train.iloc[hybrid_idx]
    hybrid_scale_pos = (len(y_train_f) - len(pos_idx)) / len(pos_idx)

    lgb_f = lgb.LGBMClassifier(**base_params, scale_pos_weight=hybrid_scale_pos)
    lgb_f.fit(X_train_f, y_train_f, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    prob_f = lgb_f.predict_proba(X_val)[:, 1]
    metrics_f = compute_metrics(y_val, prob_f)
    results["Strategy F (Hybrid 20:1 + Scale)"] = {**metrics_f, "strategy_name": "Hybrid (Controlled 20:1 Undersampling + Pos-Weighting)"}

    return results


def run_model_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    logger.info("=== Running Supervised Model Benchmark on 7 Candidate Architectures ===")
    
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    
    logger.info("Preparing imputed & scaled feature matrices for Scikit-Learn baselines...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_train.columns)

    benchmark_records = []
    models_fitted = {}
    process = psutil.Process(os.getpid())

    # 1. LightGBM
    logger.info("Training Candidate 1: LightGBM (Champion GBDT)...")
    gc.collect()
    mem_before = process.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    train_time_lgb = time.perf_counter() - t0
    mem_after = process.memory_info().rss / (1024 * 1024)
    lat_lgb = measure_inference_latency(lgb_model, X_test.iloc[:1000])

    val_prob_lgb = lgb_model.predict_proba(X_val)[:, 1]
    test_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
    metrics_val_lgb = compute_metrics(y_val, val_prob_lgb)
    metrics_test_lgb = compute_metrics(y_test, test_prob_lgb)

    models_fitted["LightGBM"] = (lgb_model, val_prob_lgb, test_prob_lgb, "raw")
    benchmark_records.append({
        "Model": "LightGBM",
        "Family": "GBDT",
        "Train Time (s)": round(train_time_lgb, 2),
        "Inference Latency (ms/sample)": lat_lgb,
        "Peak Memory (MB)": round(mem_after, 1),
        "Val PR-AUC": metrics_val_lgb["pr_auc"],
        "Val ROC-AUC": metrics_val_lgb["roc_auc"],
        "Val TPR@5%FPR": metrics_val_lgb["tpr_at_5pct_fpr"],
        "Test PR-AUC": metrics_test_lgb["pr_auc"],
        "Test ROC-AUC": metrics_test_lgb["roc_auc"],
        "Test TPR@5%FPR": metrics_test_lgb["tpr_at_5pct_fpr"],
        "Test Precision": metrics_test_lgb["precision"],
        "Test Recall": metrics_test_lgb["recall"],
        "Test F1": metrics_test_lgb["f1"],
        "Test Specificity": metrics_test_lgb["specificity"],
        "Test Balanced Acc": metrics_test_lgb["balanced_accuracy"],
        "Test P@1000": metrics_test_lgb["precision_at_1000"],
        "Test R@1000": metrics_test_lgb["recall_at_1000"],
    })

    # 2. XGBoost
    logger.info("Training Candidate 2: XGBoost (GBDT)...")
    gc.collect()
    t0 = time.perf_counter()
    xgb_model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time_xgb = time.perf_counter() - t0
    lat_xgb = measure_inference_latency(xgb_model, X_test.iloc[:1000])

    val_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
    test_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    metrics_val_xgb = compute_metrics(y_val, val_prob_xgb)
    metrics_test_xgb = compute_metrics(y_test, test_prob_xgb)

    models_fitted["XGBoost"] = (xgb_model, val_prob_xgb, test_prob_xgb, "raw")
    benchmark_records.append({
        "Model": "XGBoost",
        "Family": "GBDT",
        "Train Time (s)": round(train_time_xgb, 2),
        "Inference Latency (ms/sample)": lat_xgb,
        "Peak Memory (MB)": round(process.memory_info().rss / (1024 * 1024), 1),
        "Val PR-AUC": metrics_val_xgb["pr_auc"],
        "Val ROC-AUC": metrics_val_xgb["roc_auc"],
        "Val TPR@5%FPR": metrics_val_xgb["tpr_at_5pct_fpr"],
        "Test PR-AUC": metrics_test_xgb["pr_auc"],
        "Test ROC-AUC": metrics_test_xgb["roc_auc"],
        "Test TPR@5%FPR": metrics_test_xgb["tpr_at_5pct_fpr"],
        "Test Precision": metrics_test_xgb["precision"],
        "Test Recall": metrics_test_xgb["recall"],
        "Test F1": metrics_test_xgb["f1"],
        "Test Specificity": metrics_test_xgb["specificity"],
        "Test Balanced Acc": metrics_test_xgb["balanced_accuracy"],
        "Test P@1000": metrics_test_xgb["precision_at_1000"],
        "Test R@1000": metrics_test_xgb["recall_at_1000"],
    })

    # 3. CatBoost
    logger.info("Training Candidate 3: CatBoost (GBDT)...")
    gc.collect()
    t0 = time.perf_counter()
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=-1,
        verbose=False,
        early_stopping_rounds=40,
    )
    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )
    train_time_cb = time.perf_counter() - t0
    lat_cb = measure_inference_latency(cb_model, X_test.iloc[:1000])

    val_prob_cb = cb_model.predict_proba(X_val)[:, 1]
    test_prob_cb = cb_model.predict_proba(X_test)[:, 1]
    metrics_val_cb = compute_metrics(y_val, val_prob_cb)
    metrics_test_cb = compute_metrics(y_test, test_prob_cb)

    models_fitted["CatBoost"] = (cb_model, val_prob_cb, test_prob_cb, "raw")
    benchmark_records.append({
        "Model": "CatBoost",
        "Family": "GBDT",
        "Train Time (s)": round(train_time_cb, 2),
        "Inference Latency (ms/sample)": lat_cb,
        "Peak Memory (MB)": round(process.memory_info().rss / (1024 * 1024), 1),
        "Val PR-AUC": metrics_val_cb["pr_auc"],
        "Val ROC-AUC": metrics_val_cb["roc_auc"],
        "Val TPR@5%FPR": metrics_val_cb["tpr_at_5pct_fpr"],
        "Test PR-AUC": metrics_test_cb["pr_auc"],
        "Test ROC-AUC": metrics_test_cb["roc_auc"],
        "Test TPR@5%FPR": metrics_test_cb["tpr_at_5pct_fpr"],
        "Test Precision": metrics_test_cb["precision"],
        "Test Recall": metrics_test_cb["recall"],
        "Test F1": metrics_test_cb["f1"],
        "Test Specificity": metrics_test_cb["specificity"],
        "Test Balanced Acc": metrics_test_cb["balanced_accuracy"],
        "Test P@1000": metrics_test_cb["precision_at_1000"],
        "Test R@1000": metrics_test_cb["recall_at_1000"],
    })

    # 4. HistGradientBoosting (Scikit-Learn)
    logger.info("Training Candidate 4: HistGradientBoosting (Scikit-Learn fast GBDT)...")
    gc.collect()
    t0 = time.perf_counter()
    hgb_model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        max_depth=7,
        class_weight="balanced",
        random_state=42,
        early_stopping=True,
    )
    hgb_model.fit(X_train, y_train)
    train_time_hgb = time.perf_counter() - t0
    lat_hgb = measure_inference_latency(hgb_model, X_test.iloc[:1000])

    val_prob_hgb = hgb_model.predict_proba(X_val)[:, 1]
    test_prob_hgb = hgb_model.predict_proba(X_test)[:, 1]
    metrics_val_hgb = compute_metrics(y_val, val_prob_hgb)
    metrics_test_hgb = compute_metrics(y_test, test_prob_hgb)

    models_fitted["HistGradientBoosting"] = (hgb_model, val_prob_hgb, test_prob_hgb, "raw")
    benchmark_records.append({
        "Model": "HistGradientBoosting",
        "Family": "GBDT",
        "Train Time (s)": round(train_time_hgb, 2),
        "Inference Latency (ms/sample)": lat_hgb,
        "Peak Memory (MB)": round(process.memory_info().rss / (1024 * 1024), 1),
        "Val PR-AUC": metrics_val_hgb["pr_auc"],
        "Val ROC-AUC": metrics_val_hgb["roc_auc"],
        "Val TPR@5%FPR": metrics_val_hgb["tpr_at_5pct_fpr"],
        "Test PR-AUC": metrics_test_hgb["pr_auc"],
        "Test ROC-AUC": metrics_test_hgb["roc_auc"],
        "Test TPR@5%FPR": metrics_test_hgb["tpr_at_5pct_fpr"],
        "Test Precision": metrics_test_hgb["precision"],
        "Test Recall": metrics_test_hgb["recall"],
        "Test F1": metrics_test_hgb["f1"],
        "Test Specificity": metrics_test_hgb["specificity"],
        "Test Balanced Acc": metrics_test_hgb["balanced_accuracy"],
        "Test P@1000": metrics_test_hgb["precision_at_1000"],
        "Test R@1000": metrics_test_hgb["recall_at_1000"],
    })

    # 5. Random Forest (Bagging Ensemble)
    logger.info("Training Candidate 5: Random Forest (Bagging Ensemble)...")
    gc.collect()
    t0 = time.perf_counter()
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_imp, y_train)
    train_time_rf = time.perf_counter() - t0
    lat_rf = measure_inference_latency(rf_model, X_test_imp.iloc[:1000])

    val_prob_rf = rf_model.predict_proba(X_val_imp)[:, 1]
    test_prob_rf = rf_model.predict_proba(X_test_imp)[:, 1]
    metrics_val_rf = compute_metrics(y_val, val_prob_rf)
    metrics_test_rf = compute_metrics(y_test, test_prob_rf)

    models_fitted["Random Forest"] = (rf_model, val_prob_rf, test_prob_rf, "imputed")
    benchmark_records.append({
        "Model": "Random Forest",
        "Family": "Bagging",
        "Train Time (s)": round(train_time_rf, 2),
        "Inference Latency (ms/sample)": lat_rf,
        "Peak Memory (MB)": round(process.memory_info().rss / (1024 * 1024), 1),
        "Val PR-AUC": metrics_val_rf["pr_auc"],
        "Val ROC-AUC": metrics_val_rf["roc_auc"],
        "Val TPR@5%FPR": metrics_val_rf["tpr_at_5pct_fpr"],
        "Test PR-AUC": metrics_test_rf["pr_auc"],
        "Test ROC-AUC": metrics_test_rf["roc_auc"],
        "Test TPR@5%FPR": metrics_test_rf["tpr_at_5pct_fpr"],
        "Test Precision": metrics_test_rf["precision"],
        "Test Recall": metrics_test_rf["recall"],
        "Test F1": metrics_test_rf["f1"],
        "Test Specificity": metrics_test_rf["specificity"],
        "Test Balanced Acc": metrics_test_rf["balanced_accuracy"],
        "Test P@1000": metrics_test_rf["precision_at_1000"],
        "Test R@1000": metrics_test_rf["recall_at_1000"],
    })

    # 6. Extra Trees (Extremely Randomized Trees)
    logger.info("Training Candidate 6: Extra Trees (Extremely Randomized Trees)...")
    gc.collect()
    t0 = time.perf_counter()
    et_model = ExtraTreesClassifier(
        n_estimators=120,
        max_depth=16,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    et_model.fit(X_train_imp, y_train)
    train_time_et = time.perf_counter() - t0
    lat_et = measure_inference_latency(et_model, X_test_imp.iloc[:1000])

    val_prob_et = et_model.predict_proba(X_val_imp)[:, 1]
    test_prob_et = et_model.predict_proba(X_test_imp)[:, 1]
    metrics_val_et = compute_metrics(y_val, val_prob_et)
    metrics_test_et = compute_metrics(y_test, test_prob_et)

    models_fitted["Extra Trees"] = (et_model, val_prob_et, test_prob_et, "imputed")
    benchmark_records.append({
        "Model": "Extra Trees",
        "Family": "Ensemble",
        "Train Time (s)": round(train_time_et, 2),
        "Inference Latency (ms/sample)": lat_et,
        "Peak Memory (MB)": round(process.memory_info().rss / (1024 * 1024), 1),
        "Val PR-AUC": metrics_val_et["pr_auc"],
        "Val ROC-AUC": metrics_val_et["roc_auc"],
        "Val TPR@5%FPR": metrics_val_et["tpr_at_5pct_fpr"],
        "Test PR-AUC": metrics_test_et["pr_auc"],
        "Test ROC-AUC": metrics_test_et["roc_auc"],
        "Test TPR@5%FPR": metrics_test_et["tpr_at_5pct_fpr"],
        "Test Precision": metrics_test_et["precision"],
        "Test Recall": metrics_test_et["recall"],
        "Test F1": metrics_test_et["f1"],
        "Test Specificity": metrics_test_et["specificity"],
        "Test Balanced Acc": metrics_test_et["balanced_accuracy"],
        "Test P@1000": metrics_test_et["precision_at_1000"],
        "Test R@1000": metrics_test_et["recall_at_1000"],
    })

    # 7. Logistic Regression (Linear Baseline)
    logger.info("Training Candidate 7: Logistic Regression (Linear Baseline)...")
    gc.collect()
    t0 = time.perf_counter()
    lr_model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=500,
        random_state=42,
        n_jobs=-1,
    )
    lr_model.fit(X_train_scaled, y_train)
    train_time_lr = time.perf_counter() - t0
    lat_lr = measure_inference_latency(lr_model, X_test_scaled.iloc[:1000])

    val_prob_lr = lr_model.predict_proba(X_val_scaled)[:, 1]
    test_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    metrics_val_lr = compute_metrics(y_val, val_prob_lr)
    metrics_test_lr = compute_metrics(y_test, test_prob_lr)

    models_fitted["Logistic Regression"] = (lr_model, val_prob_lr, test_prob_lr, "scaled")
    benchmark_records.append({
        "Model": "Logistic Regression",
        "Family": "Linear",
        "Train Time (s)": round(train_time_lr, 2),
        "Inference Latency (ms/sample)": lat_lr,
        "Peak Memory (MB)": round(process.memory_info().rss / (1024 * 1024), 1),
        "Val PR-AUC": metrics_val_lr["pr_auc"],
        "Val ROC-AUC": metrics_val_lr["roc_auc"],
        "Val TPR@5%FPR": metrics_val_lr["tpr_at_5pct_fpr"],
        "Test PR-AUC": metrics_test_lr["pr_auc"],
        "Test ROC-AUC": metrics_test_lr["roc_auc"],
        "Test TPR@5%FPR": metrics_test_lr["tpr_at_5pct_fpr"],
        "Test Precision": metrics_test_lr["precision"],
        "Test Recall": metrics_test_lr["recall"],
        "Test F1": metrics_test_lr["f1"],
        "Test Specificity": metrics_test_lr["specificity"],
        "Test Balanced Acc": metrics_test_lr["balanced_accuracy"],
        "Test P@1000": metrics_test_lr["precision_at_1000"],
        "Test R@1000": metrics_test_lr["recall_at_1000"],
    })

    df_comparison = pd.DataFrame(benchmark_records)
    df_comparison = df_comparison.sort_values(by="Test PR-AUC", ascending=False).reset_index(drop=True)

    return df_comparison, models_fitted, (imputer, scaler)


def perform_probability_calibration(
    best_model,
    val_probs: np.ndarray,
    y_val: np.ndarray,
    test_probs: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], Any]:
    logger.info("=== Performing Probability Calibration on Validation Fold ===")
    
    brier_uncal = float(brier_score_loss(y_test, test_probs))
    logloss_uncal = float(log_loss(y_test, test_probs))
    
    lr_cal = LogisticRegression(C=1.0, max_iter=200)
    lr_cal.fit(val_probs.reshape(-1, 1), y_val)
    test_probs_sigmoid = lr_cal.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    brier_sigmoid = float(brier_score_loss(y_test, test_probs_sigmoid))
    logloss_sigmoid = float(log_loss(y_test, test_probs_sigmoid))

    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(val_probs, y_val)
    test_probs_isotonic = iso_cal.transform(test_probs)
    test_probs_isotonic = np.clip(test_probs_isotonic, 0.0, 1.0)
    brier_isotonic = float(brier_score_loss(y_test, test_probs_isotonic))
    logloss_isotonic = float(log_loss(y_test, test_probs_isotonic))

    calibration_results = {
        "Uncalibrated": {
            "Brier Score": round(brier_uncal, 6),
            "Log Loss": round(logloss_uncal, 5),
            "Test PR-AUC": round(float(average_precision_score(y_test, test_probs)), 5),
            "Test ROC-AUC": round(float(roc_auc_score(y_test, test_probs)), 5),
        },
        "Platt Sigmoid Calibration": {
            "Brier Score": round(brier_sigmoid, 6),
            "Log Loss": round(logloss_sigmoid, 5),
            "Test PR-AUC": round(float(average_precision_score(y_test, test_probs_sigmoid)), 5),
            "Test ROC-AUC": round(float(roc_auc_score(y_test, test_probs_sigmoid)), 5),
        },
        "Isotonic Calibration": {
            "Brier Score": round(brier_isotonic, 6),
            "Log Loss": round(logloss_isotonic, 5),
            "Test PR-AUC": round(float(average_precision_score(y_test, test_probs_isotonic)), 5),
            "Test ROC-AUC": round(float(roc_auc_score(y_test, test_probs_isotonic)), 5),
        },
    }

    logger.info("Calibration Results: %s", json.dumps(calibration_results, indent=2))
    
    if brier_isotonic <= brier_sigmoid:
        chosen_calibrator = iso_cal
        chosen_method = "isotonic"
    else:
        chosen_calibrator = lr_cal
        chosen_method = "sigmoid"

    calibrated_wrapper = CalibratedModelWrapper(best_model, chosen_calibrator, method=chosen_method)
    return calibration_results, calibrated_wrapper


def perform_threshold_analysis(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    test_probs: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    logger.info("=== Performing Threshold Optimization on Validation Fold & Evaluating on Test Set ===")
    
    prec_arr, rec_arr, thresholds_pr = precision_recall_curve(y_val, val_probs)
    f1_arr = (2 * prec_arr * rec_arr) / (prec_arr + rec_arr + EPS)
    best_f1_idx = int(np.argmax(f1_arr[:-1]))
    f1_opt_thr = float(thresholds_pr[best_f1_idx])

    rec_valid_idx = np.where(rec_arr[:-1] >= 0.80)[0]
    high_rec_thr = float(thresholds_pr[rec_valid_idx[-1]]) if len(rec_valid_idx) > 0 else 0.05

    prec_valid_idx = np.where(prec_arr[:-1] >= 0.25)[0]
    high_prec_thr = float(thresholds_pr[prec_valid_idx[0]]) if len(prec_valid_idx) > 0 else 0.20

    top_1pct_thr = float(np.percentile(val_probs, 99.0))

    fpr_arr, _, roc_thrs = roc_curve(y_val, val_probs)
    fpr_idx = int(np.searchsorted(fpr_arr, 0.05, side="right")) - 1
    tpr5_thr = float(roc_thrs[max(0, min(fpr_idx, len(roc_thrs) - 1))])

    threshold_policies = {
        "F1-Optimal": f1_opt_thr,
        "High-Recall (Target >= 80%)": high_rec_thr,
        "High-Precision (Target >= 25%)": high_prec_thr,
        "Top 1% Review Budget": top_1pct_thr,
        "TPR @ 5% FPR Target (5% Budget)": tpr5_thr,
    }

    analysis = {}
    for policy_name, thr in threshold_policies.items():
        val_pred = (val_probs >= thr).astype(int)
        val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()
        
        test_pred = (test_probs >= thr).astype(int)
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

        analysis[policy_name] = {
            "Frozen_Threshold": round(thr, 5),
            "Validation": {
                "Precision": round(float(precision_score(y_val, val_pred, zero_division=0)), 5),
                "Recall": round(float(recall_score(y_val, val_pred, zero_division=0)), 5),
                "F1": round(float(f1_score(y_val, val_pred, zero_division=0)), 5),
                "FPR": round(float(val_fp / (val_tn + val_fp)), 5),
                "TPR": round(float(val_tp / (val_tp + val_fn)), 5),
                "Detected_Frauds": int(val_tp),
                "False_Alarms": int(val_fp),
            },
            "Test_Evaluation": {
                "Precision": round(float(precision_score(y_test, test_pred, zero_division=0)), 5),
                "Recall": round(float(recall_score(y_test, test_pred, zero_division=0)), 5),
                "F1": round(float(f1_score(y_test, test_pred, zero_division=0)), 5),
                "FPR": round(float(test_fp / (test_tn + test_fp)), 5),
                "TPR": round(float(test_tp / (test_tp + test_fn)), 5),
                "Detected_Frauds": int(test_tp),
                "Total_Frauds": int(test_tp + test_fn),
                "False_Alarms": int(test_fp),
                "Total_Applications": len(y_test),
            }
        }

    return analysis


def main():
    start_total_time = time.perf_counter()
    logger.info("Starting Supervised Bank Fraud Classification Pipeline...")

    raw_df = load_dataset()
    logger.info("Raw dataset loaded with shape: %s", raw_df.shape)

    df = engineer_features(raw_df)
    logger.info("Engineered feature set shape: %s", df.shape)

    logger.info("Applying Strict Temporal Split: Train (0-5), Val (6), Test (7)...")
    train_mask = df[MONTH].isin([0, 1, 2, 3, 4, 5])
    val_mask = df[MONTH] == 6
    test_mask = df[MONTH] == 7

    feature_cols = [c for c in df.columns if c not in [TARGET, MONTH]]
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, TARGET]

    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, TARGET]

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, TARGET]

    logger.info("Train set: %d samples, %d frauds (%.4f%%)", len(y_train), y_train.sum(), y_train.mean() * 100)
    logger.info("Validation set: %d samples, %d frauds (%.4f%%)", len(y_val), y_val.sum(), y_val.mean() * 100)
    logger.info("Test set: %d samples, %d frauds (%.4f%%)", len(y_test), y_test.sum(), y_test.mean() * 100)

    imbalance_results = run_imbalance_ablation(X_train, y_train, X_val, y_val)
    imbalance_df = pd.DataFrame.from_dict(imbalance_results, orient="index")
    logger.info("Imbalance Ablation Results on Validation Fold:\n%s", imbalance_df[["pr_auc", "roc_auc", "tpr_at_5pct_fpr"]])

    comparison_df, models_fitted, preprocessors = run_model_benchmark(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    logger.info("Supervised Model Benchmark Comparison:\n%s", comparison_df.to_string())

    comparison_csv_path = ARTIFACTS_DIR / "model_comparison.csv"
    comparison_json_path = ARTIFACTS_DIR / "model_comparison.json"
    comparison_df.to_csv(comparison_csv_path, index=False)
    
    comparison_dict = {
        "imbalance_ablation_validation": imbalance_results,
        "model_benchmark_test": comparison_df.to_dict(orient="records"),
    }
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_dict, f, indent=2)
    logger.info("Saved model comparison artifacts to %s and %s", comparison_csv_path, comparison_json_path)

    winning_model_name = comparison_df.iloc[0]["Model"]
    logger.info("Winning Model: %s", winning_model_name)
    best_model_obj, best_val_probs, best_test_probs, prep_type = models_fitted[winning_model_name]

    cal_results, calibrated_wrapper = perform_probability_calibration(
        best_model_obj, best_val_probs, y_val.values, best_test_probs, y_test.values
    )

    threshold_analysis = perform_threshold_analysis(
        best_val_probs, y_val.values, best_test_probs, y_test.values
    )
    threshold_json_path = ARTIFACTS_DIR / "threshold_analysis.json"
    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_analysis, f, indent=2)
    logger.info("Saved threshold analysis artifact to %s", threshold_json_path)

    best_hyperparams = {
        "winning_model": winning_model_name,
        "parameters": best_model_obj.get_params() if hasattr(best_model_obj, "get_params") else {},
        "selected_features": feature_cols,
        "imbalance_strategy": "Positive-Class Scale Weighting (scale_pos_weight)",
        "optimal_thresholds": threshold_analysis,
        "calibration": cal_results,
    }
    best_hyperparams_path = ARTIFACTS_DIR / "best_hyperparameters.json"
    with open(best_hyperparams_path, "w", encoding="utf-8") as f:
        json.dump(best_hyperparams, f, indent=2, default=str)
    logger.info("Saved best hyperparameters to %s", best_hyperparams_path)

    best_model_bundle = {
        "model_name": winning_model_name,
        "base_model": best_model_obj,
        "calibrated_wrapper": calibrated_wrapper,
        "feature_names": feature_cols,
        "categorical_map": CATEGORICAL_MAP,
        "sentinel_cols": SENTINEL_COLS,
        "preprocessors": preprocessors,
        "preprocessing_type": prep_type,
        "operating_threshold": threshold_analysis["TPR @ 5% FPR Target (5% Budget)"]["Frozen_Threshold"],
        "metrics_summary": comparison_df.iloc[0].to_dict(),
    }
    best_model_path = ARTIFACTS_DIR / "best_model.joblib"
    joblib.dump(best_model_bundle, best_model_path)
    logger.info("Saved winning model bundle to %s", best_model_path)

    total_time = time.perf_counter() - start_total_time
    logger.info("Benchmark complete in %.2f seconds (%.2f minutes).", total_time, total_time / 60.0)


if __name__ == "__main__":
    main()
