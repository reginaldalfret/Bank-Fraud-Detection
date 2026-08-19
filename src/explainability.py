"""
src/explainability.py -- Explainability and Evidence Generation Engine for
Bank Account Opening Fraud Detection.

Domain: Bank Account Opening Fraud (applications, applicants, account opening).
Key Fraud Archetypes:
  1. Synthetic Identity: Fabricated profiles, thin address/banking history,
     low email-name coherence, disposable email domains.
  2. Identity Theft: Real victim credentials submitted from mismatched devices,
     invalidated phone contacts, anomalous session behavior.
  3. Mule Account Farming: Scripted bulk account openings, velocity bursts,
     shared birthdates/emails across applications.

Core Capabilities:
  - TreeSHAP Feature Attribution: Global feature importance rankings and local
    per-application waterfall breakdown (expected value, feature contributions,
    base margin to probability conversion).
  - Behavioral Deviation Analyzer: Benchmark applicant parameters (limit-to-income,
    credit risk score, velocity bursts, thin-file score, session length) against
    population distributions to flag statistical outliers and behavioral shifts.
  - Structured Evidence Builder: Consolidates model predictions, confidence levels,
    top SHAP positive/negative drivers, and heuristic risk flags into a unified,
    audit-ready applicant evidence package for downstream GenAI synthesis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("fraud_detection.explainability")

EPS = 1e-6


# =====================================================================
# Data Structures: Explainability & Evidence
# =====================================================================

@dataclass
class FeatureContributionItem:
    """Individual feature contribution within a local explanation waterfall."""
    feature: str
    display_name: str
    feature_value: Any
    shap_value: float
    contribution_direction: str  # 'INCREASES_FRAUD_RISK' or 'DECREASES_FRAUD_RISK'
    importance_rank: int
    magnitude: float
    archetype: str  # 'SYNTHETIC_IDENTITY', 'IDENTITY_THEFT', 'MULE_FARMING', 'CREDIT_RISK', 'GENERAL'
    domain_explanation: str


@dataclass
class LocalWaterfallExplanation:
    """Complete TreeSHAP waterfall explanation for a single account opening application."""
    application_id: Optional[str]
    base_value: float  # Base expected value (log-odds or probability)
    prediction_score: float  # Predicted fraud probability or raw margin
    top_fraud_drivers: List[FeatureContributionItem]  # Pushing towards fraud (positive SHAP)
    top_mitigating_factors: List[FeatureContributionItem]  # Pushing towards legitimate (negative SHAP)
    total_features_evaluated: int
    sum_shap_contributions: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application_id": self.application_id,
            "base_value": self.base_value,
            "prediction_score": self.prediction_score,
            "top_fraud_drivers": [asdict(item) for item in self.top_fraud_drivers],
            "top_mitigating_factors": [asdict(item) for item in self.top_mitigating_factors],
            "total_features_evaluated": self.total_features_evaluated,
            "sum_shap_contributions": self.sum_shap_contributions,
        }


@dataclass
class MetricDeviation:
    """Statistical deviation of an applicant metric against baseline population."""
    metric_name: str
    applicant_value: float
    population_mean: float
    population_median: float
    population_std: float
    z_score: float
    percentile_rank: float
    severity: str  # 'NORMAL', 'MODERATE_DEVIATION', 'EXTREME_DEVIATION', 'CRITICAL_ANOMALY'
    anomaly_direction: str  # 'HIGH', 'LOW', 'EXPECTED'
    description: str


@dataclass
class ApplicantEvidencePackage:
    """Consolidated evidence package for human analysts and GenAI reasoning engines."""
    application_id: str
    timestamp: str
    fraud_probability: float
    fraud_prediction: int
    decision_threshold: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    confidence_tier: str  # 'VERY_HIGH', 'HIGH', 'MODERATE', 'BORDERLINE'
    margin_from_threshold: float
    top_fraud_drivers: List[FeatureContributionItem]
    top_mitigating_factors: List[FeatureContributionItem]
    behavioral_deviations: List[MetricDeviation]
    triggered_risk_flags: List[str]
    applicant_summary_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application_id": self.application_id,
            "timestamp": self.timestamp,
            "fraud_probability": self.fraud_probability,
            "fraud_prediction": self.fraud_prediction,
            "decision_threshold": self.decision_threshold,
            "risk_level": self.risk_level,
            "confidence_tier": self.confidence_tier,
            "margin_from_threshold": self.margin_from_threshold,
            "top_fraud_drivers": [asdict(d) for d in self.top_fraud_drivers],
            "top_mitigating_factors": [asdict(m) for m in self.top_mitigating_factors],
            "behavioral_deviations": [asdict(b) for b in self.behavioral_deviations],
            "triggered_risk_flags": self.triggered_risk_flags,
            "applicant_summary_metrics": self.applicant_summary_metrics,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def format_prompt_context(self) -> str:
        """Formats the evidence package into a structured text prompt for LLM consumption."""
        drivers_text = "\n".join(
            f"  - {d.display_name} ({d.feature}): value={d.feature_value}, SHAP impact=+{d.shap_value:.4f} "
            f"[{d.archetype}] -> {d.domain_explanation}"
            for d in self.top_fraud_drivers
        ) or "  - None identified"

        mitigating_text = "\n".join(
            f"  - {m.display_name} ({m.feature}): value={m.feature_value}, SHAP impact={m.shap_value:.4f} "
            f"[{m.archetype}] -> {m.domain_explanation}"
            for m in self.top_mitigating_factors
        ) or "  - None identified"

        deviations_text = "\n".join(
            f"  - {dev.metric_name}: value={dev.applicant_value:.3f} "
            f"(z-score={dev.z_score:+.2f}, percentile={dev.percentile_rank:.1f}%) "
            f"[{dev.severity}] -> {dev.description}"
            for dev in self.behavioral_deviations
            if dev.severity != "NORMAL"
        ) or "  - All monitored metrics within normal population boundaries"

        flags_text = "\n".join(f"  - [FLAG] {flag}" for flag in self.triggered_risk_flags) or "  - No critical rule flags triggered"

        return f"""=== BANK ACCOUNT OPENING APPLICATION EVIDENCE ===
Application ID: {self.application_id}
Evaluation Timestamp: {self.timestamp}
Model Assessment:
  - Fraud Probability: {self.fraud_probability:.4f} (Threshold: {self.decision_threshold:.4f})
  - Decision: {"FLAGGED AS SUSPECTED FRAUD" if self.fraud_prediction == 1 else "CLEARED AS LEGITIMATE"}
  - Risk Level: {self.risk_level}
  - Model Confidence Tier: {self.confidence_tier} (Margin: {self.margin_from_threshold:+.4f})

Key Applicant Summary Metrics:
{json.dumps(self.applicant_summary_metrics, indent=4)}

Top Fraud Risk Drivers (TreeSHAP Positive Attribution):
{drivers_text}

Top Mitigating Factors (TreeSHAP Negative Attribution):
{mitigating_text}

Behavioral & Population Deviations:
{deviations_text}

Triggered Account-Opening Risk Flags:
{flags_text}
=================================================="""


# =====================================================================
# Domain Feature Taxonomy & Plain-English Explanations
# =====================================================================

FEATURE_TAXONOMY: Dict[str, Dict[str, str]] = {
    "income": {
        "display": "Applicant Income Decile",
        "archetype": "CREDIT_RISK",
        "desc": "Declared applicant income rank. Discrepancies with requested limit signal synthetic credit inflation.",
    },
    "customer_age": {
        "display": "Applicant Age (Decade)",
        "archetype": "GENERAL",
        "desc": "Applicant age rounded to decade (fairness-protected attribute).",
    },
    "name_email_similarity": {
        "display": "Name-to-Email Similarity",
        "archetype": "SYNTHETIC_IDENTITY",
        "desc": "String coherence between applicant name and email handle; low scores indicate automated email generation.",
    },
    "email_mismatch_free": {
        "display": "Synthetic Free Email Mismatch",
        "archetype": "SYNTHETIC_IDENTITY",
        "desc": "Interaction of low name-email similarity on disposable/free email domains.",
    },
    "prev_address_months_count": {
        "display": "Previous Address Tenure (Months)",
        "archetype": "SYNTHETIC_IDENTITY",
        "desc": "Months residing at previous address; missing/sentinel values indicate synthetic lack of physical footprint.",
    },
    "current_address_months_count": {
        "display": "Current Address Tenure (Months)",
        "archetype": "SYNTHETIC_IDENTITY",
        "desc": "Tenure at current residence; extremely short tenure coupled with thin credit signals high synthetic risk.",
    },
    "bank_months_count": {
        "display": "Prior Banking Tenure (Months)",
        "archetype": "SYNTHETIC_IDENTITY",
        "desc": "Duration of previous institutional banking relationship; absent history indicates thin-file synthetic profile.",
    },
    "thin_file_score": {
        "display": "Thin File Risk Score",
        "archetype": "SYNTHETIC_IDENTITY",
        "desc": "Composite score quantifying absent residential and institutional banking history.",
    },
    "credit_risk_score": {
        "display": "Bureau Credit Risk Score",
        "archetype": "CREDIT_RISK",
        "desc": "Internal credit bureau scoring metric. Highly negative or degraded scores strongly correlate with fraud risk.",
    },
    "proposed_credit_limit": {
        "display": "Requested Credit Limit",
        "archetype": "CREDIT_RISK",
        "desc": "Proposed credit facility limit requested during account opening.",
    },
    "limit_to_income": {
        "display": "Limit-to-Income Exposure Ratio",
        "archetype": "CREDIT_RISK",
        "desc": "Ratio of requested credit limit to income decile; excessive values indicate aggressive credit extraction.",
    },
    "velocity_6h": {
        "display": "Application Velocity (6 Hours)",
        "archetype": "MULE_FARMING",
        "desc": "Short-term account opening application frequency per hour.",
    },
    "velocity_24h": {
        "display": "Application Velocity (24 Hours)",
        "archetype": "MULE_FARMING",
        "desc": "Daily account opening velocity baseline.",
    },
    "velocity_4w": {
        "display": "Application Velocity (4 Weeks)",
        "archetype": "MULE_FARMING",
        "desc": "Long-term application velocity baseline across the banking channel.",
    },
    "velocity_burst_6h_4w": {
        "display": "Velocity Surge Ratio (6h / 4w)",
        "archetype": "MULE_FARMING",
        "desc": "Application surge acceleration comparing recent 6-hour burst against historical 4-week channel baseline.",
    },
    "velocity_ratio_6h_24h": {
        "display": "Velocity Surge Ratio (6h / 24h)",
        "archetype": "MULE_FARMING",
        "desc": "Immediate application rate surge against 24-hour baseline.",
    },
    "date_of_birth_distinct_emails_4w": {
        "display": "DOB Email Multiplicity (4 Weeks)",
        "archetype": "MULE_FARMING",
        "desc": "Count of distinct email addresses sharing the applicant's exact birthdate; elevated count indicates identity syndication.",
    },
    "device_distinct_emails_8w": {
        "display": "Device Email Multiplicity (8 Weeks)",
        "archetype": "MULE_FARMING",
        "desc": "Number of separate applicant emails originating from this specific device within 8 weeks.",
    },
    "session_length_in_minutes": {
        "display": "Online Application Session Length",
        "archetype": "IDENTITY_THEFT",
        "desc": "Time elapsed during application completion; ultra-rapid sessions betray automated bots, whereas prolonged sessions suggest scripted spoofing.",
    },
    "keep_alive_session": {
        "display": "Keep-Alive Session Preference",
        "archetype": "IDENTITY_THEFT",
        "desc": "User session preference toggle; lack of session customization is common in headless automated bot filings.",
    },
    "phone_home_valid": {
        "display": "Home Phone Validation",
        "archetype": "IDENTITY_THEFT",
        "desc": "Verification status of home landline contact number.",
    },
    "phone_mobile_valid": {
        "display": "Mobile Phone Validation",
        "archetype": "IDENTITY_THEFT",
        "desc": "Verification status of mobile contact number.",
    },
    "no_valid_phone": {
        "display": "Unreachable Contact Flag (No Valid Phone)",
        "archetype": "IDENTITY_THEFT",
        "desc": "Flag indicating that neither home nor mobile phone numbers could be validated by telecommunication registries.",
    },
    "zip_count_4w": {
        "display": "Regional Application Density (ZIP, 4w)",
        "archetype": "MULE_FARMING",
        "desc": "Total account applications submitted from applicant ZIP code over 4 weeks.",
    },
    "bank_branch_count_8w": {
        "display": "Branch Application Volume (8w)",
        "archetype": "MULE_FARMING",
        "desc": "Account opening volume allocated to the target branch over 8 weeks.",
    },
    "foreign_request": {
        "display": "Foreign Origin Connection",
        "archetype": "IDENTITY_THEFT",
        "desc": "Application network traffic originates from an IP address outside the domestic jurisdiction.",
    },
}


def _get_feature_meta(feature_name: str) -> Dict[str, str]:
    if feature_name in FEATURE_TAXONOMY:
        return FEATURE_TAXONOMY[feature_name]
    # Fallback heuristic classification
    clean_name = feature_name.replace("_", " ").title()
    if "velocity" in feature_name or "burst" in feature_name or "zip" in feature_name:
        return {"display": clean_name, "archetype": "MULE_FARMING", "desc": "Velocity or clustering metric associated with bulk application patterns."}
    if "email" in feature_name or "name" in feature_name or "address" in feature_name or "thin" in feature_name:
        return {"display": clean_name, "archetype": "SYNTHETIC_IDENTITY", "desc": "Identity profile consistency or history metric."}
    if "phone" in feature_name or "session" in feature_name or "device" in feature_name:
        return {"display": clean_name, "archetype": "IDENTITY_THEFT", "desc": "Device, contactability, or telecommunication validation indicator."}
    if "limit" in feature_name or "income" in feature_name or "risk" in feature_name or "credit" in feature_name:
        return {"display": clean_name, "archetype": "CREDIT_RISK", "desc": "Financial capacity and credit risk parameter."}
    return {"display": clean_name, "archetype": "GENERAL", "desc": "Application attribute contributing to model risk scoring."}


# =====================================================================
# TreeSHAP Feature Attribution Engine
# =====================================================================

def lightgbm_feature_importance(model: Any, top_n: int = 20) -> pd.DataFrame:
    """Compute gain-based global feature importance for LightGBM models."""
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    imp["share"] = imp["gain"] / (imp["gain"].sum() + EPS)
    return imp.head(top_n)


def xgboost_feature_importance(model: Any, top_n: int = 20) -> pd.DataFrame:
    """Compute gain-based global feature importance for XGBoost models."""
    score = model.get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": list(score.keys()), "gain": list(score.values())})
    if imp.empty:
        return pd.DataFrame(columns=["feature", "gain", "share"])
    imp = imp.sort_values("gain", ascending=False)
    imp["share"] = imp["gain"] / (imp["gain"].sum() + EPS)
    return imp.head(top_n)


def shap_summary(model: Any, X_sample: pd.DataFrame, model_type: str = "lightgbm") -> Tuple[Any, Any]:
    """
    Computes TreeSHAP values for a sample dataset.
    Returns (explainer, shap_values).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values


def explain_single_prediction(explainer: Any, X_row: pd.DataFrame) -> Any:
    """Execute SHAP explanation on a single application row."""
    return explainer(X_row)


def top_shap_features_for_row(shap_values_row: Any, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """Extract top SHAP contributors by absolute value for a single application row."""
    if hasattr(shap_values_row, "values"):
        vals = np.asarray(shap_values_row.values).ravel()
    else:
        vals = np.asarray(shap_values_row).ravel()

    df = pd.DataFrame({"feature": feature_names, "shap_value": vals})
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")


class TreeSHAPExplainer:
    """
    Comprehensive TreeSHAP Engine for supervised Bank Account Opening Fraud models.
    Supports LightGBM, XGBoost, CatBoost, and scikit-learn tree ensembles.
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "lightgbm",
        background_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.model_type = model_type.lower()
        self.background_data = background_data
        self.feature_names = feature_names
        self._explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        import shap

        try:
            if self.background_data is not None:
                self._explainer = shap.TreeExplainer(self.model, data=self.background_data)
            else:
                self._explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            logger.warning("Standard TreeExplainer init encountered issue (%s), attempting generic wrapper.", exc)
            self._explainer = shap.TreeExplainer(self.model)

    def compute_global_importance(self, X_sample: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Calculates global feature importance via mean absolute SHAP value across population sample.
        """
        shap_values = self._explainer(X_sample)
        vals = np.asarray(shap_values.values)
        if vals.ndim == 3:  # Multiclass / binary with dual outputs
            vals = vals[:, :, 1]

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        mean_val = np.mean(vals, axis=0)

        feat_names = self.feature_names or list(X_sample.columns)
        df = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap_direction": mean_val,
        }).sort_values("mean_abs_shap", ascending=False)

        total_mag = df["mean_abs_shap"].sum() + EPS
        df["importance_share"] = df["mean_abs_shap"] / total_mag
        df["display_name"] = df["feature"].apply(lambda f: _get_feature_meta(f)["display"])
        df["archetype"] = df["feature"].apply(lambda f: _get_feature_meta(f)["archetype"])

        return df.head(top_n).reset_index(drop=True)

    def explain_application(
        self,
        X_row: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        application_id: Optional[str] = None,
        top_positive: int = 5,
        top_negative: int = 3,
    ) -> LocalWaterfallExplanation:
        """
        Generates a detailed per-application waterfall breakdown.
        Identifies top positive drivers (increasing fraud likelihood) and
        top negative anchors (decreasing fraud likelihood).
        """
        if isinstance(X_row, dict):
            X_df = pd.DataFrame([X_row])
        elif isinstance(X_row, pd.Series):
            X_df = X_row.to_frame().T
        else:
            X_df = X_row.copy()

        if len(X_df) != 1:
            X_df = X_df.iloc[[0]]

        feat_names = self.feature_names or list(X_df.columns)
        shap_res = self._explainer(X_df)

        raw_vals = np.asarray(shap_res.values)
        if raw_vals.ndim == 3:
            raw_vals = raw_vals[:, :, 1]
        vals = raw_vals.reshape(-1)

        base_val = float(shap_res.base_values[0]) if hasattr(shap_res, "base_values") and len(shap_res.base_values) > 0 else 0.0
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[-1])

        items: List[FeatureContributionItem] = []
        for idx, (fname, sval) in enumerate(zip(feat_names, vals)):
            meta = _get_feature_meta(fname)
            val = X_df[fname].iloc[0]
            direction = "INCREASES_FRAUD_RISK" if sval > 0 else "DECREASES_FRAUD_RISK"
            items.append(FeatureContributionItem(
                feature=fname,
                display_name=meta["display"],
                feature_value=val,
                shap_value=float(sval),
                contribution_direction=direction,
                importance_rank=0,  # Assigned after sort
                magnitude=float(abs(sval)),
                archetype=meta["archetype"],
                domain_explanation=meta["desc"],
            ))

        # Sort positive and negative contributors
        pos_items = sorted([it for it in items if it.shap_value > 0], key=lambda x: x.shap_value, reverse=True)
        neg_items = sorted([it for it in items if it.shap_value < 0], key=lambda x: x.shap_value)  # Most negative first

        for i, it in enumerate(pos_items, 1):
            it.importance_rank = i
        for i, it in enumerate(neg_items, 1):
            it.importance_rank = i

        top_pos = pos_items[:top_positive]
        top_neg = neg_items[:top_negative]

        # Calculate prediction score approximation in probability space if base_val is log-odds
        sum_shap = float(np.sum(vals))
        total_margin = base_val + sum_shap
        # If output appears in margin space, convert to logistic probability
        if -20.0 <= total_margin <= 20.0 and (base_val < 0 or base_val > 1):
            pred_score = 1.0 / (1.0 + np.exp(-total_margin))
        else:
            pred_score = float(np.clip(total_margin, 0.0, 1.0))

        return LocalWaterfallExplanation(
            application_id=str(application_id) if application_id is not None else None,
            base_value=base_val,
            prediction_score=pred_score,
            top_fraud_drivers=top_pos,
            top_mitigating_factors=top_neg,
            total_features_evaluated=len(items),
            sum_shap_contributions=sum_shap,
        )


# =====================================================================
# Behavioral Deviation Analyzer
# =====================================================================

class BehavioralDeviationAnalyzer:
    """
    Evaluates individual applicant metrics against reference population distributions.
    Highlights abnormal behavioral deviations across synthetic identity, credit stretch,
    and velocity burst archetypes.
    """

    MONITORED_METRICS = [
        "limit_to_income",
        "credit_risk_score",
        "velocity_burst_6h_4w",
        "velocity_6h",
        "velocity_24h",
        "thin_file_score",
        "name_email_similarity",
        "date_of_birth_distinct_emails_4w",
        "session_length_in_minutes",
        "device_distinct_emails_8w",
        "zip_count_4w",
    ]

    def __init__(self, reference_df: Optional[pd.DataFrame] = None):
        self.stats: Dict[str, Dict[str, float]] = {}
        if reference_df is not None:
            self.fit(reference_df)
        else:
            self._set_default_benchmarks()

    def _set_default_benchmarks(self) -> None:
        """Default reference statistics derived from BAF Base distribution."""
        self.stats = {
            "limit_to_income": {"mean": 2500.0, "std": 3200.0, "median": 1500.0, "p05": 300.0, "p95": 8000.0, "p99": 15000.0},
            "credit_risk_score": {"mean": 130.0, "std": 98.0, "median": 125.0, "p05": -40.0, "p95": 290.0, "p99": 350.0},
            "velocity_burst_6h_4w": {"mean": 0.85, "std": 0.65, "median": 0.72, "p05": 0.10, "p95": 2.10, "p99": 3.80},
            "velocity_6h": {"mean": 3200.0, "std": 2400.0, "median": 2800.0, "p05": 150.0, "p95": 7800.0, "p99": 11500.0},
            "velocity_24h": {"mean": 4800.0, "std": 1900.0, "median": 4600.0, "p05": 2100.0, "p95": 8200.0, "p99": 9100.0},
            "thin_file_score": {"mean": 0.45, "std": 0.65, "median": 0.0, "p05": 0.0, "p95": 2.0, "p99": 2.0},
            "name_email_similarity": {"mean": 0.52, "std": 0.28, "median": 0.51, "p05": 0.05, "p95": 0.95, "p99": 0.99},
            "date_of_birth_distinct_emails_4w": {"mean": 9.5, "std": 6.2, "median": 8.0, "p05": 1.0, "p95": 22.0, "p99": 31.0},
            "session_length_in_minutes": {"mean": 12.5, "std": 14.0, "median": 7.5, "p05": 1.2, "p95": 42.0, "p99": 68.0},
            "device_distinct_emails_8w": {"mean": 0.82, "std": 0.45, "median": 1.0, "p05": 0.0, "p95": 1.0, "p99": 2.0},
            "zip_count_4w": {"mean": 1450.0, "std": 1200.0, "median": 1100.0, "p05": 80.0, "p95": 3800.0, "p99": 5600.0},
        }

    def fit(self, df: pd.DataFrame) -> "BehavioralDeviationAnalyzer":
        """Calculates benchmark population distributions from training data."""
        self.stats = {}
        for col in self.MONITORED_METRICS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) > 10:
                    self.stats[col] = {
                        "mean": float(series.mean()),
                        "std": float(series.std() + EPS),
                        "median": float(series.median()),
                        "p05": float(series.quantile(0.05)),
                        "p95": float(series.quantile(0.95)),
                        "p99": float(series.quantile(0.99)),
                    }
        return self

    def analyze(self, row: Union[pd.Series, Dict[str, Any]]) -> List[MetricDeviation]:
        """
        Analyzes an applicant row for statistical anomalies against population baselines.
        """
        if isinstance(row, pd.Series):
            row_dict = row.to_dict()
        else:
            row_dict = dict(row)

        deviations: List[MetricDeviation] = []

        # Derive helper ratios if missing
        if "limit_to_income" not in row_dict and "proposed_credit_limit" in row_dict and "income" in row_dict:
            inc = float(row_dict["income"])
            lim = float(row_dict["proposed_credit_limit"])
            row_dict["limit_to_income"] = lim / (inc + EPS)

        if "velocity_burst_6h_4w" not in row_dict and "velocity_6h" in row_dict and "velocity_4w" in row_dict:
            v6 = max(0.0, float(row_dict["velocity_6h"]))
            v4 = float(row_dict["velocity_4w"])
            row_dict["velocity_burst_6h_4w"] = v6 / (v4 + EPS)

        for metric, stat in self.stats.items():
            if metric not in row_dict or row_dict[metric] is None or pd.isna(row_dict[metric]):
                continue

            try:
                val = float(row_dict[metric])
            except (ValueError, TypeError):
                continue

            z = (val - stat["mean"]) / stat["std"]
            pct = float(np.clip(50.0 + (z * 34.0), 0.1, 99.9))  # Gaussian approximation for percentile

            severity = "NORMAL"
            direction = "EXPECTED"
            desc = f"Normal profile: {val:.2f} aligns with population median {stat['median']:.2f}."

            # Archetype-specific anomaly checks
            if metric == "limit_to_income":
                if val > stat["p99"] or z > 3.0:
                    severity = "CRITICAL_ANOMALY"
                    direction = "HIGH"
                    desc = f"Extreme credit extraction: requested credit limit is {val:.1f}x income decile (top 1% of applicants, z={z:+.1f})."
                elif val > stat["p95"] or z > 2.0:
                    severity = "EXTREME_DEVIATION"
                    direction = "HIGH"
                    desc = f"High credit exposure: requested limit exceeds 95th percentile relative to income decile (z={z:+.1f})."

            elif metric == "credit_risk_score":
                if val < stat["p05"] or val < -50.0:
                    severity = "CRITICAL_ANOMALY"
                    direction = "LOW"
                    desc = f"Severely degraded credit bureau score ({val:.1f}), well below 5th percentile ({stat['p05']:.1f})."
                elif val < 50.0:
                    severity = "MODERATE_DEVIATION"
                    direction = "LOW"
                    desc = f"Sub-prime credit risk profile ({val:.1f}) below population median ({stat['median']:.1f})."

            elif metric in ("velocity_burst_6h_4w", "velocity_6h"):
                if val > stat["p99"] or z > 3.0:
                    severity = "CRITICAL_ANOMALY"
                    direction = "HIGH"
                    desc = f"Severe velocity burst: application submitted during an intense surge window (+{z:.1f} std dev above channel mean)."
                elif val > stat["p95"] or z > 2.0:
                    severity = "EXTREME_DEVIATION"
                    direction = "HIGH"
                    desc = f"Elevated application rate: velocity is significantly elevated (+{z:.1f} std dev above normal)."

            elif metric == "thin_file_score":
                if val >= 2.0:
                    severity = "EXTREME_DEVIATION"
                    direction = "HIGH"
                    desc = "Thin file signature: both residential and banking history checks returned empty/missing sentinels."
                elif val >= 1.0:
                    severity = "MODERATE_DEVIATION"
                    direction = "HIGH"
                    desc = "Partial thin file: applicant lacks institutional banking or residential tenure history."

            elif metric == "name_email_similarity":
                if val < 0.15:
                    severity = "EXTREME_DEVIATION"
                    direction = "LOW"
                    desc = f"Low name-to-email coherence ({val:.2f}): email address bears minimal resemblance to declared applicant name."

            elif metric == "date_of_birth_distinct_emails_4w":
                if val > stat["p99"] or val >= 25.0:
                    severity = "CRITICAL_ANOMALY"
                    direction = "HIGH"
                    desc = f"Mule syndicate indicator: {int(val)} distinct emails observed sharing this exact birthdate in past 4 weeks."

            elif metric == "session_length_in_minutes":
                if 0.0 <= val < 2.0:
                    severity = "MODERATE_DEVIATION"
                    direction = "LOW"
                    desc = f"Ultra-fast online application ({val:.1f} mins): potential automated script/bot submission."

            elif abs(z) >= 3.0:
                severity = "EXTREME_DEVIATION"
                direction = "HIGH" if z > 0 else "LOW"
                desc = f"Metric value {val:.2f} shows extreme statistical deviation ({z:+.1f} std dev from mean {stat['mean']:.2f})."
            elif abs(z) >= 2.0:
                severity = "MODERATE_DEVIATION"
                direction = "HIGH" if z > 0 else "LOW"
                desc = f"Metric value {val:.2f} shows moderate deviation ({z:+.1f} std dev from mean {stat['mean']:.2f})."

            deviations.append(MetricDeviation(
                metric_name=metric,
                applicant_value=val,
                population_mean=stat["mean"],
                population_median=stat["median"],
                population_std=stat["std"],
                z_score=z,
                percentile_rank=pct,
                severity=severity,
                anomaly_direction=direction,
                description=desc,
            ))

        return deviations


# =====================================================================
# Structured Evidence Builder
# =====================================================================

class StructuredEvidenceBuilder:
    """
    Synthesizes model predictions, TreeSHAP feature attributions, behavioral deviations,
    and domain-specific heuristic risk flags into an audit-ready ApplicantEvidencePackage.
    """

    def __init__(
        self,
        shap_explainer: Optional[TreeSHAPExplainer] = None,
        deviation_analyzer: Optional[BehavioralDeviationAnalyzer] = None,
        default_threshold: float = 0.50,
    ):
        self.shap_explainer = shap_explainer
        self.deviation_analyzer = deviation_analyzer or BehavioralDeviationAnalyzer()
        self.default_threshold = default_threshold

    def evaluate_risk_flags(self, row_dict: Dict[str, Any]) -> List[str]:
        """Evaluates domain-specific account-opening risk heuristics."""
        flags: List[str] = []

        # 1. Synthetic Identity Profile Flag
        sim = float(row_dict.get("name_email_similarity", 0.5) or 0.5)
        free_email = int(row_dict.get("email_is_free", 0) or 0)
        thin_score = float(row_dict.get("thin_file_score", 0) or 0)
        if (sim < 0.20 and free_email == 1) or thin_score >= 2:
            flags.append("FLAG_SYNTHETIC_IDENTITY_RISK: Low name-email coherence on disposable domain and/or thin file history.")

        # 2. Velocity Spike Flag
        v6 = float(row_dict.get("velocity_6h", 0) or 0)
        v4 = float(row_dict.get("velocity_4w", 1) or 1)
        if v6 > 8000 or (v4 > 0 and (v6 / (v4 + EPS)) > 2.5):
            flags.append("FLAG_VELOCITY_BURST: Application surge rate indicates potential bulk script submission or mule ring activity.")

        # 3. Disproportionate Credit Request
        inc = float(row_dict.get("income", 0.5) or 0.5)
        lim = float(row_dict.get("proposed_credit_limit", 1000) or 1000)
        ratio = lim / (inc + EPS)
        if ratio > 8000 or (inc <= 0.2 and lim >= 1500):
            flags.append("FLAG_DISPROPORTIONATE_CREDIT_REQUEST: Excessive credit limit requested relative to declared income decile.")

        # 4. Contactability Failures
        phone_home = int(row_dict.get("phone_home_valid", 1) or 1)
        phone_mobile = int(row_dict.get("phone_mobile_valid", 1) or 1)
        if phone_home == 0 and phone_mobile == 0:
            flags.append("FLAG_UNVERIFIABLE_CONTACT: Neither home nor mobile phone numbers validated against telecommunication registries.")

        # 5. Mule Farm / Shared Attribute Flag
        dob_emails = float(row_dict.get("date_of_birth_distinct_emails_4w", 0) or 0)
        device_emails = float(row_dict.get("device_distinct_emails_8w", 0) or 0)
        if dob_emails >= 25 or device_emails >= 2:
            flags.append("FLAG_MULE_FARM_INDICATOR: Shared identity markers (multiple emails across matching DOB or shared physical device).")

        # 6. Bot / Automated Script Session
        session_len = float(row_dict.get("session_length_in_minutes", 10) or 10)
        keepalive = int(row_dict.get("keep_alive_session", 1) or 1)
        if 0.0 <= session_len < 2.0 and keepalive == 0:
            flags.append("FLAG_BOT_AUTOMATION_PROFILE: Rapid submission window (< 2 minutes) with default session parameters.")

        # 7. Bureau Credit Degradation
        credit_score = float(row_dict.get("credit_risk_score", 100) or 100)
        if credit_score < -50:
            flags.append("FLAG_SEVERE_CREDIT_RISK: Credit bureau scoring indicates high default / delinquent profile.")

        return flags

    def _determine_confidence_tier(self, prob: float, threshold: float) -> str:
        """Determines model classification confidence based on distance from threshold."""
        margin = abs(prob - threshold)
        if margin >= 0.35:
            return "VERY_HIGH"
        if margin >= 0.20:
            return "HIGH"
        if margin >= 0.08:
            return "MODERATE"
        return "BORDERLINE"

    def _determine_risk_level(self, prob: float) -> str:
        """Maps probability to risk tier."""
        if prob >= 0.70:
            return "CRITICAL"
        if prob >= 0.35:
            return "HIGH"
        if prob >= 0.10:
            return "MEDIUM"
        return "LOW"

    def build_evidence_package(
        self,
        application_data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        fraud_probability: Optional[float] = None,
        decision_threshold: Optional[float] = None,
        application_id: Optional[str] = None,
    ) -> ApplicantEvidencePackage:
        """
        Constructs the comprehensive ApplicantEvidencePackage.
        """
        if isinstance(application_data, pd.DataFrame):
            row_dict = application_data.iloc[0].to_dict()
            X_df = application_data.iloc[[0]]
        elif isinstance(application_data, pd.Series):
            row_dict = application_data.to_dict()
            X_df = application_data.to_frame().T
        else:
            row_dict = dict(application_data)
            X_df = pd.DataFrame([row_dict])

        app_id = application_id or row_dict.get("application_id") or row_dict.get("id") or f"APP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:18]}"
        threshold = decision_threshold if decision_threshold is not None else self.default_threshold

        # 1. SHAP Waterfall Breakdown
        if self.shap_explainer is not None:
            waterfall = self.shap_explainer.explain_application(
                X_df, application_id=str(app_id), top_positive=5, top_negative=3
            )
            prob = fraud_probability if fraud_probability is not None else waterfall.prediction_score
            top_drivers = waterfall.top_fraud_drivers
            top_mitigating = waterfall.top_mitigating_factors
        else:
            prob = fraud_probability if fraud_probability is not None else 0.50
            top_drivers = []
            top_mitigating = []

        prob = float(np.clip(prob, 0.0, 1.0))
        pred = 1 if prob >= threshold else 0
        risk_level = self._determine_risk_level(prob)
        confidence = self._determine_confidence_tier(prob, threshold)
        margin = prob - threshold

        # 2. Behavioral Deviations
        deviations = self.deviation_analyzer.analyze(row_dict)

        # 3. Domain Risk Flags
        risk_flags = self.evaluate_risk_flags(row_dict)

        # 4. Summary Key Metrics
        summary_metrics = {
            "income_decile": row_dict.get("income"),
            "proposed_credit_limit": row_dict.get("proposed_credit_limit"),
            "customer_age_decade": row_dict.get("customer_age"),
            "housing_status": row_dict.get("housing_status"),
            "employment_status": row_dict.get("employment_status"),
            "credit_risk_score": row_dict.get("credit_risk_score"),
            "name_email_similarity": row_dict.get("name_email_similarity"),
            "email_is_free": row_dict.get("email_is_free"),
            "velocity_6h": row_dict.get("velocity_6h"),
            "phone_contacts_valid": f"Home={row_dict.get('phone_home_valid', 'N/A')}, Mobile={row_dict.get('phone_mobile_valid', 'N/A')}",
        }

        return ApplicantEvidencePackage(
            application_id=str(app_id),
            timestamp=datetime.now(timezone.utc).isoformat(),
            fraud_probability=prob,
            fraud_prediction=pred,
            decision_threshold=threshold,
            risk_level=risk_level,
            confidence_tier=confidence,
            margin_from_threshold=margin,
            top_fraud_drivers=top_drivers,
            top_mitigating_factors=top_mitigating,
            behavioral_deviations=deviations,
            triggered_risk_flags=risk_flags,
            applicant_summary_metrics=summary_metrics,
        )
