"""
tests/test_models.py -- Model scoring, probability calibration, risk scoring, and threshold optimization tests.

Verifies:
1. Model scoring output format (probabilities, binary predictions, risk categories).
2. Predicted probabilities strictly lie within [0.0, 1.0].
3. Derived risk scores strictly lie within [0.0, 100.0].
4. Threshold logic correctly separates positive and negative predictions.
5. F1 and cost-sensitive threshold optimization routines operate accurately.
6. Risk level assignment adheres strictly to defined cutoff boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import models
from src.evaluation import confusion_at_threshold, cost_sensitive_eval, evaluate_scores
from src.prediction import predict_dataframe, risk_level
from src.preprocessing import Preprocessor
from src.threshold_optimization import best_threshold_by_cost, best_threshold_by_f1, sweep_cost_sensitive, sweep_thresholds


@pytest.fixture()
def fitted_pipeline(synthetic_baf_df, cfg):
    prep = Preprocessor(cfg)
    prep.fit(synthetic_baf_df)
    X_dense = prep.transform_dense(synthetic_baf_df)
    y = prep.get_target(synthetic_baf_df)

    lr_model = models.train_logistic_regression(X_dense, y, cfg, {"class_weight": "balanced"})
    meta = {
        "model_type": "logistic_regression",
        "strategy": "class_weight",
        "threshold": 0.05,
        "feature_columns": list(X_dense.columns),
        "model_iteration": "test_v1",
    }
    return lr_model, prep, meta


def test_model_scoring_output_format(synthetic_baf_df, fitted_pipeline, cfg):
    """Verify that scoring returns properly formatted predictions and metadata."""
    model, prep, meta = fitted_pipeline
    sample = synthetic_baf_df.head(20).copy()

    scored = predict_dataframe(sample, model, prep, meta, cfg)

    # Required output columns
    assert "fraud_probability" in scored.columns
    assert "fraud_prediction" in scored.columns
    assert "risk_level" in scored.columns

    # Type and range constraints
    assert pd.api.types.is_float_dtype(scored["fraud_probability"])
    assert pd.api.types.is_integer_dtype(scored["fraud_prediction"])
    assert scored["fraud_prediction"].isin([0, 1]).all()
    assert set(scored["risk_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def test_calibrated_probability_range(synthetic_baf_df, fitted_pipeline):
    """Verify all model probability estimates fall strictly within [0.0, 1.0]."""
    model, prep, meta = fitted_pipeline
    X = prep.transform_dense(synthetic_baf_df)

    probs = models.predict_proba(model, X, meta["model_type"])

    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(synthetic_baf_df)
    assert not np.isnan(probs).any(), "Probabilities contain NaN values"
    assert not np.isinf(probs).any(), "Probabilities contain Infinite values"
    assert (probs >= 0.0).all(), f"Found probability < 0.0: min is {probs.min()}"
    assert (probs <= 1.0).all(), f"Found probability > 1.0: max is {probs.max()}"


def test_risk_score_range(synthetic_baf_df, fitted_pipeline):
    """Verify derived risk scores (prob * 100) are strictly in [0.0, 100.0]."""
    model, prep, meta = fitted_pipeline
    X = prep.transform_dense(synthetic_baf_df)
    probs = models.predict_proba(model, X, meta["model_type"])

    risk_scores = np.round(probs * 100.0, 2)

    assert (risk_scores >= 0.0).all()
    assert (risk_scores <= 100.0).all()


def test_risk_level_threshold_boundaries(cfg):
    """Verify exact categorization of probabilities into risk levels."""
    probs = np.array([0.005, 0.019, 0.020, 0.050, 0.099, 0.100, 0.250, 0.299, 0.300, 0.850, 1.0])
    levels = risk_level(probs, cfg)

    assert levels[0] == "LOW"      # 0.005 < 0.02
    assert levels[1] == "LOW"      # 0.019 < 0.02
    assert levels[2] == "MEDIUM"   # 0.020 in [0.02, 0.10)
    assert levels[3] == "MEDIUM"   # 0.050 in [0.02, 0.10)
    assert levels[4] == "MEDIUM"   # 0.099 in [0.02, 0.10)
    assert levels[5] == "HIGH"     # 0.100 in [0.10, 0.30)
    assert levels[6] == "HIGH"     # 0.250 in [0.10, 0.30)
    assert levels[7] == "HIGH"     # 0.299 in [0.10, 0.30)
    assert levels[8] == "CRITICAL" # 0.300 >= 0.30
    assert levels[9] == "CRITICAL" # 0.850 >= 0.30
    assert levels[10] == "CRITICAL"# 1.000 >= 0.30


def test_binary_threshold_decision_logic():
    """Verify binary decision threshold application."""
    probs = np.array([0.01, 0.04, 0.05, 0.06, 0.90])
    threshold = 0.05

    preds = (probs >= threshold).astype(int)
    np.testing.assert_array_equal(preds, [0, 0, 1, 1, 1])


def test_threshold_optimization_f1_and_cost():
    """Verify sweep_thresholds and sweep_cost_sensitive logic."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 0] * 50)
    y_score = np.array([0.01, 0.02, 0.03, 0.05, 0.10, 0.30, 0.60, 0.04, 0.70, 0.02] * 50)

    thresholds = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80]

    f1_sweep = sweep_thresholds(y_true, y_score, thresholds)
    assert len(f1_sweep) == len(thresholds)
    assert "f1" in f1_sweep.columns
    assert "precision" in f1_sweep.columns
    assert "recall" in f1_sweep.columns

    best_f1 = best_threshold_by_f1(f1_sweep)
    assert best_f1 in thresholds

    # Cost sweep with asymmetric costs
    cost_fp = 50.0
    cost_fn = 500.0
    cost_sweep = sweep_cost_sensitive(y_true, y_score, thresholds, cost_fp, cost_fn)
    assert "total_cost" in cost_sweep.columns
    assert "cost_per_application" in cost_sweep.columns

    best_cost_thr = best_threshold_by_cost(cost_sweep)
    assert best_cost_thr in thresholds


def test_evaluation_metrics_reporting():
    """Verify evaluation score reporting structure."""
    y_true = np.array([0] * 990 + [1] * 10)
    y_score = np.linspace(0.001, 0.999, 1000)

    res = evaluate_scores(y_true, y_score, target_fpr=0.05)

    assert "roc_auc" in res
    assert "pr_auc" in res
    assert "tpr_at_5pct_fpr" in res
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["pr_auc"] <= 1.0
    assert 0.0 <= res["tpr_at_5pct_fpr"] <= 1.0
