"""
Tests for src/explainability.py:
- TreeSHAP feature attribution (global importance & local per-application waterfall breakdown)
- Behavioral deviation analyzer (applicant metrics benchmarked against population distributions)
- Structured evidence builder (consolidated evidence package with SHAP drivers, deviations, flags, and confidence)
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.explainability import (
    BehavioralDeviationAnalyzer,
    LocalWaterfallExplanation,
    StructuredEvidenceBuilder,
    TreeSHAPExplainer,
    explain_single_prediction,
    lightgbm_feature_importance,
    shap_summary,
    top_shap_features_for_row,
)


@pytest.fixture
def trained_lgb_model():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "income": rng.uniform(0.1, 0.9, n),
        "name_email_similarity": rng.uniform(0, 1, n),
        "credit_risk_score": rng.uniform(-100, 300, n),
        "proposed_credit_limit": rng.uniform(200, 2000, n),
        "velocity_6h": rng.uniform(0, 5000, n),
        "velocity_4w": rng.uniform(1000, 5000, n),
        "thin_file_score": rng.integers(0, 3, n),
        "phone_home_valid": rng.integers(0, 2, n),
        "phone_mobile_valid": rng.integers(0, 2, n),
    })
    y = ((X["credit_risk_score"] < 0) | (X["name_email_similarity"] < 0.2)).astype(int)
    dtrain = lgb.Dataset(X, y)
    model = lgb.train({"objective": "binary", "verbosity": -1, "seed": 42}, dtrain, num_boost_round=10)
    return model, X


def test_lightgbm_feature_importance(trained_lgb_model):
    model, _ = trained_lgb_model
    imp = lightgbm_feature_importance(model, top_n=5)
    assert isinstance(imp, pd.DataFrame)
    assert not imp.empty
    assert "feature" in imp.columns
    assert "gain" in imp.columns
    assert "share" in imp.columns
    assert len(imp) <= 5


def test_shap_summary_and_single_prediction(trained_lgb_model):
    model, X = trained_lgb_model
    explainer, shap_vals = shap_summary(model, X.iloc[:20], model_type="lightgbm")
    assert explainer is not None
    assert shap_vals is not None

    single = explain_single_prediction(explainer, X.iloc[[0]])
    assert single is not None

    top_feats = top_shap_features_for_row(single, list(X.columns), top_n=3)
    assert len(top_feats) <= 3
    assert "feature" in top_feats.columns
    assert "shap_value" in top_feats.columns


def test_treeshap_explainer(trained_lgb_model):
    model, X = trained_lgb_model
    explainer = TreeSHAPExplainer(model, model_type="lightgbm", feature_names=list(X.columns))

    # Global importance
    global_imp = explainer.compute_global_importance(X.iloc[:50], top_n=5)
    assert isinstance(global_imp, pd.DataFrame)
    assert len(global_imp) == 5
    assert "mean_abs_shap" in global_imp.columns
    assert "importance_share" in global_imp.columns
    assert "archetype" in global_imp.columns

    # Local per-application explanation
    row = X.iloc[0].to_dict()
    explanation = explainer.explain_application(row, application_id="APP-TEST-001", top_positive=3, top_negative=2)
    assert isinstance(explanation, LocalWaterfallExplanation)
    assert explanation.application_id == "APP-TEST-001"
    assert 0.0 <= explanation.prediction_score <= 1.0
    assert len(explanation.top_fraud_drivers) <= 3
    assert len(explanation.top_mitigating_factors) <= 2
    assert explanation.total_features_evaluated == len(X.columns)

    d = explanation.to_dict()
    assert "top_fraud_drivers" in d
    assert "base_value" in d


def test_behavioral_deviation_analyzer():
    analyzer = BehavioralDeviationAnalyzer()

    # Normal application
    normal_row = {
        "limit_to_income": 1500.0,
        "credit_risk_score": 130.0,
        "velocity_burst_6h_4w": 0.80,
        "thin_file_score": 0,
        "name_email_similarity": 0.65,
    }
    devs = analyzer.analyze(normal_row)
    assert len(devs) > 0
    assert all(d.severity == "NORMAL" for d in devs)

    # Anomaly application (Credit stretch + synthetic thin file)
    fraud_row = {
        "proposed_credit_limit": 2000.0,
        "income": 0.1,  # limit_to_income = 20,000 (critical anomaly)
        "credit_risk_score": -90.0,  # critical credit anomaly
        "thin_file_score": 2,  # extreme deviation
        "name_email_similarity": 0.05,  # extreme mismatch
        "velocity_6h": 12000.0,
        "velocity_4w": 2000.0,  # burst = 6.0 (critical burst)
    }
    devs_fraud = analyzer.analyze(fraud_row)
    severities = {d.metric_name: d.severity for d in devs_fraud}
    assert severities.get("limit_to_income") in ("CRITICAL_ANOMALY", "EXTREME_DEVIATION")
    assert severities.get("credit_risk_score") == "CRITICAL_ANOMALY"
    assert severities.get("thin_file_score") == "EXTREME_DEVIATION"
    assert severities.get("name_email_similarity") == "EXTREME_DEVIATION"
    assert severities.get("velocity_burst_6h_4w") == "CRITICAL_ANOMALY"


def test_structured_evidence_builder(trained_lgb_model):
    model, X = trained_lgb_model
    explainer = TreeSHAPExplainer(model, model_type="lightgbm", feature_names=list(X.columns))
    analyzer = BehavioralDeviationAnalyzer()
    builder = StructuredEvidenceBuilder(shap_explainer=explainer, deviation_analyzer=analyzer, default_threshold=0.45)

    test_row = {
        "income": 0.1,
        "proposed_credit_limit": 2000.0,
        "credit_risk_score": -80.0,
        "name_email_similarity": 0.08,
        "email_is_free": 1,
        "thin_file_score": 2,
        "velocity_6h": 9000.0,
        "velocity_4w": 3000.0,
        "phone_home_valid": 0,
        "phone_mobile_valid": 0,
        "date_of_birth_distinct_emails_4w": 28,
        "session_length_in_minutes": 1.2,
        "keep_alive_session": 0,
        "customer_age": 30,
        "housing_status": "BC",
        "employment_status": "CA",
    }

    evidence = builder.build_evidence_package(test_row, fraud_probability=0.88, application_id="APP-998877")

    assert evidence.application_id == "APP-998877"
    assert evidence.fraud_probability == 0.88
    assert evidence.fraud_prediction == 1
    assert evidence.risk_level == "CRITICAL"
    assert evidence.confidence_tier in ("VERY_HIGH", "HIGH")
    assert len(evidence.top_fraud_drivers) > 0
    assert len(evidence.triggered_risk_flags) >= 4  # multiple critical flags triggered

    # Check serialization
    d = evidence.to_dict()
    assert d["application_id"] == "APP-998877"
    json_str = evidence.to_json()
    assert "APP-998877" in json_str

    # Check prompt context formatting
    prompt_ctx = evidence.format_prompt_context()
    assert "BANK ACCOUNT OPENING APPLICATION EVIDENCE" in prompt_ctx
    assert "FLAG_SYNTHETIC_IDENTITY_RISK" in prompt_ctx
    assert "FLAG_UNVERIFIABLE_CONTACT" in prompt_ctx
    assert "FLAG_DISPROPORTIONATE_CREDIT_REQUEST" in prompt_ctx
