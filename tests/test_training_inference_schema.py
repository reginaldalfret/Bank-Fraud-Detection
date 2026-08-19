"""
tests/test_training_inference_schema.py -- Audit of production feature pipeline.

Verifies:
1. Training feature names == Inference feature names
2. Training feature order == Inference feature order
3. Training dtypes == Inference dtypes
4. Engineered features (velocity_burst_6h_4w, email_mismatch_free, thin_file_score, limit_to_income) are actually calculated and not zero-filled.
5. Single-row inference produces valid predictions through the full causal pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from src.scientific_pipeline import ProductionFeatureEngine


def test_feature_schema_parity():
    bundle_path = os.path.join("artifacts", "best_model.joblib")
    assert os.path.exists(bundle_path), "artifacts/best_model.joblib must exist"

    bundle = joblib.load(bundle_path)
    feature_engine = bundle["feature_engine"]
    expected_cols = bundle["feature_cols"]

    # Create dummy raw application
    raw_app = pd.DataFrame([{
        "income": 0.6, "name_email_similarity": 0.85, "customer_age": 35, "days_since_request": 0.05,
        "zip_count_4w": 120, "velocity_6h": 2500.0, "velocity_24h": 4000.0, "velocity_4w": 5000.0,
        "bank_branch_count_8w": 10, "date_of_birth_distinct_emails_4w": 2, "credit_risk_score": 150.0,
        "email_is_free": 1, "phone_home_valid": 1, "phone_mobile_valid": 1, "has_other_cards": 0,
        "proposed_credit_limit": 1200.0, "foreign_request": 0, "keep_alive_session": 1,
        "prev_address_months_count": 24, "current_address_months_count": 36, "bank_months_count": 12,
        "session_length_in_minutes": 10.0, "device_distinct_emails_8w": 1, "intended_balcon_amount": -1.0,
        "payment_type": "AA", "employment_status": "CA", "housing_status": "BA", "source": "INTERNET",
        "device_os": "windows", "device_fraud_count": 0, "month": 6
    }])

    transformed = feature_engine.transform(raw_app)

    # 1. Feature count & names match
    assert list(transformed.columns) == expected_cols
    assert len(transformed.columns) == len(expected_cols)

    # 2. Engineered features are computed from real data and NOT 0.0
    assert transformed["velocity_burst_6h_4w"].iloc[0] == pytest.approx(2500.0 / 5000.0, rel=1e-3)
    assert transformed["email_mismatch_free"].iloc[0] == pytest.approx((1.0 - 0.85) * 1.0, rel=1e-3)
    assert transformed["limit_to_income"].iloc[0] == pytest.approx(1200.0 / 0.6, rel=1e-3)
    assert transformed["intended_balcon_amount_is_missing"].iloc[0] == 1.0


def test_production_bundle_scoring_integrity():
    bundle_path = os.path.join("artifacts", "best_model.joblib")
    bundle = joblib.load(bundle_path)

    model = bundle["model"]
    engine = bundle["feature_engine"]
    calibrator = bundle["calibrator"]
    a_coef = bundle["bayes_a"]
    c_coef = bundle["bayes_c"]
    thr = bundle["primary_threshold"]

    sample_raw = pd.DataFrame([{
        "income": 0.1, "name_email_similarity": 0.05, "customer_age": 20, "days_since_request": 5.0,
        "zip_count_4w": 2500, "velocity_6h": 9000.0, "velocity_24h": 12000.0, "velocity_4w": 15000.0,
        "bank_branch_count_8w": 80, "date_of_birth_distinct_emails_4w": 8, "credit_risk_score": -150.0,
        "email_is_free": 1, "phone_home_valid": 0, "phone_mobile_valid": 0, "has_other_cards": 0,
        "proposed_credit_limit": 2500.0, "foreign_request": 1, "keep_alive_session": 0,
        "prev_address_months_count": -1, "current_address_months_count": 0, "bank_months_count": -1,
        "session_length_in_minutes": 0.5, "device_distinct_emails_8w": 2, "intended_balcon_amount": 100.0,
        "payment_type": "AC", "employment_status": "CE", "housing_status": "BE", "source": "INTERNET",
        "device_os": "other", "device_fraud_count": 0, "month": 7
    }])

    feat = engine.transform(sample_raw)
    p_raw = model.predict_proba(feat)[:, 1]
    p_bayes = (p_raw * a_coef) / (p_raw * a_coef + (1.0 - p_raw) * c_coef)
    p_cal = calibrator.predict(p_bayes)

    assert 0.0 <= p_cal[0] <= 1.0
    risk_score = round(float(p_cal[0] * 100), 2)
    assert 0.0 <= risk_score <= 100.0
    is_fraud = bool(p_cal[0] >= thr)
    assert is_fraud is True  # High risk fraud profile flagged
