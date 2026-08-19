"""
tests/test_data_leakage.py -- Data leakage prevention tests.

Verifies:
1. Test set remains untouched during preprocessing and model fitting.
2. Temporal validation integrity (months 0-5 train vs months 6-7 test).
3. Encoders/imputers/scalers fit strictly on the training fold only.
4. Target column is strictly excluded from all feature matrices.
5. Sentinel negative values (-1) are correctly converted to NaN + missingness indicators.
6. Legitimate negative numeric columns preserve valid negative values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_validation import check_no_leakage
from src.preprocessing import Preprocessor, to_nan_and_flag


def test_test_set_untouched(synthetic_baf_df, cfg):
    """Ensure preprocessor fitting is 100% isolated to the train fold."""
    train_df = synthetic_baf_df.iloc[:1000].copy()
    test_df = synthetic_baf_df.iloc[1000:].copy()

    # Extreme distribution shift in test set
    test_df["income"] = 999.0
    test_df["velocity_6h"] = 88888.0

    prep = Preprocessor(cfg)
    prep.fit(train_df)

    # Scaler mean must reflect train set only, unaffected by test shift
    income_idx = prep.numeric_cols_.index("income")
    train_mean = train_df["income"].mean()
    scaler_mean = prep._scaler.mean_[income_idx]

    np.testing.assert_almost_equal(scaler_mean, train_mean, decimal=4)

    # Imputer statistics must reflect train set only
    imputer_median = prep._imputer.statistics_[income_idx]
    train_median = train_df["income"].median()
    np.testing.assert_almost_equal(imputer_median, train_median, decimal=4)


def test_no_temporal_leakage(synthetic_baf_df, cfg):
    """Verify temporal split protocol (months 0-5 train vs months 6-7 test)."""
    df = synthetic_baf_df.copy()
    train_months = set(range(0, 6))
    test_months = {6, 7}

    train_mask = df["month"].isin(train_months)
    test_mask = df["month"].isin(test_months)

    train_data = df[train_mask]
    test_data = df[test_mask]

    # Temporal boundaries strictly disjoint
    assert set(train_data["month"].unique()).issubset(train_months)
    assert set(test_data["month"].unique()).issubset(test_months)
    assert len(set(train_data["month"].unique()) & set(test_data["month"].unique())) == 0

    # Max train month is strictly less than min test month
    assert train_data["month"].max() < test_data["month"].min()

    # Fitting on train months only
    prep = Preprocessor(cfg)
    prep.fit(train_data)
    X_test = prep.transform_tree(test_data)

    assert len(X_test) == len(test_data)
    assert "fraud_bool" not in X_test.columns


def test_encoders_fit_only_on_train_fold(synthetic_baf_df, cfg):
    """Verify that unseen categorical values in test do not alter feature matrix structure."""
    train_df = synthetic_baf_df.iloc[:1000].copy()
    test_df = synthetic_baf_df.iloc[1000:].copy()

    prep = Preprocessor(cfg)
    prep.fit(train_df)

    expected_dense_cols = list(prep.feature_cols_dense_)
    expected_tree_cols = list(prep.feature_cols_tree_)

    # Inject completely unseen categories in test set
    test_df["payment_type"] = "UNSEEN_CAT_99"
    test_df["employment_status"] = "NEW_STATUS_X"
    test_df["device_os"] = "alien_os_1.0"

    X_dense_test = prep.transform_dense(test_df)
    X_tree_test = prep.transform_tree(test_df)

    # Columns and shape must match exactly what was learned from train
    assert list(X_dense_test.columns) == expected_dense_cols
    assert list(X_tree_test.columns) == expected_tree_cols
    assert not np.isnan(X_dense_test.to_numpy()).any()


def test_target_exclusion(synthetic_baf_df, cfg):
    """Verify target column is never included in the transformed feature matrices."""
    prep = Preprocessor(cfg)
    prep.fit(synthetic_baf_df)

    X_tree = prep.transform_tree(synthetic_baf_df)
    X_dense = prep.transform_dense(synthetic_baf_df)

    assert "fraud_bool" not in X_tree.columns
    assert "fraud_bool" not in X_dense.columns
    assert "is_fraud" not in X_tree.columns
    assert "is_fraud" not in X_dense.columns
    assert "target" not in X_tree.columns
    assert "target" not in X_dense.columns


def test_sentinel_correctness(synthetic_baf_df, cfg):
    """Verify sentinel values (-1) are mapped to NaN and flagged with _is_missing."""
    sentinel_cols = list(cfg.sentinel_cols)
    assert len(sentinel_cols) == 6

    # Create controlled df with -1 in all sentinel columns
    df = synthetic_baf_df.copy()
    for col in sentinel_cols:
        df.loc[:10, col] = -1.0

    transformed = to_nan_and_flag(df, sentinel_cols)

    for col in sentinel_cols:
        flag_col = f"{col}_is_missing"
        assert flag_col in transformed.columns
        # First 11 rows should have NaN in feature and 1 in flag
        assert transformed.loc[:10, col].isna().all()
        assert (transformed.loc[:10, flag_col] == 1).all()
        # Non-negative rows should have 0 in flag
        valid_rows = df[col] >= 0
        if valid_rows.any():
            assert (transformed.loc[valid_rows, flag_col] == 0).all()


def test_legitimate_negatives_preserved(synthetic_baf_df, cfg):
    """Verify legitimate negative columns (credit_risk_score, velocity_6h) are not converted to NaN."""
    legit_cols = list(cfg.legitimate_negative_cols)
    df = synthetic_baf_df.copy()
    df.loc[:10, "credit_risk_score"] = -150.0
    df.loc[:10, "velocity_6h"] = -50.0

    # Sentinel transform with only cfg.sentinel_cols
    transformed = to_nan_and_flag(df, list(cfg.sentinel_cols))

    for col in legit_cols:
        assert f"{col}_is_missing" not in transformed.columns
        # Negative values must be preserved
        assert (transformed.loc[:10, col] < 0).all()
        assert not transformed.loc[:10, col].isna().any()


def test_check_no_leakage_detector(synthetic_baf_df):
    """Verify check_no_leakage catches post-decision columns."""
    clean_df = synthetic_baf_df.copy()
    assert len(check_no_leakage(clean_df)) == 0

    leaky_df = synthetic_baf_df.copy()
    leaky_df["chargeback_amount"] = 100.0
    leaky_df["investigation_status"] = "closed"
    leaky_df["outcome_label"] = 1

    hits = check_no_leakage(leaky_df)
    assert "chargeback_amount" in hits
    assert "investigation_status" in hits
    assert "outcome_label" in hits
