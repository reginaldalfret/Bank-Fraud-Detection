"""Feature Engineering Service for Bank Account Opening Fraud Detection.

Performs causal feature transformations, sentinel missing-value flagging,
velocity ratios, synthetic identity coherence metrics, thin-file scoring,
and one-hot / category encoding for both single and chunked batch inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import numpy as np
import pandas as pd

from src.api.schemas import ApplicationRequest

logger = logging.getLogger("fraud_api.feature_service")

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

# The canonical ordered feature names expected by the LightGBM champion model
CANONICAL_FEATURE_NAMES = [
    'income', 'name_email_similarity', 'customer_age', 'days_since_request', 'zip_count_4w',
    'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
    'credit_risk_score', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'has_other_cards',
    'proposed_credit_limit', 'foreign_request', 'keep_alive_session',
    'prev_address_months_count_is_missing', 'prev_address_months_count',
    'current_address_months_count_is_missing', 'current_address_months_count',
    'bank_months_count_is_missing', 'bank_months_count',
    'session_length_in_minutes_is_missing', 'session_length_in_minutes',
    'device_distinct_emails_8w_is_missing', 'device_distinct_emails_8w',
    'intended_balcon_amount_is_missing', 'intended_balcon_amount',
    'velocity_burst_6h_4w', 'velocity_ratio_6h_24h', 'velocity_burst_24h_4w',
    'email_mismatch_free', 'dob_emails_x_mismatch', 'thin_file_score', 'n_missing',
    'no_valid_phone', 'limit_to_income', 'limit_per_risk', 'risk_x_income',
    'emails_per_session_min',
    'payment_type_AA', 'payment_type_AB', 'payment_type_AC', 'payment_type_AD', 'payment_type_AE',
    'employment_status_CA', 'employment_status_CB', 'employment_status_CC', 'employment_status_CD',
    'employment_status_CE', 'employment_status_CF', 'employment_status_CG',
    'housing_status_BA', 'housing_status_BB', 'housing_status_BC', 'housing_status_BD',
    'housing_status_BE', 'housing_status_BF', 'housing_status_BG',
    'source_INTERNET', 'source_TELEAPP',
    'device_os_linux', 'device_os_macintosh', 'device_os_other', 'device_os_windows', 'device_os_x11'
]


class FeatureService:
    """Enterprise Feature Engineering and Transformation Engine."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        from src.feature_engine import ProductionFeatureEngine
        self.engine = ProductionFeatureEngine()
        self.feature_names = feature_names or CANONICAL_FEATURE_NAMES
        self.sentinel_cols = SENTINEL_COLS
        self.categorical_map = CATEGORICAL_MAP

    def transform_single_dict(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single raw dictionary into engineered model features."""
        df = pd.DataFrame([raw])
        X = self.engine.transform(df)
        return X.iloc[0].to_dict()

    def transform_single(self, application: ApplicationRequest) -> np.ndarray:
        """Transform an ApplicationRequest into a feature vector matching canonical order."""
        df = pd.DataFrame([application.model_dump()])
        X = self.engine.transform(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.feature_names].values[0].astype(np.float64)

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a raw DataFrame of applications into fully engineered features."""
        X = self.engine.transform(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.feature_names]


    def transform_chunks(
        self,
        df_iterator: Iterable[pd.DataFrame],
        chunk_size: int = 5000
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream/chunk generator for large batch scoring."""
        for chunk in df_iterator:
            yield self.transform_dataframe(chunk)


_feature_service_instance: Optional[FeatureService] = None


def get_feature_service() -> FeatureService:
    global _feature_service_instance
    if _feature_service_instance is None:
        _feature_service_instance = FeatureService()
    return _feature_service_instance
