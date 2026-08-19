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

# The canonical ordered feature names expected by the tree models
CANONICAL_FEATURE_NAMES = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "month",
    "prev_address_months_count_is_missing",
    "current_address_months_count_is_missing",
    "bank_months_count_is_missing",
    "session_length_in_minutes_is_missing",
    "device_distinct_emails_8w_is_missing",
    "intended_balcon_amount_is_missing",
    "velocity_burst_6h_4w",
    "velocity_ratio_6h_24h",
    "velocity_burst_24h_4w",
    "email_mismatch_free",
    "dob_emails_x_mismatch",
    "total_address_history",
    "thin_file_score",
    "n_missing",
    "n_valid_phones",
    "no_valid_phone",
    "limit_to_income",
    "limit_per_risk",
    "risk_x_income",
    "emails_per_session_min",
    "short_session_no_keepalive",
    "zip_density_vs_velocity",
    "payment_type_AA",
    "payment_type_AB",
    "payment_type_AC",
    "payment_type_AD",
    "payment_type_AE",
    "employment_status_CA",
    "employment_status_CB",
    "employment_status_CC",
    "employment_status_CD",
    "employment_status_CE",
    "employment_status_CF",
    "employment_status_CG",
    "housing_status_BA",
    "housing_status_BB",
    "housing_status_BC",
    "housing_status_BD",
    "housing_status_BE",
    "housing_status_BF",
    "housing_status_BG",
    "source_INTERNET",
    "source_TELEAPP",
    "device_os_linux",
    "device_os_macintosh",
    "device_os_other",
    "device_os_windows",
    "device_os_x11",
]


class FeatureService:
    """Enterprise Feature Engineering and Transformation Engine."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or CANONICAL_FEATURE_NAMES
        self.sentinel_cols = SENTINEL_COLS
        self.categorical_map = CATEGORICAL_MAP

    def transform_single_dict(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single raw dictionary into engineered model features."""
        r = dict(raw)
        
        # 1. Sentinel negative conversion & missing indicators
        miss_count = 0
        for c in self.sentinel_cols:
            val = r.get(c, -1.0)
            is_miss = 1 if (val is None or (isinstance(val, (int, float)) and val < 0) or pd.isna(val)) else 0
            r[f"{c}_is_missing"] = is_miss
            miss_count += is_miss
            r[c] = np.nan if is_miss else float(val)

        # 2. Velocity burst ratios (clip velocity_6h at 0)
        v6 = max(0.0, float(r.get("velocity_6h", 0.0)))
        v24 = float(r.get("velocity_24h", 4000.0))
        v4w = float(r.get("velocity_4w", 4000.0))
        r["velocity_burst_6h_4w"] = v6 / (v4w + EPS)
        r["velocity_ratio_6h_24h"] = v6 / (v24 + EPS)
        r["velocity_burst_24h_4w"] = v24 / (v4w + EPS)

        # 3. Synthetic Identity: Email & Name Coherence
        name_email_sim = float(r.get("name_email_similarity", 0.5))
        email_is_free = float(r.get("email_is_free", 1))
        dob_emails = float(r.get("date_of_birth_distinct_emails_4w", 0.0))
        
        r["email_mismatch_free"] = (1.0 - name_email_sim) * email_is_free
        r["dob_emails_x_mismatch"] = dob_emails * (1.0 - name_email_sim)

        # 4. Thin File: Address & Banking History
        prev_addr = 0.0 if pd.isna(r.get("prev_address_months_count")) else float(r["prev_address_months_count"])
        cur_addr = 0.0 if pd.isna(r.get("current_address_months_count")) else float(r["current_address_months_count"])
        r["total_address_history"] = prev_addr + cur_addr
        r["thin_file_score"] = r.get("prev_address_months_count_is_missing", 0) + r.get("bank_months_count_is_missing", 0)
        r["n_missing"] = miss_count

        # 5. Contactability
        p_home = int(r.get("phone_home_valid", 0))
        p_mob = int(r.get("phone_mobile_valid", 1))
        r["n_valid_phones"] = p_home + p_mob
        r["no_valid_phone"] = 1 if r["n_valid_phones"] == 0 else 0

        # 6. Financial Coherence
        income = float(r.get("income", 0.5))
        credit_limit = float(r.get("proposed_credit_limit", 200.0))
        credit_risk = float(r.get("credit_risk_score", 100.0))

        r["limit_to_income"] = credit_limit / (income + EPS)
        r["limit_per_risk"] = credit_limit / (credit_risk + 200.0)
        r["risk_x_income"] = credit_risk * income

        # 7. Device & Session Behavior
        session_len = 0.0 if pd.isna(r.get("session_length_in_minutes")) else float(r["session_length_in_minutes"])
        dev_emails = 0.0 if pd.isna(r.get("device_distinct_emails_8w")) else float(r["device_distinct_emails_8w"])
        r["emails_per_session_min"] = dev_emails / (session_len + 1.0)
        
        keep_alive = int(r.get("keep_alive_session", 1))
        r["short_session_no_keepalive"] = 1 if (session_len < 5 and keep_alive == 0) else 0

        # 8. Geographic Clustering
        zip_cnt = float(r.get("zip_count_4w", 1000.0))
        r["zip_density_vs_velocity"] = zip_cnt / (v4w + EPS)

        # 9. Month default if missing
        if "month" not in r or r["month"] is None or pd.isna(r["month"]):
            r["month"] = 3.0
        else:
            r["month"] = float(r["month"])

        # 10. One-hot categoricals
        for cat_col, categories in self.categorical_map.items():
            raw_val = str(r.get(cat_col, "")).strip().upper()
            if cat_col == "device_os":
                raw_val = str(r.get(cat_col, "")).strip().lower()
            if cat_col == "payment_type" and len(raw_val) > 2 and raw_val.startswith("AA"):
                raw_val = raw_val[-2:]
            
            for cat in categories:
                col_name = f"{cat_col}_{cat}"
                r[col_name] = 1.0 if raw_val == cat.upper() or raw_val == cat.lower() else 0.0

        return r

    def transform_single(self, application: ApplicationRequest) -> np.ndarray:
        """Transform an ApplicationRequest into a feature vector matching canonical order."""
        transformed = self.transform_single_dict(application.model_dump())
        vector = np.zeros(len(self.feature_names), dtype=np.float64)
        for i, fname in enumerate(self.feature_names):
            val = transformed.get(fname, np.nan)
            vector[i] = val if val is not None else np.nan
        return vector

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a raw DataFrame of applications into fully engineered features."""
        df = df.copy()

        # 1. Sentinels & Missing Flags
        miss_df = pd.DataFrame(index=df.index)
        for col in self.sentinel_cols:
            if col in df.columns:
                miss = df[col].isna() | (df[col] < 0)
                miss_df[f"{col}_is_missing"] = miss.astype("int8")
                df[f"{col}_is_missing"] = miss.astype("int8")
                df.loc[miss, col] = np.nan
            else:
                df[f"{col}_is_missing"] = 1
                df[col] = np.nan

        # 2. Velocity Ratios
        if "velocity_6h" in df.columns:
            v6 = df["velocity_6h"].clip(lower=0)
            if "velocity_4w" in df.columns:
                df["velocity_burst_6h_4w"] = v6 / (df["velocity_4w"] + EPS)
            if "velocity_24h" in df.columns:
                df["velocity_ratio_6h_24h"] = v6 / (df["velocity_24h"] + EPS)
        if {"velocity_24h", "velocity_4w"}.issubset(df.columns):
            df["velocity_burst_24h_4w"] = df["velocity_24h"] / (df["velocity_4w"] + EPS)

        # 3. Synthetic Identity Coherence
        if {"name_email_similarity", "email_is_free"}.issubset(df.columns):
            df["email_mismatch_free"] = (1.0 - df["name_email_similarity"]) * df["email_is_free"]
        if {"date_of_birth_distinct_emails_4w", "name_email_similarity"}.issubset(df.columns):
            df["dob_emails_x_mismatch"] = df["date_of_birth_distinct_emails_4w"] * (1.0 - df["name_email_similarity"])

        # 4. Thin File
        if {"prev_address_months_count", "current_address_months_count"}.issubset(df.columns):
            df["total_address_history"] = df["prev_address_months_count"].fillna(0) + df["current_address_months_count"].fillna(0)
        
        thin_parts = [c for c in ["prev_address_months_count_is_missing", "bank_months_count_is_missing"] if c in df.columns]
        if thin_parts:
            df["thin_file_score"] = df[thin_parts].sum(axis=1).astype("int8")
        
        miss_cols = [f"{c}_is_missing" for c in self.sentinel_cols if f"{c}_is_missing" in df.columns]
        if miss_cols:
            df["n_missing"] = df[miss_cols].sum(axis=1).astype("int8")

        # 5. Contactability
        phones = [c for c in ["phone_home_valid", "phone_mobile_valid"] if c in df.columns]
        if len(phones) == 2:
            df["n_valid_phones"] = df["phone_home_valid"] + df["phone_mobile_valid"]
            df["no_valid_phone"] = (df["n_valid_phones"] == 0).astype("int8")

        # 6. Financial Coherence
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
            df["short_session_no_keepalive"] = (
                (df["session_length_in_minutes"] < 5) & (df["keep_alive_session"] == 0)
            ).astype("int8")

        # 8. Geographic Clustering
        if {"zip_count_4w", "velocity_4w"}.issubset(df.columns):
            df["zip_density_vs_velocity"] = df["zip_count_4w"] / (df["velocity_4w"] + EPS)

        # 9. Month default
        if "month" not in df.columns:
            df["month"] = 3.0
        else:
            df["month"] = df["month"].fillna(3.0)

        # 10. One-hot categoricals
        for cat_col, categories in self.categorical_map.items():
            if cat_col in df.columns:
                series = df[cat_col].astype(str).str.strip()
                if cat_col == "payment_type":
                    series = series.apply(lambda s: s[-2:] if len(s) > 2 and s.startswith("AA") else s)
                for cat in categories:
                    col_name = f"{cat_col}_{cat}"
                    if cat_col == "device_os":
                        df[col_name] = (series.str.lower() == cat.lower()).astype("float64")
                    else:
                        df[col_name] = (series.str.upper() == cat.upper()).astype("float64")
            else:
                for cat in categories:
                    df[f"{cat_col}_{cat}"] = 0.0

        # Ensure all canonical columns exist in exact order
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan

        return df[self.feature_names]

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
