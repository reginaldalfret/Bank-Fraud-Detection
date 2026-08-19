# src/feature_engine.py
import numpy as np
import pandas as pd

SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "intended_balcon_amount",
]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

class ProductionFeatureEngine:
    def __init__(self):
        self.version = "2.0.0-scientific"
        self.feature_names = []
        self.cat_categories = {}
        self.stats = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                self.cat_categories[col] = sorted(list(df[col].dropna().astype(str).unique()))
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        eps = 1e-6

        raw_num = [
            "income", "name_email_similarity", "customer_age", "days_since_request",
            "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w",
            "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
            "credit_risk_score", "email_is_free", "phone_home_valid",
            "phone_mobile_valid", "has_other_cards", "proposed_credit_limit",
            "foreign_request", "keep_alive_session"
        ]
        for c in raw_num:
            if c in df.columns:
                out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

        for col in SENTINEL_COLS:
            if col in df.columns:
                raw_val = pd.to_numeric(df[col], errors="coerce").fillna(-1.0).to_numpy()
                is_miss = (raw_val < 0).astype(np.float32)
                out[f"{col}_is_missing"] = is_miss
                clean_val = np.where(raw_val < 0, np.nan, raw_val).astype(np.float32)
                out[col] = clean_val

        if "velocity_6h" in df.columns and "velocity_4w" in df.columns:
            v6_clipped = np.maximum(pd.to_numeric(df["velocity_6h"], errors="coerce").fillna(0.0).to_numpy(), 0.0)
            v4w = np.maximum(pd.to_numeric(df["velocity_4w"], errors="coerce").fillna(0.0).to_numpy(), eps)
            v24h = np.maximum(pd.to_numeric(df["velocity_24h"], errors="coerce").fillna(0.0).to_numpy(), eps) if "velocity_24h" in df.columns else v4w
            out["velocity_burst_6h_4w"] = (v6_clipped / v4w).astype(np.float32)
            out["velocity_ratio_6h_24h"] = (v6_clipped / v24h).astype(np.float32)
            out["velocity_burst_24h_4w"] = (v24h / v4w).astype(np.float32)

        if "name_email_similarity" in df.columns and "email_is_free" in df.columns:
            sim = pd.to_numeric(df["name_email_similarity"], errors="coerce").fillna(0.5).to_numpy()
            free = pd.to_numeric(df["email_is_free"], errors="coerce").fillna(0.0).to_numpy()
            out["email_mismatch_free"] = ((1.0 - sim) * free).astype(np.float32)

        if "date_of_birth_distinct_emails_4w" in df.columns and "name_email_similarity" in df.columns:
            dob_em = pd.to_numeric(df["date_of_birth_distinct_emails_4w"], errors="coerce").fillna(0.0).to_numpy()
            sim = pd.to_numeric(df["name_email_similarity"], errors="coerce").fillna(0.5).to_numpy()
            out["dob_emails_x_mismatch"] = (dob_em * (1.0 - sim)).astype(np.float32)

        thin_cols = [f"{c}_is_missing" for c in ["prev_address_months_count", "bank_months_count"] if f"{c}_is_missing" in out.columns]
        if thin_cols:
            out["thin_file_score"] = out[thin_cols].sum(axis=1).astype(np.float32)
        
        miss_cols = [f"{c}_is_missing" for c in SENTINEL_COLS if f"{c}_is_missing" in out.columns]
        out["n_missing"] = out[miss_cols].sum(axis=1).astype(np.float32)

        if "phone_home_valid" in df.columns and "phone_mobile_valid" in df.columns:
            ph = pd.to_numeric(df["phone_home_valid"], errors="coerce").fillna(0.0).to_numpy()
            pm = pd.to_numeric(df["phone_mobile_valid"], errors="coerce").fillna(0.0).to_numpy()
            out["no_valid_phone"] = ((ph == 0) & (pm == 0)).astype(np.float32)

        if "proposed_credit_limit" in df.columns and "income" in df.columns:
            lim = pd.to_numeric(df["proposed_credit_limit"], errors="coerce").fillna(500.0).to_numpy()
            inc = pd.to_numeric(df["income"], errors="coerce").fillna(0.5).to_numpy()
            out["limit_to_income"] = (lim / (inc + eps)).astype(np.float32)

        if "proposed_credit_limit" in df.columns and "credit_risk_score" in df.columns:
            lim = pd.to_numeric(df["proposed_credit_limit"], errors="coerce").fillna(500.0).to_numpy()
            c_risk = pd.to_numeric(df["credit_risk_score"], errors="coerce").fillna(0.0).to_numpy()
            out["limit_per_risk"] = (lim / (c_risk + 200.0)).astype(np.float32)
            out["risk_x_income"] = (c_risk * pd.to_numeric(df["income"], errors="coerce").fillna(0.5).to_numpy()).astype(np.float32)

        if "session_length_in_minutes" in df.columns and "device_distinct_emails_8w" in df.columns:
            sess = np.nan_to_num(out["session_length_in_minutes"].to_numpy(), nan=5.0)
            dev_em = np.nan_to_num(out["device_distinct_emails_8w"].to_numpy(), nan=1.0)
            out["emails_per_session_min"] = (dev_em / (sess + 1.0)).astype(np.float32)

        for col in CATEGORICAL_COLS:
            if col in df.columns:
                col_str = df[col].astype(str)
                known_cats = self.cat_categories.get(col, sorted(list(col_str.unique())))
                for cat in known_cats:
                    out[f"{col}_{cat}"] = (col_str == cat).astype(np.float32)

        if not self.feature_names:
            self.feature_names = list(out.columns)
        else:
            out = out.reindex(columns=self.feature_names, fill_value=0.0)

        return out
