"""Application Data Management and Provenance Service.

Maintains in-memory repository of application submissions, pre-seeds
sample benchmark records, handles caching, pagination, filtering,
and data provenance tracking.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.api.schemas import ApplicationRequest

logger = logging.getLogger("fraud_api.data_service")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DataService:
    """Enterprise Data Service for Bank Account Opening Applications."""

    def __init__(self):
        self._lock = threading.Lock()
        self.applications: Dict[str, Dict[str, Any]] = {}
        self.predictions_cache: Dict[str, Dict[str, Any]] = {}
        self.dataset_meta: Dict[str, Any] = {
            "dataset_name": "Bank Account Fraud (BAF) - Base Variant (NeurIPS 2022)",
            "domain": "Bank Account Opening Fraud Detection",
            "total_records": 1000000,
            "fraud_rate": 0.01103,
            "features_count": 31,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sha256": "3a8b29c91f08e434f0c746e50f5a709be734293f0b2f518e38d786016e789f21"
        }
        
        self._seed_initial_applications()

    def _seed_initial_applications(self):
        """Seed repository with realistic benchmark applications from sample data or defaults."""
        sample_paths = [
            WORKSPACE_ROOT / "BAF-Fraud-Detection-Kit" / "code" / "sample_train.csv",
            WORKSPACE_ROOT / "placement-round-fraud-kit" / "artifacts" / "labeled.csv",
        ]
        
        loaded = 0
        for path in sample_paths:
            if path.exists():
                try:
                    logger.info("Seeding initial applications from: %s", path)
                    # Load top 100 samples
                    df = pd.read_csv(path, nrows=100)
                    for i, row in df.iterrows():
                        app_id = f"APP-{10000 + i}"
                        row_dict = row.to_dict()
                        # Sanitize row dict
                        clean_dict = {
                            k: (None if pd.isna(v) else v)
                            for k, v in row_dict.items()
                            if not k.lower().startswith("unnamed")
                        }
                        clean_dict["application_id"] = app_id
                        self.applications[app_id] = clean_dict
                        loaded += 1
                    logger.info("Successfully seeded %d applications from %s", loaded, path)
                    break
                except Exception as ex:
                    logger.warning("Could not seed from %s: %s", path, ex)

        if loaded == 0:
            logger.info("Generating synthetic benchmark applications for seeding.")
            # Seed curated archetype cases:
            # 1. Normal legitimate customer
            self.save_application(ApplicationRequest(
                application_id="APP-10001",
                income=0.8,
                name_email_similarity=0.92,
                prev_address_months_count=48.0,
                current_address_months_count=72.0,
                customer_age=40,
                days_since_request=0.01,
                intended_balcon_amount=25.0,
                payment_type="AA",
                zip_count_4w=850.0,
                velocity_6h=3200.0,
                velocity_24h=3500.0,
                velocity_4w=3600.0,
                bank_branch_count_8w=5.0,
                date_of_birth_distinct_emails_4w=1.0,
                employment_status="CA",
                credit_risk_score=280.0,
                email_is_free=0,
                housing_status="BA",
                phone_home_valid=1,
                phone_mobile_valid=1,
                bank_months_count=24.0,
                has_other_cards=1,
                proposed_credit_limit=500.0,
                foreign_request=0,
                source="INTERNET",
                session_length_in_minutes=12.5,
                device_os="windows",
                keep_alive_session=1,
                device_distinct_emails_8w=1.0,
                device_fraud_count=0,
                month=6
            ).model_dump())

            # 2. Synthetic Identity Fraud Case
            self.save_application(ApplicationRequest(
                application_id="APP-10002",
                income=0.2,
                name_email_similarity=0.08,
                prev_address_months_count=-1.0,
                current_address_months_count=2.0,
                customer_age=30,
                days_since_request=0.02,
                intended_balcon_amount=-1.0,
                payment_type="AB",
                zip_count_4w=4500.0,
                velocity_6h=8200.0,
                velocity_24h=6000.0,
                velocity_4w=4000.0,
                bank_branch_count_8w=32.0,
                date_of_birth_distinct_emails_4w=14.0,
                employment_status="CB",
                credit_risk_score=-40.0,
                email_is_free=1,
                housing_status="BC",
                phone_home_valid=0,
                phone_mobile_valid=0,
                bank_months_count=-1.0,
                has_other_cards=0,
                proposed_credit_limit=1800.0,
                foreign_request=1,
                source="INTERNET",
                session_length_in_minutes=1.8,
                device_os="linux",
                keep_alive_session=0,
                device_distinct_emails_8w=3.0,
                device_fraud_count=0,
                month=7
            ).model_dump())

    def save_application(self, app_data: Dict[str, Any]) -> str:
        """Store or update an application in the repository."""
        with self._lock:
            app_id = app_data.get("application_id") or f"APP-{uuid.uuid4().hex[:10].upper()}"
            app_data["application_id"] = app_id
            app_data["stored_at"] = datetime.now(timezone.utc).isoformat()
            self.applications[app_id] = app_data
            return app_id

    def get_application(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve application by unique ID."""
        with self._lock:
            return self.applications.get(app_id)

    def list_applications(
        self,
        page: int = 1,
        page_size: int = 50,
        risk_level: Optional[str] = None,
        month: Optional[int] = None,
        min_probability: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List applications with pagination and filters."""
        with self._lock:
            items = list(self.applications.values())
            
        # Filtering
        if month is not None:
            items = [x for x in items if x.get("month") == month]
            
        if min_probability is not None:
            items = [
                x for x in items
                if self.predictions_cache.get(x["application_id"], {}).get("fraud_probability", 0.0) >= min_probability
            ]

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        return items[start:end], total

    def cache_prediction(self, app_id: str, prediction_res: Dict[str, Any]):
        """Cache latest inference output for quick lookup."""
        with self._lock:
            self.predictions_cache[app_id] = prediction_res

    def get_cached_prediction(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction for an application ID."""
        with self._lock:
            return self.predictions_cache.get(app_id)


_data_service_instance: Optional[DataService] = None


def get_data_service() -> DataService:
    global _data_service_instance
    if _data_service_instance is None:
        _data_service_instance = DataService()
    return _data_service_instance
