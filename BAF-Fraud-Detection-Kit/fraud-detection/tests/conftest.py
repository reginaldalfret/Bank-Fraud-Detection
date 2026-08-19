"""
tests/conftest.py -- Master test configuration and fixtures.

Sets up:
- Isolated SQLite database for API testing.
- Pre-trained demo model artifacts for fast unit & integration tests.
- Synthetic BAF dataframe fixture for ML tests.
- TestClient and authenticated user tokens across all RBAC roles (ADMIN, FRAUD_ANALYST, RISK_MANAGER, AUDITOR, VIEWER).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FRAUD_DETECTION_ROOT = Path(__file__).resolve().parent.parent
TEST_DB_PATH = FRAUD_DETECTION_ROOT / "tests" / "test_fraud_api.db"

# Fresh isolated DB for test session
if TEST_DB_PATH.exists():
    try:
        TEST_DB_PATH.unlink()
    except Exception:
        pass

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["JWT_SECRET"] = "test-only-secret-not-for-production-0123456789"
os.environ["MFA_REQUIRED"] = "false"
os.environ["ENFORCE_MODEL_CHECKSUM"] = "false"
os.environ["LOGIN_RATE_LIMIT"] = "100/minute"
os.environ["PREDICT_RATE_LIMIT"] = "100/minute"
os.environ["CORS_ALLOW_ORIGINS"] = "http://localhost:3000,http://127.0.0.1:3000"
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = "test-admin@example.com"
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = "TestAdminPassword123!"

if str(FRAUD_DETECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAUD_DETECTION_ROOT))

ADMIN_EMAIL = "test-admin@example.com"
ADMIN_PASSWORD = "TestAdminPassword123!"


def _ensure_model_exists() -> None:
    models_dir = FRAUD_DETECTION_ROOT / "models"
    model_path = models_dir / "final_model.joblib"
    checksum_path = models_dir / "model_checksum.json"
    if not model_path.exists():
        from api.scripts import train_demo_model
        train_demo_model.main()
    if not checksum_path.exists():
        from api.scripts import record_model_checksum
        record_model_checksum.main()


_ensure_model_exists()

from api.main import app  # noqa: E402
from api.rate_limit import limiter  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from src.config import load_config  # noqa: E402

N = 2000


@pytest.fixture(scope="session")
def cfg():
    return load_config(FRAUD_DETECTION_ROOT / "config.yaml")


@pytest.fixture()
def synthetic_baf_df():
    rng = np.random.default_rng(42)
    n = N
    df = pd.DataFrame({
        "income": rng.uniform(0.1, 0.9, n),
        "name_email_similarity": rng.uniform(0, 1, n),
        "prev_address_months_count": rng.choice([-1] + list(range(0, 380)), n).astype(float),
        "current_address_months_count": rng.choice([-1] + list(range(0, 429)), n).astype(float),
        "customer_age": rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], n),
        "days_since_request": rng.uniform(0, 79, n),
        "intended_balcon_amount": rng.choice(list(range(-16, 0)) + list(range(0, 114)), n).astype(float),
        "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], n),
        "zip_count_4w": rng.integers(1, 6830, n),
        "velocity_6h": rng.uniform(-175, 16818, n),
        "velocity_24h": rng.uniform(1297, 9586, n),
        "velocity_4w": rng.uniform(2825, 7020, n),
        "bank_branch_count_8w": rng.integers(0, 2404, n),
        "date_of_birth_distinct_emails_4w": rng.integers(0, 39, n),
        "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE"], n),
        "credit_risk_score": rng.uniform(-191, 389, n),
        "email_is_free": rng.integers(0, 2, n),
        "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE"], n),
        "phone_home_valid": rng.integers(0, 2, n),
        "phone_mobile_valid": rng.integers(0, 2, n),
        "bank_months_count": rng.choice([-1] + list(range(0, 32)), n).astype(float),
        "has_other_cards": rng.integers(0, 2, n),
        "proposed_credit_limit": rng.uniform(200, 2000, n),
        "foreign_request": rng.integers(0, 2, n),
        "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[0.95, 0.05]),
        "session_length_in_minutes": rng.choice([-1] + list(range(0, 107)), n).astype(float),
        "device_os": rng.choice(["windows", "macintosh", "linux", "x11", "other"], n),
        "keep_alive_session": rng.integers(0, 2, n),
        "device_distinct_emails_8w": rng.choice([-1, 0, 1, 2], n).astype(float),
        "device_fraud_count": np.zeros(n),
        "month": rng.integers(0, 8, n),
    })
    fraud_prob = 0.011 + 0.05 * (df["credit_risk_score"] < -50).astype(float)
    df["fraud_bool"] = (rng.uniform(0, 1, n) < fraud_prob).astype(int)
    return df


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        from api.scripts import init_db as init_db_script
        init_db_script.main()
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    limiter.reset()
    yield


@pytest.fixture()
def admin_credentials():
    return ADMIN_EMAIL, ADMIN_PASSWORD


@pytest.fixture()
def admin_token(client: TestClient) -> str:
    r = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def make_user(client: TestClient, admin_token: str, email: str, password: str, role: str) -> str:
    r = client.post(
        "/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": email, "password": password, "role": role},
    )
    assert r.status_code in (200, 201, 409), r.text
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture()
def analyst_token(client: TestClient, admin_token: str) -> str:
    return make_user(client, admin_token, "analyst-fixture@example.com", "AnalystPass123!", "FRAUD_ANALYST")


@pytest.fixture()
def auditor_token(client: TestClient, admin_token: str) -> str:
    return make_user(client, admin_token, "auditor-fixture@example.com", "AuditorPass123!", "AUDITOR")


@pytest.fixture()
def viewer_token(client: TestClient, admin_token: str) -> str:
    return make_user(client, admin_token, "viewer-fixture@example.com", "ViewerPass123!", "VIEWER")
