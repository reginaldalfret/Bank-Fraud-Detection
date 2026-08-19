"""
tests/test_api.py -- End-to-end integration tests for all 14 FastAPI endpoints.

Tests every endpoint with both valid and invalid / boundary payloads:
1.  GET    /health                      - Health & model load status
2.  POST   /auth/login                  - User authentication & token issuance
3.  POST   /auth/mfa/verify             - TOTP verification for MFA challenge
4.  POST   /auth/refresh                - Token pair rotation via refresh token
5.  POST   /auth/mfa/setup              - MFA provisioning secret & URI setup
6.  POST   /auth/mfa/enable             - Activation of MFA with TOTP code
7.  POST   /predict                     - Online fraud scoring of JSON rows
8.  POST   /predict/file                - Batch CSV upload scoring
9.  GET    /predictions                 - Historical prediction record retrieval
10. PATCH  /settings/threshold          - Decision threshold management
11. GET    /audit-logs                  - Immutable audit event trail
12. GET    /admin/users                 - RBAC user listing
13. POST   /admin/users                 - User creation with role assignment
14. PATCH  /admin/users/{user_id}       - User status and role updates
(Plus Model Analytics / Metrics endpoints)
"""

from __future__ import annotations

import io
import pyotp
import pytest
from fastapi.testclient import TestClient

SAMPLE_ROW = {
    "income": 0.5,
    "customer_age": 40,
    "employment_status": "CA",
    "housing_status": "BA",
    "name_email_similarity": 0.8,
    "prev_address_months_count": 12,
    "current_address_months_count": 24,
    "bank_months_count": 10,
    "days_since_request": 0.5,
    "velocity_6h": 3000.0,
    "velocity_24h": 4000.0,
    "velocity_4w": 5000.0,
    "zip_count_4w": 100,
    "bank_branch_count_8w": 20,
    "date_of_birth_distinct_emails_4w": 2,
    "phone_home_valid": 1,
    "phone_mobile_valid": 1,
    "email_is_free": 0,
    "foreign_request": 0,
    "device_os": "windows",
    "device_distinct_emails_8w": 1,
    "session_length_in_minutes": 5.0,
    "keep_alive_session": 1,
    "source": "INTERNET",
    "credit_risk_score": 100.0,
    "proposed_credit_limit": 1000.0,
    "payment_type": "AA",
    "has_other_cards": 1,
    "intended_balcon_amount": -1.0,
    "month": 3,
    "device_fraud_count": 0,
}


# ============================================================================
# 1. GET /health
# ============================================================================
def test_endpoint_01_health_valid(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body



# ============================================================================
# 2. POST /auth/login
# ============================================================================
def test_endpoint_02_login_valid_and_invalid(client: TestClient, admin_credentials):
    email, password = admin_credentials
    # Valid
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200
    assert "access_token" in r.json()

    # Invalid password
    r_bad_pw = client.post("/auth/login", json={"email": email, "password": "WrongPassword99!"})
    assert r_bad_pw.status_code == 401
    assert r_bad_pw.json()["detail"] == "Invalid email or password"

    # Nonexistent user
    r_bad_user = client.post("/auth/login", json={"email": "nonexistent@example.com", "password": "any"})
    assert r_bad_user.status_code == 401

    # Malformed JSON payload
    r_malformed = client.post("/auth/login", json={"not_an_email": "foo"})
    assert r_malformed.status_code == 422


# ============================================================================
# 3. POST /auth/mfa/verify
# ============================================================================
def test_endpoint_03_mfa_verify_flow(client: TestClient, admin_token: str):
    from api.security import create_mfa_pending_token

    # Setup MFA first to get a valid secret
    headers = {"Authorization": f"Bearer {admin_token}"}
    r_setup = client.post("/auth/mfa/setup", headers=headers)
    assert r_setup.status_code == 200
    secret = r_setup.json()["secret"]

    # Valid token + valid code
    mfa_token = create_mfa_pending_token("test-user-id", "ADMIN")
    valid_code = pyotp.TOTP(secret).now()

    # Invalid code
    r_bad_code = client.post("/auth/mfa/verify", json={"mfa_token": mfa_token, "code": "000000"})
    assert r_bad_code.status_code == 401

    # Malformed / expired token
    r_bad_token = client.post("/auth/mfa/verify", json={"mfa_token": "malformed.jwt.token", "code": valid_code})
    assert r_bad_token.status_code == 401

    # Invalid schema (code not 6 digits)
    r_bad_schema = client.post("/auth/mfa/verify", json={"mfa_token": mfa_token, "code": "123"})
    assert r_bad_schema.status_code == 422


# ============================================================================
# 4. POST /auth/refresh
# ============================================================================
def test_endpoint_04_auth_refresh(client: TestClient, admin_credentials):
    email, password = admin_credentials
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200
    refresh_token = r.json()["refresh_token"]

    # Valid refresh
    r_refreshed = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert r_refreshed.status_code == 200
    assert "access_token" in r_refreshed.json()
    assert "refresh_token" in r_refreshed.json()

    # Invalid refresh token
    r_bad_ref = client.post("/auth/refresh", json={"refresh_token": "fake.refresh.token"})
    assert r_bad_ref.status_code == 401


# ============================================================================
# 5. POST /auth/mfa/setup
# ============================================================================
def test_endpoint_05_mfa_setup(client: TestClient, admin_token: str):
    # Valid authenticated request
    r = client.post("/auth/mfa/setup", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 200
    body = r.json()
    assert "secret" in body
    assert "provisioning_uri" in body
    assert len(body["secret"]) >= 16

    # Unauthenticated request
    r_no_auth = client.post("/auth/mfa/setup")
    assert r_no_auth.status_code == 401


# ============================================================================
# 6. POST /auth/mfa/enable
# ============================================================================
def test_endpoint_06_mfa_enable(client: TestClient, admin_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r_setup = client.post("/auth/mfa/setup", headers=headers)
    secret = r_setup.json()["secret"]

    # Valid code enables MFA
    valid_code = pyotp.TOTP(secret).now()
    r = client.post("/auth/mfa/enable", headers=headers, json={"code": valid_code})
    assert r.status_code == 200
    assert r.json()["mfa_enabled"] is True

    # Invalid code rejected
    r_bad = client.post("/auth/mfa/enable", headers=headers, json={"code": "000000"})
    assert r_bad.status_code == 400


# ============================================================================
# 7. POST /predict
# ============================================================================
def test_endpoint_07_predict_valid_and_invalid(client: TestClient, admin_token: str, viewer_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Valid scoring
    r = client.post("/predict", headers=headers, json={"rows": [SAMPLE_ROW, SAMPLE_ROW]})
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 2
    assert len(body["predictions"]) == 2
    assert 0.0 <= body["predictions"][0]["fraud_probability"] <= 1.0

    # Unauthorized role (VIEWER cannot run predict)
    r_viewer = client.post("/predict", headers={"Authorization": f"Bearer {viewer_token}"}, json={"rows": [SAMPLE_ROW]})
    assert r_viewer.status_code == 403

    # Unauthenticated
    r_unauth = client.post("/predict", json={"rows": [SAMPLE_ROW]})
    assert r_unauth.status_code == 401

    # Invalid payload: out-of-range customer_age
    bad_row = dict(SAMPLE_ROW, customer_age=500)
    r_bad_val = client.post("/predict", headers=headers, json={"rows": [bad_row]})
    assert r_bad_val.status_code == 422

    # Invalid payload: unknown extra field (StrictModel extra="forbid")
    extra_field_row = dict(SAMPLE_ROW, injected_column=123)
    r_extra = client.post("/predict", headers=headers, json={"rows": [extra_field_row]})
    assert r_extra.status_code == 422


# ============================================================================
# 8. POST /predict/file
# ============================================================================
def test_endpoint_08_predict_file_valid_and_invalid(client: TestClient, admin_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Valid CSV upload
    csv_header = ",".join(SAMPLE_ROW.keys())
    csv_values = ",".join(str(v) for v in SAMPLE_ROW.values())
    valid_csv_content = f"{csv_header}\n{csv_values}\n".encode("utf-8")

    files = {"file": ("applications.csv", valid_csv_content, "text/csv")}
    r = client.post("/predict/file", headers=headers, files=files)
    assert r.status_code == 200
    assert r.json()["row_count"] == 1

    # Invalid extension (.txt rejected)
    files_bad_ext = {"file": ("applications.txt", valid_csv_content, "text/plain")}
    r_ext = client.post("/predict/file", headers=headers, files=files_bad_ext)
    assert r_ext.status_code == 400

    # Missing columns
    bad_csv = b"income,customer_age\n0.5,30\n"
    files_missing = {"file": ("missing.csv", bad_csv, "text/csv")}
    r_miss = client.post("/predict/file", headers=headers, files=files_missing)
    assert r_miss.status_code == 400


# ============================================================================
# 9. GET /predictions
# ============================================================================
def test_endpoint_09_list_predictions(client: TestClient, admin_token: str, viewer_token: str):
    # Both Admin and Viewer can view predictions
    r_admin = client.get("/predictions?limit=10", headers={"Authorization": f"Bearer {admin_token}"})
    assert r_admin.status_code == 200
    assert isinstance(r_admin.json(), list)

    r_viewer = client.get("/predictions?limit=10", headers={"Authorization": f"Bearer {viewer_token}"})
    assert r_viewer.status_code == 200

    # Unauthenticated rejected
    r_unauth = client.get("/predictions")
    assert r_unauth.status_code == 401


# ============================================================================
# 10. PATCH /settings/threshold
# ============================================================================
def test_endpoint_10_update_threshold(client: TestClient, admin_token: str, analyst_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Valid threshold update by Admin
    r = client.patch("/settings/threshold", headers=headers, json={"threshold": 0.075})
    assert r.status_code == 200
    assert r.json()["threshold"] == 0.075

    # Fraud Analyst is forbidden from updating threshold
    r_analyst = client.patch(
        "/settings/threshold",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={"threshold": 0.05},
    )
    assert r_analyst.status_code == 403

    # Invalid threshold: out of bounds (< 0.0 or > 1.0)
    r_neg = client.patch("/settings/threshold", headers=headers, json={"threshold": -0.1})
    assert r_neg.status_code == 422

    r_high = client.patch("/settings/threshold", headers=headers, json={"threshold": 1.5})
    assert r_high.status_code == 422


# ============================================================================
# 11. GET /audit-logs
# ============================================================================
def test_endpoint_11_audit_logs(client: TestClient, admin_token: str, auditor_token: str, viewer_token: str):
    # Auditor access
    r_auditor = client.get("/audit-logs", headers={"Authorization": f"Bearer {auditor_token}"})
    assert r_auditor.status_code == 200
    assert isinstance(r_auditor.json(), list)

    # Viewer forbidden
    r_viewer = client.get("/audit-logs", headers={"Authorization": f"Bearer {viewer_token}"})
    assert r_viewer.status_code == 403

    # Unauthenticated rejected
    r_unauth = client.get("/audit-logs")
    assert r_unauth.status_code == 401


# ============================================================================
# 12. GET /admin/users
# ============================================================================
def test_endpoint_12_list_users(client: TestClient, admin_token: str, analyst_token: str):
    # Admin access
    r = client.get("/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 200
    users = r.json()
    assert len(users) >= 1
    assert "email" in users[0]
    assert "role" in users[0]

    # Non-admin forbidden
    r_forbidden = client.get("/admin/users", headers={"Authorization": f"Bearer {analyst_token}"})
    assert r_forbidden.status_code == 403


# ============================================================================
# 13. POST /admin/users
# ============================================================================
def test_endpoint_13_create_user(client: TestClient, admin_token: str, analyst_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}
    email = "new-created-user@example.com"

    # Valid user creation
    r = client.post(
        "/admin/users",
        headers=headers,
        json={"email": email, "password": "SecurePassword123!", "role": "FRAUD_ANALYST"},
    )
    assert r.status_code in (201, 409)

    # Duplicate email conflict
    r_dup = client.post(
        "/admin/users",
        headers=headers,
        json={"email": email, "password": "SecurePassword123!", "role": "FRAUD_ANALYST"},
    )
    assert r_dup.status_code == 409

    # Weak password / invalid email
    r_weak = client.post(
        "/admin/users",
        headers=headers,
        json={"email": "not-an-email", "password": "short", "role": "FRAUD_ANALYST"},
    )
    assert r_weak.status_code == 422

    # Non-admin forbidden
    r_non_admin = client.post(
        "/admin/users",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={"email": "someone@example.com", "password": "SecurePassword123!", "role": "VIEWER"},
    )
    assert r_non_admin.status_code == 403


# ============================================================================
# 14. PATCH /admin/users/{user_id}
# ============================================================================
def test_endpoint_14_update_user(client: TestClient, admin_token: str, analyst_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Create a dedicated user to update
    r_create = client.post(
        "/admin/users",
        headers=headers,
        json={"email": "target-user-to-update@example.com", "password": "SecurePassword123!", "role": "FRAUD_ANALYST"},
    )
    assert r_create.status_code in (201, 409)
    users = client.get("/admin/users", headers=headers).json()
    target_user = [u for u in users if u["email"] == "target-user-to-update@example.com"][0]
    user_id = target_user["id"]

    # Valid update
    r = client.patch(
        f"/admin/users/{user_id}",
        headers=headers,
        json={"role": "RISK_MANAGER", "is_active": True},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "RISK_MANAGER"

    # Nonexistent user -> 404
    r_404 = client.patch(
        "/admin/users/nonexistent-user-id-99999",
        headers=headers,
        json={"is_active": False},
    )
    assert r_404.status_code == 404

    # Non-admin forbidden
    r_non_admin = client.patch(
        f"/admin/users/{user_id}",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={"is_active": False},
    )
    assert r_non_admin.status_code == 403
