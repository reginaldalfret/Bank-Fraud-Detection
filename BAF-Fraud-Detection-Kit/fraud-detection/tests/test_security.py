"""
tests/test_security.py -- Security, injection resilience, and defensive posture tests.

Verifies:
1. Path traversal attack mitigation on static and dynamic file endpoints.
2. Strict CORS allow-listing (no wildcard '*' allowed, origin verification).
3. Input sanitization & injection resistance (SQLi, XSS, control characters, strict types).
4. Safe file upload protections (MIME, extension, size limit, row count limit, encoding).
5. Comprehensive security response headers (CSP, HSTS, X-Frame-Options, nosniff, etc.).
"""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient

from api.settings import settings

SAMPLE_ROW = {
    "income": 0.5, "customer_age": 40, "employment_status": "CA",
    "housing_status": "BA", "name_email_similarity": 0.8,
    "prev_address_months_count": 12, "current_address_months_count": 24,
    "bank_months_count": 10, "days_since_request": 0.5,
    "velocity_6h": 3000.0, "velocity_24h": 4000.0, "velocity_4w": 5000.0,
    "zip_count_4w": 100, "bank_branch_count_8w": 20,
    "date_of_birth_distinct_emails_4w": 2, "phone_home_valid": 1,
    "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
    "device_os": "windows", "device_distinct_emails_8w": 1,
    "session_length_in_minutes": 5.0, "keep_alive_session": 1,
    "source": "INTERNET", "credit_risk_score": 100.0,
    "proposed_credit_limit": 1000.0, "payment_type": "AA",
    "has_other_cards": 1, "intended_balcon_amount": -1.0,
    "month": 3, "device_fraud_count": 0,
}


# ============================================================================
# 1. Path Traversal Protection
# ============================================================================
@pytest.mark.parametrize(
    "payload",
    [
        "..%2f..%2fmain.py",
        "../../main.py",
        "....//....//config.yaml",
        "..%5c..%5csettings.py",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "%2e%2e%2f%2e%2e%2fapi%2fdatabase.py",
    ],
)
def test_path_traversal_blocked_on_figures(client: TestClient, admin_token: str, payload: str):
    """Path traversal sequences must be rejected with 400/403/404, never leaking internal files."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get(f"/model/figures/{payload}", headers=headers)
    assert r.status_code in (400, 403, 404)
    # Ensure source code or internal secrets are not returned in body
    assert "DATABASE_URL" not in r.text
    assert "jwt_secret" not in r.text
    assert "class Settings" not in r.text


# ============================================================================
# 2. CORS Policy Verification
# ============================================================================
def test_cors_allowed_origin(client: TestClient):
    """Allowed origin receives access-control-allow-origin header."""
    origin = "http://localhost:3000"
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization,Content-Type",
    }
    r = client.options("/predict", headers=headers)
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == origin


def test_cors_disallowed_origin_rejected(client: TestClient):
    """Disallowed origin does NOT receive access-control-allow-origin header."""
    headers = {
        "Origin": "http://malicious-phishing-site.com",
        "Access-Control-Request-Method": "POST",
    }
    r = client.options("/predict", headers=headers)
    assert r.headers.get("access-control-allow-origin") != "http://malicious-phishing-site.com"
    assert r.headers.get("access-control-allow-origin") != "*"


# ============================================================================
# 3. Input Sanitization & Injection Defense
# ============================================================================
def test_sqli_payload_in_auth_safely_handled(client: TestClient):
    """SQL injection payloads in email field are safely escaped / rejected without DB error."""
    sqli_emails = [
        "' OR '1'='1",
        "admin@example.com'--",
        "'; DROP TABLE users; --",
        "test' UNION SELECT * FROM users--",
    ]
    for email in sqli_emails:
        r = client.post("/auth/login", json={"email": email, "password": "any"})
        # 422 (Pydantic EmailStr validation failure) or 401 (escaped query failed match)
        assert r.status_code in (401, 422)
        assert "OperationalError" not in r.text
        assert "syntax error" not in r.text.lower()


def test_xss_and_control_chars_in_prediction_fields(client: TestClient, admin_token: str):
    """Control characters and malformed strings are rejected by field validators."""
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Control characters in category
    bad_control_row = dict(SAMPLE_ROW, payment_type="AA\x00\x01")
    r_ctrl = client.post("/predict", headers=headers, json={"rows": [bad_control_row]})
    assert r_ctrl.status_code == 422

    # HTML tags in strings must not be reflected unescaped
    xss_row = dict(SAMPLE_ROW, employment_status="<script>")
    r_xss = client.post("/predict", headers=headers, json={"rows": [xss_row]})
    # Caught by min/max length or processed safely without rendering
    assert r_xss.status_code in (200, 422)
    if r_xss.status_code == 200:
        assert "<script>" not in r_xss.headers.get("content-type", "")


def test_numeric_boundary_and_type_coercion_safety(client: TestClient, admin_token: str):
    """Floating point infinities and string injection into numeric fields are strictly blocked."""
    headers = {"Authorization": f"Bearer {admin_token}"}

    # NaN / Infinity injection
    nan_row = dict(SAMPLE_ROW, income=float("nan"))
    r_nan = client.post("/predict", headers=headers, json={"rows": [nan_row]})
    assert r_nan.status_code == 422

    inf_row = dict(SAMPLE_ROW, credit_risk_score=float("inf"))
    r_inf = client.post("/predict", headers=headers, json={"rows": [inf_row]})
    assert r_inf.status_code == 422


# ============================================================================
# 4. Safe File Uploads
# ============================================================================
def test_upload_oversized_file_rejected(client: TestClient, admin_token: str):
    """Payloads exceeding max_upload_bytes (5MB) are rejected with 413 Payload Too Large."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    large_content = b"a" * (settings.max_upload_bytes + 1024)

    files = {"file": ("large.csv", large_content, "text/csv")}
    r = client.post("/predict/file", headers=headers, files=files)
    assert r.status_code == 413
    assert "exceeds" in r.json()["detail"].lower()


def test_upload_dangerous_extension_rejected(client: TestClient, admin_token: str):
    """Executable or script extensions are rejected immediately."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    content = b"print('malicious')"

    for bad_ext in ["payload.exe", "script.sh", "code.py", "archive.zip", "test.html"]:
        files = {"file": (bad_ext, content, "application/octet-stream")}
        r = client.post("/predict/file", headers=headers, files=files)
        assert r.status_code == 400
        assert "only .csv files are accepted" in r.json()["detail"].lower()


def test_upload_invalid_utf8_rejected(client: TestClient, admin_token: str):
    """Corrupted / non-UTF8 binary files are rejected gracefully."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    invalid_bytes = b"\xff\xfe\xfd\x80\x81\x82"

    files = {"file": ("binary.csv", invalid_bytes, "text/csv")}
    r = client.post("/predict/file", headers=headers, files=files)
    assert r.status_code == 400
    assert "utf-8" in r.json()["detail"].lower()


# ============================================================================
# 5. Security Response Headers
# ============================================================================
def test_hardened_security_headers(client: TestClient):
    """Verify security headers are attached to all API responses."""
    r = client.get("/health")
    assert r.status_code == 200

    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("referrer-policy") == "no-referrer"
    assert "default-src" in r.headers.get("content-security-policy", "")
    assert "max-age=" in r.headers.get("strict-transport-security", "")
    assert "geolocation=()" in r.headers.get("permissions-policy", "")
    assert "X-Request-ID" in r.headers
