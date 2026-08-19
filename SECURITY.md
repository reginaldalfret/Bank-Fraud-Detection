# SECURITY POLICY & THREAT MODEL

## 1. Vulnerability Reporting
If you discover a security vulnerability in this project, please report it via private disclosure rather than opening a public issue.
- **Contact:** `reginaldalfret` (Repository Maintainer)
- **Response SLA:** Critical vulnerabilities evaluated within 24 hours.

---

## 2. Authentication & Access Control (RBAC)
The API enforces 4-tier Role-Based Access Control:
- **`ADMIN`:** Full access to model configuration, user management, threshold overrides, and audits.
- **`RISK_MANAGER`:** Access to detailed forensic SHAP attributions, batch simulation, and threshold recommendations.
- **`FRAUD_ANALYST`:** Investigation queue triage, applicant dossier inspection, and disposition review.
- **`AUDITOR`:** Read-only access to model performance cards, compliance metrics, and fairness audits.

---

## 3. Defense-in-Depth Mechanisms
1. **Secret Isolation:** Zero hardcoded API keys, private certificates, or database credentials exist in the codebase (verified via automated regex secret scanning).
2. **Pydantic v2 Input Validation:** Strict type enforcement and bounds checking on all 31 application features. Out-of-bounds inputs (e.g. `customer_age < 0` or `customer_age > 120`) reject with `422 Unprocessable Entity`.
3. **Path Traversal Defense:** Sanitization on all file ingestion and export endpoints preventing directory traversal (`..%2f`, `..\`).
4. **SQLi & XSS Mitigation:** Parametrized ORM queries and explicit JSON payload parsing prevent injection attacks.
5. **Rate Limiting:** IP-level and token-level rate limiting powered by SlowAPI on `/auth/login` and `/predict`.
6. **Encrypted TOTP Storage:** MFA secrets encrypted via symmetric Fernet keys prior to database persistence.
