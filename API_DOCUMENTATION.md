# Enterprise REST & OpenAPI Specification
### Bank Account Opening Fraud Detection & AI Forensics Platform

**API Version:** `2.0.0`  
**Base URL:** `http://127.0.0.1:8000`  
**OpenAPI Interactive UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
**ReDoc Reference:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)  
**Main Module:** [`src/api/main.py`](file:///e:/Fraud%20Detection/src/api/main.py)  

---

## 1. Endpoints Quick Reference (14 Enterprise Endpoints)

| # | HTTP Verb | Path | Tag | Description |
|:---:|:---:|:---|:---|:---|
| 1 | `GET` | `/api/health` | System Health | Service uptime, sub-service status, active model health |
| 2 | `GET` | `/api/meta` | Metadata | BAF schema, valid numeric ranges, sentinels, typologies |
| 3 | `GET` | `/api/model-info` | Model Governance | Active ensemble details, tree counts, calibration method |
| 4 | `POST` | `/api/predict` | Inference | Single application scoring, risk level, decision triage |
| 5 | `POST` | `/api/batch-predict` | Inference | Bulk scoring via JSON array or multipart CSV/Parquet upload |
| 6 | `GET` | `/api/applications` | Applications | Paginated application registry with risk and month filters |
| 7 | `GET` | `/api/applications/{id}` | Applications | Deep inspection: raw fields, engineered vector, model score |
| 8 | `POST` | `/api/explain` | Explainability | Local Tree SHAP attributions, positive and mitigating drivers |
| 9 | `GET` | `/api/model-comparison` | Model Governance | Multi-model benchmark evaluation matrix (7 models) |
| 10 | `GET` | `/api/metrics` | Model Governance | Test set metrics, confusion matrix, age fairness analysis |
| 11 | `GET` | `/api/ai/health` | AI & LLM | NVIDIA Nemotron connectivity and fallback engine readiness |
| 12 | `GET` | `/api/queue` | Investigation Queue | Paginated investigation queue and case management status |
| 13 | `POST` | `/api/queue/action` | Investigation Queue | Execute triage action (Review, Escalate, Confirm Fraud) |
| 14 | `GET` | `/api/queue/export` | Investigation Queue | Download current investigation queue as CSV file |

*Enterprise Alias Routes:*
- `GET /api/transactions` -> Alias for `/api/applications`
- `GET /api/transactions/{id}` -> Alias for `/api/applications/{id}`
- `POST /api/ai/analyze` -> Deep forensic briefing via Nemotron reasoning engine
- `GET /dashboard` and `GET /` -> Web Triage Console frontend

---

## 2. Detailed Endpoint Specifications

### 1. `GET /api/health`
Checks overall service health, sub-service operational readiness, model initialization, and process uptime.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/health"
```

#### Example Response (`200 OK`)
```json
{
  "status": "healthy",
  "timestamp": "2026-08-19T17:40:23.102Z",
  "version": "2.0.0",
  "uptime_seconds": 3612.45,
  "model_loaded": true,
  "model_name": "LightGBM-BAF-Champion",
  "services": {
    "model_service": "operational",
    "data_service": "operational",
    "threshold_service": "operational",
    "feature_service": "operational",
    "explanation_service": "operational",
    "queue_service": "operational",
    "nemotron_ai_service": "operational"
  }
}
```

---

### 2. `GET /api/meta`
Returns dataset metadata, canonical feature names, sentinel encodings, categorical levels, and fraud typology definitions.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/meta"
```

#### Example Response (`200 OK`)
```json
{
  "dataset_name": "Bank Account Fraud (BAF) - Base Variant (NeurIPS 2022)",
  "domain": "Bank Account Opening Fraud Detection (Applications, not Transactions)",
  "total_raw_features": 31,
  "total_engineered_features": 72,
  "sentinel_columns": [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "intended_balcon_amount"
  ],
  "categorical_columns": [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os"
  ],
  "protected_attributes": [
    "customer_age"
  ],
  "fraud_typologies": [
    {
      "name": "Synthetic Identity",
      "description": "Fabricated identities combining disparate PII. Betrayed by thin address/banking history and email-name mismatch."
    },
    {
      "name": "Identity Theft",
      "description": "Compromised real consumer credentials. Betrayed by device/session discrepancies and unverified phone contactability."
    },
    {
      "name": "Mule Account Farming",
      "description": "Organized criminal syndicates opening accounts in bulk. Betrayed by 6h/4w velocity bursts and branch/ZIP clustering."
    },
    {
      "name": "Financial Incoherence",
      "description": "Excessive credit limit requests disproportionate to applicant income decile or adverse internal credit ratings."
    }
  ],
  "field_ranges": {
    "income": { "min": 0.1, "max": 0.9, "step": 0.1 },
    "name_email_similarity": { "min": 0.0, "max": 1.0, "step": 0.01 },
    "customer_age": { "min": 10.0, "max": 90.0, "step": 10.0 },
    "proposed_credit_limit": { "min": 200.0, "max": 2000.0, "step": 50.0 }
  }
}
```

---

### 3. `GET /api/model-info`
Retrieves model architecture metadata, tree ensemble specifications, Platt calibration profile, and global feature importance.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/model-info"
```

#### Example Response (`200 OK`)
```json
{
  "model_name": "LightGBM-BAF-Champion",
  "model_version": "v2026.1-production",
  "model_type": "Gradient Boosted Decision Trees (Ensemble)",
  "trees_count": 100,
  "calibration_method": "Platt Sigmoid / Temperature Scaling",
  "training_protocol": "Temporal Split (Months 0-5 Train, Months 6-7 Out-of-Time Test)",
  "benchmark_eval": {
    "roc_auc": 0.8985,
    "pr_auc": 0.1675,
    "tpr_at_5pct_fpr": 0.5536,
    "positive_rate": 0.01103,
    "n": 300000
  },
  "threshold_profiles": {
    "balanced": { "threshold": 0.0446, "block_threshold": 0.08, "review_threshold": 0.015 },
    "strict": { "threshold": 0.025, "block_threshold": 0.05, "review_threshold": 0.01 },
    "conservative": { "threshold": 0.075, "block_threshold": 0.12, "review_threshold": 0.03 }
  },
  "top_feature_importance": [
    { "feature": "housing_status_BC", "split_count": 142, "importance_score": 0.142 },
    { "feature": "credit_risk_score", "split_count": 128, "importance_score": 0.128 },
    { "feature": "name_email_similarity", "split_count": 115, "importance_score": 0.115 }
  ],
  "total_features": 72
}
```

---

### 4. `POST /api/predict`
Scores a single account opening application, returning calibrated fraud probability, operational decision (`APPROVE`, `REVIEW`, `BLOCK`), risk level, and top local risk factors.

#### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "income": 0.1,
       "name_email_similarity": 0.05,
       "prev_address_months_count": -1,
       "current_address_months_count": 1,
       "customer_age": 22,
       "days_since_request": 10.5,
       "intended_balcon_amount": 100.0,
       "payment_type": "AC",
       "zip_count_4w": 2500,
       "velocity_6h": 9800.0,
       "velocity_24h": 15000.0,
       "velocity_4w": 20000.0,
       "bank_branch_count_8w": 150,
       "date_of_birth_distinct_emails_4w": 8,
       "employment_status": "CE",
       "credit_risk_score": -180.0,
       "email_is_free": 1,
       "housing_status": "BE",
       "phone_home_valid": 0,
       "phone_mobile_valid": 0,
       "bank_months_count": -1,
       "has_other_cards": 0,
       "proposed_credit_limit": 2000.0,
       "foreign_request": 1,
       "source": "INTERNET",
       "session_length_in_minutes": 1.0,
       "device_os": "other",
       "keep_alive_session": 0,
       "device_distinct_emails_8w": 2,
       "month": 5
     }'
```

#### Example Response (`200 OK`)
```json
{
  "application_id": "APP-20260819-001",
  "fraud_probability": 0.1485,
  "fraud_prediction": 1,
  "risk_level": "CRITICAL",
  "action": "BLOCK",
  "threshold_used": 0.0446,
  "threshold_profile": "balanced",
  "model_name": "LightGBM-BAF-Champion",
  "model_version": "v2026.1-production",
  "top_risk_factors": [
    {
      "signal_name": "Synthetic Identity Indicator",
      "feature_name": "name_email_similarity",
      "value": 0.05,
      "risk_impact": "positive",
      "score_delta": 0.35,
      "description": "Low applicant name/email similarity on a free provider suggests machine-generated identity"
    },
    {
      "signal_name": "Thin File (Missing Prior Address)",
      "feature_name": "prev_address_months_count",
      "value": "Missing (-1)",
      "risk_impact": "positive",
      "score_delta": 0.28,
      "description": "No verifiable prior residential tenure available on record"
    },
    {
      "signal_name": "DOB Email Farming Cluster",
      "feature_name": "date_of_birth_distinct_emails_4w",
      "value": 8,
      "risk_impact": "positive",
      "score_delta": 0.40,
      "description": "8 distinct emails sharing same DOB in 4 weeks indicates automated application mill"
    }
  ],
  "timestamp": "2026-08-19T17:40:23.210Z",
  "latency_ms": 1.35
}
```

---

### 5. `POST /api/batch-predict`
Processes batches of applications via JSON array or multipart CSV/Parquet file upload with chunked memory streaming.

#### Example Request (Multipart CSV Upload)
```bash
curl -X POST "http://127.0.0.1:8000/api/batch-predict?threshold_profile=balanced" \
     -F "file=@sample_applications.csv"
```

#### Example Response (`200 OK`)
```json
{
  "total_applications": 500,
  "approved_count": 472,
  "review_count": 18,
  "blocked_count": 10,
  "average_fraud_probability": 0.01214,
  "processing_time_ms": 48.60,
  "threshold_profile_used": "balanced",
  "predictions": [
    {
      "application_id": "APP-20260819-001",
      "fraud_probability": 0.1485,
      "fraud_prediction": 1,
      "risk_level": "CRITICAL",
      "action": "BLOCK",
      "threshold_used": 0.0446,
      "threshold_profile": "balanced",
      "model_name": "LightGBM-BAF-Champion",
      "model_version": "v2026.1-production",
      "top_risk_factors": [],
      "timestamp": "2026-08-19T17:40:23.250Z",
      "latency_ms": 0.09
    }
  ]
}
```

---

### 6. `GET /api/applications` & `GET /api/transactions`
Retrieves a paginated list of applications with optional filtering by risk tier, month, or minimum fraud score.

#### Query Parameters
- `page` (integer, default: 1)
- `page_size` (integer, default: 50, max: 500)
- `risk_level` (string, optional: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
- `month` (integer, optional: `0` to `7`)
- `min_probability` (float, optional: `0.0` to `1.0`)

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/applications?risk_level=CRITICAL&page=1&page_size=10"
```

---

### 7. `GET /api/applications/{id}` & `GET /api/transactions/{id}`
Inspects a specific application by ID, providing raw attributes, 72-dimension engineered feature vector, live score, and forensic signals.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/applications/APP-20260819-001"
```

---

### 8. `POST /api/explain`
Computes local Tree SHAP feature attributions, identifying positive risk drivers and mitigating trust indicators.

#### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/api/explain" \
     -H "Content-Type: application/json" \
     -d '{
       "income": 0.7,
       "name_email_similarity": 0.95,
       "prev_address_months_count": 60,
       "current_address_months_count": 120,
       "customer_age": 45,
       "credit_risk_score": 280.0,
       "proposed_credit_limit": 1500.0,
       "phone_home_valid": 1,
       "phone_mobile_valid": 1
     }'
```

#### Example Response (`200 OK`)
```json
{
  "application_id": "APP-20260819-0045",
  "base_value": -4.50,
  "output_value": -6.20,
  "fraud_probability": 0.00202,
  "top_positive_factors": [
    {
      "feature": "proposed_credit_limit",
      "value": 1500.0,
      "attribution": 0.12,
      "description": "Requested credit limit adds minor baseline exposure"
    }
  ],
  "top_mitigating_factors": [
    {
      "feature": "name_email_similarity",
      "value": 0.95,
      "attribution": -0.85,
      "description": "High name-email concordance strongly indicates authentic identity"
    },
    {
      "feature": "credit_risk_score",
      "value": 280.0,
      "attribution": -0.62,
      "description": "Favorable internal credit rating"
    }
  ],
  "summary": "Application demonstrates high identity coherence and strong residential stability. Low fraud risk."
}
```

---

### 9. `GET /api/model-comparison`
Returns the multi-model benchmark evaluation matrix comparing all 7 model families on the BAF Base dataset.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/model-comparison"
```

---

### 10. `GET /api/metrics`
Returns production model test metrics, confusion matrix, and demographic fairness evaluation across age cohorts.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/metrics"
```

---

### 11. `GET /api/ai/health`
Checks connectivity to NVIDIA Nemotron LLM endpoint and readiness of the offline deterministic fallback reasoning engine.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/ai/health"
```

#### Example Response (`200 OK`)
```json
{
  "status": "online",
  "provider": "nvidia-nemotron-70b",
  "model": "nvidia/nemotron-4-340b-instruct",
  "has_api_credentials": true,
  "features": [
    "Synthetic Identity Detection",
    "Identity Theft Forensic Triangulation",
    "Mule Farm Burst Recognition",
    "Financial Incoherence Auditing",
    "Automated Analyst Checklist Generation"
  ]
}
```

---

### 12. `GET /api/queue`
Retrieves paginated investigation queue items with triage statistics and filters by status, risk level, or analyst.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/queue?status=PENDING&risk_level=CRITICAL"
```

---

### 13. `POST /api/queue/action`
Executes an analyst triage disposition action on an investigation queue item.

#### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/api/queue/action" \
     -H "Content-Type: application/json" \
     -d '{
       "application_id": "APP-20260819-001",
       "action": "CONFIRM_FRAUD",
       "analyst_id": "ANALYST-94",
       "notes": "Verified synthetic identity: invalid phone carrier, disposable email cluster detected."
     }'
```

---

### 14. `GET /api/queue/export`
Exports current investigation queue to a downloadable CSV file.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/api/queue/export" -o investigation_queue.csv
```
