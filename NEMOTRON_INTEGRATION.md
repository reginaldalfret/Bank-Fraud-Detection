# NVIDIA Nemotron Local AI Analyst Integration
### Architecture, Structured Evidence Contracts, and Zero-Downtime Offline Fallback

**Document Reference:** `NEM-2026-INTEGRATION-V2`  
**Client Module:** [`src/nemotron_client.py`](file:///e:/Fraud%20Detection/src/nemotron_client.py) & [`src/api/services/nemotron_service.py`](file:///e:/Fraud%20Detection/src/api/services/nemotron_service.py)  
**Supported Models:** `nvidia/nemotron-4-340b-instruct`, `nvidia/nemotron-mini-4b-instruct`, and compatible local OpenAI-format LLM endpoints  

---

## 1. System Overview & Problem Statement

While machine learning models output numerical fraud probabilities and SHAP score deltas, human fraud analysts require **contextual forensic briefings** explaining *why* an account opening application is suspicious and *what specific KYC verification steps* must be taken.

The **Nemotron Local AI Analyst** bridges this gap by synthesizing structured tabular evidence into forensic investigation reports.

```
+----------------------------------------------------------------------------------------------------+
|                                    NEMOTRON INTEGRATION ARCHITECTURE                               |
+----------------------------------------------------------------------------------------------------+

   [ Inbound Application Request ]
                 |
                 v
   +-----------------------------+
   |  Model & SHAP Engine        | ---> Calibrated Probability, Risk Tier, Top 10 Local SHAP Values
   +-------------+---------------+
                 |
                 v
   +-----------------------------+
   | Structured Evidence Builder | ---> Assembles Evidence Contract: PII attributes, velocity ratios,
   +-------------+---------------+      contactability checks, thin-file flags, policy triggers
                 |
                 v
   +-----------------------------+
   |   Nemotron Routing Gate     |
   +-------------+---------------+
                 |
        +--------+--------+
        |                 |
  (Endpoint Online)  (Offline / Timeout / Malformed)
        |                 |
        v                 v
+---------------+ +---------------+
| Online LLM    | | Deterministic | ---> Guaranteed zero-downtime offline reasoning engine
| Nemotron Call | | Report Engine |      synthesizes domain rules directly from SHAP evidence
+-------+-------+ +-------+-------+
        |                 |
        +--------+--------+
                 |
                 v
   +-----------------------------+
   | Validated Structured JSON   | ---> Standardized AnalystInvestigationReport
   +-----------------------------+
```

---

## 2. Structured Evidence Contract

Before invoking the AI engine, the pipeline constructs a standardized **Evidence Package** containing mathematical features, domain policy triggers, and behavioral telemetry.

### 2.1 Evidence Schema Definition

```json
{
  "application_id": "APP-20260819-0042",
  "fraud_probability": 0.1245,
  "decision_threshold": 0.0446,
  "risk_level": "CRITICAL",
  "confidence_tier": "HIGH_CONVICTION",
  "top_fraud_drivers": [
    {
      "feature": "email_mismatch_free",
      "display_name": "Email Mismatch on Free Host",
      "shap_value": 0.384,
      "feature_value": 0.95,
      "domain_explanation": "Severe disparity between legal name and email address on disposable domain."
    },
    {
      "feature": "prev_address_months_count_is_missing",
      "display_name": "Thin File (Missing Prior Address)",
      "shap_value": 0.292,
      "feature_value": 1.0,
      "domain_explanation": "No prior residential tenure on file with credit bureau."
    }
  ],
  "top_mitigating_factors": [
    {
      "feature": "income",
      "display_name": "Declared Income Decile",
      "shap_value": -0.115,
      "feature_value": 0.70
    }
  ],
  "triggered_risk_flags": [
    "FLAG_SYNTHETIC_IDENTITY_DISPARITY",
    "FLAG_ZERO_CARRIER_VERIFIED_PHONES",
    "FLAG_VELOCITY_SPIKE_6H"
  ],
  "applicant_summary_metrics": {
    "income_rank": 0.7,
    "proposed_credit_limit": 2000.0,
    "phone_contacts_valid": "Home=0, Mobile=0",
    "session_length_minutes": 1.2
  }
}
```

---

## 3. Standardized Output Specification (`AnalystInvestigationReport`)

Both the live LLM endpoint and the offline deterministic fallback adhere strictly to the same output schema:

```json
{
  "investigation_id": "INV-7A4B9F2C10E8",
  "application_id": "APP-20260819-0042",
  "timestamp": "2026-08-19T17:40:23Z",
  "fraud_probability": 0.1245,
  "decision_threshold": 0.0446,
  "model_prediction": "SUSPECTED_FRAUD",
  "investigation_priority": "CRITICAL_IMMEDIATE_ACTION",
  "disposition_recommendation": "DECLINE_FRAUD_SUSPECTED",
  "executive_summary": "Application APP-20260819-0042 presents critical fraud exposure (Model Score: 12.45%, Threshold: 4.46%). The applicant demonstrates severe synthetic identity signals, characterized by a completely disconnected email handle on a free provider combined with a zero-tenure residential history and unreachable carrier phone records. Immediate decline and SAR filing recommended.",
  "typology_analysis": {
    "synthetic_identity": {
      "risk_score": 0.88,
      "level": "CRITICAL",
      "notes": [
        "Severe applicant name vs email address disparity on a free email host",
        "No prior residential address history found (thin credit bureau file)"
      ]
    },
    "identity_theft": {
      "risk_score": 0.65,
      "level": "ELEVATED",
      "notes": [
        "Both home and mobile phone numbers failed carrier KYC lookup"
      ]
    },
    "mule_farming": {
      "risk_score": 0.72,
      "level": "CRITICAL",
      "notes": [
        "6-hour application velocity spike (2.4x baseline)"
      ]
    },
    "financial_incoherence": {
      "risk_score": 0.30,
      "level": "LOW",
      "notes": [
        "Requested credit line within broad income decile parameters"
      ]
    }
  },
  "primary_risk_factors": [
    "Synthetic Identity: Disconnected email format indicating programmatic identity generation",
    "Thin File: Applicant lacks verifiable residential tenure records",
    "Contactability Failure: Both home and mobile numbers failed carrier KYC lookup",
    "Velocity Anomaly: 6-hour burst is 2.4x higher than 4-week baseline"
  ],
  "mitigating_factors": [
    "Established high income tier (Decile 0.7)"
  ],
  "recommended_verification_checklist": [
    "Verify device fingerprint against blacklisted hardware clusters",
    "Issue SAR (Suspicious Activity Report) notification for synthetic profile syndicate",
    "Place applicant email and identity indicators onto organizational blocklist",
    "Cross-reference recent accounts opened from same branch and ZIP zone"
  ],
  "confidence_score": 0.94,
  "engine_mode": "NEMOTRON_LLM",
  "metadata": {
    "model_name": "nvidia/nemotron-4-340b-instruct"
  }
}
```

---

## 4. Zero-Downtime Deterministic Fallback Engine

A critical requirement in banking operations is **100% API availability**. If an LLM endpoint fails, the system must not return HTTP 500 errors or halt onboarding queues.

The [`DeterministicReportGenerator`](file:///e:/Fraud%20Detection/src/nemotron_client.py#L123-L284) implements an offline expert reasoning engine:

### 4.1 Fallback Triggers
1. **Network Disconnection / Connection Refused (`httpx.ConnectError`)**
2. **Read Timeout (`httpx.TimeoutException` exceeding `NEMOTRON_TIMEOUT`)**
3. **HTTP 5xx Server Errors from LLM host**
4. **Malformed JSON or Code Block Output from LLM**
5. **Missing API Credentials in Offline Air-Gapped Environments**

### 4.2 Fallback Synthesis Logic
- **Priority Calculation:** Evaluates $P(\text{fraud})$ against operational threshold bands.
- **Typology Scorer:** Audits multi-column feature flags (e.g. `email_mismatch_free > 0.5` triggers Synthetic Identity elevated score).
- **Checklist Generator:** Synthesizes actionable KYC verification steps (e.g. carrier phone check, paystub verification, proof of address) based directly on the triggered risk drivers.

---

## 5. Local Endpoint Configuration & Environment Variables

Configure the client via environment variables:

| Environment Variable | Default Value | Description |
|:---|:---|:---|
| `NEMOTRON_BASE_URL` | `http://127.0.0.1:8000/v1` | Local or remote LLM endpoint base URL |
| `NEMOTRON_MODEL` | `nvidia/nemotron-mini-4b-instruct` | LLM model identifier string |
| `NEMOTRON_TIMEOUT` | `10.0` | HTTP request timeout in seconds |
| `NEMOTRON_MAX_RETRIES` | `2` | Number of exponential backoff retry attempts |
| `NEMOTRON_API_KEY` | `EMPTY` | API Key (or `EMPTY` for local vLLM / Ollama servers) |
| `NEMOTRON_TEMPERATURE`| `0.1` | Sampling temperature (low for deterministic JSON) |

### Quickstart Test via Python

```python
from src.nemotron_client import NemotronClient, NemotronConfig

config = NemotronConfig(
    base_url="http://127.0.0.1:8000/v1",
    model="nvidia/nemotron-mini-4b-instruct",
    timeout=5.0
)
client = NemotronClient(config)

# Generates report with automatic fallback if endpoint is unreachable
report = client.generate_investigation_report(evidence_dict)
print(f"Report Generated via: {report.engine_mode} (Priority: {report.investigation_priority})")
```
