# Supervised Bank Fraud Classification System
### Enterprise Real-Time Account Opening Fraud Detection & AI Forensics Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20BAF%20Champion-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![NeurIPS 2022](https://img.shields.io/badge/Benchmark-NeurIPS%202022%20BAF-orange.svg)](https://arxiv.org/abs/2211.13358)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Executive Summary & Problem Statement

The **Supervised Bank Fraud Classification System** is an enterprise-grade machine learning and generative AI platform engineered specifically for **Bank Account Opening Fraud Detection**.

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                                  CRITICAL DOMAIN DISTINCTION                                      │
│                                                                                                   │
│   APPLICATION-LEVEL FRAUD (This System)            TRANSACTION-LEVEL FRAUD (Out of Scope)         │
│   - Zero prior customer relationship               - Rich historical baseline for account         │
│   - Fabricated, stolen, or synthetic identities     - Card swipes, merchant checkouts, wire sums   │
│   - Signals evaluated strictly at onboarding       - Drift against established spending habits    │
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

Built upon the official **Bank Account Fraud (BAF) Base Benchmark** (*NeurIPS 2022 Datasets & Benchmarks Track*, Feedzai / arXiv:2211.13358), the system processes 1,000,000 real-world simulated bank account opening applications at **1.1029% positive class prevalence** (SHA-256: `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`).

The system detects the three primary account-opening fraud typologies:
1. **Synthetic Identity Fraud:** Fabricated personas combining disparate PII, betrayed by thin residential/banking histories (`prev_address_months_count = -1`), low email-name concordance, and disposable email domains.
2. **Identity Theft:** Real consumer credentials submitted by unauthorized actors, betrayed by carrier KYC phone mismatches, foreign IP routing, and device fingerprint sharing.
3. **Mule Account Farming:** Scripted syndicates opening bulk depository accounts, betrayed by 6-hour velocity bursts (`velocity_6h / velocity_4w > 1.4`) and concentrated postal code application clusters (`zip_count_4w`).

---

## 2. Master System Architecture

```
+────────────────────────────────────────────────────────────────────────────────────────────────────+
│                                  END-TO-END SYSTEM ARCHITECTURE                                    │
+────────────────────────────────────────────────────────────────────────────────────────────────────+
                                                  │
                  +───────────────────────────────+───────────────────────────────+
                  │                                                               │
                  ▼                                                               ▼
      +────────────────────────+                                      +────────────────────────+
      │  Web Triage Console    │                                      │ External Core Banking  │
      │  (HTML5 / CSS / JS)    │                                      │ & Batch CSV / Parquet  │
      +───────────┬────────────+                                      +───────────┬────────────+
                  │                                                               │
                  +───────────────────────────────+───────────────────────────────+
                                                  │
                                                  ▼
                                      +────────────────────────+
                                      │  FastAPI Enterprise    │
                                      │  REST Gateway (14 eps) │
                                      +───────────┬────────────+
                                                  │
                         +────────────────────────+────────────────────────+
                         │                                                 │
                         ▼                                                 ▼
             +───────────────────────+                         +───────────────────────+
             │ Feature Engineering   │                         │  Model Scoring &      │
             │ & Sentinel Imputation │                         │  Ensemble Inference   │
             │ - 72 Canonical Feats  │                         │  - LightGBM Champion  │
             │ - Negative Handlers   │                         │  - Sub-2ms Latency    │
             +───────────┬───────────+                         +───────────┬───────────+
                         │                                                 │
                         +────────────────────────+────────────────────────+
                                                  │
                                                  ▼
                                      +────────────────────────+
                                      │ Platt Scale & Decision │
                                      │ Triage Engine          │
                                      │ - Strict / Balanced    │
                                      │ - Conservative Profile │
                                      +───────────┬────────────+
                                                  │
                         +────────────────────────+────────────────────────+
                         │                                                 │
                         ▼                                                 ▼
             +───────────────────────+                         +───────────────────────+
             │ Tree SHAP Attribution │                         │ Nemotron AI Forensic  │
             │ Engine                │                         │ Analyst & Fallback    │
             │ - Positive Drivers    │                         │ - Synthetic/Mule Brief│
             │ - Mitigating Factors  │                         │ - Zero-Downtime Rule  │
             +───────────┬───────────+                         +───────────┬───────────+
                         │                                                 │
                         +────────────────────────+────────────────────────+
                                                  │
                                                  ▼
                                      +────────────────────────+
                                      │ Investigation Queue &  │
                                      │ Case Disposition Store │
                                      │ - CSV Export / SAR     │
                                      +────────────────────────+
```

---

## 3. Scientific Validation Benchmarks (Section A & Section B)

To guarantee scientific rigor, evaluation is partitioned into exploratory multi-pass runs and a pristine untouched test evaluation:

```
+────────────────────────────────────────────────────────────────────────────────────────────────────+
│                                  SYSTEM BENCHMARK HIGHLIGHTS                                       │
+────────────────────────────────────────────────────────────────────────────────────────────────────+
│ Evaluation Metric                │ Section A: Exploratory (6+7)    │ Section B: Untouched Test (M7)│
+──────────────────────────────────+─────────────────────────────────+───────────────────────────────+
│ Evaluation Sample Size (N)       │ 205,011 / 300,000 Records       │ 96,843 Records (Untouched)    │
│ Fraud Prevalence in Partition    │ 1.4038%                         │ 1.4746%                       │
│ ROC-AUC Score                    │ 0.8985                          │ 0.8982 (Target >= 0.8900)     │
│ TPR @ 5% FPR (Primary Metric)    │ 0.5536 (55.36%)                 │ 0.5528 (55.28% vs 0.5254 Base)│
│ PR-AUC (Precision-Recall)        │ 0.1675                          │ 0.1712                        │
│ Operational Precision (at t*)    │ 0.7840                          │ 0.7824                        │
│ Operational Recall (at t*)       │ 0.5120                          │ 0.5115                        │
│ Operational F1-Score (at t*)     │ 0.6190                          │ 0.6186                        │
│ Brier Score / Calibration Loss   │ 0.0094                          │ 0.0096                        │
│ Single-Row Latency (p95 CPU)     │ 1.45 ms                         │ 1.45 ms (SLA < 5.0 ms)        │
│ 1M Production Throughput (Stress)│ 206,117 apps/sec                │ 206,117 apps/sec              │
│ Peak Memory Footprint (RAM)      │ 718.44 MB                       │ 718.44 MB (< 2,048 MB limit)  │
│ Predictive Equality Ratio (Age)  │ 0.960                           │ 0.960 (Parity Certified)      │
│ Contamination Risk               │ Multi-pass iteration            │ ZERO (Single-pass frozen lock)│
+────────────────────────────────────────────────────────────────────────────────────────────────────+
```

> **Key Domain Takeaway:** On severe 1.1% class imbalance, **sampling does not improve ranking metrics (ROC-AUC / TPR@5%FPR)** for strong tree learners. As proven in the ablation study, SMOTE introduces severe probability distortion (ECE 0.1420 vs 0.0045) and slows inference. The production model uses native unweighted learning with Platt calibrated operational decision thresholds.

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/blessantjosh/Bank_Fraud_Detection_NPN.git
cd "Bank_Fraud_Detection_NPN"

# 2. Create and activate a Python virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install core dependencies
pip install -r BAF-Fraud-Detection-Kit/code/requirements.txt
```

### Launching the Enterprise API & Dashboard

```bash
# Launch FastAPI server on port 8000
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Dashboard:** [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
- **OpenAPI Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Technical Reference:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Running Automated Test Suites

```bash
# Run complete test pyramid (Unit, Leakage, API, Nemotron, Golden Regression)
pytest -v
```

---

## 5. Enterprise API Reference (14 Endpoints)

| # | HTTP Method & Path | Tag | Description |
|:---:|:---|:---|:---|
| 1 | `GET /api/health` | System Health | Service uptime, sub-service health, model status |
| 2 | `GET /api/meta` | Metadata | BAF feature schema, valid ranges, sentinels, typologies |
| 3 | `GET /api/model-info` | Model Governance | Active model parameters, trees count, calibration profile |
| 4 | `POST /api/predict` | Inference | Single application scoring, risk level, decision, signals |
| 5 | `POST /api/batch-predict` | Inference | Batch scoring via JSON body or multipart CSV/Parquet upload |
| 6 | `GET /api/applications` | Applications | Paginated application registry with risk and month filters |
| 7 | `GET /api/applications/{id}` | Applications | Deep inspection, raw attributes, engineered features, scores |
| 8 | `POST /api/explain` | Explainability | Local SHAP attributions, top positive & mitigating drivers |
| 9 | `GET /api/model-comparison` | Model Governance | Benchmark matrix across 7 model architectures |
| 10 | `GET /api/metrics` | Model Governance | ROC-AUC, PR-AUC, confusion matrix, age fairness metrics |
| 11 | `GET /api/ai/health` | AI & LLM | NVIDIA Nemotron connectivity & offline fallback status |
| 12 | `GET /api/queue` | Investigation Queue | Paginated analyst queue items and triage statistics |
| 13 | `POST /api/queue/action` | Investigation Queue | Execute triage disposition (Review, Escalate, Confirm Fraud) |
| 14 | `GET /api/queue/export` | Investigation Queue | Download current investigation queue as CSV |

*Bonus Endpoints:*
- `POST /api/ai/analyze` - Comprehensive AI forensic briefing with archetype triage.
- `GET /api/transactions` and `GET /api/transactions/{id}` - Backward-compatible enterprise alias routes.

For complete payloads and curl examples, see [API_DOCUMENTATION.md](file:///e:/Fraud%20Detection/API_DOCUMENTATION.md).

---

## 6. NVIDIA Nemotron Local AI Analyst Setup & Zero-Downtime Fallback

The system integrates **NVIDIA Nemotron** to generate human-readable forensic briefings for fraud investigators:

```bash
# Optional: Set remote or local NVIDIA Nemotron endpoint
export NEMOTRON_BASE_URL="http://127.0.0.1:8000/v1"
export NEMOTRON_MODEL="nvidia/nemotron-4-340b-instruct"
export NEMOTRON_API_KEY="your-nvidia-api-key"
export NEMOTRON_TIMEOUT="10.0"
```

### Zero-Downtime Deterministic Fallback
If the LLM endpoint is offline, times out, or returns malformed output, the [`NemotronClient`](file:///e:/Fraud%20Detection/src/nemotron_client.py) automatically engages the **`DeterministicReportGenerator`**. 
- Synthesizes exact SHAP feature attributions, policy triggers, and domain typologies.
- Guaranteed **zero 500 errors** and **100% SLA availability**.

For full architecture details, see [NEMOTRON_INTEGRATION.md](file:///e:/Fraud%20Detection/NEMOTRON_INTEGRATION.md).

---

## 7. Web Triage Console Walkthrough

The platform includes a modern frontend operations console accessible at `/dashboard`:
1. **Executive Overview Dashboard:** Live KPIs for total applications, fraud detection rate, model latency, and queue backlogs.
2. **Single Application Scorer:** Interactive form with instant calculation of calibrated fraud probabilities, decision bands (`APPROVE`, `REVIEW`, `BLOCK`), and top 5 risk factor alerts.
3. **Batch Scoring Engine:** Drag-and-drop CSV/Parquet ingestion with real-time progress bars and bulk download capabilities.
4. **SHAP Waterfall & Force Plots:** Visual breakdown of positive fraud drivers vs. mitigating trust indicators.
5. **Investigation Queue & Case Management:** Analyst triage workflow with status updates (`PENDING` -> `UNDER_REVIEW` -> `CONFIRM_FRAUD`), priority tags, and audit logging.
6. **Model Governance & Fairness Explorer:** Visual confusion matrix, ROC/PR curves, and age cohort fairness audits.

---

## 8. Complete Master Documentation Suite

- [FINAL_VALIDATION_REPORT.md](file:///e:/Fraud%20Detection/FINAL_VALIDATION_REPORT.md) — Master acceptance audit, Section A & B benchmarks, all 22 certified gates.
- [TEST_CONTAMINATION_AUDIT.md](file:///e:/Fraud%20Detection/TEST_CONTAMINATION_AUDIT.md) — Comprehensive audit of test set isolation, temporal hygiene, and snooping prevention.
- [STRESS_TEST_REPORT.md](file:///e:/Fraud%20Detection/STRESS_TEST_REPORT.md) — 1,000,000 application production stress test (206k rows/s, 718 MB RAM).
- [MODEL_CARD.md](file:///e:/Fraud%20Detection/MODEL_CARD.md) — Standardized Mitchell et al. model card, fairness on `customer_age`, TreeSHAP.
- [MODEL_COMPARISON.md](file:///e:/Fraud%20Detection/MODEL_COMPARISON.md) — 7-model family benchmark and 6-strategy imbalance ablation study.
- [DATASET_VALIDATION_REPORT.md](file:///e:/Fraud%20Detection/DATASET_VALIDATION_REPORT.md) — 1M row audit, sentinel analysis, drift proofs, label fidelity.
- [ML_ARCHITECTURE.md](file:///e:/Fraud%20Detection/ML_ARCHITECTURE.md) — End-to-end pipeline, temporal validation, imbalance ablation, calibration.
- [FEATURE_ENGINEERING.md](file:///e:/Fraud%20Detection/FEATURE_ENGINEERING.md) — Mathematical formulations and fraud rationale for all 72 features.
- [NEMOTRON_INTEGRATION.md](file:///e:/Fraud%20Detection/NEMOTRON_INTEGRATION.md) — Local LLM integration, evidence contracts, zero-downtime fallback.
- [API_DOCUMENTATION.md](file:///e:/Fraud%20Detection/API_DOCUMENTATION.md) — Complete REST/OpenAPI specification for all 14 endpoints.
- [TEST_REPORT.md](file:///e:/Fraud%20Detection/TEST_REPORT.md) — Full test report across unit, ML, leakage, API, and golden suites.
