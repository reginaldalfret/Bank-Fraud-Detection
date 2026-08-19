# RELEASE AUDIT & SYSTEM INVENTORY
## Supervised Bank Fraud Classification System

**Release Version:** `v2.0.0-scientific`  
**Git Branch:** `main` (Promoted from `Pardhavika_final-model`)  
**Deployment Target:** Windows 11 Production Laptop + Cloudflare Ingress  
**Public Endpoint:** `https://frauddetection.reginaldalfret.tech`  
**GitHub Repository:** `https://github.com/reginaldalfret/Bank-Fraud-Detection`  
**Audit Date:** August 19, 2026  
**Auditor:** Release Engineer & Systems Architect  

---

## 1. System Inventory

### Core Components
| Subsystem | File / Module | Purpose | Status |
|---|---|---|---|
| **Champion Model** | `artifacts/best_model.joblib` | LightGBM + 10:1 RUS + Bayes Prior Shift + Isotonic Calibrator | **Frozen & Verified** |
| **Causal Feature Engine** | `src/feature_engine.py` | 72 leak-free features (velocity bursts, thin-file, synthetic signals) | **Verified (Parity)** |
| **Explainability Engine** | `src/explainability.py` | TreeSHAP exact attributions & deviation radar metrics | **Additivity Verified** |
| **Forensic AI Analyst** | `src/nemotron_client.py` | Local Nemotron LLM client + deterministic offline fallback | **Zero-Downtime Verified** |
| **Enterprise REST API** | `src/api/main.py` | 14 FastAPI endpoints with Pydantic v2 validation and RBAC | **100% Verified** |
| **Frontend Console** | `dashboard/frontend/` | 5 interactive UI views with real-data binding & zero hardcoded metrics | **Verified** |
| **Unified Runner** | `src/serve.py` | Unified FastAPI + Frontend static server on port 8050 | **Operational** |
| **Windows Auto-Start** | `deployment/windows/` | Windows Task Scheduler watchdog + auto-recovery scripts | **Installed & Active** |
| **Cloudflare Tunnel** | `C:\Cloudflared\` | Ingress routing `frauddetection.reginaldalfret.tech` -> `:8050` | **Live & Secured** |

---

## 2. Runtime Dependencies & Environment

### Python Runtime
- **Version:** Python 3.12.10 (64-bit Windows)
- **Key Packages:** `lightgbm==4.6.0`, `polars==1.38.0`, `pandas==2.2.3`, `scikit-learn==1.6.1`, `shap==0.46.0`, `fastapi==0.115.8`, `uvicorn==0.34.0`, `pydantic==2.10.6`, `pytest==9.1.1`.

### Port Allocation & Process Table
| Port | Protocol | Binding | Service Name | Status |
|---|---|---|---|---|
| **8050** | TCP | `0.0.0.0` | Bank Fraud Classification FastAPI + Dashboard | **LISTENING** (Active) |
| **8088** | TCP | `127.0.0.1` | Local LLM / llama-server (Nemotron Backend) | **OPTIONAL** (Fallback Active) |
| **443** | HTTPS | Cloudflare Edge | `frauddetection.reginaldalfret.tech` | **LIVE (200 OK)** |

---

## 3. Dataset & Model Provenance

- **Dataset Name:** Feedzai NeurIPS 2022 Bank Account Fraud (BAF Base).
- **Total Volume:** 1,000,000 application records x 32 columns.
- **Fraud Prevalence:** 1.1029% (11,029 confirmed fraud cases).
- **Source SHA-256:** `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`.
- **Temporal Splitting:** Months 0–5 Train (794,989), Month 6 Validation (108,168), Month 7 Test (96,843).

---

## 4. Frozen Benchmark Results

- **Untouched Test PR-AUC (Month 7):** **`0.1905`** ($12.92\times$ lift over test prevalence $1.4746\%$; $17.27\times$ over global prevalence $1.1029\%$).
- **Untouched Test ROC-AUC (Month 7):** **`0.8895`**.
- **Untouched Test TPR @ 5% FPR:** **`56.02%`** ($800 / 1,428$ frauds caught at $5.0002\%$ FPR).
- **1M Production Pipeline Throughput:** **`206,117 applications/second`** ($4.85\text{s}$ elapsed, $718.44\text{ MB}$ peak RAM).
- **Automated Test Suite:** **62 / 62 tests passing (100% pass rate)**.
