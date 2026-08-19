# CHANGELOG

All notable changes to this project are documented below.

## [2.0.0-scientific] - 2026-08-19

### Scientific ML Pipeline & Validation
- Standardized on the authoritative 1,000,000 application Feedzai NeurIPS 2022 dataset.
- Enforced strict 3-tier temporal splitting: Months 0–5 (Train: 794,989), Month 6 (Validation: 108,168), Month 7 (Test: 96,843).
- Conducted 8-model benchmark and 6-strategy imbalance ablation strictly on Month 6 Validation.
- Deployed champion LightGBM (10:1 RUS + Bayes Prior Correction + Isotonic Calibration) achieving **0.1905 PR-AUC**, **0.8895 ROC-AUC**, and **56.02% TPR @ 5% FPR** on untouched Month 7 Test.
- Completed 1,000,000-application production stress test achieving **206,117 applications/second** throughput and **718.44 MB** peak RAM footprint.

### Explainability & Forensic AI
- Implemented TreeSHAP exact local attributions ($\sum \text{SHAP} + \text{base} \approx \text{margin}$ verified within $< 10^{-3}$).
- Built local NVIDIA Nemotron client with zero-downtime deterministic fallback engine across 8 error modes.

### Backend & UI Console
- Built 14 modular FastAPI REST endpoints with Pydantic v2 data contracts, JWT authentication, and TOTP MFA.
- Upgraded financial operations dashboard with 5 interactive views (Monitor, Queue, Inspector, Lab, Batch Sandbox).
- Added Windows auto-start watchdog service and Cloudflare Ingress integration (`https://frauddetection.reginaldalfret.tech`).
- Reached 100% automated test coverage across 62 pytest cases.
