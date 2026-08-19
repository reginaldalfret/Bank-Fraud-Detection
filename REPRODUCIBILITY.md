# SCIENTIFIC REPRODUCIBILITY GUIDE

This guide allows an independent researcher to reproduce all experimental findings, model weights, probability calibrations, and test evaluations from scratch.

---

## 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/reginaldalfret/Bank-Fraud-Detection.git
cd Bank-Fraud-Detection

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install exact pinned dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Setup & Validation

Download the official Feedzai NeurIPS 2022 `Base.csv` file into `data/Base.csv`. Validate its integrity:

```powershell
python scripts/validate_data.py
```
Expected SHA-256: `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`.

---

## 3. Run Scientific Selection & Evaluation Pipeline

```powershell
python src/scientific_pipeline.py
```

This single command executes the complete protocol:
1. Splits dataset into Months 0–5 (Train: 794,989), Month 6 (Validation: 108,168), Month 7 (Test: 96,843).
2. Fits `ProductionFeatureEngine` on Train.
3. Benchmarks all 8 model families across 6 imbalance strategies strictly on Month 6 Validation.
4. Fits Isotonic Calibration and freezes operational thresholds ($T=0.0382$ for $5\%$ FPR) on Month 6 Validation.
5. Evaluates the frozen champion model exactly once on Month 7 Test.

---

## 4. Run Automated QA, Security & Consistency Tests

```powershell
pytest tests/ -v
```
All 62 tests should pass with 100% success rate.
