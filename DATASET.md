# DATASET PROVENANCE & BENCHMARK SPECIFICATION
## Feedzai NeurIPS 2022 Bank Account Fraud (BAF) Dataset

### 1. Dataset Overview
- **Official Benchmark:** NeurIPS 2022 Bank Account Fraud (BAF) Suite (Base Variant).
- **Domain Context:** Retail Bank Account Opening Applications (Application Fraud).
- **Total Application Records:** **1,000,000**
- **Feature Dimensionality:** 31 input features + 1 binary ground truth target (`fraud_bool`).
- **Fraud Class Prevalence:** **1.1029%** (11,029 confirmed fraud applications out of 1,000,000).
- **Class Imbalance Ratio:** ~89.7 legitimate accounts to 1 fraudulent application.
- **SHA-256 Hash of `Base.csv`:** `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`

---

## 2. Authoritative Temporal Split

The dataset features a discrete `month` integer column ranging from 0 to 7. The authoritative leak-free temporal partition is:

| Split | Month Index | Applications ($N$) | Fraud Count | Prevalence | Purpose |
|---|:---:|:---:|:---:|:---:|---|
| **TRAIN** | Months 0–5 | **`794,989`** | 8,151 | 1.0253% | Fit feature transformers & train candidate models |
| **VALIDATION** | Month 6 | **`108,168`** | 1,450 | 1.3405% | Model selection, calibration, threshold freezing |
| **TEST** | Month 7 | **`96,843`** | 1,428 | 1.4746% | Untouched single-pass out-of-sample evaluation |
| **TOTAL** | Months 0–7 | **`1,000,000`** | **`11,029`** | **1.1029%** | Full Benchmark Suite |

---

## 3. Feature Taxonomy & Sentinel Value Definitions

### Missingness Encoding (Sentinel Values)
In accordance with official NeurIPS 2022 benchmark specifications, missing numerical attributes are encoded using negative sentinels (`-1.0`). The `ProductionFeatureEngine` explicitly captures binary missingness indicators prior to clipping:
1. `prev_address_months_count` (< 0 indicates no previous address / thin file)
2. `current_address_months_count` (< 0 indicates unverified current address)
3. `bank_months_count` (< 0 indicates no prior banking relationship)
4. `session_length_in_minutes` (< 0 indicates automated / scripted submission)
5. `device_distinct_emails_8w` (< 0 indicates device cookie clearing / emulator)
6. `intended_balcon_amount` (< 0 indicates missing transfer intent)

### Target and Constant Attributes
- **Target (`fraud_bool`):** Ground truth label (1 = Fraud, 0 = Legitimate). Excluded strictly from model inputs.
- **Constant Column (`device_fraud_count`):** Constant value (0.0) across all 1,000,000 rows. Pruned from feature matrices to prevent zero-variance warnings.

---

## 4. Dataset Acquisition & Verification Instructions

### Step 1: Download
Download `Base.csv` from the official Feedzai repository or Kaggle mirror:
```bash
# Kaggle CLI (if configured)
kaggle datasets download -d feedzai/bank-account-fraud-dataset-neurips-2022 -f Base.csv
```
Place the file into `data/Base.csv` in the repository root.

### Step 2: Validate SHA-256 Hash
Run the built-in validation script:
```powershell
python scripts/validate_data.py
```
Expected output:
```
[PASS] Base.csv verified: 1,000,000 rows, 32 columns.
[PASS] SHA-256: 7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809
```
