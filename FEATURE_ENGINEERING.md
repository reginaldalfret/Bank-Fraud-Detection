# Feature Engineering & Forensic Signal Taxonomy
### Mathematical Formulations, Fraud Typology Rationale, and Implementation Reference

**Document Reference:** `FE-2026-TAXONOMY-V2`  
**Pipeline Module:** [`src/api/services/feature_service.py`](file:///e:/Fraud%20Detection/src/api/services/feature_service.py) & [`src/preprocessing.py`](file:///e:/Fraud%20Detection/src/preprocessing.py)  
**Total Canonical Features:** 72 Features (31 Base Raw + 6 Missing Indicators + 16 Engineered Domain Transforms + 19 One-Hot Categoricals)  

---

## 1. Executive Summary & Design Philosophy

In bank account opening fraud detection, standard linear transformations and naive polynomial expansions fail to capture criminal behavior. Fraudsters manipulate application attributes to evade traditional heuristic rules.

The feature engineering pipeline is built around **three primary fraud typologies**:

```
+----------------------------------------------------------------------------------------------------+
|                                    FRAUD TYPOLOGY MAPPING                                          |
+----------------------------------------------------------------------------------------------------+
| Typology               | Core Criminal Mechanism             | Target Engineered Signal Group     |
+------------------------+-------------------------------------+------------------------------------+
| 1. Synthetic Identity  | Fabricated persona with thin history| Email-Name mismatch, DOB clusters, |
|                        | and unlinked credit records         | thin-file composite scores         |
+------------------------+-------------------------------------+------------------------------------+
| 2. Identity Theft      | Stolen real PII submitted from an   | Contactability failures, device OS |
|                        | unauthorized location/hardware      | anomalies, session keepalive flags |
+------------------------+-------------------------------------+------------------------------------+
| 3. Mule Account Farming| Automated bulk account openings for | Velocity burst acceleration ratios,|
|                        | money laundering infrastructure     | ZIP code spatial concentration     |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Feature Group 1: Velocity Bursts & Acceleration Ratios

### Fraud Rationale
Criminal syndicates and automated bots open accounts in bursts during specific operational windows (e.g. night hours or preceding a coordinated mule cash-out campaign). While raw velocity metrics (`velocity_6h`, `velocity_24h`, `velocity_4w`) are informative, their **ratios detect sudden acceleration against regional baseline activity**.

```
+----------------------------------------------------------------------------------------------------+
| Feature Name            | Mathematical Formula                       | Description & Logic         |
+-------------------------+--------------------------------------------+-----------------------------+
| `velocity_burst_6h_4w`  | $\frac{\max(0, v_{6h})}{v_{4w} + \epsilon}$| Short-term (6h) spike       |
|                         |                                            | relative to 4-week baseline |
+-------------------------+--------------------------------------------+-----------------------------+
| `velocity_ratio_6h_24h` | $\frac{\max(0, v_{6h})}{v_{24h} + \epsilon}$| Immediate 6h activity       |
|                         |                                            | relative to daily baseline  |
+-------------------------+--------------------------------------------+-----------------------------+
| `velocity_burst_24h_4w` | $\frac{v_{24h}}{v_{4w} + \epsilon}$        | Daily velocity acceleration |
|                         |                                            | relative to 4-week baseline |
+----------------------------------------------------------------------------------------------------+
```
*Note: $\epsilon = 10^{-6}$ prevents zero-division. $v_{6h}$ is clipped at $0.0$ prior to ratio computation to prevent negative generator artefacts from flipping the sign.*

### Implementation Reference
```python
v6 = max(0.0, float(r.get("velocity_6h", 0.0)))
v24 = float(r.get("velocity_24h", 4000.0))
v4w = float(r.get("velocity_4w", 4000.0))

r["velocity_burst_6h_4w"] = v6 / (v4w + 1e-6)
r["velocity_ratio_6h_24h"] = v6 / (v24 + 1e-6)
r["velocity_burst_24h_4w"] = v24 / (v4w + 1e-6)
```

---

## 3. Feature Group 2: Synthetic Identity Signals

### Fraud Rationale
Synthetic identity fraudsters often generate disposable email addresses that do not match the applicant's declared legal name (e.g. `john_smith_1984@gmail.com` applying under the name `Robert Taylor`). Furthermore, synthetic identity syndicates reuse identical fake birthdates across hundreds of generated email identities.

```
+----------------------------------------------------------------------------------------------------+
| Feature Name            | Mathematical Formula                       | Description & Logic         |
+-------------------------+--------------------------------------------+-----------------------------+
| `email_mismatch_free`   | $(1.0 - \text{sim}_{\text{name, email}})   | Compound risk: Low name-    |
|                         | \times \text{email\_is\_free}$             | email match on a free host  |
+-------------------------+--------------------------------------------+-----------------------------+
| `dob_emails_x_mismatch` | $\text{dob\_emails}_{4w} \times            | High-conviction synthetic:  |
|                         | (1.0 - \text{sim}_{\text{name, email}})$   | Shared DOB across multiple  |
|                         |                                            | emails with mismatch names  |
+----------------------------------------------------------------------------------------------------+
```

### Implementation Reference
```python
name_email_sim = float(r.get("name_email_similarity", 0.5))
email_is_free = float(r.get("email_is_free", 1))
dob_emails = float(r.get("date_of_birth_distinct_emails_4w", 0.0))

r["email_mismatch_free"] = (1.0 - name_email_sim) * email_is_free
r["dob_emails_x_mismatch"] = dob_emails * (1.0 - name_email_sim)
```

---

## 4. Feature Group 3: Thin-File Composite Scores & Missing Indicators

### Fraud Rationale
A newly manufactured synthetic identity has no verifiable past residential history, no previous banking tenure, and an incomplete credit footprint. 

Six fields use `-1` as a sentinel for missing data. Rather than imputing them, we convert `-1 \to \text{NaN}` and aggregate multi-column missingness.

```
+----------------------------------------------------------------------------------------------------+
| Feature Name            | Mathematical Formula                       | Description & Logic         |
+-------------------------+--------------------------------------------+-----------------------------+
| `prev_addr_is_missing`  | $\mathbb{I}(\text{prev\_address} < 0)$     | Missing prior address flag  |
| `curr_addr_is_missing`  | $\mathbb{I}(\text{curr\_address} < 0)$     | Missing current address flag|
| `bank_months_is_missing`| $\mathbb{I}(\text{bank\_months} < 0)$      | Missing bank history flag   |
| `session_len_is_missing`| $\mathbb{I}(\text{session\_len} < 0)$      | Missing session length flag |
| `dev_emails_is_missing` | $\mathbb{I}(\text{dev\_emails} < 0)$       | Missing device emails flag  |
| `intended_bal_is_missing`|$\mathbb{I}(\text{intended\_bal} < 0)$     | Missing transfer amount flag|
+-------------------------+--------------------------------------------+-----------------------------+
| `total_address_history` | $\text{prev\_addr} + \text{curr\_addr}$    | Total residential footprint |
+-------------------------+--------------------------------------------+-----------------------------+
| `thin_file_score`       | $\text{prev\_missing} + \text{bank\_missing}$| Count of core thin markers|
+-------------------------+--------------------------------------------+-----------------------------+
| `n_missing`             | $\sum_{i=1}^6 \text{sentinel\_missing}_i$  | Cross-column missing sum    |
+----------------------------------------------------------------------------------------------------+
```

### Implementation Reference
```python
prev_addr = 0.0 if pd.isna(r.get("prev_address_months_count")) else float(r["prev_address_months_count"])
cur_addr = 0.0 if pd.isna(r.get("current_address_months_count")) else float(r["current_address_months_count"])

r["total_address_history"] = prev_addr + cur_addr
r["thin_file_score"] = r.get("prev_address_months_count_is_missing", 0) + r.get("bank_months_count_is_missing", 0)
r["n_missing"] = miss_count
```

---

## 5. Feature Group 4: Financial Coherence Ratios

### Fraud Rationale
Legitimate banking applicants request credit limits proportionate to their income decile and credit rating. Fraudsters consistently request the maximum possible credit limit (`proposed_credit_limit = 2000`) despite declaring minimal income ranks (`income = 0.1`) or possessing adverse internal ratings (`credit_risk_score < 0`).

```
+----------------------------------------------------------------------------------------------------+
| Feature Name            | Mathematical Formula                       | Description & Logic         |
+-------------------------+--------------------------------------------+-----------------------------+
| `limit_to_income`       | $\frac{\text{proposed\_credit\_limit}}{\text{income} + \epsilon}$ | Credit limit leverage vs income rank |
+-------------------------+--------------------------------------------+-----------------------------+
| `limit_per_risk`        | $\frac{\text{proposed\_credit\_limit}}{\text{credit\_risk\_score} + 200}$ | Credit request normalized by risk score |
+-------------------------+--------------------------------------------+-----------------------------+
| `risk_x_income`         | $\text{credit\_risk\_score} \times \text{income}$ | Credit score weighted by income decile |
+----------------------------------------------------------------------------------------------------+
```

### Implementation Reference
```python
income = float(r.get("income", 0.5))
credit_limit = float(r.get("proposed_credit_limit", 200.0))
credit_risk = float(r.get("credit_risk_score", 100.0))

r["limit_to_income"] = credit_limit / (income + 1e-6)
r["limit_per_risk"] = credit_limit / (credit_risk + 200.0)
r["risk_x_income"] = credit_risk * income
```

---

## 6. Feature Group 5: Contactability & Carrier Verification

### Fraud Rationale
Fraudsters often supply fake phone numbers that fail carrier KYC validation. Having **both** home and mobile phones fail validation is a severe identity theft indicator.

```
+----------------------------------------------------------------------------------------------------+
| Feature Name            | Mathematical Formula                       | Description & Logic         |
+-------------------------+--------------------------------------------+-----------------------------+
| `n_valid_phones`        | $\text{phone\_home\_valid} + \text{phone\_mobile\_valid}$ | Total carrier-verified telephone lines |
+-------------------------+--------------------------------------------+-----------------------------+
| `no_valid_phone`        | $\mathbb{I}(\text{n\_valid\_phones} == 0)$ | 1 if applicant is completely unreachable |
+----------------------------------------------------------------------------------------------------+
```

---

## 7. Feature Group 6: Session Telemetry & Device Anomalies

### Fraud Rationale
Automated bot scripts fill out account applications in seconds, without toggling session persistence options. Fraud syndicates also originate multiple accounts from a single device hardware fingerprint.

```
+----------------------------------------------------------------------------------------------------+
| Feature Name                | Mathematical Formula                   | Description & Logic         |
+-----------------------------+----------------------------------------+-----------------------------+
| `emails_per_session_min`    | $\frac{\text{device\_distinct\_emails}_{8w}}{\text{session\_len} + 1.0}$ | Velocity of accounts per active minute |
+-----------------------------+----------------------------------------+-----------------------------+
| `short_session_no_keepalive`| $\mathbb{I}(\text{session\_len} < 5 \land \text{keep\_alive} == 0)$ | Automated scripted session indicator |
+----------------------------------------------------------------------------------------------------+
```

---

## 8. Feature Group 7 & 8: Spatial Density & Categorical Encodings

### Spatial Clustering
- **`zip_density_vs_velocity`:** $\frac{\text{zip\_count}_{4w}}{\text{velocity}_{4w} + \epsilon}$ measures geographical concentration of application bursts within the applicant's postal zone.

### Categorical One-Hot Encodings (19 Features)
- **`payment_type` (5):** `payment_type_AA`, `payment_type_AB`, `payment_type_AC`, `payment_type_AD`, `payment_type_AE`
- **`employment_status` (7):** `employment_status_CA` through `employment_status_CG`
- **`housing_status` (7):** `housing_status_BA` through `housing_status_BG` (where `housing_status_BC` is among the strongest single predictors of fraud)
- **`source` (2):** `source_INTERNET`, `source_TELEAPP`
- **`device_os` (5):** `device_os_windows`, `device_os_macintosh`, `device_os_linux`, `device_os_x11`, `device_os_other`

---

## 9. Canonical Feature Vector Order

All tree models in the ensemble expect the 72 features in exact canonical index order:

```python
CANONICAL_FEATURE_NAMES = [
    "income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
    "customer_age", "days_since_request", "intended_balcon_amount", "zip_count_4w",
    "velocity_6h", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "credit_risk_score", "email_is_free", "phone_home_valid",
    "phone_mobile_valid", "bank_months_count", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "session_length_in_minutes", "keep_alive_session", "device_distinct_emails_8w",
    "month", "prev_address_months_count_is_missing", "current_address_months_count_is_missing",
    "bank_months_count_is_missing", "session_length_in_minutes_is_missing",
    "device_distinct_emails_8w_is_missing", "intended_balcon_amount_is_missing",
    "velocity_burst_6h_4w", "velocity_ratio_6h_24h", "velocity_burst_24h_4w",
    "email_mismatch_free", "dob_emails_x_mismatch", "total_address_history",
    "thin_file_score", "n_missing", "n_valid_phones", "no_valid_phone",
    "limit_to_income", "limit_per_risk", "risk_x_income", "emails_per_session_min",
    "short_session_no_keepalive", "zip_density_vs_velocity",
    "payment_type_AA", "payment_type_AB", "payment_type_AC", "payment_type_AD", "payment_type_AE",
    "employment_status_CA", "employment_status_CB", "employment_status_CC", "employment_status_CD",
    "employment_status_CE", "employment_status_CF", "employment_status_CG",
    "housing_status_BA", "housing_status_BB", "housing_status_BC", "housing_status_BD",
    "housing_status_BE", "housing_status_BF", "housing_status_BG",
    "source_INTERNET", "source_TELEAPP",
    "device_os_linux", "device_os_macintosh", "device_os_other", "device_os_windows", "device_os_x11"
]
```
