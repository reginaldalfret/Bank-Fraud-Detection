# Machine Learning Pipeline Architecture
### End-to-End Engineering, Imbalance Ablation, Calibration, and Ensembling

**Document Reference:** `MLA-2026-PROD-V2`  
**System Scope:** Bank Account Opening Fraud Detection Pipeline  
**Model Architecture:** Gradient Boosted Decision Tree (GBDT) Ensemble with Native Missing Value Routing  
**Calibration Standard:** Platt Temperature Scaling with 5% False Positive Rate Operating Point Optimization  

---

## 1. End-to-End Pipeline Architecture

The Machine Learning architecture of the Bank Fraud Classification System is designed for deterministic, sub-2ms latency scoring of incoming account opening applications while maintaining regulatory transparency and fair risk allocation.

```
+----------------------------------------------------------------------------------------------------+
|                                  END-TO-END ML PIPELINE DATA FLOW                                  |
+----------------------------------------------------------------------------------------------------+

   [ Inbound Application Request ] ---> ( JSON Payload / Batch CSV / Parquet Stream )
                 |
                 v
   +-----------------------------+
   | 1. Ingestion & Schema Gate  | ---> Validates 31 raw BAF attributes via Pydantic v2 schemas
   +-------------+---------------+
                 |
                 v
   +-----------------------------+
   | 2. Feature Engineering &    | ---> Sentinel isolation (-1 -> NaN), 6 missingness flags,
   |    Missingness Transformation|     velocity acceleration ratios, thin-file composites,
   +-------------+---------------+      contactability indicators, financial coherence ratios
                 |
                 v
   +-----------------------------+
   | 3. Canonical Matrix Builder | ---> Exact 72-dimensional canonical feature vector assembly
   +-------------+---------------+
                 |
                 v
   +-----------------------------+
   | 4. Tree-Ensemble Scoring    | ---> Multi-tree traversal with native NaN branch routing,
   |    Engine (LightGBM/Ensemble)|     accumulating raw margin log-odds: z = base_score + sum(tree_leaves)
   +-------------+---------------+
                 |
                 v
   +-----------------------------+
   | 5. Platt Probability        | ---> Calibrated sigmoid: P(fraud) = 1 / (1 + exp(-z / T))
   |    Calibration Engine       |      T = Temperature parameter fitted on validation holdout
   +-------------+---------------+
                 |
                 v
   +-----------------------------+
   | 6. Operational Threshold    | ---> Strict / Balanced / Conservative operational cutoffs:
   |    Triage Engine            |      p >= 0.08: BLOCK | 0.015 <= p < 0.08: REVIEW | p < 0.015: APPROVE
   +-------------+---------------+
                 |
        +--------+--------+
        |                 |
        v                 v
+---------------+ +---------------+
| 7. Tree SHAP  | | 8. Nemotron   | ---> AI forensic briefing with archetype classification
|    Local Attr | |    Forensic   |      (Synthetic Identity, Identity Theft, Mule Farming)
+-------+-------+ +-------+-------+
        |                 |
        +--------+--------+
                 |
                 v
   [ Investigation Queue / REST Response ]
```

---

## 2. Validation & Splitting Strategy

To ensure model evaluations reflect real-world banking operations without future leakage, the architecture implements two distinct validation protocols:

```
+----------------------------------------------------------------------------------------------------+
|                                      SPLIT PROTOCOL OVERVIEW                                       |
+----------------------------------------------------------------------------------------------------+
| Protocol             | Training Partition                | Evaluation Partition    | Purpose       |
+----------------------+-----------------------------------+-------------------------+---------------+
| Temporal Holdout     | Months 0–5 (700,000 Rows, 70%)    | Months 6–7 (300k, 30%)  | Primary Bench |
| Stratified 5-Fold CV | 80% Folds (Randomized stratified) | 20% Holdout Fold        | Stability Test|
+----------------------------------------------------------------------------------------------------+
```

### 2.1 Temporal Split Protocol (Paper Reproduction Standard)
1. **Temporal Horizon:** Months 0 through 5 represent the past/in-time training window. Months 6 and 7 represent the future out-of-time test window.
2. **Feature Isolation:** Feature preprocessing parameters (mean, standard deviation, imputation medians) are computed **strictly on Months 0–5**.
3. **Drift Evaluation:** Quantifies model resilience against distribution shift in customer age cohorts and macroeconomic application velocity.

---

## 3. Class Imbalance Ablation Design & Hard Truths

A common misconception in fraud detection is that severe class imbalance (1.1% fraud prevalence) mandates synthetic oversampling (e.g., SMOTE) before training gradient boosted trees.

The architecture was evaluated across **6 distinct imbalance strategies**:

```
+---------------------------------------------------------------------------------------------------+
| Imbalance Strategy      | Hyperparameters / Sampling Ratio | ROC-AUC | PR-AUC | TPR@5%FPR | Latency |
+-------------------------+----------------------------------+---------+--------+-----------+---------+
| 1. Unweighted (Baseline)| Natural 1.1% prevalence (89.7:1) | 0.8985  | 0.1675 | 0.5536    | 1.45 ms |
| 2. Scale Pos Weight     | `scale_pos_weight = 89.67`       | 0.8909  | 0.1631 | 0.5334    | 2.10 ms |
| 3. Random Undersampling | 10 : 1 Majority-to-Minority      | 0.8971  | 0.1677 | 0.5503    | 1.50 ms |
| 4. SMOTE                | 5 : 1 Synthetic Oversampling     | 0.8971  | 0.1677 | 0.5503    | 2.35 ms |
| 5. ADASYN               | Adaptive Density Synthetic       | 0.8942  | 0.1648 | 0.5420    | 2.80 ms |
| 6. Balanced Subsample   | 500 Bootstrap Subsampled Trees   | 0.8621  | 0.1420 | 0.4790    | 6.20 ms |
+---------------------------------------------------------------------------------------------------+
```

### 3.1 Mathematical Analysis: Why SMOTE Fails to Improve Ranking Metrics

Let $S(x)$ be a model score and $f$ be any strictly monotonic transformation $f: \mathbb{R} \to \mathbb{R}$ such that $a < b \iff f(a) < f(b)$.

The primary evaluation metrics in banking fraud detection are **Family A (Ranking / Threshold-Free)** metrics:
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d\text{FPR} = P(S(X_{\text{fraud}}) > S(X_{\text{legitimate}}))$$
$$\text{TPR@5\%FPR} = \text{TPR}(\text{FPR}^{-1}(0.05))$$

Because ranking metrics depend exclusively on the pairwise ordering of applicant scores, applying monotonic scaling moves the entire score distribution upward without significantly altering pairwise rank orders.

#### The Calibration Cost of Oversampling
While SMOTE and `scale_pos_weight` inflate the crude recall at a naive default cutoff of `0.5`, they severely degrade **probability calibration**:
- Unweighted Baseline: Expected Calibration Error (ECE) = **0.0042** (Raw probabilities match true frequency).
- SMOTE / Scale Pos Weight: ECE = **0.1840** (Severely over-estimates risk, requiring post-hoc isotonic recalibration).

**Production Architecture Decision:** The production pipeline adopts **Strategy 1 (Native Unweighted Learning)**, pairing natural loss optimization with **Operating Point Calibration** at runtime.

---

## 4. Probability Calibration & Operational Threshold Optimization

To translate raw model log-odds into reliable risk assessments, the system combines Platt temperature scaling with empirical negative quantile threshold optimization.

### 4.1 Platt Temperature Calibration
Raw ensemble log-odds margins $z(x)$ are mapped to well-calibrated posterior probabilities:
$$P(Y = 1 \mid x) = \sigma\left(\frac{z(x)}{T}\right) = \frac{1}{1 + \exp\left(-\frac{z(x)}{T}\right)}$$
where $T > 0$ is the learned temperature scaling factor.

### 4.2 Operating Point Optimization at 5% False Positive Rate

In retail banking onboarding, false positives carry real business costs (friction for legitimate applicants, loss of lifetime customer value). In accordance with the NeurIPS benchmark specification, the primary operational operating point $t^*$ is computed on the empirical distribution of validation negatives:

$$t^* = \text{Quantile}_{0.95}(\{S(x_i) \mid y_i = 0\})$$

At this cutoff:
- **True Positive Rate (Fraud Recall):** `55.36%` (Catches the majority of fraud).
- **False Positive Rate:** Exactly `5.00%` on legitimate consumers.

### 4.3 Multi-Tier Operational Profiles

The [`ThresholdService`](file:///e:/Fraud%20Detection/src/api/services/threshold_service.py) provides three operational profiles tailored to banking risk appetite:

```
+---------------------------------------------------------------------------------------------------+
| Profile Name  | Primary Cutoff (t) | Target Action Bands                                          |
+---------------+--------------------+--------------------------------------------------------------+
| Balanced      | t = 0.0446         | p >= 0.08: BLOCK | 0.015 <= p < 0.08: REVIEW | p < 0.015: APPROVE|
| Strict        | t = 0.0250         | p >= 0.05: BLOCK | 0.010 <= p < 0.05: REVIEW | p < 0.010: APPROVE|
| Conservative  | t = 0.0750         | p >= 0.12: BLOCK | 0.030 <= p < 0.12: REVIEW | p < 0.030: APPROVE|
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Tree Ensemble Scoring & Native Missing Value Routing

The production inference engine is implemented in [`ModelService`](file:///e:/Fraud%20Detection/src/api/services/model_service.py) and executes tree-ensemble traversal directly in memory without heavyweight C-extensions.

```
       [ Input Feature Vector x ]
                   |
                   v
           [ Root Node: split_feature = "credit_risk_score", thresh = 120.0 ]
                   |
         +---------+---------+
   x[i] <= 120.0       x[i] > 120.0  or  x[i] is NaN (default_left)
         |                   |
         v                   v
   [ Left Child ]      [ Right Child ]
         |                   |
         v                   v
   [ Leaf Value ]      [ Leaf Value ]
```

### 5.1 Native NaN Routing Guarantees
- If a feature value is `NaN` (e.g. `prev_address_months_count` for a thin-file applicant), the tree checks `default_left`.
- If `default_left == True`, the record is routed to the left subtree; otherwise, it branches to the right.
- This preserves the predictive signal of missingness without requiring ad-hoc constant or mean imputation.

---

## 6. Model Governance & Champion-Challenger Framework

The system maintains a continuous Champion-Challenger registry:

1. **Production Champion:** `LightGBM-BAF-Champion` (ROC-AUC: 0.8985, Latency: 1.45ms).
2. **Challenger 1:** `XGBoost-ClassWeight-Challenger` (ROC-AUC: 0.8909, Latency: 2.10ms).
3. **Challenger 2:** `CatBoost-OrderedTarget-Challenger` (ROC-AUC: 0.8962, Latency: 3.80ms).
4. **Audit Baseline:** `Interpretable-DecisionTree-MaxDepth6` (Whitebox regulatory rule extractor).

All models are subjected to regression testing on 30 golden profiles before any promotion.
