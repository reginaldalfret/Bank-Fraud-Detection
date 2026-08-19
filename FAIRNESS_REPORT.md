# DEMOGRAPHIC FAIRNESS & REGULATORY COMPLIANCE AUDIT
## Protected Attribute Analysis: `customer_age`

In compliance with fair lending guidelines (ECOA / FCRA) and the Feedzai NeurIPS 2022 benchmark specifications, this system was evaluated for algorithmic bias across applicant age cohorts.

---

## 1. Metric Definitions
- **Predictive Equality (FPR Parity):** Evaluates whether legitimate applicants in protected groups experience an equal False Positive Rate:
  $$\text{Ratio} = \frac{\min(\text{FPR}_A, \text{FPR}_B)}{\max(\text{FPR}_A, \text{FPR}_B)} \quad (\text{Target: } \ge 0.800)$$
- **Equal Opportunity (TPR Parity):** Evaluates whether fraud across groups is detected with equal sensitivity ($|\text{TPR}_A - \text{TPR}_B| \le 0.05$).

---

## 2. Quantitative Evaluation on Month 7 Test Set ($N=96,843$)

| Age Demographic Cohort | Sample Size ($N$) | Ground Truth Frauds | Fraud Prevalence | Group FPR at $T=0.0382$ | Group TPR at $T=0.0382$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Young / Middle (`customer_age` $\le 50$)** | 92,678 | 1,291 | 1.393% | **4.98%** | **55.93%** |
| **Senior / Mature (`customer_age` $> 50$)** | 4,165 | 137 | 3.289% | **5.19%** | **56.93%** |

### Fairness Ratios:
- **Predictive Equality Ratio (FPR Parity):** $\frac{4.98\%}{5.19\%} = \mathbf{0.960}$ *(Exceeds the 0.800 four-fifths rule threshold)*.
- **Equal Opportunity Disparity (TPR Parity):** $|56.93\% - 55.93\%| = \mathbf{1.00\%}$ *(Well below the 5.0% maximum allowable disparity ceiling)*.

---

## 3. Compliance Summary
The champion LightGBM model exhibits strong algorithmic parity across customer age cohorts without requiring coercive post-processing adjustments that degrade fraud detection precision.
