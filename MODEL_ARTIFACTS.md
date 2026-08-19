# MODEL ARTIFACTS INVENTORY & SPECIFICATIONS

## Production Model Bundle: `artifacts/best_model.joblib`

- **Bundle Version:** `2.0.0-scientific`
- **Creation Date:** 2026-08-19
- **Size:** `2,088,431 bytes` (~2.0 MB)
- **SHA-256 Checksum:** `B2386B1C24EC869FEB2E79E63EB6959A23F74E260259CF7CA74D4C54FA82CF08`

### Bundle Components
| Object Key | Class / Type | Purpose |
|---|---|---|
| `model` | `lightgbm.sklearn.LGBMClassifier` | 300 Trees, Depth 6, Learning Rate 0.05, Subsample 0.8 |
| `feature_engine` | `src.feature_engine.ProductionFeatureEngine` | 72-feature causal transformer |
| `feature_cols` | `list[str]` | Canonical ordered feature names list |
| `bayes_a`, `bayes_c` | `float` | Bayes prior shift multipliers for 10:1 undersampling correction |
| `calibrator` | `sklearn.isotonic.IsotonicRegression` | Out-of-bounds clipped isotonic probability calibrator |
| `frozen_thresholds` | `dict[str, float]` | Frozen operational decision points (`F1-Optimal`, `5% FPR`, etc.) |
| `primary_threshold` | `float` (`0.0382`) | Operating threshold ensuring $\le 5\%$ False Positive Rate |

---

## All Scientific Audit Artifacts

| Artifact Path | Size (Bytes) | SHA-256 Hash | Purpose |
|---|:---:|:---:|---|
| `artifacts/best_model.joblib` | 2,088,431 | `B2386B1C24EC869FEB2E79E63EB6959A23F74E260259CF7CA74D4C54FA82CF08` | Serialized champion model bundle |
| `artifacts/validation_model_comparison.json` | 1,926 | `807C28301A1701E474D4D28E0FCBA80E1885751F35514B92C9B4760AADD9F821` | Validation-only model leaderboard (Month 6) |
| `artifacts/validation_model_comparison.csv` | 933 | `B4DD865F782C9F46EADB78AB3564EE0F1E69B0A17047AE1CDA98A851AA45DDA0` | CSV format validation comparison |
| `artifacts/threshold_analysis.json` | 3,372 | `47825A5779CEE2B46555B89F43FAF56B0EF86DD190589EBC3F7A0DAF4D33146A` | Frozen threshold performance metrics |
| `artifacts/experimental_protocol_audit.json` | 1,355 | `B276E8BAA736A3D9AE0FE6B4EFDF52C28C96A308908C2530EE51DFA50C8B431E` | Proof of zero test contamination |
| `artifacts/stress_test_results.json` | 546 | `89909F19EA526AA26E9CE9B60139244239DF65BE6B6C546B373FE90ACBE30224` | 1,000,000 row production stress benchmark |
| `artifacts/nemotron_integration_test.json` | 260 | `D3A68EDF7507E7D0FFCAA1494D42AF6EAD5AE00451F718106D21E7AACC74A6B7` | 8-mode Nemotron offline fallback audit |
| `artifacts/http_endpoints_audit.json` | 3,135 | `588E40BB8FA8BDF87298BE78A25A46505C7DEC9758D2A2C037A861D5B7B6978E` | Live HTTP verification across 14 API endpoints |
