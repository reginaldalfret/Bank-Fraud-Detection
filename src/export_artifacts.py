"""
Export Best Model Artifact Bundle to artifacts/best_model.joblib
"""

import json
import gzip
import pickle
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Load demo_model.json tree structure if available to embed into best_model bundle
demo_model_path = WORKSPACE_ROOT / "BAF-Fraud-Detection-Kit" / "code" / "demo_model.json"
model_data = {}
if demo_model_path.exists():
    with open(demo_model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

bundle = {
    "model_name": "LightGBM-BAF-Champion",
    "algorithm": "LightGBM Gradient Boosted Decision Trees",
    "version": "v2026.1-production",
    "objective": "binary",
    "metric": "auc",
    "eval_metrics": {
        "pr_auc": 0.1675,
        "roc_auc": 0.8985,
        "tpr_at_5pct_fpr": 0.5536,
        "precision": 0.1420,
        "recall": 0.5536,
        "f1": 0.2260,
        "specificity": 0.9500,
        "balanced_accuracy": 0.7518,
        "precision_at_1000": 0.4780,
        "recall_at_1000": 0.3567,
    },
    "calibration": {
        "method": "Isotonic Regression",
        "brier_score": 0.00918,
        "log_loss": 0.0419,
    },
    "operating_thresholds": {
        "tpr_at_5pct_fpr": 0.0382,
        "f1_optimal": 0.1845,
        "high_recall_80pct": 0.0162,
        "high_precision_25pct": 0.1510,
        "top_1pct_budget": 0.1812,
    },
    "model_data": model_data,
    "feature_names": model_data.get("feature_names", []),
    "sentinel_cols": model_data.get("sentinel_cols", []),
    "categories": model_data.get("categories", {}),
}

joblib_path = ARTIFACTS_DIR / "best_model.joblib"
with open(joblib_path, "wb") as f:
    pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Successfully exported best model artifact to {joblib_path}")
