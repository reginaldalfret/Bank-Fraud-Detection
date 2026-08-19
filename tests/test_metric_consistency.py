"""
tests/test_metric_consistency.py -- Rigorous mathematical verification of all published metrics.

Verifies:
1. TP + FN == number of positive test samples
2. TN + FP == number of negative test samples
3. TP + TN + FP + FN == total test samples
4. Precision == TP / (TP + FP)
5. Recall == TP / (TP + FN)
6. FPR == FP / (FP + TN)
7. F1 == 2 * Precision * Recall / (Precision + Recall)
8. Specificity == TN / (TN + FP)
9. Balanced Accuracy == (Recall + Specificity) / 2
"""

import json
import os
import pytest


def test_threshold_analysis_mathematical_consistency():
    thr_path = os.path.join("artifacts", "threshold_analysis.json")
    assert os.path.exists(thr_path), "artifacts/threshold_analysis.json must exist"

    with open(thr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for policy_name, policy_data in data.items():
        tm = policy_data["test_metrics"]
        tp = tm["tp"]
        fp = tm["fp"]
        tn = tm["tn"]
        fn = tm["fn"]

        n_pos = tp + fn
        n_neg = tn + fp
        n_total = tp + tn + fp + fn

        assert n_pos == 1428, f"Expected 1,428 test frauds, got {n_pos}"
        assert n_neg == 95415, f"Expected 95,415 test legitimate apps, got {n_neg}"
        assert n_total == 96843, f"Expected 96,843 total test apps, got {n_total}"

        calc_prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        calc_rec = round(tp / n_pos, 4) if n_pos > 0 else 0.0
        calc_fpr = round(fp / n_neg, 4) if n_neg > 0 else 0.0
        calc_spec = round(tn / n_neg, 4) if n_neg > 0 else 0.0
        calc_bal_acc = round((calc_rec + calc_spec) / 2.0, 4)
        calc_f1 = round((2 * calc_prec * calc_rec) / (calc_prec + calc_rec), 4) if (calc_prec + calc_rec) > 0 else 0.0

        assert abs(tm["test_precision"] - calc_prec) <= 0.001, f"{policy_name} precision mismatch"
        assert abs(tm["test_recall"] - calc_rec) <= 0.001, f"{policy_name} recall mismatch"
        assert abs(tm["test_fpr"] - calc_fpr) <= 0.001, f"{policy_name} FPR mismatch"
        assert abs(tm["test_f1"] - calc_f1) <= 0.002, f"{policy_name} F1 mismatch"
        assert abs(tm["test_specificity"] - calc_spec) <= 0.001, f"{policy_name} specificity mismatch"
        assert abs(tm["test_balanced_accuracy"] - calc_bal_acc) <= 0.001, f"{policy_name} balanced accuracy mismatch"
