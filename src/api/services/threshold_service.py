"""Threshold Service for Bank Account Opening Fraud Detection.

Manages operational decision boundaries, review vs block cutoffs,
risk tiering, capacity allocation, and cost-optimal operating points.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.api.schemas import ThresholdProfileInfo

logger = logging.getLogger("fraud_api.threshold_service")

# Operational Threshold Profiles designed for 1.1% BAF base fraud rate
THRESHOLD_PROFILES: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "profile_name": "balanced",
        "review_threshold": 0.015,
        "block_threshold": 0.080,
        "target_objective": "Maximized Net Financial Utility balancing review queue capacity and fraud loss prevention",
        "expected_recall": 0.78,
        "expected_fpr": 0.035,
        "description": "Recommended production standard. Approves 95%+, flags ~3.5% for queue triage, blocks severe synthetic bursts."
    },
    "high_recall": {
        "profile_name": "high_recall",
        "review_threshold": 0.008,
        "block_threshold": 0.040,
        "target_objective": "Maximum fraud capture (90%+ recall) for high-risk product lines",
        "expected_recall": 0.91,
        "expected_fpr": 0.075,
        "description": "Catches 90%+ of all fraudulent applications with expanded investigation queue budget."
    },
    "high_precision": {
        "profile_name": "high_precision",
        "review_threshold": 0.060,
        "block_threshold": 0.220,
        "target_objective": "Minimal customer friction and high investigator precision (80%+ precision)",
        "expected_recall": 0.52,
        "expected_fpr": 0.008,
        "description": "Ultra-low false alarm rate for high-value priority customers."
    },
    "top_1pct": {
        "profile_name": "top_1pct",
        "review_threshold": 0.045,
        "block_threshold": 0.180,
        "target_objective": "Cap investigation queue volume strictly at top 1% alert rate",
        "expected_recall": 0.44,
        "expected_fpr": 0.010,
        "description": "Strict queue volume constraint during limited analyst staffing shifts."
    },
    "top_5pct": {
        "profile_name": "top_5pct",
        "review_threshold": 0.012,
        "block_threshold": 0.070,
        "target_objective": "Standard 5% alert rate (TPR @ 5% FPR benchmark point)",
        "expected_recall": 0.82,
        "expected_fpr": 0.050,
        "description": "Aligns directly with NeurIPS BAF benchmark metric (TPR at 5% FPR: 55.4% instant block / 82% review capture)."
    }
}


class ThresholdService:
    """Enterprise Operational Threshold Management Service."""

    def __init__(self):
        self.profiles = THRESHOLD_PROFILES

    def get_profile(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Get the configuration for a named profile or default to balanced."""
        if not profile_name:
            profile_name = "balanced"
        key = profile_name.strip().lower()
        if key not in self.profiles:
            logger.warning("Unknown threshold profile '%s'. Falling back to 'balanced'.", profile_name)
            key = "balanced"
        return self.profiles[key]

    def get_all_profiles(self) -> Dict[str, ThresholdProfileInfo]:
        """Get all available operational profiles as Pydantic models."""
        return {
            name: ThresholdProfileInfo(
                profile_name=cfg["profile_name"],
                review_threshold=cfg["review_threshold"],
                block_threshold=cfg["block_threshold"],
                target_objective=cfg["target_objective"],
                expected_recall=cfg["expected_recall"],
                expected_fpr=cfg["expected_fpr"],
            )
            for name, cfg in self.profiles.items()
        }

    def evaluate_decision(
        self,
        fraud_probability: float,
        profile_name: Optional[str] = "balanced",
        custom_threshold: Optional[float] = None
    ) -> Tuple[Literal["APPROVE", "REVIEW", "BLOCK"], Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"], float, str]:
        """
        Evaluate operational decision and risk level based on fraud probability.
        
        Returns:
            (action, risk_level, threshold_used, profile_used)
        """
        cfg = self.get_profile(profile_name)
        review_thresh = cfg["review_threshold"]
        block_thresh = cfg["block_threshold"]
        profile_used = cfg["profile_name"]

        if custom_threshold is not None and custom_threshold > 0:
            review_thresh = custom_threshold
            block_thresh = min(1.0, custom_threshold * 3.5)
            profile_used = "custom"

        # Operational Action Triaging
        if fraud_probability >= block_thresh:
            action = "BLOCK"
            risk_level = "CRITICAL"
            threshold_used = block_thresh
        elif fraud_probability >= review_thresh:
            action = "REVIEW"
            risk_level = "HIGH" if fraud_probability >= (review_thresh + (block_thresh - review_thresh) * 0.5) else "MEDIUM"
            threshold_used = review_thresh
        else:
            action = "APPROVE"
            risk_level = "LOW"
            threshold_used = review_thresh

        return action, risk_level, round(threshold_used, 4), profile_used


_threshold_service_instance: Optional[ThresholdService] = None


def get_threshold_service() -> ThresholdService:
    global _threshold_service_instance
    if _threshold_service_instance is None:
        _threshold_service_instance = ThresholdService()
    return _threshold_service_instance
