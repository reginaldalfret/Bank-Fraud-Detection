"""
src/nemotron.py -- Nemotron LLM client for explainable fraud detection.

Provides automated narrative explanations of fraud predictions for risk analysts.
Supports online inference against Nvidia Nemotron API endpoints, with deterministic
offline rule-based fallbacks for network failure, timeouts, or missing API keys.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("fraud_detection.nemotron")


class NemotronClient:
    """Client for generating fraud prediction explanations via Nvidia Nemotron or offline fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-4-340b-instruct",
        timeout: float = 5.0,
    ):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NEMOTRON_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def is_online_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def explain_application(
        self,
        applicant_data: dict[str, Any],
        fraud_probability: float,
        risk_level: str,
        top_features: list[dict[str, Any]] | None = None,
        threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Generate structured explanation for a scored applicant."""
        prob = float(fraud_probability)
        risk_score = round(prob * 100.0, 2)
        top_features = top_features or []

        if not self.is_online_configured:
            return self._offline_fallback(
                applicant_data=applicant_data,
                fraud_probability=prob,
                risk_score=risk_score,
                risk_level=risk_level,
                top_features=top_features,
                threshold=threshold,
                reason="api_key_missing",
            )

        try:
            return self._call_online_api(
                applicant_data=applicant_data,
                fraud_probability=prob,
                risk_score=risk_score,
                risk_level=risk_level,
                top_features=top_features,
                threshold=threshold,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Nemotron API request failed (%s), using offline fallback", exc)
            return self._offline_fallback(
                applicant_data=applicant_data,
                fraud_probability=prob,
                risk_score=risk_score,
                risk_level=risk_level,
                top_features=top_features,
                threshold=threshold,
                reason=f"network_error: {exc}",
            )
        except json.JSONDecodeError as exc:
            logger.warning("Nemotron response was malformed JSON (%s), using offline fallback", exc)
            return self._offline_fallback(
                applicant_data=applicant_data,
                fraud_probability=prob,
                risk_score=risk_score,
                risk_level=risk_level,
                top_features=top_features,
                threshold=threshold,
                reason=f"malformed_json: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error calling Nemotron (%s), using offline fallback", exc, exc_info=True)
            return self._offline_fallback(
                applicant_data=applicant_data,
                fraud_probability=prob,
                risk_score=risk_score,
                risk_level=risk_level,
                top_features=top_features,
                threshold=threshold,
                reason=f"unexpected_error: {exc}",
            )

    def _call_online_api(
        self,
        applicant_data: dict[str, Any],
        fraud_probability: float,
        risk_score: float,
        risk_level: str,
        top_features: list[dict[str, Any]],
        threshold: float,
    ) -> dict[str, Any]:
        """Perform HTTP call to Nemotron API and parse response."""
        prompt = self._build_prompt(
            applicant_data=applicant_data,
            fraud_probability=fraud_probability,
            risk_score=risk_score,
            risk_level=risk_level,
            top_features=top_features,
            threshold=threshold,
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Bank Fraud Risk Analyst. "
                        "Respond ONLY with a valid JSON object matching this schema: "
                        '{"summary": "...", "key_factors": ["..."], "recommended_action": "APPROVE|FLAG_FOR_REVIEW|MANUAL_INVESTIGATION|REJECT_OR_BLOCK"}'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Bank-Fraud-Detection-Nemotron/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp_bytes = resp.read()

        resp_json = json.loads(resp_bytes.decode("utf-8"))
        content = resp_json["choices"][0]["message"]["content"]

        # Parse JSON from content (stripping code fence if present)
        clean_content = content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

        parsed = json.loads(clean_content)

        return {
            "risk_level": risk_level,
            "fraud_probability": fraud_probability,
            "risk_score": risk_score,
            "summary": parsed.get("summary", "Analysis generated by Nemotron."),
            "key_factors": parsed.get("key_factors", []),
            "recommended_action": parsed.get("recommended_action", self._default_action(risk_level)),
            "provider": "nemotron",
            "model": self.model,
        }

    def _build_prompt(
        self,
        applicant_data: dict[str, Any],
        fraud_probability: float,
        risk_score: float,
        risk_level: str,
        top_features: list[dict[str, Any]],
        threshold: float,
    ) -> str:
        features_summary = ", ".join(
            f"{f.get('feature')} (contrib: {f.get('contribution', 0.0):+.3f})"
            for f in top_features[:5]
        ) or "None provided"

        return (
            f"Applicant Data: {json.dumps(applicant_data)}\n"
            f"Predicted Fraud Probability: {fraud_probability:.4f} (Threshold: {threshold:.4f})\n"
            f"Risk Score: {risk_score}/100\n"
            f"Risk Level: {risk_level}\n"
            f"Top Predictive Drivers: {features_summary}\n\n"
            "Provide an executive summary, list of key risk factors, and recommended action in JSON format."
        )

    def _offline_fallback(
        self,
        applicant_data: dict[str, Any],
        fraud_probability: float,
        risk_score: float,
        risk_level: str,
        top_features: list[dict[str, Any]],
        threshold: float,
        reason: str = "offline",
    ) -> dict[str, Any]:
        """Deterministic offline rule-based explanation."""
        factors = []
        for feat in top_features[:5]:
            fname = feat.get("feature", "unknown")
            contrib = feat.get("contribution", 0.0)
            fval = feat.get("value", "")
            direction = "elevating" if contrib > 0 else "reducing"
            factors.append(f"{fname}={fval} ({direction} risk by {abs(contrib):.3f})")

        if not factors:
            if applicant_data.get("velocity_6h", 0) and applicant_data.get("velocity_6h", 0) > 4000:
                factors.append("Elevated 6-hour request velocity")
            if applicant_data.get("credit_risk_score", 0) and applicant_data.get("credit_risk_score", 0) < 0:
                factors.append("Sub-zero credit risk score")
            if applicant_data.get("foreign_request") == 1:
                factors.append("Application originated from foreign IP / network")
            if applicant_data.get("email_is_free") == 1:
                factors.append("Free webmail domain used")

        action = self._default_action(risk_level)
        summary = (
            f"Applicant scored with fraud probability of {fraud_probability:.2%} "
            f"(Risk Score: {risk_score}/100, Category: {risk_level}). "
            f"Decision threshold is {threshold:.2%}. {action}"
        )

        return {
            "risk_level": risk_level,
            "fraud_probability": fraud_probability,
            "risk_score": risk_score,
            "summary": summary,
            "key_factors": factors if factors else ["Risk profile aligns with standard baseline distributions."],
            "recommended_action": action,
            "provider": "offline_fallback",
            "fallback_reason": reason,
        }

    @staticmethod
    def _default_action(risk_level: str) -> str:
        actions = {
            "LOW": "APPROVE: Low fraud indicators, standard automated onboarding.",
            "MEDIUM": "FLAG_FOR_REVIEW: Moderate risk signals detected, secondary KYC recommended.",
            "HIGH": "MANUAL_INVESTIGATION: High anomaly detected, hold application for fraud analyst.",
            "CRITICAL": "REJECT_OR_BLOCK: Severe fraud markers detected, escalate to security.",
        }
        return actions.get(risk_level.upper(), "FLAG_FOR_REVIEW")
