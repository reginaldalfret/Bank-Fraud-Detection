"""
src/nemotron_client.py -- Nemotron GenAI Fraud Analyst Client & Zero-Downtime Deterministic Fallback.

Domain: Bank Account Opening Fraud (applications, applicants, account-opening).
Key Fraud Archetypes:
  1. Synthetic Identity: Fabricated profiles, thin address/banking history,
     low email-name coherence, disposable email domains.
  2. Identity Theft: Real victim credentials submitted from mismatched devices,
     invalidated phone contacts, anomalous session behavior.
  3. Mule Account Farming: Scripted bulk account openings, velocity bursts,
     shared birthdates/emails across applications.

Features:
  - Async & Sync Client for local Nemotron LLM endpoint (e.g., http://127.0.0.1:8000/v1)
  - Configurable via environment variables:
      NEMOTRON_BASE_URL (default: http://127.0.0.1:8000/v1)
      NEMOTRON_MODEL    (default: nvidia/nemotron-mini-4b-instruct)
      NEMOTRON_TIMEOUT  (default: 10.0 seconds)
      NEMOTRON_MAX_RETRIES (default: 2)
      NEMOTRON_API_KEY  (default: EMPTY)
  - Health check, retry with exponential backoff, timeout, and malformed response handling.
  - ZERO-DOWNTIME DETERMINISTIC OFFLINE FALLBACK:
      If Nemotron is unreachable, offline, times out, or returns malformed output,
      it returns a high-quality deterministic structured analyst investigation report
      (executive summary, primary risk factors, mitigating factors, recommended verification
      checklist, investigation priority, disposition recommendation) without throwing any 500 errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

try:
    import httpx
except ImportError:
    httpx = None

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None
    Field = None

logger = logging.getLogger("fraud_detection.nemotron_client")


# =====================================================================
# Configuration Model
# =====================================================================

@dataclass
class NemotronConfig:
    """Configuration for local Nemotron LLM inference service."""
    base_url: str = field(
        default_factory=lambda: os.getenv("NEMOTRON_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    )
    model: str = field(
        default_factory=lambda: os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-mini-4b-instruct")
    )
    timeout: float = field(
        default_factory=lambda: float(os.getenv("NEMOTRON_TIMEOUT", "10.0"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("NEMOTRON_MAX_RETRIES", "2"))
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("NEMOTRON_API_KEY", "EMPTY")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("NEMOTRON_TEMPERATURE", "0.1"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("NEMOTRON_MAX_TOKENS", "1024"))
    )


# =====================================================================
# Structured Report Schema
# =====================================================================

@dataclass
class AnalystInvestigationReport:
    """
    Standardized, high-quality fraud analyst investigation report
    generated either via Nemotron LLM or zero-downtime deterministic fallback.
    """
    investigation_id: str
    application_id: str
    timestamp: str
    fraud_probability: float
    decision_threshold: float
    model_prediction: str  # 'SUSPECTED_FRAUD' or 'LEGITIMATE_APPLICATION'
    investigation_priority: str  # 'CRITICAL_IMMEDIATE_ACTION', 'HIGH_PRIORITY_REVIEW', 'STANDARD_REVIEW_QUEUE', 'EXPEDITED_CLEARANCE'
    disposition_recommendation: str  # 'DECLINE_FRAUD_SUSPECTED', 'MANUAL_REVIEW_TIER_2', 'MANUAL_REVIEW_TIER_1', 'AUTO_APPROVE'
    executive_summary: str
    primary_risk_factors: List[str]
    mitigating_factors: List[str]
    recommended_verification_checklist: List[str]
    engine_mode: str  # 'NEMOTRON_LLM' or 'DETERMINISTIC_OFFLINE_FALLBACK'
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =====================================================================
# Deterministic Offline Fallback Generator
# =====================================================================

class DeterministicReportGenerator:
    """
    Guarantees zero-downtime analyst report generation when Nemotron LLM is offline,
    unreachable, or experiencing high latency.
    """

    @classmethod
    def generate_report(
        cls,
        evidence: Union[Dict[str, Any], Any],
        application_id: Optional[str] = None,
    ) -> AnalystInvestigationReport:
        """
        Synthesizes a rule-guided, highly detailed fraud analyst report directly from
        the applicant evidence package.
        """
        if hasattr(evidence, "to_dict"):
            data = evidence.to_dict()
        elif isinstance(evidence, dict):
            data = evidence
        else:
            data = {}

        app_id = str(
            application_id
            or data.get("application_id")
            or f"APP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:18]}"
        )
        prob = float(data.get("fraud_probability", 0.50))
        threshold = float(data.get("decision_threshold", 0.50))
        risk_level = str(data.get("risk_level", "MEDIUM")).upper()
        confidence_tier = str(data.get("confidence_tier", "MODERATE")).upper()

        top_drivers = data.get("top_fraud_drivers", [])
        top_mitigating = data.get("top_mitigating_factors", [])
        deviations = data.get("behavioral_deviations", [])
        risk_flags = data.get("triggered_risk_flags", [])
        summary_metrics = data.get("applicant_summary_metrics", {})

        # Determine Priority & Disposition
        if prob >= 0.70 or "CRITICAL" in risk_level:
            priority = "CRITICAL_IMMEDIATE_ACTION"
            disposition = "DECLINE_FRAUD_SUSPECTED"
            confidence_score = 0.95
        elif prob >= threshold:
            priority = "HIGH_PRIORITY_REVIEW"
            disposition = "MANUAL_REVIEW_TIER_2"
            confidence_score = 0.85
        elif prob >= (threshold - 0.15):
            priority = "STANDARD_REVIEW_QUEUE"
            disposition = "MANUAL_REVIEW_TIER_1"
            confidence_score = 0.75
        else:
            priority = "EXPEDITED_CLEARANCE"
            disposition = "AUTO_APPROVE"
            confidence_score = 0.92

        pred_label = "SUSPECTED_FRAUD" if prob >= threshold else "LEGITIMATE_APPLICATION"

        # 1. Build Primary Risk Factors
        primary_risks: List[str] = []
        for flag in risk_flags:
            clean_flag = flag.replace("FLAG_", "").replace("_", " ").strip()
            primary_risks.append(f"Policy Trigger: {clean_flag}")

        for driver in top_drivers[:4]:
            if isinstance(driver, dict):
                fname = driver.get("display_name", driver.get("feature", "Feature"))
                sval = driver.get("shap_value", 0.0)
                fval = driver.get("feature_value", "N/A")
                desc = driver.get("domain_explanation", "")
                primary_risks.append(f"Model Driver (+{sval:.3f} SHAP): {fname} (value: {fval}) -- {desc}")

        for dev in deviations:
            if isinstance(dev, dict) and dev.get("severity") in ("CRITICAL_ANOMALY", "EXTREME_DEVIATION"):
                mname = dev.get("metric_name", "").replace("_", " ").title()
                desc = dev.get("description", "")
                primary_risks.append(f"Statistical Anomaly ({mname}): {desc}")

        if not primary_risks:
            if prob >= threshold:
                primary_risks.append(f"Aggregated fraud risk score ({prob:.4f}) exceeds decision threshold ({threshold:.4f}).")
            else:
                primary_risks.append("No critical account-opening risk factors identified.")

        # 2. Build Mitigating Factors
        mitigating: List[str] = []
        for factor in top_mitigating[:3]:
            if isinstance(factor, dict):
                fname = factor.get("display_name", factor.get("feature", "Feature"))
                sval = factor.get("shap_value", 0.0)
                fval = factor.get("feature_value", "N/A")
                mitigating.append(f"Trust Indicator ({sval:.3f} SHAP): {fname} (value: {fval})")

        if summary_metrics.get("housing_status") in ("BA", "BB"):
            mitigating.append("Established residential profile reported on application.")
        if "Home=1" in str(summary_metrics.get("phone_contacts_valid", "")) or "Mobile=1" in str(summary_metrics.get("phone_contacts_valid", "")):
            mitigating.append("Verified telecommunications registry contact on file.")

        if not mitigating:
            mitigating.append("No significant mitigating credibility factors noted in submitted profile.")

        # 3. Build Actionable Verification Checklist
        checklist: List[str] = []
        if any("SYNTHETIC" in r for r in primary_risks) or any("THIN" in r for r in primary_risks):
            checklist.append("Perform documentary identity verification: Request government-issued photo ID and primary utility statement.")
            checklist.append("Conduct credit bureau cross-reference check for date of credit file establishment.")

        if any("VELOCITY" in r for r in primary_risks) or any("MULE" in r for r in primary_risks):
            checklist.append("Scan device fingerprint and IP subnet against historical application clusters for syndicate linkage.")
            checklist.append("Review recent account openings sharing identical postal code or employer references within the past 4 weeks.")

        if any("CONTACT" in r for r in primary_risks) or any("PHONE" in r for r in primary_risks):
            checklist.append("Initiate mandatory Out-of-Band (OOB) SMS/Voice OTP verification to verified carrier record.")

        if any("CREDIT" in r for r in primary_risks) or any("LIMIT" in r for r in primary_risks):
            checklist.append("Require income verification: Request 2 consecutive paystubs or verified tax transcript before line assignment.")

        if not checklist:
            if prob >= threshold:
                checklist.append("Trigger standard Level-2 identity and KYC compliance check.")
                checklist.append("Review application contact details and credit bureau profile.")
            else:
                checklist.append("Proceed with standard automated identity verification and KYC onboarding protocol.")

        # 4. Generate Executive Summary
        if prob >= threshold:
            summary = (
                f"Application {app_id} has been evaluated by the bank account opening fraud system with an "
                f"elevated fraud probability of {prob:.4f}, exceeding the operational threshold of {threshold:.4f} "
                f"(Risk Tier: {risk_level}, Model Confidence: {confidence_tier}). "
                f"The primary risk profile is driven by {len(primary_risks)} high-impact indicators, specifically "
                f"concerning {primary_risks[0] if primary_risks else 'elevated risk score'}. "
                f"Recommended disposition is {disposition} under priority {priority}."
            )
        else:
            summary = (
                f"Application {app_id} has been evaluated with a low fraud probability of {prob:.4f}, safely below "
                f"the decision threshold of {threshold:.4f} (Risk Tier: {risk_level}, Model Confidence: {confidence_tier}). "
                f"The applicant profile demonstrates consistent identity indicators with no anomalous velocity bursts "
                f"or synthetic markers. Clearance is recommended under {disposition}."
            )

        return AnalystInvestigationReport(
            investigation_id=f"INV-{uuid.uuid4().hex[:12].upper()}",
            application_id=app_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            fraud_probability=prob,
            decision_threshold=threshold,
            model_prediction=pred_label,
            investigation_priority=priority,
            disposition_recommendation=disposition,
            executive_summary=summary,
            primary_risk_factors=primary_risks,
            mitigating_factors=mitigating,
            recommended_verification_checklist=checklist,
            engine_mode="DETERMINISTIC_OFFLINE_FALLBACK",
            confidence_score=confidence_score,
            metadata={"evidence_items_count": len(primary_risks) + len(mitigating)},
        )


# =====================================================================
# Nemotron LLM Client (Sync & Async)
# =====================================================================

class NemotronClient:
    """
    Robust Client for local NVIDIA Nemotron LLM endpoints.
    Provides synchronous and asynchronous query execution, automatic retry with
    exponential backoff, health checking, response cleaning, and zero-downtime
    deterministic offline fallback.
    """

    SYSTEM_PROMPT = """You are the Lead Bank Account Opening Fraud Analyst.
Your role is to analyze new bank account opening applications and generate concise, structured, high-conviction fraud investigation reports for risk managers.

CRITICAL DOMAIN RULES:
- The domain is BANK ACCOUNT OPENING FRAUD (applications, applicants, new account onboarding).
- NEVER mention transactions, purchases, merchant payments, credit card swipes, or checkout sessions (this is an application-time evaluation dataset).
- Focus on the three account opening fraud archetypes:
  1. Synthetic Identity (fabricated person, thin credit/address tenure, low email-name coherence, disposable domain).
  2. Identity Theft (real credentials submitted by unauthorized actor, contactability mismatch, device spoofing).
  3. Mule Account Farming (scripted bulk openings, application velocity bursts, shared birthdate/device clusters).

You MUST respond strictly with a valid JSON object matching this schema:
{
  "investigation_priority": "CRITICAL_IMMEDIATE_ACTION" | "HIGH_PRIORITY_REVIEW" | "STANDARD_REVIEW_QUEUE" | "EXPEDITED_CLEARANCE",
  "disposition_recommendation": "DECLINE_FRAUD_SUSPECTED" | "MANUAL_REVIEW_TIER_2" | "MANUAL_REVIEW_TIER_1" | "AUTO_APPROVE",
  "executive_summary": "Thorough analytical summary of the application risk assessment...",
  "primary_risk_factors": ["Risk factor 1 with evidence", "Risk factor 2 with evidence"],
  "mitigating_factors": ["Mitigating factor 1", "Mitigating factor 2"],
  "recommended_verification_checklist": ["Specific actionable verification task 1", "Specific actionable verification task 2"],
  "confidence_score": 0.95
}
"""

    def __init__(self, config: Optional[NemotronConfig] = None):
        self.config = config or NemotronConfig()

    # -----------------------------------------------------------------
    # Health Check Methods
    # -----------------------------------------------------------------

    def health_check(self) -> bool:
        """Synchronously check if local Nemotron endpoint is available and healthy."""
        if httpx is None:
            return False
        try:
            url = f"{self.config.base_url}/models"
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            with httpx.Client(timeout=2.0) as client:
                res = client.get(url, headers=headers)
                return res.status_code == 200
        except Exception as exc:
            logger.debug("Nemotron health check failed: %s", exc)
            return False

    async def health_check_async(self) -> bool:
        """Asynchronously check if local Nemotron endpoint is available and healthy."""
        if httpx is None:
            return False
        try:
            url = f"{self.config.base_url}/models"
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(url, headers=headers)
                return res.status_code == 200
        except Exception as exc:
            logger.debug("Async Nemotron health check failed: %s", exc)
            return False

    # -----------------------------------------------------------------
    # LLM Completion Methods
    # -----------------------------------------------------------------

    def _clean_llm_json(self, raw_text: str) -> Dict[str, Any]:
        """Strips markdown codeblocks, thought blocks, and parses JSON output."""
        cleaned = raw_text.strip()
        # Remove reasoning thought tags if present
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL)

        # Strip markdown JSON fences
        if "```json" in cleaned:
            match = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
        elif "```" in cleaned:
            match = re.search(r"```\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)

        cleaned = cleaned.strip()
        return json.loads(cleaned)

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Synchronously sends chat completion request to Nemotron endpoint with retries."""
        if httpx is None:
            raise RuntimeError("httpx package is required for NemotronClient HTTP communication.")

        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        return data["choices"][0]["message"]["content"]
                    elif response.status_code in (502, 503, 504):
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    else:
                        response.raise_for_status()
            except Exception as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                else:
                    break

        raise RuntimeError(f"Nemotron complete failed after {self.config.max_retries} retries: {last_exc}") from last_exc

    async def complete_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Asynchronously sends chat completion request to Nemotron endpoint with retries."""
        if httpx is None:
            raise RuntimeError("httpx package is required for NemotronClient HTTP communication.")

        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        return data["choices"][0]["message"]["content"]
                    elif response.status_code in (502, 503, 504):
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    else:
                        response.raise_for_status()
            except Exception as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                else:
                    break

        raise RuntimeError(f"Nemotron complete_async failed after {self.config.max_retries} retries: {last_exc}") from last_exc

    # -----------------------------------------------------------------
    # High-Level Report Generation with Zero-Downtime Fallback
    # -----------------------------------------------------------------

    def generate_investigation_report(
        self,
        evidence: Union[Dict[str, Any], Any],
        application_id: Optional[str] = None,
        force_fallback: bool = False,
    ) -> AnalystInvestigationReport:
        """
        Generates a comprehensive investigation report.
        Attempts Nemotron LLM first; if unreachable, times out, or returns invalid format,
        seamlessly returns the deterministic offline fallback report.
        """
        if force_fallback:
            return DeterministicReportGenerator.generate_report(evidence, application_id)

        prompt = evidence.format_prompt_context() if hasattr(evidence, "format_prompt_context") else str(evidence)
        app_id = str(
            application_id
            or (evidence.application_id if hasattr(evidence, "application_id") else None)
            or (evidence.get("application_id") if isinstance(evidence, dict) else None)
            or f"APP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:18]}"
        )

        try:
            raw_response = self.complete(prompt)
            parsed = self._clean_llm_json(raw_response)

            prob = float(getattr(evidence, "fraud_probability", 0.50) if not isinstance(evidence, dict) else evidence.get("fraud_probability", 0.50))
            threshold = float(getattr(evidence, "decision_threshold", 0.50) if not isinstance(evidence, dict) else evidence.get("decision_threshold", 0.50))

            return AnalystInvestigationReport(
                investigation_id=f"INV-{uuid.uuid4().hex[:12].upper()}",
                application_id=app_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                fraud_probability=prob,
                decision_threshold=threshold,
                model_prediction="SUSPECTED_FRAUD" if prob >= threshold else "LEGITIMATE_APPLICATION",
                investigation_priority=str(parsed.get("investigation_priority", "HIGH_PRIORITY_REVIEW")),
                disposition_recommendation=str(parsed.get("disposition_recommendation", "MANUAL_REVIEW_TIER_2")),
                executive_summary=str(parsed.get("executive_summary", "")),
                primary_risk_factors=list(parsed.get("primary_risk_factors", [])),
                mitigating_factors=list(parsed.get("mitigating_factors", [])),
                recommended_verification_checklist=list(parsed.get("recommended_verification_checklist", [])),
                engine_mode="NEMOTRON_LLM",
                confidence_score=float(parsed.get("confidence_score", 0.90)),
                metadata={"model_name": self.config.model},
            )
        except Exception as exc:
            logger.warning(
                "Nemotron LLM generation failed or endpoint unavailable (%s). "
                "Engaging zero-downtime deterministic fallback engine.",
                exc,
            )
            return DeterministicReportGenerator.generate_report(evidence, application_id)

    async def generate_investigation_report_async(
        self,
        evidence: Union[Dict[str, Any], Any],
        application_id: Optional[str] = None,
        force_fallback: bool = False,
    ) -> AnalystInvestigationReport:
        """
        Asynchronously generates a comprehensive investigation report with zero-downtime fallback.
        """
        if force_fallback:
            return DeterministicReportGenerator.generate_report(evidence, application_id)

        prompt = evidence.format_prompt_context() if hasattr(evidence, "format_prompt_context") else str(evidence)
        app_id = str(
            application_id
            or (evidence.application_id if hasattr(evidence, "application_id") else None)
            or (evidence.get("application_id") if isinstance(evidence, dict) else None)
            or f"APP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:18]}"
        )

        try:
            raw_response = await self.complete_async(prompt)
            parsed = self._clean_llm_json(raw_response)

            prob = float(getattr(evidence, "fraud_probability", 0.50) if not isinstance(evidence, dict) else evidence.get("fraud_probability", 0.50))
            threshold = float(getattr(evidence, "decision_threshold", 0.50) if not isinstance(evidence, dict) else evidence.get("decision_threshold", 0.50))

            return AnalystInvestigationReport(
                investigation_id=f"INV-{uuid.uuid4().hex[:12].upper()}",
                application_id=app_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                fraud_probability=prob,
                decision_threshold=threshold,
                model_prediction="SUSPECTED_FRAUD" if prob >= threshold else "LEGITIMATE_APPLICATION",
                investigation_priority=str(parsed.get("investigation_priority", "HIGH_PRIORITY_REVIEW")),
                disposition_recommendation=str(parsed.get("disposition_recommendation", "MANUAL_REVIEW_TIER_2")),
                executive_summary=str(parsed.get("executive_summary", "")),
                primary_risk_factors=list(parsed.get("primary_risk_factors", [])),
                mitigating_factors=list(parsed.get("mitigating_factors", [])),
                recommended_verification_checklist=list(parsed.get("recommended_verification_checklist", [])),
                engine_mode="NEMOTRON_LLM",
                confidence_score=float(parsed.get("confidence_score", 0.90)),
                metadata={"model_name": self.config.model},
            )
        except Exception as exc:
            logger.warning(
                "Async Nemotron LLM generation failed or endpoint unavailable (%s). "
                "Engaging zero-downtime deterministic fallback engine.",
                exc,
            )
            return DeterministicReportGenerator.generate_report(evidence, application_id)


NemotronAnalystClient = NemotronClient

