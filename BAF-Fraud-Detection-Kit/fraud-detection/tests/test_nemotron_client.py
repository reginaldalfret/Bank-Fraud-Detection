"""
Tests for src/nemotron_client.py:
- Async and Sync client for Nemotron LLM endpoint
- Health check handling
- Retry and timeout behavior
- Malformed response cleaning
- Zero-Downtime Deterministic Offline Fallback validation
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.explainability import ApplicantEvidencePackage, FeatureContributionItem, MetricDeviation
from src.nemotron_client import (
    AnalystInvestigationReport,
    DeterministicReportGenerator,
    NemotronClient,
    NemotronConfig,
)


@pytest.fixture
def mock_evidence():
    return ApplicantEvidencePackage(
        application_id="APP-UNIT-100",
        timestamp="2026-08-19T12:00:00Z",
        fraud_probability=0.82,
        fraud_prediction=1,
        decision_threshold=0.50,
        risk_level="CRITICAL",
        confidence_tier="VERY_HIGH",
        margin_from_threshold=0.32,
        top_fraud_drivers=[
            FeatureContributionItem(
                feature="limit_to_income",
                display_name="Limit-to-Income Exposure Ratio",
                feature_value=18500.0,
                shap_value=0.28,
                contribution_direction="INCREASES_FRAUD_RISK",
                importance_rank=1,
                magnitude=0.28,
                archetype="CREDIT_RISK",
                domain_explanation="Extreme credit limit relative to declared income decile.",
            ),
            FeatureContributionItem(
                feature="velocity_6h",
                display_name="Application Velocity (6h)",
                feature_value=9400.0,
                shap_value=0.19,
                contribution_direction="INCREASES_FRAUD_RISK",
                importance_rank=2,
                magnitude=0.19,
                archetype="MULE_FARMING",
                domain_explanation="Elevated burst in hourly application submissions.",
            ),
        ],
        top_mitigating_factors=[
            FeatureContributionItem(
                feature="customer_age",
                display_name="Applicant Age Decade",
                feature_value=40,
                shap_value=-0.04,
                contribution_direction="DECREASES_FRAUD_RISK",
                importance_rank=1,
                magnitude=0.04,
                archetype="GENERAL",
                domain_explanation="Standard applicant demographic profile.",
            )
        ],
        behavioral_deviations=[
            MetricDeviation(
                metric_name="limit_to_income",
                applicant_value=18500.0,
                population_mean=2500.0,
                population_median=1500.0,
                population_std=3200.0,
                z_score=5.0,
                percentile_rank=99.9,
                severity="CRITICAL_ANOMALY",
                anomaly_direction="HIGH",
                description="Extreme credit limit extraction relative to income decile.",
            )
        ],
        triggered_risk_flags=[
            "FLAG_SYNTHETIC_IDENTITY_RISK: Low name-email coherence and thin credit file.",
            "FLAG_VELOCITY_BURST: Application surge rate indicates bulk script submission.",
        ],
        applicant_summary_metrics={
            "income_decile": 0.1,
            "proposed_credit_limit": 1850.0,
            "housing_status": "BC",
            "phone_contacts_valid": "Home=0, Mobile=0",
        },
    )


def test_deterministic_offline_fallback(mock_evidence):
    # Test direct deterministic offline report generation
    report = DeterministicReportGenerator.generate_report(mock_evidence)

    assert isinstance(report, AnalystInvestigationReport)
    assert report.application_id == "APP-UNIT-100"
    assert report.fraud_probability == 0.82
    assert report.model_prediction == "SUSPECTED_FRAUD"
    assert report.investigation_priority in ("CRITICAL_IMMEDIATE_ACTION", "HIGH_PRIORITY_REVIEW")
    assert report.disposition_recommendation == "DECLINE_FRAUD_SUSPECTED"
    assert report.engine_mode == "DETERMINISTIC_OFFLINE_FALLBACK"
    assert len(report.primary_risk_factors) >= 2
    assert len(report.recommended_verification_checklist) >= 1
    assert "APP-UNIT-100" in report.executive_summary
    assert "0.8200" in report.executive_summary

    # Ensure JSON conversion works
    rep_dict = report.to_dict()
    assert rep_dict["engine_mode"] == "DETERMINISTIC_OFFLINE_FALLBACK"
    rep_json = report.to_json()
    assert "APP-UNIT-100" in rep_json


def test_nemotron_client_zero_downtime_fallback_on_unreachable_endpoint(mock_evidence):
    # Client pointing to a non-existent local port
    cfg = NemotronConfig(
        base_url="http://127.0.0.1:59999/v1",
        timeout=0.2,
        max_retries=0,
    )
    client = NemotronClient(config=cfg)

    # Health check should return False, without raising an exception
    assert client.health_check() is False

    # Synchronous report generation must seamlessly return deterministic fallback
    report = client.generate_investigation_report(mock_evidence)
    assert isinstance(report, AnalystInvestigationReport)
    assert report.engine_mode == "DETERMINISTIC_OFFLINE_FALLBACK"
    assert report.application_id == "APP-UNIT-100"
    assert report.fraud_probability == 0.82


def test_async_nemotron_client_zero_downtime_fallback(mock_evidence):
    cfg = NemotronConfig(
        base_url="http://127.0.0.1:59999/v1",
        timeout=0.2,
        max_retries=0,
    )
    client = NemotronClient(config=cfg)

    async def run_async_test():
        # Health check
        healthy = await client.health_check_async()
        assert healthy is False

        # Async report generation with fallback
        rep = await client.generate_investigation_report_async(mock_evidence)
        assert isinstance(rep, AnalystInvestigationReport)
        assert rep.engine_mode == "DETERMINISTIC_OFFLINE_FALLBACK"
        assert rep.disposition_recommendation == "DECLINE_FRAUD_SUSPECTED"

    asyncio.run(run_async_test())


def test_nemotron_client_mock_successful_response(mock_evidence):
    cfg = NemotronConfig(base_url="http://127.0.0.1:8000/v1")
    client = NemotronClient(config=cfg)

    mock_llm_json = """```json
    {
      "investigation_priority": "CRITICAL_IMMEDIATE_ACTION",
      "disposition_recommendation": "DECLINE_FRAUD_SUSPECTED",
      "executive_summary": "Application exhibits strong indicators of synthetic identity creation with severe velocity burst.",
      "primary_risk_factors": ["Extreme limit-to-income mismatch", "Unverifiable phone contact"],
      "mitigating_factors": ["None noted"],
      "recommended_verification_checklist": ["Request physical notarized ID proof", "Cross-check IP subnet cluster"],
      "confidence_score": 0.98
    }
    ```"""

    with patch.object(client, "complete", return_value=mock_llm_json):
        report = client.generate_investigation_report(mock_evidence)
        assert report.engine_mode == "NEMOTRON_LLM"
        assert report.confidence_score == 0.98
        assert report.investigation_priority == "CRITICAL_IMMEDIATE_ACTION"
        assert "Extreme limit-to-income mismatch" in report.primary_risk_factors


def test_nemotron_client_malformed_response_handling(mock_evidence):
    cfg = NemotronConfig(base_url="http://127.0.0.1:8000/v1")
    client = NemotronClient(config=cfg)

    # Malformed response from LLM (e.g. truncated or bad syntax)
    with patch.object(client, "complete", return_value="INVALID NON-JSON OUTPUT THAT CANNOT BE PARSED"):
        report = client.generate_investigation_report(mock_evidence)
        # Should gracefully fall back to deterministic report without 500 error
        assert report.engine_mode == "DETERMINISTIC_OFFLINE_FALLBACK"
        assert report.application_id == "APP-UNIT-100"
        assert report.fraud_probability == 0.82
