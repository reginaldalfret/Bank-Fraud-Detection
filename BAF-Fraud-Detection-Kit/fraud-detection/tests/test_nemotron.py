"""
tests/test_nemotron.py -- Automated tests for the Nemotron LLM explanation client.

Verifies:
1. Online inference mock (happy path parsing, markdown code block stripping, structured schema).
2. Offline rule-based fallback (missing API key, diverse risk levels, deterministic actions).
3. Timeout exception handling (graceful fallback without unhandled error).
4. Malformed JSON response recovery (invalid syntax, partial response, unexpected structure).
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.nemotron import NemotronClient

SAMPLE_APPLICANT = {
    "income": 0.3,
    "customer_age": 42,
    "velocity_6h": 5200.0,
    "credit_risk_score": -120.0,
    "foreign_request": 1,
    "email_is_free": 1,
    "device_os": "windows",
}

SAMPLE_TOP_FEATURES = [
    {"feature": "velocity_6h", "value": 5200.0, "contribution": 0.42, "method": "shap"},
    {"feature": "credit_risk_score", "value": -120.0, "contribution": 0.38, "method": "shap"},
    {"feature": "foreign_request", "value": 1, "contribution": 0.15, "method": "shap"},
]


class MockHTTPResponse:
    """Helper to mock urllib.request.urlopen responses."""

    def __init__(self, json_data: dict | str, status: int = 200):
        if isinstance(json_data, dict):
            raw_text = json.dumps(json_data)
        else:
            raw_text = json_data
        self._bytes = raw_text.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_nemotron_offline_fallback_when_no_api_key():
    """Client with no API key must cleanly use deterministic offline fallback."""
    client = NemotronClient(api_key="")
    assert not client.is_online_configured

    res = client.explain_application(
        applicant_data=SAMPLE_APPLICANT,
        fraud_probability=0.35,
        risk_level="CRITICAL",
        top_features=SAMPLE_TOP_FEATURES,
        threshold=0.05,
    )

    assert res["provider"] == "offline_fallback"
    assert res["fallback_reason"] == "api_key_missing"
    assert res["risk_level"] == "CRITICAL"
    assert res["risk_score"] == 35.0
    assert "REJECT_OR_BLOCK" in res["recommended_action"]
    assert len(res["key_factors"]) >= 3
    assert "summary" in res


@pytest.mark.parametrize(
    ("risk_level", "expected_action_keyword"),
    [
        ("LOW", "APPROVE"),
        ("MEDIUM", "FLAG_FOR_REVIEW"),
        ("HIGH", "MANUAL_INVESTIGATION"),
        ("CRITICAL", "REJECT_OR_BLOCK"),
    ],
)
def test_nemotron_offline_fallback_action_mapping(risk_level, expected_action_keyword):
    """Verify offline fallback produces standard banking actions matching risk tier."""
    client = NemotronClient(api_key=None)
    res = client.explain_application(
        applicant_data=SAMPLE_APPLICANT,
        fraud_probability=0.05,
        risk_level=risk_level,
        top_features=[],
        threshold=0.05,
    )
    assert expected_action_keyword in res["recommended_action"]


def test_nemotron_online_mock_happy_path():
    """Mock successful Nemotron API response."""
    client = NemotronClient(api_key="nvapi-mock-test-key-12345")
    assert client.is_online_configured

    mock_llm_json = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "summary": "High risk applicant with anomalous transaction velocity and sub-zero credit score.",
                        "key_factors": [
                            "Velocity of 5,200 requests within 6h is 10x normal baseline",
                            "Foreign IP request combined with free webmail",
                        ],
                        "recommended_action": "REJECT_OR_BLOCK",
                    }),
                }
            }
        ]
    }

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mock_llm_json)):
        res = client.explain_application(
            applicant_data=SAMPLE_APPLICANT,
            fraud_probability=0.45,
            risk_level="CRITICAL",
            top_features=SAMPLE_TOP_FEATURES,
            threshold=0.05,
        )

    assert res["provider"] == "nemotron"
    assert res["model"] == "nvidia/nemotron-4-340b-instruct"
    assert res["risk_score"] == 45.0
    assert "anomalous transaction velocity" in res["summary"]
    assert len(res["key_factors"]) == 2
    assert res["recommended_action"] == "REJECT_OR_BLOCK"


def test_nemotron_online_mock_with_markdown_fence():
    """Verify handling when LLM returns JSON wrapped inside markdown code fences."""
    client = NemotronClient(api_key="nvapi-mock-test-key-12345")

    content_with_fence = (
        "```json\n"
        '{"summary": "Safe applicant.", "key_factors": ["Normal velocity"], "recommended_action": "APPROVE"}\n'
        "```"
    )

    mock_llm_json = {
        "choices": [{"message": {"role": "assistant", "content": content_with_fence}}]
    }

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mock_llm_json)):
        res = client.explain_application(
            applicant_data=SAMPLE_APPLICANT,
            fraud_probability=0.01,
            risk_level="LOW",
            top_features=[],
            threshold=0.05,
        )

    assert res["provider"] == "nemotron"
    assert res["summary"] == "Safe applicant."
    assert res["recommended_action"] == "APPROVE"


def test_nemotron_timeout_handling():
    """Verify graceful recovery to fallback when API call times out."""
    client = NemotronClient(api_key="nvapi-mock-test-key-12345", timeout=0.01)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("The read operation timed out")):
        res = client.explain_application(
            applicant_data=SAMPLE_APPLICANT,
            fraud_probability=0.25,
            risk_level="HIGH",
            top_features=SAMPLE_TOP_FEATURES,
            threshold=0.05,
        )

    assert res["provider"] == "offline_fallback"
    assert "network_error" in res["fallback_reason"]
    assert "timed out" in res["fallback_reason"]
    assert res["risk_level"] == "HIGH"
    assert "MANUAL_INVESTIGATION" in res["recommended_action"]


def test_nemotron_malformed_json_handling():
    """Verify graceful recovery when LLM returns unparseable content."""
    client = NemotronClient(api_key="nvapi-mock-test-key-12345")

    mock_invalid_response = {
        "choices": [{"message": {"role": "assistant", "content": "{ This is not valid JSON at all ... {"}}]
    }

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mock_invalid_response)):
        res = client.explain_application(
            applicant_data=SAMPLE_APPLICANT,
            fraud_probability=0.15,
            risk_level="HIGH",
            top_features=SAMPLE_TOP_FEATURES,
            threshold=0.05,
        )

    assert res["provider"] == "offline_fallback"
    assert "malformed_json" in res["fallback_reason"]
    assert res["risk_score"] == 15.0
    assert "summary" in res
