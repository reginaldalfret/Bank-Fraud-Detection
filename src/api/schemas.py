"""Pydantic v2 schemas and data models for Bank Fraud Classification API.

Domain: Bank Account Opening Fraud (Application-level risk evaluation).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class ApplicationRequest(BaseModel):
    """Raw application data for Bank Account Opening Fraud detection."""
    application_id: Optional[str] = Field(
        default=None,
        description="Unique application identifier. Generated automatically if omitted."
    )
    income: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Applicant income decile (0.1 to 0.9 rank)"
    )
    name_email_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity between applicant name and email (0.0 to 1.0)"
    )
    prev_address_months_count: float = Field(
        default=-1.0,
        description="Months at previous address (-1 sentinel indicates missing/thin file)"
    )
    current_address_months_count: float = Field(
        default=50.0,
        description="Months at current address (-1 sentinel indicates missing)"
    )
    customer_age: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Customer age rounded to decade (10, 20, ..., 90). Protected attribute."
    )
    days_since_request: float = Field(
        default=0.01,
        ge=0.0,
        description="Days elapsed since application request was submitted"
    )
    intended_balcon_amount: float = Field(
        default=-1.0,
        description="Initial balance consolidation amount (-1/negative indicates missing)"
    )
    payment_type: str = Field(
        default="AB",
        description="Payment plan code: AA, AB, AC, AD, AE (or AAA, AAB, etc.)"
    )
    zip_count_4w: float = Field(
        default=1200.0,
        ge=0.0,
        description="Total applications from the same postal code in past 4 weeks"
    )
    velocity_6h: float = Field(
        default=5000.0,
        description="Application velocity in last 6 hours (applications/hour)"
    )
    velocity_24h: float = Field(
        default=4500.0,
        ge=0.0,
        description="Application velocity in last 24 hours (applications/hour)"
    )
    velocity_4w: float = Field(
        default=4800.0,
        ge=0.0,
        description="Application velocity baseline in last 4 weeks (applications/hour)"
    )
    bank_branch_count_8w: float = Field(
        default=10.0,
        ge=0.0,
        description="Application count from the associated branch in past 8 weeks"
    )
    date_of_birth_distinct_emails_4w: float = Field(
        default=5.0,
        ge=0.0,
        description="Number of distinct emails sharing same Date of Birth in past 4 weeks"
    )
    employment_status: str = Field(
        default="CA",
        description="Anonymized employment status code: CA, CB, CC, CD, CE, CF, CG"
    )
    credit_risk_score: float = Field(
        default=120.0,
        description="Bank's internal credit risk score (-191 to +389)"
    )
    email_is_free: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 if email is hosted on free provider (Gmail/Yahoo/etc), 0 if corporate/custom"
    )
    housing_status: str = Field(
        default="BC",
        description="Anonymized residential status code: BA, BB, BC, BD, BE, BF, BG"
    )
    phone_home_valid: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if home telephone verified, 0 if unverified/missing"
    )
    phone_mobile_valid: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 if mobile telephone verified, 0 if unverified/missing"
    )
    bank_months_count: float = Field(
        default=6.0,
        description="Months with previous banking relationship (-1 sentinel indicates missing/new customer)"
    )
    has_other_cards: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if applicant holds other credit cards with this institution, 0 otherwise"
    )
    proposed_credit_limit: float = Field(
        default=200.0,
        ge=0.0,
        description="Requested initial credit line"
    )
    foreign_request: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if IP/location origin does not match bank jurisdiction, 0 otherwise"
    )
    source: str = Field(
        default="INTERNET",
        description="Application intake channel: INTERNET, TELEAPP"
    )
    session_length_in_minutes: float = Field(
        default=5.0,
        description="Web application session duration in minutes (-1 if missing)"
    )
    device_os: str = Field(
        default="windows",
        description="Operating system used: windows, macintosh, linux, x11, other"
    )
    keep_alive_session: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 if user selected session keep-alive (genuine human tell), 0 otherwise"
    )
    device_distinct_emails_8w: float = Field(
        default=1.0,
        description="Distinct emails submitted from this hardware device fingerprint in 8 weeks (-1 if missing)"
    )
    device_fraud_count: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Historical fraud incidents associated with device fingerprint"
    )
    month: Optional[int] = Field(
        default=None,
        ge=0,
        le=7,
        description="Temporal cohort month (0-7)"
    )
    threshold_profile: Optional[str] = Field(
        default="balanced",
        description="Operational threshold profile: balanced, high_recall, high_precision, top_1pct, top_5pct"
    )

    @model_validator(mode="after")
    def populate_defaults(self) -> ApplicationRequest:
        if not self.application_id:
            self.application_id = f"APP-{uuid.uuid4().hex[:10].upper()}"
        # Normalize categoricals
        if self.payment_type:
            self.payment_type = self.payment_type.strip().upper()
            if len(self.payment_type) > 2 and self.payment_type.startswith("AA"):
                self.payment_type = self.payment_type[-2:]
        if self.employment_status:
            self.employment_status = self.employment_status.strip().upper()
        if self.housing_status:
            self.housing_status = self.housing_status.strip().upper()
        if self.source:
            self.source = self.source.strip().upper()
        if self.device_os:
            self.device_os = self.device_os.strip().lower()
        return self


class ApplicationBatchRequest(BaseModel):
    """Batch request of multiple account opening applications."""
    applications: List[ApplicationRequest] = Field(..., min_length=1, description="List of applications to score")
    threshold_profile: Optional[str] = Field(
        default="balanced",
        description="Threshold profile to evaluate all applications against"
    )


class RiskFactor(BaseModel):
    """Individual risk driver or mitigating signal for an application."""
    signal_name: str
    feature_name: str
    value: Any
    risk_impact: str = Field(description="positive (increases fraud probability) or negative (mitigating)")
    score_delta: float
    description: str


class PredictionResponse(BaseModel):
    """Single application fraud risk assessment response."""
    application_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of account opening fraud")
    fraud_prediction: int = Field(..., ge=0, le=1, description="Binary classification (1 = Fraudulent, 0 = Legitimate)")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(..., description="Categorical risk tier")
    action: Literal["APPROVE", "REVIEW", "BLOCK"] = Field(..., description="Operational decision recommendation")
    threshold_used: float = Field(..., description="Operational decision boundary applied")
    threshold_profile: str = Field(..., description="Threshold profile applied (balanced, high_recall, etc.)")
    model_name: str = Field(..., description="Active fraud detection model identifier")
    model_version: str = Field(..., description="Model version and iteration")
    top_risk_factors: List[RiskFactor] = Field(default_factory=list, description="Top positive and negative signals")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Aggregate batch prediction summary with individual records."""
    total_applications: int
    approved_count: int
    review_count: int
    blocked_count: int
    average_fraud_probability: float
    processing_time_ms: float
    threshold_profile_used: str
    predictions: List[PredictionResponse]


class ShapDriver(BaseModel):
    """Detailed SHAP feature attribution element."""
    feature: str
    feature_name_human: str
    value: Any
    shap_value: float
    contribution_pct: float
    direction: Literal["increases_risk", "decreases_risk"]
    description: str


class ExplanationResponse(BaseModel):
    """Local SHAP feature attribution & explainability response."""
    application_id: str
    fraud_probability: float
    base_value: float = Field(..., description="Expected baseline log-odds / score before individual features")
    output_value: float = Field(..., description="Model final output score")
    top_positive_drivers: List[ShapDriver] = Field(description="Features increasing risk the most")
    top_negative_drivers: List[ShapDriver] = Field(description="Features reducing risk / mitigating factors")
    all_shap_values: Dict[str, float] = Field(description="Map of feature names to raw SHAP attribution values")
    explanation_summary: str = Field(description="Human-readable synthesis of primary fraud signals")
    risk_summary: str = Field(description="Business context explanation for fraud investigators")
    latency_ms: float


class NemotronAnalysisResponse(BaseModel):
    """AI / LLM fraud forensic investigation briefing."""
    application_id: str
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    executive_summary: str
    typology_analysis: Dict[str, Any] = Field(
        description="Breakdown across Synthetic Identity, Identity Theft, Mule Farming, Incoherent Financials"
    )
    key_findings: List[str] = Field(description="Key evidence items identified in the application")
    mitigating_factors: List[str] = Field(description="Positive applicant attributes reducing suspicion")
    recommended_action: Literal[
        "APPROVE",
        "MANUAL_INVESTIGATION",
        "BLOCK_IMMEDIATELY",
        "REQUEST_ADDITIONAL_DOCUMENTS"
    ]
    analyst_checklist: List[str] = Field(description="Specific investigative steps recommended for fraud triage")
    confidence_score: float = Field(ge=0.0, le=1.0)
    provider: str = Field(description="Model provider: nvidia-nemotron-70b or offline_deterministic_fallback")
    latency_ms: float


class QueueActionRequest(BaseModel):
    """Triage action performed by a fraud analyst on an investigation item."""
    application_id: str
    action: Literal["Review", "Escalate", "Mark Legitimate", "Confirm Fraud", "Add Notes"]
    analyst_id: str = Field(default="analyst_1", description="Identifier of the investigating analyst")
    notes: Optional[str] = Field(default=None, description="Investigator note or rationale")
    tags: Optional[List[str]] = Field(default=None, description="Categorization tags (e.g. mule_network, bot_attack)")


class NoteEntry(BaseModel):
    """Note history item on a queue case."""
    timestamp: str
    analyst_id: str
    action: str
    note: str


class QueueItem(BaseModel):
    """Item representation in the active fraud triage queue."""
    application_id: str
    fraud_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    status: Literal["PENDING", "UNDER_REVIEW", "ESCALATED", "RESOLVED_LEGITIMATE", "RESOLVED_FRAUD"]
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    assigned_to: Optional[str] = None
    notes_history: List[NoteEntry] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    application_data: Optional[Dict[str, Any]] = None


class QueueListResponse(BaseModel):
    """Investigation queue list with triage metrics."""
    total: int
    pending_count: int
    under_review_count: int
    escalated_count: int
    resolved_count: int
    items: List[QueueItem]


class ThresholdProfileInfo(BaseModel):
    """Operational threshold configuration."""
    profile_name: str
    review_threshold: float
    block_threshold: float
    target_objective: str
    expected_recall: float
    expected_fpr: float


class ModelMetricsResponse(BaseModel):
    """Detailed model performance and fairness evaluation metrics."""
    model_name: str
    model_type: str
    dataset_variant: str
    total_evaluation_samples: int
    roc_auc: float
    pr_auc: float
    tpr_at_5pct_fpr: float
    balanced_accuracy: float
    threshold_profiles: Dict[str, ThresholdProfileInfo]
    confusion_matrix: Dict[str, int]
    fairness_metrics: Dict[str, Any]
    feature_importance: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """System health and dependency monitoring status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: str
    version: str
    uptime_seconds: float
    model_loaded: bool
    model_name: str
    services: Dict[str, str]


class MetaResponse(BaseModel):
    """Metadata describing dataset schema, feature distributions, and sentinel handling."""
    dataset_name: str
    domain: str
    total_raw_features: int
    total_engineered_features: int
    sentinel_columns: List[str]
    categorical_columns: List[str]
    protected_attributes: List[str]
    fraud_typologies: List[Dict[str, str]]
    field_ranges: Dict[str, Dict[str, float]]
