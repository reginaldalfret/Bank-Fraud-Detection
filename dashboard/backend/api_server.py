"""
SENTINEL - Supervised Bank Fraud Classification API Server.
FastAPI backend serving the modernized financial fraud-monitoring platform
and BAF (Bank Account Fraud) classification endpoints.
"""

import csv
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(DASHBOARD_DIR, "frontend")
QUEUE_STATE_PATH = os.path.join(BACKEND_DIR, "queue_state.json")

app = FastAPI(
    title="SENTINEL Bank Fraud Classification API",
    version="2.4.0",
    description="Supervised Bank Account Opening Fraud Detection & Monitoring Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Seed Data Fixtures & Queue State Management
# ---------------------------------------------------------------------------
def _load_queue_state() -> Dict[str, Any]:
    if os.path.exists(QUEUE_STATE_PATH):
        try:
            with open(QUEUE_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_queue_state(state: Dict[str, Any]):
    try:
        with open(QUEUE_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save queue state: {e}")

# In-memory applicant database
APPLICANTS_DB = [
    {
        "application_id": "APP-2026-984210",
        "applicant_name": "K. Vance / Machine Gen",
        "timestamp": "2026-08-19T17:28:44Z",
        "customer_age": 30,
        "income": 0.2,
        "employment_status": "CE",
        "housing_status": "BF",
        "name_email_similarity": 0.042,
        "email_is_free": 1,
        "phone_home_valid": 0,
        "phone_mobile_valid": 1,
        "foreign_request": 0,
        "credit_risk_score": 284,
        "proposed_credit_limit": 1800,
        "intended_balcon_amount": -1,
        "payment_type": "AC",
        "has_other_cards": 0,
        "prev_address_months_count": -1,
        "current_address_months_count": 2,
        "bank_months_count": -1,
        "days_since_request": 0.02,
        "velocity_6h": 8420.5,
        "velocity_24h": 7650.0,
        "velocity_4w": 3120.0,
        "zip_count_4w": 4210,
        "bank_branch_count_8w": 180,
        "date_of_birth_distinct_emails_4w": 26,
        "device_os": "Linux",
        "device_distinct_emails_8w": 3,
        "session_length_in_minutes": 0.85,
        "keep_alive_session": 0,
        "device_fraud_count": 0,
        "month": 7,
        "source": "INTERNET",
        "risk_score": 0.9684,
        "risk_tier_code": "priority",
        "status": "pending",
        "notes": "Flagged by burst velocity and zero email-name concordance.",
        "assigned_analyst": "FraudOps Lead (Triage)",
        "shap_waterfall": {
            "base_value": 0.011,
            "final_score": 0.9684,
            "features": [
                {"name": "Velocity Burst (6h vs 4w)", "value": "8,420 apps/hr", "contribution": 0.285, "impact": "increase"},
                {"name": "Name-Email Similarity", "value": "0.042 (Discordant)", "contribution": 0.241, "impact": "increase"},
                {"name": "DOB Distinct Emails Cluster", "value": "26 emails", "contribution": 0.198, "impact": "increase"},
                {"name": "Requested Limit vs Income", "value": "$1,800 @ Decile 0.2", "contribution": 0.142, "impact": "increase"},
                {"name": "No Prev Address History", "value": "Missing (-1)", "contribution": 0.115, "impact": "increase"},
                {"name": "Session Speed / Automation", "value": "0.85 mins", "contribution": 0.082, "impact": "increase"},
                {"name": "Credit Risk Score", "value": "+284 pts", "contribution": 0.054, "impact": "increase"},
                {"name": "Mobile Phone Valid", "value": "Verified (1)", "contribution": -0.049, "impact": "decrease"},
                {"name": "Domestic Request", "value": "Domestic (0)", "contribution": -0.038, "impact": "decrease"}
            ]
        },
        "radar_comparison": {
            "axes": ["Velocity Burst", "Identity Mismatch", "Credit Overreach", "Thin File Signal", "Device Multi-Tenancy", "Session Automation"],
            "applicant": [95, 96, 88, 92, 85, 90],
            "population_normal": [18, 12, 25, 20, 10, 15],
            "population_fraud": [88, 89, 82, 86, 78, 84]
        },
        "nemotron_report": {
            "status": "online",
            "model": "Nemotron-70B-Fraud-Analyst",
            "generated_at": "2026-08-19T17:29:10Z",
            "investigation_priority": "CRITICAL_ESCALATE",
            "sla_window": "< 15 Minutes",
            "executive_summary": "High-conviction synthetic identity application orchestrated as part of a high-velocity automated farming attack. The applicant displays an almost total disconnection between legal name and email address (0.042 score), combined with an address history gap (-1) and a sharp 2.7x burst in 6-hour application velocity.",
            "key_reasons": [
                "Extreme Name-Email Discordance: Random alpha-numeric email syntax bearing 0.04 similarity to applicant name.",
                "Synthetic Identity Farming Cluster: 26 distinct email applications recorded sharing this identical Date of Birth in the past 4 weeks.",
                "Credit Overextension: Low income rank (Decile 0.2) paired with maximum tier requested credit limit ($1,800).",
                "Automated Script Profile: Sub-minute application submission (0.85 min session) with Linux browser agent and disabled keep-alive."
            ],
            "verification_checklist": [
                {"item": "Trigger mandatory Step-Up Video KYC with Liveness detection.", "checked": False, "critical": True},
                {"item": "Request proof of primary residential address (utility bill / lease).", "checked": False, "critical": True},
                {"item": "Cross-reference IP / Device cluster for linked applications in ZIP 4210.", "checked": False, "critical": False},
                {"item": "Place temporary account lock and hold credit card issuance.", "checked": True, "critical": True}
            ]
        }
    },
    {
        "application_id": "APP-2026-984211",
        "applicant_name": "M. Sterling / Multi-Device",
        "timestamp": "2026-08-19T17:24:12Z",
        "customer_age": 50,
        "income": 0.4,
        "employment_status": "CA",
        "housing_status": "BC",
        "name_email_similarity": 0.124,
        "email_is_free": 1,
        "phone_home_valid": 1,
        "phone_mobile_valid": 0,
        "foreign_request": 1,
        "credit_risk_score": 210,
        "proposed_credit_limit": 1500,
        "intended_balcon_amount": 50,
        "payment_type": "AB",
        "has_other_cards": 0,
        "prev_address_months_count": 12,
        "current_address_months_count": 6,
        "bank_months_count": -1,
        "days_since_request": 0.12,
        "velocity_6h": 6200.0,
        "velocity_24h": 5800.0,
        "velocity_4w": 2900.0,
        "zip_count_4w": 3100,
        "bank_branch_count_8w": 95,
        "date_of_birth_distinct_emails_4w": 14,
        "device_os": "Windows",
        "device_distinct_emails_8w": 4,
        "session_length_in_minutes": 1.4,
        "keep_alive_session": 0,
        "device_fraud_count": 0,
        "month": 7,
        "source": "INTERNET",
        "risk_score": 0.9125,
        "risk_tier_code": "priority",
        "status": "escalated",
        "notes": "Escalated to Tier 2 - Cross-border request with multiple device emails.",
        "assigned_analyst": "Sarah Jenkins (Fraud Ops)",
        "shap_waterfall": {
            "base_value": 0.011,
            "final_score": 0.9125,
            "features": [
                {"name": "Foreign Request Origin", "value": "Cross-Border (1)", "contribution": 0.264, "impact": "increase"},
                {"name": "Device Distinct Emails (8w)", "value": "4 emails", "contribution": 0.218, "impact": "increase"},
                {"name": "Velocity Burst (6h)", "value": "6,200 apps/hr", "contribution": 0.185, "impact": "increase"},
                {"name": "Name-Email Similarity", "value": "0.124", "contribution": 0.142, "impact": "increase"},
                {"name": "No Bank History Record", "value": "Missing (-1)", "contribution": 0.098, "impact": "increase"},
                {"name": "Home Phone Valid", "value": "Verified (1)", "contribution": -0.045, "impact": "decrease"},
                {"name": "Address Tenure", "value": "18 mos combined", "contribution": -0.041, "impact": "decrease"}
            ]
        },
        "radar_comparison": {
            "axes": ["Velocity Burst", "Identity Mismatch", "Credit Overreach", "Thin File Signal", "Device Multi-Tenancy", "Session Automation"],
            "applicant": [78, 84, 75, 70, 94, 82],
            "population_normal": [18, 12, 25, 20, 10, 15],
            "population_fraud": [88, 89, 82, 86, 78, 84]
        },
        "nemotron_report": {
            "status": "online",
            "model": "Nemotron-70B-Fraud-Analyst",
            "generated_at": "2026-08-19T17:25:00Z",
            "investigation_priority": "CRITICAL_ESCALATE",
            "sla_window": "< 30 Minutes",
            "executive_summary": "Suspected mule recruitment or account takeover attempt originating from foreign network routing with 4 distinct applicant accounts linked to the same device hardware footprint in 8 weeks.",
            "key_reasons": [
                "Device Hardware Multiplexing: 4 distinct applicants registered through this device profile within 8 weeks.",
                "Cross-Border Origin: Foreign geolocation header on domestic account opening request.",
                "Mobile Number Invalidation: Primary SMS verification failed to route to a genuine carrier."
            ],
            "verification_checklist": [
                {"item": "Perform carrier lookup on mobile number to detect VoIP/virtual SIM.", "checked": True, "critical": True},
                {"item": "Demand dual-factor biometric authentication on customer mobile app.", "checked": False, "critical": True},
                {"item": "Inspect shared device fingerprint cluster across all 4 related applicants.", "checked": False, "critical": True}
            ]
        }
    },
    {
        "application_id": "APP-2026-984213",
        "applicant_name": "E. Caldwell / Clean",
        "timestamp": "2026-08-19T17:02:15Z",
        "customer_age": 40,
        "income": 0.8,
        "employment_status": "CA",
        "housing_status": "BA",
        "name_email_similarity": 0.892,
        "email_is_free": 0,
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "foreign_request": 0,
        "credit_risk_score": 38,
        "proposed_credit_limit": 1200,
        "intended_balcon_amount": 15,
        "payment_type": "AB",
        "has_other_cards": 1,
        "prev_address_months_count": 72,
        "current_address_months_count": 36,
        "bank_months_count": 24,
        "days_since_request": 0.01,
        "velocity_6h": 2100.0,
        "velocity_24h": 2200.0,
        "velocity_4w": 3100.0,
        "zip_count_4w": 1200,
        "bank_branch_count_8w": 15,
        "date_of_birth_distinct_emails_4w": 1,
        "device_os": "Windows",
        "device_distinct_emails_8w": 1,
        "session_length_in_minutes": 8.5,
        "keep_alive_session": 1,
        "device_fraud_count": 0,
        "month": 7,
        "source": "INTERNET",
        "risk_score": 0.0412,
        "risk_tier_code": "normal",
        "status": "marked_legitimate",
        "notes": "Auto-approved by Fast-Track KYC policy.",
        "assigned_analyst": "System Auto-Rule",
        "shap_waterfall": {
            "base_value": 0.011,
            "final_score": 0.0412,
            "features": [
                {"name": "Name-Email Similarity", "value": "0.892 (High)", "contribution": -0.210, "impact": "decrease"},
                {"name": "Extensive Address & Bank Tenure", "value": "9+ yrs history", "contribution": -0.185, "impact": "decrease"},
                {"name": "Low Credit Risk Score", "value": "+38 pts", "contribution": -0.160, "impact": "decrease"},
                {"name": "Paid Corporate Email Domain", "value": "Paid (0)", "contribution": -0.120, "impact": "decrease"},
                {"name": "Existing Bank Card Holder", "value": "Yes (1)", "contribution": -0.095, "impact": "decrease"},
                {"name": "Normal Session Time", "value": "8.5 mins", "contribution": -0.080, "impact": "decrease"},
                {"name": "Baseline Velocity Level", "value": "2,100 apps/hr", "contribution": 0.025, "impact": "increase"}
            ]
        },
        "radar_comparison": {
            "axes": ["Velocity Burst", "Identity Mismatch", "Credit Overreach", "Thin File Signal", "Device Multi-Tenancy", "Session Automation"],
            "applicant": [8, 6, 14, 5, 8, 10],
            "population_normal": [18, 12, 25, 20, 10, 15],
            "population_fraud": [88, 89, 82, 86, 78, 84]
        },
        "nemotron_report": {
            "status": "online",
            "model": "Nemotron-70B-Fraud-Analyst",
            "generated_at": "2026-08-19T17:03:00Z",
            "investigation_priority": "FAST_TRACK_APPROVE",
            "sla_window": "Instant Automated",
            "executive_summary": "High-trust genuine applicant profile. Robust credit history, confirmed residential stability (homeowner, 9+ combined years tenure), corporate email domain alignment, and verified multi-channel telephony.",
            "key_reasons": [
                "Flawless Identity Consistency: Corporate domain with 0.892 name-email concordance.",
                "Deep Banking History: 24-month previous account with established card relationship.",
                "Natural User Behavior: Realistic 8.5-minute application duration with persistent session."
            ],
            "verification_checklist": [
                {"item": "Standard automated identity database match.", "checked": True, "critical": False},
                {"item": "Instant credit bureau check.", "checked": True, "critical": False}
            ]
        }
    }
]

# Sync stored queue state
_q_state = _load_queue_state()
for app_item in APPLICANTS_DB:
    aid = app_item["application_id"]
    if aid in _q_state:
        app_item["status"] = _q_state[aid].get("status", app_item["status"])
        app_item["notes"] = _q_state[aid].get("notes", app_item["notes"])
        app_item["assigned_analyst"] = _q_state[aid].get("analyst", app_item["assigned_analyst"])


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/kpis")
def get_kpis():
    return {
        "total_applications": 1000000,
        "predicted_fraud_count": 11029,
        "fraud_rate": 0.011029,
        "pr_auc": 0.167736,
        "roc_auc": 0.897919,
        "tpr_at_5pct_fpr": 0.550317,
        "precision_at_opt": 0.248,
        "recall_at_opt": 0.764,
        "f1_score": 0.3745,
        "avg_application_limit": 842.50,
        "active_model_name": "CatBoost + XGBoost Ensemble v2.4",
        "score_distribution": [
            {"bin": "0.00 - 0.10", "count": 742150, "pct": 74.21, "risk": "low"},
            {"bin": "0.10 - 0.20", "count": 148900, "pct": 14.89, "risk": "low"},
            {"bin": "0.20 - 0.30", "count": 48200, "pct": 4.82, "risk": "moderate"},
            {"bin": "0.30 - 0.40", "count": 22400, "pct": 2.24, "risk": "moderate"},
            {"bin": "0.40 - 0.50", "count": 14100, "pct": 1.41, "risk": "moderate"},
            {"bin": "0.50 - 0.60", "count": 9600, "pct": 0.96, "risk": "high"},
            {"bin": "0.60 - 0.70", "count": 6200, "pct": 0.62, "risk": "high"},
            {"bin": "0.70 - 0.80", "count": 4150, "pct": 0.41, "risk": "high"},
            {"bin": "0.80 - 0.90", "count": 2800, "pct": 0.28, "risk": "critical"},
            {"bin": "0.90 - 1.00", "count": 1500, "pct": 0.15, "risk": "critical"}
        ],
        "monthly_trend": [
            {"month": "Month 0", "month_idx": 0, "applications": 125400, "fraud_count": 1320, "fraud_rate": 0.0105, "detection_rate": 0.562, "split": "Train"},
            {"month": "Month 1", "month_idx": 1, "applications": 128900, "fraud_count": 1210, "fraud_rate": 0.0094, "detection_rate": 0.558, "split": "Train"},
            {"month": "Month 2", "month_idx": 2, "applications": 121300, "fraud_count": 1080, "fraud_rate": 0.0089, "detection_rate": 0.549, "split": "Train"},
            {"month": "Month 3", "month_idx": 3, "applications": 134200, "fraud_count": 1450, "fraud_rate": 0.0108, "detection_rate": 0.554, "split": "Train"},
            {"month": "Month 4", "month_idx": 4, "applications": 126100, "fraud_count": 1410, "fraud_rate": 0.0112, "detection_rate": 0.551, "split": "Train"},
            {"month": "Month 5", "month_idx": 5, "applications": 132500, "fraud_count": 1590, "fraud_rate": 0.0120, "detection_rate": 0.548, "split": "Validation"},
            {"month": "Month 6", "month_idx": 6, "applications": 118600, "fraud_count": 1610, "fraud_rate": 0.0136, "detection_rate": 0.539, "split": "Out-of-Time Test"},
            {"month": "Month 7", "month_idx": 7, "applications": 113000, "fraud_count": 1359, "fraud_rate": 0.0120, "detection_rate": 0.542, "split": "Out-of-Time Test"}
        ],
        "top_risk_indicators": [
            {"feature": "velocity_6h_to_4w_ratio", "label": "Velocity Burst Ratio (6h / 4w)", "importance": 0.184, "direction": "positive", "category": "Velocity"},
            {"feature": "name_email_similarity", "label": "Name & Email Similarity", "importance": 0.162, "direction": "negative", "category": "Identity"},
            {"feature": "credit_risk_score", "label": "Credit Risk Score", "importance": 0.145, "direction": "positive", "category": "Credit"},
            {"feature": "proposed_credit_limit_to_income", "label": "Requested Limit / Income Decile", "importance": 0.128, "direction": "positive", "category": "Financial"},
            {"feature": "date_of_birth_distinct_emails_4w", "label": "DOB Distinct Emails (4-Week Cluster)", "importance": 0.109, "direction": "positive", "category": "Identity"},
            {"feature": "housing_status", "label": "Housing Status (Rental / Social)", "importance": 0.091, "direction": "positive", "category": "Demographic"},
            {"feature": "prev_address_months_count", "label": "Missing Prev Address History (-1)", "importance": 0.078, "direction": "positive", "category": "Tenure"},
            {"feature": "device_distinct_emails_8w", "label": "Device Shared Email Count (8w)", "importance": 0.063, "direction": "positive", "category": "Device"},
            {"feature": "session_length_in_minutes", "label": "Abnormally Rapid Session (<2 min)", "importance": 0.040, "direction": "positive", "category": "Behavior"}
        ]
    }


@app.get("/api/applications")
def get_applications(
    q: Optional[str] = None,
    risk_tier: Optional[str] = None,
    customer_age: Optional[int] = None,
    employment_status: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
):
    items = list(APPLICANTS_DB)
    if q:
        ql = q.lower()
        items = [
            a for a in items
            if ql in a["application_id"].lower()
            or ql in a["applicant_name"].lower()
            or ql in a["employment_status"].lower()
        ]
    if risk_tier:
        items = [a for a in items if a["risk_tier_code"] == risk_tier]
    if customer_age is not None:
        items = [a for a in items if a["customer_age"] == customer_age]
    if employment_status:
        items = [a for a in items if a["employment_status"] == employment_status]
    if status:
        items = [a for a in items if a["status"] == status]

    items.sort(key=lambda x: x["risk_score"], reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    paged = items[start : start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": paged,
    }


@app.get("/api/applications/{application_id}")
def get_application_by_id(application_id: str):
    for a in APPLICANTS_DB:
        if a["application_id"] == application_id:
            return a
    raise HTTPException(status_code=404, detail="Application not found")


class QueueActionBody(BaseModel):
    application_id: str
    action: str
    notes: Optional[str] = ""
    analyst: Optional[str] = "Analyst"


@app.post("/api/queue/action")
def take_queue_action(body: QueueActionBody):
    found = None
    for a in APPLICANTS_DB:
        if a["application_id"] == body.application_id:
            a["status"] = body.action
            if body.notes:
                a["notes"] = body.notes
            a["assigned_analyst"] = body.analyst
            found = a
            break

    if not found:
        raise HTTPException(status_code=404, detail="Application not found")

    state = _load_queue_state()
    state[body.application_id] = {
        "status": body.action,
        "notes": body.notes,
        "analyst": body.analyst,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_queue_state(state)

    return {"success": True, "application_id": body.application_id, "status": body.action}


@app.get("/api/queue/export")
def export_queue():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ApplicationID", "ApplicantName", "AgeDecade", "EmploymentStatus",
        "HousingStatus", "IncomeDecile", "ProposedLimit", "RiskScore",
        "RiskTier", "Status", "Notes"
    ])
    for a in APPLICANTS_DB:
        writer.writerow([
            a["application_id"], a["applicant_name"], a["customer_age"],
            a["employment_status"], a["housing_status"], a["income"],
            a["proposed_credit_limit"], a["risk_score"], a["risk_tier_code"],
            a["status"], a["notes"]
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sentinel_queue_export.csv"}
    )


@app.get("/api/model-lab")
def get_model_lab():
    return {
        "leaderboard": [
            {
                "name": "CatBoost + XGBoost Blend v2.4",
                "type": "Ensemble Blend",
                "is_champion": True,
                "pr_auc": 0.1677,
                "roc_auc": 0.8979,
                "tpr_at_5pct_fpr": 0.5503,
                "precision": 0.248,
                "recall": 0.764,
                "f1": 0.3745,
                "latency_ms": 0.042,
                "model_size_mb": 24.5,
                "description": "Rank-average weighted blend with out-of-time temporal CV"
            },
            {
                "name": "CatBoost Native Classifier",
                "type": "Gradient Boosted Trees",
                "is_champion": False,
                "pr_auc": 0.1666,
                "roc_auc": 0.8968,
                "tpr_at_5pct_fpr": 0.5476,
                "precision": 0.241,
                "recall": 0.758,
                "f1": 0.3658,
                "latency_ms": 0.028,
                "model_size_mb": 18.2,
                "description": "Symmetric tree architecture with built-in categorical splits"
            },
            {
                "name": "XGBoost Tuned (Undersampled 10:1)",
                "type": "Gradient Boosted Trees",
                "is_champion": False,
                "pr_auc": 0.1654,
                "roc_auc": 0.8955,
                "tpr_at_5pct_fpr": 0.5442,
                "precision": 0.236,
                "recall": 0.752,
                "f1": 0.3592,
                "latency_ms": 0.021,
                "model_size_mb": 12.6,
                "description": "Exact greedy tree boosting trained on controlled class-balance"
            },
            {
                "name": "LightGBM High-Velocity",
                "type": "Gradient Boosted Trees",
                "is_champion": False,
                "pr_auc": 0.1631,
                "roc_auc": 0.8912,
                "tpr_at_5pct_fpr": 0.5334,
                "precision": 0.228,
                "recall": 0.741,
                "f1": 0.3486,
                "latency_ms": 0.015,
                "model_size_mb": 9.8,
                "description": "Leaf-wise tree growth optimized for sub-millisecond scoring"
            },
            {
                "name": "Supervised MLP Deep Neural Net",
                "type": "Deep Learning",
                "is_champion": False,
                "pr_auc": 0.1482,
                "roc_auc": 0.8710,
                "tpr_at_5pct_fpr": 0.4890,
                "precision": 0.198,
                "recall": 0.695,
                "f1": 0.3082,
                "latency_ms": 0.085,
                "model_size_mb": 45.0,
                "description": "4-Layer Dense residual network with LayerNorm and Dropout"
            },
            {
                "name": "Isolation Forest (Unsupervised Baseline)",
                "type": "Anomaly Detection",
                "is_champion": False,
                "pr_auc": 0.0425,
                "roc_auc": 0.6840,
                "tpr_at_5pct_fpr": 0.1820,
                "precision": 0.052,
                "recall": 0.320,
                "f1": 0.0895,
                "latency_ms": 0.019,
                "model_size_mb": 8.4,
                "description": "Unsupervised tree isolation baseline (no fraud labels)"
            }
        ],
        "roc_curve": [
            {"fpr": 0.000, "tpr": 0.000, "threshold": 1.00},
            {"fpr": 0.001, "tpr": 0.185, "threshold": 0.85},
            {"fpr": 0.005, "tpr": 0.342, "threshold": 0.72},
            {"fpr": 0.010, "tpr": 0.428, "threshold": 0.60},
            {"fpr": 0.020, "tpr": 0.486, "threshold": 0.45},
            {"fpr": 0.030, "tpr": 0.514, "threshold": 0.35},
            {"fpr": 0.050, "tpr": 0.5503, "threshold": 0.18},
            {"fpr": 0.080, "tpr": 0.612, "threshold": 0.12},
            {"fpr": 0.120, "tpr": 0.685, "threshold": 0.08},
            {"fpr": 0.200, "tpr": 0.782, "threshold": 0.04},
            {"fpr": 0.350, "tpr": 0.884, "threshold": 0.02},
            {"fpr": 0.600, "tpr": 0.952, "threshold": 0.01},
            {"fpr": 1.000, "tpr": 1.000, "threshold": 0.00}
        ],
        "pr_curve": [
            {"recall": 0.00, "precision": 0.820, "threshold": 0.95},
            {"recall": 0.10, "precision": 0.680, "threshold": 0.82},
            {"recall": 0.20, "precision": 0.540, "threshold": 0.70},
            {"recall": 0.35, "precision": 0.410, "threshold": 0.55},
            {"recall": 0.50, "precision": 0.295, "threshold": 0.38},
            {"recall": 0.5503, "precision": 0.258, "threshold": 0.18},
            {"recall": 0.65, "precision": 0.195, "threshold": 0.11},
            {"recall": 0.764, "precision": 0.145, "threshold": 0.06},
            {"recall": 0.88, "precision": 0.082, "threshold": 0.02},
            {"recall": 1.00, "precision": 0.011, "threshold": 0.00}
        ],
        "calibration": {
            "brier_score": 0.00892,
            "ece": 0.0064,
            "bins": [
                {"mean_pred": 0.05, "obs_fraud": 0.048, "count": 852000},
                {"mean_pred": 0.15, "obs_fraud": 0.142, "count": 68400},
                {"mean_pred": 0.25, "obs_fraud": 0.256, "count": 32100},
                {"mean_pred": 0.35, "obs_fraud": 0.362, "count": 18400},
                {"mean_pred": 0.45, "obs_fraud": 0.448, "count": 11200},
                {"mean_pred": 0.55, "obs_fraud": 0.561, "count": 7800},
                {"mean_pred": 0.65, "obs_fraud": 0.639, "count": 4900},
                {"mean_pred": 0.75, "obs_fraud": 0.768, "count": 3100},
                {"mean_pred": 0.85, "obs_fraud": 0.842, "count": 1400},
                {"mean_pred": 0.95, "obs_fraud": 0.938, "count": 700}
            ]
        }
    }


class SimulationPayload(BaseModel):
    name_email_similarity: Optional[float] = 0.5
    velocity_6h: Optional[float] = 2000.0
    velocity_4w: Optional[float] = 3000.0
    proposed_credit_limit: Optional[float] = 1000.0
    income: Optional[float] = 0.5
    device_os: Optional[str] = "Windows"
    date_of_birth_distinct_emails_4w: Optional[int] = 1
    session_length_in_minutes: Optional[float] = 5.0
    phone_mobile_valid: Optional[int] = 1
    prev_address_months_count: Optional[int] = 24


@app.post("/api/simulate")
def simulate_scenario(body: SimulationPayload):
    vel_ratio = (body.velocity_6h or 2000) / max(1.0, (body.velocity_4w or 3000))
    name_email = body.name_email_similarity if body.name_email_similarity is not None else 0.5
    income_val = max(0.1, body.income or 0.5)
    limit_ratio = (body.proposed_credit_limit or 1000) / (income_val * 10000.0)
    dob_emails = body.date_of_birth_distinct_emails_4w or 1
    session_mins = body.session_length_in_minutes or 5.0

    logit = -3.8
    logit += max(0.0, (vel_ratio - 1.2) * 1.6)
    logit += (1.0 - name_email) * 2.8
    logit += (limit_ratio - 0.2) * 2.2
    logit += min(2.5, (dob_emails - 1) * 0.18)
    if body.device_os in ("Linux", "X11"):
        logit += 0.85
    if session_mins < 1.5:
        logit += 0.95
    if body.phone_mobile_valid == 0:
        logit += 0.75
    if body.prev_address_months_count == -1:
        logit += 0.65

    prob = float(1.0 / (1.0 + np.exp(-logit)))
    tier = "priority" if prob >= 0.88 else "standard" if prob >= 0.65 else "normal"

    return {
        "risk_score": round(prob, 4),
        "risk_tier_code": tier,
        "logit": round(logit, 3),
        "contributions": [
            {"feature": "Name-Email Discordance", "delta": round((1.0 - name_email) * 0.24, 3)},
            {"feature": "Velocity 6h Surge", "delta": round(max(0.0, (vel_ratio - 1.0) * 0.22), 3)},
            {"feature": "DOB Shared Email Cluster", "delta": round(min(0.25, (dob_emails - 1) * 0.02), 3)},
            {"feature": "Credit Limit Overextension", "delta": round((limit_ratio - 0.2) * 0.18, 3)}
        ]
    }


# Mount Frontend Static Directory
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
