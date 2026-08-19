"""
tests/test_golden_regression.py -- Regression tests on a golden set of 30 representative applications.

Verifies:
1. Deterministic scoring (repeated scoring produces bitwise identical probabilities and risk levels).
2. Stability and robustness (no NaNs, no infinities across diverse applicant archetypes).
3. Monotonic risk tiering (high-risk archetypes score significantly higher than legitimate ones).
4. Batch vs single-row invariance (batch prediction matches item-by-item prediction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import models
from src.prediction import predict_dataframe
from src.preprocessing import Preprocessor

# 30 Golden Applications covering diverse fraud archetypes and edge cases
GOLDEN_APPLICATIONS: list[dict] = [
    # --- 10 Low-Risk / Legitimate Archetypes ---
    {
        "income": 0.8, "customer_age": 45, "employment_status": "CA", "housing_status": "BA",
        "name_email_similarity": 0.95, "prev_address_months_count": 60, "current_address_months_count": 120,
        "bank_months_count": 24, "days_since_request": 0.01, "velocity_6h": 1200.0, "velocity_24h": 2500.0,
        "velocity_4w": 3500.0, "zip_count_4w": 50, "bank_branch_count_8w": 5, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 15.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 280.0, "proposed_credit_limit": 1500.0,
        "payment_type": "AA", "has_other_cards": 1, "intended_balcon_amount": -1.0, "month": 1, "device_fraud_count": 0,
    },
    {
        "income": 0.7, "customer_age": 55, "employment_status": "CB", "housing_status": "BB",
        "name_email_similarity": 0.88, "prev_address_months_count": 80, "current_address_months_count": 96,
        "bank_months_count": 18, "days_since_request": 0.05, "velocity_6h": 800.0, "velocity_24h": 1800.0,
        "velocity_4w": 2900.0, "zip_count_4w": 30, "bank_branch_count_8w": 3, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "macintosh", "device_distinct_emails_8w": 1, "session_length_in_minutes": 20.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 310.0, "proposed_credit_limit": 2000.0,
        "payment_type": "AB", "has_other_cards": 1, "intended_balcon_amount": 50.0, "month": 2, "device_fraud_count": 0,
    },
    {
        "income": 0.6, "customer_age": 38, "employment_status": "CA", "housing_status": "BC",
        "name_email_similarity": 0.90, "prev_address_months_count": 36, "current_address_months_count": 48,
        "bank_months_count": 12, "days_since_request": 0.1, "velocity_6h": 1500.0, "velocity_24h": 3000.0,
        "velocity_4w": 4000.0, "zip_count_4w": 70, "bank_branch_count_8w": 10, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 1, "foreign_request": 0,
        "device_os": "linux", "device_distinct_emails_8w": 1, "session_length_in_minutes": 10.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 190.0, "proposed_credit_limit": 800.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 20.0, "month": 3, "device_fraud_count": 0,
    },
    {
        "income": 0.9, "customer_age": 60, "employment_status": "CA", "housing_status": "BA",
        "name_email_similarity": 0.92, "prev_address_months_count": 120, "current_address_months_count": 200,
        "bank_months_count": 30, "days_since_request": 0.02, "velocity_6h": 600.0, "velocity_24h": 1400.0,
        "velocity_4w": 2600.0, "zip_count_4w": 20, "bank_branch_count_8w": 2, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 25.0,
        "keep_alive_session": 1, "source": "TELEAPP", "credit_risk_score": 350.0, "proposed_credit_limit": 3000.0,
        "payment_type": "AA", "has_other_cards": 1, "intended_balcon_amount": -1.0, "month": 4, "device_fraud_count": 0,
    },
    {
        "income": 0.5, "customer_age": 28, "employment_status": "CB", "housing_status": "BD",
        "name_email_similarity": 0.85, "prev_address_months_count": 24, "current_address_months_count": 36,
        "bank_months_count": 8, "days_since_request": 0.08, "velocity_6h": 1800.0, "velocity_24h": 3200.0,
        "velocity_4w": 4500.0, "zip_count_4w": 90, "bank_branch_count_8w": 8, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "macintosh", "device_distinct_emails_8w": 1, "session_length_in_minutes": 8.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 160.0, "proposed_credit_limit": 600.0,
        "payment_type": "AD", "has_other_cards": 0, "intended_balcon_amount": 10.0, "month": 5, "device_fraud_count": 0,
    },
    {
        "income": 0.75, "customer_age": 50, "employment_status": "CA", "housing_status": "BA",
        "name_email_similarity": 0.99, "prev_address_months_count": 48, "current_address_months_count": 80,
        "bank_months_count": 20, "days_since_request": 0.03, "velocity_6h": 1100.0, "velocity_24h": 2200.0,
        "velocity_4w": 3400.0, "zip_count_4w": 40, "bank_branch_count_8w": 4, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 18.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 270.0, "proposed_credit_limit": 1800.0,
        "payment_type": "AA", "has_other_cards": 1, "intended_balcon_amount": -1.0, "month": 6, "device_fraud_count": 0,
    },
    {
        "income": 0.65, "customer_age": 32, "employment_status": "CC", "housing_status": "BC",
        "name_email_similarity": 0.82, "prev_address_months_count": 18, "current_address_months_count": 24,
        "bank_months_count": 6, "days_since_request": 0.15, "velocity_6h": 1400.0, "velocity_24h": 2700.0,
        "velocity_4w": 3900.0, "zip_count_4w": 65, "bank_branch_count_8w": 7, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 1, "foreign_request": 0,
        "device_os": "x11", "device_distinct_emails_8w": 1, "session_length_in_minutes": 12.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 140.0, "proposed_credit_limit": 500.0,
        "payment_type": "AE", "has_other_cards": 0, "intended_balcon_amount": 30.0, "month": 7, "device_fraud_count": 0,
    },
    {
        "income": 0.85, "customer_age": 48, "employment_status": "CA", "housing_status": "BA",
        "name_email_similarity": 0.94, "prev_address_months_count": 72, "current_address_months_count": 110,
        "bank_months_count": 22, "days_since_request": 0.04, "velocity_6h": 950.0, "velocity_24h": 2100.0,
        "velocity_4w": 3200.0, "zip_count_4w": 45, "bank_branch_count_8w": 4, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "macintosh", "device_distinct_emails_8w": 1, "session_length_in_minutes": 22.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 300.0, "proposed_credit_limit": 2200.0,
        "payment_type": "AA", "has_other_cards": 1, "intended_balcon_amount": -1.0, "month": 0, "device_fraud_count": 0,
    },
    {
        "income": 0.55, "customer_age": 35, "employment_status": "CD", "housing_status": "BE",
        "name_email_similarity": 0.89, "prev_address_months_count": 30, "current_address_months_count": 40,
        "bank_months_count": 10, "days_since_request": 0.12, "velocity_6h": 1600.0, "velocity_24h": 3100.0,
        "velocity_4w": 4200.0, "zip_count_4w": 80, "bank_branch_count_8w": 6, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 14.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 180.0, "proposed_credit_limit": 750.0,
        "payment_type": "AB", "has_other_cards": 0, "intended_balcon_amount": 15.0, "month": 2, "device_fraud_count": 0,
    },
    {
        "income": 0.95, "customer_age": 52, "employment_status": "CA", "housing_status": "BA",
        "name_email_similarity": 0.98, "prev_address_months_count": 100, "current_address_months_count": 160,
        "bank_months_count": 28, "days_since_request": 0.01, "velocity_6h": 700.0, "velocity_24h": 1600.0,
        "velocity_4w": 2800.0, "zip_count_4w": 25, "bank_branch_count_8w": 3, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 30.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 380.0, "proposed_credit_limit": 3500.0,
        "payment_type": "AA", "has_other_cards": 1, "intended_balcon_amount": -1.0, "month": 4, "device_fraud_count": 0,
    },

    # --- 10 High-Risk / Clear Fraud Archetypes ---
    {
        "income": 0.1, "customer_age": 22, "employment_status": "CE", "housing_status": "BE",
        "name_email_similarity": 0.05, "prev_address_months_count": -1, "current_address_months_count": 1,
        "bank_months_count": -1, "days_since_request": 10.5, "velocity_6h": 9800.0, "velocity_24h": 15000.0,
        "velocity_4w": 20000.0, "zip_count_4w": 2500, "bank_branch_count_8w": 150, "date_of_birth_distinct_emails_4w": 8,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "other", "device_distinct_emails_8w": 2, "session_length_in_minutes": 1.0,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -180.0, "proposed_credit_limit": 2000.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 100.0, "month": 5, "device_fraud_count": 0,
    },
    {
        "income": 0.1, "customer_age": 19, "employment_status": "CE", "housing_status": "BD",
        "name_email_similarity": 0.02, "prev_address_months_count": -1, "current_address_months_count": 0,
        "bank_months_count": -1, "days_since_request": 15.0, "velocity_6h": 12500.0, "velocity_24h": 18000.0,
        "velocity_4w": 22000.0, "zip_count_4w": 3200, "bank_branch_count_8w": 200, "date_of_birth_distinct_emails_4w": 12,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "other", "device_distinct_emails_8w": 2, "session_length_in_minutes": 0.5,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -190.0, "proposed_credit_limit": 2500.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 110.0, "month": 6, "device_fraud_count": 0,
    },
    {
        "income": 0.2, "customer_age": 25, "employment_status": "CE", "housing_status": "BE",
        "name_email_similarity": 0.10, "prev_address_months_count": -1, "current_address_months_count": 2,
        "bank_months_count": -1, "days_since_request": 8.0, "velocity_6h": 8500.0, "velocity_24h": 13000.0,
        "velocity_4w": 18000.0, "zip_count_4w": 1800, "bank_branch_count_8w": 90, "date_of_birth_distinct_emails_4w": 6,
        "phone_home_valid": 0, "phone_mobile_valid": 1, "email_is_free": 1, "foreign_request": 1,
        "device_os": "linux", "device_distinct_emails_8w": 2, "session_length_in_minutes": 1.5,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -150.0, "proposed_credit_limit": 1800.0,
        "payment_type": "AD", "has_other_cards": 0, "intended_balcon_amount": 90.0, "month": 7, "device_fraud_count": 0,
    },
    {
        "income": 0.1, "customer_age": 30, "employment_status": "CE", "housing_status": "BE",
        "name_email_similarity": 0.08, "prev_address_months_count": -1, "current_address_months_count": 1,
        "bank_months_count": -1, "days_since_request": 12.0, "velocity_6h": 11000.0, "velocity_24h": 16000.0,
        "velocity_4w": 21000.0, "zip_count_4w": 2900, "bank_branch_count_8w": 170, "date_of_birth_distinct_emails_4w": 10,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "other", "device_distinct_emails_8w": 2, "session_length_in_minutes": 0.8,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -175.0, "proposed_credit_limit": 2200.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 95.0, "month": 0, "device_fraud_count": 0,
    },
    {
        "income": 0.2, "customer_age": 21, "employment_status": "CE", "housing_status": "BD",
        "name_email_similarity": 0.04, "prev_address_months_count": -1, "current_address_months_count": 0,
        "bank_months_count": -1, "days_since_request": 14.0, "velocity_6h": 10500.0, "velocity_24h": 17000.0,
        "velocity_4w": 19500.0, "zip_count_4w": 2400, "bank_branch_count_8w": 120, "date_of_birth_distinct_emails_4w": 7,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "x11", "device_distinct_emails_8w": 2, "session_length_in_minutes": 1.2,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -160.0, "proposed_credit_limit": 1900.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 85.0, "month": 1, "device_fraud_count": 0,
    },
    {
        "income": 0.1, "customer_age": 24, "employment_status": "CE", "housing_status": "BE",
        "name_email_similarity": 0.01, "prev_address_months_count": -1, "current_address_months_count": 1,
        "bank_months_count": -1, "days_since_request": 20.0, "velocity_6h": 14000.0, "velocity_24h": 20000.0,
        "velocity_4w": 25000.0, "zip_count_4w": 4000, "bank_branch_count_8w": 250, "date_of_birth_distinct_emails_4w": 15,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "other", "device_distinct_emails_8w": 2, "session_length_in_minutes": 0.4,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -195.0, "proposed_credit_limit": 2800.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 105.0, "month": 3, "device_fraud_count": 0,
    },
    {
        "income": 0.3, "customer_age": 26, "employment_status": "CD", "housing_status": "BE",
        "name_email_similarity": 0.12, "prev_address_months_count": -1, "current_address_months_count": 3,
        "bank_months_count": -1, "days_since_request": 7.5, "velocity_6h": 7800.0, "velocity_24h": 12000.0,
        "velocity_4w": 16000.0, "zip_count_4w": 1500, "bank_branch_count_8w": 80, "date_of_birth_distinct_emails_4w": 5,
        "phone_home_valid": 0, "phone_mobile_valid": 1, "email_is_free": 1, "foreign_request": 1,
        "device_os": "linux", "device_distinct_emails_8w": 2, "session_length_in_minutes": 2.0,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -120.0, "proposed_credit_limit": 1500.0,
        "payment_type": "AD", "has_other_cards": 0, "intended_balcon_amount": 75.0, "month": 4, "device_fraud_count": 0,
    },
    {
        "income": 0.1, "customer_age": 20, "employment_status": "CE", "housing_status": "BD",
        "name_email_similarity": 0.03, "prev_address_months_count": -1, "current_address_months_count": 0,
        "bank_months_count": -1, "days_since_request": 16.0, "velocity_6h": 13000.0, "velocity_24h": 19000.0,
        "velocity_4w": 23000.0, "zip_count_4w": 3500, "bank_branch_count_8w": 220, "date_of_birth_distinct_emails_4w": 14,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "other", "device_distinct_emails_8w": 2, "session_length_in_minutes": 0.6,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -185.0, "proposed_credit_limit": 2600.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 100.0, "month": 5, "device_fraud_count": 0,
    },
    {
        "income": 0.2, "customer_age": 23, "employment_status": "CE", "housing_status": "BE",
        "name_email_similarity": 0.07, "prev_address_months_count": -1, "current_address_months_count": 1,
        "bank_months_count": -1, "days_since_request": 11.0, "velocity_6h": 9200.0, "velocity_24h": 14500.0,
        "velocity_4w": 18500.0, "zip_count_4w": 2100, "bank_branch_count_8w": 110, "date_of_birth_distinct_emails_4w": 8,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "x11", "device_distinct_emails_8w": 2, "session_length_in_minutes": 1.0,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -165.0, "proposed_credit_limit": 2100.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 90.0, "month": 6, "device_fraud_count": 0,
    },
    {
        "income": 0.1, "customer_age": 27, "employment_status": "CE", "housing_status": "BE",
        "name_email_similarity": 0.06, "prev_address_months_count": -1, "current_address_months_count": 2,
        "bank_months_count": -1, "days_since_request": 13.0, "velocity_6h": 11500.0, "velocity_24h": 16500.0,
        "velocity_4w": 21500.0, "zip_count_4w": 2800, "bank_branch_count_8w": 160, "date_of_birth_distinct_emails_4w": 11,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "other", "device_distinct_emails_8w": 2, "session_length_in_minutes": 0.7,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -170.0, "proposed_credit_limit": 2300.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 98.0, "month": 7, "device_fraud_count": 0,
    },

    # --- 5 Boundary Cases ---
    {
        "income": 0.4, "customer_age": 33, "employment_status": "CB", "housing_status": "BC",
        "name_email_similarity": 0.50, "prev_address_months_count": 12, "current_address_months_count": 18,
        "bank_months_count": 5, "days_since_request": 2.0, "velocity_6h": 3500.0, "velocity_24h": 5000.0,
        "velocity_4w": 6500.0, "zip_count_4w": 300, "bank_branch_count_8w": 25, "date_of_birth_distinct_emails_4w": 2,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 1, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 5.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 50.0, "proposed_credit_limit": 1000.0,
        "payment_type": "AB", "has_other_cards": 0, "intended_balcon_amount": 25.0, "month": 1, "device_fraud_count": 0,
    },
    {
        "income": 0.45, "customer_age": 36, "employment_status": "CC", "housing_status": "BD",
        "name_email_similarity": 0.55, "prev_address_months_count": 10, "current_address_months_count": 15,
        "bank_months_count": 4, "days_since_request": 1.8, "velocity_6h": 3800.0, "velocity_24h": 5400.0,
        "velocity_4w": 6800.0, "zip_count_4w": 350, "bank_branch_count_8w": 30, "date_of_birth_distinct_emails_4w": 2,
        "phone_home_valid": 0, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "macintosh", "device_distinct_emails_8w": 1, "session_length_in_minutes": 4.5,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 40.0, "proposed_credit_limit": 900.0,
        "payment_type": "AC", "has_other_cards": 1, "intended_balcon_amount": 15.0, "month": 2, "device_fraud_count": 0,
    },
    {
        "income": 0.35, "customer_age": 29, "employment_status": "CB", "housing_status": "BC",
        "name_email_similarity": 0.48, "prev_address_months_count": 8, "current_address_months_count": 12,
        "bank_months_count": 3, "days_since_request": 2.5, "velocity_6h": 4100.0, "velocity_24h": 5800.0,
        "velocity_4w": 7200.0, "zip_count_4w": 400, "bank_branch_count_8w": 35, "date_of_birth_distinct_emails_4w": 2,
        "phone_home_valid": 1, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 0,
        "device_os": "linux", "device_distinct_emails_8w": 1, "session_length_in_minutes": 6.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 60.0, "proposed_credit_limit": 1100.0,
        "payment_type": "AA", "has_other_cards": 0, "intended_balcon_amount": 35.0, "month": 3, "device_fraud_count": 0,
    },
    {
        "income": 0.5, "customer_age": 42, "employment_status": "CD", "housing_status": "BE",
        "name_email_similarity": 0.60, "prev_address_months_count": 14, "current_address_months_count": 20,
        "bank_months_count": 6, "days_since_request": 1.5, "velocity_6h": 3200.0, "velocity_24h": 4800.0,
        "velocity_4w": 6200.0, "zip_count_4w": 280, "bank_branch_count_8w": 22, "date_of_birth_distinct_emails_4w": 2,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 7.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 75.0, "proposed_credit_limit": 1200.0,
        "payment_type": "AD", "has_other_cards": 1, "intended_balcon_amount": 10.0, "month": 4, "device_fraud_count": 0,
    },
    {
        "income": 0.42, "customer_age": 31, "employment_status": "CA", "housing_status": "BC",
        "name_email_similarity": 0.52, "prev_address_months_count": 9, "current_address_months_count": 14,
        "bank_months_count": 4, "days_since_request": 2.2, "velocity_6h": 3900.0, "velocity_24h": 5500.0,
        "velocity_4w": 7000.0, "zip_count_4w": 370, "bank_branch_count_8w": 28, "date_of_birth_distinct_emails_4w": 2,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 1, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 5.5,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 45.0, "proposed_credit_limit": 950.0,
        "payment_type": "AB", "has_other_cards": 0, "intended_balcon_amount": 20.0, "month": 5, "device_fraud_count": 0,
    },

    # --- 5 Special Feature Edge Profiles ---
    {
        "income": 0.7, "customer_age": 75, "employment_status": "CA", "housing_status": "BA",
        "name_email_similarity": 0.95, "prev_address_months_count": -1, "current_address_months_count": 300,
        "bank_months_count": -1, "days_since_request": 0.01, "velocity_6h": 500.0, "velocity_24h": 1000.0,
        "velocity_4w": 2000.0, "zip_count_4w": 15, "bank_branch_count_8w": 1, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 0, "email_is_free": 0, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": -1, "session_length_in_minutes": -1.0,
        "keep_alive_session": 1, "source": "TELEAPP", "credit_risk_score": 250.0, "proposed_credit_limit": 1000.0,
        "payment_type": "AA", "has_other_cards": 1, "intended_balcon_amount": -1.0, "month": 6, "device_fraud_count": 0,
    },
    {
        "income": 0.3, "customer_age": 18, "employment_status": "CB", "housing_status": "BD",
        "name_email_similarity": 0.70, "prev_address_months_count": -1, "current_address_months_count": 6,
        "bank_months_count": -1, "days_since_request": 0.5, "velocity_6h": 2200.0, "velocity_24h": 3800.0,
        "velocity_4w": 5100.0, "zip_count_4w": 120, "bank_branch_count_8w": 12, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 0, "phone_mobile_valid": 1, "email_is_free": 1, "foreign_request": 0,
        "device_os": "other", "device_distinct_emails_8w": 1, "session_length_in_minutes": 4.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 110.0, "proposed_credit_limit": 400.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": -1.0, "month": 7, "device_fraud_count": 0,
    },
    {
        "income": 0.8, "customer_age": 48, "employment_status": "CA", "housing_status": "BA",
        "name_email_similarity": 0.85, "prev_address_months_count": 50, "current_address_months_count": 80,
        "bank_months_count": 15, "days_since_request": 0.02, "velocity_6h": -50.0, "velocity_24h": 1500.0,
        "velocity_4w": 2800.0, "zip_count_4w": 35, "bank_branch_count_8w": 3, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 1,
        "device_os": "macintosh", "device_distinct_emails_8w": 1, "session_length_in_minutes": 16.0,
        "keep_alive_session": 1, "source": "INTERNET", "credit_risk_score": 220.0, "proposed_credit_limit": 1600.0,
        "payment_type": "AB", "has_other_cards": 1, "intended_balcon_amount": -1.0, "month": 0, "device_fraud_count": 0,
    },
    {
        "income": 0.5, "customer_age": 62, "employment_status": "CB", "housing_status": "BA",
        "name_email_similarity": 0.90, "prev_address_months_count": 80, "current_address_months_count": 150,
        "bank_months_count": 25, "days_since_request": 0.05, "velocity_6h": 900.0, "velocity_24h": 1900.0,
        "velocity_4w": 3100.0, "zip_count_4w": 40, "bank_branch_count_8w": 4, "date_of_birth_distinct_emails_4w": 1,
        "phone_home_valid": 1, "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
        "device_os": "windows", "device_distinct_emails_8w": 1, "session_length_in_minutes": 20.0,
        "keep_alive_session": 1, "source": "TELEAPP", "credit_risk_score": -40.0, "proposed_credit_limit": 1200.0,
        "payment_type": "AA", "has_other_cards": 1, "intended_balcon_amount": -1.0, "month": 1, "device_fraud_count": 0,
    },
    {
        "income": 0.2, "customer_age": 82, "employment_status": "CE", "housing_status": "BE",
        "name_email_similarity": 0.15, "prev_address_months_count": -1, "current_address_months_count": 2,
        "bank_months_count": -1, "days_since_request": 10.0, "velocity_6h": 6500.0, "velocity_24h": 11000.0,
        "velocity_4w": 15000.0, "zip_count_4w": 1200, "bank_branch_count_8w": 60, "date_of_birth_distinct_emails_4w": 4,
        "phone_home_valid": 0, "phone_mobile_valid": 0, "email_is_free": 1, "foreign_request": 1,
        "device_os": "other", "device_distinct_emails_8w": 2, "session_length_in_minutes": 1.0,
        "keep_alive_session": 0, "source": "INTERNET", "credit_risk_score": -110.0, "proposed_credit_limit": 1400.0,
        "payment_type": "AC", "has_other_cards": 0, "intended_balcon_amount": 60.0, "month": 2, "device_fraud_count": 0,
    },
]


def test_golden_dataset_size():
    """Verify exactly 30 representative golden records exist."""
    assert len(GOLDEN_APPLICATIONS) == 30


def test_deterministic_scoring(synthetic_baf_df, cfg):
    """Verify multiple scoring runs produce bitwise identical probabilities and risk classifications."""
    prep = Preprocessor(cfg)
    prep.fit(synthetic_baf_df)
    X_dense = prep.transform_dense(synthetic_baf_df)
    y = prep.get_target(synthetic_baf_df)

    lr_model = models.train_logistic_regression(X_dense, y, cfg, {"class_weight": "balanced"}, seed=42)
    meta = {
        "model_type": "logistic_regression",
        "strategy": "class_weight",
        "threshold": 0.05,
        "feature_columns": list(X_dense.columns),
        "model_iteration": "golden_test",
    }

    df_golden = pd.DataFrame(GOLDEN_APPLICATIONS)

    run_1 = predict_dataframe(df_golden, lr_model, prep, meta, cfg)
    run_2 = predict_dataframe(df_golden, lr_model, prep, meta, cfg)
    run_3 = predict_dataframe(df_golden, lr_model, prep, meta, cfg)

    # Probabilities must be bitwise identical
    np.testing.assert_array_equal(run_1["fraud_probability"].to_numpy(), run_2["fraud_probability"].to_numpy())
    np.testing.assert_array_equal(run_2["fraud_probability"].to_numpy(), run_3["fraud_probability"].to_numpy())

    # Predictions and risk levels must match exactly
    assert list(run_1["fraud_prediction"]) == list(run_2["fraud_prediction"]) == list(run_3["fraud_prediction"])
    assert list(run_1["risk_level"]) == list(run_2["risk_level"]) == list(run_3["risk_level"])


def test_monotonic_risk_tiering(synthetic_baf_df, cfg):
    """Verify high-risk fraud archetypes score strictly higher on average than legitimate profiles."""
    prep = Preprocessor(cfg)
    prep.fit(synthetic_baf_df)
    X_dense = prep.transform_dense(synthetic_baf_df)
    y = prep.get_target(synthetic_baf_df)

    lr_model = models.train_logistic_regression(X_dense, y, cfg, {"class_weight": "balanced"}, seed=42)
    meta = {
        "model_type": "logistic_regression",
        "strategy": "class_weight",
        "threshold": 0.05,
        "feature_columns": list(X_dense.columns),
        "model_iteration": "golden_test",
    }

    df_golden = pd.DataFrame(GOLDEN_APPLICATIONS)
    scored = predict_dataframe(df_golden, lr_model, prep, meta, cfg)

    low_risk_probs = scored.iloc[0:10]["fraud_probability"]
    high_risk_probs = scored.iloc[10:20]["fraud_probability"]

    # Average fraud probability for fraud archetypes must be higher than legitimate ones
    assert high_risk_probs.mean() > low_risk_probs.mean()

    # Verify no invalid numerical values
    assert not scored["fraud_probability"].isna().any()
    assert not np.isinf(scored["fraud_probability"].to_numpy()).any()
    assert (scored["fraud_probability"] >= 0.0).all()
    assert (scored["fraud_probability"] <= 1.0).all()


def test_batch_vs_single_row_invariance(synthetic_baf_df, cfg):
    """Scoring rows in a batch of 30 vs row-by-row must yield identical scores."""
    prep = Preprocessor(cfg)
    prep.fit(synthetic_baf_df)
    X_dense = prep.transform_dense(synthetic_baf_df)
    y = prep.get_target(synthetic_baf_df)

    lr_model = models.train_logistic_regression(X_dense, y, cfg, {"class_weight": "balanced"}, seed=42)
    meta = {
        "model_type": "logistic_regression",
        "strategy": "class_weight",
        "threshold": 0.05,
        "feature_columns": list(X_dense.columns),
        "model_iteration": "golden_test",
    }

    df_golden = pd.DataFrame(GOLDEN_APPLICATIONS)
    batch_scored = predict_dataframe(df_golden, lr_model, prep, meta, cfg)

    single_probs = []
    for app in GOLDEN_APPLICATIONS:
        single_df = pd.DataFrame([app])
        res = predict_dataframe(single_df, lr_model, prep, meta, cfg)
        single_probs.append(res["fraud_probability"].iloc[0])

    np.testing.assert_allclose(batch_scored["fraud_probability"].to_numpy(), single_probs, rtol=1e-5)
