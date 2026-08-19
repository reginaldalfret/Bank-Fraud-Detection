/* =========================================================================
   SENTINEL Bank Fraud Classification Platform - Authentic BAF Dataset & Mock Fixtures
   Based on NeurIPS 2022 Bank Account Fraud (BAF) Benchmark & Blend Models
   ========================================================================= */

const MockData = (() => {
  // 1. Overview & System Monitoring Data
  const monitorKPIs = {
    total_applications: 1000000,
    predicted_fraud_count: 11029,
    fraud_rate: 0.011029, // 1.10%
    pr_auc: 0.167736,
    roc_auc: 0.897919,
    tpr_at_5pct_fpr: 0.550317, // 55.03% caught at 5% false positive rate
    precision_at_opt: 0.248,
    recall_at_opt: 0.764,
    f1_score: 0.3745,
    avg_application_limit: 842.50,
    active_model_name: "CatBoost + XGBoost Ensemble v2.4",
    model_timestamp: "2026-08-19T00:00:00Z",
    
    // Score Distribution Histogram (0.0 to 1.0)
    score_distribution: [
      { bin: "0.00 - 0.10", count: 742150, pct: 74.21, risk: "low" },
      { bin: "0.10 - 0.20", count: 148900, pct: 14.89, risk: "low" },
      { bin: "0.20 - 0.30", count: 48200, pct: 4.82, risk: "moderate" },
      { bin: "0.30 - 0.40", count: 22400, pct: 2.24, risk: "moderate" },
      { bin: "0.40 - 0.50", count: 14100, pct: 1.41, risk: "moderate" },
      { bin: "0.50 - 0.60", count: 9600, pct: 0.96, risk: "high" },
      { bin: "0.60 - 0.70", count: 6200, pct: 0.62, risk: "high" },
      { bin: "0.70 - 0.80", count: 4150, pct: 0.41, risk: "high" },
      { bin: "0.80 - 0.90", count: 2800, pct: 0.28, risk: "critical" },
      { bin: "0.90 - 1.00", count: 1500, pct: 0.15, risk: "critical" }
    ],

    // Temporal Drift Analysis across Month 0 to Month 7
    monthly_trend: [
      { month: "Month 0", month_idx: 0, applications: 125400, fraud_count: 1320, fraud_rate: 0.0105, detection_rate: 0.562, split: "Train" },
      { month: "Month 1", month_idx: 1, applications: 128900, fraud_count: 1210, fraud_rate: 0.0094, detection_rate: 0.558, split: "Train" },
      { month: "Month 2", month_idx: 2, applications: 121300, fraud_count: 1080, fraud_rate: 0.0089, detection_rate: 0.549, split: "Train" },
      { month: "Month 3", month_idx: 3, applications: 134200, fraud_count: 1450, fraud_rate: 0.0108, detection_rate: 0.554, split: "Train" },
      { month: "Month 4", month_idx: 4, applications: 126100, fraud_count: 1410, fraud_rate: 0.0112, detection_rate: 0.551, split: "Train" },
      { month: "Month 5", month_idx: 5, applications: 132500, fraud_count: 1590, fraud_rate: 0.0120, detection_rate: 0.548, split: "Validation" },
      { month: "Month 6", month_idx: 6, applications: 118600, fraud_count: 1610, fraud_rate: 0.0136, detection_rate: 0.539, split: "Out-of-Time Test" },
      { month: "Month 7", month_idx: 7, applications: 113000, fraud_count: 1359, fraud_rate: 0.0120, detection_rate: 0.542, split: "Out-of-Time Test" }
    ],

    // Top Global Risk Drivers (Global SHAP Importance)
    top_risk_indicators: [
      { feature: "velocity_6h_to_4w_ratio", label: "Velocity Burst Ratio (6h / 4w)", importance: 0.184, direction: "positive", category: "Velocity" },
      { feature: "name_email_similarity", label: "Name & Email Similarity", importance: 0.162, direction: "negative", category: "Identity" },
      { feature: "credit_risk_score", label: "Credit Risk Score", importance: 0.145, direction: "positive", category: "Credit" },
      { feature: "proposed_credit_limit_to_income", label: "Requested Limit / Income Decile", importance: 0.128, direction: "positive", category: "Financial" },
      { feature: "date_of_birth_distinct_emails_4w", label: "DOB Distinct Emails (4-Week Cluster)", importance: 0.109, direction: "positive", category: "Identity" },
      { feature: "housing_status", label: "Housing Status (Rental / Social)", importance: 0.091, direction: "positive", category: "Demographic" },
      { feature: "prev_address_months_count", label: "Missing Prev Address History (-1)", importance: 0.078, direction: "positive", category: "Tenure" },
      { feature: "device_distinct_emails_8w", label: "Device Shared Email Count (8w)", importance: 0.063, direction: "positive", category: "Device" },
      { feature: "session_length_in_minutes", label: "Abnormally Rapid Session (<2 min)", importance: 0.040, direction: "positive", category: "Behavior" }
    ]
  };

  // 2. Applicant Database & Investigation Queue
  const applicants = [
    {
      application_id: "APP-2026-984210",
      applicant_name: "K. Vance / Machine Gen",
      timestamp: "2026-08-19T17:28:44Z",
      customer_age: 30, // Age bracket 30s
      income: 0.2, // Decile 0.2
      employment_status: "CE", // Unemployed / Seeking
      housing_status: "BF", // Temporary / Hostel
      name_email_similarity: 0.042, // Extreme mismatch
      email_is_free: 1,
      phone_home_valid: 0,
      phone_mobile_valid: 1,
      foreign_request: 0,
      credit_risk_score: 284, // High risk
      proposed_credit_limit: 1800,
      intended_balcon_amount: -1,
      payment_type: "AC",
      has_other_cards: 0,
      prev_address_months_count: -1, // Missing history
      current_address_months_count: 2,
      bank_months_count: -1, // No banking history
      days_since_request: 0.02,
      velocity_6h: 8420.5,
      velocity_24h: 7650.0,
      velocity_4w: 3120.0,
      zip_count_4w: 4210,
      bank_branch_count_8w: 180,
      date_of_birth_distinct_emails_4w: 26, // High cluster
      device_os: "Linux",
      device_distinct_emails_8w: 3,
      session_length_in_minutes: 0.85,
      keep_alive_session: 0,
      device_fraud_count: 0,
      month: 7,
      source: "INTERNET",
      
      // Model Output
      risk_score: 0.9684,
      risk_tier_code: "priority", // Priority Review (P99)
      status: "pending", // pending, under_review, escalated, marked_legitimate, confirmed_fraud
      notes: "Flagged by burst velocity and zero email-name concordance.",
      assigned_analyst: "FraudOps Lead (Triage)",
      
      // SHAP Waterfall Attribution (Base logit: -4.49 -> final score 0.9684)
      shap_waterfall: {
        base_value: 0.011, // Expected baseline
        final_score: 0.9684,
        features: [
          { name: "Velocity Burst (6h vs 4w)", value: "8,420 apps/hr", contribution: 0.285, impact: "increase" },
          { name: "Name-Email Similarity", value: "0.042 (Discordant)", contribution: 0.241, impact: "increase" },
          { name: "DOB Distinct Emails Cluster", value: "26 emails", contribution: 0.198, impact: "increase" },
          { name: "Requested Limit vs Income", value: "$1,800 @ Decile 0.2", contribution: 0.142, impact: "increase" },
          { name: "No Prev Address History", value: "Missing (-1)", contribution: 0.115, impact: "increase" },
          { name: "Session Speed / Automation", value: "0.85 mins", contribution: 0.082, impact: "increase" },
          { name: "Credit Risk Score", value: "+284 pts", contribution: 0.054, impact: "increase" },
          { name: "Mobile Phone Valid", value: "Verified (1)", contribution: -0.049, impact: "decrease" },
          { name: "Domestic Request", value: "Domestic (0)", contribution: -0.038, impact: "decrease" }
        ]
      },

      // 6-Axis Behavioral Deviation Radar (Scale 0 to 100)
      radar_comparison: {
        axes: ["Velocity Burst", "Identity Mismatch", "Credit Overreach", "Thin File Signal", "Device Multi-Tenancy", "Session Automation"],
        applicant: [95, 96, 88, 92, 85, 90],
        population_normal: [18, 12, 25, 20, 10, 15],
        population_fraud: [88, 89, 82, 86, 78, 84]
      },

      // Nemotron AI Analyst Report
      nemotron_report: {
        status: "online",
        model: "Nemotron-70B-Fraud-Analyst",
        generated_at: "2026-08-19T17:29:10Z",
        investigation_priority: "CRITICAL_ESCALATE",
        sla_window: "< 15 Minutes",
        executive_summary: "High-conviction synthetic identity application orchestrated as part of a high-velocity automated farming attack. The applicant displays an almost total disconnection between legal name and email address (0.042 score), combined with an address history gap (-1) and a sharp 2.7x burst in 6-hour application velocity.",
        key_reasons: [
          "Extreme Name-Email Discordance: Random alpha-numeric email syntax bearing 0.04 similarity to applicant name.",
          "Synthetic Identity Farming Cluster: 26 distinct email applications recorded sharing this identical Date of Birth in the past 4 weeks.",
          "Credit Overextension: Low income rank (Decile 0.2) paired with maximum tier requested credit limit ($1,800).",
          "Automated Script Profile: Sub-minute application submission (0.85 min session) with Linux browser agent and disabled keep-alive."
        ],
        verification_checklist: [
          { item: "Trigger mandatory Step-Up Video KYC with Liveness detection.", checked: false, critical: true },
          { item: "Request proof of primary residential address (utility bill / lease).", checked: false, critical: true },
          { item: "Cross-reference IP / Device cluster for linked applications in ZIP 4210.", checked: false, critical: false },
          { item: "Place temporary account lock and hold credit card issuance.", checked: true, critical: true }
        ]
      }
    },

    {
      application_id: "APP-2026-984211",
      applicant_name: "M. Sterling / Multi-Device",
      timestamp: "2026-08-19T17:24:12Z",
      customer_age: 50,
      income: 0.4,
      employment_status: "CA",
      housing_status: "BC",
      name_email_similarity: 0.124,
      email_is_free: 1,
      phone_home_valid: 1,
      phone_mobile_valid: 0,
      foreign_request: 1,
      credit_risk_score: 210,
      proposed_credit_limit: 1500,
      intended_balcon_amount: 50,
      payment_type: "AB",
      has_other_cards: 0,
      prev_address_months_count: 12,
      current_address_months_count: 6,
      bank_months_count: -1,
      days_since_request: 0.12,
      velocity_6h: 6200.0,
      velocity_24h: 5800.0,
      velocity_4w: 2900.0,
      zip_count_4w: 3100,
      bank_branch_count_8w: 95,
      date_of_birth_distinct_emails_4w: 14,
      device_os: "Windows",
      device_distinct_emails_8w: 4,
      session_length_in_minutes: 1.4,
      keep_alive_session: 0,
      device_fraud_count: 0,
      month: 7,
      source: "INTERNET",
      risk_score: 0.9125,
      risk_tier_code: "priority",
      status: "escalated",
      notes: "Escalated to Tier 2 - Cross-border request with multiple device emails.",
      assigned_analyst: "Sarah Jenkins (Fraud Ops)",
      shap_waterfall: {
        base_value: 0.011,
        final_score: 0.9125,
        features: [
          { name: "Foreign Request Origin", value: "Cross-Border (1)", contribution: 0.264, impact: "increase" },
          { name: "Device Distinct Emails (8w)", value: "4 emails", contribution: 0.218, impact: "increase" },
          { name: "Velocity Burst (6h)", value: "6,200 apps/hr", contribution: 0.185, impact: "increase" },
          { name: "Name-Email Similarity", value: "0.124", contribution: 0.142, impact: "increase" },
          { name: "No Bank History Record", value: "Missing (-1)", contribution: 0.098, impact: "increase" },
          { name: "Home Phone Valid", value: "Verified (1)", contribution: -0.045, impact: "decrease" },
          { name: "Address Tenure", value: "18 mos combined", contribution: -0.041, impact: "decrease" }
        ]
      },
      radar_comparison: {
        axes: ["Velocity Burst", "Identity Mismatch", "Credit Overreach", "Thin File Signal", "Device Multi-Tenancy", "Session Automation"],
        applicant: [78, 84, 75, 70, 94, 82],
        population_normal: [18, 12, 25, 20, 10, 15],
        population_fraud: [88, 89, 82, 86, 78, 84]
      },
      nemotron_report: {
        status: "online",
        model: "Nemotron-70B-Fraud-Analyst",
        generated_at: "2026-08-19T17:25:00Z",
        investigation_priority: "CRITICAL_ESCALATE",
        sla_window: "< 30 Minutes",
        executive_summary: "Suspected mule recruitment or account takeover attempt originating from foreign network routing with 4 distinct applicant accounts linked to the same device hardware footprint in 8 weeks.",
        key_reasons: [
          "Device Hardware Multiplexing: 4 distinct applicants registered through this device profile within 8 weeks.",
          "Cross-Border Origin: Foreign geolocation header on domestic account opening request.",
          "Mobile Number Invalidation: Primary SMS verification failed to route to a genuine carrier."
        ],
        verification_checklist: [
          { item: "Perform carrier lookup on mobile number to detect VoIP/virtual SIM.", checked: true, critical: true },
          { item: "Demand dual-factor biometric authentication on customer mobile app.", checked: false, critical: true },
          { item: "Inspect shared device fingerprint cluster across all 4 related applicants.", checked: false, critical: true }
        ]
      }
    },

    {
      application_id: "APP-2026-984212",
      applicant_name: "D. Thorne / Overlimit",
      timestamp: "2026-08-19T17:15:30Z",
      customer_age: 20,
      income: 0.1, // Lowest decile
      employment_status: "CF", // Student
      housing_status: "BE", // Living with parents
      name_email_similarity: 0.385,
      email_is_free: 1,
      phone_home_valid: 1,
      phone_mobile_valid: 1,
      foreign_request: 0,
      credit_risk_score: 175,
      proposed_credit_limit: 2000, // Maximum allowable
      intended_balcon_amount: -1,
      payment_type: "AA",
      has_other_cards: 0,
      prev_address_months_count: -1,
      current_address_months_count: 8,
      bank_months_count: -1,
      days_since_request: 0.05,
      velocity_6h: 4200.0,
      velocity_24h: 3800.0,
      velocity_4w: 3200.0,
      zip_count_4w: 1850,
      bank_branch_count_8w: 42,
      date_of_birth_distinct_emails_4w: 4,
      device_os: "macOS",
      device_distinct_emails_8w: 1,
      session_length_in_minutes: 3.2,
      keep_alive_session: 1,
      device_fraud_count: 0,
      month: 7,
      source: "INTERNET",
      risk_score: 0.8740,
      risk_tier_code: "standard", // Standard Review (P95)
      status: "under_review",
      notes: "Student profile requesting $2,000 credit limit at decile 0.1 income.",
      assigned_analyst: "David Zhao (Risk Analyst)",
      shap_waterfall: {
        base_value: 0.011,
        final_score: 0.8740,
        features: [
          { name: "Max Limit to Income Ratio", value: "$2,000 / Decile 0.1", contribution: 0.380, impact: "increase" },
          { name: "Thin File / No Bank History", value: "Missing (-1)", contribution: 0.220, impact: "increase" },
          { name: "Young Age / High Credit Tier", value: "Age 20 (Decade 20)", contribution: 0.160, impact: "increase" },
          { name: "Name-Email Concordance", value: "0.385 (Moderate)", contribution: 0.085, impact: "increase" },
          { name: "Both Phones Valid", value: "Home + Mobile (1)", contribution: -0.095, impact: "decrease" },
          { name: "Single Device Usage", value: "1 Email / 8w", contribution: -0.065, impact: "decrease" }
        ]
      },
      radar_comparison: {
        axes: ["Velocity Burst", "Identity Mismatch", "Credit Overreach", "Thin File Signal", "Device Multi-Tenancy", "Session Automation"],
        applicant: [42, 52, 98, 86, 20, 30],
        population_normal: [18, 12, 25, 20, 10, 15],
        population_fraud: [88, 89, 82, 86, 78, 84]
      },
      nemotron_report: {
        status: "online",
        model: "Nemotron-70B-Fraud-Analyst",
        generated_at: "2026-08-19T17:16:00Z",
        investigation_priority: "MANUAL_REVIEW",
        sla_window: "< 2 Hours",
        executive_summary: "High credit limit mismatch against student applicant profile. While identity signals and device footprint appear natural, the disproportionate initial credit limit ($2,000 against income decile 0.1) warrants income verification or automatic limit down-tiering.",
        key_reasons: [
          "Extreme Debt-to-Income Exposure: $2,000 max limit requested by student with lowest income bracket (0.1).",
          "Zero Prior Credit Footprint: No previous banking tenure or address history on file."
        ],
        verification_checklist: [
          { item: "Request proof of enrolled student status or employment stipend.", checked: false, critical: true },
          { item: "Propose counter-offer with reduced initial credit limit ($300).", checked: false, critical: false },
          { item: "Verify home landline phone registration.", checked: true, critical: false }
        ]
      }
    },

    {
      application_id: "APP-2026-984213",
      applicant_name: "E. Caldwell / Clean",
      timestamp: "2026-08-19T17:02:15Z",
      customer_age: 40,
      income: 0.8,
      employment_status: "CA",
      housing_status: "BA", // Homeowner
      name_email_similarity: 0.892, // Strong match
      email_is_free: 0, // Paid / custom domain
      phone_home_valid: 1,
      phone_mobile_valid: 1,
      foreign_request: 0,
      credit_risk_score: 38, // Healthy low risk
      proposed_credit_limit: 1200,
      intended_balcon_amount: 15,
      payment_type: "AB",
      has_other_cards: 1,
      prev_address_months_count: 72, // 6 years
      current_address_months_count: 36, // 3 years
      bank_months_count: 24, // 2 years
      days_since_request: 0.01,
      velocity_6h: 2100.0,
      velocity_24h: 2200.0,
      velocity_4w: 3100.0,
      zip_count_4w: 1200,
      bank_branch_count_8w: 15,
      date_of_birth_distinct_emails_4w: 1,
      device_os: "Windows",
      device_distinct_emails_8w: 1,
      session_length_in_minutes: 8.5,
      keep_alive_session: 1,
      device_fraud_count: 0,
      month: 7,
      source: "INTERNET",
      risk_score: 0.0412,
      risk_tier_code: "normal",
      status: "marked_legitimate",
      notes: "Auto-approved by Fast-Track KYC policy.",
      assigned_analyst: "System Auto-Rule",
      shap_waterfall: {
        base_value: 0.011,
        final_score: 0.0412,
        features: [
          { name: "Name-Email Similarity", value: "0.892 (High)", contribution: -0.210, impact: "decrease" },
          { name: "Extensive Address & Bank Tenure", value: "9+ yrs history", contribution: -0.185, impact: "decrease" },
          { name: "Low Credit Risk Score", value: "+38 pts", contribution: -0.160, impact: "decrease" },
          { name: "Paid Corporate Email Domain", value: "Paid (0)", contribution: -0.120, impact: "decrease" },
          { name: "Existing Bank Card Holder", value: "Yes (1)", contribution: -0.095, impact: "decrease" },
          { name: "Normal Session Time", value: "8.5 mins", contribution: -0.080, impact: "decrease" },
          { name: "Baseline Velocity Level", value: "2,100 apps/hr", contribution: 0.025, impact: "increase" }
        ]
      },
      radar_comparison: {
        axes: ["Velocity Burst", "Identity Mismatch", "Credit Overreach", "Thin File Signal", "Device Multi-Tenancy", "Session Automation"],
        applicant: [8, 6, 14, 5, 8, 10],
        population_normal: [18, 12, 25, 20, 10, 15],
        population_fraud: [88, 89, 82, 86, 78, 84]
      },
      nemotron_report: {
        status: "online",
        model: "Nemotron-70B-Fraud-Analyst",
        generated_at: "2026-08-19T17:03:00Z",
        investigation_priority: "FAST_TRACK_APPROVE",
        sla_window: "Instant Automated",
        executive_summary: "High-trust genuine applicant profile. Robust credit history, confirmed residential stability (homeowner, 9+ combined years tenure), corporate email domain alignment, and verified multi-channel telephony.",
        key_reasons: [
          "Flawless Identity Consistency: Corporate domain with 0.892 name-email concordance.",
          "Deep Banking History: 24-month previous account with established card relationship.",
          "Natural User Behavior: Realistic 8.5-minute application duration with persistent session."
        ],
        verification_checklist: [
          { item: "Standard automated identity database match.", checked: true, critical: false },
          { item: "Instant credit bureau check.", checked: true, critical: false }
        ]
      }
    },

    {
      application_id: "APP-2026-984214",
      applicant_name: "R. Patel / Rapid Velocity",
      timestamp: "2026-08-19T16:55:00Z",
      customer_age: 30,
      income: 0.5,
      employment_status: "CB",
      housing_status: "BC",
      name_email_similarity: 0.210,
      email_is_free: 1,
      phone_home_valid: 0,
      phone_mobile_valid: 1,
      foreign_request: 0,
      credit_risk_score: 195,
      proposed_credit_limit: 1000,
      intended_balcon_amount: -1,
      payment_type: "AC",
      has_other_cards: 0,
      prev_address_months_count: 4,
      current_address_months_count: 3,
      bank_months_count: 6,
      days_since_request: 0.03,
      velocity_6h: 7800.0,
      velocity_24h: 6900.0,
      velocity_4w: 3100.0,
      zip_count_4w: 3900,
      bank_branch_count_8w: 120,
      date_of_birth_distinct_emails_4w: 19,
      device_os: "Linux",
      device_distinct_emails_8w: 2,
      session_length_in_minutes: 1.1,
      keep_alive_session: 0,
      device_fraud_count: 0,
      month: 7,
      source: "INTERNET",
      risk_score: 0.9410,
      risk_tier_code: "priority",
      status: "confirmed_fraud",
      notes: "Confirmed Mule Farm node during Tier 2 phone interview.",
      assigned_analyst: "Sarah Jenkins (Fraud Ops)",
      shap_waterfall: {
        base_value: 0.011,
        final_score: 0.9410,
        features: [
          { name: "Velocity 6h Surge", value: "7,800 apps/hr", contribution: 0.290, impact: "increase" },
          { name: "DOB Distinct Emails Cluster", value: "19 emails", contribution: 0.230, impact: "increase" },
          { name: "Name-Email Discordance", value: "0.210", contribution: 0.175, impact: "increase" },
          { name: "Linux OS on Retail Site", value: "Linux", contribution: 0.120, impact: "increase" },
          { name: "Short Address Tenure", value: "7 mos combined", contribution: 0.085, impact: "increase" },
          { name: "Mobile Phone Valid", value: "Verified (1)", contribution: -0.040, impact: "decrease" }
        ]
      },
      radar_comparison: {
        axes: ["Velocity Burst", "Identity Mismatch", "Credit Overreach", "Thin File Signal", "Device Multi-Tenancy", "Session Automation"],
        applicant: [92, 79, 65, 78, 68, 88],
        population_normal: [18, 12, 25, 20, 10, 15],
        population_fraud: [88, 89, 82, 86, 78, 84]
      },
      nemotron_report: {
        status: "online",
        model: "Nemotron-70B-Fraud-Analyst",
        generated_at: "2026-08-19T16:56:00Z",
        investigation_priority: "CRITICAL_ESCALATE",
        sla_window: "< 15 Minutes",
        executive_summary: "Mule Account Ring node. Elevated regional velocity across ZIP code combined with multiple distinct email registrations tied to one birth date in 4 weeks.",
        key_reasons: [
          "Coordinated Application Spike: 7,800 apps/hr in 6h window.",
          "Identity Collusion: 19 applications using the same DOB."
        ],
        verification_checklist: [
          { item: "Blacklist applicant device signature and IP range.", checked: true, critical: true },
          { item: "Lodge SAR (Suspicious Activity Report) with compliance.", checked: true, critical: true }
        ]
      }
    }
  ];

  // 3. Model Lab Leaderboard & Metrics
  const modelLeaderboard = [
    {
      name: "CatBoost + XGBoost Blend v2.4",
      type: "Ensemble Blend",
      is_champion: true,
      pr_auc: 0.1677,
      roc_auc: 0.8979,
      tpr_at_5pct_fpr: 0.5503, // 55.03%
      precision: 0.248,
      recall: 0.764,
      f1: 0.3745,
      latency_ms: 0.042,
      model_size_mb: 24.5,
      description: "Rank-average weighted blend with out-of-time temporal CV"
    },
    {
      name: "CatBoost Native Classifier",
      type: "Gradient Boosted Trees",
      is_champion: false,
      pr_auc: 0.1666,
      roc_auc: 0.8968,
      tpr_at_5pct_fpr: 0.5476,
      precision: 0.241,
      recall: 0.758,
      f1: 0.3658,
      latency_ms: 0.028,
      model_size_mb: 18.2,
      description: "Symmetric tree architecture with built-in categorical splits"
    },
    {
      name: "XGBoost Tuned (Undersampled 10:1)",
      type: "Gradient Boosted Trees",
      is_champion: false,
      pr_auc: 0.1654,
      roc_auc: 0.8955,
      tpr_at_5pct_fpr: 0.5442,
      precision: 0.236,
      recall: 0.752,
      f1: 0.3592,
      latency_ms: 0.021,
      model_size_mb: 12.6,
      description: "Exact greedy tree boosting trained on controlled class-balance"
    },
    {
      name: "LightGBM High-Velocity",
      type: "Gradient Boosted Trees",
      is_champion: false,
      pr_auc: 0.1631,
      roc_auc: 0.8912,
      tpr_at_5pct_fpr: 0.5334,
      precision: 0.228,
      recall: 0.741,
      f1: 0.3486,
      latency_ms: 0.015,
      model_size_mb: 9.8,
      description: "Leaf-wise tree growth optimized for sub-millisecond scoring"
    },
    {
      name: "Supervised MLP Deep Neural Net",
      type: "Deep Learning",
      is_champion: false,
      pr_auc: 0.1482,
      roc_auc: 0.8710,
      tpr_at_5pct_fpr: 0.4890,
      precision: 0.198,
      recall: 0.695,
      f1: 0.3082,
      latency_ms: 0.085,
      model_size_mb: 45.0,
      description: "4-Layer Dense residual network with LayerNorm and Dropout"
    },
    {
      name: "Isolation Forest (Unsupervised Baseline)",
      type: "Anomaly Detection",
      is_champion: false,
      pr_auc: 0.0425,
      roc_auc: 0.6840,
      tpr_at_5pct_fpr: 0.1820,
      precision: 0.052,
      recall: 0.320,
      f1: 0.0895,
      latency_ms: 0.019,
      model_size_mb: 8.4,
      description: "Unsupervised tree isolation baseline (no fraud labels)"
    }
  ];

  // 4. ROC and PR Curve Data Points (Synthesized from results.json)
  const rocCurvePoints = [
    { fpr: 0.000, tpr: 0.000, threshold: 1.00 },
    { fpr: 0.001, tpr: 0.185, threshold: 0.85 },
    { fpr: 0.005, tpr: 0.342, threshold: 0.72 },
    { fpr: 0.010, tpr: 0.428, threshold: 0.60 },
    { fpr: 0.020, tpr: 0.486, threshold: 0.45 },
    { fpr: 0.030, tpr: 0.514, threshold: 0.35 },
    { fpr: 0.050, tpr: 0.5503, threshold: 0.18 }, // Official benchmark point (TPR@5%FPR)
    { fpr: 0.080, tpr: 0.612, threshold: 0.12 },
    { fpr: 0.120, tpr: 0.685, threshold: 0.08 },
    { fpr: 0.200, tpr: 0.782, threshold: 0.04 },
    { fpr: 0.350, tpr: 0.884, threshold: 0.02 },
    { fpr: 0.600, tpr: 0.952, threshold: 0.01 },
    { fpr: 1.000, tpr: 1.000, threshold: 0.00 }
  ];

  const prCurvePoints = [
    { recall: 0.00, precision: 0.820, threshold: 0.95 },
    { recall: 0.10, precision: 0.680, threshold: 0.82 },
    { recall: 0.20, precision: 0.540, threshold: 0.70 },
    { recall: 0.35, precision: 0.410, threshold: 0.55 },
    { recall: 0.50, precision: 0.295, threshold: 0.38 },
    { recall: 0.5503, precision: 0.258, threshold: 0.18 },
    { recall: 0.65, precision: 0.195, threshold: 0.11 },
    { recall: 0.764, precision: 0.145, threshold: 0.06 },
    { recall: 0.88, precision: 0.082, threshold: 0.02 },
    { recall: 1.00, precision: 0.011, threshold: 0.00 }
  ];

  // 5. Calibration Reliability Diagram Data
  const calibrationData = {
    brier_score: 0.00892,
    ece: 0.0064, // Expected Calibration Error (0.64%)
    bins: [
      { mean_pred: 0.05, obs_fraud: 0.048, count: 852000 },
      { mean_pred: 0.15, obs_fraud: 0.142, count: 68400 },
      { mean_pred: 0.25, obs_fraud: 0.256, count: 32100 },
      { mean_pred: 0.35, obs_fraud: 0.362, count: 18400 },
      { mean_pred: 0.45, obs_fraud: 0.448, count: 11200 },
      { mean_pred: 0.55, obs_fraud: 0.561, count: 7800 },
      { mean_pred: 0.65, obs_fraud: 0.639, count: 4900 },
      { mean_pred: 0.75, obs_fraud: 0.768, count: 3100 },
      { mean_pred: 0.85, obs_fraud: 0.842, count: 1400 },
      { mean_pred: 0.95, obs_fraud: 0.938, count: 700 }
    ]
  };

  return {
    monitorKPIs,
    applicants,
    modelLeaderboard,
    rocCurvePoints,
    prCurvePoints,
    calibrationData
  };
})();
