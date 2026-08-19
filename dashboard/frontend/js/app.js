/* =========================================================================
   SENTINEL Bank Fraud Classification Platform - Core Controller & SPA Engine
   ========================================================================= */

const App = (() => {
  let currentPage = "monitor";
  let liveFeedActive = true;
  let liveFeedTimer = null;
  let activeApplicantId = "APP-2026-984210";
  let activeThreshold = 0.50;
  let cachedKpis = null;
  let cachedLab = null;
  let selectedQueueIds = new Set();
  let currentModalAppId = null;

  // -----------------------------------------------------------------------
  // 1. Theme Management & Icons
  // -----------------------------------------------------------------------
  function initTheme() {
    const savedTheme = localStorage.getItem("sentinel-theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    const toggleBtn = document.getElementById("theme-toggle-btn");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") || "dark";
        const next = current === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("sentinel-theme", next);
        updateThemeIcon(next);
        renderCurrentPageCharts();
      });
    }
  }

  function updateThemeIcon(theme) {
    const iconEl = document.getElementById("theme-icon");
    if (iconEl) iconEl.innerHTML = theme === "dark" ? Icons.sun : Icons.moon;
  }

  function injectIcons() {
    document.querySelectorAll("[data-icon]").forEach((el) => {
      const name = el.getAttribute("data-icon");
      if (Icons[name]) el.innerHTML = Icons[name];
    });
  }

  function showToast(msg) {
    const stack = document.getElementById("toast-stack");
    if (!stack) return;
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = msg;
    stack.appendChild(t);
    setTimeout(() => {
      t.style.transition = "opacity 200ms ease, transform 200ms ease";
      t.style.opacity = "0";
      t.style.transform = "translateX(50px)";
      setTimeout(() => t.remove(), 220);
    }, 2800);
  }

  // -----------------------------------------------------------------------
  // 2. Navigation Routing
  // -----------------------------------------------------------------------
  function navigateTo(pageId) {
    currentPage = pageId;

    // Update Sidebar Active state
    document.querySelectorAll(".nav-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-page") === pageId);
    });

    // Update View Panels
    document.querySelectorAll(".view-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `view-${pageId}`);
    });

    // Update Topbar Title
    const titles = {
      monitor: { title: "Monitor Dashboard", sub: "Portfolio-wide bank account opening application risk & real-time telemetry" },
      queue: { title: "Investigation Queue", sub: "Triage high-risk applications, assign analysts & record dispositions" },
      inspector: { title: "Application Deep-Dive Inspector", sub: "SHAP waterfall attribution, behavioral deviation & Nemotron AI analysis" },
      lab: { title: "Model Lab & Decision Tuner", sub: "Out-of-time model benchmark leaderboard, interactive threshold tuning & ROI modeling" },
      "batch-sim": { title: "1M Batch Inference & Scenario Sandbox", sub: "High-throughput streaming scoring engine & what-if application simulator" }
    };

    if (titles[pageId]) {
      document.getElementById("page-title").textContent = titles[pageId].title;
      document.getElementById("page-subtitle").textContent = titles[pageId].sub;
    }

    renderCurrentPage();
  }

  function renderCurrentPage() {
    if (currentPage === "monitor") loadMonitorDashboard();
    if (currentPage === "queue") loadInvestigationQueue();
    if (currentPage === "inspector") loadInspector(activeApplicantId);
    if (currentPage === "lab") loadModelLab();
    if (currentPage === "batch-sim") loadBatchSimulator();
  }

  function renderCurrentPageCharts() {
    if (currentPage === "monitor" && cachedKpis) {
      Charts.renderScoreDistribution(document.getElementById("chart-score-dist"), cachedKpis.score_distribution, activeThreshold);
      Charts.renderMonthlyTrend(document.getElementById("chart-monthly-trend"), cachedKpis.monthly_trend);
      Charts.renderTopIndicators(document.getElementById("chart-top-indicators"), cachedKpis.top_risk_indicators);
    } else if (currentPage === "inspector") {
      loadInspector(activeApplicantId);
    } else if (currentPage === "lab" && cachedLab) {
      Charts.renderRocCurve(document.getElementById("chart-roc-curve"), cachedLab.roc_curve, activeThreshold);
      Charts.renderPrCurve(document.getElementById("chart-pr-curve"), cachedLab.pr_curve, activeThreshold);
      Charts.renderCalibrationCurve(document.getElementById("chart-calibration-curve"), cachedLab.calibration);
    }
  }

  async function updateQueueBadge() {
    try {
      const res = await Api.applications({ status: "pending" });
      const badge = document.getElementById("queue-badge-count");
      if (badge) {
        badge.textContent = res.total !== undefined ? res.total : res.items.length;
      }
    } catch (e) {
      console.warn("Could not update queue badge:", e);
    }
  }

  // -----------------------------------------------------------------------
  // 3. Monitor Dashboard Controller
  // -----------------------------------------------------------------------
  async function loadMonitorDashboard() {
    try {
      cachedKpis = await Api.kpis();
      Fmt.countUp(document.getElementById("kpi-total-apps"), cachedKpis.total_applications, { formatter: Fmt.int });
      Fmt.countUp(document.getElementById("kpi-fraud-count"), cachedKpis.predicted_fraud_count, { formatter: Fmt.int });
      Fmt.countUp(document.getElementById("kpi-fraud-rate"), cachedKpis.fraud_rate * 100, { formatter: (v) => `${v.toFixed(2)}%`, decimals: 2 });
      Fmt.countUp(document.getElementById("kpi-pr-auc"), cachedKpis.pr_auc, { formatter: Fmt.float, decimals: 4 });
      Fmt.countUp(document.getElementById("kpi-tpr-benchmark"), cachedKpis.tpr_at_5pct_fpr * 100, { formatter: (v) => `${v.toFixed(2)}%`, decimals: 2 });

      // Update sidebar champion pill
      const champTpr = document.getElementById("sidebar-champion-tpr");
      if (champTpr && cachedKpis.tpr_at_5pct_fpr) {
        champTpr.textContent = `TPR@5%: ${(cachedKpis.tpr_at_5pct_fpr * 100).toFixed(1)}%`;
      }
      const champName = document.getElementById("sidebar-champion-name");
      if (champName && cachedKpis.active_model_name) {
        champName.textContent = cachedKpis.active_model_name;
      }

      await updateQueueBadge();

      Charts.renderScoreDistribution(document.getElementById("chart-score-dist"), cachedKpis.score_distribution, activeThreshold);
      Charts.renderMonthlyTrend(document.getElementById("chart-monthly-trend"), cachedKpis.monthly_trend);
      Charts.renderTopIndicators(document.getElementById("chart-top-indicators"), cachedKpis.top_risk_indicators);

      // Load Priority Triage List
      const apps = await Api.applications({ risk_tier: "priority" });
      const tbody = document.querySelector("#table-priority-triage tbody");
      if (tbody) {
        tbody.innerHTML = apps.items.slice(0, 5).map((app) => `
          <tr data-app-id="${app.application_id}">
            <td class="font-mono font-semibold" style="color:var(--accent-light);">${app.application_id}</td>
            <td>${Fmt.escapeHtml(app.applicant_name)}</td>
            <td>Age ${app.customer_age}s</td>
            <td class="font-mono font-semibold">${Fmt.money(app.proposed_credit_limit)}</td>
            <td><span class="truncate" style="max-width:140px; display:inline-block; font-size:11.5px; color:var(--crimson);">${app.notes}</span></td>
            <td class="num font-mono font-bold" style="color:var(--crimson);">${Fmt.score(app.risk_score)}</td>
            <td><span class="badge badge-critical">Priority</span></td>
            <td>
              <button class="btn btn-primary btn-sm btn-quick-inspect" data-id="${app.application_id}">Inspect</button>
            </td>
          </tr>
        `).join("");

        tbody.querySelectorAll(".btn-quick-inspect").forEach((btn) => {
          btn.addEventListener("click", (e) => {
            e.stopPropagation();
            activeApplicantId = btn.getAttribute("data-id");
            navigateTo("inspector");
          });
        });

        tbody.querySelectorAll("tr").forEach((row) => {
          row.addEventListener("click", () => {
            activeApplicantId = row.getAttribute("data-app-id");
            navigateTo("inspector");
          });
        });
      }

      startLiveFeedEngine();
    } catch (e) {
      showToast(`Error loading Monitor Dashboard: ${e.message}`);
    }
  }

  // Live Application Feed Simulation
  function startLiveFeedEngine() {
    if (liveFeedTimer) clearInterval(liveFeedTimer);
    const container = document.getElementById("live-feed-container");
    if (!container) return;

    // Seed initial feed cards
    if (container.children.length === 0) {
      MockData.applicants.forEach((app) => addFeedCard(app, false));
    }

    liveFeedTimer = setInterval(() => {
      if (!liveFeedActive) return;
      generateRandomFeedApplication();
    }, 2800);
  }

  function generateRandomFeedApplication() {
    const isAnomalous = Math.random() < 0.22;
    const score = isAnomalous ? 0.85 + Math.random() * 0.13 : 0.02 + Math.random() * 0.25;
    const tier = score >= 0.88 ? "priority" : score >= 0.65 ? "standard" : "normal";
    const app = {
      application_id: `APP-2026-${Math.floor(100000 + Math.random() * 900000)}`,
      applicant_name: isAnomalous ? "Synthetic Identity Ring" : "Verified Retail Customer",
      customer_age: [20, 30, 40, 50, 60][Math.floor(Math.random() * 5)],
      proposed_credit_limit: isAnomalous ? 1800 : 800,
      risk_score: score,
      risk_tier_code: tier,
      notes: isAnomalous ? "Velocity spike & name discordance" : "Standard verification passed"
    };
    addFeedCard(app, true);
  }

  function addFeedCard(app, prepend = true) {
    const container = document.getElementById("live-feed-container");
    if (!container) return;

    const card = document.createElement("div");
    card.className = `feed-item-card is-${app.risk_tier_code}`;
    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px;">
        <span class="font-mono font-bold" style="color:var(--accent-light);">${app.application_id}</span>
        <span class="font-mono font-bold text-${app.risk_tier_code === "priority" ? "crimson" : app.risk_tier_code === "standard" ? "amber" : "emerald"}">${Fmt.score(app.risk_score)}</span>
      </div>
      <div style="font-size:12px; font-weight:600;" class="truncate">${Fmt.escapeHtml(app.applicant_name)}</div>
      <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Limit: ${Fmt.money(app.proposed_credit_limit)} &bull; Age ${app.customer_age}s</div>
    `;

    card.addEventListener("click", () => {
      activeApplicantId = app.application_id;
      navigateTo("inspector");
    });

    if (prepend) {
      container.insertBefore(card, container.firstChild);
      if (container.children.length > 8) container.removeChild(container.lastChild);
    } else {
      container.appendChild(card);
    }
  }

  // -----------------------------------------------------------------------
  // 4. Investigation Queue Controller
  // -----------------------------------------------------------------------
  async function loadInvestigationQueue() {
    const searchVal = document.getElementById("queue-search-input")?.value || "";
    const tierVal = document.getElementById("queue-filter-tier")?.value || "";
    const ageVal = document.getElementById("queue-filter-age")?.value || "";
    const empVal = document.getElementById("queue-filter-employment")?.value || "";
    const statusVal = document.getElementById("queue-filter-status")?.value || "";

    const res = await Api.applications({
      q: searchVal,
      risk_tier: tierVal,
      customer_age: ageVal,
      employment_status: empVal,
      status: statusVal
    });

    const tbody = document.querySelector("#table-investigation-queue tbody");
    if (!tbody) return;

    if (res.items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; padding:32px; color:var(--text-muted);">No applications match the active filter criteria.</td></tr>`;
      return;
    }

    tbody.innerHTML = res.items.map((app) => {
      const isSelected = selectedQueueIds.has(app.application_id);
      let statusBadgeClass = "badge-neutral";
      let statusText = "Pending";
      if (app.status === "under_review") { statusBadgeClass = "badge-indigo"; statusText = "Under Review"; }
      if (app.status === "escalated") { statusBadgeClass = "badge-warning"; statusText = "Escalated"; }
      if (app.status === "marked_legitimate") { statusBadgeClass = "badge-good"; statusText = "Legitimate"; }
      if (app.status === "confirmed_fraud") { statusBadgeClass = "badge-critical"; statusText = "Confirmed Fraud"; }

      const tierBadge = app.risk_tier_code === "priority" ? `<span class="badge badge-critical">Priority</span>` :
        app.risk_tier_code === "standard" ? `<span class="badge badge-warning">Standard</span>` :
        `<span class="badge badge-good">Normal</span>`;

      return `
        <tr data-id="${app.application_id}">
          <td><input type="checkbox" class="queue-row-cb" data-id="${app.application_id}" ${isSelected ? "checked" : ""} /></td>
          <td class="font-mono font-semibold" style="color:var(--accent-light);">${app.application_id}</td>
          <td>
            <div style="font-weight:600;">${Fmt.escapeHtml(app.applicant_name)}</div>
            <div style="font-size:11px; color:var(--text-muted);">${Fmt.dateTime(app.timestamp)}</div>
          </td>
          <td>Age ${app.customer_age}s &bull; ${app.employment_status}</td>
          <td class="font-mono">${Fmt.money(app.proposed_credit_limit)}</td>
          <td>Decile ${app.income}</td>
          <td><span style="font-size:11.5px; color:var(--crimson);">${app.notes || "—"}</span></td>
          <td class="num font-mono font-bold" style="color:${app.risk_score >= 0.88 ? "var(--crimson)" : app.risk_score >= 0.65 ? "var(--amber)" : "var(--emerald)"};">${Fmt.score(app.risk_score)}</td>
          <td>${tierBadge}</td>
          <td><span class="badge ${statusBadgeClass}">${statusText}</span></td>
          <td>
            <div style="display:flex; gap:6px;">
              <button class="btn btn-secondary btn-sm btn-inspect-queue" data-id="${app.application_id}">Inspect</button>
              <button class="btn btn-primary btn-sm btn-action-queue" data-id="${app.application_id}">Review</button>
            </div>
          </td>
        </tr>
      `;
    }).join("");

    // Wire Select All Checkbox
    const selectAllCb = document.getElementById("queue-select-all");
    if (selectAllCb) {
      selectAllCb.checked = res.items.length > 0 && res.items.every((a) => selectedQueueIds.has(a.application_id));
      selectAllCb.onchange = () => {
        if (selectAllCb.checked) {
          res.items.forEach((a) => selectedQueueIds.add(a.application_id));
        } else {
          res.items.forEach((a) => selectedQueueIds.delete(a.application_id));
        }
        tbody.querySelectorAll(".queue-row-cb").forEach((cb) => {
          cb.checked = selectAllCb.checked;
        });
        updateBatchActionBar();
      };
    }

    // Wire Row Checkboxes
    tbody.querySelectorAll(".queue-row-cb").forEach((cb) => {
      cb.addEventListener("change", (e) => {
        const id = cb.getAttribute("data-id");
        if (cb.checked) selectedQueueIds.add(id);
        else selectedQueueIds.delete(id);
        if (selectAllCb) {
          selectAllCb.checked = res.items.length > 0 && res.items.every((a) => selectedQueueIds.has(a.application_id));
        }
        updateBatchActionBar();
      });
    });

    // Wire Inspect & Action Buttons
    tbody.querySelectorAll(".btn-inspect-queue").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        activeApplicantId = btn.getAttribute("data-id");
        navigateTo("inspector");
      });
    });

    tbody.querySelectorAll(".btn-action-queue").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        openCaseModal(btn.getAttribute("data-id"));
      });
    });

    tbody.querySelectorAll("tr").forEach((row) => {
      row.addEventListener("click", (e) => {
        if (e.target.tagName.toLowerCase() === "input" || e.target.tagName.toLowerCase() === "button") return;
        activeApplicantId = row.getAttribute("data-id");
        navigateTo("inspector");
      });
    });

    await updateQueueBadge();
  }

  function updateBatchActionBar() {
    const bar = document.getElementById("batch-action-bar");
    const countEl = document.getElementById("batch-selected-count");
    if (!bar || !countEl) return;

    if (selectedQueueIds.size > 0) {
      bar.style.display = "flex";
      countEl.textContent = `${selectedQueueIds.size} Application${selectedQueueIds.size > 1 ? "s" : ""} Selected`;
    } else {
      bar.style.display = "none";
    }
  }

  async function handleBatchAction(actionStatus, label) {
    if (selectedQueueIds.size === 0) return;
    const count = selectedQueueIds.size;
    for (const appId of selectedQueueIds) {
      await Api.queueAction(appId, actionStatus, `Bulk action: ${label}`, "Investigator Lead");
    }
    showToast(`Bulk updated ${count} application${count > 1 ? "s" : ""} to ${label}`);
    selectedQueueIds.clear();
    updateBatchActionBar();
    await loadInvestigationQueue();
    await updateQueueBadge();
  }

  // Case Disposition Modal
  function openCaseModal(appId) {
    currentModalAppId = appId;
    const modal = document.getElementById("case-modal");
    document.getElementById("modal-app-title").textContent = `Case Review: ${appId}`;
    if (modal) modal.classList.add("active");
  }

  function closeCaseModal() {
    const modal = document.getElementById("case-modal");
    if (modal) modal.classList.remove("active");
    currentModalAppId = null;
  }

  async function saveCaseDisposition() {
    if (!currentModalAppId) return;
    const action = document.getElementById("modal-action-select").value;
    const notes = document.getElementById("modal-notes-input").value;

    await Api.queueAction(currentModalAppId, action, notes, "Senior Analyst");
    showToast(`Case ${currentModalAppId} updated to ${action.replace("_", " ")}`);
    closeCaseModal();
    if (currentPage === "queue") loadInvestigationQueue();
    if (currentPage === "inspector") loadInspector(currentModalAppId);
  }

  // -----------------------------------------------------------------------
  // 5. Application Deep-Dive Inspector Controller
  // -----------------------------------------------------------------------
  async function loadInspector(appId) {
    try {
      const app = await Api.application(appId);
      if (!app) return;
      activeApplicantId = app.application_id;

      // Populate Quick Switcher Dropdown
      const selector = document.getElementById("insp-app-selector");
      if (selector && selector.children.length === 0) {
        selector.innerHTML = MockData.applicants.map((a) => `
          <option value="${a.application_id}" ${a.application_id === app.application_id ? "selected" : ""}>
            ${a.application_id} - ${a.applicant_name} (${(a.risk_score * 100).toFixed(1)}%)
          </option>
        `).join("");
        selector.addEventListener("change", () => loadInspector(selector.value));
      } else if (selector) {
        selector.value = app.application_id;
      }

      // Header Meta
      document.getElementById("insp-app-id").textContent = app.application_id;
      document.getElementById("insp-applicant-name").textContent = `${app.applicant_name} • Submitted ${Fmt.dateTime(app.timestamp)}`;
      document.getElementById("insp-risk-score").textContent = Fmt.score(app.risk_score);
      document.getElementById("insp-risk-score").style.color = app.risk_score >= 0.88 ? "var(--crimson)" : app.risk_score >= 0.65 ? "var(--amber)" : "var(--emerald)";

      const tierBadgeEl = document.getElementById("insp-tier-badge");
      if (tierBadgeEl) {
        tierBadgeEl.innerHTML = app.risk_tier_code === "priority" ?
          `<span class="badge badge-critical"><span data-icon="critical"></span>Priority Review (P99)</span>` :
          app.risk_tier_code === "standard" ?
          `<span class="badge badge-warning"><span data-icon="warning"></span>Standard Review (P95)</span>` :
          `<span class="badge badge-good"><span data-icon="good"></span>Normal Approval</span>`;
      }

      // Dossier Fields
      document.getElementById("dossier-age").textContent = `${app.customer_age}s`;
      document.getElementById("dossier-income").textContent = `Decile ${app.income}`;
      document.getElementById("dossier-employment").textContent = Fmt.formatEmployment(app.employment_status);
      document.getElementById("dossier-housing").textContent = Fmt.formatHousing(app.housing_status);

      document.getElementById("dossier-email-match").textContent = `${app.name_email_similarity.toFixed(3)} ${app.name_email_similarity < 0.15 ? "(Discordant)" : "(Concordant)"}`;
      document.getElementById("dossier-email-match").style.color = app.name_email_similarity < 0.15 ? "var(--crimson)" : "var(--emerald)";
      document.getElementById("dossier-email-type").textContent = app.email_is_free ? "Free Webmail (1)" : "Paid Domain (0)";
      document.getElementById("dossier-phones").textContent = `Mobile: ${app.phone_mobile_valid ? "Valid" : "Invalid"}, Home: ${app.phone_home_valid ? "Valid" : "Invalid"}`;
      document.getElementById("dossier-foreign").textContent = app.foreign_request ? "Foreign (1)" : "Domestic (0)";

      document.getElementById("dossier-velocity").textContent = `${Fmt.int(app.velocity_6h)} vs ${Fmt.int(app.velocity_4w)} apps/hr`;
      document.getElementById("dossier-dob-cluster").textContent = `${app.date_of_birth_distinct_emails_4w} Emails in 4w`;
      document.getElementById("dossier-dob-cluster").style.color = app.date_of_birth_distinct_emails_4w > 10 ? "var(--crimson)" : "var(--text-primary)";
      document.getElementById("dossier-address-tenure").innerHTML = Fmt.formatMissingMonths(app.prev_address_months_count);
      document.getElementById("dossier-bank-tenure").innerHTML = Fmt.formatMissingMonths(app.bank_months_count);

      document.getElementById("dossier-credit-score").textContent = `${app.credit_risk_score >= 0 ? "+" : ""}${app.credit_risk_score} pts`;
      document.getElementById("dossier-credit-limit").textContent = Fmt.money(app.proposed_credit_limit);
      document.getElementById("dossier-device-os").textContent = app.device_os;
      document.getElementById("dossier-session").textContent = `${app.session_length_in_minutes} min`;

      // Render Visualizations
      Charts.renderShapWaterfall(document.getElementById("chart-shap-waterfall"), app.shap_waterfall);
      Charts.renderBehavioralRadar(document.getElementById("chart-behavioral-radar"), app.radar_comparison);

      // Nemotron AI Report
      const report = app.nemotron_report;
      document.getElementById("nemotron-summary-text").textContent = report.executive_summary;

      const reasonsList = document.getElementById("nemotron-reasons-list");
      if (reasonsList) {
        reasonsList.innerHTML = report.key_reasons.map((r) => `<li>${Fmt.escapeHtml(r)}</li>`).join("");
      }

      const checklistWrap = document.getElementById("nemotron-checklist-container");
      if (checklistWrap) {
        checklistWrap.innerHTML = report.verification_checklist.map((item, idx) => `
          <label class="checklist-item">
            <input type="checkbox" ${item.checked ? "checked" : ""} />
            <span>${Fmt.escapeHtml(item.item)} ${item.critical ? '<strong style="color:var(--crimson); font-size:10px;">(MANDATORY)</strong>' : ""}</span>
          </label>
        `).join("");
      }

      injectIcons();
    } catch (e) {
      showToast(`Error loading application inspector: ${e.message}`);
    }
  }

  // -----------------------------------------------------------------------
  // 6. Model Lab Controller & Threshold Tuner
  // -----------------------------------------------------------------------
  async function loadModelLab() {
    try {
      cachedLab = await Api.modelLab();

      // Render Leaderboard
      const tbody = document.querySelector("#table-model-leaderboard tbody");
      if (tbody) {
        tbody.innerHTML = cachedLab.leaderboard.map((m) => `
          <tr class="${m.is_champion ? "font-bold" : ""}">
            <td>
              <div style="display:flex; align-items:center; gap:8px;">
                ${m.is_champion ? '<span class="badge badge-good">CHAMPION</span>' : ""}
                <span>${Fmt.escapeHtml(m.name)}</span>
              </div>
            </td>
            <td style="color:var(--text-secondary); font-size:11.5px;">${m.type}</td>
            <td class="num font-mono font-bold" style="color:var(--emerald);">${m.pr_auc.toFixed(4)}</td>
            <td class="num font-mono">${m.roc_auc.toFixed(4)}</td>
            <td class="num font-mono font-bold" style="color:var(--cyan);">${(m.tpr_at_5pct_fpr * 100).toFixed(2)}%</td>
            <td class="num font-mono">${(m.precision * 100).toFixed(1)}%</td>
            <td class="num font-mono">${(m.recall * 100).toFixed(1)}%</td>
            <td class="num font-mono">${m.f1.toFixed(4)}</td>
            <td class="num font-mono text-muted">${m.latency_ms.toFixed(3)} ms</td>
            <td><span class="badge ${m.is_champion ? "badge-good" : "badge-neutral"}">${m.is_champion ? "Deployed (Active)" : "Benchmarked"}</span></td>
          </tr>
        `).join("");
      }

      // Render Curves
      Charts.renderRocCurve(document.getElementById("chart-roc-curve"), cachedLab.roc_curve, activeThreshold);
      Charts.renderPrCurve(document.getElementById("chart-pr-curve"), cachedLab.pr_curve, activeThreshold);
      Charts.renderCalibrationCurve(document.getElementById("chart-calibration-curve"), cachedLab.calibration);

      updateThresholdCalculations(activeThreshold);
    } catch (e) {
      showToast(`Error loading model lab: ${e.message}`);
    }
  }

  function updateThresholdCalculations(thresh) {
    activeThreshold = thresh;
    document.getElementById("label-active-threshold").textContent = `T = ${thresh.toFixed(2)}`;
    document.getElementById("input-threshold-slider").value = thresh;

    // Reactive Confusion Matrix Simulation for 300,000 Out-of-Time Test Applications
    const totalN = 300000;
    const totalFrauds = Math.round(totalN * 0.01103); // ~3,309 frauds
    const totalLegit = totalN - totalFrauds;

    // Recall & Precision curves interpolation
    const recall = Math.max(0.10, Math.min(0.98, 0.95 - Math.pow(thresh, 0.85) * 0.88));
    const fpr = Math.max(0.001, Math.min(0.80, Math.pow(1.0 - thresh, 3.2)));

    const tp = Math.round(totalFrauds * recall);
    const fn = totalFrauds - tp;
    const fp = Math.round(totalLegit * fpr);
    const tn = totalLegit - fp;

    document.getElementById("cm-tp-val").textContent = Fmt.int(tp);
    document.getElementById("cm-fp-val").textContent = Fmt.int(fp);
    document.getElementById("cm-fn-val").textContent = Fmt.int(fn);
    document.getElementById("cm-tn-val").textContent = Fmt.int(tn);

    // Business ROI Modeling
    const costFN = parseFloat(document.getElementById("cost-fn-input")?.value || "2500");
    const costFP = parseFloat(document.getElementById("cost-fp-input")?.value || "35");

    const lossesPrevented = tp * costFN;
    const investigationCost = fp * costFP;
    const netValue = lossesPrevented - investigationCost;

    document.getElementById("roi-losses-prevented").textContent = Fmt.money(lossesPrevented);
    document.getElementById("roi-investigation-cost").textContent = Fmt.money(investigationCost);
    document.getElementById("roi-net-value").textContent = `+${Fmt.money(netValue)}`;

    if (cachedLab) {
      Charts.renderRocCurve(document.getElementById("chart-roc-curve"), cachedLab.roc_curve, activeThreshold);
      Charts.renderPrCurve(document.getElementById("chart-pr-curve"), cachedLab.pr_curve, activeThreshold);
    }
  }

  // -----------------------------------------------------------------------
  // 7. 1M Batch Inference & Scenario Sandbox Controller
  // -----------------------------------------------------------------------
  function loadBatchSimulator() {
    initScenarioListeners();
  }

  function initScenarioListeners() {
    const inputs = [
      "sim-input-name-email", "sim-input-velocity", "sim-input-limit",
      "sim-input-income", "sim-input-os", "sim-input-dob-emails"
    ];

    inputs.forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("input", runLiveScenarioSimulation);
    });

    runLiveScenarioSimulation();
  }

  async function runLiveScenarioSimulation() {
    const nameEmail = parseFloat(document.getElementById("sim-input-name-email")?.value || "0.05");
    const velocity = parseFloat(document.getElementById("sim-input-velocity")?.value || "8200");
    const limit = parseFloat(document.getElementById("sim-input-limit")?.value || "1800");
    const income = parseFloat(document.getElementById("sim-input-income")?.value || "0.2");
    const os = document.getElementById("sim-input-os")?.value || "Linux";
    const dobEmails = parseFloat(document.getElementById("sim-input-dob-emails")?.value || "22");

    // Update Slider Value Labels
    document.getElementById("sim-val-name-email").textContent = nameEmail.toFixed(2);
    document.getElementById("sim-val-velocity").textContent = Fmt.int(velocity);
    document.getElementById("sim-val-limit").textContent = Fmt.money(limit);
    document.getElementById("sim-val-income").textContent = `Decile ${income.toFixed(1)}`;

    const res = await Api.simulateScenario({
      name_email_similarity: nameEmail,
      velocity_6h: velocity,
      velocity_4w: 3100,
      proposed_credit_limit: limit,
      income,
      device_os: os,
      date_of_birth_distinct_emails_4w: dobEmails
    });

    document.getElementById("sim-output-score").textContent = Fmt.score(res.risk_score);
    document.getElementById("sim-output-score").style.color = res.risk_score >= 0.88 ? "var(--crimson)" : res.risk_score >= 0.65 ? "var(--amber)" : "var(--emerald)";

    const verdictEl = document.getElementById("sim-output-verdict");
    if (verdictEl) {
      verdictEl.innerHTML = res.risk_tier_code === "priority" ?
        `<span class="badge badge-critical">Priority Review (P99)</span>` :
        res.risk_tier_code === "standard" ?
        `<span class="badge badge-warning">Standard Review (P95)</span>` :
        `<span class="badge badge-good">Fast-Track Auto Approve</span>`;
    }

    const deltasContainer = document.getElementById("sim-deltas-container");
    if (deltasContainer && res.contributions) {
      deltasContainer.innerHTML = res.contributions.map((c) => `
        <span class="badge ${c.delta > 0 ? "badge-critical" : "badge-good"}">
          ${c.delta > 0 ? "+" : ""}${c.delta} ${c.feature}
        </span>
      `).join("");
    }
  }

  // Batch Inference Runner
  async function triggerBatchScoring(presetType) {
    const section = document.getElementById("batch-progress-section");
    const pBar = document.getElementById("batch-progress-bar");
    const pLbl = document.getElementById("batch-percent-label");
    const statusLbl = document.getElementById("batch-status-label");
    const pRows = document.getElementById("batch-processed-rows");
    const tPut = document.getElementById("batch-throughput");
    const fCount = document.getElementById("batch-flagged-count");
    const downloadBtn = document.getElementById("btn-download-batch-results");

    if (section) section.style.display = "block";
    if (downloadBtn) downloadBtn.style.display = "none";

    showToast(`Starting high-throughput batch scoring: ${presetType}...`);

    const result = await Api.batchScore(presetType, (progress) => {
      if (pBar) pBar.style.width = `${progress.percent}%`;
      if (pLbl) pLbl.textContent = `${progress.percent}%`;
      if (statusLbl) statusLbl.textContent = `Streaming Chunk ${progress.chunk}/${progress.totalChunks}...`;
      if (pRows) pRows.textContent = Fmt.int(progress.processed);
      if (tPut) tPut.textContent = `${Fmt.int(progress.throughput)} apps/s`;
      if (fCount) fCount.textContent = Fmt.int(progress.flaggedFraud);
    });

    showToast(`Batch completed: ${Fmt.int(result.total_rows)} applications scored in ${result.elapsed_seconds}s!`);
    if (downloadBtn) downloadBtn.style.display = "flex";

    downloadBtn.onclick = () => {
      const headers = ["application_id", "risk_score", "risk_tier", "primary_flag", "action"];
      const rows = [];
      for (let i = 0; i < 50; i++) {
        const isF = Math.random() < 0.05;
        rows.push([
          `APP-2026-${100000 + i}`,
          (isF ? 0.92 : 0.04).toFixed(4),
          isF ? "priority" : "normal",
          isF ? "velocity_burst" : "none",
          isF ? "ESCALATE" : "APPROVE"
        ]);
      }
      Fmt.downloadCsv("sentinel_scored_batch_applications.csv", headers, rows);
    };
  }

  // -----------------------------------------------------------------------
  // 8. Event Wiring & Bootstrap
  // -----------------------------------------------------------------------
  function bindGlobalEvents() {
    // Navigation Buttons
    document.querySelectorAll(".nav-btn").forEach((btn) => {
      btn.addEventListener("click", () => navigateTo(btn.getAttribute("data-page")));
    });

    document.getElementById("btn-view-all-queue")?.addEventListener("click", () => navigateTo("queue"));

    // Live Feed Controls
    document.getElementById("btn-toggle-live-feed")?.addEventListener("click", () => {
      liveFeedActive = !liveFeedActive;
      document.getElementById("feed-play-text").textContent = liveFeedActive ? "Pause Feed" : "Resume Feed";
      document.getElementById("feed-play-icon").innerHTML = liveFeedActive ? Icons.pause : Icons.play;
    });

    document.getElementById("btn-add-test-app")?.addEventListener("click", () => {
      generateRandomFeedApplication();
      showToast("Injected synthetic high-risk opening application.");
    });

    // Investigation Queue Filter Events
    ["queue-search-input", "queue-filter-tier", "queue-filter-age", "queue-filter-employment", "queue-filter-status"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("input", Fmt.debounce(loadInvestigationQueue, 180));
    });

    document.getElementById("btn-export-queue-csv")?.addEventListener("click", async () => {
      const res = await Api.applications();
      const headers = ["ApplicationID", "ApplicantName", "Age", "Employment", "Housing", "ProposedLimit", "IncomeDecile", "RiskScore", "Tier", "Status"];
      const rows = res.items.map((a) => [a.application_id, a.applicant_name, a.customer_age, a.employment_status, a.housing_status, a.proposed_credit_limit, a.income, a.risk_score, a.risk_tier_code, a.status]);
      Fmt.downloadCsv("sentinel_investigation_queue.csv", headers, rows);
      showToast("Exported investigation queue to CSV.");
    });

    // Batch Action Toolbar Buttons
    document.getElementById("btn-batch-approve")?.addEventListener("click", () => handleBatchAction("marked_legitimate", "Marked Legitimate"));
    document.getElementById("btn-batch-escalate")?.addEventListener("click", () => handleBatchAction("escalated", "Escalated"));
    document.getElementById("btn-batch-block")?.addEventListener("click", () => handleBatchAction("confirmed_fraud", "Confirmed Fraud"));

    // Threshold Slider Events
    const threshSlider = document.getElementById("input-threshold-slider");
    if (threshSlider) {
      threshSlider.addEventListener("input", (e) => updateThresholdCalculations(parseFloat(e.target.value)));
    }

    document.getElementById("preset-thresh-fpr")?.addEventListener("click", () => updateThresholdCalculations(0.18));
    document.getElementById("preset-thresh-f1")?.addEventListener("click", () => updateThresholdCalculations(0.32));
    document.getElementById("preset-thresh-precision")?.addEventListener("click", () => updateThresholdCalculations(0.75));

    ["cost-fn-input", "cost-fp-input"].forEach((id) => {
      document.getElementById(id)?.addEventListener("input", () => updateThresholdCalculations(activeThreshold));
    });

    // Batch Presets
    document.getElementById("btn-preset-10k")?.addEventListener("click", () => triggerBatchScoring("preset-10k"));
    document.getElementById("btn-preset-100k")?.addEventListener("click", () => triggerBatchScoring("preset-100k"));
    document.getElementById("btn-preset-1m")?.addEventListener("click", () => triggerBatchScoring("preset-1m"));

    document.getElementById("btn-browse-file")?.addEventListener("click", () => {
      document.getElementById("batch-file-input")?.click();
    });

    document.getElementById("batch-file-input")?.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        triggerBatchScoring(e.target.files[0]);
      }
    });

    // Inspector Action Buttons
    document.getElementById("btn-insp-confirm-fraud")?.addEventListener("click", async () => {
      await Api.queueAction(activeApplicantId, "confirmed_fraud", "Confirmed Fraud by Analyst in Inspector");
      showToast(`Application ${activeApplicantId} marked as CONFIRMED FRAUD.`);
      loadInspector(activeApplicantId);
    });

    document.getElementById("btn-insp-escalate")?.addEventListener("click", async () => {
      await Api.queueAction(activeApplicantId, "escalated", "Escalated to Tier 2 Fraud Ops");
      showToast(`Application ${activeApplicantId} ESCALATED to Tier 2.`);
      loadInspector(activeApplicantId);
    });

    document.getElementById("btn-insp-approve")?.addEventListener("click", async () => {
      await Api.queueAction(activeApplicantId, "marked_legitimate", "Approved and Verified Legitimate");
      showToast(`Application ${activeApplicantId} MARKED LEGITIMATE.`);
      loadInspector(activeApplicantId);
    });

    // Modal Close / Save
    document.getElementById("btn-close-modal")?.addEventListener("click", closeCaseModal);
    document.getElementById("btn-cancel-modal")?.addEventListener("click", closeCaseModal);
    document.getElementById("btn-save-modal")?.addEventListener("click", saveCaseDisposition);

    // Responsive Window Resize (debounced redraw)
    window.addEventListener("resize", Fmt.debounce(renderCurrentPageCharts, 250));
  }

  function init() {
    initTheme();
    injectIcons();
    bindGlobalEvents();
    navigateTo("monitor");
  }

  return { init, navigateTo, showToast };
})();

document.addEventListener("DOMContentLoaded", App.init);
