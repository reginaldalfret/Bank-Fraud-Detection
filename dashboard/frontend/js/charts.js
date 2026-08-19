/* =========================================================================
   SENTINEL Bank Fraud Classification Platform - Interactive SVG Chart Suite
   High-performance, zero-dependency, theme-reactive inline SVGs
   ========================================================================= */

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#64748b";

  function el(tag, attrs = {}) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    return node;
  }

  // --- Shared Tooltip Singleton ---
  let tooltipEl = null;
  function tooltip() {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className = "chart-tooltip";
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  }

  function showTooltip(html, clientX, clientY) {
    const t = tooltip();
    t.innerHTML = html;
    t.style.display = "block";
    const rect = t.getBoundingClientRect();
    let left = clientX + 14;
    let top = clientY + 14;
    if (left + rect.width > window.innerWidth - 12) left = clientX - rect.width - 14;
    if (top + rect.height > window.innerHeight - 12) top = clientY - rect.height - 14;
    t.style.left = `${Math.max(6, left)}px`;
    t.style.top = `${Math.max(6, top)}px`;
  }

  function hideTooltip() {
    if (tooltipEl) tooltipEl.style.display = "none";
  }

  // -------------------------------------------------------------------------
  // 1. Risk Score Distribution Histogram
  // -------------------------------------------------------------------------
  function renderScoreDistribution(container, distData, activeThreshold = 0.5) {
    if (!container) return;
    container.innerHTML = "";
    const width = Math.max(container.clientWidth || 460, 300);
    const height = 220;
    const pad = { top: 20, right: 24, bottom: 36, left: 54 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const maxCount = Math.max(...distData.map((d) => d.count), 1000) * 1.08;
    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });

    // Gridlines
    const gridColor = cssVar("--border");
    const textColor = cssVar("--text-muted");
    const numYGrid = 4;
    for (let i = 0; i <= numYGrid; i++) {
      const yVal = (maxCount / numYGrid) * i;
      const yPos = pad.top + plotH - (yVal / maxCount) * plotH;
      svg.appendChild(el("line", {
        x1: pad.left, x2: pad.left + plotW, y1: yPos, y2: yPos,
        stroke: gridColor, "stroke-width": "1", "stroke-dasharray": i === 0 ? "none" : "3,3"
      }));
      const label = el("text", {
        x: pad.left - 8, y: yPos + 4, "text-anchor": "end", "font-size": "10",
        fill: textColor, class: "font-mono"
      });
      label.textContent = Fmt.int(yVal);
      svg.appendChild(label);
    }

    const nBars = distData.length;
    const barW = (plotW / nBars) * 0.82;
    const barSpacing = plotW / nBars;

    distData.forEach((d, i) => {
      const xPos = pad.left + i * barSpacing + (barSpacing - barW) / 2;
      const barH = (d.count / maxCount) * plotH;
      const yPos = pad.top + plotH - barH;

      let fillColor = "#10b981"; // Emerald low
      if (d.risk === "moderate") fillColor = "#38bdf8"; // Sky blue
      if (d.risk === "high") fillColor = "#f59e0b"; // Amber
      if (d.risk === "critical") fillColor = "#ef4444"; // Crimson

      const bar = el("rect", {
        x: xPos, y: yPos, width: barW, height: Math.max(barH, 2), rx: 3,
        fill: fillColor, opacity: "0.88", class: "chart-interactive-bar"
      });

      bar.addEventListener("mouseenter", (ev) => {
        bar.setAttribute("opacity", "1");
        showTooltip(
          `<div class="font-semibold">${d.bin} Risk Score</div>
           <div class="text-sm font-mono mt-1">${Fmt.int(d.count)} Applications (${d.pct.toFixed(2)}%)</div>
           <div class="text-xs text-muted mt-1">Tier: <span class="capitalize font-semibold text-${fillColor}">${d.risk}</span></div>`,
          ev.clientX, ev.clientY
        );
      });
      bar.addEventListener("mousemove", (ev) => showTooltip(tooltip().innerHTML, ev.clientX, ev.clientY));
      bar.addEventListener("mouseleave", () => {
        bar.setAttribute("opacity", "0.88");
        hideTooltip();
      });

      svg.appendChild(bar);

      // X-Axis Label
      if (i % 2 === 0 || i === nBars - 1) {
        const xLabel = el("text", {
          x: xPos + barW / 2, y: pad.top + plotH + 18,
          "text-anchor": "middle", "font-size": "10", fill: textColor, class: "font-mono"
        });
        xLabel.textContent = d.bin.split(" - ")[0];
        svg.appendChild(xLabel);
      }
    });

    // Active Threshold Marker Line
    const threshX = pad.left + activeThreshold * plotW;
    svg.appendChild(el("line", {
      x1: threshX, x2: threshX, y1: pad.top - 6, y2: pad.top + plotH,
      stroke: "#ef4444", "stroke-width": "2", "stroke-dasharray": "4,2"
    }));
    const tBadge = el("text", {
      x: threshX + 4, y: pad.top + 10, fill: "#ef4444", "font-size": "10",
      "font-weight": "bold", class: "font-mono"
    });
    tBadge.textContent = `T=${activeThreshold.toFixed(2)}`;
    svg.appendChild(tBadge);

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------------
  // 2. Temporal Drift Across Months (Line / Area Chart)
  // -------------------------------------------------------------------------
  function renderMonthlyTrend(container, trendData) {
    if (!container) return;
    container.innerHTML = "";
    const width = Math.max(container.clientWidth || 500, 320);
    const height = 230;
    const pad = { top: 24, right: 54, bottom: 36, left: 54 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const maxRate = 0.02; // 2.0% top scale
    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });

    // Background Grid
    const gridColor = cssVar("--border");
    const textColor = cssVar("--text-muted");
    for (let i = 0; i <= 4; i++) {
      const yRatio = i / 4;
      const yPos = pad.top + plotH - yRatio * plotH;
      svg.appendChild(el("line", {
        x1: pad.left, x2: pad.left + plotW, y1: yPos, y2: yPos,
        stroke: gridColor, "stroke-width": "1", "stroke-dasharray": i === 0 ? "none" : "3,3"
      }));
      // Left Y-axis (Fraud Rate %)
      const leftLbl = el("text", {
        x: pad.left - 8, y: yPos + 4, "text-anchor": "end", "font-size": "10",
        fill: "#ef4444", class: "font-mono"
      });
      leftLbl.textContent = `${(yRatio * maxRate * 100).toFixed(1)}%`;
      svg.appendChild(leftLbl);

      // Right Y-axis (Applications Count)
      const rightLbl = el("text", {
        x: pad.left + plotW + 8, y: yPos + 4, "text-anchor": "start", "font-size": "10",
        fill: "#38bdf8", class: "font-mono"
      });
      rightLbl.textContent = `${Math.round(yRatio * 150)}k`;
      svg.appendChild(rightLbl);
    }

    const n = trendData.length;
    const xStep = plotW / (n - 1);

    // Points calculation
    const ratePoints = trendData.map((d, i) => ({
      x: pad.left + i * xStep,
      y: pad.top + plotH - (d.fraud_rate / maxRate) * plotH,
      data: d
    }));

    const volPoints = trendData.map((d, i) => ({
      x: pad.left + i * xStep,
      y: pad.top + plotH - (d.applications / 150000) * plotH,
      data: d
    }));

    // Draw Volume Path (Cyan area)
    const volPathD = `M${volPoints[0].x},${pad.top + plotH} ` +
      volPoints.map((p) => `L${p.x},${p.y}`).join(" ") +
      ` L${volPoints[n - 1].x},${pad.top + plotH} Z`;
    svg.appendChild(el("path", {
      d: volPathD, fill: "url(#cyanGradient)", opacity: "0.15"
    }));

    // Gradient Defs
    const defs = el("defs");
    defs.innerHTML = `
      <linearGradient id="cyanGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.6"/>
        <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
      </linearGradient>
      <linearGradient id="redGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ef4444" stop-opacity="0.5"/>
        <stop offset="100%" stop-color="#ef4444" stop-opacity="0.0"/>
      </linearGradient>
    `;
    svg.appendChild(defs);

    // Volume Line
    const volLineD = "M" + volPoints.map((p) => `${p.x},${p.y}`).join(" L");
    svg.appendChild(el("path", {
      d: volLineD, fill: "none", stroke: "#38bdf8", "stroke-width": "2", "stroke-dasharray": "3,3"
    }));

    // Fraud Rate Area & Line
    const rateAreaD = `M${ratePoints[0].x},${pad.top + plotH} ` +
      ratePoints.map((p) => `L${p.x},${p.y}`).join(" ") +
      ` L${ratePoints[n - 1].x},${pad.top + plotH} Z`;
    svg.appendChild(el("path", { d: rateAreaD, fill: "url(#redGradient)" }));

    const rateLineD = "M" + ratePoints.map((p) => `${p.x},${p.y}`).join(" L");
    svg.appendChild(el("path", {
      d: rateLineD, fill: "none", stroke: "#ef4444", "stroke-width": "2.5"
    }));

    // Data Dots & Interactions
    ratePoints.forEach((p) => {
      const dot = el("circle", {
        cx: p.x, cy: p.y, r: "5", fill: "#ef4444", stroke: cssVar("--surface-1"), "stroke-width": "2",
        class: "cursor-pointer"
      });
      dot.addEventListener("mouseenter", (ev) => {
        dot.setAttribute("r", "7");
        showTooltip(
          `<div class="font-bold">${p.data.month} (${p.data.split})</div>
           <div class="text-sm mt-1 text-crimson font-mono font-semibold">Fraud Rate: ${(p.data.fraud_rate * 100).toFixed(2)}% (${Fmt.int(p.data.fraud_count)} frauds)</div>
           <div class="text-xs text-muted font-mono mt-0.5">Volume: ${Fmt.int(p.data.applications)} applications</div>
           <div class="text-xs text-emerald font-mono mt-0.5">Model TPR@5%FPR: ${(p.data.detection_rate * 100).toFixed(1)}%</div>`,
          ev.clientX, ev.clientY
        );
      });
      dot.addEventListener("mouseleave", () => {
        dot.setAttribute("r", "5");
        hideTooltip();
      });
      svg.appendChild(dot);

      // Month Label
      const xLbl = el("text", {
        x: p.x, y: pad.top + plotH + 18, "text-anchor": "middle", "font-size": "10",
        fill: textColor, class: "font-mono"
      });
      xLbl.textContent = p.data.month.replace("Month ", "M");
      svg.appendChild(xLbl);
    });

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------------
  // 3. Top Global Risk Indicators (Horizontal Feature Importance)
  // -------------------------------------------------------------------------
  function renderTopIndicators(container, indicators) {
    if (!container) return;
    container.innerHTML = "";
    const width = Math.max(container.clientWidth || 460, 280);
    const rowH = 26;
    const pad = { top: 10, right: 60, bottom: 10, left: 180 };
    const height = pad.top + indicators.length * rowH + pad.bottom;
    const plotW = width - pad.left - pad.right;

    const maxImp = Math.max(...indicators.map((d) => d.importance), 0.05);
    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });

    indicators.forEach((d, i) => {
      const yPos = pad.top + i * rowH;
      const barW = (d.importance / maxImp) * plotW;

      // Label
      const label = el("text", {
        x: pad.left - 10, y: yPos + 14, "text-anchor": "end", "font-size": "11",
        fill: cssVar("--text-secondary"), class: "truncate"
      });
      label.textContent = d.label;
      svg.appendChild(label);

      // Bar Background
      svg.appendChild(el("rect", {
        x: pad.left, y: yPos + 2, width: plotW, height: 16, rx: 4,
        fill: cssVar("--surface-2"), opacity: "0.6"
      }));

      // Bar Foreground
      const barColor = d.direction === "positive" ? "#ef4444" : "#10b981";
      const bar = el("rect", {
        x: pad.left, y: yPos + 2, width: Math.max(barW, 3), height: 16, rx: 4,
        fill: barColor, opacity: "0.85", class: "chart-interactive-bar"
      });

      bar.addEventListener("mouseenter", (ev) => {
        bar.setAttribute("opacity", "1");
        showTooltip(
          `<div class="font-bold">${d.label}</div>
           <div class="text-xs text-muted font-mono mt-0.5">Feature: ${d.feature}</div>
           <div class="text-sm mt-1 font-mono font-semibold">Importance: ${(d.importance * 100).toFixed(1)}%</div>
           <div class="text-xs mt-0.5 text-${d.direction === "positive" ? "crimson" : "emerald"} font-semibold">Risk Vector: ${d.direction === "positive" ? "Increases Fraud Risk" : "Indicates Legitimate"}</div>`,
          ev.clientX, ev.clientY
        );
      });
      bar.addEventListener("mouseleave", () => {
        bar.setAttribute("opacity", "0.85");
        hideTooltip();
      });
      svg.appendChild(bar);

      // Value text
      const valText = el("text", {
        x: pad.left + barW + 6, y: yPos + 14, "font-size": "10",
        fill: cssVar("--text-primary"), class: "font-mono font-semibold"
      });
      valText.textContent = `${(d.importance * 100).toFixed(1)}%`;
      svg.appendChild(valText);
    });

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------------
  // 4. Interactive SHAP Waterfall Plot
  // -------------------------------------------------------------------------
  function renderShapWaterfall(container, shapData) {
    if (!container) return;
    container.innerHTML = "";
    const width = Math.max(container.clientWidth || 540, 320);
    const features = shapData.features;
    const rowH = 34;
    const pad = { top: 36, right: 70, bottom: 40, left: 190 };
    const height = pad.top + (features.length + 2) * rowH + pad.bottom;
    const plotW = width - pad.left - pad.right;

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });

    // Track running sum
    let runningVal = shapData.base_value;
    const values = [runningVal];
    features.forEach((f) => {
      runningVal += f.contribution;
      values.push(runningVal);
    });
    values.push(shapData.final_score);

    const minV = Math.min(0, ...values);
    const maxV = Math.max(1, ...values) * 1.05;
    const scaleX = (v) => pad.left + Math.max(0, Math.min(plotW, ((v - minV) / (maxV - minV)) * plotW));

    // Zero / Base line
    const baseLineX = scaleX(shapData.base_value);
    svg.appendChild(el("line", {
      x1: baseLineX, x2: baseLineX, y1: pad.top, y2: height - pad.bottom,
      stroke: cssVar("--text-muted"), "stroke-width": "1.5", "stroke-dasharray": "3,3"
    }));

    // Header Label: Base Expected Value
    const baseLbl = el("text", {
      x: baseLineX, y: pad.top - 12, "text-anchor": "middle", "font-size": "10",
      fill: cssVar("--text-muted"), class: "font-mono"
    });
    baseLbl.textContent = `E[f(x)] = ${shapData.base_value.toFixed(3)}`;
    svg.appendChild(baseLbl);

    let currentVal = shapData.base_value;

    features.forEach((f, i) => {
      const yPos = pad.top + i * rowH;
      const startX = scaleX(currentVal);
      const nextVal = currentVal + f.contribution;
      const endX = scaleX(nextVal);
      const isPositive = f.contribution >= 0;

      const barX = Math.min(startX, endX);
      const barW = Math.max(Math.abs(endX - startX), 3);
      const barColor = isPositive ? "#ef4444" : "#10b981";

      // Row Name
      const rowLbl = el("text", {
        x: pad.left - 12, y: yPos + 18, "text-anchor": "end", "font-size": "11",
        fill: cssVar("--text-primary"), class: "font-medium"
      });
      rowLbl.textContent = f.name;
      svg.appendChild(rowLbl);

      // Connecting guide line
      if (i > 0) {
        svg.appendChild(el("line", {
          x1: startX, x2: startX, y1: yPos - 6, y2: yPos + 6,
          stroke: cssVar("--border"), "stroke-width": "1"
        }));
      }

      // Feature Push Bar
      const bar = el("rect", {
        x: barX, y: yPos + 4, width: barW, height: 20, rx: 4,
        fill: barColor, opacity: "0.88", class: "chart-interactive-bar"
      });

      bar.addEventListener("mouseenter", (ev) => {
        bar.setAttribute("opacity", "1");
        showTooltip(
          `<div class="font-bold">${f.name}</div>
           <div class="text-xs text-muted font-mono mt-0.5">Observed Value: <span class="text-white font-semibold">${f.value}</span></div>
           <div class="text-sm mt-1 font-mono font-semibold text-${isPositive ? "crimson" : "emerald"}">SHAP Push: ${isPositive ? "+" : ""}${f.contribution.toFixed(4)}</div>
           <div class="text-xs text-muted mt-0.5">Shifted Risk Probability to ${(nextVal * 100).toFixed(1)}%</div>`,
          ev.clientX, ev.clientY
        );
      });
      bar.addEventListener("mouseleave", () => {
        bar.setAttribute("opacity", "0.88");
        hideTooltip();
      });
      svg.appendChild(bar);

      // Contribution badge
      const contribLbl = el("text", {
        x: endX + (isPositive ? 6 : -6), y: yPos + 18,
        "text-anchor": isPositive ? "start" : "end", "font-size": "10",
        fill: barColor, class: "font-mono font-semibold"
      });
      contribLbl.textContent = `${isPositive ? "+" : ""}${f.contribution.toFixed(3)}`;
      svg.appendChild(contribLbl);

      currentVal = nextVal;
    });

    // Final Score Row
    const finalY = pad.top + features.length * rowH + 10;
    svg.appendChild(el("line", {
      x1: pad.left - 40, x2: pad.left + plotW + 40, y1: finalY - 4, y2: finalY - 4,
      stroke: cssVar("--border"), "stroke-width": "1"
    }));

    const finalLbl = el("text", {
      x: pad.left - 12, y: finalY + 16, "text-anchor": "end", "font-size": "12",
      fill: cssVar("--text-primary"), class: "font-bold"
    });
    finalLbl.textContent = "Final Model Risk Output f(x)";
    svg.appendChild(finalLbl);

    const finalX = scaleX(shapData.final_score);
    const finalBar = el("rect", {
      x: pad.left, y: finalY + 2, width: Math.max(finalX - pad.left, 4), height: 20, rx: 4,
      fill: "#6366f1", opacity: "0.95"
    });
    svg.appendChild(finalBar);

    const finalValText = el("text", {
      x: finalX + 8, y: finalY + 16, "font-size": "12",
      fill: "#6366f1", class: "font-mono font-bold"
    });
    finalValText.textContent = `${(shapData.final_score * 100).toFixed(2)}%`;
    svg.appendChild(finalValText);

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------------
  // 5. 6-Axis Behavioral Deviation Radar Chart
  // -------------------------------------------------------------------------
  function renderBehavioralRadar(container, radarData) {
    if (!container) return;
    container.innerHTML = "";
    const size = Math.min(container.clientWidth || 380, container.clientHeight || 340, 360);
    const center = size / 2;
    const radius = center * 0.72;
    const axes = radarData.axes;
    const n = axes.length;
    const angleStep = (Math.PI * 2) / n;

    const svg = el("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}`, class: "chart-svg" });

    // Grid Concentric Polygons (20%, 40%, 60%, 80%, 100%)
    for (let level = 1; level <= 5; level++) {
      const r = (radius / 5) * level;
      const points = [];
      for (let i = 0; i < n; i++) {
        const a = i * angleStep - Math.PI / 2;
        points.push(`${center + r * Math.cos(a)},${center + r * Math.sin(a)}`);
      }
      svg.appendChild(el("polygon", {
        points: points.join(" "), fill: "none", stroke: cssVar("--border"),
        "stroke-width": "1", opacity: "0.7"
      }));
    }

    // Spokes & Axis Labels
    for (let i = 0; i < n; i++) {
      const a = i * angleStep - Math.PI / 2;
      const x2 = center + radius * Math.cos(a);
      const y2 = center + radius * Math.sin(a);
      svg.appendChild(el("line", {
        x1: center, y1: center, x2, y2, stroke: cssVar("--border"), "stroke-width": "1"
      }));

      // Label positioning
      const lblR = radius + 22;
      const lblX = center + lblR * Math.cos(a);
      const lblY = center + lblR * Math.sin(a);
      const label = el("text", {
        x: lblX, y: lblY + 4, "text-anchor": "middle", "font-size": "9.5",
        fill: cssVar("--text-secondary"), class: "font-medium"
      });
      label.textContent = axes[i];
      svg.appendChild(label);
    }

    // Helper to get polygon points from values [0-100]
    function getPolygon(vals) {
      return vals.map((v, i) => {
        const r = (radius * Math.max(0, Math.min(100, v))) / 100;
        const a = i * angleStep - Math.PI / 2;
        return `${center + r * Math.cos(a)},${center + r * Math.sin(a)}`;
      }).join(" ");
    }

    // 1. Normal Population Baseline (Emerald polygon)
    svg.appendChild(el("polygon", {
      points: getPolygon(radarData.population_normal),
      fill: "#10b981", "fill-opacity": "0.15", stroke: "#10b981", "stroke-width": "1.5",
      "stroke-dasharray": "3,3"
    }));

    // 2. Fraud Population Average (Crimson dashed polygon)
    svg.appendChild(el("polygon", {
      points: getPolygon(radarData.population_fraud),
      fill: "#ef4444", "fill-opacity": "0.12", stroke: "#ef4444", "stroke-width": "1.5",
      "stroke-dasharray": "4,2"
    }));

    // 3. This Applicant (Solid Bright Violet / Indigo polygon)
    const appPoly = el("polygon", {
      points: getPolygon(radarData.applicant),
      fill: "#818cf8", "fill-opacity": "0.35", stroke: "#6366f1", "stroke-width": "2.5"
    });
    svg.appendChild(appPoly);

    // Dots for applicant
    radarData.applicant.forEach((v, i) => {
      const r = (radius * v) / 100;
      const a = i * angleStep - Math.PI / 2;
      const dot = el("circle", {
        cx: center + r * Math.cos(a), cy: center + r * Math.sin(a), r: "4",
        fill: "#6366f1", stroke: "#ffffff", "stroke-width": "1.5"
      });
      svg.appendChild(dot);
    });

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------------
  // 6. ROC and Precision-Recall Curves with Threshold Marker
  // -------------------------------------------------------------------------
  function renderRocCurve(container, points, currentThreshold = 0.5) {
    if (!container) return;
    container.innerHTML = "";
    const width = Math.max(container.clientWidth || 300, 240);
    const height = 220;
    const pad = { top: 20, right: 20, bottom: 34, left: 44 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });

    // Grid & Diagonal random classifier
    svg.appendChild(el("line", {
      x1: pad.left, x2: pad.left + plotW, y1: pad.top + plotH, y2: pad.top,
      stroke: cssVar("--border"), "stroke-width": "1.5", "stroke-dasharray": "4,4"
    }));

    const linePoints = points.map((p) => ({
      x: pad.left + p.fpr * plotW,
      y: pad.top + plotH - p.tpr * plotH,
      p
    }));

    const pathD = "M" + linePoints.map((p) => `${p.x},${p.y}`).join(" L");
    svg.appendChild(el("path", {
      d: pathD, fill: "none", stroke: "#38bdf8", "stroke-width": "2.5"
    }));

    // Operating point on curve closest to currentThreshold
    let closest = linePoints[0];
    let minDiff = 999;
    linePoints.forEach((pt) => {
      const diff = Math.abs(pt.p.threshold - currentThreshold);
      if (diff < minDiff) {
        minDiff = diff;
        closest = pt;
      }
    });

    const activeDot = el("circle", {
      cx: closest.x, cy: closest.y, r: "6", fill: "#ef4444", stroke: "#ffffff", "stroke-width": "2"
    });
    svg.appendChild(activeDot);

    // Axis labels
    const xLbl = el("text", {
      x: pad.left + plotW / 2, y: height - 6, "text-anchor": "middle", "font-size": "10",
      fill: cssVar("--text-muted"), class: "font-mono"
    });
    xLbl.textContent = "False Positive Rate (FPR)";
    svg.appendChild(xLbl);

    const yLbl = el("text", {
      x: -(pad.top + plotH / 2), y: 14, "text-anchor": "middle", "font-size": "10",
      fill: cssVar("--text-muted"), class: "font-mono", transform: "rotate(-90)"
    });
    yLbl.textContent = "True Positive Rate (Recall)";
    svg.appendChild(yLbl);

    container.appendChild(svg);
  }

  function renderPrCurve(container, points, currentThreshold = 0.5) {
    if (!container) return;
    container.innerHTML = "";
    const width = Math.max(container.clientWidth || 300, 240);
    const height = 220;
    const pad = { top: 20, right: 20, bottom: 34, left: 44 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });

    // Baseline fraud prevalence horizontal line (~1.1%)
    const basePrevalenceY = pad.top + plotH - 0.011 * plotH;
    svg.appendChild(el("line", {
      x1: pad.left, x2: pad.left + plotW, y1: basePrevalenceY, y2: basePrevalenceY,
      stroke: "#64748b", "stroke-width": "1", "stroke-dasharray": "3,3"
    }));

    const linePoints = points.map((p) => ({
      x: pad.left + p.recall * plotW,
      y: pad.top + plotH - p.precision * plotH,
      p
    }));

    const pathD = "M" + linePoints.map((p) => `${p.x},${p.y}`).join(" L");
    svg.appendChild(el("path", {
      d: pathD, fill: "none", stroke: "#10b981", "stroke-width": "2.5"
    }));

    // Operating point
    let closest = linePoints[0];
    let minDiff = 999;
    linePoints.forEach((pt) => {
      const diff = Math.abs(pt.p.threshold - currentThreshold);
      if (diff < minDiff) {
        minDiff = diff;
        closest = pt;
      }
    });

    const activeDot = el("circle", {
      cx: closest.x, cy: closest.y, r: "6", fill: "#ef4444", stroke: "#ffffff", "stroke-width": "2"
    });
    svg.appendChild(activeDot);

    // Axis labels
    const xLbl = el("text", {
      x: pad.left + plotW / 2, y: height - 6, "text-anchor": "middle", "font-size": "10",
      fill: cssVar("--text-muted"), class: "font-mono"
    });
    xLbl.textContent = "Recall (Fraud Caught)";
    svg.appendChild(xLbl);

    const yLbl = el("text", {
      x: -(pad.top + plotH / 2), y: 14, "text-anchor": "middle", "font-size": "10",
      fill: cssVar("--text-muted"), class: "font-mono", transform: "rotate(-90)"
    });
    yLbl.textContent = "Precision";
    svg.appendChild(yLbl);

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------------
  // 7. Calibration Curve (Reliability Diagram)
  // -------------------------------------------------------------------------
  function renderCalibrationCurve(container, calData) {
    if (!container) return;
    container.innerHTML = "";
    const width = Math.max(container.clientWidth || 300, 240);
    const height = 220;
    const pad = { top: 20, right: 20, bottom: 34, left: 44 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });

    // Perfect calibration diagonal
    svg.appendChild(el("line", {
      x1: pad.left, x2: pad.left + plotW, y1: pad.top + plotH, y2: pad.top,
      stroke: cssVar("--border"), "stroke-width": "1.5", "stroke-dasharray": "3,3"
    }));

    // Bins
    const points = calData.bins.map((b) => ({
      x: pad.left + b.mean_pred * plotW,
      y: pad.top + plotH - b.obs_fraud * plotH,
      b
    }));

    const pathD = "M" + points.map((p) => `${p.x},${p.y}`).join(" L");
    svg.appendChild(el("path", {
      d: pathD, fill: "none", stroke: "#818cf8", "stroke-width": "2"
    }));

    points.forEach((p) => {
      const dot = el("circle", {
        cx: p.x, cy: p.y, r: "4.5", fill: "#6366f1", stroke: "#ffffff", "stroke-width": "1.5"
      });
      dot.addEventListener("mouseenter", (ev) => {
        dot.setAttribute("r", "6.5");
        showTooltip(
          `<div class="font-bold">Calibration Bin</div>
           <div class="text-xs font-mono mt-0.5">Mean Pred Probability: ${(p.b.mean_pred * 100).toFixed(1)}%</div>
           <div class="text-xs font-mono mt-0.5 text-emerald font-semibold">Observed Fraud: ${(p.b.obs_fraud * 100).toFixed(1)}%</div>
           <div class="text-xs text-muted font-mono mt-0.5">Sample Size: ${Fmt.int(p.b.count)} apps</div>`,
          ev.clientX, ev.clientY
        );
      });
      dot.addEventListener("mouseleave", () => {
        dot.setAttribute("r", "4.5");
        hideTooltip();
      });
      svg.appendChild(dot);
    });

    // Axis labels
    const xLbl = el("text", {
      x: pad.left + plotW / 2, y: height - 6, "text-anchor": "middle", "font-size": "10",
      fill: cssVar("--text-muted"), class: "font-mono"
    });
    xLbl.textContent = "Mean Predicted Probability";
    svg.appendChild(xLbl);

    const yLbl = el("text", {
      x: -(pad.top + plotH / 2), y: 14, "text-anchor": "middle", "font-size": "10",
      fill: cssVar("--text-muted"), class: "font-mono", transform: "rotate(-90)"
    });
    yLbl.textContent = "Observed Fraud Fraction";
    svg.appendChild(yLbl);

    container.appendChild(svg);
  }

  return {
    renderScoreDistribution,
    renderMonthlyTrend,
    renderTopIndicators,
    renderShapWaterfall,
    renderBehavioralRadar,
    renderRocCurve,
    renderPrCurve,
    renderCalibrationCurve
  };
})();
