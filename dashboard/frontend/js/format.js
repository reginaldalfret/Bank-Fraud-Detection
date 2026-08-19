/* =========================================================================
   SENTINEL Bank Fraud Classification Platform - Formatting & DOM Utilities
   ========================================================================= */

const Fmt = (() => {
  const numberFmt = new Intl.NumberFormat("en-US");
  const currencyFmt = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  const currencyPreciseFmt = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2 });
  const pctFmt = new Intl.NumberFormat("en-US", { style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const pctPreciseFmt = new Intl.NumberFormat("en-US", { style: "percent", minimumFractionDigits: 4, maximumFractionDigits: 4 });

  const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function int(n) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    return numberFmt.format(Math.round(n));
  }

  function float(n, decimals = 2) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    return Number(n).toFixed(decimals);
  }

  function money(n, precise = false) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    return precise ? currencyPreciseFmt.format(n) : currencyFmt.format(n);
  }

  function pct(n, decimals = 2) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    return decimals === 4 ? pctPreciseFmt.format(n) : pctFmt.format(n);
  }

  function score(n) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    return `${(Number(n) * 100).toFixed(1)}%`;
  }

  function scoreRaw(n) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    return Number(n).toFixed(4);
  }

  function dateTime(isoOrTs) {
    if (!isoOrTs) return "—";
    const d = new Date(isoOrTs);
    if (isNaN(d.getTime())) return String(isoOrTs);
    return d.toLocaleString("en-US", {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit"
    });
  }

  function dateOnly(isoOrTs) {
    if (!isoOrTs) return "—";
    const d = new Date(isoOrTs);
    if (isNaN(d.getTime())) return String(isoOrTs);
    return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  }

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function debounce(fn, wait = 200) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), wait);
    };
  }

  /** Animate a number counting up inside `el` */
  function countUp(el, target, { duration = 650, formatter = int, decimals = 0 } = {}) {
    if (!el) return;
    if (target === null || target === undefined || isNaN(target)) {
      el.textContent = "—";
      return;
    }
    if (prefersReducedMotion()) {
      el.textContent = formatter(target);
      return;
    }
    const start = performance.now();
    const from = 0;
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = from + (target - from) * eased;
      el.textContent = formatter(decimals ? Number(value.toFixed(decimals)) : value);
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = formatter(target);
    }
    requestAnimationFrame(tick);
  }

  /** Helper to trigger instant CSV file download */
  function downloadCsv(filename, headers, rows) {
    const csvContent = [
      headers.map((h) => `"${String(h).replace(/"/g, '""')}"`).join(","),
      ...rows.map((row) =>
        row.map((cell) => `"${String(cell ?? "").replace(/"/g, '""')}"`).join(",")
      ),
    ].join("\r\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  // Friendly human interpretations of BAF Bank Account Opening variables
  const EMPLOYMENT_MAP = {
    CA: "CA (Full-Time Corporate)",
    CB: "CB (Self-Employed / Freelance)",
    CC: "CC (Contract / Temp)",
    CD: "CD (Public Sector / Civil)",
    CE: "CE (Unemployed / Seeking)",
    CF: "CF (Student / Academic)",
    CG: "CG (Retired / Pensioner)",
  };

  const HOUSING_MAP = {
    BA: "BA (Homeowner w/ Mortgage)",
    BB: "BB (Homeowner Outright)",
    BC: "BC (Private Rental)",
    BD: "BD (Social / Subsidized)",
    BE: "BE (Living with Family)",
    BF: "BF (Temporary / Hostel)",
    BG: "BG (Other / Unverified)",
  };

  function formatEmployment(code) {
    return EMPLOYMENT_MAP[code] || `Code ${code}`;
  }

  function formatHousing(code) {
    return HOUSING_MAP[code] || `Code ${code}`;
  }

  function formatMissingMonths(val) {
    if (val === -1 || val === "-1" || val === null || val === undefined) {
      return '<span class="text-amber font-mono">No prior record (-1)</span>';
    }
    return `${val} months (${(val / 12).toFixed(1)} yrs)`;
  }

  return {
    int,
    float,
    money,
    pct,
    score,
    scoreRaw,
    dateTime,
    dateOnly,
    escapeHtml,
    debounce,
    countUp,
    downloadCsv,
    formatEmployment,
    formatHousing,
    formatMissingMonths,
  };
})();
