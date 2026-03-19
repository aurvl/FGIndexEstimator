console.log("[BOOT] app.js loaded", window.location.href);

import { fetchChart } from "./api.js";
import { createChart, renderChart, THEMES } from "./chart.js";
import { setActiveTab, setActiveRange, applyTheme, initMarketDropdown, getSelectedMarkets, toggleDebugPanel, showDebugPayload } from "./ui.js";

const DEFAULT_INCLUDE = ["fgi", "sp500", "nasdaq", "cac40", "msciworld"];
const TIMEFRAMES = {
  "1M": { months: 1 },
  "3M": { months: 3 },
  "6M": { months: 6 },
  "1Y": { years: 1 },
  "5Y": { years: 5 },
  "MAX": null
};
const state = {
  theme: localStorage.getItem('theme') || "dark",
  tab: "tab-fgi",
  range: "1Y",
  use_calibrated_model: true,
  with_components: false,
  viewIndex: 0,
  cache: {
    chartMax: null, // {datasets, meta, components}
    chartMaxWithComponents: null
  },
  selectedMarketKey: null
};

const chart = createChart(document.getElementById("chart"));

const MARKET_LABELS = {
  sp500: "S&P 500",
  nasdaq: "NASDAQ Composite",
  cac40: "CAC 40",
  msciworld: "MSCI World",
};

async function fetchAndCacheChartMax(withComponents = false) {
  const data = await fetchChart({
    range: "MAX",
    include: DEFAULT_INCLUDE,
    with_components: withComponents,
    use_calibrated_model: state.use_calibrated_model
  });
  if (withComponents) {
    state.cache.chartMaxWithComponents = data;
  } else {
    state.cache.chartMax = data;
  }
  console.debug("Fetched chart MAX", { withComponents, keys: Object.keys(data.datasets || {}) });
}

function filterSeriesByTimeframe(series, range) {
  if (!series || !Array.isArray(series) || range === "MAX") return series || [];
  if (!series.length) return [];
  const parse = d => new Date(d);
  const endDate = parse(series[series.length - 1].date);
  let startDate = new Date(endDate);
  const tf = TIMEFRAMES[range];
  if (tf.years) startDate.setFullYear(startDate.getFullYear() - tf.years);
  if (tf.months) startDate.setMonth(startDate.getMonth() - tf.months);
  return series.filter(p => parse(p.date) >= startDate);
}

function getFilteredDatasets(range, withComponents = false) {
  const data = withComponents ? state.cache.chartMaxWithComponents : state.cache.chartMax;
  if (!data || !data.datasets) return {};
  const filtered = {};
  for (const key of Object.keys(data.datasets)) {
    filtered[key] = filterSeriesByTimeframe(data.datasets[key], range);
    console.debug(`Filtered ${key} for ${range}:`, filtered[key].length);
  }
  return filtered;
}

function updateChartForState() {
  const datasets = getFilteredDatasets(state.range, state.with_components);
  renderChart(chart, datasets, state.selectedMarketKey, THEMES[state.theme]);
}

async function refresh() {
  try {
    // On first load, fetch MAX if not cached
    if (!state.cache.chartMax) {
      await fetchAndCacheChartMax(false);
    }
    if (state.with_components && !state.cache.chartMaxWithComponents) {
      await fetchAndCacheChartMax(true);
    }
    updateChartForState();
    const data = state.with_components ? state.cache.chartMaxWithComponents : state.cache.chartMax;
    showDebugPayload({
      request: { range: "MAX", include: DEFAULT_INCLUDE, with_components: state.with_components },
      responseSummary: { keys: Object.keys(data.datasets || {}) },
      response: data
    });
  } catch (e) {
    console.error(e);
  }
}

// --- Tabs
// Keep tab behavior (single tab)
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", async () => {
    state.tab = btn.dataset.tab;
    setActiveTab(state.tab);
    await refresh();
  });
});

// --- Theme toggle
document.getElementById("btnTheme").addEventListener("click", () => {
  state.theme = state.theme === "dark" ? "light" : "dark";
  localStorage.setItem('theme', state.theme);
  applyTheme(state.theme);
  updateChartForState();
  chart.resize();
});

// --- Arrows (views)
function updateNavButtons(viewIndex) {
  const btnPrev = document.getElementById("btnPrev");
  const btnNext = document.getElementById("btnNext");
  if (viewIndex === 0) {
    if (btnPrev) btnPrev.style.display = "none";
    if (btnNext) btnNext.style.display = "inline-flex";
  } else if (viewIndex === 1) {
    if (btnPrev) btnPrev.style.display = "inline-flex";
    if (btnNext) btnNext.style.display = "inline-flex";
  }
}

document.getElementById("btnPrev").addEventListener("click", async () => {
  // single view - just trigger a refresh
  await refresh();
});

const btnNext = document.getElementById("btnNext");
if (btnNext) {
  btnNext.addEventListener("click", () => {
    window.location.href = "/web/overview.html";
  });
}

// --- Range buttons (tab 1)
const rangeGroup = document.getElementById("rangeGroup");
rangeGroup.addEventListener("click", async (e) => {
  const btn = e.target.closest(".seg");
  if (!btn) return;
  state.range = btn.dataset.range;
  setActiveRange(rangeGroup, state.range);
  updateChartForState(); // just filter and update, no fetch
});

// --- Market dropdown
initMarketDropdown('marketDropdown');
const mdd = document.getElementById('marketDropdown');
if (mdd) {
  mdd.addEventListener('market-change', async () => {
    const selected = getSelectedMarkets()[0] || null;
    state.selectedMarketKey = selected;
    updateChartForState();
  });
}

// Debug toggle
const btnDebug = document.getElementById('btnDebug');
if (btnDebug) btnDebug.addEventListener('click', () => toggleDebugPanel());
const debugClose = document.getElementById('debugClose');
if (debugClose) debugClose.addEventListener('click', () => toggleDebugPanel());

// --- Components toggle (removed from UI, but keep logic safe)
const chkComponents = document.getElementById("chkComponents");
if (chkComponents) {
  chkComponents.addEventListener("change", async (e) => {
    state.with_components = e.target.checked;
    if (state.with_components && !state.cache.chartMaxWithComponents) {
      await fetchAndCacheChartMax(true);
    }
    updateChartForState();
  });
}

// Init
applyTheme(state.theme);
setActiveTab(state.tab);
setActiveRange(rangeGroup, state.range);
updateNavButtons(state.viewIndex);
document.addEventListener("DOMContentLoaded", async () => {
  await refresh(); // initial load: fetch MAX and plot
  updateChartForState();
});
