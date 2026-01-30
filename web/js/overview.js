/* overview.js */
console.log("overview.js loaded", window.location.href);

const CACHE_KEY = "FG_CHART_MAX_V1";
const API_URL =
  "/v1/chart?range=MAX&include=fgi&with_components=true&use_calibrated_model=true";

let gaugeChart = null;

/* -----------------------------
   Theme helpers
------------------------------ */
function getThemeTokens() {
  const cs = getComputedStyle(document.documentElement);
  return {
    text: cs.getPropertyValue("--text").trim(),
    muted: cs.getPropertyValue("--muted").trim(),
    stroke: cs.getPropertyValue("--stroke").trim(),
  };
}

/* -----------------------------
   Sentiment mapping (CNN-like)
------------------------------ */
const MAP = [0, 25, 45, 55, 75, 100];
const LABS = ["extreme fear", "fear", "neutral", "greed", "extreme greed"];

function sentimentLabel(v) {
  const x = Number(v);
  if (!Number.isFinite(x)) return "neutral";
  if (x < MAP[1]) return LABS[0];
  if (x < MAP[2]) return LABS[1];
  if (x < MAP[3]) return LABS[2];
  if (x < MAP[4]) return LABS[3];
  return LABS[4];
}

function badgeClass(label) {
  switch (label) {
    case "extreme fear":
      return "badge--extreme-fear";
    case "fear":
      return "badge--fear";
    case "neutral":
      return "badge--neutral";
    case "greed":
      return "badge--greed";
    case "extreme greed":
      return "badge--extreme-greed";
    default:
      return "badge--neutral";
  }
}

/* -----------------------------
   Data helpers
------------------------------ */
function normalizeSeries(points) {
  return (points || [])
    .map((p) => ({ date: String(p.date), value: Number(p.value) }))
    .filter((p) => p.date && Number.isFinite(p.value))
    .sort((a, b) => a.date.localeCompare(b.date));
}

function isoToMs(iso) {
  return new Date(iso + "T00:00:00Z").getTime();
}

// nearest point at or before target date
function getNearestPoint(series, targetISO) {
  const t = isoToMs(targetISO);
  for (let i = series.length - 1; i >= 0; i--) {
    if (isoToMs(series[i].date) <= t) return series[i];
  }
  return series[0] || null;
}

function shiftISO(iso, days) {
  const d = new Date(iso + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

/* -----------------------------
   API + cache
------------------------------ */
async function loadChartPayload() {
  const cached = sessionStorage.getItem(CACHE_KEY);
  if (cached) {
    try {
      return JSON.parse(cached);
    } catch (e) {}
  }

  const res = await fetch(API_URL, { credentials: "same-origin" });
  const txt = await res.text();
  if (!res.ok) throw new Error(`Overview API error ${res.status}: ${txt}`);

  const payload = JSON.parse(txt);
  try {
    sessionStorage.setItem(CACHE_KEY, JSON.stringify(payload));
  } catch (e) {}
  return payload;
}

/* -----------------------------
   Gauge
------------------------------ */
function ensureGauge() {
  if (gaugeChart) return gaugeChart;
  const el = document.getElementById("gaugeChart");
  gaugeChart = echarts.init(el);
  window.addEventListener("resize", () => gaugeChart && gaugeChart.resize());
  return gaugeChart;
}

function renderGauge(value) {
  const tokens = getThemeTokens();
  const label = sentimentLabel(value).toUpperCase();
  const chart = ensureGauge();

  const seg = [
    [0.25, "#e74c3c"], // extreme fear
    [0.45, "#f39c12"], // fear
    [0.55, "#f1c40f"], // neutral
    [0.75, "#2ecc71"], // greed
    [1.0, "#16a085"],  // extreme greed
  ];

  chart.setOption(
    {
      animation: true,
      tooltip: { show: false },
      series: [
        {
          type: "gauge",
          min: 0,
          max: 100,
          startAngle: 200,
          endAngle: -20,

          radius: "88%",
          center: ["50%", "56%"],

          splitNumber: 4,

          axisLine: {
            lineStyle: {
              width: 22,
              color: seg,
            },
          },

          axisTick: {
            distance: -18,
            length: 6,
            lineStyle: { color: tokens.stroke, width: 1 },
          },

          splitLine: {
            distance: -18,
            length: 14,
            lineStyle: { color: tokens.stroke, width: 2 },
          },

          axisLabel: {
            distance: 18,
            color: tokens.muted,
            fontSize: 13,
            formatter: (v) =>
              v === 0 || v === 25 || v === 50 || v === 75 || v === 100
                ? String(v)
                : "",
          },

          pointer: {
            length: "82%",
            width: 6,
          },

          itemStyle: {
            color: tokens.text,
          },

          anchor: {
            show: true,
            showAbove: true,
            size: 12,
            itemStyle: { color: tokens.text },
          },

          detail: {
            valueAnimation: true,
            offsetCenter: [0, "24%"],
            formatter: () => `{val|${Math.round(value)}}\n{lab|${label}}`,
            rich: {
              val: {
                fontSize: 64,
                fontWeight: 900,
                lineHeight: 70,
                color: tokens.text,
              },
              lab: {
                fontSize: 18,
                fontWeight: 800,
                lineHeight: 22,
                color: tokens.muted,
              },
            },
          },

          data: [{ value }],
        },
      ],
    },
    { notMerge: true }
  );

  chart.resize();
}

/* -----------------------------
   Right-side stats
------------------------------ */
function renderStats(series) {
  const statsEl = document.getElementById("overviewStats");
  const lastUpdatedEl = document.getElementById("lastUpdated");

  if (!series.length) {
    statsEl.innerHTML = `<div class="muted">No data</div>`;
    lastUpdatedEl.textContent = "";
    return;
  }

  const last = series[series.length - 1];
  const lastISO = last.date;

  const pPrev = last;
  const pWeek = getNearestPoint(series, shiftISO(lastISO, -7));
  const pMonth = getNearestPoint(series, shiftISO(lastISO, -30));
  const pYear = getNearestPoint(series, shiftISO(lastISO, -365));

  const rows = [
    { title: "Previous close", point: pPrev },
    { title: "1 week ago", point: pWeek },
    { title: "1 month ago", point: pMonth },
    { title: "1 year ago", point: pYear },
  ];

//   statsEl.innerHTML = rows
//     .map((r) => {
//       const val = Math.round(r.point?.value ?? NaN);
//       const lab = sentimentLabel(val);
//       const cls = badgeClass(lab);

//       return `
//         <div class="stat-row">
//           <div class="stat-left">
//             <div class="stat-title">${r.title}</div>
//             <div class="stat-sentiment">${lab}</div>
//           </div>
//           <div class="stat-dots" aria-hidden="true"></div>
//           <div class="overview-badge ${cls}">
//             ${Number.isFinite(val) ? val : "-"}
//           </div>
//         </div>
//       `;
//     })
//     .join("");
    statsEl.innerHTML = rows.map(({ title, point }) => {
      const value = Math.round(point?.value ?? NaN);
      const label = sentimentLabel(value); // "neutral", "fear", ...
      const slug = label.replace(/\s+/g, "-"); // "extreme-fear"

      return `
        <div class="overview-row">
          <div class="overview-lefttext">
            <div class="overview-label">${title}</div>
            <div class="overview-sentiment">${label}</div>
          </div>

          <div class="overview-dots" aria-hidden="true"></div>

          <div class="overview-badge overview-badge--${slug}">
            ${Number.isFinite(value) ? value : "-"}
          </div>
        </div>
      `;
    }).join("");

  lastUpdatedEl.textContent = `Last updated: ${lastISO}`;

  // Jauge = previous close
  renderGauge(pPrev.value);
}

/* -----------------------------
   Main
------------------------------ */
async function main() {
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute("data-theme", savedTheme);

  // Update theme label on load
  const themeLabel = document.getElementById("themeLabel");
  if (themeLabel) {
    themeLabel.textContent = savedTheme === "dark" ? "Dark" : "Light";
  }

  // Theme toggle
  const btnTheme = document.getElementById("btnTheme");
  if (btnTheme && !btnTheme.__bound) {
    btnTheme.__bound = true;
    btnTheme.addEventListener("click", () => {
      const html = document.documentElement;
      const cur = html.getAttribute("data-theme") || "dark";
      const newTheme = cur === "dark" ? "light" : "dark";
      html.setAttribute("data-theme", newTheme);
      localStorage.setItem('theme', newTheme);
      if (themeLabel) {
        themeLabel.textContent = newTheme === "dark" ? "Dark" : "Light";
      }
      main().catch(console.error); // re-render sans refetch
    });
  }

  // Navigation back to timeline
  const btnPrev = document.getElementById("btnPrev");
  if (btnPrev && !btnPrev.__bound) {
    btnPrev.__bound = true;
    btnPrev.addEventListener("click", () => {
      window.location.href = "/web/";
    });
  }

  const payload = await loadChartPayload();
  const series = normalizeSeries(payload?.datasets?.fgi || []);
  renderStats(series);
}

document.addEventListener("DOMContentLoaded", () => {
  main().catch((err) => console.error(err));
});
