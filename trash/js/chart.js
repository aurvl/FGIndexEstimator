export const THEMES = {
  dark: {
    text: "rgba(255,255,255,0.82)",
    textMuted: "rgba(255,255,255,0.60)",
    axisLine: "rgba(255,255,255,0.28)",
    gridLine: "rgba(255,255,255,0.12)",
    tooltipBg: "rgba(15,18,25,0.92)",
    tooltipText: "rgba(255,255,255,0.92)",
    zoomBg: "rgba(255,255,255,0.10)",
    zoomFiller: "rgba(255,255,255,0.18)",
    seriesFGI: "#0077b6",
    seriesMarket: "#9BE56C"
  },
  light: {
    text: "rgba(10,14,22,0.82)",
    textMuted: "rgba(10,14,22,0.58)",
    axisLine: "rgba(10,14,22,0.20)",
    gridLine: "rgba(10,14,22,0.10)",
    tooltipBg: "rgba(255,255,255,0.95)",
    tooltipText: "rgba(10,14,22,0.92)",
    zoomBg: "rgba(10,14,22,0.06)",
    zoomFiller: "rgba(10,14,22,0.10)",
    seriesFGI: "#0077b6",
    seriesMarket: "#2E9B3A"
  }
};

// Build a robust ECharts option for FGI + market time series
export function buildOption(datasets, selectedMarketId, themeTokens) {
  // Defensive: require FGI data
  if (!datasets || !datasets.fgi || !Array.isArray(datasets.fgi) || datasets.fgi.length === 0) {
    return null;
  }
  const fgiSeries = {
    name: "FGI",
    type: "line",
    showSymbol: false,
    symbol: "none",
    yAxisIndex: 0,
    lineStyle: { color: themeTokens.seriesFGI, width: 2 },
    data: datasets.fgi.map(p => [p.date, Number(p.value)])
  };

  let marketSeries = null;
  if (selectedMarketId && datasets[selectedMarketId] && Array.isArray(datasets[selectedMarketId])) {
    marketSeries = {
      name: selectedMarketId.toUpperCase(),
      type: "line",
      showSymbol: false,
      symbol: "none",
      yAxisIndex: 1,
      lineStyle: { color: themeTokens.seriesMarket, width: 2 },
      data: datasets[selectedMarketId].map(p => [p.date, Number(p.value)])
    };
  }

  return {
    animation: false,
    grid: { left: 50, right: 60, top: 35, bottom: 60 },
    textStyle: { color: themeTokens.text },
    tooltip: {
      trigger: "axis",
      backgroundColor: themeTokens.tooltipBg,
      borderColor: themeTokens.textMuted,
      textStyle: { color: themeTokens.tooltipText },
      formatter: function(params) {
        if (!params || !params.length) return "";
        let s = `<div style='font-weight:bold;margin-bottom:2px'>${params[0].axisValueLabel}</div>`;
        for (const p of params) {
          const isFGI = p.seriesName === "FGI";
          const val = isFGI ? Number(p.data[1]).toFixed(4) : Number(p.data[1]).toFixed(2);
          s += `<div><span style='color:${p.color};font-weight:bold'>${p.seriesName}:</span> ${val}</div>`;
        }
        return s;
      }
    },
    legend: { top: 6, textStyle: { color: themeTokens.text } },
    xAxis: {
      type: "time",
      boundaryGap: false,
      axisLine: { lineStyle: { opacity: 0.35, color: themeTokens.axisLine } },
      axisTick: { lineStyle: { color: themeTokens.axisLine } },
      axisLabel: { color: themeTokens.text },
      nameTextStyle: { color: themeTokens.text },
      splitLine: { show: false }
    },
    yAxis: [
      {
        type: "value",
        name: "FGI (0–100)",
        min: 0,
        max: 100,
        splitNumber: 6,
        axisLine: { lineStyle: { opacity: 0.35, color: themeTokens.axisLine } },
        axisTick: { lineStyle: { color: themeTokens.axisLine } },
        axisLabel: { color: themeTokens.text },
        nameTextStyle: { color: themeTokens.text },
        splitLine: { show: true, lineStyle: { color: themeTokens.gridLine, opacity: 1, type: 'dotted' } }
      },
      {
        type: "value",
        name: "Close",
        scale: true,
        alignTicks: true,
        splitNumber: 6,
        axisLine: { lineStyle: { opacity: 0.35, color: themeTokens.axisLine } },
        axisTick: { lineStyle: { color: themeTokens.axisLine } },
        axisLabel: { color: themeTokens.text },
        nameTextStyle: { color: themeTokens.text },
        splitLine: { show: false }
      }
    ],
    toolbox: {
      right: 10,
      top: 6,
      iconStyle: { borderColor: themeTokens.textMuted, color: themeTokens.textMuted },
      feature: {
        dataZoom: { yAxisIndex: "none" },
        restore: {},
        saveAsImage: {}
      }
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0, textStyle: { color: themeTokens.textMuted }, backgroundColor: themeTokens.zoomBg, fillerColor: themeTokens.zoomFiller },
      { type: "slider", xAxisIndex: 0, height: 16, bottom: 30, textStyle: { color: themeTokens.textMuted }, handleStyle: { color: themeTokens.textMuted, borderColor: themeTokens.textMuted }, brushStyle: { color: themeTokens.textMuted }, backgroundColor: themeTokens.zoomBg, fillerColor: themeTokens.zoomFiller }
    ],
    series: marketSeries ? [fgiSeries, marketSeries] : [fgiSeries]
  };
}

// Render chart with defensive guards and debug logs
export function renderChart(chart, datasets, selectedMarketId, themeTokens) {
  const option = buildOption(datasets, selectedMarketId, themeTokens);
  if (!option) {
    const banner = document.getElementById("chartErrorBanner");
    if (banner) {
      banner.textContent = "Error: No FGI data available.";
      banner.style.display = "block";
    }
    chart.clear();
    return;
  } else {
    const banner = document.getElementById("chartErrorBanner");
    if (banner) banner.style.display = "none";
  }
  try {
    console.log("ECharts option keys:", Object.keys(option));
    console.log("series length:", option.series.length);
    if (option.series[0]) console.log("first 3 points:", option.series[0].data.slice(0,3));
    chart.setOption(option, { notMerge: true, lazyUpdate: false });
  } catch (e) {
    console.error("ECharts setOption error", e, option);
    chart.clear();
    return;
  }
  chart.resize();
  const rect = chart.getDom().getBoundingClientRect();
  console.log("chart bounding rect", rect.width, rect.height);
}
export function createChart(domEl) {
  const chart = echarts.init(domEl);

  const option = {
    animation: false,
    grid: { left: 50, right: 60, top: 35, bottom: 40 },
    tooltip: { trigger: "axis" },
    legend: { top: 6 },
    xAxis: { type: "time", boundaryGap: false, axisLine: { lineStyle: { opacity: 0.35 } }, splitLine: { show: false } },
    yAxis: [
      { type: "value", name: "FG (0–100)", min: 0, max: 100, axisLine: { lineStyle: { opacity: 0.35 } }, splitLine: { lineStyle: { opacity: 0.18 } } },
      { type: "value", name: "Close", scale: true, axisLine: { lineStyle: { opacity: 0.35 } }, splitLine: { lineStyle: { opacity: 0.12 } } }
    ],
    toolbox: {
      right: 10,
      top: 6,
      feature: {
        dataZoom: { yAxisIndex: "none" },
        restore: {},
        saveAsImage: {}
      }
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0 },
      { type: "slider", xAxisIndex: 0, height: 18, bottom: 10 }
    ],
    series: []
  };

  chart.setOption(option);
  window.addEventListener("resize", () => chart.resize());
  return chart;
}

export function updateChart(chart, datasets, { showFG = true, showMarkets = true, labels = {}, marketKey = null, marketLabel = "" } = {}) {
  // datasets: { key: [{date,value}, ...], ...}
  const keys = Object.keys(datasets);
  if (!datasets.fgi || !Array.isArray(datasets.fgi) || datasets.fgi.length === 0) {
    // Show error banner if FGI missing
    const banner = document.getElementById("chartErrorBanner");
    if (banner) {
      banner.textContent = "Error: No FGI data available.";
      banner.style.display = "block";
    }
    chart.clear();
    return;
  } else {
    const banner = document.getElementById("chartErrorBanner");
    if (banner) banner.style.display = "none";
  }

  // Build series for ECharts time axis
  const series = [];
  // Always plot FGI
  if (showFG && datasets.fgi) {
    series.push({
        name: "FGI",
        type: "line",
        yAxisIndex: 0,
        data: datasets.fgi.map(p => [p.date, Number(p.value)]),

        // pas de points visibles par défaut
        showSymbol: false,

        // forme du point au hover
        symbol: "circle",
        symbolSize: 8,

        // style normal (ligne + point)
        lineStyle: {
            width: 2
        },
        itemStyle: {
            borderWidth: 0
        },

        // style AU HOVER (tooltip)
        emphasis: {
            focus: "series",
            scale: true,
            itemStyle: {
            borderWidth: 0
            }
        }
    });
  }
  // Plot selected market if present
  if (marketKey && datasets[marketKey]) {
    series.push({
      name: marketLabel || marketKey.toUpperCase(),
      type: "line",
      showSymbol: false,
      yAxisIndex: 1,
      data: datasets[marketKey].map(p => [p.date, Number(p.value)])
    });
  }

  chart.setOption({ series }, { notMerge: true });
  chart.resize();
  // Debug logs
  console.log("updateChart: series count", series.length);
  if (series[0]) console.log("series[0] data sample", series[0].data.slice(0,5));
  const rect = chart.getDom().getBoundingClientRect();
  console.log("chart bounding rect", rect.width, rect.height);
}
