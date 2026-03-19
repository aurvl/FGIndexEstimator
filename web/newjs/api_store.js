/**
 * api_store.js
 *
 * Goal
 * ----
 * Fetch the dashboard datasets ONCE from the FastAPI backend and share them
 * across pages (line_chart.html, gauge_chart.html, point_chart.html) without
 * re-fetching on each navigation.
 *
 * How it works
 * ------------
 * - We call `/v1/chart?range=MAX&include=...` one time.
 * - The full payload is stored in `sessionStorage`.
 * - Other pages read from `sessionStorage` and reuse the exact same datasets.
 * - We do NOT use ES modules here (no bundler). Everything is attached to
 *   `window.DashboardStore`.
 *
 * Notes
 * -----
 * - `sessionStorage` is per-tab: opening a new tab will fetch again.
 * - We fetch MAX once, then slice client-side for 1M/3M/... to avoid refetch.
 */

(function () {
  'use strict';

  // Bump this if you change the cached shape.
  var CACHE_VERSION = 1;

  // Data cache (big) + UI state (small) are stored separately.
  var DATA_KEY = 'fgi_dashboard_data_v' + CACHE_VERSION;
  var STATE_KEY = 'fgi_dashboard_state_v' + CACHE_VERSION;

  // Cache expires after 6 hours by default (adjust as you like).
  var DEFAULT_MAX_AGE_MS = 6 * 60 * 60 * 1000;

  // Keep this list in sync with `deploy/services/market_service.py`.
  // We request them all in ONE call so the user can switch without refetch.
  var DEFAULT_INCLUDE = ['fgi', 'sp500', 'nasdaq', 'cac40', 'msciworld'];

  function safeJsonParse(str) {
    try {
      return JSON.parse(str);
    } catch (_e) {
      return null;
    }
  }

  function nowMs() {
    return Date.now();
  }

  function isFresh(fetchedAtMs, maxAgeMs) {
    if (!fetchedAtMs) return false;
    return (nowMs() - fetchedAtMs) <= (maxAgeMs || DEFAULT_MAX_AGE_MS);
  }

  function loadDataCache() {
    var raw = sessionStorage.getItem(DATA_KEY);
    if (!raw) return null;
    return safeJsonParse(raw);
  }

  function saveDataCache(obj) {
    try {
      sessionStorage.setItem(DATA_KEY, JSON.stringify(obj));
    } catch (_e) {
      // If storage is full or disabled, we silently continue without caching.
    }
  }

  function loadState() {
    var raw = sessionStorage.getItem(STATE_KEY);
    if (!raw) return null;
    return safeJsonParse(raw);
  }

  function saveState(state) {
    try {
      sessionStorage.setItem(STATE_KEY, JSON.stringify(state));
    } catch (_e) {}
  }

  /**
   * Build a relative URL to the API.
   *
   * When the UI is served by FastAPI (recommended), `/v1/...` is same-origin.
   */
  function apiUrl(pathAndQuery) {
    if (!pathAndQuery) return '/v1/chart';
    if (pathAndQuery.charAt(0) === '/') return pathAndQuery;
    return '/' + pathAndQuery;
  }

  async function fetchChartMax(includeKeys) {
    var keys = Array.isArray(includeKeys) && includeKeys.length ? includeKeys : DEFAULT_INCLUDE;
    var url = apiUrl('v1/chart?range=MAX&include=' + encodeURIComponent(keys.join(',')));

    var res = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!res.ok) {
      var text = await res.text().catch(function () { return ''; });
      throw new Error('API error ' + res.status + ' for ' + url + (text ? ('\n' + text) : ''));
    }

    return await res.json();
  }

  /**
   * Compute an ISO date string for (today - days).
   * Used for client-side slicing.
   */
  function isoDateMinusDays(days) {
    var d = new Date();
    d.setHours(0, 0, 0, 0);
    d.setDate(d.getDate() - days);
    var yyyy = d.getFullYear();
    var mm = String(d.getMonth() + 1).padStart(2, '0');
    var dd = String(d.getDate()).padStart(2, '0');
    return yyyy + '-' + mm + '-' + dd;
  }

  function rangeToDays(range) {
    switch (String(range || '').toUpperCase()) {
      case '1M': return 30;
      case '3M': return 90;
      case '6M': return 180;
      case '1Y': return 365;
      case '5Y': return 1825;
      case 'MAX': return null;
      default: return 365;
    }
  }

  /**
   * Slice a list of `{date, value}` points by a preset range.
   *
   * This assumes `date` is `YYYY-MM-DD`.
   */
  function slicePointsByRange(points, range) {
    if (!Array.isArray(points)) return [];

    var days = rangeToDays(range);
    if (days == null) return points;

    var startIso = isoDateMinusDays(days);
    return points.filter(function (p) {
      return p && p.date && p.date >= startIso;
    });
  }

  function lastFiniteValue(points) {
    if (!Array.isArray(points)) return null;
    for (var i = points.length - 1; i >= 0; i--) {
      var v = points[i] && points[i].value;
      if (typeof v === 'number' && isFinite(v)) return v;
    }
    return null;
  }

  /**
   * Public store API (attached to `window.DashboardStore`).
   */
  var DashboardStore = {
    /**
     * Ensure data is present in sessionStorage (fetch if missing/stale).
     *
     * @param {Object} opts
     * @param {number} [opts.maxAgeMs] - cache TTL
     * @param {string[]} [opts.include] - datasets to include in the one-shot call
     * @returns {Promise<Object>} chart response JSON
     */
    ensureData: async function (opts) {
      opts = opts || {};
      var maxAgeMs = opts.maxAgeMs || DEFAULT_MAX_AGE_MS;

      var cached = loadDataCache();
      if (cached && cached.fetchedAtMs && cached.chart && isFresh(cached.fetchedAtMs, maxAgeMs)) {
        return cached.chart;
      }

      var chart = await fetchChartMax(opts.include);
      saveDataCache({
        fetchedAtMs: nowMs(),
        chart: chart
      });
      return chart;
    },

    /**
     * Get the cached chart response without triggering a fetch.
     * Returns null if missing.
     */
    peekData: function () {
      var cached = loadDataCache();
      return cached && cached.chart ? cached.chart : null;
    },

    /**
     * Get / set UI state shared between pages (range, selected market, ...)
     */
    getState: function () {
      var s = loadState() || {};
      return {
        range: s.range || '1Y',
        marketId: s.marketId || 'none'
      };
    },

    setState: function (partial) {
      var current = this.getState();
      var next = {
        range: (partial && partial.range) ? partial.range : current.range,
        marketId: (partial && partial.marketId != null) ? partial.marketId : current.marketId
      };
      saveState(next);
      return next;
    },

    /**
     * Convenience helpers used by pages.
     */
    slicePointsByRange: slicePointsByRange,
    lastFiniteValue: lastFiniteValue,
    DEFAULT_INCLUDE: DEFAULT_INCLUDE.slice()
  };

  window.DashboardStore = DashboardStore;
})();
