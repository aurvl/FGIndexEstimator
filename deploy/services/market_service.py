import os
import pandas as pd
from datetime import datetime # noqa
from typing import Optional, Tuple # noqa
from utils import get_yf_close
from deploy.services.fgi_service import get_range_dates, CACHE_DIR

MANUAL_CACHE_LOOKBACK_DAYS = 7

MARKET_REGISTRY = {
    "sp500": {"label": "S&P 500", "ticker": "^GSPC"},
    "nasdaq": {"label": "NASDAQ Composite", "ticker": "^IXIC"},
    "cac40": {"label": "CAC 40", "ticker": "^FCHI"},
    "msciworld": {"label": "MSCI World", "ticker": "^990100-USD-STRD"},
}

def build_market_cache_key(market_id: str, range_name: str, end_date: str) -> str:
    return f"market_{market_id}_{range_name}_{end_date}.parquet"


def _find_latest_market_max_cache(market_id: str) -> tuple[pd.Series, str] | None:
    """Find latest market_{id}_MAX_YYYY-MM-DD.parquet in CACHE_DIR."""
    suffix = ".parquet"
    prefix = f"market_{market_id}_MAX_"
    try:
        files = [f for f in os.listdir(CACHE_DIR) if f.startswith(prefix) and f.endswith(suffix)]
    except FileNotFoundError:
        files = []
    best_file = None
    best_end = None
    for name in files:
        stem = name[:-len(suffix)]
        parts = stem.split("_")
        # [market, id, MAX, YYYY-MM-DD]
        if len(parts) < 4:
            continue
        end_str = parts[3]
        try:
            pd.to_datetime(end_str)
        except Exception:
            continue
        if best_end is None or end_str > best_end:
            best_end = end_str
            best_file = name
    if best_file is None or best_end is None:
        return None
    cached = read_cache(best_file)
    if cached is None:
        return None
    s = cached
    # normalize index to datetime
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s, best_end

def cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key)

def read_cache(key: str) -> Optional[pd.Series]:
    path = cache_path(key)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            # If parquet contains a DataFrame with a 'value' column, return it as Series
            if "value" in df.columns:
                s = df["value"]
                # keep index from parquet if present
                if df.index is not None:
                    s.index = df.index
                return s
            # If single-column DataFrame, return that column
            if df.shape[1] == 1:
                col = df.columns[0]
                s = df[col]
                if df.index is not None:
                    s.index = df.index
                return s
            return None
        except Exception:
            return None
    return None

def write_cache(key: str, series: pd.Series):
    import numpy as np
    path = cache_path(key)
    # Ensure index is datetime before formatting
    idx = series.index
    if not hasattr(idx, "strftime"):
        idx = pd.to_datetime(idx)
    date_strs = np.array(idx.strftime("%Y-%m-%d"))
    values = np.array(series.values).flatten()
    # Defensive: ensure both are 1D and same length
    if date_strs.ndim != 1 or values.ndim != 1 or len(date_strs) != len(values):
        raise ValueError(f"Cache write: date_strs shape {date_strs.shape}, values shape {values.shape}, lengths: {len(date_strs)}, {len(values)}")
    df = pd.DataFrame({"date": date_strs, "value": values})
    df.set_index("date", inplace=True)
    df.to_parquet(path)

def get_market_series(market_id: str, range_name: str, end_date: Optional[str]):
    if market_id not in MARKET_REGISTRY:
        raise ValueError(f"Unknown market_id: {market_id}")
    ticker = MARKET_REGISTRY[market_id]["ticker"]
    start_date, end_date = get_range_dates(range_name, end_date)

    # Prefer the latest MAX manual cache, then slice to requested range.
    cached_info = _find_latest_market_max_cache(market_id)
    if cached_info is not None:
        cached_series, _cached_end = cached_info
        end_ts = pd.to_datetime(end_date)
        last_cached = pd.to_datetime(cached_series.index.max()).normalize()

        if end_ts.normalize() <= last_cached:
            merged = cached_series
        else:
            recompute_start_ts = last_cached - pd.Timedelta(days=MANUAL_CACHE_LOOKBACK_DAYS)
            fresh = get_yf_close(ticker, start=recompute_start_ts.strftime("%Y-%m-%d"))
            if isinstance(fresh, pd.DataFrame):
                fresh = fresh.iloc[:, 0]
            fresh = pd.Series(fresh, dtype=float)
            fresh.index = pd.to_datetime(fresh.index, errors="coerce")
            fresh = fresh[~fresh.index.isna()].sort_index()
            fresh = fresh.loc[fresh.index <= end_ts]

            cutoff = recompute_start_ts
            merged = pd.concat([cached_series.loc[cached_series.index < cutoff], fresh], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        mask = (merged.index >= pd.to_datetime(start_date)) & (merged.index <= pd.to_datetime(end_date))
        out = merged.loc[mask]
        out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
        return out, start_date, end_date

    # Fallback to legacy per-request cache
    key = build_market_cache_key(market_id, range_name, end_date)
    series = read_cache(key)
    if series is not None:
        if isinstance(series, pd.DataFrame):
            if "value" in series.columns:
                series = series["value"]
            else:
                series = series.iloc[:, 0]
        series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
        return series, start_date, end_date

    full_series = get_yf_close(ticker, start="1990-01-01")
    full_series = full_series.sort_index()
    mask = (full_series.index >= pd.to_datetime(start_date)) & (full_series.index <= pd.to_datetime(end_date))
    series = full_series.loc[mask]
    if isinstance(series, pd.DataFrame):
        if "Close" in series.columns and series.shape[1] == 1:
            series = series.iloc[:, 0]
        elif series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            series = series.iloc[:, 0]
    series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
    write_cache(key, series)
    return series, start_date, end_date
