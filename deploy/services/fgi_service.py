import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional

CACHE_DIR = os.path.join('data', 'cache_api')
MANUAL_CACHE_LOOKBACK_DAYS = 7
WARMUP_DAYS = 2000  # enough for window=1260 + MA/pct-change buffers

RANGE_PRESETS = {
    '1M': 30,
    '3M': 90,
    '6M': 180,
    '1Y': 365,
    '5Y': 1825,
    'MAX': None
}

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _find_latest_fgi_max_cache(with_components: bool, use_calibrated_model: bool) -> tuple[pd.DataFrame, str] | None:
    """Find latest fgi_MAX_YYYY-MM-DD_componentsX_calibY.parquet in CACHE_DIR."""
    pattern = f"fgi_MAX_*_components{int(with_components)}_calib{int(use_calibrated_model)}.parquet"
    try:
        files = [f for f in os.listdir(CACHE_DIR) if f.startswith("fgi_MAX_") and f.endswith(f"_components{int(with_components)}_calib{int(use_calibrated_model)}.parquet")]
    except FileNotFoundError:
        files = []
    best_file = None
    best_end = None
    for name in files:
        parts = name.replace(".parquet", "").split("_")
        # [fgi, MAX, YYYY-MM-DD, componentsX, calibY]
        if len(parts) < 5:
            continue
        end_str = parts[2]
        try:
            pd.to_datetime(end_str)
        except Exception:
            continue
        if best_end is None or end_str > best_end:
            best_end = end_str
            best_file = name
    if best_file is None or best_end is None:
        return None
    path = os.path.join(CACHE_DIR, best_file)
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df, best_end
    except Exception:
        return None

def get_range_dates(range_name: str, end_date: Optional[str] = None) -> Tuple[str, str]:
    if end_date is None:
        end_dt = datetime.today()
    else:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    days = RANGE_PRESETS.get(range_name.upper())
    if days is None:
        start_dt = datetime(2000, 1, 1)  # earliest possible
    else:
        start_dt = end_dt - timedelta(days=days)
    return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

def build_cache_key(range_name: str, end_date: str, with_components: bool, use_calibrated_model: bool) -> str:
    return f"fgi_{range_name}_{end_date}_components{int(with_components)}_calib{int(use_calibrated_model)}.parquet"

def cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key)

def read_cache(key: str) -> Optional[pd.DataFrame]:
    path = cache_path(key)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None

def write_cache(key: str, df: pd.DataFrame):
    path = cache_path(key)
    df.to_parquet(path)

def get_fgi_series(range_name: str, end_date: Optional[str], with_components: bool, use_calibrated_model: bool):
    from fg_core import get_fgi_estimation  # import here to avoid circular
    ensure_cache_dir()
    start_date, end_date = get_range_dates(range_name, end_date)

    # 1) Load the latest MAX cache file (versioned in the repo and updated manually)
    cached_info = _find_latest_fgi_max_cache(with_components=with_components, use_calibrated_model=use_calibrated_model)
    cached = cached_info[0] if cached_info else None

    end_ts = pd.to_datetime(end_date)
    start_ts = pd.to_datetime(start_date)

    if cached is None:
        # Fallback: compute directly (slower). Recommended: run `python -m get_fg` and update the manual cache.
        df = get_fgi_estimation(
            start_date=start_date,
            end_date=end_date,
            with_components=with_components,
            use_calibrated_model=use_calibrated_model,
        )
        return df, start_date, end_date

    # 2) Incremental recompute from (last_cached_date - 7d) to end_date
    last_cached = pd.to_datetime(cached.index.max()).normalize()
    if end_ts.normalize() <= last_cached:
        merged = cached
    else:
        recompute_start_ts = last_cached - pd.Timedelta(days=MANUAL_CACHE_LOOKBACK_DAYS)
        recompute_start = recompute_start_ts.strftime("%Y-%m-%d")
        warmup_start = (recompute_start_ts - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

        fresh = get_fgi_estimation(
            start_date=recompute_start,
            end_date=end_date,
            with_components=with_components,
            use_calibrated_model=use_calibrated_model,
            history_start=warmup_start,
        )

        cutoff = pd.to_datetime(recompute_start)
        merged = pd.concat([cached.loc[cached.index < cutoff], fresh], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    # 3) Slice to requested range
    merged = merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()
    if not with_components:
        # Keep only the main index column
        if "FG_estimation" in merged.columns:
            merged = merged[["FG_estimation"]]
        elif "FG_like" in merged.columns:
            merged = merged[["FG_like"]]
    return merged, start_date, end_date
