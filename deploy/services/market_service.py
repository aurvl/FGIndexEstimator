import os
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
from utils import get_yf_close
from deploy.services.fgi_service import get_range_dates, CACHE_DIR

MARKET_REGISTRY = {
    "sp500": {"label": "S&P 500", "ticker": "^GSPC"},
    "nasdaq": {"label": "NASDAQ Composite", "ticker": "^IXIC"},
    "cac40": {"label": "CAC 40", "ticker": "^FCHI"},
    "msciworld": {"label": "MSCI World", "ticker": "^990100-USD-STRD"},
}

def build_market_cache_key(market_id: str, range_name: str, end_date: str) -> str:
    return f"market_{market_id}_{range_name}_{end_date}.parquet"

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
    key = build_market_cache_key(market_id, range_name, end_date)
    series = read_cache(key)
    if series is not None:
        # If read_cache returned a DataFrame column as Series, ensure it's a 1-D Series
        if isinstance(series, pd.DataFrame):
            if "value" in series.columns:
                series = series["value"]
            else:
                # fallback to first column
                series = series.iloc[:, 0]
        # Ensure index is string for API output
        series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
        return series, start_date, end_date
    # Download full history, slice to range
    full_series = get_yf_close(ticker, start="1990-01-01")
    # Ensure index is datetime and sorted
    full_series = full_series.sort_index()
    mask = (full_series.index >= pd.to_datetime(start_date)) & (full_series.index <= pd.to_datetime(end_date))
    series = full_series.loc[mask]
    # If result is DataFrame, try to reduce to single Series
    if isinstance(series, pd.DataFrame):
        if "Close" in series.columns and series.shape[1] == 1:
            series = series.iloc[:, 0]
        elif series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            # unexpected multi-column result -> take first column
            series = series.iloc[:, 0]
    # Index as string for JSON
    series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
    write_cache(key, series)
    return series, start_date, end_date
