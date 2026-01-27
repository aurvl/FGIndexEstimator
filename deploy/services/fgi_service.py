import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional

CACHE_DIR = os.path.join('data', 'cache_api')

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
    key = build_cache_key(range_name, end_date, with_components, use_calibrated_model)
    df = read_cache(key)
    if df is None:
        df = get_fgi_estimation(start_date=start_date, end_date=end_date, with_components=with_components, use_calibrated_model=use_calibrated_model)
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError('get_fgi_estimation did not return a DataFrame')
        write_cache(key, df)
    return df, start_date, end_date
