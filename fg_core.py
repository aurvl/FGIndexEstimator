import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from utils import build_raw_indicators, compute_components, compute_fear_greed
import warnings

try:
    from pandas.errors import Pandas4Warning
except ImportError:
    Pandas4Warning = FutureWarning

warnings.filterwarnings(
    "ignore",
    category=Pandas4Warning,
    message="Timestamp.utcnow is deprecated and will be removed in a future version. Use Timestamp.now\\('UTC'\\) instead.",
)

# Constants
CALIB_MODEL_PATH = Path("models") / "fg_linear_weights.pkl"
load_dotenv()
API_KEY = os.getenv("API_KEY", "")

def load_model(
    path: str | Path,
) -> tuple[dict[str, float] | None, float]:
    path = Path(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    intercept = float(model.intercept_)
    feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
    if feature_names is None:
        feature_names = [f"score_{i}" for i in range(len(model.coef_))]
    coefs = pd.Series(model.coef_, index=feature_names, dtype=float)
    weights = {}
    for col, w in coefs.items():
        base = col.replace("score_", "")
        weights[base] = float(w)
    return weights, intercept

def get_fgi_estimation(
    start_date: str | None = None,
    end_date: str | None = None,
    with_components: bool = True,
    use_calibrated_model: bool = True,
    history_start: str = "2005-01-01",
    data_dir: str = "data/fred_cache",
    min_periods: int = 252,
    window: int | None = 1260,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Pipeline d'estimation du score Fear & Greed "FG_like".
    ...existing docstring...
    """
    if end_date is None:
        end_ts = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
    else:
        end_ts = pd.to_datetime(end_date)
    if start_date is None:
        start_ts = end_ts - pd.DateOffset(years=1)
    else:
        start_ts = pd.to_datetime(start_date)
    if use_calibrated_model:
        weights, intercept = load_model(CALIB_MODEL_PATH)
    else:
        weights, intercept = None, 0.0
    df_raw = build_raw_indicators(
        api_key_fred=API_KEY,
        data_dir=data_dir,
        start=history_start,
    )
    components = compute_components(df_raw)
    scores, fg_col_name = compute_fear_greed(
        components=components,
        min_periods=min_periods,
        window=window,
        lower_q=lower_q,
        upper_q=upper_q,
        weights=weights,
        intercept=intercept,
    )
    if fg_col_name == "FG_est_mean":
        print("[INFO]: Using simple average for F&G estimation.")
    else:
        print("[INFO]: Using calibrated linear model for F&G estimation.")
    fg_like = scores[fg_col_name].rename("FG_estimation")
    score_cols = scores.drop(columns=[fg_col_name], errors="ignore").add_prefix("score_")
    result_full = pd.concat(
        [components.add_prefix("raw_"), score_cols, fg_like],
        axis=1,
    )
    result_full = result_full.dropna(subset=["FG_estimation"])
    mask = (result_full.index >= start_ts) & (result_full.index <= end_ts)
    result = result_full.loc[mask].copy()
    if not with_components:
        return result[["FG_estimation"]]
    return result
