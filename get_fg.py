import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from fg_core import get_fgi_estimation
import warnings
import tempfile
from utils import get_yf_close

try:
    # Single source of truth for tickers
    from deploy.services.market_service import MARKET_REGISTRY
except Exception:
    MARKET_REGISTRY = {}

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
# 
CALIB_MODEL_PATH = Path("models") / "fg_linear_weights.pkl"
load_dotenv()
API_KEY = os.getenv("API_KEY", "")

MANUAL_CACHE_DIR = Path("data") / "cache_api"
RECOMPUTE_LOOKBACK_DAYS = 7
WARMUP_DAYS = 2000  # enough for window=1260 + MA/pct-change buffers


def _fgi_cache_key(range_name: str, end_date: str, with_components: bool, use_calibrated_model: bool) -> str:
    return f"fgi_{range_name}_{end_date}_components{int(with_components)}_calib{int(use_calibrated_model)}.parquet"


def _find_latest_fgi_cache(range_name: str, with_components: bool, use_calibrated_model: bool) -> tuple[Path, str] | None:
    """Return (path, end_date_str) for the latest matching cache file, or None."""
    pattern = f"fgi_{range_name}_*_components{int(with_components)}_calib{int(use_calibrated_model)}.parquet"
    candidates = sorted(MANUAL_CACHE_DIR.glob(pattern))
    if not candidates:
        return None
    # Extract end_date from filename: fgi_{range}_{end}_componentsX_calibY.parquet
    best = None
    best_end = None
    for p in candidates:
        stem = p.stem
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        # parts: [fgi, RANGE, YYYY-MM-DD, componentsX, calibY]
        end = parts[2]
        try:
            pd.to_datetime(end)
        except Exception:
            continue
        if best_end is None or end > best_end:
            best = p
            best_end = end
    if best is None or best_end is None:
        return None
    return best, best_end


def _market_cache_key(market_id: str, range_name: str, end_date: str) -> str:
    return f"market_{market_id}_{range_name}_{end_date}.parquet"


def _find_latest_market_cache(market_id: str, range_name: str) -> tuple[Path, str] | None:
    pattern = f"market_{market_id}_{range_name}_*.parquet"
    candidates = sorted(MANUAL_CACHE_DIR.glob(pattern))
    if not candidates:
        return None
    best = None
    best_end = None
    for p in candidates:
        parts = p.stem.split("_")
        # [market, id, RANGE, YYYY-MM-DD]
        if len(parts) < 4:
            continue
        end = parts[3]
        try:
            pd.to_datetime(end)
        except Exception:
            continue
        if best_end is None or end > best_end:
            best = p
            best_end = end
    if best is None or best_end is None:
        return None
    return best, best_end


def _read_market_cache_as_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    try:
        obj = pd.read_parquet(path)
    except Exception:
        return None
    if isinstance(obj, pd.Series):
        s = obj
    elif isinstance(obj, pd.DataFrame):
        if "value" in obj.columns:
            s = obj["value"]
        elif obj.shape[1] == 1:
            s = obj.iloc[:, 0]
        else:
            return None
    else:
        return None
    idx = pd.to_datetime(s.index, errors="coerce")
    s.index = idx
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return s


def _write_market_cache_from_series(series: pd.Series, path: Path) -> None:
    # store as DataFrame with string date index + 'value' column (compatible with API reader)
    df = pd.DataFrame({"value": series.values}, index=pd.to_datetime(series.index).strftime("%Y-%m-%d"))
    df.index.name = "date"
    _atomic_write_parquet(df, path)


def _read_manual_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        # common case if index got persisted as strings
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=str(path.parent),
        prefix=path.stem + ".tmp.",
        suffix=path.suffix,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path)
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

def load_model(
    path: str | Path,
) -> tuple[dict[str, float] | None, float]:
    """
    Charge un modèle linéaire scikit-learn sauvegardé en pickle
    et en extrait l'intercept et les poids par composante.

    Hypothèses :
    - modèle avec attributs `coef_` et `intercept_`
    - entraîné sur des colonnes de type 'score_momentum_spx', etc.
      => on retire le préfixe 'score_' pour revenir aux noms bruts
         attendus par `compute_fear_greed`.
    """
    path = Path(path)
    if not path.exists():
        # pas de modèle : on utilisera la moyenne simple
        return None, 0.0

    with open(path, "rb") as f:
        model = pickle.load(f)

    intercept = float(model.intercept_)

    # Récupération des noms de features utilisés à l'entraînement
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        # fallback : à adapter si tu stockes les noms ailleurs
        raise ValueError(
            "The loaded model has no `feature_names_in_`. "
            "Save the model with this attribute or adapt the loader."
        )

    coefs = pd.Series(model.coef_, index=feature_names, dtype=float)

    # 'score_momentum_spx' -> 'momentum_spx', etc.
    weights = {}
    for col, w in coefs.items():
        base = col.replace("score_", "")
        weights[base] = float(w)

    return weights, intercept


if __name__ == "__main__":
    update_cache_input = input(
        "Update the manual API cache in data/cache_api? (y/n) [default: y]: "
    ).strip().lower()
    update_cache = update_cache_input != "n"

    calib_input = input("Use calibrated model? (y/n) [default: y]: ").strip().lower()
    use_calibrated_model = calib_input != "n"

    end = input("End date (YYYY-MM-DD) [default: today]: ").strip()
    end_date = end if end else None

    if update_cache:
        MANUAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        range_name = "MAX"

        comp_cache_input = input("Also build components cache file? (y/n) [default: y]: ").strip().lower()
        build_components_cache = comp_cache_input != "n"

        end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.today().normalize()
        end_str = end_dt.strftime("%Y-%m-%d")

        def _update_one(with_components_flag: bool) -> Path:
            latest = _find_latest_fgi_cache(range_name, with_components_flag, use_calibrated_model)
            if latest is None:
                start = input("Start date (YYYY-MM-DD) [default: 2005-01-01]: ").strip()
                start_date_local = start if start else "2005-01-01"
                cache_df_local = None
            else:
                latest_path, _latest_end = latest
                cache_df_local = _read_manual_cache(latest_path)
                if cache_df_local is None or cache_df_local.empty:
                    start_date_local = "2005-01-01"
                    cache_df_local = None
                else:
                    last_date = pd.to_datetime(cache_df_local.index.max()).normalize()
                    start_date_local = (last_date - pd.Timedelta(days=RECOMPUTE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

            warmup_start = (pd.to_datetime(start_date_local) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

            new_df_local = get_fgi_estimation(
                start_date=start_date_local,
                end_date=end_str,
                with_components=with_components_flag,
                use_calibrated_model=use_calibrated_model,
                history_start=warmup_start,
            )

            if cache_df_local is None:
                merged_local = new_df_local
            else:
                cutoff = pd.to_datetime(start_date_local)
                merged_local = pd.concat([cache_df_local.loc[cache_df_local.index < cutoff], new_df_local], axis=0)
                merged_local = merged_local[~merged_local.index.duplicated(keep="last")].sort_index()

            out_key = _fgi_cache_key(range_name, end_str, with_components_flag, use_calibrated_model)
            out_path = MANUAL_CACHE_DIR / out_key
            _atomic_write_parquet(merged_local, out_path)

            # cleanup older versions for same pattern (keep only the newest file we just wrote)
            pattern = f"fgi_{range_name}_*_components{int(with_components_flag)}_calib{int(use_calibrated_model)}.parquet"
            for p in MANUAL_CACHE_DIR.glob(pattern):
                if p != out_path:
                    try:
                        p.unlink()
                    except OSError:
                        pass

            print(f"[OK] Cache updated: {out_path}")
            return out_path

        # Always build the light cache (FG only)
        _update_one(with_components_flag=False)
        if build_components_cache:
            _update_one(with_components_flag=True)

        market_cache_input = input("Also update market MAX caches? (y/n) [default: y]: ").strip().lower()
        update_markets = market_cache_input != "n"
        if update_markets and MARKET_REGISTRY:
            for market_id, info in MARKET_REGISTRY.items():
                ticker = info.get("ticker")
                if not ticker:
                    continue
                latest = _find_latest_market_cache(market_id, range_name)
                if latest is None:
                    cached_series = None
                    recompute_start_ts = pd.Timestamp("1990-01-01")
                else:
                    latest_path, _latest_end = latest
                    cached_series = _read_market_cache_as_series(latest_path)
                    if cached_series is None or cached_series.empty:
                        cached_series = None
                        recompute_start_ts = pd.Timestamp("1990-01-01")
                    else:
                        last_date = pd.to_datetime(cached_series.index.max()).normalize()
                        recompute_start_ts = last_date - pd.Timedelta(days=RECOMPUTE_LOOKBACK_DAYS)

                fresh = get_yf_close(ticker, start=recompute_start_ts.strftime("%Y-%m-%d"))
                if isinstance(fresh, pd.DataFrame):
                    fresh = fresh.iloc[:, 0]
                fresh = pd.Series(fresh, dtype=float)
                fresh.index = pd.to_datetime(fresh.index, errors="coerce")
                fresh = fresh[~fresh.index.isna()].sort_index()
                fresh = fresh.loc[fresh.index <= end_dt]

                if cached_series is None:
                    merged = fresh
                else:
                    cutoff = recompute_start_ts
                    merged = pd.concat([cached_series.loc[cached_series.index < cutoff], fresh], axis=0)
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

                out_key = _market_cache_key(market_id, range_name, end_str)
                out_path = MANUAL_CACHE_DIR / out_key
                _write_market_cache_from_series(merged, out_path)

                # cleanup older versions
                pattern = f"market_{market_id}_{range_name}_*.parquet"
                for p in MANUAL_CACHE_DIR.glob(pattern):
                    if p != out_path:
                        try:
                            p.unlink()
                        except OSError:
                            pass

                print(f"[OK] Market cache updated: {out_path}")
    else:
        start = input("Start date (YYYY-MM-DD) [default: 1 year ago]: ").strip()
        with_components_input = input("Include component components? (y/n) [default: y]: ").strip().lower()
        with_components = with_components_input != "n"
        start_date = start if start else None

        fg_df = get_fgi_estimation(
            start_date=start_date,
            end_date=end_date,
            with_components=with_components,
            use_calibrated_model=use_calibrated_model,
        )
        print(fg_df.head())