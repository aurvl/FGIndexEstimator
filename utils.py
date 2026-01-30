import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from threading import Lock

logger = logging.getLogger(__name__)
_YF_LOCK = Lock()

# -----------------------------------
# 1. fonction FRED
# -----------------------------------
def _get_data_fred(path, api_key, series_id, rename):
    path = str(path)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obs = data.get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            return pd.DataFrame(columns=[rename])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[['date','value']].rename(columns={'value': rename})
        df = df.set_index("date").sort_index()
        return df
    else:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
            }
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            obs = data.get("observations", [])
            df = pd.DataFrame(obs)
            if df.empty:
                return pd.DataFrame(columns=[rename])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[['date','value']].rename(columns={'value': rename})
            df = df.set_index("date").sort_index()
            return df
        except Exception:
            # Série inconnue / API key / etc. -> retourne vide (on gérera via ffill/skipna)
            return pd.DataFrame(columns=[rename])

# -----------------------------------
# 2. Percentile -> score between 0–100
# -----------------------------------
def percentile_score(
    series: pd.Series,
    invert: bool = False,
    min_periods: int = 252,
    window: int | None = 1260,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """
    Transforme une série en score 0–100 via rang percentile, sans look-ahead.

    - min_periods : nb min d'observations avant de commencer à produire un score
    - window      : taille max de la fenêtre (en points) pour estimer la CDF.
                    None = fenêtre "expanding" (toute l'historique).
                    1260 ~ 5 ans de jours ouvrés.
    - lower_q / upper_q : quantiles pour winsoriser l'historique (robuste aux gros
                          outliers tout en gardant un gradient dans les extrêmes).

    Retourne une série alignée sur 'series' avec des scores dans [0, 100].
    """

    s = series.astype(float).copy()
    scores = pd.Series(index=s.index, dtype=float)

    for i, (idx, val) in enumerate(s.items()):
        if pd.isna(val):
            continue

        # Fenêtre historique : rolling (window) ou expanding si None
        if window is None:
            hist = s.iloc[: i + 1].dropna()
        else:
            start = max(0, i - window + 1)
            hist = s.iloc[start : i + 1].dropna()

        if len(hist) < min_periods:
            continue

        # Winsorisation de l'historique pour limiter l'effet des outliers extrêmes
        if lower_q is not None and upper_q is not None:
            q_low, q_high = hist.quantile([lower_q, upper_q])
            hist = hist.clip(q_low, q_high)

        # La dernière valeur de hist est celle de t (val)
        rank = hist.rank(pct=True).iloc[-1]  # entre 0 et 1
        score = rank * 100.0
        if invert:
            score = 100.0 - score

        scores.loc[idx] = score

    return scores

# -----------------------------------
# 3. Récupération des données marché
# -----------------------------------
def get_yf_close(ticker: str, start: str = "1990-01-01") -> pd.Series:
    with _YF_LOCK:
        data = yf.download(
            ticker,
            start=start,
            progress=False,
            auto_adjust=False,
            threads=False,   # CRUCIAL
        )

    if data is None or data.empty:
        return pd.Series(dtype=float)

    # yfinance renvoie souvent colonnes: Open High Low Close Adj Close Volume
    if "Adj Close" in data.columns:
        s = data["Adj Close"].copy()
    else:
        s = data["Close"].copy()

    s.name = ticker
    return s

def build_raw_indicators(
    api_key_fred: str,
    data_dir: str = "data_fred",
    start: str = "1990-01-01",
) -> pd.DataFrame:
    """
    Récupère et assemble les séries brutes nécessaires à l'indice :
      - SPX, VIX, TLT, RSP, HYG via yfinance (closes)
      - HY_spread via FRED (BAMLH0A0HYM2)
      - put_call via FRED (PUTCALL, si dispo)

    Retour:
      DataFrame sur calendrier business (B), index trié, colonnes stables :
        ['spx', 'vix', 'tlt', 'rsp', 'hyg', 'HY_spread', 'put_call'] (selon dispo)
    """

    os.makedirs(data_dir, exist_ok=True)

    # --- 1) mapping ticker -> nom de colonne stable ---
    market_map = {
        "^GSPC": "spx",
        "^VIX":  "vix",
        "TLT":   "tlt",
        "RSP":   "rsp",
        "HYG":   "hyg",
    }

    # --- 2) télécharger chaque série séparément et la renommer ---
    market_series = []
    for ticker, colname in market_map.items():
        s = get_yf_close(ticker, start=start)

        if s is None or len(s) == 0:
            logger.warning("yfinance returned empty series for %s", ticker)
            continue

        # s peut être Series ou DataFrame selon ta fonction -> on force Series
        if isinstance(s, pd.DataFrame):
            # si c'est un DF, on prend la 1ere colonne
            if s.shape[1] == 0:
                logger.warning("Empty DataFrame for %s", ticker)
                continue
            s = s.iloc[:, 0]

        s = pd.Series(s, name=colname)

        # normaliser l'index : datetime, tz-naive, trié, doublons gérés
        idx = pd.to_datetime(s.index, errors="coerce")
        s.index = idx
        s = s[~s.index.isna()]
        s = s.sort_index()

        # si index tz-aware -> tz-naive
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_convert(None)

        # si doublons de dates -> on garde le dernier
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep="last")]

        # force float
        s = pd.to_numeric(s, errors="coerce").astype(float)

        market_series.append(s)

    # --- 3) concat marché avec colonnes uniques garanties ---
    if not market_series:
        raise RuntimeError("No market series could be loaded from yfinance.")

    df = pd.concat(market_series, axis=1)

    # sécurité extra : aucune colonne dupliquée
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        logger.warning("Duplicate market columns detected (dropping): %s", dupes)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # --- 4) calendrier business + ffill global ---
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    # calendrier business, puis ffill (évite les trous de cotation)
    df = df.asfreq("B")
    df = df.ffill()

    # --- 5) FRED (join safe) ---
    # HY spread (BofA High Yield Option-Adjusted Spread)
    hy_path = os.path.join(data_dir, "BAMLH0A0HYM2.json")
    hy_spread = _get_data_fred(
        hy_path,
        api_key_fred,
        series_id="BAMLH0A0HYM2",
        rename="HY_spread",
    )

    if hy_spread is not None and len(hy_spread) > 0:
        hy_spread.index = pd.to_datetime(hy_spread.index, errors="coerce")
        hy_spread = hy_spread[~hy_spread.index.isna()].sort_index()
        if getattr(hy_spread.index, "tz", None) is not None:
            hy_spread.index = hy_spread.index.tz_convert(None)
        hy_spread = hy_spread[~hy_spread.index.duplicated(keep="last")]

        # align sur df (B) + fill
        hy_spread = hy_spread.reindex(df.index).ffill()
        df = df.join(hy_spread, how="left")

    # Put/Call ratio (si dispo FRED)
    pc_path = os.path.join(data_dir, "PUTCALL.json")
    put_call = _get_data_fred(
        pc_path,
        api_key_fred,
        series_id="PUTCALL",
        rename="put_call",
    )

    if put_call is not None and len(put_call) > 0:
        put_call.index = pd.to_datetime(put_call.index, errors="coerce")
        put_call = put_call[~put_call.index.isna()].sort_index()
        if getattr(put_call.index, "tz", None) is not None:
            put_call.index = put_call.index.tz_convert(None)
        put_call = put_call[~put_call.index.duplicated(keep="last")]

        put_call = put_call.reindex(df.index).ffill()
        df = df.join(put_call, how="left")

    # --- 6) dernière passe de nettoyage ---
    # float + aucun doublon
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        logger.warning("Duplicate columns detected (dropping): %s", dupes)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # enlever lignes entièrement vides (rare après ffill)
    df = df.dropna(how="all")

    return df

# -----------------------------------
# 4. Calcul des 7 composantes brutes
# -----------------------------------
def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les composantes brutes du Fear & Greed.

    Colonnes acceptées (alias) :
      - SPX : 'spx' ou '^GSPC'
      - VIX : 'vix' ou '^VIX'
      - TLT : 'tlt' ou 'TLT'
      - RSP : 'rsp' ou 'RSP'
      - HYG : 'hyg' ou 'HYG'
      - FRED : 'HY_spread' (optionnel), 'put_call' (optionnel)

    Retourne un DataFrame indexé comme df, avec :
      - momentum_spx         : (SPX - MA125) / MA125
      - strength_proxy       : (SPX - MA200) / MA200
      - breadth_rsp_spx      : ret60(RSP) - ret60(SPX)
      - junk_bond_mom_20d    : ret20(HYG)
      - hy_spread            : spread HY brut (à inverser au scoring)
      - vix_rel              : (VIX - MA50) / MA50
      - safe_haven_20d       : ret20(SPX) - ret20(TLT)
      - put_call             : put/call brut (si dispo)
    """

    # --- resolve aliases ---
    aliases = {
        "spx": ["spx", "^GSPC"],
        "vix": ["vix", "^VIX"],
        "tlt": ["tlt", "TLT"],
        "rsp": ["rsp", "RSP"],
        "hyg": ["hyg", "HYG"],
    }

    def _pick_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_spx = _pick_col(aliases["spx"])
    col_vix = _pick_col(aliases["vix"])
    col_tlt = _pick_col(aliases["tlt"])
    col_rsp = _pick_col(aliases["rsp"])
    col_hyg = _pick_col(aliases["hyg"])

    missing = [name for name, col in {
        "spx": col_spx, "vix": col_vix, "tlt": col_tlt, "rsp": col_rsp, "hyg": col_hyg
    }.items() if col is None]

    if missing:
        raise KeyError(f"Missing required market series: {missing}. Available: {list(df.columns)}")

    # --- typed series (float), aligned on df.index ---
    spx = pd.to_numeric(df[col_spx], errors="coerce").astype(float)
    vix = pd.to_numeric(df[col_vix], errors="coerce").astype(float)
    tlt = pd.to_numeric(df[col_tlt], errors="coerce").astype(float)
    rsp = pd.to_numeric(df[col_rsp], errors="coerce").astype(float)
    hyg = pd.to_numeric(df[col_hyg], errors="coerce").astype(float)

    out = pd.DataFrame(index=df.index)

    # 1) Momentum SPX (MA125)
    ma125 = spx.rolling(125, min_periods=60).mean()
    out["momentum_spx"] = (spx - ma125) / ma125

    # 2) Strength proxy (MA200)
    ma200 = spx.rolling(200, min_periods=80).mean()
    out["strength_proxy"] = (spx - ma200) / ma200

    # 3) Breadth : surperformance 60j RSP vs SPX
    out["breadth_rsp_spx"] = rsp.pct_change(60) - spx.pct_change(60)

    # 4) Junk bond momentum : rendement 20j HYG
    out["junk_bond_mom_20d"] = hyg.pct_change(20)

    # 5) High-yield spread brut (optionnel)
    out["hy_spread"] = pd.to_numeric(df["HY_spread"], errors="coerce") if "HY_spread" in df.columns else np.nan

    # 6) VIX relatif : (VIX - MA50) / MA50
    ma_vix_50 = vix.rolling(50, min_periods=20).mean()
    out["vix_rel"] = (vix - ma_vix_50) / ma_vix_50

    # 7) Safe haven demand : ret20(SPX) - ret20(TLT)
    out["safe_haven_20d"] = spx.pct_change(20) - tlt.pct_change(20)

    # 8) Put/Call brut (optionnel)
    out["put_call"] = pd.to_numeric(df["put_call"], errors="coerce") if "put_call" in df.columns else np.nan

    return out


# ---------------------------------------------
# 5. Specs composantes & Fear/Greed calculation
# ---------------------------------------------
FG_COMPONENT_SPECS = [
    # plus grand = plus de greed
    ("momentum_spx",      False),
    ("strength_proxy",    False),
    ("breadth_rsp_spx",   False),
    ("safe_haven_20d",    False),
    ("junk_bond_mom_20d", False),

    # plus grand = plus de PEUR -> invert=True
    ("put_call",  True),      # + de puts = + de fear
    ("hy_spread", True),      # spread HY plus large = + de fear
    ("vix_rel",   True),      # VIX >> moyenne = + de fear
]

def compute_fear_greed(
    components: pd.DataFrame,
    min_periods: int = 252,
    window: int | None = 1260,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    weights: dict | None = None,
    intercept: float = 0.0,
):
    """
    Calcule les scores 0–100 pour chaque composante + un indice Fear/Greed agrégé.

    - components : DataFrame des composantes brutes (sortie de compute_components)
    - min_periods : nb min d'observations avant de sortir un score
    - window      : taille max de la fenêtre pour la CDF (None = expanding)
    - lower_q / upper_q : quantiles de winsorisation pour la CDF
    - weights     : dict {nom_composante: poids}. None -> moyenne simple.
    - intercept   : biais additif (si calibration sur un index externe).
    """

    scores = pd.DataFrame(index=components.index)

    for name, invert in FG_COMPONENT_SPECS:
        if name not in components.columns:
            continue

        scores[name] = percentile_score(
            components[name],
            invert=invert,
            min_periods=min_periods,
            window=window,
            lower_q=lower_q,
            upper_q=upper_q,
        )

    comp_cols = [c for c in scores.columns if c in dict(FG_COMPONENT_SPECS)]
    if not comp_cols:
        scores["FG_est"] = np.nan
        return scores, "FG_est"

    if weights is None:
        # Moyenne simple des composantes disponibles
        scores["FG_est_mean"] = scores[comp_cols].mean(axis=1)
        name = "FG_est_mean"
    else:
        # Moyenne pondérée calibrée
        w = pd.Series(
            {k: float(v) for k, v in weights.items() if k in comp_cols},
            dtype=float,
        )
        scores["FG_est_cal"] = intercept + (scores[w.index] * w).sum(axis=1)
        name = "FG_est_cal"

    return scores, name

