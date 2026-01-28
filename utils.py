import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf

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
def get_yf_close(ticker, start="1990-01-01"):
    data = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if data is None or len(data) == 0:
        raise ValueError(f"No data returned by yfinance for {ticker}")
    close = data["Close"].copy()
    close.name = ticker
    return close

def build_raw_indicators(
    api_key_fred: str,
    data_dir: str = "data_fred",
    start: str = "1990-01-01",
) -> pd.DataFrame:
    """
    Récupère et assemble les séries brutes nécessaires à l'indice :
      - '^GSPC', '^VIX', 'TLT', 'RSP', 'HYG' via yfinance
      - 'HY_spread' via FRED (BAMLH0A0HYM2)
      - 'put_call' via FRED si disponible (PUTCALL)

    Renvoie un DataFrame journalier business days, ffill sur les jours sans cotation.
    """

    os.makedirs(data_dir, exist_ok=True)

    spx = get_yf_close("^GSPC", start=start)
    vix = get_yf_close("^VIX", start=start)
    tlt = get_yf_close("TLT", start=start)
    rsp = get_yf_close("RSP", start=start)
    hyg = get_yf_close("HYG", start=start)

    # HY spread (BofA High Yield Option-Adjusted Spread)
    hy_path = os.path.join(data_dir, "BAMLH0A0HYM2.json")
    hy_spread = _get_data_fred(
        hy_path,
        api_key_fred,
        series_id="BAMLH0A0HYM2",
        rename="HY_spread",
    )

    # Put/Call ratio (si dispo sur FRED avec ton API)
    pc_path = os.path.join(data_dir, "PUTCALL.json")
    put_call = _get_data_fred(
        pc_path,
        api_key_fred,
        series_id="PUTCALL",
        rename="put_call",
    )

    # Assemblage des prix marché
    df = pd.concat([spx, vix, tlt, rsp, hyg], axis=1)
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().asfreq("B")   # calendrier business
    df = df.dropna(how="all")

    market_cols = ["^GSPC", "^VIX", "TLT", "RSP", "HYG"]
    existing = [c for c in market_cols if c in df.columns]
    existing = pd.Index(existing).unique().tolist()
    
    if existing:
        df[existing] = df[existing].ffill()

    # Join HY spread et put/call
    df = df.join(hy_spread, how="left")
    df = df.join(put_call, how="left")

    if "HY_spread" in df.columns:
        df["HY_spread"] = df["HY_spread"].ffill()

    if "put_call" in df.columns:
        df["put_call"] = df["put_call"].ffill()

    if df.columns.has_duplicates:
        dupes = df.columns[df.columns.duplicated()].tolist()
        print("[WARN] Duplicate columns in build_raw_indicators:", dupes)
        print("[WARN] Columns:", list(df.columns))
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # 2) Ensure 'existing' has unique keys too (defensive)
    existing = list(dict.fromkeys([c for c in market_cols if c in df.columns]))

    # 3) Safe forward-fill
    if existing:
        df = df.copy()
        df[existing] = df[existing].ffill()ffill()

    return df

# -----------------------------------
# 4. Calcul des 7 composantes brutes
# -----------------------------------
def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les composantes brutes de l'indice Fear & Greed.

    Colonnes attendues dans df :
      - '^GSPC' : prix S&P 500
      - '^VIX'  : VIX
      - 'TLT'   : Treasuries long terme
      - 'RSP'   : ETF equal-weight S&P 500 (breadth proxy)
      - 'HYG'   : ETF high yield (junk bond proxy)
      - 'HY_spread' : spread high-yield (FRED)
      - 'put_call'  : ratio put/call (si dispo)

    Retourne un DataFrame 'out' avec :
      - momentum_spx         : (P - MA125) / MA125
      - strength_proxy       : (P - MA200) / MA200
      - breadth_rsp_spx      : surperformance 60 j RSP - SPX
      - junk_bond_mom_20d    : rendement 20 j de HYG
      - hy_spread            : spread HY brut (pour un indicateur de peur)
      - vix_rel              : (VIX - MA50) / MA50
      - safe_haven_20d       : perf_20j SPX - perf_20j TLT
      - put_call             : ratio brut si dispo
    """

    out = pd.DataFrame(index=df.index)

    # 1) Momentum SPX (inchangé)
    ma125 = df["^GSPC"].rolling(125, min_periods=60).mean()
    out["momentum_spx"] = (df["^GSPC"] - ma125) / ma125

    # 2) Strength proxy (toujours MA200 : très corrélé au momentum, mais on le garde)
    ma200 = df["^GSPC"].rolling(200, min_periods=80).mean()
    out["strength_proxy"] = (df["^GSPC"] - ma200) / ma200

    # 3) Breadth : sur-performance 60 j de RSP vs SPX (plus stationnaire qu'un simple ratio de niveaux)
    if "RSP" in df.columns:
        ret_rsp_60 = df["RSP"].pct_change(60)
        ret_spx_60 = df["^GSPC"].pct_change(60)
        out["breadth_rsp_spx"] = ret_rsp_60 - ret_spx_60
    else:
        out["breadth_rsp_spx"] = np.nan

    # 4) Junk bond momentum : rendement 20 j de HYG
    if "HYG" in df.columns:
        out["junk_bond_mom_20d"] = df["HYG"].pct_change(20)
    else:
        out["junk_bond_mom_20d"] = np.nan

    # 5) High-yield spread brut (plus grand = plus de peur, on inversera au scoring)
    out["hy_spread"] = df.get("HY_spread")

    # 6) Volatilité relative : (VIX - MA50) / MA50
    vix = df["^VIX"].astype(float)
    ma_vix_50 = vix.rolling(50, min_periods=20).mean()
    out["vix_rel"] = (vix - ma_vix_50) / ma_vix_50

    # 7) Safe haven demand : différence de performance 20 j actions - Treasuries
    spx_ret_20 = df["^GSPC"].pct_change(20)
    tlt_ret_20 = df["TLT"].pct_change(20)
    out["safe_haven_20d"] = spx_ret_20 - tlt_ret_20

    # 8) Put/Call brut (si présent)
    out["put_call"] = df.get("put_call")

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

