from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from deploy.schemas import (
    FGIResponse, FGISeriesPoint, FGIResponseMeta,
    MarketInfo, MarketListResponse, MarketSeriesResponse, MarketSeriesMeta, ChartResponse
)
from deploy.services.fgi_service import get_fgi_series
from deploy.services.market_service import MARKET_REGISTRY, get_market_series


load_dotenv()

app = FastAPI()

# Allow all origins for dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Market endpoints ---
@app.get("/v1/markets", response_model=MarketListResponse)
def list_markets():
    """
    List supported market datasets.
    """
    return {"markets": [
        MarketInfo(id=k, label=v["label"], ticker=v["ticker"]) for k, v in MARKET_REGISTRY.items()
    ]}

@app.get("/v1/markets/{market_id}", response_model=MarketSeriesResponse)
def get_market(
    market_id: str,
    range: str = Query("1Y", regex="^(1M|3M|6M|1Y|5Y|MAX)$"),
    end_date: str = Query(None, description="YYYY-MM-DD")
):
    """
    Return a single market CLOSE price time series (raw), for charting.

    Parameters
    ----------
    * market_id : str
    
        Market identifier. Must be one of the keys from `MARKET_REGISTRY`
        (e.g. `sp500`, `nasdaq`, `cac40`, `msciworld`).
    * range : str, optional
    
        Date range preset. Allowed values: `1M`, `3M`, `6M`, `1Y`, `5Y`, `MAX`.
        Defaults to `1Y`.
    * end_date : str | None, optional
    
        End date (inclusive) in `YYYY-MM-DD` format. Defaults to today.

    Returns
    -------
    MarketSeriesResponse
    
    A payload containing:
        
    - `id`, `label`, `ticker`
    - `series`: list of `{date, value}` points where `value` is the raw CLOSE price
    - `meta`: runtime metadata (range, resolved start/end dates, generation timestamp, duration, number of points)

    Raises
    ------
    HTTPException
    
    - 404 if `market_id` is unknown.
    - 500 if data download / parsing fails.

    Notes
    -----
    - Values are *raw close prices* (no rebasing / scaling). The frontend can plot these
    on a right y-axis while plotting FGI (0–100) on a left y-axis.
    - The underlying download uses the project helper `utils.get_yf_close`.
    - If caching is enabled inside `get_market_series`, repeated calls for the same
    (market_id, range, end_date) should be fast.
    """
    start_time = datetime.utcnow()
    if market_id not in MARKET_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown market_id")
    try:
        series, start_date, end_date = get_market_series(market_id, range, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    points = [FGISeriesPoint(date=str(idx), value=float(val)) for idx, val in series.items()]
    
    duration = (datetime.utcnow() - start_time).total_seconds()
    meta = MarketSeriesMeta(
        range=range,
        start_date=start_date,
        end_date=end_date,
        generated_at=datetime.utcnow().isoformat(),
        duration_seconds= duration,
        points=len(points)
    )
    info = MARKET_REGISTRY[market_id]
    return MarketSeriesResponse(
        id=market_id,
        label=info["label"],
        ticker=info["ticker"],
        series=points,
        meta=meta
    )

# --- Chart endpoint ---
@app.get("/v1/chart", response_model=ChartResponse)
def get_chart(
    range: str = Query("1Y", regex="^(1M|3M|6M|1Y|5Y|MAX)$"),
    end_date: str = Query(None, description="YYYY-MM-DD"),
    include: str = Query("fgi"),
    with_components: bool = Query(False),
    use_calibrated_model: bool = Query(True)
):
    """
    Return multiple datasets in a single response for dashboard charting.

    This endpoint is designed for the frontend to perform **one** HTTP request and
    receive all the series it needs to plot:
    - `fgi` (0–100 sentiment index, left axis)
    - market close prices (raw, right axis)

    Parameters
    ----------
    * range : str, optional
    
        Date range preset. Allowed values: `1M`, `3M`, `6M`, `1Y`, `5Y`, `MAX`.
        Defaults to `1Y`.
    * end_date : str | None, optional
    
        End date (inclusive) in `YYYY-MM-DD` format. Defaults to today.
    * include : str, optional
    
        Comma-separated list of dataset keys to include.

        Supported keys:
        - `fgi`
        - any market id present in `MARKET_REGISTRY` (e.g. `sp500`, `nasdaq`, `cac40`, `msciworld`)

        Examples:
        - `include=fgi`
        - `include=fgi,sp500,cac40`

        Defaults to `fgi`.
    * with_components : bool, optional
    
        If `True`, include per-date component score columns for the FGI dataset under
        the top-level `components` field. Defaults to `False`.
    * use_calibrated_model : bool, optional
    
        If `True`, the FGI computation uses the calibrated linear model (if available).
        Defaults to `True`.

    Returns
    -------
    ChartResponse
    
    A payload containing:
    
    * `datasets`: a dict mapping each requested key to a list of `{date, value}` points
    * `datasets["fgi"]`: sentiment series (0–100)
    * `datasets["sp500"]`, etc.: raw close price series
    * `meta`: global metadata (range, resolved start/end dates, included keys,
    generation timestamp, runtime duration)
    * `components` (optional): list of per-date dicts containing component score values
    when `with_components=True`

    Raises
    ------
    HTTPException
    
    - 500 if a requested dataset fails to load.
    - 500 if the FGI output does not contain a valid FGI column (`FG_estimation` or `FG_like`).

    Notes
    -----
    - Datasets may have different date grids (trading calendars differ across indices).
    The frontend is expected to handle alignment visually.
    - Market series are returned as raw close prices (intended for a right y-axis).
    - FGI is returned as 0–100 (intended for a left y-axis).
    """
    start_time = datetime.utcnow()
    keys = [k.strip() for k in include.split(",") if k.strip()]
    datasets = {}
    start_date, end_date_actual = None, None
    for key in keys:
        if key == "fgi":
            df, s, e = get_fgi_series(range, end_date, with_components, use_calibrated_model)
            col = None
            for c in ["FG_estimation", "FG_like"]:
                if c in df.columns:
                    col = c
                    break
            if not col:
                raise HTTPException(status_code=500, detail="FG_estimation or FG_like column not found in result.")
            points = [FGISeriesPoint(date=str(idx)[:10], value=float(val)) for idx, val in df[col].items()]
            datasets["fgi"] = points
            if start_date is None:
                start_date, end_date_actual = s, e
        elif key in MARKET_REGISTRY:
            try:
                series, s, e = get_market_series(key, range, end_date)
            except Exception as ex:
                raise HTTPException(status_code=500, detail=f"Error for {key}: {ex}")
            points = [FGISeriesPoint(date=str(idx), value=float(val)) for idx, val in series.items()]
            datasets[key] = points
            if start_date is None:
                start_date, end_date_actual = s, e
    
    components_list = None
    if with_components:
        # collect score_* columns
        score_cols = df.drop([col], axis=1).columns.to_list()
        components_list = []
        for idx, row in df.iterrows():
            entry = {"date": str(idx)[:10]}
            for sc in score_cols:
                val = row.get(sc)
                entry[sc] = None if pd.isna(val) else float(val)
            components_list.append(entry)
    
    duration = (datetime.utcnow() - start_time).total_seconds()
    meta = {
        "range": range,
        "start_date": start_date,
        "end_date": end_date_actual,
        "include": keys,
        "generated_at": datetime.utcnow().isoformat(),
        "duration_seconds": duration,
    }
    return ChartResponse(datasets=datasets, meta=meta, components=components_list)

@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.get("/v1/fgi", response_model=FGIResponse)
def get_fgi(
    range: str = Query("1Y", regex="^(1M|3M|6M|1Y|5Y|MAX)$"),
    end_date: str = Query(None, description="YYYY-MM-DD"),
    with_components: bool = Query(False),
    use_calibrated_model: bool = Query(True)
):
    """
    Return the Fear & Greed estimation time series (0–100).

    Parameters
    ----------
    * range : str, optional
    
        Date range preset. Allowed values: `1M`, `3M`, `6M`, `1Y`, `5Y`, `MAX`.
        Defaults to `1Y`.
    * end_date : str | None, optional
    
        End date (inclusive) in `YYYY-MM-DD` format. Defaults to today.
    * with_components : bool, optional
    
        If `True`, include per-date component score values (e.g. `score_*` columns)
        in the response under the `components` field. Defaults to `False`.
    * use_calibrated_model : bool, optional
    
        If `True`, use the calibrated linear model (if available) for the composite
        FGI estimation. Defaults to `True`.

    Returns
    -------
    FGIResponse
    
    A payload containing:
    
    - `series`: list of `{date, value}` points where `value` is the FGI estimate (0–100)
    - `last_value`: last non-null FGI value (or null if empty)
    - `meta`: computation metadata (range, resolved start/end dates, timestamp, runtime duration, points)
    - `components` (optional): list of per-date dictionaries including component score values

    Raises
    ------
    RuntimeError: 
        If no valid FGI column is found in the computed DataFrame.

    Notes
    -----
    - The implementation relies on `get_fgi_series(...)` from `deploy.services.fgi_service`.
    - The DataFrame returned by the pipeline may expose the aggregate as either
    `FG_estimation` or `FG_like`; this endpoint detects both.
    - Component details can increase payload size significantly; keep `with_components=False`
    for normal dashboard use.
    """
    start_time = datetime.utcnow()
    df, start_date, end_date = get_fgi_series(range, end_date, with_components, use_calibrated_model)
    col = None
    for c in ["FG_estimation", "FG_like"]:
        if c in df.columns:
            col = c
            break
    if not col:
        raise RuntimeError("FG_estimation or FG_like column not found in result.")
    series = [FGISeriesPoint(date=str(idx)[:10], value=float(val)) for idx, val in df[col].items()]
    last_value = float(df[col].iloc[-1]) if not df.empty else None
    components_list = None
    if with_components:
        # collect score_* columns
        score_cols = df.drop([col], axis=1).columns.to_list()
        components_list = []
        for idx, row in df.iterrows():
            entry = {"date": str(idx)[:10]}
            for sc in score_cols:
                val = row.get(sc)
                entry[sc] = None if pd.isna(val) else float(val)
            components_list.append(entry)
    duration = (datetime.utcnow() - start_time).total_seconds()
    meta = FGIResponseMeta(
        range=range,
        start_date=start_date,
        end_date=end_date,
        with_components=with_components,
        use_calibrated_model=use_calibrated_model,
        generated_at=datetime.utcnow().isoformat(),
        duration_seconds = duration,
        points=len(series)
    )
    return FGIResponse(series=series, last_value=last_value, meta=meta, components=components_list)


# --- Serve frontend (web/) ---
WEB_DIR = Path(__file__).resolve().parents[1] / "web"   # ajuste si besoin
app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

@app.get("/")
def root():
    return RedirectResponse(url="/web/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        reload=True
    )