
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FGISeriesPoint(BaseModel):
    date: str
    value: float

class FGIResponseMeta(BaseModel):
    range: str
    start_date: str
    end_date: str
    with_components: bool
    use_calibrated_model: bool
    generated_at: str
    duration_seconds: float
    points: int

class FGIResponse(BaseModel):
    series: List[FGISeriesPoint]
    last_value: Optional[float]
    meta: FGIResponseMeta
    # Optional per-date component components when requested
    components: Optional[List[Dict[str, Any]]] = None

# --- Market endpoints ---
class MarketInfo(BaseModel):
    id: str
    label: str
    ticker: str

class MarketListResponse(BaseModel):
    markets: List[MarketInfo]

class MarketSeriesMeta(BaseModel):
    range: str
    start_date: str
    end_date: str
    generated_at: str
    duration_seconds: float
    points: int

class MarketSeriesResponse(BaseModel):
    id: str
    label: str
    ticker: str
    series: List[FGISeriesPoint]
    meta: MarketSeriesMeta

# --- Chart endpoint ---
class ChartResponse(BaseModel):
    datasets: Dict[str, List[FGISeriesPoint]]
    meta: Dict[str, Any]
    # Optional per-date component components when requested
    components: Optional[List[Dict[str, Any]]] = None
