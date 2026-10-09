"""Load CNN Fear & Greed history from White Rabbit's public dataset."""

from __future__ import annotations

from io import StringIO
from typing import Literal

import pandas as pd
import requests


DATA_URL = (
    "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/"
    "main/fear-greed.csv"
)
CNN_SOURCE_START = pd.Timestamp("2021-02-01")
_REQUIRED_COLUMNS = {"Date", "Fear Greed", "Rating"}


def _normalize_date_bound(
    value: str | pd.Timestamp | None,
    name: str,
) -> pd.Timestamp | None:
    """Parse a date bound and normalize it to a timezone-naive UTC day."""
    if value is None:
        return None

    try:
        date = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date; got {value!r}.") from exc

    if pd.isna(date):
        raise ValueError(f"{name} must be a valid date; got {value!r}.")
    if date.tzinfo is not None:
        date = date.tz_convert("UTC").tz_localize(None)
    return date.normalize()


def get_fg_historic(
    source: Literal["cnn", "all"] = "cnn",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    *,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch and return White Rabbit's Fear & Greed history.

    The default keeps only observations from the CNN-backed endpoint era,
    starting on 2021-02-01. Use ``source="all"`` to include the older
    third-party archive as well; a ``source`` column marks each row's origin.

    The result is sorted by date and indexed by ``date``. Its columns are
    ``fear_greed_score`` (float, 0-100), ``sentiment`` (lowercase rating), and
    ``source`` (``cnn_endpoint`` or ``third_party_archive``).

    Args:
        source: ``"cnn"`` for the CNN-backed period, or ``"all"`` to include
            White Rabbit's older third-party archive.
        start_date: Optional inclusive first date. Dates are interpreted as
            UTC when timezone-aware and as calendar dates when timezone-naive.
        end_date: Optional inclusive last date, with the same date handling.
        timeout: HTTP request timeout in seconds.

    Returns:
        A DataFrame indexed by unique, normalized UTC dates.

    Raises:
        ValueError: If the source or date range is invalid, or the downloaded
            data has an unexpected schema or invalid observations.
        RuntimeError: If the dataset cannot be downloaded or parsed.
    """
    if source not in ("cnn", "all"):
        raise ValueError("source must be either 'cnn' or 'all'.")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    start = _normalize_date_bound(start_date, "start_date")
    end = _normalize_date_bound(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date.")
    if source == "cnn" and start is not None and start < CNN_SOURCE_START:
        raise ValueError(
            "start_date is before the CNN-backed period (2021-02-01). "
            "Use source='all' to include the older third-party archive."
        )

    try:
        response = requests.get(
            DATA_URL,
            headers={
                "Accept": "text/csv",
                "User-Agent": "FGIndexEstimator/1.0",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download White Rabbit data: {exc}") from exc
    except (pd.errors.ParserError, UnicodeError) as exc:
        raise RuntimeError(f"Could not parse White Rabbit CSV data: {exc}") from exc

    missing_columns = _REQUIRED_COLUMNS.difference(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"White Rabbit CSV is missing required columns: {missing}.")

    dates = pd.to_datetime(raw["Date"], errors="coerce", utc=True)
    scores = pd.to_numeric(raw["Fear Greed"], errors="coerce")
    ratings = raw["Rating"].astype("string").str.strip().str.casefold()

    if dates.isna().any():
        raise ValueError("White Rabbit CSV contains invalid dates.")
    if scores.isna().any() or not scores.between(0, 100).all():
        raise ValueError("White Rabbit CSV contains invalid scores outside 0-100.")
    if ratings.isna().any() or ratings.eq("").any():
        raise ValueError("White Rabbit CSV contains empty ratings.")

    frame = pd.DataFrame(
        {
            "date": dates.dt.tz_convert(None).dt.normalize(),
            "fear_greed_score": scores.astype(float),
            "sentiment": ratings,
        }
    )
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(False), "date"]
        duplicate_days = ", ".join(
            date.strftime("%Y-%m-%d") for date in duplicates.drop_duplicates().head(5)
        )
        raise ValueError(
            "White Rabbit CSV contains multiple observations per date "
            f"({duplicate_days}); resolve them explicitly before calibration."
        )

    frame["source"] = "cnn_endpoint"
    frame.loc[frame["date"] < CNN_SOURCE_START, "source"] = "third_party_archive"
    if source == "cnn":
        frame = frame.loc[frame["date"] >= CNN_SOURCE_START]
    if start is not None:
        frame = frame.loc[frame["date"] >= start]
    if end is not None:
        frame = frame.loc[frame["date"] <= end]
    if frame.empty:
        raise ValueError("No observations found for the requested source and dates.")

    result = frame.set_index("date").sort_index()
    result.index.name = "date"
    result.attrs["source_url"] = DATA_URL
    result.attrs["fetched_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    return result
