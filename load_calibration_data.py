"""Load estimator features and White Rabbit targets for calibration."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from fg_core import get_fgi_estimation
from get_fg_historic import get_fg_historic


def _as_date_string(value: str | pd.Timestamp | None) -> str | None:
    """Normalize a date bound to the UTC calendar day used by both loaders."""
    if value is None:
        return None
    date = pd.Timestamp(value)
    if date.tzinfo is not None:
        date = date.tz_convert("UTC").tz_localize(None)
    return date.normalize().strftime("%Y-%m-%d")


def load_calibration_data(
    start_date: str | pd.Timestamp = "2021-02-01",
    end_date: str | pd.Timestamp | None = None,
    *,
    history_start: str = "2005-01-01",
    data_dir: str = "data/fred_cache",
    source: Literal["cnn", "all"] = "cnn",
    min_periods: int = 252,
    window: int | None = 1260,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Load component scores, the equal-weight estimate, and Rabbit's target.

    Market data and component calculations come from the project's existing
    ``fg_core``/``utils`` pipeline. The estimator is deliberately run without
    saved calibration weights, so ``FG_est_mean`` remains the equal-weight
    baseline that can be compared with a newly fitted Bayesian regressor.

    The returned frame is indexed by date and contains raw component columns
    (``raw_*``), component score columns (``score_*``), ``FG_est_mean``, and
    White Rabbit columns ``fear_greed_score``, ``sentiment``, and ``source``.
    It is left-joined on the estimator dates, so recent estimator values remain
    available even when Rabbit has not published a target for that date yet.

    Args:
        start_date: First estimator/target date to return, inclusive. The
            default starts at the CNN-backed White Rabbit period.
        end_date: Last estimator/target date to return, inclusive. ``None``
            uses the current date in the estimator and the latest Rabbit row.
        history_start: Earlier date from which to load market inputs so rolling
            indicators and percentile scores have a warm-up period.
        data_dir: Directory used by the existing FRED cache pipeline.
        source: ``"cnn"`` uses the CNN-backed Rabbit period; ``"all"`` also
            includes its older third-party archive.
        min_periods: Minimum observations for each rolling percentile score.
        window: Maximum rolling percentile window, or ``None`` for expanding.
        lower_q: Lower winsorization quantile used by the estimator.
        upper_q: Upper winsorization quantile used by the estimator.
        timeout: HTTP timeout in seconds for downloading White Rabbit's CSV.

    Returns:
        A date-indexed DataFrame ready for exploratory analysis and calibration.
        Select ``score_*`` columns as model features and ``fear_greed_score``
        as the supervised target on rows where that target is present.
    """
    normalized_start = _as_date_string(start_date)
    normalized_end = _as_date_string(end_date)
    estimator_frame = get_fgi_estimation(
        start_date=normalized_start,
        end_date=normalized_end,
        with_components=True,
        use_calibrated_model=False,
        history_start=history_start,
        data_dir=data_dir,
        min_periods=min_periods,
        window=window,
        lower_q=lower_q,
        upper_q=upper_q,
    )

    if "FG_estimation" not in estimator_frame.columns:
        raise KeyError("The estimator output does not contain 'FG_estimation'.")

    estimator_frame = estimator_frame.rename(
        columns={"FG_estimation": "FG_est_mean"}
    )
    rabbit_frame = get_fg_historic(
        source=source,
        start_date=normalized_start,
        end_date=normalized_end,
        timeout=timeout,
    )

    result = estimator_frame.join(rabbit_frame, how="left").sort_index()
    result.index.name = "date"
    result.attrs["target_column"] = "fear_greed_score"
    result.attrs["feature_columns"] = [
        column for column in result.columns if column.startswith("score_")
    ]
    result.attrs["white_rabbit_source_url"] = rabbit_frame.attrs.get("source_url")
    result.attrs["white_rabbit_fetched_at_utc"] = rabbit_frame.attrs.get(
        "fetched_at_utc"
    )
    return result
