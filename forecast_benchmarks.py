"""Leakage-aware weekly F&G target construction and forecast benchmarks."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "fear_greed_score"
CNN_SOURCE = "cnn_endpoint"
ARCHIVE_SOURCE = "third_party_archive"
WHITE_RABBIT_SOURCES = (ARCHIVE_SOURCE, CNN_SOURCE)
AGGREGATIONS = ("last", "mean", "median")
MODEL_NAMES = (
    "Seasonal Naive",
    "Moving Average",
    "ARIMA",
    "VAR (first differences)",
    "VECM",
    "GRU",
)


def build_weekly_datasets(
    history_frame: pd.DataFrame,
    *,
    aggregations: Iterable[str] = AGGREGATIONS,
    week_rule: str = "W-FRI",
) -> dict[str, pd.DataFrame]:
    """Build regular weekly targets and inputs from all White Rabbit eras.

    All observed Rabbit target values are aggregated, including the pre-2021
    third-party archive and the post-2021 CNN endpoint. Features are aggregated
    from the estimator's full daily history, not only dates with Rabbit values.
    Internal missing target or feature weeks are forward-filled from the past
    to preserve weekly spacing. Target-observation and source masks are kept in
    frame attrs so imputed targets never count as backtest actuals.

    Args:
        history_frame: Date-indexed frame from load_calibration_data.
        aggregations: Candidate weekly summaries: last, mean, or median.
        week_rule: Pandas weekly resampling rule; defaults to Friday-ending
            weeks.

    Returns:
        Mapping from aggregation name to regular weekly frames. Each frame
        carries source coverage, feature-selection, and target-observation
        details in its attrs.
    """
    requested_features = list(history_frame.attrs.get("feature_columns", []))
    target_column = history_frame.attrs.get("target_column", TARGET_COLUMN)
    required = requested_features + [target_column, "source"]
    missing = set(required).difference(history_frame.columns)
    if missing:
        raise ValueError(f"History is missing columns: {', '.join(sorted(missing))}.")

    requested = tuple(aggregations)
    invalid = set(requested).difference(AGGREGATIONS)
    if invalid:
        raise ValueError(f"Unsupported aggregations: {', '.join(sorted(invalid))}.")
    if not requested:
        raise ValueError("Request at least one weekly target aggregation.")

    rabbit_daily = history_frame.loc[
        history_frame["source"].isin(WHITE_RABBIT_SOURCES)
        & history_frame[target_column].notna(),
        [*requested_features, target_column, "source"],
    ].sort_index()
    if rabbit_daily.empty:
        raise ValueError("No White Rabbit target observations are available.")

    present_sources = [
        source for source in WHITE_RABBIT_SOURCES
        if rabbit_daily["source"].eq(source).any()
    ]
    feature_columns = [
        feature for feature in requested_features
        if all(
            rabbit_daily.loc[rabbit_daily["source"].eq(source), feature].notna().any()
            for source in present_sources
        )
    ]
    excluded_features = [
        feature for feature in requested_features if feature not in feature_columns
    ]
    if not feature_columns:
        raise ValueError(
            "No component feature has observations in every White Rabbit source era. "
            "The target history is available, but multivariate forecasts need "
            "a common input set."
        )

    target_source_days = {
        source: int(rabbit_daily["source"].eq(source).sum())
        for source in present_sources
    }
    complete_days_by_source = {
        source: int(
            rabbit_daily.loc[rabbit_daily["source"].eq(source), feature_columns]
            .notna()
            .all(axis=1)
            .sum()
        )
        for source in present_sources
    }

    def source_label(values: pd.Series) -> str:
        sources = set(values.dropna().unique())
        if len(sources) > 1:
            return "mixed_sources"
        return next(iter(sources)) if sources else "unobserved"

    source_by_week = (
        rabbit_daily.groupby(pd.Grouper(freq=week_rule))["source"]
        .agg(source_label)
        .to_dict()
    )

    weekly_frames: dict[str, pd.DataFrame] = {}
    feature_imputed_counts: dict[str, int] = {}
    for aggregation in requested:
        target_weekly = rabbit_daily[target_column].resample(week_rule).agg(aggregation)
        if len(target_weekly) and target_weekly.index[-1] > rabbit_daily.index[-1].normalize():
            target_weekly = target_weekly.iloc[:-1]
        if target_weekly.empty:
            raise ValueError("No complete White Rabbit target weeks are available.")

        week_index = pd.date_range(
            target_weekly.index.min(),
            target_weekly.index.max(),
            freq=week_rule,
        )
        target_weekly = target_weekly.reindex(week_index)
        target_observed = target_weekly.notna()
        target_filled = target_weekly.ffill()
        if target_filled.isna().any():
            raise ValueError("The weekly target has missing values before its first observation.")

        feature_weekly = (
            history_frame[feature_columns]
            .sort_index()
            .resample(week_rule)
            .agg(aggregation)
            .reindex(week_index)
        )
        feature_imputed = feature_weekly.isna().any(axis=1)
        feature_weekly = feature_weekly.ffill()
        if feature_weekly.isna().any().any():
            missing_feature_names = feature_weekly.columns[
                feature_weekly.isna().any()
            ].tolist()
            raise ValueError(
                "Component inputs have no earlier value to carry forward at the "
                f"start of the target period: {missing_feature_names}."
            )

        weekly = pd.concat(
            [target_filled.rename("target"), feature_weekly],
            axis=1,
        ).astype(float)
        feature_imputed_counts[aggregation] = int(feature_imputed.sum())

        weekly.attrs["aggregation"] = aggregation
        weekly.attrs["week_rule"] = week_rule
        weekly.attrs["feature_columns"] = feature_columns
        weekly.attrs["excluded_feature_columns"] = excluded_features
        weekly.attrs["target_column"] = "target"
        weekly.attrs["source_target_column"] = target_column
        weekly.attrs["target_source"] = "white_rabbit_all_sources"
        weekly.attrs["target_source_by_week"] = source_by_week
        weekly.attrs["target_observed_by_week"] = target_observed.to_dict()
        weekly.attrs["source_target_days"] = target_source_days
        weekly.attrs["complete_days_by_source"] = complete_days_by_source
        weekly.attrs["complete_daily_rows"] = sum(complete_days_by_source.values())
        weekly.attrs["observed_target_weeks"] = int(target_observed.sum())
        weekly.attrs["carried_target_weeks"] = int((~target_observed).sum())
        weekly.attrs["feature_imputed_weeks"] = feature_imputed_counts[aggregation]
        weekly.attrs["observed_target_mean"] = float(target_weekly.mean())
        weekly.attrs["observed_target_std"] = float(target_weekly.std())
        weekly.attrs["observed_target_min"] = float(target_weekly.min())
        weekly.attrs["observed_target_max"] = float(target_weekly.max())
        weekly_frames[aggregation] = weekly
    return weekly_frames


def _seasonal_naive(y: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    if len(y) < season_length:
        raise ValueError(
            f"Seasonal Naive needs at least {season_length} training weeks."
        )
    start = len(y) - season_length
    return np.asarray(
        [y[start + ((step - 1) % season_length)] for step in range(1, horizon + 1)],
        dtype=float,
    )


def _moving_average(y: np.ndarray, horizon: int, window: int) -> np.ndarray:
    if len(y) < window:
        raise ValueError(f"Moving Average needs at least {window} training weeks.")
    return np.repeat(float(np.mean(y[-window:])), horizon)


def _forecast_arima(y: np.ndarray, horizon: int) -> tuple[np.ndarray, str]:
    from statsmodels.tsa.arima.model import ARIMA

    candidate_orders = (
        (1, 0, 0),
        (0, 1, 1),
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 0),
    )
    best_aic = float("inf")
    best_fit = None
    best_order = None
    for order in candidate_orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(y, order=order).fit()
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = float(fit.aic)
                best_fit = fit
                best_order = order
        except (ValueError, np.linalg.LinAlgError):
            continue
    if best_fit is None:
        raise RuntimeError("All candidate ARIMA orders failed to fit.")
    return np.asarray(best_fit.forecast(steps=horizon), dtype=float), str(best_order)


def _forecast_var(
    train: pd.DataFrame,
    horizon: int,
    target_column: str,
) -> tuple[np.ndarray, str]:
    from statsmodels.tsa.api import VAR

    differences = train.diff().dropna()
    if len(differences) < 12:
        raise ValueError("Too few weekly differences to fit VAR.")
    model = VAR(differences)
    max_lags = min(4, max(1, len(differences) // (2 * len(train.columns) + 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = model.select_order(maxlags=max_lags)
    lag_order = selection.selected_orders.get("bic") or 1
    lag_order = max(1, int(lag_order))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(maxlags=lag_order)
    difference_forecast = fitted.forecast(
        differences.to_numpy()[-fitted.k_ar :],
        steps=horizon,
    )
    target_position = train.columns.get_loc(target_column)
    level_forecast = train.iloc[-1].to_numpy() + np.cumsum(difference_forecast, axis=0)
    return level_forecast[:, target_position], f"lag={fitted.k_ar}"


def _forecast_vecm(
    train: pd.DataFrame,
    horizon: int,
    target_column: str,
) -> tuple[np.ndarray, str]:
    from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

    rank_result = select_coint_rank(
        train.to_numpy(dtype=float),
        det_order=0,
        k_ar_diff=1,
        method="trace",
        signif=0.05,
    )
    rank = int(rank_result.rank)
    if rank <= 0 or rank >= train.shape[1]:
        raise ValueError(
            f"Johansen trace test selected rank {rank}; VECM rank must be between 1 and K-1."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = VECM(
            train,
            k_ar_diff=1,
            coint_rank=rank,
            deterministic="co",
        ).fit()
    forecast = np.asarray(fitted.predict(steps=horizon), dtype=float)
    return forecast[:, train.columns.get_loc(target_column)], f"rank={rank}, diff_lags=1"


def _forecast_gru(
    train: pd.DataFrame,
    horizon: int,
    *,
    lookback: int,
    hidden_size: int,
    max_epochs: int,
    patience: int,
    validation_fraction: float,
    batch_size: int,
    learning_rate: float,
    random_state: int,
) -> tuple[np.ndarray, str]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "The forecast GRU requires PyTorch. Install it with `pip install torch`."
        ) from exc

    values = train.to_numpy(dtype=np.float32)
    target_position = train.columns.get_loc("target")
    origins = np.arange(lookback - 1, len(train) - horizon)
    if len(origins) < 20:
        raise ValueError("Too few training examples for the multivariate forecast GRU.")

    validation_count = max(2, int(np.ceil(len(origins) * validation_fraction)))
    validation_origins = origins[-validation_count:]
    first_validation_origin = int(validation_origins[0])
    fit_origins = origins[origins + horizon < first_validation_origin]
    if len(fit_origins) < 10:
        raise ValueError("Too few GRU training origins after the validation split.")

    validation_scaler = StandardScaler().fit(values[: first_validation_origin + 1])
    validation_values = validation_scaler.transform(values).astype(np.float32)

    def make_sequences(scaled: np.ndarray, forecast_origins: np.ndarray) -> np.ndarray:
        return np.stack(
            [scaled[origin - lookback + 1 : origin + 1] for origin in forecast_origins]
        )

    def make_targets(forecast_origins: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                values[origin + 1 : origin + horizon + 1, target_position]
                for origin in forecast_origins
            ]
        ).astype(np.float32)

    X_fit = make_sequences(validation_values, fit_origins)
    y_fit = make_targets(fit_origins)
    X_val = make_sequences(validation_values, validation_origins)
    y_val = make_targets(validation_origins)

    class MultiHorizonGRU(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, horizon)

        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            encoded, _ = self.encoder(sequence)
            return self.head(encoded[:, -1, :])

    def loader(X: np.ndarray, y: np.ndarray, seed: int) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    def train_epoch(
        model: MultiHorizonGRU,
        batches: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        model.train()
        for batch_X, batch_y in batches:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch_X), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    torch.manual_seed(random_state)
    model = MultiHorizonGRU(input_size=train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    fit_loader = loader(X_fit, y_fit, random_state)
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    for epoch in range(1, max_epochs + 1):
        train_epoch(model, fit_loader, optimizer)
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                nn.functional.mse_loss(model(torch.from_numpy(X_val)), torch.from_numpy(y_val))
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    # Refit on every fully observed training origin for the selected epochs.
    final_scaler = StandardScaler().fit(values)
    scaled_values = final_scaler.transform(values).astype(np.float32)
    all_X = make_sequences(scaled_values, origins)
    all_y = make_targets(origins)
    torch.manual_seed(random_state)
    final_model = MultiHorizonGRU(input_size=train.shape[1])
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=learning_rate,
        weight_decay=1e-3,
    )
    final_loader = loader(all_X, all_y, random_state)
    for _ in range(best_epoch):
        train_epoch(final_model, final_loader, final_optimizer)

    final_model.eval()
    final_input = torch.from_numpy(scaled_values[-lookback:]).unsqueeze(0)
    with torch.no_grad():
        forecast = final_model(final_input).squeeze(0).numpy()
    return forecast.astype(float), f"lookback={lookback}, selected_epochs={best_epoch}"


def forecast_candidate(
    model_name: str,
    train_frame: pd.DataFrame,
    *,
    horizon: int = 6,
    season_length: int = 52,
    moving_average_window: int = 4,
    gru_lookback: int = 26,
    gru_hidden_size: int = 32,
    gru_max_epochs: int = 100,
    gru_patience: int = 12,
    gru_validation_fraction: float = 0.15,
    gru_batch_size: int = 16,
    gru_learning_rate: float = 0.001,
    random_state: int = 42,
) -> tuple[np.ndarray, str]:
    """Fit one requested weekly model and return its six-step target path."""
    y = train_frame["target"].to_numpy(dtype=float)
    if model_name == "Seasonal Naive":
        return _seasonal_naive(y, horizon, season_length), f"season={season_length}"
    if model_name == "Moving Average":
        return _moving_average(y, horizon, moving_average_window), f"window={moving_average_window}"
    if model_name == "ARIMA":
        return _forecast_arima(y, horizon)
    if model_name == "VAR (first differences)":
        return _forecast_var(train_frame, horizon, "target")
    if model_name == "VECM":
        return _forecast_vecm(train_frame, horizon, "target")
    if model_name == "GRU":
        return _forecast_gru(
            train_frame,
            horizon,
            lookback=gru_lookback,
            hidden_size=gru_hidden_size,
            max_epochs=gru_max_epochs,
            patience=gru_patience,
            validation_fraction=gru_validation_fraction,
            batch_size=gru_batch_size,
            learning_rate=gru_learning_rate,
            random_state=random_state,
        )
    raise ValueError(f"Unknown forecast model: {model_name!r}.")


def rolling_origin_backtest(
    weekly_frame: pd.DataFrame,
    *,
    horizon: int = 6,
    initial_train_weeks: int = 104,
    origin_step_weeks: int = 6,
    models: Iterable[str] = MODEL_NAMES,
    season_length: int = 52,
    moving_average_window: int = 4,
    gru_lookback: int = 26,
    gru_hidden_size: int = 32,
    gru_max_epochs: int = 100,
    gru_patience: int = 12,
    gru_validation_fraction: float = 0.15,
    gru_batch_size: int = 16,
    gru_learning_rate: float = 0.001,
    random_state: int = 42,
) -> pd.DataFrame:
    """Backtest multi-step weekly forecasts at expanding, non-overlap origins.

    Each training window expands by ``origin_step_weeks`` after the previous
    six-step outcome is observed. Statistical order/rank selection and GRU
    epoch selection use only the data available at that origin. Weeks whose
    target was carried forward to fill a data gap are excluded from scoring.
    """
    if horizon < 1 or initial_train_weeks < 1 or origin_step_weeks < horizon:
        raise ValueError("Use positive sizes and origin_step_weeks >= horizon.")
    if "target" not in weekly_frame.columns:
        raise ValueError("Weekly frame must include a 'target' column.")
    requested_models = tuple(models)
    origins = range(
        initial_train_weeks - 1,
        len(weekly_frame) - horizon,
        origin_step_weeks,
    )
    prediction_rows: list[dict[str, object]] = []
    for origin_position in origins:
        train = weekly_frame.iloc[: origin_position + 1].copy()
        actual_frame = weekly_frame.iloc[origin_position + 1 : origin_position + horizon + 1]
        actual = actual_frame["target"].to_numpy(dtype=float)
        observed_by_week = weekly_frame.attrs.get("target_observed_by_week", {})
        actual_observed = np.asarray(
            [observed_by_week.get(date, True) for date in actual_frame.index],
            dtype=bool,
        )
        actual[~actual_observed] = np.nan
        for model_name in requested_models:
            status = "ok"
            details = ""
            try:
                prediction, details = forecast_candidate(
                    model_name,
                    train,
                    horizon=horizon,
                    season_length=season_length,
                    moving_average_window=moving_average_window,
                    gru_lookback=gru_lookback,
                    gru_hidden_size=gru_hidden_size,
                    gru_max_epochs=gru_max_epochs,
                    gru_patience=gru_patience,
                    gru_validation_fraction=gru_validation_fraction,
                    gru_batch_size=gru_batch_size,
                    gru_learning_rate=gru_learning_rate,
                    random_state=random_state,
                )
            except Exception as exc:  # Preserve a visible row for unavailable candidates.
                prediction = np.full(horizon, np.nan)
                details = ""
                status = f"unavailable: {type(exc).__name__}: {exc}"

            for step, (target_date, observed, predicted) in enumerate(
                zip(actual_frame.index, actual, prediction),
                start=1,
            ):
                prediction_rows.append(
                    {
                        "origin_date": train.index[-1],
                        "target_date": target_date,
                        "horizon": step,
                        "model": model_name,
                        "actual": observed,
                        "prediction": predicted,
                        "status": status,
                        "model_details": details,
                    }
                )
    return pd.DataFrame(prediction_rows)


def summarize_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return per-aggregation, per-model and per-horizon error metrics."""
    required = {"aggregation", "model", "horizon", "actual", "prediction", "status"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Backtest predictions are missing: {', '.join(sorted(missing))}.")
    rows: list[dict[str, object]] = []
    group_columns = ["aggregation", "model", "horizon"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        valid = group.dropna(subset=["actual", "prediction"])
        agg_name, model_name, horizon = keys
        rows.append(
            {
                "aggregation": agg_name,
                "model": model_name,
                "horizon_weeks": int(horizon),
                "n_origins": len(valid),
                "MAE": (
                    float(mean_absolute_error(valid["actual"], valid["prediction"]))
                    if len(valid)
                    else np.nan
                ),
                "RMSE": (
                    float(np.sqrt(mean_squared_error(valid["actual"], valid["prediction"])))
                    if len(valid)
                    else np.nan
                ),
                "R2": (
                    float(r2_score(valid["actual"], valid["prediction"]))
                    if len(valid) > 1
                    else np.nan
                ),
                "failed_origins": int(group["prediction"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def forecast_latest_scenarios(
    weekly_frames: dict[str, pd.DataFrame],
    *,
    horizon: int = 6,
    models: Iterable[str] = MODEL_NAMES,
    **model_settings: object,
) -> pd.DataFrame:
    """Fit every candidate to the latest full week under each target definition."""
    rows: list[dict[str, object]] = []
    for aggregation, weekly_frame in weekly_frames.items():
        for model_name in models:
            status = "ok"
            details = ""
            try:
                forecast, details = forecast_candidate(
                    model_name,
                    weekly_frame,
                    horizon=horizon,
                    **model_settings,
                )
            except Exception as exc:
                forecast = np.full(horizon, np.nan)
                details = ""
                status = f"unavailable: {type(exc).__name__}: {exc}"
            future_dates = pd.date_range(
                weekly_frame.index[-1] + pd.offsets.Week(weekday=4),
                periods=horizon,
                freq="W-FRI",
            )
            for step, (target_date, predicted) in enumerate(zip(future_dates, forecast), start=1):
                rows.append(
                    {
                        "aggregation": aggregation,
                        "model": model_name,
                        "origin_date": weekly_frame.index[-1],
                        "target_date": target_date,
                        "horizon": step,
                        "prediction": predicted,
                        "status": status,
                        "model_details": details,
                    }
                )
    return pd.DataFrame(rows)


def plot_backtest_metrics(metrics: pd.DataFrame) -> tuple[Figure, np.ndarray]:
    """Plot MAE by horizon in separate panels for the three weekly targets."""
    aggregations = [name for name in AGGREGATIONS if name in set(metrics["aggregation"])]
    fig, axes = plt.subplots(
        len(aggregations),
        1,
        figsize=(13, 4 * len(aggregations)),
        sharex=True,
        squeeze=False,
    )
    colors = {
        "Seasonal Naive": "#777777",
        "Moving Average": "#D18F00",
        "ARIMA": "#4C72B0",
        "VAR (first differences)": "#55A868",
        "VECM": "#8172B3",
        "GRU": "#C44E52",
    }
    for ax, aggregation in zip(axes[:, 0], aggregations):
        subset = metrics.loc[metrics["aggregation"] == aggregation]
        for model_name in MODEL_NAMES:
            series = subset.loc[subset["model"] == model_name].sort_values("horizon_weeks")
            if series["MAE"].notna().any():
                ax.plot(
                    series["horizon_weeks"],
                    series["MAE"],
                    marker="o",
                    color=colors[model_name],
                    label=model_name,
                )
        ax.set_title(f"Weekly target aggregation: {aggregation}")
        ax.set_ylabel("MAE (F&G points)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, framealpha=0.9)
    axes[-1, 0].set_xlabel("Forecast horizon (weeks)")
    fig.tight_layout()
    return fig, axes[:, 0]


def plot_latest_forecasts(
    weekly_frames: dict[str, pd.DataFrame],
    forecasts: pd.DataFrame,
) -> tuple[Figure, np.ndarray]:
    """Plot each aggregation's recent history and six-week candidate paths."""
    aggregations = [name for name in AGGREGATIONS if name in weekly_frames]
    fig, axes = plt.subplots(
        len(aggregations),
        1,
        figsize=(14, 4.2 * len(aggregations)),
        squeeze=False,
    )
    colors = {
        "Seasonal Naive": "#777777",
        "Moving Average": "#D18F00",
        "ARIMA": "#4C72B0",
        "VAR (first differences)": "#55A868",
        "VECM": "#8172B3",
        "GRU": "#C44E52",
    }
    for ax, aggregation in zip(axes[:, 0], aggregations):
        history = weekly_frames[aggregation]["target"].tail(52)
        ax.plot(history.index, history, color="#444444", linewidth=1.5, label="White Rabbit target")
        subset = forecasts.loc[
            (forecasts["aggregation"] == aggregation)
            & forecasts["prediction"].notna()
        ]
        for model_name in MODEL_NAMES:
            series = subset.loc[subset["model"] == model_name].sort_values("target_date")
            if not series.empty:
                ax.plot(
                    series["target_date"],
                    series["prediction"],
                    marker="o",
                    linewidth=1.2,
                    color=colors[model_name],
                    label=model_name,
                )
        visible_values = pd.concat(
            [history.reset_index(drop=True), subset["prediction"].reset_index(drop=True)],
            ignore_index=True,
        ).dropna()
        if not visible_values.empty:
            padding = max(2.0, 0.05 * float(visible_values.max() - visible_values.min()))
            ax.set_ylim(
                min(0.0, float(visible_values.min()) - padding),
                max(100.0, float(visible_values.max()) + padding),
            )
        ax.set_title(f"Six-week forecast: {aggregation} weekly target")
        ax.set_ylabel("Fear & Greed score")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, framealpha=0.9)
    axes[-1, 0].set_xlabel("Week ending Friday")
    fig.tight_layout()
    return fig, axes[:, 0]
