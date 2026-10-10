"""Weekly F&G targets, chronological holdouts, VAR diagnostics, and forecasts."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    "VAR (levels)",
    "VAR (first differences)",
    "VECM",
    "GRU",
)
BASE_MODEL_NAMES = ("Seasonal Naive", "Moving Average", "ARIMA", "GRU")
GRU_QUANTILES = (0.1, 0.5, 0.9)


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
    *,
    first_differences: bool,
) -> tuple[np.ndarray, str]:
    from statsmodels.tsa.api import VAR

    model_data = train.diff().dropna() if first_differences else train
    if len(model_data) < 12:
        raise ValueError("Too few weekly observations to fit VAR.")
    model = VAR(model_data)
    max_lags = min(4, max(1, len(model_data) // (2 * len(train.columns) + 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = model.select_order(maxlags=max_lags)
    lag_order = selection.selected_orders.get("bic") or 1
    lag_order = max(1, int(lag_order))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(maxlags=lag_order)
    model_forecast = fitted.forecast(
        model_data.to_numpy()[-fitted.k_ar :],
        steps=horizon,
    )
    target_position = train.columns.get_loc(target_column)
    if first_differences:
        model_forecast = train.iloc[-1].to_numpy() + np.cumsum(model_forecast, axis=0)
    return model_forecast[:, target_position], f"lag={fitted.k_ar}"


def _select_vecm_difference_lags(train: pd.DataFrame, max_lags: int = 4) -> int:
    from statsmodels.tsa.vector_ar.vecm import select_order

    capped_lags = min(max_lags, max(1, len(train) // (2 * train.shape[1] + 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = select_order(train, maxlags=capped_lags, deterministic="co")
    selected = selection.selected_orders.get("bic")
    return max(0, int(selected)) if selected is not None else 1


def _forecast_vecm(
    train: pd.DataFrame,
    horizon: int,
    target_column: str,
) -> tuple[np.ndarray, str]:
    from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

    k_ar_diff = _select_vecm_difference_lags(train)

    rank_result = select_coint_rank(
        train.to_numpy(dtype=float),
        det_order=0,
        k_ar_diff=k_ar_diff,
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
            k_ar_diff=k_ar_diff,
            coint_rank=rank,
            deterministic="co",
        ).fit()
    forecast = np.asarray(fitted.predict(steps=horizon), dtype=float)
    return forecast[:, train.columns.get_loc(target_column)], f"rank={rank}, diff_lags={k_ar_diff}"


def diagnose_var_specification(
    train_frame: pd.DataFrame,
    *,
    significance: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Use training-only ADF/KPSS and Johansen tests to screen VAR forms.

    A VAR in levels is recommended when every input is classified I(0). When
    every input is classified I(1), the Johansen trace test distinguishes a
    differenced VAR (rank zero) from a VECM (intermediate rank). Mixed or
    inconclusive stationarity results do not force a multivariate model.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.vector_ar.vecm import select_coint_rank

    if train_frame.empty or train_frame.isna().any().any():
        raise ValueError("Use a complete, non-empty training frame for diagnostics.")
    if not 0 < significance < 1:
        raise ValueError("significance must be between zero and one.")

    def stationarity_pvalues(values: np.ndarray) -> tuple[float, float]:
        if (
            len(values) < 20
            or not np.isfinite(values).all()
            or np.isclose(float(np.std(values)), 0.0)
        ):
            return np.nan, np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_p = float(adfuller(values, regression="c", autolag="BIC")[1])
            kpss_p = float(kpss(values, regression="c", nlags="auto")[1])
        return adf_p, kpss_p

    rows: list[dict[str, object]] = []
    for column in train_frame.columns:
        levels = train_frame[column].to_numpy(dtype=float)
        differences = np.diff(levels)
        adf_level, kpss_level = stationarity_pvalues(levels)
        adf_difference, kpss_difference = stationarity_pvalues(differences)
        level_stationary = adf_level <= significance and kpss_level > significance
        level_nonstationary = adf_level > significance and kpss_level <= significance
        difference_stationary = (
            adf_difference <= significance and kpss_difference > significance
        )
        if level_stationary:
            integration_order = "I(0)"
        elif level_nonstationary and difference_stationary:
            integration_order = "I(1)"
        else:
            integration_order = "inconclusive"
        rows.append(
            {
                "variable": column,
                "ADF p (level)": adf_level,
                "KPSS p (level)": kpss_level,
                "ADF p (difference)": adf_difference,
                "KPSS p (difference)": kpss_difference,
                "integration_order": integration_order,
            }
        )

    variable_tests = pd.DataFrame(rows)
    orders = variable_tests["integration_order"].tolist()
    summary: dict[str, object] = {
        "system_integration": "mixed/inconclusive",
        "cointegration_rank": None,
        "vecm_difference_lags": None,
        "recommended_model": None,
        "reason": "The series do not share a clearly identified integration order.",
    }
    if orders and all(order == "I(0)" for order in orders):
        summary.update(
            system_integration="all I(0)",
            recommended_model="VAR (levels)",
            reason="All inputs pass the level-stationarity rule (ADF and KPSS).",
        )
    elif orders and all(order == "I(1)" for order in orders):
        summary["system_integration"] = "all I(1)"
        try:
            k_ar_diff = _select_vecm_difference_lags(train_frame)
            summary["vecm_difference_lags"] = k_ar_diff
            rank_result = select_coint_rank(
                train_frame.to_numpy(dtype=float),
                det_order=0,
                k_ar_diff=k_ar_diff,
                method="trace",
                signif=significance,
            )
            rank = int(rank_result.rank)
            summary["cointegration_rank"] = rank
            if rank == 0:
                summary.update(
                    recommended_model="VAR (first differences)",
                    reason="All inputs look I(1); Johansen trace test selects rank zero.",
                )
            elif rank < train_frame.shape[1]:
                summary.update(
                    recommended_model="VECM",
                    reason=(
                        "All inputs look I(1); Johansen trace test finds an "
                        "intermediate cointegration rank."
                    ),
                )
            else:
                summary["reason"] = (
                    "The selected rank is full; it does not support the usual "
                    "I(1) VECM branch."
                )
        except Exception as exc:
            summary["reason"] = f"Johansen rank test unavailable: {type(exc).__name__}: {exc}"
    return variable_tests, summary


def _fit_gru_quantiles_impl(
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
    """Fit a direct six-week GRU with q10/q50/q90 pinball loss."""
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
    validation_target = torch.from_numpy(y_val)

    class MultiHorizonGRU(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, horizon * len(GRU_QUANTILES))

        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            encoded, _ = self.encoder(sequence)
            raw = self.head(encoded[:, -1, :]).view(-1, horizon, len(GRU_QUANTILES))
            median = raw[..., 1]
            lower = median - nn.functional.softplus(raw[..., 0])
            upper = median + nn.functional.softplus(raw[..., 2])
            return torch.stack([lower, median, upper], dim=-1)

    quantile_levels = torch.tensor(GRU_QUANTILES, dtype=torch.float32).view(1, 1, -1)

    def quantile_loss(actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        error = actual.unsqueeze(-1) - predicted
        return torch.maximum(quantile_levels * error, (quantile_levels - 1) * error).mean()

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
            loss = quantile_loss(batch_y, model(batch_X))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    torch.manual_seed(random_state)
    model = MultiHorizonGRU(input_size=train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    fit_loader = loader(X_fit, y_fit, random_state)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    for epoch in range(1, max_epochs + 1):
        train_epoch(model, fit_loader, optimizer)
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                quantile_loss(validation_target, model(torch.from_numpy(X_val)))
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("GRU training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    final_input = torch.from_numpy(validation_values[-lookback:]).unsqueeze(0)
    with torch.no_grad():
        forecast = model(final_input).squeeze(0).numpy()
    return forecast.astype(float), f"lookback={lookback}, selected_epochs={best_epoch}"


def _forecast_gru_quantiles(
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
    """Run the small CPU GRU with one intra-op thread, then restore PyTorch."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The forecast GRU requires PyTorch. Install it with `pip install torch`."
        ) from exc

    previous_threads = torch.get_num_threads()
    try:
        if previous_threads != 1:
            torch.set_num_threads(1)
        return _fit_gru_quantiles_impl(
            train,
            horizon,
            lookback=lookback,
            hidden_size=hidden_size,
            max_epochs=max_epochs,
            patience=patience,
            validation_fraction=validation_fraction,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)


def forecast_candidate_with_quantiles(
    model_name: str,
    train_frame: pd.DataFrame,
    *,
    horizon: int = 6,
    season_length: int = 52,
    moving_average_window: int = 4,
    gru_lookback: int = 26,
    gru_hidden_size: int = 32,
    gru_max_epochs: int = 50,
    gru_patience: int = 8,
    gru_validation_fraction: float = 0.15,
    gru_batch_size: int = 16,
    gru_learning_rate: float = 0.001,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Fit one candidate and return its point and optional quantile forecasts."""
    y = train_frame["target"].to_numpy(dtype=float)
    if model_name == "Seasonal Naive":
        point = _seasonal_naive(y, horizon, season_length)
        return point, None, f"season={season_length}"
    if model_name == "Moving Average":
        point = _moving_average(y, horizon, moving_average_window)
        return point, None, f"window={moving_average_window}"
    if model_name == "ARIMA":
        point, details = _forecast_arima(y, horizon)
        return point, None, details
    if model_name == "VAR (levels)":
        point, details = _forecast_var(
            train_frame,
            horizon,
            "target",
            first_differences=False,
        )
        return point, None, details
    if model_name == "VAR (first differences)":
        point, details = _forecast_var(
            train_frame,
            horizon,
            "target",
            first_differences=True,
        )
        return point, None, details
    if model_name == "VECM":
        point, details = _forecast_vecm(train_frame, horizon, "target")
        return point, None, details
    if model_name == "GRU":
        quantile_forecast, details = _forecast_gru_quantiles(
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
        return quantile_forecast[:, 1], quantile_forecast, details
    raise ValueError(f"Unknown forecast model: {model_name!r}.")


def forecast_candidate(
    model_name: str,
    train_frame: pd.DataFrame,
    **model_settings: object,
) -> tuple[np.ndarray, str]:
    """Fit a candidate and return its point path (q50 for the quantile GRU)."""
    point, _, details = forecast_candidate_with_quantiles(
        model_name,
        train_frame,
        **model_settings,
    )
    return point, details


def train_validation_test_evaluation(
    weekly_frame: pd.DataFrame,
    *,
    horizon: int = 6,
    validation_folds: int = 1,
    include_test: bool = True,
    models: Iterable[str] = MODEL_NAMES,
    season_length: int = 52,
    moving_average_window: int = 4,
    gru_lookback: int = 26,
    gru_hidden_size: int = 32,
    gru_max_epochs: int = 50,
    gru_patience: int = 8,
    gru_validation_fraction: float = 0.15,
    gru_batch_size: int = 16,
    gru_learning_rate: float = 0.001,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate six-week forecasts on chronological validation and test blocks.

    The final ``horizon`` weeks are reserved for the untouched test. The
    immediately preceding ``validation_folds * horizon`` weeks are validation,
    split into non-overlapping walk-forward blocks. Each validation fit sees
    only earlier weeks. The test fit sees all pre-test observations, including
    validation, and is run once. Set ``include_test=False`` for optional
    validation-only walk-forward analysis.
    """
    if horizon < 1 or validation_folds < 1:
        raise ValueError("horizon and validation_folds must be positive.")
    if "target" not in weekly_frame.columns:
        raise ValueError("Weekly frame must include a 'target' column.")

    test_start = len(weekly_frame) - horizon
    validation_start = test_start - validation_folds * horizon
    minimum_history = max(season_length, gru_lookback + horizon + 20)
    if validation_start < minimum_history:
        raise ValueError(
            "Not enough history before the validation window. "
            f"Need at least {minimum_history} training weeks, got {validation_start}."
        )

    requested_models = tuple(models)
    observed_by_week = weekly_frame.attrs.get("target_observed_by_week", {})
    source_by_week = weekly_frame.attrs.get("target_source_by_week", {})
    prediction_rows: list[dict[str, object]] = []

    def evaluate_block(
        *,
        split: str,
        fold: int | None,
        train_end: int,
        forecast_start: int,
    ) -> None:
        train = weekly_frame.iloc[:train_end].copy()
        actual_frame = weekly_frame.iloc[forecast_start : forecast_start + horizon]
        if len(actual_frame) != horizon:
            raise ValueError(f"The {split} block does not contain {horizon} weeks.")

        actual = actual_frame["target"].to_numpy(dtype=float).copy()
        observed_mask = np.asarray(
            [observed_by_week.get(date, True) for date in actual_frame.index],
            dtype=bool,
        )
        actual[~observed_mask] = np.nan

        for model_name in requested_models:
            status = "ok"
            details = ""
            quantile_forecast: np.ndarray | None = None
            try:
                point_forecast, quantile_forecast, details = forecast_candidate_with_quantiles(
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
                point_forecast = np.asarray(point_forecast, dtype=float)
                if point_forecast.shape != (horizon,):
                    raise ValueError(
                        f"Expected {horizon} point forecasts, got {point_forecast.shape}."
                    )
            except Exception as exc:  # Keep failed candidates visible in the table.
                point_forecast = np.full(horizon, np.nan)
                quantile_forecast = None
                status = f"unavailable: {type(exc).__name__}: {exc}"

            if quantile_forecast is None:
                q10 = q50 = q90 = np.full(horizon, np.nan)
            else:
                quantile_forecast = np.asarray(quantile_forecast, dtype=float)
                if quantile_forecast.shape != (horizon, len(GRU_QUANTILES)):
                    raise ValueError(
                        "Quantile forecasts must have shape "
                        f"({horizon}, {len(GRU_QUANTILES)})."
                    )
                q10, q50, q90 = quantile_forecast.T

            for step, (target_date, observed, predicted) in enumerate(
                zip(actual_frame.index, actual, point_forecast),
                start=1,
            ):
                prediction_rows.append(
                    {
                        "split": split,
                        "validation_fold": fold,
                        "origin_date": train.index[-1],
                        "target_date": target_date,
                        "horizon": step,
                        "model": model_name,
                        "actual": observed,
                        "prediction": predicted,
                        "prediction_q10": q10[step - 1],
                        "prediction_q50": q50[step - 1],
                        "prediction_q90": q90[step - 1],
                        "target_source": source_by_week.get(target_date, "unknown"),
                        "status": status,
                        "model_details": details,
                    }
                )

    for fold in range(validation_folds):
        forecast_start = validation_start + fold * horizon
        evaluate_block(
            split="validation",
            fold=fold + 1,
            train_end=forecast_start,
            forecast_start=forecast_start,
        )

    if include_test:
        evaluate_block(
            split="test",
            fold=None,
            train_end=test_start,
            forecast_start=test_start,
        )

    return pd.DataFrame(prediction_rows)


def summarize_forecast_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize point errors and q10/q50/q90 pinball loss by split and lead."""
    required = {"aggregation", "split", "model", "horizon", "actual", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing: {', '.join(sorted(missing))}.")

    rows: list[dict[str, object]] = []
    group_columns = ["aggregation", "split", "model", "horizon"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        valid = group.dropna(subset=["actual", "prediction"])
        agg_name, split, model_name, horizon = keys
        row: dict[str, object] = {
            "aggregation": agg_name,
            "split": split,
            "model": model_name,
            "horizon_weeks": int(horizon),
            "n_weeks": len(valid),
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
            "failed_predictions": int(group["prediction"].isna().sum()),
            "pinball_q10": np.nan,
            "pinball_q50": np.nan,
            "pinball_q90": np.nan,
            "q10_q90_coverage": np.nan,
        }
        quantile_valid = group.dropna(
            subset=["actual", "prediction_q10", "prediction_q50", "prediction_q90"]
        )
        if not quantile_valid.empty:
            actual = quantile_valid["actual"].to_numpy(dtype=float)
            for quantile, column in zip(GRU_QUANTILES, (
                "prediction_q10", "prediction_q50", "prediction_q90"
            )):
                error = actual - quantile_valid[column].to_numpy(dtype=float)
                row[f"pinball_q{int(quantile * 100):02d}"] = float(
                    np.maximum(quantile * error, (quantile - 1) * error).mean()
                )
            lower = quantile_valid["prediction_q10"].to_numpy(dtype=float)
            upper = quantile_valid["prediction_q90"].to_numpy(dtype=float)
            row["q10_q90_coverage"] = float(((actual >= lower) & (actual <= upper)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_latest_scenarios(
    weekly_frames: dict[str, pd.DataFrame],
    *,
    horizon: int = 6,
    models: Iterable[str] = MODEL_NAMES,
    **model_settings: object,
) -> pd.DataFrame:
    """Fit each requested candidate to all observed history and forecast ahead."""
    rows: list[dict[str, object]] = []
    requested_models = tuple(models)
    for aggregation, weekly_frame in weekly_frames.items():
        for model_name in requested_models:
            status = "ok"
            details = ""
            try:
                point_forecast, quantile_forecast, details = forecast_candidate_with_quantiles(
                    model_name,
                    weekly_frame,
                    horizon=horizon,
                    **model_settings,
                )
            except Exception as exc:
                point_forecast = np.full(horizon, np.nan)
                quantile_forecast = None
                details = ""
                status = f"unavailable: {type(exc).__name__}: {exc}"

            if quantile_forecast is None:
                q10 = q50 = q90 = np.full(horizon, np.nan)
            else:
                q10, q50, q90 = np.asarray(quantile_forecast, dtype=float).T
            future_dates = pd.date_range(
                weekly_frame.index[-1] + pd.offsets.Week(weekday=4),
                periods=horizon,
                freq="W-FRI",
            )
            for step, target_date in enumerate(future_dates):
                rows.append(
                    {
                        "aggregation": aggregation,
                        "model": model_name,
                        "origin_date": weekly_frame.index[-1],
                        "target_date": target_date,
                        "horizon": step + 1,
                        "prediction": point_forecast[step],
                        "prediction_q10": q10[step],
                        "prediction_q50": q50[step],
                        "prediction_q90": q90[step],
                        "status": status,
                        "model_details": details,
                    }
                )
    return pd.DataFrame(rows)


def plot_holdout_metrics(metrics: pd.DataFrame) -> tuple[Figure, np.ndarray]:
    """Plot MAE by forecast lead for validation and the final test block."""
    aggregations = [name for name in AGGREGATIONS if name in set(metrics["aggregation"])]
    fig, axes = plt.subplots(
        len(aggregations),
        2,
        figsize=(14, 4 * len(aggregations)),
        sharex=True,
        squeeze=False,
    )
    colors = {
        "Seasonal Naive": "#777777",
        "Moving Average": "#D18F00",
        "ARIMA": "#4C72B0",
        "VAR (levels)": "#8172B3",
        "VAR (first differences)": "#55A868",
        "VECM": "#CCB974",
        "GRU": "#C44E52",
    }
    for row, aggregation in enumerate(aggregations):
        for col, split in enumerate(("validation", "test")):
            ax = axes[row, col]
            subset = metrics.loc[
                (metrics["aggregation"] == aggregation)
                & (metrics["split"] == split)
            ]
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
            ax.set_title(f"{aggregation.capitalize()} target - {split.capitalize()}")
            ax.set_ylabel("MAE (F&G points)")
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, framealpha=0.9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Forecast horizon (weeks)")
    fig.tight_layout()
    return fig, axes


def plot_latest_forecasts(
    weekly_frames: dict[str, pd.DataFrame],
    forecasts: pd.DataFrame,
) -> tuple[Figure, np.ndarray]:
    """Plot recent history, candidate paths, and the GRU q10-q90 interval."""
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
        "VAR (levels)": "#8172B3",
        "VAR (first differences)": "#55A868",
        "VECM": "#CCB974",
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
                if (
                    model_name == "GRU"
                    and series[["prediction_q10", "prediction_q90"]].notna().all().all()
                ):
                    ax.fill_between(
                        pd.to_datetime(series["target_date"]),
                        series["prediction_q10"].to_numpy(dtype=float),
                        series["prediction_q90"].to_numpy(dtype=float),
                        color=colors[model_name],
                        alpha=0.16,
                        label="GRU q10-q90 range",
                    )
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
