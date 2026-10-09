"""Measure how White Rabbit's pre-2021 archive affects F&G calibration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_calibration_data import load_calibration_data


CNN_SOURCE = "cnn_endpoint"
ARCHIVE_SOURCE = "third_party_archive"
CNN_ONLY_HISTORY = "CNN-only training"
EXTENDED_HISTORY = "Extended archive + CNN training"


def _make_model(model_name: str) -> Pipeline:
    """Create a scaled regression pipeline for one calibration method."""
    if model_name == "OLS":
        regressor = LinearRegression()
    elif model_name == "Bayesian Ridge":
        regressor = BayesianRidge()
    else:
        raise ValueError(f"Unsupported calibration model: {model_name!r}.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def _predict(
    model: Pipeline,
    model_name: str,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return predictions and Bayesian predictive standard deviations."""
    scaled_features = model.named_steps["scaler"].transform(features)
    regressor = model.named_steps["regressor"]
    if model_name == "Bayesian Ridge":
        prediction, prediction_std = regressor.predict(
            scaled_features,
            return_std=True,
        )
        return prediction, prediction_std
    return regressor.predict(scaled_features), None


def compare_archive_calibration(
    *,
    history_frame: pd.DataFrame | None = None,
    start_date: str = "2011-01-03",
    history_start: str = "2005-01-01",
    test_years: int = 1,
    data_dir: str = "data/fred_cache",
    min_periods: int = 252,
    window: int | None = 1260,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    timeout: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare three baselines on the same recent CNN-backed test year.

    The function loads White Rabbit's combined history and builds two training
    sets ending before the test window: CNN-backed observations only, and the
    older third-party archive plus all available CNN-backed observations.
    Equal-weight, OLS, and Bayesian Ridge predictions are evaluated on the
    exact same latest CNN-backed dates in both scenarios. Equal-weight is not
    fitted; its row is repeated to make the comparison table explicit.

    Args:
        start_date: First date returned by the estimator and target loader.
            This must include White Rabbit's pre-2021 archive for the extended
            training scenario.
        history_frame: Optional preloaded combined history from
            ``load_calibration_data(source="all")``. Pass the frame created by
            the preceding source diagnostic to avoid downloading/recomputing
            the same data twice.
        history_start: Earlier date used to warm the estimator's rolling
            indicators and percentile transforms.
        test_years: Calendar-year width of the most recent CNN test period.
        data_dir: Directory used by the existing FRED data cache.
        min_periods: Minimum observations for component percentile scores.
        window: Rolling percentile window, or ``None`` for expanding history.
        lower_q: Lower winsorization quantile for component scores.
        upper_q: Upper winsorization quantile for component scores.
        timeout: HTTP timeout in seconds for White Rabbit's CSV download.

    Returns:
        A pair ``(metrics, predictions)``. The first DataFrame has one row for
        each training-history/model combination. The second contains daily
        actuals, predictions, and Bayesian predictive standard deviations for
        plotting the same test window.

    Raises:
        ValueError: If the history, archive era, CNN test period, or a training
            set is missing after complete-case alignment.
    """
    if test_years < 1:
        raise ValueError("test_years must be at least 1.")

    if history_frame is None:
        history = load_calibration_data(
            start_date=start_date,
            history_start=history_start,
            data_dir=data_dir,
            source="all",
            min_periods=min_periods,
            window=window,
            lower_q=lower_q,
            upper_q=upper_q,
            timeout=timeout,
        )
    else:
        history = history_frame.copy()
    feature_candidates = history.attrs.get("feature_columns", [])
    if "source" not in history.columns:
        raise ValueError("The loaded history has no target-source column.")
    cnn_mask = history["source"] == CNN_SOURCE
    feature_columns = [
        column
        for column in feature_candidates
        if history.loc[cnn_mask, column].notna().any()
    ]
    if not feature_columns:
        raise ValueError(
            "No non-empty component score columns are available in the CNN era."
        )

    required_columns = feature_columns + [
        "FG_est_mean",
        "fear_greed_score",
        "source",
    ]
    missing_columns = set(required_columns).difference(history.columns)
    if missing_columns:
        raise ValueError(
            "The loaded history is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    complete_history = history.dropna(subset=required_columns).sort_index()
    cnn_history = complete_history.loc[
        complete_history["source"] == CNN_SOURCE
    ]
    if cnn_history.empty:
        raise ValueError("No complete CNN-backed rows are available.")

    latest_cnn_date = cnn_history.index.max()
    test_cutoff = latest_cnn_date - pd.DateOffset(years=test_years)
    test_frame = cnn_history.loc[cnn_history.index > test_cutoff].copy()
    cnn_train = cnn_history.loc[cnn_history.index <= test_cutoff].copy()
    extended_train = complete_history.loc[
        complete_history.index <= test_cutoff
    ].copy()

    if test_frame.empty:
        raise ValueError("No complete CNN rows fall inside the requested test window.")
    if cnn_train.empty or extended_train.empty:
        raise ValueError("Not enough complete history is available before the test window.")
    if not extended_train["source"].eq(ARCHIVE_SOURCE).any():
        raise ValueError(
            "The extended training set contains no pre-2021 archive rows. "
            "Check start_date and White Rabbit's source column."
        )

    training_scenarios = [
        (CNN_ONLY_HISTORY, cnn_train),
        (EXTENDED_HISTORY, extended_train),
    ]
    actual = test_frame["fear_greed_score"].to_numpy(dtype=float)
    test_dates = test_frame.index
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    def record_result(
        history_name: str,
        model_name: str,
        prediction: np.ndarray,
        prediction_std: np.ndarray | None,
        training_frame: pd.DataFrame | None,
    ) -> None:
        if prediction_std is None:
            coverage_95 = np.nan
            std_values = np.full(len(prediction), np.nan)
        else:
            lower_95 = prediction - 1.96 * prediction_std
            upper_95 = prediction + 1.96 * prediction_std
            coverage_95 = float(
                np.mean((actual >= lower_95) & (actual <= upper_95))
            )
            std_values = prediction_std

        metric_rows.append(
            {
                "training_history": history_name,
                "model": model_name,
                "n_train": None if training_frame is None else len(training_frame),
                "n_archive_train": (
                    None
                    if training_frame is None
                    else int(training_frame["source"].eq(ARCHIVE_SOURCE).sum())
                ),
                "n_cnn_train": (
                    None
                    if training_frame is None
                    else int(training_frame["source"].eq(CNN_SOURCE).sum())
                ),
                "train_start": (
                    None if training_frame is None else training_frame.index.min().date()
                ),
                "train_end": (
                    None if training_frame is None else training_frame.index.max().date()
                ),
                "test_start": test_dates.min().date(),
                "test_end": test_dates.max().date(),
                "n_test": len(test_frame),
                "MAE": float(mean_absolute_error(actual, prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
                "R2": float(r2_score(actual, prediction)),
                "coverage_95": coverage_95,
            }
        )

        for date, observed, predicted, predicted_std in zip(
            test_dates,
            actual,
            prediction,
            std_values,
        ):
            prediction_rows.append(
                {
                    "date": date,
                    "training_history": history_name,
                    "model": model_name,
                    "actual": observed,
                    "prediction": predicted,
                    "prediction_std": predicted_std,
                }
            )

    for history_name, training_frame in training_scenarios:
        # Equal-weight scores do not use reference targets, so this baseline is
        # identical across the two training-history scenarios.
        record_result(
            history_name=history_name,
            model_name="Equal-weight",
            prediction=test_frame["FG_est_mean"].to_numpy(dtype=float),
            prediction_std=None,
            training_frame=None,
        )

        X_train = training_frame[feature_columns]
        y_train = training_frame["fear_greed_score"]
        X_test = test_frame[feature_columns]
        for model_name in ("OLS", "Bayesian Ridge"):
            model = _make_model(model_name)
            model.fit(X_train, y_train)
            prediction, prediction_std = _predict(model, model_name, X_test)
            record_result(
                history_name=history_name,
                model_name=model_name,
                prediction=prediction,
                prediction_std=prediction_std,
                training_frame=training_frame,
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    for frame in (metrics, predictions):
        frame.attrs["test_start"] = test_dates.min().strftime("%Y-%m-%d")
        frame.attrs["test_end"] = test_dates.max().strftime("%Y-%m-%d")
        frame.attrs["test_target_source"] = CNN_SOURCE
        frame.attrs["white_rabbit_source_url"] = history.attrs.get(
            "white_rabbit_source_url"
        )
        frame.attrs["white_rabbit_fetched_at_utc"] = history.attrs.get(
            "white_rabbit_fetched_at_utc"
        )

    return metrics, predictions


def plot_archive_sensitivity(
    predictions: pd.DataFrame,
) -> tuple[Figure, np.ndarray]:
    """Plot the equal-weight, OLS, and Bayesian comparisons over the test year."""
    required = {
        "date",
        "training_history",
        "model",
        "actual",
        "prediction",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions are missing required columns: {', '.join(sorted(missing))}."
        )

    actual_rows = predictions.loc[
        (predictions["training_history"] == CNN_ONLY_HISTORY)
        & (predictions["model"] == "Equal-weight")
    ].sort_values("date")
    if actual_rows.empty:
        raise ValueError("No CNN-only equal-weight predictions are available to plot.")

    actual = actual_rows.set_index("date")["actual"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    actual_color = "#555555"
    scenario_colors = {
        CNN_ONLY_HISTORY: "#2878B5",
        EXTENDED_HISTORY: "#E07A3F",
    }

    panels = [
        (axes[0], "Equal-weight", "Equal-weight baseline"),
        (axes[1], "OLS", "OLS calibration"),
        (axes[2], "Bayesian Ridge", "Bayesian Ridge calibration"),
    ]
    for ax, model_name, title in panels:
        ax.plot(
            actual.index,
            actual,
            color=actual_color,
            linewidth=1.5,
            label="White Rabbit CNN target",
        )

        histories = (
            [CNN_ONLY_HISTORY]
            if model_name == "Equal-weight"
            else [CNN_ONLY_HISTORY, EXTENDED_HISTORY]
        )
        for history_name in histories:
            series = predictions.loc[
                (predictions["training_history"] == history_name)
                & (predictions["model"] == model_name)
            ].sort_values("date")
            ax.plot(
                series["date"],
                series["prediction"],
                color=scenario_colors[history_name],
                linewidth=1.2,
                label=history_name,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(100, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel("F&G score")
        ax.grid(alpha=0.2)
        ax.legend(framealpha=0.95)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Effect of Adding Pre-2021 Archive History: Latest CNN Year")
    fig.tight_layout()
    plt.show()
    return fig, axes
