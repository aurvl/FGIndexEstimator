"""Compare linear and sequence-based calibrators on CNN-backed F&G data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


CNN_SOURCE = "cnn_endpoint"


def compare_gru_with_bayesian(
    *,
    history_frame: pd.DataFrame,
    test_years: int = 1,
    sequence_length: int = 20,
    hidden_size: int = 32,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    max_epochs: int = 250,
    patience: int = 20,
    validation_fraction: float = 0.15,
    batch_size: int = 32,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare four calibrators and their simple mean on the same CNN year.

    The GRU predicts each day's CNN target from a rolling sequence of the
    component scores ending on that date. Training and test targets are
    restricted to the CNN-backed period. A chronological validation tail of
    the training period selects the epoch count; then the GRU is re-fitted on
    all pre-test sequences for that fixed number of epochs. The latest
    ``test_years`` calendar years are never used for model selection.

    PyTorch is optional for the base application and is imported only when
    this function is called.

    Args:
        history_frame: Date-indexed frame returned by
            ``load_calibration_data(source="cnn")``. It must include feature
            and target metadata in ``DataFrame.attrs``.
        test_years: Calendar-year width of the latest test window.
        sequence_length: Number of consecutive daily rows in each GRU input.
        hidden_size: Number of hidden units in each GRU layer.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between stacked GRU layers.
        learning_rate: Adam learning rate.
        weight_decay: L2 penalty applied by Adam.
        max_epochs: Maximum epochs used for chronological validation.
        patience: Epochs without validation improvement before stopping.
        validation_fraction: Chronological tail fraction used for early
            stopping and epoch selection.
        batch_size: Number of sequences per optimizer step.
        random_state: Seed for reproducible initialization and batch order.

    Returns:
        ``(metrics, predictions)``. Metrics contain one row per calibrator
        and the fixed-weight simple-mean ensemble; predictions are in long
        format for plotting.

    Raises:
        ImportError: If PyTorch is not installed in the notebook environment.
        ValueError: If inputs or chronological train/validation/test windows
            are insufficient for the comparison.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from gru_model import ComponentGRURegressor
    except ImportError as exc:
        raise ImportError(
            "The GRU benchmark requires PyTorch. Install it in the notebook "
            "environment with `pip install torch`."
        ) from exc

    if test_years < 1:
        raise ValueError("test_years must be at least 1.")
    if sequence_length < 1 or hidden_size < 1 or num_layers < 1:
        raise ValueError("sequence_length, hidden_size, and num_layers must be positive.")
    if num_layers > 1 and not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1) for a multi-layer GRU.")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    history = history_frame.copy()
    feature_candidates = history.attrs.get("feature_columns", [])
    target_column = history.attrs.get("target_column", "fear_greed_score")
    if "source" not in history.columns:
        raise ValueError("The loaded history has no target-source column.")

    cnn_mask = history["source"] == CNN_SOURCE
    feature_columns = [
        column
        for column in feature_candidates
        if column in history.columns and history.loc[cnn_mask, column].notna().any()
    ]
    required_columns = feature_columns + ["FG_est_mean", target_column, "source"]
    missing_columns = set(required_columns).difference(history.columns)
    if missing_columns:
        raise ValueError(
            "The loaded history is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    if not feature_columns:
        raise ValueError("No non-empty component score columns are available.")

    complete = history.dropna(subset=required_columns).sort_index()
    cnn_history = complete.loc[complete["source"] == CNN_SOURCE]
    if cnn_history.empty:
        raise ValueError("No complete CNN-backed observations are available.")

    latest_date = cnn_history.index.max()
    cutoff = latest_date - pd.DateOffset(years=test_years)
    train_frame = cnn_history.loc[cnn_history.index <= cutoff]
    test_frame = cnn_history.loc[cnn_history.index > cutoff]
    if train_frame.empty or test_frame.empty:
        raise ValueError("The requested CNN-only train/test split is empty.")

    # Fit Bayesian Ridge on the same pre-test dates and component columns.
    X_train = train_frame[feature_columns]
    X_test = test_frame[feature_columns]
    bayesian_scaler = StandardScaler().fit(train_frame[feature_columns])
    X_train_bayesian = bayesian_scaler.transform(train_frame[feature_columns])
    X_test_bayesian = bayesian_scaler.transform(test_frame[feature_columns])
    bayesian = BayesianRidge().fit(X_train_bayesian, train_frame[target_column])
    bayesian_prediction, bayesian_std = bayesian.predict(
        X_test_bayesian,
        return_std=True,
    )

    # Build all rolling inputs once. Each sequence ends on its target date;
    # the first test sequence may use earlier training-period features.
    dates = cnn_history.index
    raw_features = cnn_history[feature_columns].to_numpy(dtype=np.float32)
    targets = cnn_history[target_column].to_numpy(dtype=np.float32)
    train_end_position = int(dates.get_indexer([train_frame.index[-1]])[0])
    test_start_position = int(dates.get_indexer([test_frame.index[0]])[0])
    first_target_position = sequence_length - 1
    if train_end_position < first_target_position:
        raise ValueError("There are too few CNN rows to build the requested sequences.")

    train_positions = np.arange(first_target_position, train_end_position + 1)
    validation_count = max(1, int(np.ceil(len(train_positions) * validation_fraction)))
    if validation_count >= len(train_positions):
        raise ValueError("Not enough training sequences for a chronological validation tail.")
    fit_positions = train_positions[:-validation_count]
    validation_positions = train_positions[-validation_count:]
    validation_start_position = int(validation_positions[0])
    if len(fit_positions) < 2:
        raise ValueError("Too few pre-validation sequences to fit the GRU.")

    # Fit the epoch-selection scaler without using validation or test features.
    selection_scaler = StandardScaler().fit(raw_features[:validation_start_position])
    selection_features = selection_scaler.transform(raw_features).astype(np.float32)

    def make_sequences(end_positions: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                selection_features[position - sequence_length + 1 : position + 1]
                for position in end_positions
            ]
        )

    X_fit = make_sequences(fit_positions)
    y_fit = targets[fit_positions]
    X_validation = make_sequences(validation_positions)
    y_validation = targets[validation_positions]

    def make_model() -> ComponentGRURegressor:
        return ComponentGRURegressor(
            input_size=len(feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def make_loader(
        features: np.ndarray,
        values: np.ndarray,
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(values),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def train_one_epoch(
        model: ComponentGRURegressor,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            prediction = model(batch_features)
            loss = nn.functional.mse_loss(prediction, batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_targets)
            total_rows += len(batch_targets)
        return total_loss / total_rows

    def evaluate_loss(
        model: ComponentGRURegressor,
        features: np.ndarray,
        values: np.ndarray,
    ) -> float:
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(features))
            return float(nn.functional.mse_loss(prediction, torch.from_numpy(values)))

    torch.manual_seed(random_state)
    selection_model = make_model()
    selection_optimizer = torch.optim.Adam(
        selection_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    fit_loader = make_loader(X_fit, y_fit, shuffle=True, seed=random_state)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        train_one_epoch(selection_model, fit_loader, selection_optimizer)
        validation_loss = evaluate_loss(selection_model, X_validation, y_validation)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_epoch == 0:
        raise RuntimeError("GRU training failed to record a valid validation epoch.")

    # Refit on all available training sequences for the selected number of epochs.
    final_scaler = StandardScaler().fit(raw_features[: train_end_position + 1])
    scaled_features = final_scaler.transform(raw_features).astype(np.float32)

    def make_final_sequences(end_positions: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                scaled_features[position - sequence_length + 1 : position + 1]
                for position in end_positions
            ]
        )

    final_train_positions = np.arange(first_target_position, train_end_position + 1)
    test_positions = np.arange(test_start_position, len(cnn_history))
    X_final_train = make_final_sequences(final_train_positions)
    y_final_train = targets[final_train_positions]
    X_test_gru = make_final_sequences(test_positions)
    torch.manual_seed(random_state)
    gru = make_model()
    optimizer = torch.optim.Adam(
        gru.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    final_loader = make_loader(
        X_final_train,
        y_final_train,
        shuffle=True,
        seed=random_state,
    )
    for _ in range(best_epoch):
        gru_final_loss = train_one_epoch(gru, final_loader, optimizer)
    gru.eval()
    with torch.no_grad():
        gru_prediction = gru(torch.from_numpy(X_test_gru)).numpy()

    actual = test_frame[target_column].to_numpy(dtype=float)
    ols_scaler = StandardScaler().fit(train_frame[feature_columns])
    ols = LinearRegression().fit(
        ols_scaler.transform(train_frame[feature_columns]),
        train_frame[target_column],
    )
    ols_prediction = ols.predict(ols_scaler.transform(X_test))

    predictions_by_model = {
        "Equal-weight": test_frame["FG_est_mean"].to_numpy(dtype=float),
        "OLS": ols_prediction,
        "Bayesian Ridge": bayesian_prediction,
        "GRU": gru_prediction,
    }
    # A transparent, non-fitted ensemble: every candidate gets the same
    # contribution on each date in the common test window.
    ensemble_members = ("Equal-weight", "OLS", "Bayesian Ridge", "GRU")
    predictions_by_model["Simple Mean Ensemble"] = np.mean(
        np.column_stack([predictions_by_model[name] for name in ensemble_members]),
        axis=1,
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_name, prediction in predictions_by_model.items():
        if model_name == "Bayesian Ridge":
            lower_95 = prediction - 1.96 * bayesian_std
            upper_95 = prediction + 1.96 * bayesian_std
            coverage_95 = float(np.mean((actual >= lower_95) & (actual <= upper_95)))
        else:
            coverage_95 = np.nan

        metric_rows.append(
            {
                "model": model_name,
                "n_train": (
                    len(final_train_positions)
                    if model_name == "GRU"
                    else len(train_frame)
                    if model_name == "Bayesian Ridge"
                    else np.nan
                ),
                "train_start": (
                    dates[first_target_position].date()
                    if model_name == "GRU"
                    else train_frame.index.min().date()
                    if model_name == "Bayesian Ridge"
                    else None
                ),
                "train_end": (
                    train_frame.index.max().date() if model_name != "Equal-weight" else None
                ),
                "test_start": test_frame.index.min().date(),
                "test_end": test_frame.index.max().date(),
                "n_test": len(test_frame),
                "MAE": float(mean_absolute_error(actual, prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
                "R2": float(r2_score(actual, prediction)),
                "coverage_95": coverage_95,
                "gru_sequence_length": sequence_length if model_name == "GRU" else np.nan,
                "gru_best_epoch": best_epoch if model_name == "GRU" else np.nan,
                "gru_validation_best_loss": (
                    best_validation_loss if model_name == "GRU" else np.nan
                ),
                "gru_final_train_loss": gru_final_loss if model_name == "GRU" else np.nan,
            }
        )
        for date, observed, predicted in zip(test_frame.index, actual, prediction):
            prediction_rows.append(
                {
                    "date": date,
                    "model": model_name,
                    "actual": observed,
                    "prediction": predicted,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    # Preserve the fitted post-2021 Bayesian Ridge parameters so the notebook
    # can serialize this tested model without fitting it a second time.
    metrics.attrs["bayesian_checkpoint"] = {
        "feature_columns": feature_columns,
        "scaler_mean": bayesian_scaler.mean_.astype(float).tolist(),
        "scaler_scale": bayesian_scaler.scale_.astype(float).tolist(),
        "coef": bayesian.coef_.astype(float).tolist(),
        "intercept": float(bayesian.intercept_),
        "posterior_covariance": bayesian.sigma_.astype(float).tolist(),
        "alpha": float(bayesian.alpha_),
        "lambda": float(bayesian.lambda_),
        "train_start": train_frame.index.min().strftime("%Y-%m-%d"),
        "train_end": train_frame.index.max().strftime("%Y-%m-%d"),
        "test_start": test_frame.index.min().strftime("%Y-%m-%d"),
        "test_end": test_frame.index.max().strftime("%Y-%m-%d"),
        "target_source": CNN_SOURCE,
    }
    for frame in (metrics, predictions):
        frame.attrs["test_start"] = test_frame.index.min().strftime("%Y-%m-%d")
        frame.attrs["test_end"] = test_frame.index.max().strftime("%Y-%m-%d")
        frame.attrs["test_target_source"] = CNN_SOURCE
        frame.attrs["feature_columns"] = feature_columns
        frame.attrs["gru_sequence_length"] = sequence_length
        frame.attrs["gru_hidden_size"] = hidden_size
        frame.attrs["gru_num_layers"] = num_layers
    return metrics, predictions


def plot_gru_comparison(
    predictions: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Plot the latest CNN test year and four calibrators plus their ensemble."""
    required = {"date", "model", "actual", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions are missing required columns: {', '.join(sorted(missing))}."
        )

    fig, ax = plt.subplots(figsize=(15, 6))
    actual_rows = predictions.loc[
        predictions["model"] == "Equal-weight"
    ].sort_values("date")
    if actual_rows.empty:
        raise ValueError("No equal-weight rows are available to recover test actuals.")
    ax.plot(
        actual_rows["date"],
        actual_rows["actual"],
        color="#666666",
        linewidth=1.7,
        label="White Rabbit CNN target",
    )

    colors = {
        "Equal-weight": "#D18F00",
        "OLS": "#777777",
        "Bayesian Ridge": "#2878B5",
        "GRU": "#C44E52",
        "Simple Mean Ensemble": "#7A5195",
    }
    for model_name in (
        "Equal-weight",
        "OLS",
        "Bayesian Ridge",
        "GRU",
        "Simple Mean Ensemble",
    ):
        series = predictions.loc[predictions["model"] == model_name].sort_values("date")
        if series.empty:
            continue
        ax.plot(
            series["date"],
            series["prediction"],
            color=colors[model_name],
            linewidth=1.25,
            alpha=0.9,
            label=model_name,
        )

    ax.set_title("CNN-only calibration: latest-year benchmark comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fear & Greed score")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return fig, ax
