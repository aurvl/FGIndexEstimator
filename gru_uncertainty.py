"""Compare point and probabilistic GRUs using all archive and CNN history."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


CNN_SOURCE = "cnn_endpoint"
ARCHIVE_SOURCE = "third_party_archive"
QUANTILES = (0.1, 0.5, 0.9)


def _pinball_loss(predictions, targets, quantiles, torch):
    errors = targets.unsqueeze(1) - predictions
    levels = torch.as_tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
    return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()


def compare_probabilistic_grus(
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
    """Fit three GRUs on all history and score the latest CNN-backed year.

    Training and validation use every complete target row, including the
    pre-2021 third-party archive and post-2021 CNN-backed scores. Evaluation is
    restricted to the latest ``test_years`` of CNN-backed scores. All models
    estimate the same-day target from a rolling sequence ending on that date.
    Epoch selection uses a chronological validation tail before the CNN test.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from gru_model import (
            ComponentGRURegressor,
            GaussianComponentGRU,
            QuantileComponentGRU,
        )
    except ImportError as exc:
        raise ImportError("Install PyTorch with `pip install torch` to run these GRUs.") from exc

    if test_years < 1 or sequence_length < 1 or hidden_size < 1 or num_layers < 1:
        raise ValueError("test_years and GRU dimensions must be positive.")
    if num_layers > 1 and not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1) for a multi-layer GRU.")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    target_column = history_frame.attrs.get("target_column", "fear_greed_score")
    feature_candidates = list(history_frame.attrs.get("feature_columns", []))
    required_base = [target_column, "source"]
    missing = set(required_base).difference(history_frame.columns)
    if missing:
        raise ValueError(f"History is missing columns: {', '.join(sorted(missing))}.")

    cnn_source_rows = history_frame.loc[history_frame["source"] == CNN_SOURCE]
    feature_columns = [
        column
        for column in feature_candidates
        if column in history_frame.columns
        and cnn_source_rows[column].notna().any()
    ]
    excluded_features = [
        column
        for column in feature_candidates
        if column not in feature_columns
    ]
    if not feature_columns:
        raise ValueError(
            "No component-score feature has any non-null values in the CNN period."
        )

    # Equal-weight is not a GRU input. Also exclude candidate features that
    # have no CNN-era values, since they cannot be supplied at evaluation time.
    required = feature_columns + [target_column, "source"]

    complete = history_frame.dropna(subset=required).sort_index()
    cnn = complete.loc[complete["source"] == CNN_SOURCE]
    if cnn.empty:
        source_rows = cnn_source_rows
        available_targets = int(source_rows[target_column].notna().sum())
        raise ValueError(
            "No complete CNN-backed rows remain after aligning component features "
            f"with '{target_column}'. CNN source rows={len(source_rows)}, "
            f"non-null targets={available_targets}, features used={feature_columns}, "
            f"features excluded for zero CNN coverage={excluded_features}."
        )

    cutoff = cnn.index.max() - pd.DateOffset(years=test_years)
    train_frame = complete.loc[complete.index <= cutoff]
    test_frame = cnn.loc[cnn.index > cutoff]
    if train_frame.empty or test_frame.empty:
        raise ValueError("The requested all-history train/CNN-test split is empty.")
    dates = complete.index
    train_end_position = int(dates.get_indexer([train_frame.index[-1]])[0])
    test_end_positions = np.flatnonzero(
        (complete.index > cutoff) & (complete["source"].to_numpy() == CNN_SOURCE)
    )
    first_target = sequence_length - 1
    if train_end_position < first_target:
        raise ValueError("There are too few historical rows for the requested sequence length.")

    train_positions = np.arange(first_target, train_end_position + 1)
    validation_count = max(1, int(np.ceil(len(train_positions) * validation_fraction)))
    if validation_count >= len(train_positions) - 1:
        raise ValueError("Too few pre-test sequences for the chronological validation split.")
    fit_positions = train_positions[:-validation_count]
    validation_positions = train_positions[-validation_count:]

    raw_X = complete[feature_columns].to_numpy(dtype=np.float32)
    y = complete[target_column].to_numpy(dtype=np.float32)
    validation_start = int(validation_positions[0])
    selection_scaler = StandardScaler().fit(raw_X[:validation_start])
    selection_X = selection_scaler.transform(raw_X).astype(np.float32)

    def sequences(scaled: np.ndarray, end_positions: np.ndarray) -> np.ndarray:
        return np.stack(
            [scaled[pos - sequence_length + 1 : pos + 1] for pos in end_positions]
        )

    X_fit = sequences(selection_X, fit_positions)
    y_fit = y[fit_positions]
    X_validation = sequences(selection_X, validation_positions)
    y_validation = y[validation_positions]

    def make_loader(x, target, *, seed: int):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(target)),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

    results: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_gru_models: dict[str, dict[str, object]] = {}
    model_specs = (
        ("GRU point (MSE)", ComponentGRURegressor, "point"),
        ("GRU Gaussian NLL", GaussianComponentGRU, "gaussian_nll"),
        ("GRU Quantile 10/50/90", QuantileComponentGRU, "quantile"),
    )

    for model_offset, (model_name, model_class, objective) in enumerate(model_specs):
        def make_model():
            return model_class(
                input_size=len(feature_columns),
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

        def objective_loss(prediction, target):
            if objective == "gaussian_nll":
                mu, sigma = prediction
                return (
                    torch.log(sigma)
                    + 0.5 * ((target - mu) / sigma) ** 2
                    + 0.5 * np.log(2.0 * np.pi)
                ).mean()
            if objective == "quantile":
                return _pinball_loss(prediction, target, QUANTILES, torch)
            return nn.functional.mse_loss(prediction, target)

        def train_epoch(model, loader, optimizer) -> float:
            model.train()
            loss_total = 0.0
            count = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = objective_loss(model(batch_X), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_total += float(loss.detach()) * len(batch_y)
                count += len(batch_y)
            return loss_total / count

        torch.manual_seed(random_state + model_offset)
        selection_model = make_model()
        optimizer = torch.optim.Adam(
            selection_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        fit_loader = make_loader(X_fit, y_fit, seed=random_state + model_offset)
        best_loss = float("inf")
        best_epoch = 0
        wait = 0
        for epoch in range(1, max_epochs + 1):
            train_epoch(selection_model, fit_loader, optimizer)
            selection_model.eval()
            with torch.no_grad():
                val_loss = float(
                    objective_loss(
                        selection_model(torch.from_numpy(X_validation)),
                        torch.from_numpy(y_validation),
                    )
                )
            if val_loss < best_loss:
                best_loss, best_epoch, wait = val_loss, epoch, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_epoch == 0:
            raise RuntimeError(f"No valid epoch selected for {model_name}.")

        final_scaler = StandardScaler().fit(raw_X[: train_end_position + 1])
        final_X = final_scaler.transform(raw_X).astype(np.float32)
        final_train_positions = np.arange(first_target, train_end_position + 1)
        X_final_train = sequences(final_X, final_train_positions)
        y_final_train = y[final_train_positions]
        X_test = sequences(final_X, test_end_positions)
        torch.manual_seed(random_state + model_offset)
        final_model = make_model()
        final_optimizer = torch.optim.Adam(
            final_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        final_loader = make_loader(
            X_final_train,
            y_final_train,
            seed=random_state + model_offset,
        )
        for _ in range(best_epoch):
            train_epoch(final_model, final_loader, final_optimizer)
        final_model.eval()
        with torch.no_grad():
            output = final_model(torch.from_numpy(X_test))

        fitted_gru_models[model_name] = {
            "objective": objective,
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in final_model.state_dict().items()
            },
            "feature_columns": feature_columns,
            "sequence_length": sequence_length,
            "input_size": len(feature_columns),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "scaler_mean": final_scaler.mean_.astype(float).tolist(),
            "scaler_scale": final_scaler.scale_.astype(float).tolist(),
            "selected_epochs": best_epoch,
            "validation_objective": best_loss,
            "train_start": train_frame.index.min().strftime("%Y-%m-%d"),
            "train_end": train_frame.index.max().strftime("%Y-%m-%d"),
            "training_sources": sorted(train_frame["source"].dropna().unique().tolist()),
            "n_archive_train": int(train_frame["source"].eq(ARCHIVE_SOURCE).sum()),
            "n_cnn_train": int(train_frame["source"].eq(CNN_SOURCE).sum()),
            "test_start": test_frame.index.min().strftime("%Y-%m-%d"),
            "test_end": test_frame.index.max().strftime("%Y-%m-%d"),
        }

        actual = test_frame[target_column].to_numpy(dtype=float)
        if objective == "point":
            point_prediction = output.numpy().astype(float)
            interval_coverage = np.nan
            mean_width = np.nan
            nll = np.nan
            sigma_abs_error_spearman = np.nan
            predictions = pd.DataFrame(
                {
                    "date": test_frame.index,
                    "actual": actual,
                    "prediction": point_prediction,
                }
            )
        elif objective == "gaussian_nll":
            mu, sigma = (part.numpy().astype(float) for part in output)
            lower = mu - 1.6448536269514722 * sigma
            upper = mu + 1.6448536269514722 * sigma
            nll = float(
                np.mean(
                    np.log(sigma)
                    + 0.5 * ((actual - mu) / sigma) ** 2
                    + 0.5 * np.log(2.0 * np.pi)
                )
            )
            interval_coverage = float(np.mean((actual >= lower) & (actual <= upper)))
            mean_width = float(np.mean(upper - lower))
            sigma_abs_error_spearman = float(
                pd.Series(sigma).corr(pd.Series(np.abs(actual - mu)), method="spearman")
            )
            predictions = pd.DataFrame(
                {"date": test_frame.index, "actual": actual, "prediction": mu,
                 "sigma": sigma, "lower_90": lower, "upper_90": upper}
            )
        else:
            quantile_predictions = output.numpy().astype(float)
            q10, q50, q90 = quantile_predictions.T
            interval_coverage = float(np.mean((actual >= q10) & (actual <= q90)))
            mean_width = float(np.mean(q90 - q10))
            nll = np.nan
            sigma_abs_error_spearman = np.nan
            predictions = pd.DataFrame(
                {"date": test_frame.index, "actual": actual, "prediction": q50,
                 "q10": q10, "q50": q50, "q90": q90}
            )

        results.append(
            {
                "model": model_name,
                "n_test": len(test_frame),
                "MAE": float(mean_absolute_error(actual, predictions["prediction"])),
                "RMSE": float(np.sqrt(mean_squared_error(actual, predictions["prediction"]))),
                "R2": float(r2_score(actual, predictions["prediction"])),
                "validation_objective": best_loss,
                "best_epoch": best_epoch,
                "n_train": len(train_frame),
                "n_archive_train": int(train_frame["source"].eq(ARCHIVE_SOURCE).sum()),
                "n_cnn_train": int(train_frame["source"].eq(CNN_SOURCE).sum()),
                "NLL": nll,
                "interval_90_coverage": interval_coverage,
                "interval_90_mean_width": mean_width,
                "sigma_abs_error_spearman": sigma_abs_error_spearman,
            }
        )
        predictions["model"] = model_name
        prediction_frames.append(predictions)

    metrics = pd.DataFrame(results)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.attrs["test_start"] = test_frame.index.min().strftime("%Y-%m-%d")
    metrics.attrs["test_end"] = test_frame.index.max().strftime("%Y-%m-%d")
    metrics.attrs["target_source"] = CNN_SOURCE
    metrics.attrs["training_sources"] = [ARCHIVE_SOURCE, CNN_SOURCE]
    metrics.attrs["feature_columns"] = feature_columns
    metrics.attrs["excluded_features_zero_cnn_coverage"] = excluded_features
    metrics.attrs["fitted_gru_models"] = fitted_gru_models
    return metrics, predictions


def plot_probabilistic_grus(
    predictions: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Plot all three GRU point estimates and uncertainty intervals."""
    fig, ax = plt.subplots(figsize=(15, 6))
    point = predictions.loc[predictions["model"] == "GRU point (MSE)"].sort_values("date")
    if point.empty:
        raise ValueError("Point-GRU predictions are needed to provide target values.")
    ax.plot(point["date"], point["actual"], color="#555555", label="White Rabbit CNN target")
    ax.plot(point["date"], point["prediction"], color="#C44E52", label="GRU point (MSE)")

    gaussian = predictions.loc[predictions["model"] == "GRU Gaussian NLL"].sort_values("date")
    ax.plot(gaussian["date"], gaussian["prediction"], color="#2878B5", label="GRU Gaussian mean")
    ax.fill_between(gaussian["date"], gaussian["lower_90"], gaussian["upper_90"],
                    color="#2878B5", alpha=0.16, label="Gaussian 90% interval")

    quantile = predictions.loc[predictions["model"] == "GRU Quantile 10/50/90"].sort_values("date")
    ax.plot(quantile["date"], quantile["prediction"], color="#7A5195", label="GRU median")
    ax.fill_between(quantile["date"], quantile["q10"], quantile["q90"],
                    color="#7A5195", alpha=0.14, label="Quantile 10-90% interval")

    ax.set_title("Same-day CNN calibration: point and uncertainty-aware GRUs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fear & Greed score")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return fig, ax
