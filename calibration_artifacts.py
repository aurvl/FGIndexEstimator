"""Serialize already-fitted Bayesian Ridge and GRU models with PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def bayesian_checkpoint_from_pipeline(
    fitted_pipeline: Any,
    *,
    feature_columns: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    target_source: str = "cnn_endpoint",
) -> dict[str, Any]:
    """Extract a fitted sklearn Bayesian Ridge pipeline as serializable values."""
    try:
        scaler = fitted_pipeline.named_steps["scaler"]
        regressor = fitted_pipeline.named_steps["regressor"]
        covariance = regressor.sigma_
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "Expected a fitted Pipeline with 'scaler' and Bayesian 'regressor' steps."
        ) from exc

    return {
        "feature_columns": list(feature_columns),
        "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "coef": np.asarray(regressor.coef_, dtype=float).tolist(),
        "intercept": float(regressor.intercept_),
        "posterior_covariance": np.asarray(covariance, dtype=float).tolist(),
        "alpha": float(regressor.alpha_),
        "lambda": float(regressor.lambda_),
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "target_source": target_source,
    }


def save_calibration_artifacts(
    *,
    bayesian_checkpoint: Mapping[str, Any],
    gru_metrics: pd.DataFrame,
    output_path: str | Path = "models/calibration/fg_calibrators.pt",
) -> dict[str, str]:
    """Save the tested Bayesian model and lowest-RMSE tested GRU without fitting.

    ``bayesian_checkpoint`` is returned in ``gru_comparison.attrs`` and
    ``gru_metrics`` is returned by ``compare_probabilistic_grus``. The latter
    stores fitted GRU state dictionaries in ``attrs['fitted_gru_models']``.
    The selected GRU is the candidate with the lowest common-holdout RMSE.

    Both models, scalers, feature order, source information, and selection
    metadata are written into a single ``torch.save`` checkpoint.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install PyTorch with `pip install torch` to save the checkpoint.") from exc

    required_columns = {"model", "RMSE"}
    missing = required_columns.difference(gru_metrics.columns)
    if missing:
        raise ValueError(f"GRU metrics are missing columns: {', '.join(sorted(missing))}.")

    fitted_grus = gru_metrics.attrs.get("fitted_gru_models", {})
    if not fitted_grus:
        raise ValueError(
            "No fitted GRU weights are attached to these metrics. Rerun only the "
            "probabilistic GRU comparison cell after reloading `gru_uncertainty`."
        )
    if not bayesian_checkpoint:
        raise ValueError(
            "No fitted Bayesian checkpoint was attached to the benchmark metrics. "
            "Rerun only the Bayesian/GRU benchmark cell after reloading `gru_benchmark`."
        )

    eligible = gru_metrics.loc[
        gru_metrics["model"].isin(fitted_grus) & np.isfinite(gru_metrics["RMSE"])
    ]
    if eligible.empty:
        raise ValueError("No fitted GRU candidate has a finite RMSE for selection.")
    best_row = eligible.loc[eligible["RMSE"].idxmin()]
    selected_name = str(best_row["model"])
    selected_gru = fitted_grus[selected_name]

    bayesian_tensors = {
        "scaler_mean": torch.tensor(bayesian_checkpoint["scaler_mean"], dtype=torch.float64),
        "scaler_scale": torch.tensor(bayesian_checkpoint["scaler_scale"], dtype=torch.float64),
        "coef": torch.tensor(bayesian_checkpoint["coef"], dtype=torch.float64),
        "intercept": torch.tensor(bayesian_checkpoint["intercept"], dtype=torch.float64),
        "posterior_covariance": torch.tensor(
            bayesian_checkpoint["posterior_covariance"], dtype=torch.float64
        ),
        "alpha": torch.tensor(bayesian_checkpoint["alpha"], dtype=torch.float64),
        "lambda": torch.tensor(bayesian_checkpoint["lambda"], dtype=torch.float64),
    }
    gru_tensors = {
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in selected_gru["state_dict"].items()
        },
        "scaler_mean": torch.tensor(selected_gru["scaler_mean"], dtype=torch.float32),
        "scaler_scale": torch.tensor(selected_gru["scaler_scale"], dtype=torch.float32),
    }
    gru_metadata = {
        key: value
        for key, value in selected_gru.items()
        if key not in {"state_dict", "scaler_mean", "scaler_scale"}
    }

    artifact = {
        "artifact_type": "fg_calibrator_bundle",
        "format_version": 1,
        "selection_rule": "lowest RMSE among point, Gaussian-NLL, and quantile GRUs on the common CNN holdout",
        "selected_gru_name": selected_name,
        "selected_gru_rmse": float(best_row["RMSE"]),
        "bayesian_ridge": {
            "model_type": "BayesianRidge",
            "feature_columns": list(bayesian_checkpoint["feature_columns"]),
            "train_start": bayesian_checkpoint["train_start"],
            "train_end": bayesian_checkpoint["train_end"],
            "test_start": bayesian_checkpoint["test_start"],
            "test_end": bayesian_checkpoint["test_end"],
            "target_source": bayesian_checkpoint["target_source"],
            **bayesian_tensors,
        },
        "gru": {
            "model_name": selected_name,
            **gru_metadata,
            **gru_tensors,
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    return {
        "checkpoint": str(destination),
        "selected_gru": selected_name,
        "selection_metric": "RMSE",
    }
