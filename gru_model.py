"""Reusable PyTorch architecture for daily component-sequence calibration."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ComponentGRURegressor(nn.Module):
    """Estimate a same-day F&G score from a sequence of component scores."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequences: Tensor) -> Tensor:
        """Return one same-day target estimate for each input sequence."""
        recurrent_output, _ = self.gru(sequences)
        return self.output(recurrent_output[:, -1, :]).squeeze(-1)


class GaussianComponentGRU(nn.Module):
    """Predict the conditional mean and standard deviation of the target."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        minimum_sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 2)
        self.minimum_sigma = minimum_sigma

    def forward(self, sequences: Tensor) -> tuple[Tensor, Tensor]:
        recurrent_output, _ = self.gru(sequences)
        parameters = self.output(recurrent_output[:, -1, :])
        mu = parameters[:, 0]
        sigma = torch.nn.functional.softplus(parameters[:, 1]) + self.minimum_sigma
        return mu, sigma


class QuantileComponentGRU(nn.Module):
    """Predict ordered 10th, 50th, and 90th conditional quantiles."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, sequences: Tensor) -> Tensor:
        recurrent_output, _ = self.gru(sequences)
        raw = self.output(recurrent_output[:, -1, :])
        q10 = raw[:, 0]
        q50 = q10 + torch.nn.functional.softplus(raw[:, 1])
        q90 = q50 + torch.nn.functional.softplus(raw[:, 2])
        return torch.stack((q10, q50, q90), dim=1)
