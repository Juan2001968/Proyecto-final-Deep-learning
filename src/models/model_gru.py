"""Forecaster GRU multistep — placeholder funcional."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseForecaster


class GRUForecaster(BaseForecaster):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            lookback=lookback,
            horizon=horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_in = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_in, horizon * n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        y = self.head(last)
        return y.view(-1, self.horizon, self.n_targets)
