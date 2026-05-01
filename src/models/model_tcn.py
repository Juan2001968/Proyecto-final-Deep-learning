"""Temporal Convolutional Network (Bai et al., 2018) — placeholder funcional."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .base import BaseForecaster


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_c, out_c, k, padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_c, out_c, k, padding=pad, dilation=dilation))
        self.chomp1 = _Chomp1d(pad)
        self.chomp2 = _Chomp1d(pad)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None

    def forward(self, x):
        out = self.drop(self.relu(self.chomp1(self.conv1(x))))
        out = self.drop(self.relu(self.chomp2(self.conv2(out))))
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNForecaster(BaseForecaster):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        channels = channels or [64, 64, 64, 64]
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            lookback=lookback,
            horizon=horizon,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        layers: list[nn.Module] = []
        prev = n_features
        for i, c in enumerate(channels):
            layers.append(_TemporalBlock(prev, c, kernel_size, 2 ** i, dropout))
            prev = c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(prev, horizon * n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)               # (B, F, L)
        h = self.tcn(x)[:, :, -1]           # (B, C)
        y = self.head(h)
        return y.view(-1, self.horizon, self.n_targets)
