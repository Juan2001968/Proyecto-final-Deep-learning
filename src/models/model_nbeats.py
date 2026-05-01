"""N-BEATS / N-HiTS (Oreshkin et al., 2020 / Challu et al., 2023) — placeholder.

Implementación mínima de N-BEATS genérico univariado; el camino N-HiTS
(con multi-rate sampling) queda marcado con TODOs para que se complete.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseForecaster


class _NBeatsBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, layer_width: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = lookback
        for _ in range(num_layers):
            layers += [nn.Linear(prev, layer_width), nn.ReLU()]
            prev = layer_width
        self.fc = nn.Sequential(*layers)
        self.backcast = nn.Linear(prev, lookback)
        self.forecast = nn.Linear(prev, horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        return self.backcast(h), self.forecast(h)


class NBEATSForecaster(BaseForecaster):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        stack_types: list[str] | None = None,
        num_blocks_per_stack: int = 3,
        num_layers: int = 4,
        layer_width: int = 256,
        expansion_coefficient_dim: int = 5,
        trend_polynomial_degree: int = 3,
        variant: str = "nbeats",
    ) -> None:
        stack_types = stack_types or ["trend", "seasonality", "generic"]
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            lookback=lookback,
            horizon=horizon,
            stack_types=stack_types,
            num_blocks_per_stack=num_blocks_per_stack,
            num_layers=num_layers,
            layer_width=layer_width,
            expansion_coefficient_dim=expansion_coefficient_dim,
            trend_polynomial_degree=trend_polynomial_degree,
            variant=variant,
        )
        # TODO(n-hits): añadir multi-rate downsampling por stack si variant == "nhits".
        if n_targets != 1:
            raise NotImplementedError("Esta plantilla N-BEATS asume univariado (n_targets=1).")

        self.blocks = nn.ModuleList(
            [
                _NBeatsBlock(lookback, horizon, layer_width, num_layers)
                for _ in range(len(stack_types) * num_blocks_per_stack)
            ]
        )
        # Selecciona la última feature como serie target — adaptar según pipeline.
        self.target_feature_idx = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) → tomamos la columna del target como serie 1D
        series = x[:, :, self.target_feature_idx]   # (B, L)
        residual = series
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)
        for block in self.blocks:
            backcast, fcst = block(residual)
            residual = residual - backcast
            forecast = forecast + fcst
        return forecast.unsqueeze(-1)               # (B, H, 1)
