"""Temporal Fusion Transformer — placeholder.

Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion
Transformers for interpretable multi-horizon time series forecasting.
International Journal of Forecasting, 37(4), 1748–1764.
https://doi.org/10.1016/j.ijforecast.2021.03.012

Componentes principales pendientes de implementar:
    - Variable Selection Networks (VSN) sobre inputs estáticos / pasados / futuros.
    - Static covariate encoders (4 contextos).
    - LSTM encoder-decoder local.
    - Multi-head self-attention para dependencias largas.
    - Quantile loss (P10/P50/P90) — no MSE.

⚠️ SKELETON — la implementación completa se difiere a una entrega posterior.
   Recomendación: usar `pytorch-forecasting` como referencia o implementación
   directa. La firma debe respetar ``BaseForecaster``.
"""

from __future__ import annotations

import torch

from .base import BaseForecaster


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer (Lim et al., 2021).

    Multi-horizonte nativo con embeddings de entidad y quantile loss.
    Encaje óptimo con el panel INMET (40 estaciones × 5 macrorregiones).
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        hidden_size: int = 64,
        attention_heads: int = 4,
        dropout: float = 0.1,
        n_static_categorical: int = 4,   # station_id, region, biome, koppen
        n_static_real: int = 3,          # latitude, longitude, altitude
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            lookback=lookback,
            horizon=horizon,
            hidden_size=hidden_size,
            attention_heads=attention_heads,
            dropout=dropout,
            n_static_categorical=n_static_categorical,
            n_static_real=n_static_real,
            quantiles=tuple(quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TFTForecaster: pendiente de implementación.")
