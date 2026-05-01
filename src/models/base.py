"""Interfaz común a todos los forecasters DL.

Todas las arquitecturas (LSTM/GRU/TCN/Transformer/N-BEATS) heredan de
``BaseForecaster``. Implementan ``forward`` (PyTorch) y opcionalmente
sobrescriben ``configure_optimizers``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseForecaster(nn.Module, ABC):
    """Forecaster multistep univariado o multivariado.

    Convención de tensores
    ----------------------
    - Entrada X:  (batch, lookback, n_features)
    - Salida ŷ:   (batch, horizon, n_targets)
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.lookback = lookback
        self.horizon = horizon
        self.hparams: dict[str, Any] = dict(kwargs)

    # -------------------------------------------------------- API obligatoria

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    # ---------------------------------------------------- API por defecto

    def configure_optimizers(self, lr: float, weight_decay: float = 0.0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hparams": self.hparams,
                "shape": dict(
                    n_features=self.n_features,
                    n_targets=self.n_targets,
                    lookback=self.lookback,
                    horizon=self.horizon,
                ),
                "class": self.__class__.__name__,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | None = None) -> "BaseForecaster":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["shape"], **ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model
