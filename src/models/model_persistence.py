"""Forecaster de persistencia (naive baseline) — sanity check obligatorio.

Predice ``ŷ_{t+h} = y_t`` para todo horizonte ``h``. Cualquier modelo de DL
en el benchmark debe superarlo significativamente (tests Diebold-Mariano).

⚠️ SKELETON — la lógica específica se implementa en una entrega posterior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseForecaster


class PersistenceForecaster(BaseForecaster):
    """Predicción naive: repite el último valor observado del target a lo largo del horizonte.

    No tiene parámetros entrenables reales. Llevamos un ``nn.Parameter`` dummy
    con ``requires_grad=False`` solo para que ``torch.optim.Adam`` no falle al
    inicializarse con una lista vacía. Sirve como **sanity check** del pipeline
    y **baseline mínimo** que cualquier modelo DL debe superar.
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        target_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            lookback=lookback,
            horizon=horizon,
            target_indices=target_indices,
        )
        # Si el target está dentro de las features de entrada, ``target_indices``
        # apunta a la(s) columna(s) correspondiente(s); si no asumimos los
        # primeros ``n_targets`` canales (convención del proyecto: el target
        # `temp_c` queda en el canal 0).
        if target_indices is None:
            target_indices = list(range(n_targets))
        self.target_indices = list(target_indices)
        # Dummy: Adam necesita al menos un parámetro para inicializar y el
        # Trainer llama ``loss.backward()``, así que necesitamos que la salida
        # tenga ``grad_fn``. Hilamos el dummy con coeficiente 0 dentro del
        # forward — el resultado numérico es idéntico al puro shift, pero el
        # grafo conecta y backward pasa sin actualizar nada en la práctica.
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """x: (B, lookback, n_features) → (B, horizon, n_targets) repitiendo y_t."""
        # Último valor observado del target: x[:, -1, target_indices] -> (B, n_targets)
        last = x[:, -1, self.target_indices]
        # Repetir a lo largo del horizonte: (B, horizon, n_targets)
        out = last.unsqueeze(1).expand(-1, self.horizon, -1).contiguous()
        # Edge cero al dummy para que `loss.backward()` no falle.
        return out + 0.0 * self._dummy.sum()
