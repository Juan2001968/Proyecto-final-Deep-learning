"""Escaladores que se ajustan ÚNICAMENTE con datos de train.

Reglas:
- ``fit`` siempre se llama con el DataFrame/array de train.
- Para val/test sólo se usa ``transform``.
- Tests/test_scaler_fit_train_only.py verifica esta invariante.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ScalerName = Literal["standard", "minmax", "robust", "none"]


def _factory(name: ScalerName):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    if name == "none":
        return _Identity()
    raise ValueError(name)


class _Identity:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.asarray(X)
    def inverse_transform(self, X):
        return np.asarray(X)


@dataclass
class FeatureScaler:
    """Wrapper que conserva metadatos (`fitted_on`) para auditoría."""
    name: ScalerName
    per_feature: bool = True
    _scaler: object | None = None
    _fitted_on: str | None = None

    def fit(self, X_train: np.ndarray, *, source: str = "train") -> "FeatureScaler":
        scaler = _factory(self.name)
        # X esperado en (N, T, F) o (N, F)
        Xf = X_train.reshape(-1, X_train.shape[-1]) if X_train.ndim > 2 else X_train
        scaler.fit(Xf)
        self._scaler = scaler
        self._fitted_on = source
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._scaler is not None, "Llama a fit() antes de transform()."
        shape = X.shape
        Xf = X.reshape(-1, shape[-1]) if X.ndim > 2 else X
        out = self._scaler.transform(Xf)
        return out.reshape(shape).astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        assert self._scaler is not None
        shape = X.shape
        Xf = X.reshape(-1, shape[-1]) if X.ndim > 2 else X
        out = self._scaler.inverse_transform(Xf)
        return out.reshape(shape).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "name": self.name, "fitted_on": self._fitted_on}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        d = joblib.load(path)
        obj = cls(name=d["name"])
        obj._scaler = d["scaler"]
        obj._fitted_on = d["fitted_on"]
        return obj
