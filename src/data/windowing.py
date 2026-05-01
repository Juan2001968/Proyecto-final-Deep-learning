"""Ventaneo *lookback → horizon* sin cruzar fronteras de split.

Construye tensores ``(N, lookback, F)`` para X y ``(N, horizon)`` (o
``(N, horizon, T)`` si hay multi-target) para Y.

Garantía: ninguna ventana incluye índices fuera del DataFrame que recibe.
Por construcción, si invocas esto **por separado** sobre cada split, no hay
leakage entre splits — esa propiedad se valida en
``tests/test_windowing_no_leakage.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Windows:
    X: np.ndarray              # (N, lookback, F_in)
    y: np.ndarray              # (N, horizon, T)  (T=1 univariado)
    timestamps: np.ndarray     # (N,) — timestamp del PRIMER paso de y (t+1)
    feature_names: list[str]
    target_names: list[str]


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> Windows:
    """Slice ``df`` en ventanas (lookback, horizon).

    Para cada índice ``t``, X = df[feature_cols].iloc[t-lookback:t]
    e y = df[target_cols].iloc[t:t+horizon]. Solo se generan ventanas que
    caben íntegramente dentro de ``df``.
    """
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    feats = df[feature_cols].to_numpy(dtype=np.float32)
    tgts = df[target_cols].to_numpy(dtype=np.float32)
    idx = df.index

    n = len(df)
    starts = np.arange(lookback, n - horizon + 1, stride, dtype=np.int64)
    if starts.size == 0:
        return Windows(
            X=np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            y=np.empty((0, horizon, len(target_cols)), dtype=np.float32),
            timestamps=np.array([], dtype="datetime64[ns]"),
            feature_names=list(feature_cols),
            target_names=list(target_cols),
        )

    X = np.stack([feats[s - lookback : s] for s in starts])
    y = np.stack([tgts[s : s + horizon] for s in starts])
    ts = idx[starts].to_numpy()  # primer paso del horizonte

    return Windows(
        X=X, y=y, timestamps=ts,
        feature_names=list(feature_cols),
        target_names=list(target_cols),
    )
