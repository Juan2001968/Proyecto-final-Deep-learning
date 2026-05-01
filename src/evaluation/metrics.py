"""Métricas de regresión para forecasting multistep.

Convención de tensores: ``y_true``, ``y_pred`` con shape ``(N, horizon, T)``
o ``(N, horizon)`` (univariado). Devuelve métricas globales y por horizonte.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

_EPS = 1e-8


def _flatten_targets(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 2 else y.reshape(y.shape[0], y.shape[1], -1).mean(axis=-1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + _EPS))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2 + _EPS
    return float(np.mean(num / den) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + _EPS
    return float(1 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, per_horizon: bool = True) -> dict:
    """Calcula RMSE/MAE/R²/MAPE/sMAPE globales y opcionalmente por paso de horizonte."""
    yt = _flatten_targets(y_true)
    yp = _flatten_targets(y_pred)

    out = {
        "rmse_total": rmse(yt, yp),
        "mae_total": mae(yt, yp),
        "r2_total": r2(yt, yp),
        "mape_total": mape(yt, yp),
        "smape_total": smape(yt, yp),
        "n_samples": int(yt.shape[0]),
        "horizon": int(yt.shape[1]),
    }
    if per_horizon and yt.ndim == 2:
        out["per_horizon"] = {
            "rmse": [rmse(yt[:, h], yp[:, h]) for h in range(yt.shape[1])],
            "mae":  [mae(yt[:, h], yp[:, h])  for h in range(yt.shape[1])],
            "r2":   [r2(yt[:, h], yp[:, h])   for h in range(yt.shape[1])],
        }
    return out


def metrics_by_region(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    region_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Agrega métricas por macrorregión IBGE.

    Args:
        predictions: ``{station_code: (y_true, y_pred)}`` con tensores
            ``(N, H)`` o ``(N, H, T)``.
        region_map: Opcional, ``{station_code: region}``. Si es ``None`` se
            resuelve con :func:`src.utils.regions.region_of`.

    Returns:
        DataFrame con columnas ``[region, n_stations, rmse_mean, rmse_std,
        mae_mean, mae_std, r2_mean, r2_std]``. La media/std son **entre
        estaciones de la misma región** (no entre muestras).
    """
    if region_map is None:
        from src.utils.regions import region_of

        region_map = {code: region_of(code) for code in predictions}

    rows = []
    for code, (yt, yp) in predictions.items():
        rows.append({
            "station": code,
            "region": region_map[code],
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
            "r2": r2(yt, yp),
        })
    df = pd.DataFrame(rows)
    grp = df.groupby("region")
    out = grp.agg(
        n_stations=("station", "nunique"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),   mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),     r2_std=("r2", "std"),
    ).reset_index()
    return out
