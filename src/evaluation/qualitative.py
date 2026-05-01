"""Plots cualitativos: predicción vs real, residuos, error por hora del día."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_pred_vs_true(
    y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray,
    out_path: Path, horizon_step: int = 0, n_show: int = 500,
) -> None:
    """Compara la primera (o ``horizon_step``-ésima) predicción con la real."""
    yt = y_true[:, horizon_step] if y_true.ndim >= 2 else y_true
    yp = y_pred[:, horizon_step] if y_pred.ndim >= 2 else y_pred
    n = min(n_show, len(yt))
    ts = pd.to_datetime(timestamps[:n])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, yt[:n], label="real", linewidth=1.4)
    ax.plot(ts, yp[:n], label="pred", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Predicción vs real (h+{horizon_step + 1})")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    res = (y_pred - y_true).reshape(-1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(res, kde=True, ax=ax[0])
    ax[0].set_title("Distribución residuos")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].scatter(y_true.reshape(-1), res, s=2, alpha=0.3)
    ax[1].set_xlabel("y real")
    ax[1].set_ylabel("residuo")
    ax[1].set_title("Residuos vs real")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_by_hour(
    y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray, out_path: Path
) -> None:
    """RMSE agrupado por hora del día (sobre el primer paso del horizonte)."""
    ts = pd.to_datetime(timestamps)
    err = (y_pred[:, 0] if y_pred.ndim >= 2 else y_pred) - (y_true[:, 0] if y_true.ndim >= 2 else y_true)
    if err.ndim > 1:
        err = err.mean(axis=-1)
    df = pd.DataFrame({"hour": ts.hour, "sq": err ** 2})
    rmse_by_h = np.sqrt(df.groupby("hour")["sq"].mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    rmse_by_h.plot(kind="bar", ax=ax)
    ax.set_xlabel("hora del día (UTC)")
    ax.set_ylabel("RMSE")
    ax.set_title("Error por hora del día")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
