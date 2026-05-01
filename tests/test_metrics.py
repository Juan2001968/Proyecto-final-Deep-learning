"""Sanity checks de las métricas de evaluación."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, mae, r2, rmse


def test_perfect_prediction_zero_error():
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = compute_metrics(y, y, per_horizon=True)
    assert out["rmse_total"] == pytest.approx(0.0)
    assert out["mae_total"] == pytest.approx(0.0)
    assert out["r2_total"] == pytest.approx(1.0)
    for h in range(3):
        assert out["per_horizon"]["rmse"][h] == pytest.approx(0.0)


def test_rmse_known_value():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    assert rmse(y_true, y_pred) == pytest.approx(0.5)
    assert mae(y_true, y_pred) == pytest.approx(0.25)


def test_r2_baseline_negative_when_worse_than_mean():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y_true, 100.0)
    assert r2(y_true, y_pred) < 0


def test_compute_metrics_3d():
    y = np.random.RandomState(0).randn(8, 5, 2)  # (N, H, T)
    out = compute_metrics(y, y, per_horizon=True)
    assert out["rmse_total"] == pytest.approx(0.0)
    assert out["horizon"] == 5
