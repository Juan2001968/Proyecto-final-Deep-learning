"""Garantiza que el FeatureScaler se ajusta SÓLO con datos de train."""

import numpy as np
import pytest

from src.data.scalers import FeatureScaler


def test_transform_before_fit_raises():
    sc = FeatureScaler(name="standard")
    with pytest.raises(AssertionError):
        sc.transform(np.zeros((4, 3)))


def test_fit_uses_only_train_stats():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=10.0, scale=2.0, size=(1000, 3)).astype(np.float32)
    test = rng.normal(loc=50.0, scale=20.0, size=(500, 3)).astype(np.float32)

    sc = FeatureScaler(name="standard").fit(train, source="train")
    train_scaled = sc.transform(train)

    # tras ajustar con train: media~0, std~1 en train (no en test)
    np.testing.assert_allclose(train_scaled.mean(axis=0), 0.0, atol=1e-2)
    np.testing.assert_allclose(train_scaled.std(axis=0), 1.0, atol=1e-2)

    test_scaled = sc.transform(test)
    assert abs(test_scaled.mean()) > 5  # confirma que test NO está re-centrado


def test_fitted_on_metadata():
    sc = FeatureScaler(name="minmax").fit(np.random.randn(100, 2), source="train")
    assert sc._fitted_on == "train"


def test_3d_tensor_shape_preserved():
    X = np.random.randn(50, 24, 5).astype(np.float32)
    sc = FeatureScaler(name="standard").fit(X, source="train")
    out = sc.transform(X)
    assert out.shape == X.shape
