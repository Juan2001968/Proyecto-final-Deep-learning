"""Verifica que set_seed produce resultados reproducibles."""

import numpy as np

from src.utils import set_seed


def test_set_seed_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(50)
    set_seed(123)
    b = np.random.rand(50)
    np.testing.assert_array_equal(a, b)


def test_set_seed_torch_reproducible():
    torch = __import__("torch")
    set_seed(7)
    a = torch.randn(64)
    set_seed(7)
    b = torch.randn(64)
    assert torch.equal(a, b)
