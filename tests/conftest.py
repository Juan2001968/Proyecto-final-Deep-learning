"""Fixtures comunes para los tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    """Serie horaria sintética 2018-2023 con dos features y un target."""
    idx = pd.date_range("2018-01-01", "2025-12-31 23:00", freq="h")
    rng = np.random.default_rng(42)
    t = np.linspace(0, 100, len(idx))
    target = (
        15 + 10 * np.sin(2 * np.pi * idx.hour / 24)
        + 5 * np.sin(2 * np.pi * idx.dayofyear / 366)
        + rng.normal(0, 1, size=len(idx))
    )
    return pd.DataFrame(
        {
            "temp_c": target,
            "humidity_pct": 60 + 10 * np.cos(2 * np.pi * idx.hour / 24) + rng.normal(0, 2, size=len(idx)),
            "pressure_mb": 1013 + rng.normal(0, 1.5, size=len(idx)),
            "drift": t,
        },
        index=idx,
    )
