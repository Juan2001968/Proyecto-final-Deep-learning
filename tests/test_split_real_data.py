"""Verifica el split temporal sobre un parquet real de ``data/processed/``.

Garantiza que la sección ``split.by_year`` de ``config/config.yaml`` produce
los rangos esperados (train 2018-2023, val 2024, test 2025) sobre los datos
realmente procesados, y que ``assert_no_leakage`` pasa.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.split import split_dataframe
from src.utils import load_yaml

_PARQUET = Path("data/processed/A001.parquet")
_CONFIG = Path("config/config.yaml")


@pytest.fixture(scope="module")
def real_split():
    if not _PARQUET.exists():
        pytest.skip(f"No existe {_PARQUET} — corre `make process` antes.")
    df = pd.read_parquet(_PARQUET)
    cfg = load_yaml(_CONFIG)
    return split_dataframe(df, cfg["split"])


def test_train_covers_2018_to_2023(real_split):
    assert real_split.train.index.min() == pd.Timestamp("2018-01-01 00:00:00")
    assert real_split.train.index.max() == pd.Timestamp("2023-12-31 23:00:00")
    assert set(real_split.train.index.year.unique()) == {2018, 2019, 2020, 2021, 2022, 2023}


def test_val_covers_2024(real_split):
    assert real_split.val.index.min() == pd.Timestamp("2024-01-01 00:00:00")
    assert real_split.val.index.max() == pd.Timestamp("2024-12-31 23:00:00")
    assert set(real_split.val.index.year.unique()) == {2024}


def test_test_covers_2025(real_split):
    assert real_split.test.index.min() == pd.Timestamp("2025-01-01 00:00:00")
    assert real_split.test.index.max() == pd.Timestamp("2025-12-31 23:00:00")
    assert set(real_split.test.index.year.unique()) == {2025}


def test_no_leakage_on_real_data(real_split):
    real_split.assert_no_leakage()
