"""Verifica que las ventanas no cruzan fronteras de split."""

import numpy as np

from src.data.split import split_dataframe
from src.data.windowing import make_windows


_BY_YEAR = {
    "mode": "by_year",
    "by_year": {
        "train_years": [2018, 2019, 2020, 2021],
        "val_years": [2022],
        "test_years": [2023],
    },
    "by_ratio": {"train": 0.7, "val": 0.15, "test": 0.15},
    "by_date": {"train_end": "2021-12-31 23:00:00", "val_end": "2022-12-31 23:00:00"},
}


def test_windows_within_split_only(hourly_df):
    s = split_dataframe(hourly_df, _BY_YEAR)
    L, H = 48, 24
    feats = ["temp_c", "humidity_pct", "pressure_mb"]
    tgts = ["temp_c"]

    w_tr = make_windows(s.train, feats, tgts, L, H)
    w_va = make_windows(s.val, feats, tgts, L, H)
    w_te = make_windows(s.test, feats, tgts, L, H)

    # primer paso del horizonte (timestamps[]) cae dentro del split correspondiente
    assert (w_tr.timestamps >= s.train.index.min()).all()
    assert (w_tr.timestamps <= s.train.index.max()).all()
    assert (w_va.timestamps >= s.val.index.min()).all()
    assert (w_va.timestamps <= s.val.index.max()).all()
    assert (w_te.timestamps >= s.test.index.min()).all()
    assert (w_te.timestamps <= s.test.index.max()).all()

    # cantidad esperada
    assert w_tr.X.shape[1] == L
    assert w_tr.y.shape[1] == H
    assert w_tr.X.shape[0] == len(s.train) - L - H + 1


def test_no_window_spans_two_splits(hourly_df):
    s = split_dataframe(hourly_df, _BY_YEAR)
    L, H = 48, 24
    feats = ["temp_c", "humidity_pct"]
    tgts = ["temp_c"]
    w_va = make_windows(s.val, feats, tgts, L, H)
    # la primera ventana de val empieza al menos L pasos despues del inicio de val
    first_start = s.val.index[L]
    assert first_start <= w_va.timestamps[0]
    # ninguna ventana de val incluye datos de train
    assert s.train.index.max() < s.val.index[0]
