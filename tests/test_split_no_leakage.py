"""Tests CRÍTICOS de splits temporales sin data leakage."""

import pytest

from src.data.split import split_dataframe


def _by_year_cfg():
    return {
        "mode": "by_year",
        "by_year": {
            "train_years": [2018, 2019, 2020, 2021],
            "val_years": [2022],
            "test_years": [2023],
        },
        "by_ratio": {"train": 0.7, "val": 0.15, "test": 0.15},
        "by_date": {"train_end": "2021-12-31 23:00:00", "val_end": "2022-12-31 23:00:00"},
    }


def test_by_year_strict_chronology(hourly_df):
    s = split_dataframe(hourly_df, _by_year_cfg())
    assert s.train.index.max() < s.val.index.min()
    assert s.val.index.max() < s.test.index.min()


def test_by_year_no_overlap_years(hourly_df):
    s = split_dataframe(hourly_df, _by_year_cfg())
    yr = lambda df: set(df.index.year)
    assert yr(s.train).isdisjoint(yr(s.val))
    assert yr(s.val).isdisjoint(yr(s.test))
    assert yr(s.train).isdisjoint(yr(s.test))


def test_by_year_overlap_raises(hourly_df):
    bad = _by_year_cfg()
    bad["by_year"]["val_years"] = [2021, 2022]   # solapa con train
    with pytest.raises(ValueError):
        split_dataframe(hourly_df, bad)


def test_by_ratio_strict_chronology(hourly_df):
    cfg = _by_year_cfg(); cfg["mode"] = "by_ratio"
    s = split_dataframe(hourly_df, cfg)
    assert s.train.index.max() < s.val.index.min()
    assert s.val.index.max() < s.test.index.min()
    assert len(s.train) + len(s.val) + len(s.test) == len(hourly_df)


def test_by_date_strict_chronology(hourly_df):
    cfg = _by_year_cfg(); cfg["mode"] = "by_date"
    s = split_dataframe(hourly_df, cfg)
    assert s.train.index.max() < s.val.index.min()
    assert s.val.index.max() < s.test.index.min()
