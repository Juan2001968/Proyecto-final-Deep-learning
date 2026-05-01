"""Resampling regular a la frecuencia objetivo y manejo de gaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_parquet, load_yaml, save_parquet

log = get_logger(__name__)


# Reglas de agregación cuando se downsampling (p.ej. H → D)
_AGG_RULES = {
    "precip_mm": "sum",
    "radiation_kj_m2": "sum",
    "wind_gust_ms": "max",
    "temp_max_c": "max",
    "temp_min_c": "min",
    "humidity_max_pct": "max",
    "humidity_min_pct": "min",
}
_DEFAULT_AGG = "mean"


def _agg_for(col: str) -> str:
    return _AGG_RULES.get(col, _DEFAULT_AGG)


def regular_grid(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Reindexa a una rejilla regular ``freq`` introduciendo NaN en los gaps."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    out = df.reindex(full_idx)
    out.index.name = "timestamp"
    return out


def resample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resampling adaptativo: en frecuencia horaria solo regulariza la rejilla;
    en frecuencias inferiores agrega con `sum/mean/max/min` según variable."""
    cur_freq = pd.infer_freq(df.index) or "H"
    if pd.tseries.frequencies.to_offset(cur_freq) == pd.tseries.frequencies.to_offset(freq):
        return regular_grid(df, freq)

    agg = {c: _agg_for(c) for c in df.columns}
    return df.resample(freq).agg(agg)


def run(config: dict) -> None:
    in_dir = Path(config["paths"]["data_processed"]) / "clean"
    out_dir = Path(config["paths"]["data_processed"]) / "resampled"
    out_dir.mkdir(parents=True, exist_ok=True)

    freq = config["task"]["freq"]
    for path in sorted(in_dir.glob("*.parquet")):
        df = load_parquet(path)
        df = resample_df(df, freq)
        gaps = df.isna().any(axis=1).sum()
        log.info("%s — %d filas, %d con gap", path.stem, len(df), gaps)
        save_parquet(df, out_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
