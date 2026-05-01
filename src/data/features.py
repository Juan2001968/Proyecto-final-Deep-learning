"""Feature engineering: codificación cíclica, lags, rolling stats e imputación.

⚠️ Las features que dependen de información pasada (lags, rolling) **no
introducen leakage** porque sólo miran timestamps anteriores; sin embargo,
la imputación se hace siempre **antes** del ventaneo y de forma idéntica
para train/val/test (no usa estadísticos globales que mezclen splits).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger, load_parquet, load_yaml, save_parquet

log = get_logger(__name__)


# ---------------------------------------------------------------------- cíclico

def _cyclic(df: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    out = df.copy()
    idx = df.index
    if "hour" in components:
        out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    if "day_of_year" in components:
        out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 366)
        out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 366)
    if "month" in components:
        out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    if "wind_dir_deg" in df.columns:
        rad = np.deg2rad(df["wind_dir_deg"])
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)
    return out


# -------------------------------------------------------------------- lags/rolls

def _lags(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    new = {f"{c}_lag{lag}": df[c].shift(lag) for c in columns for lag in lags if c in df}
    return df.assign(**new) if new else df


def _rolling(
    df: pd.DataFrame, columns: list[str], windows: list[int], stats: list[str]
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in df:
            continue
        # shift(1) garantiza que la rolling no incluya el timestamp actual
        s = df[col].shift(1)
        for w in windows:
            r = s.rolling(window=w, min_periods=max(2, w // 2))
            for st in stats:
                out[f"{col}_roll{w}_{st}"] = getattr(r, st)()
    return out


# ------------------------------------------------------------------- imputación

def _fillna(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "drop":
        return df.dropna()
    if strategy == "ffill":
        return df.ffill().bfill()
    if strategy == "time_interpolate":
        return df.interpolate(method="time").ffill().bfill()
    raise ValueError(f"Estrategia desconocida: {strategy}")


# --------------------------------------------------------------------- driver

def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    feats_cfg = config["features"]
    target = config["task"]["target"]
    exog = config["task"]["exog"]
    base_cols = list(dict.fromkeys([target, *exog]))

    df = _cyclic(df, feats_cfg.get("cyclic", []))
    df = _lags(df, base_cols, feats_cfg.get("lags", []))
    df = _rolling(
        df, base_cols, feats_cfg.get("rolling_windows", []), feats_cfg.get("rolling_stats", [])
    )
    df = _fillna(df, feats_cfg.get("fillna_strategy", "time_interpolate"))
    return df


def run(config: dict) -> None:
    in_dir = Path(config["paths"]["data_processed"]) / "resampled"
    out_dir = Path(config["paths"]["data_processed"]) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(in_dir.glob("*.parquet")):
        df = load_parquet(path)
        df = build_features(df, config)
        log.info("%s — %d filas, %d columnas", path.stem, len(df), df.shape[1])
        save_parquet(df, out_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
