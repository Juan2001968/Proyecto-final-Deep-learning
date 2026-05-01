"""Limpieza de CSVs INMET: -9999→NaN, parseo de fecha+hora a UTC, decimal coma."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_yaml, save_parquet

log = get_logger(__name__)


# Rangos físicos plausibles para recortar outliers manifiestos.
_PHYSICAL_BOUNDS = {
    "precip_mm": (0.0, 200.0),
    "pressure_mb": (700.0, 1100.0),
    "pressure_max_mb": (700.0, 1100.0),
    "pressure_min_mb": (700.0, 1100.0),
    "radiation_kj_m2": (0.0, 6000.0),
    "temp_c": (-15.0, 55.0),
    "temp_max_c": (-15.0, 55.0),
    "temp_min_c": (-15.0, 55.0),
    "dew_point_c": (-30.0, 35.0),
    "humidity_pct": (0.0, 100.0),
    "humidity_max_pct": (0.0, 100.0),
    "humidity_min_pct": (0.0, 100.0),
    "wind_dir_deg": (0.0, 360.0),
    "wind_gust_ms": (0.0, 80.0),
    "wind_speed_ms": (0.0, 60.0),
}


def _parse_timestamp(df: pd.DataFrame, date_col: str, hour_col: str) -> pd.DatetimeIndex:
    """Combina ``Data`` + ``Hora UTC`` (formatos heterogéneos) en un DatetimeIndex UTC."""
    hour_str = df[hour_col].astype(str).str.strip()
    # "0000 UTC" / "00:00 UTC" / "0000"
    hour_str = hour_str.str.replace(" UTC", "", regex=False).str.replace(":", "", regex=False)
    hour_str = hour_str.str.zfill(4).str[:2] + ":" + hour_str.str.zfill(4).str[2:]
    ts = pd.to_datetime(
        df[date_col].astype(str) + " " + hour_str,
        utc=True,
        errors="coerce",
        format="mixed",
    )
    ts = ts.dt.tz_localize(None)
    return pd.DatetimeIndex(ts)


def clean_station(station_dir: Path, config: dict) -> pd.DataFrame:
    enc = config["ingest"]["encoding"]
    sep = config["ingest"]["separator"]
    dec = config["ingest"]["decimal"]
    skip = config["ingest"]["header_rows"]
    na_vals = config["ingest"]["na_values"]
    rename = config["columns"]["rename"]
    date_col = config["columns"]["date"]
    hour_col = config["columns"]["hour"]

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(station_dir.glob("*.csv")):
        df = pd.read_csv(
            csv_path,
            sep=sep,
            decimal=dec,
            encoding=enc,
            skiprows=skip,
            na_values=na_vals,
            on_bad_lines="skip",
        )
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"Sin CSV en {station_dir}")

    df = pd.concat(frames, ignore_index=True)
    df.index = _parse_timestamp(df, date_col, hour_col)
    df.index.name = "timestamp"
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.rename(columns=rename)
    keep = [c for c in rename.values() if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")

    # Recorte físico
    for col, (lo, hi) in _PHYSICAL_BOUNDS.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = pd.NA

    log.info(
        "Estación %s — %d filas, %.1f%% NaN tras limpieza",
        station_dir.name, len(df), 100 * df.isna().mean().mean(),
    )
    return df


def run(config: dict) -> None:
    interim = Path(config["paths"]["data_interim"])
    out_dir = Path(config["paths"]["data_processed"]) / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    for station_dir in sorted(p for p in interim.iterdir() if p.is_dir()):
        df = clean_station(station_dir, config)
        save_parquet(df, out_dir / f"{station_dir.name}.parquet")
        log.info("Guardado %s.parquet", station_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
