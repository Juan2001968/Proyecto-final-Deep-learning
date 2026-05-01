"""Orquesta el procesamiento ``data/interim/<wmo>/*.csv`` → ``data/processed/<wmo>.parquet``.

Etapas (in-memory, una por estación, sin escrituras intermedias):

    1. Lectura de los CSV anuales INMET con encoding ``latin-1`` y separador ``;``.
    2. Normalización de columnas a ``snake_case`` por *pattern matching* tolerante
       a variantes de encoding y formato entre años.
    3. Parseo de timestamp (``DATA`` + ``HORA``) a ``datetime64[ns]`` **tz-naive**
       (interpretando UTC implícitamente, coherente con los splits del proyecto).
    4. Recorte físico de outliers manifiestos (rangos plausibles para Brasil).
    5. Reindex a rejilla horaria regular y reporte de gaps consecutivos > 6 h.
    6. Imputación causal con ``ffill(limit=6)`` para gaps cortos.
    7. Features cíclicas: ``hour_{sin,cos}``, ``doy_{sin,cos}``, ``month_{sin,cos}``.
    8. Identificadores de estación (``station_id`` entero, ``region``, ``biome``,
       ``koppen_class``) provenientes de ``config/stations.yaml``.

Salida: un parquet por estación en ``data/processed/<wmo>.parquet``.

Uso:
    python -m src.data.process --config config/config.yaml
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger, load_yaml, save_parquet

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Rango físico plausible para Brasil. Valores fuera → NaN.
_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "precip_mm": (0.0, 200.0),
    "pressure_mb": (700.0, 1100.0),
    "pressure_max_mb": (700.0, 1100.0),
    "pressure_min_mb": (700.0, 1100.0),
    "radiation_kj_m2": (0.0, 6000.0),
    "temp_c": (-10.0, 50.0),
    "temp_max_c": (-10.0, 50.0),
    "temp_min_c": (-10.0, 50.0),
    "dew_point_c": (-30.0, 35.0),
    "dew_point_max_c": (-30.0, 35.0),
    "dew_point_min_c": (-30.0, 35.0),
    "humidity_pct": (0.0, 100.0),
    "humidity_max_pct": (0.0, 100.0),
    "humidity_min_pct": (0.0, 100.0),
    "wind_dir_deg": (0.0, 360.0),
    "wind_gust_ms": (0.0, 80.0),
    "wind_speed_ms": (0.0, 60.0),
}

# Patrones (substring sobre nombre normalizado) → nombre snake_case.
# Orden importa: específicos antes que genéricos.
_RENAME_PATTERNS: list[tuple[str, str]] = [
    ("PRECIPITACAO TOTAL", "precip_mm"),
    ("PRESSAO ATMOSFERICA AO NIVEL", "pressure_mb"),
    ("PRESSAO ATMOSFERICA MAX", "pressure_max_mb"),
    ("PRESSAO ATMOSFERICA MIN", "pressure_min_mb"),
    ("RADIACAO GLOBAL", "radiation_kj_m2"),
    ("TEMPERATURA DO AR - BULBO SECO", "temp_c"),
    ("TEMPERATURA ORVALHO MAX", "dew_point_max_c"),
    ("TEMPERATURA ORVALHO MIN", "dew_point_min_c"),
    ("TEMPERATURA MAX", "temp_max_c"),
    ("TEMPERATURA MIN", "temp_min_c"),
    ("TEMPERATURA DO PONTO DE ORVALHO", "dew_point_c"),
    ("UMIDADE REL. MAX", "humidity_max_pct"),
    ("UMIDADE REL. MIN", "humidity_min_pct"),
    ("UMIDADE RELATIVA DO AR, HORARIA", "humidity_pct"),
    ("VENTO, DIRECAO", "wind_dir_deg"),
    ("VENTO, RAJADA", "wind_gust_ms"),
    ("VENTO, VELOCIDADE", "wind_speed_ms"),
]

_DATE_PATTERNS = ("DATA",)
_HOUR_PATTERNS = ("HORA",)

_GAP_THRESHOLD_HOURS = 6
_IMPUTE_LIMIT_HOURS = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normaliza string: ASCII upper sin tildes, espacios colapsados.

    Args:
        s: cadena posiblemente con caracteres latinos (``°``, ``ã``, etc.).

    Returns:
        Versión ASCII en mayúsculas, espacios simples.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.upper().split())


def _resolve_columns(columns: list[str]) -> tuple[str, str, dict[str, str]]:
    """Identifica las columnas ``Data`` y ``Hora`` y construye el rename map.

    Args:
        columns: nombres originales del CSV (post ``str.strip``).

    Returns:
        Tupla ``(date_col, hour_col, rename_map)`` donde ``rename_map`` mapea
        nombre original → nombre snake_case interno.

    Raises:
        ValueError: si no se identifican columnas de fecha u hora.
    """
    norm_to_orig = {_norm(c): c for c in columns}

    date_col = next(
        (orig for n, orig in norm_to_orig.items() if any(n.startswith(p) for p in _DATE_PATTERNS)),
        None,
    )
    hour_col = next(
        (orig for n, orig in norm_to_orig.items() if any(n.startswith(p) for p in _HOUR_PATTERNS)),
        None,
    )
    if date_col is None or hour_col is None:
        raise ValueError(f"No encontré columnas de fecha/hora en: {columns}")

    rename: dict[str, str] = {}
    used: set[str] = set()
    for pattern, target in _RENAME_PATTERNS:
        if target in used:
            continue
        for n, orig in norm_to_orig.items():
            if orig in (date_col, hour_col) or orig in rename:
                continue
            if pattern in n:
                rename[orig] = target
                used.add(target)
                break
    return date_col, hour_col, rename


def _parse_timestamp(df: pd.DataFrame, date_col: str, hour_col: str) -> pd.DatetimeIndex:
    """Combina ``Data`` + ``Hora`` en un ``DatetimeIndex`` tz-naive.

    Acepta variantes ``"0000 UTC"``, ``"00:00"``, ``"0000"``.
    """
    hour_str = df[hour_col].astype(str).str.strip()
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


def _read_station_csvs(station_dir: Path, encoding: str, sep: str, dec: str,
                        skip: int, na_vals: list) -> pd.DataFrame:
    """Concatena los CSV anuales de una estación con columnas pre-normalizadas.

    Los CSV INMET cambian los nombres de las columnas entre años (ej. 2018 usa
    ``DATA (YYYY-MM-DD)``/``HORA (UTC)`` mientras que 2019+ usan ``Data``/``Hora UTC``).
    Resolvemos el rename **por archivo** antes del concat para que las columnas
    queden canonizadas y no aparezcan duplicadas tras concatenar.
    """
    _DATE = "__date__"
    _HOUR = "__hour__"
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(station_dir.glob("*.csv")):
        df = pd.read_csv(
            csv_path,
            sep=sep,
            decimal=dec,
            encoding=encoding,
            skiprows=skip,
            na_values=na_vals,
            on_bad_lines="skip",
        )
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        date_col, hour_col, rename = _resolve_columns(df.columns.tolist())
        full_rename = {date_col: _DATE, hour_col: _HOUR, **rename}
        df = df.rename(columns=full_rename)
        # Mantener solo columnas canónicas conocidas
        known = [_DATE, _HOUR, *[v for v in rename.values()]]
        df = df[[c for c in known if c in df.columns]]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Sin CSV en {station_dir}")
    return pd.concat(frames, ignore_index=True)


def _apply_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Setea a NaN los valores fuera de los rangos físicos plausibles."""
    out = df.copy()
    for col, (lo, hi) in _PHYSICAL_BOUNDS.items():
        if col in out.columns:
            out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan
    return out


def _detect_long_gaps(df: pd.DataFrame, threshold_h: int) -> int:
    """Cuenta runs de NaN consecutivos > ``threshold_h`` en *cualquier* columna numérica.

    Args:
        df: serie ya reindexada a rejilla regular.
        threshold_h: umbral en horas.

    Returns:
        Número de gaps largos detectados (suma sobre columnas).
    """
    if df.empty:
        return 0
    long_gaps = 0
    for col in df.select_dtypes(include="number").columns:
        is_nan = df[col].isna().to_numpy()
        if not is_nan.any():
            continue
        # run-length: cambios donde empieza/termina un tramo NaN
        edges = np.diff(np.concatenate([[0], is_nan.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        run_lengths = ends - starts
        long_gaps += int((run_lengths > threshold_h).sum())
    return long_gaps


def _causal_impute(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Forward-fill limitado a ``limit`` pasos. Estrictamente causal."""
    return df.ffill(limit=limit)


def _add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade codificación seno/coseno de hora, día del año y mes."""
    out = df.copy()
    idx = df.index
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 366)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 366)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return out


def _load_station_metadata(stations_yaml: Path) -> tuple[dict[str, dict], dict[str, int]]:
    """Lee ``stations.yaml`` y devuelve mapas WMO → metadatos y WMO → station_id (entero)."""
    cfg = load_yaml(stations_yaml)
    by_code = {s["code"].upper(): s for s in cfg.get("stations", [])}
    # ID determinístico desde el orden alfabético de los códigos.
    id_map = {code: i for i, code in enumerate(sorted(by_code))}
    return by_code, id_map


def _add_station_identifiers(
    df: pd.DataFrame, wmo: str, by_code: dict[str, dict], id_map: dict[str, int]
) -> pd.DataFrame:
    """Añade ``station_id``, ``region``, ``biome``, ``koppen_class`` como columnas constantes."""
    info = by_code.get(wmo, {})
    out = df.copy()
    out["station_id"] = id_map.get(wmo, -1)
    out["region"] = info.get("region", "UNKNOWN")
    out["biome"] = info.get("biome", "UNKNOWN")
    out["koppen_class"] = info.get("koppen_class", "UNKNOWN")
    return out


# ---------------------------------------------------------------------------
# Driver por estación
# ---------------------------------------------------------------------------

def process_station(
    station_dir: Path, config: dict, by_code: dict[str, dict], id_map: dict[str, int]
) -> pd.DataFrame:
    """Pipeline completo para una estación INMET.

    Args:
        station_dir: ``data/interim/<wmo>/`` con los CSV anuales y ``metadata.json``.
        config: configuración global del proyecto.
        by_code: mapa WMO → dict (region, biome, koppen_class) desde stations.yaml.
        id_map: mapa WMO → station_id entero.

    Returns:
        DataFrame procesado, indexado por ``timestamp`` (tz-naive, frecuencia ``h``),
        listo para guardar como parquet.
    """
    wmo = station_dir.name.upper()
    enc = config["ingest"]["encoding"]
    sep = config["ingest"]["separator"]
    dec = config["ingest"]["decimal"]
    skip = config["ingest"]["header_rows"]
    na_vals = config["ingest"]["na_values"]

    raw = _read_station_csvs(station_dir, enc, sep, dec, skip, na_vals)
    raw.index = _parse_timestamp(raw, "__date__", "__hour__")
    raw.index.name = "timestamp"
    raw = raw[raw.index.notna()].sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]

    measure_cols = [c for c in raw.columns if c not in ("__date__", "__hour__")]
    df = raw[measure_cols].apply(pd.to_numeric, errors="coerce")

    df = _apply_physical_bounds(df)

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = "timestamp"

    long_gaps = _detect_long_gaps(df, _GAP_THRESHOLD_HOURS)
    df = _causal_impute(df, _IMPUTE_LIMIT_HOURS)
    df = _add_cyclic_features(df)
    df = _add_station_identifiers(df, wmo, by_code, id_map)

    log.info(
        "%s — %d filas, %.1f%% NaN, %d gaps > %dh",
        wmo, len(df), 100 * df.select_dtypes(include="number").isna().mean().mean(),
        long_gaps, _GAP_THRESHOLD_HOURS,
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(config: dict) -> None:
    """Procesa todas las estaciones presentes en ``data/interim/`` y guarda parquets."""
    interim = Path(config["paths"]["data_interim"])
    out_dir = Path(config["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    by_code, id_map = _load_station_metadata(Path(config["paths"]["stations_config"]))

    station_dirs = sorted(p for p in interim.iterdir() if p.is_dir())
    log.info("Procesando %d estaciones …", len(station_dirs))

    ok = 0
    for station_dir in station_dirs:
        try:
            df = process_station(station_dir, config, by_code, id_map)
            save_parquet(df, out_dir / f"{station_dir.name}.parquet")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            log.error("Falló %s: %s", station_dir.name, exc)
    log.info("Process OK: %d/%d estaciones -> %s", ok, len(station_dirs), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
