"""Ingesta de archivos INMET: descomprime ZIPs anuales y extrae metadatos.

Espera ZIPs en ``data/raw/`` (uno por año) o CSVs ya extraídos. Filtra por
los códigos WMO listados en ``config/stations.yaml`` y guarda los CSV
reorganizados como::

    data/interim/<station_code>/<year>.csv

con un sidecar JSON (``<station_code>/metadata.json``) que conserva los
metadatos de cabecera (REGIÃO, UF, lat/lon/alt).
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from src.utils import get_logger, load_yaml

log = get_logger(__name__)

_FILENAME_RE = re.compile(
    r"INMET_(?P<region>\w+)_(?P<uf>\w+)_(?P<wmo>\w+)_.*?_(?P<dini>[\d-]+)_A_(?P<dfin>[\d-]+)\.CSV",
    re.IGNORECASE,
)


def _parse_metadata(csv_path: Path, header_rows: int, encoding: str) -> dict:
    meta: dict = {}
    with open(csv_path, encoding=encoding) as f:
        for _ in range(header_rows):
            line = f.readline()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            # las líneas INMET son tipo "REGIÃO:;CO" → tras split por ':' queda ';CO'
            meta[key.strip()] = val.strip().lstrip(";").rstrip(";").strip()
    return meta


def _extract_zips(raw_dir: Path) -> None:
    for zip_path in sorted(raw_dir.glob("*.zip")):
        out_dir = raw_dir / zip_path.stem
        if out_dir.exists():
            continue
        log.info("Extrayendo %s …", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)


def ingest(config: dict) -> None:
    raw_dir = Path(config["paths"]["data_raw"])
    interim_dir = Path(config["paths"]["data_interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    stations_cfg = load_yaml(config["paths"]["stations_config"])
    wanted = {s["code"].upper() for s in stations_cfg.get("stations", [])}
    log.info("Estaciones objetivo: %s", sorted(wanted) if wanted else "TODAS")

    _extract_zips(raw_dir)

    csv_paths = list(raw_dir.rglob("*.CSV")) + list(raw_dir.rglob("*.csv"))
    log.info("CSV INMET encontrados: %d", len(csv_paths))

    metas: dict[str, dict] = {}
    moved = 0
    for csv_path in csv_paths:
        m = _FILENAME_RE.match(csv_path.name)
        if not m:
            log.warning("Nombre no reconocido, salto: %s", csv_path.name)
            continue
        wmo = m["wmo"].upper()
        if wanted and wmo not in wanted:
            continue
        year = m["dini"][-4:]

        out_dir = interim_dir / wmo
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{year}.csv"
        if not out_path.exists():
            out_path.write_bytes(csv_path.read_bytes())
            moved += 1

        if wmo not in metas:
            metas[wmo] = _parse_metadata(
                csv_path, config["ingest"]["header_rows"], config["ingest"]["encoding"]
            )

    for wmo, meta in metas.items():
        with open(interim_dir / wmo / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # Conteo verídico: archivos efectivamente parseados (incluye los que ya estaban).
    parsed = sum(1 for d in interim_dir.iterdir() if d.is_dir() for _ in d.glob("*.csv"))
    stations_done = sum(
        1 for d in interim_dir.iterdir() if d.is_dir() and any(d.glob("*.csv"))
    )
    log.info(
        "Ingesta OK: %d archivos parseados (%d nuevos), %d estaciones procesadas",
        parsed, moved, stations_done,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    ingest(load_yaml(args.config))


if __name__ == "__main__":
    main()
