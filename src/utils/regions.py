"""Helpers para trabajar con macrorregiones IBGE en todo el proyecto.

Los datos se leen una sola vez desde:

- ``config/regions.yaml``  — definicion de regiones (paleta, biomas, etc.).
- ``config/stations.yaml`` — fuente de verdad de las estaciones.

Las cargas estan cacheadas con ``functools.lru_cache`` para evitar releer
en cada acceso. ``assert_consistency()`` valida que ambos archivos estan
sincronizados.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REGIONS_PATH = Path("config/regions.yaml")
STATIONS_PATH = Path("config/stations.yaml")


# --------------------------------------------------------------------- loaders

@lru_cache(maxsize=1)
def load_regions() -> dict:
    """Devuelve el diccionario completo de ``config/regions.yaml``.

    Returns:
        Diccionario ``{nombre_region: {states, dominant_biomes, koppen_classes,
        station_codes, n_stations, color, ...}}``.
    """
    with open(REGIONS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("regions", {})


@lru_cache(maxsize=1)
def load_stations() -> list[dict]:
    """Devuelve la lista de estaciones de ``config/stations.yaml``.

    Returns:
        Lista de dicts con al menos las claves ``code``, ``name``, ``uf``,
        ``region``, ``biome``, ``koppen_class``.
    """
    with open(STATIONS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("stations", []))


# ------------------------------------------------------------------- consultas

@lru_cache(maxsize=None)
def stations_by_region(region: str) -> list[str]:
    """Devuelve los codigos WMO de las estaciones de una region.

    Args:
        region: Nombre de la macrorregion IBGE (p. ej. ``"Norte"``).

    Returns:
        Lista de codigos WMO en el orden definido en ``regions.yaml``.

    Raises:
        KeyError: Si ``region`` no existe en ``regions.yaml``.
    """
    regions = load_regions()
    if region not in regions:
        raise KeyError(f"Region desconocida: {region!r}. Validas: {sorted(regions)}")
    return list(regions[region].get("station_codes", []))


@lru_cache(maxsize=None)
def region_of(station_code: str) -> str:
    """Devuelve la macrorregion IBGE a la que pertenece una estacion.

    Args:
        station_code: Codigo WMO de la estacion.

    Returns:
        Nombre de la region.

    Raises:
        KeyError: Si la estacion no aparece en ``stations.yaml``.
    """
    code = station_code.upper()
    for s in load_stations():
        if s["code"].upper() == code:
            return s["region"]
    raise KeyError(f"Estacion {station_code!r} no encontrada en stations.yaml")


@lru_cache(maxsize=None)
def region_color(region: str) -> str:
    """Devuelve el color hex consistente para los plots de una region.

    Args:
        region: Nombre de la macrorregion IBGE.

    Returns:
        Cadena hex en formato ``"#RRGGBB"``.
    """
    regions = load_regions()
    if region not in regions:
        raise KeyError(f"Region desconocida: {region!r}")
    return regions[region]["color"]


@lru_cache(maxsize=1)
def all_regions() -> list[str]:
    """Devuelve la lista de regiones definidas en orden de aparicion.

    Returns:
        Lista de nombres de region (p. ej. ``["Norte", "Nordeste", ...]``).
    """
    return list(load_regions().keys())


@lru_cache(maxsize=1)
def region_color_map() -> dict[str, str]:
    """Devuelve un mapa region -> color hex listo para ``hue_order``/``palette``."""
    return {r: region_color(r) for r in all_regions()}


# ------------------------------------------------------------------ validacion

def assert_consistency() -> None:
    """Valida que ``regions.yaml`` y ``stations.yaml`` esten sincronizados.

    Reglas:
        1. Toda estacion en ``stations.yaml`` aparece en exactamente un
           ``region.station_codes``.
        2. Todo codigo en ``region.station_codes`` existe en ``stations.yaml``.
        3. ``n_stations == len(station_codes)``.
        4. La ``region`` declarada en cada estacion coincide con aquella
           cuya lista la contiene.

    Raises:
        ValueError: Si alguna de las reglas anteriores se viola, con un
            mensaje describiendo el problema.
    """
    regions = load_regions()
    stations = load_stations()
    station_codes_global = {s["code"].upper() for s in stations}

    # 3) consistencia n_stations <-> len(station_codes)
    for name, info in regions.items():
        codes = info.get("station_codes", [])
        n = info.get("n_stations")
        if n != len(codes):
            raise ValueError(
                f"Region {name!r}: n_stations={n} != len(station_codes)={len(codes)}"
            )

    # 1) cada estacion en exactamente una region
    seen: dict[str, str] = {}
    for region_name, info in regions.items():
        for code in info.get("station_codes", []):
            code_u = code.upper()
            if code_u in seen:
                raise ValueError(
                    f"Estacion {code!r} aparece en {seen[code_u]!r} y en {region_name!r}"
                )
            seen[code_u] = region_name

    missing_in_regions = station_codes_global - seen.keys()
    if missing_in_regions:
        raise ValueError(
            f"Estaciones en stations.yaml ausentes de regions.yaml: {sorted(missing_in_regions)}"
        )

    # 2) ningun codigo fantasma en regions.yaml
    extra_in_regions = seen.keys() - station_codes_global
    if extra_in_regions:
        raise ValueError(
            f"Codigos en regions.yaml inexistentes en stations.yaml: {sorted(extra_in_regions)}"
        )

    # 4) la region declarada en stations.yaml coincide con su contenedor
    for s in stations:
        declared = s["region"]
        actual = seen[s["code"].upper()]
        if declared != actual:
            raise ValueError(
                f"Estacion {s['code']!r}: region declarada {declared!r} != "
                f"region contenedora {actual!r}"
            )
