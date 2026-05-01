"""Split temporal SIN data leakage.

Garantías que se verifican en ``tests/test_split_no_leakage.py``:

1. Orden cronológico estricto: ``max(train.index) < min(val.index) <= max(val.index) < min(test.index)``.
2. No hay timestamps duplicados entre splits.
3. Si ``mode == by_year``, ningún año aparece en más de un split.
4. Sin barajado: el split es determinista a partir del índice temporal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_parquet, load_yaml, save_parquet

log = get_logger(__name__)


@dataclass(frozen=True)
class TemporalSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def assert_no_leakage(self) -> None:
        if not (self.train.index.is_monotonic_increasing
                and self.val.index.is_monotonic_increasing
                and self.test.index.is_monotonic_increasing):
            raise AssertionError("Algún split no está ordenado cronológicamente.")
        if not self.train.index.max() < self.val.index.min():
            raise AssertionError(
                f"Leakage train→val: max(train)={self.train.index.max()} >= min(val)={self.val.index.min()}"
            )
        if not self.val.index.max() < self.test.index.min():
            raise AssertionError(
                f"Leakage val→test: max(val)={self.val.index.max()} >= min(test)={self.test.index.min()}"
            )
        joint = (
            self.train.index.union(self.val.index).union(self.test.index)
        )
        if len(joint) != len(self.train) + len(self.val) + len(self.test):
            raise AssertionError("Timestamps duplicados entre splits.")


def _by_year(df: pd.DataFrame, cfg: dict) -> TemporalSplits:
    yrs = df.index.year
    train = df[yrs.isin(cfg["train_years"])]
    val = df[yrs.isin(cfg["val_years"])]
    test = df[yrs.isin(cfg["test_years"])]
    if set(cfg["train_years"]) & set(cfg["val_years"]) \
            or set(cfg["val_years"]) & set(cfg["test_years"]) \
            or set(cfg["train_years"]) & set(cfg["test_years"]):
        raise ValueError("Años solapados entre splits en config.split.by_year")
    return TemporalSplits(train, val, test)


def _by_ratio(df: pd.DataFrame, cfg: dict) -> TemporalSplits:
    n = len(df)
    n_tr = int(n * cfg["train"])
    n_va = int(n * cfg["val"])
    train = df.iloc[:n_tr]
    val = df.iloc[n_tr : n_tr + n_va]
    test = df.iloc[n_tr + n_va :]
    return TemporalSplits(train, val, test)


def _by_date(df: pd.DataFrame, cfg: dict) -> TemporalSplits:
    t_end = pd.Timestamp(cfg["train_end"])
    v_end = pd.Timestamp(cfg["val_end"])
    train = df.loc[: t_end]
    val = df.loc[t_end + pd.Timedelta(seconds=1) : v_end]
    test = df.loc[v_end + pd.Timedelta(seconds=1) :]
    return TemporalSplits(train, val, test)


def split_dataframe(df: pd.DataFrame, split_cfg: dict) -> TemporalSplits:
    df = df.sort_index()
    mode = split_cfg["mode"]
    if mode == "by_year":
        s = _by_year(df, split_cfg["by_year"])
    elif mode == "by_ratio":
        s = _by_ratio(df, split_cfg["by_ratio"])
    elif mode == "by_date":
        s = _by_date(df, split_cfg["by_date"])
    else:
        raise ValueError(f"Modo de split desconocido: {mode}")
    s.assert_no_leakage()
    return s


def _apply_sampling(config: dict) -> tuple[dict, list[str] | None]:
    """Devuelve (split_cfg_efectivo, lista_estaciones_permitidas | None).

    Si ``config.sampling.enabled`` es True, recorta los años de train y la
    lista de estaciones a procesar. NO toca la lógica de split.
    """
    split_cfg = dict(config["split"])
    sampling = config.get("sampling", {}) or {}
    if not sampling.get("enabled", False):
        return split_cfg, None

    # Train years recortados (val y test se mantienen)
    new_by_year = dict(split_cfg.get("by_year", {}))
    new_by_year["train_years"] = list(
        sampling.get("train_years", new_by_year.get("train_years", []))
    )
    split_cfg["by_year"] = new_by_year

    # Estaciones: 1 por región × 5 regiones (default) — usa regions util
    from src.utils.regions import all_regions, stations_by_region
    n_per_region = int(sampling.get("n_stations_per_region", 1))
    sampled = []
    for region in all_regions():
        sampled.extend(stations_by_region(region)[:n_per_region])
    log.warning(
        "MUESTREO PROVISIONAL ACTIVO: %d estaciones (%s), train_years=%s",
        len(sampled), sampled, new_by_year["train_years"],
    )
    return split_cfg, sampled


def run(config: dict) -> None:
    in_dir = Path(config["paths"]["data_processed"]) / "features"
    out_dir = Path(config["paths"]["data_processed"])
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    split_cfg, sampled_stations = _apply_sampling(config)
    sampled_set = {s.upper() for s in sampled_stations} if sampled_stations else None

    for path in sorted(in_dir.glob("*.parquet")):
        if sampled_set is not None and path.stem.upper() not in sampled_set:
            continue
        df = load_parquet(path)
        s = split_dataframe(df, split_cfg)
        log.info(
            "%s — train=%d, val=%d, test=%d (rangos: %s | %s | %s)",
            path.stem, len(s.train), len(s.val), len(s.test),
            (s.train.index.min(), s.train.index.max()) if len(s.train) else None,
            (s.val.index.min(), s.val.index.max()) if len(s.val) else None,
            (s.test.index.min(), s.test.index.max()) if len(s.test) else None,
        )
        save_parquet(s.train, out_dir / "train" / path.name)
        save_parquet(s.val,   out_dir / "val"   / path.name)
        save_parquet(s.test,  out_dir / "test"  / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
