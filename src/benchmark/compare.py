"""Agrega métricas por (modelo, semilla) → tabla con media ± std e IC bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger, load_yaml

log = get_logger(__name__)


def _load_runs(experiments_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in experiments_root.rglob("metrics.json"):
        # experiments/<model>/<station>/seed=<s>/metrics.json
        seed_dir, station_dir, model_dir = metrics_path.parents[0:3]
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "model": model_dir.name,
            "station": station_dir.name,
            "seed": int(seed_dir.name.split("=")[1]),
            "rmse": m["rmse_total"],
            "mae": m["mae_total"],
            "r2": m["r2_total"],
            "mape": m["mape_total"],
            "smape": m["smape_total"],
            "per_horizon": m.get("per_horizon"),
        })
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, n: int, alpha: float) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def aggregate(
    df: pd.DataFrame,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    group_by: str = "station",
) -> pd.DataFrame:
    """Agrega métricas por (modelo, ``group_by``).

    Args:
        df: tabla larga producida por :func:`_load_runs`.
        n_boot: nº de re-muestreos bootstrap para los IC.
        alpha: nivel del IC (0.05 → IC95%).
        group_by: ``"station"`` (default) o ``"region"``. Cuando se agrupa
            por región se mapea cada estación con
            :func:`src.utils.regions.region_of` antes de agregar.
    """
    if group_by not in {"station", "region"}:
        raise ValueError(f"group_by debe ser 'station' o 'region', no {group_by!r}")

    df = df.copy()
    if group_by == "region":
        from src.utils.regions import region_of

        df["region"] = df["station"].map(region_of)

    rows = []
    for (model, key), grp in df.groupby(["model", group_by]):
        for metric in ["rmse", "mae", "r2", "mape", "smape"]:
            v = grp[metric].to_numpy()
            lo, hi = _bootstrap_ci(v, n_boot, alpha) if len(v) > 1 else (np.nan, np.nan)
            rows.append({
                "model": model, group_by: key, "metric": metric,
                "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else 0.0,
                "ci_lo": lo, "ci_hi": hi, "n_runs": len(v),
            })
    return pd.DataFrame(rows)


def per_horizon_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE/MAE/R² por modelo × región × paso del horizonte (media entre seeds y estaciones)."""
    from src.utils.regions import region_of

    rows = []
    for _, r in df.iterrows():
        ph = r["per_horizon"]
        if not ph:
            continue
        region = region_of(r["station"])
        for h, val in enumerate(ph["rmse"]):
            rows.append({
                "model": r["model"], "region": region, "h": h + 1,
                "rmse_h": val, "mae_h": ph["mae"][h], "r2_h": ph["r2"][h],
            })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    return (
        raw.groupby(["model", "region", "h"])
        .agg(rmse_h=("rmse_h", "mean"), mae_h=("mae_h", "mean"), r2_h=("r2_h", "mean"))
        .reset_index()
    )


def per_horizon_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        ph = r["per_horizon"]
        if not ph:
            continue
        for h, val in enumerate(ph["rmse"]):
            rows.append({
                "model": r["model"], "station": r["station"], "seed": r["seed"],
                "h": h + 1, "rmse_h": val, "mae_h": ph["mae"][h], "r2_h": ph["r2"][h],
            })
    return pd.DataFrame(rows)


def run(config: dict) -> None:
    exp_root = Path(config["paths"]["experiments"])
    out_dir = Path(config["paths"]["tables"]) / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(exp_root)
    if runs.empty:
        log.warning("No se encontraron runs en %s", exp_root)
        return
    runs.to_csv(out_dir / "runs_raw.csv", index=False)

    agg = aggregate(
        runs,
        n_boot=config["evaluation"]["bootstrap_n"],
        alpha=config["evaluation"]["bootstrap_alpha"],
        group_by="station",
    )
    agg.to_csv(out_dir / "agg_mean_std_ci.csv", index=False)

    agg_region = aggregate(
        runs,
        n_boot=config["evaluation"]["bootstrap_n"],
        alpha=config["evaluation"]["bootstrap_alpha"],
        group_by="region",
    )
    agg_region.to_csv(out_dir / "agg_mean_std_ci_by_region.csv", index=False)

    ph = per_horizon_table(runs)
    if not ph.empty:
        ph.to_csv(out_dir / "per_horizon.csv", index=False)
        ph_region = per_horizon_by_region(runs)
        ph_region.to_csv(out_dir / "per_horizon_by_region.csv", index=False)

    log.info(
        "Tabla agregada guardada (modelos=%d, runs=%d, regiones=%d)",
        runs["model"].nunique(), len(runs), agg_region["region"].nunique() if not agg_region.empty else 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
