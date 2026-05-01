"""EDA general: descripción del dataset, calidad, faltantes, duplicados, target."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import get_logger, load_parquet, load_yaml, save_json

log = get_logger(__name__)


def quality_report(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_pct": df.isna().mean().mul(100).round(3).to_dict(),
        "duplicated_index": int(df.index.duplicated().sum()),
        "min_ts": str(df.index.min()),
        "max_ts": str(df.index.max()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }


def target_distribution(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[target].dropna(), kde=True, ax=ax[0])
    ax[0].set_title(f"Histograma {target}")
    sns.boxplot(x=df[target], ax=ax[1])
    ax[1].set_title(f"Boxplot {target}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"target_distribution_{target}.png", dpi=150)
    plt.close(fig)


def correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(min(0.5 * num.shape[1] + 4, 18), 8))
    sns.heatmap(num.corr(), annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlación de Pearson")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def summary_by_region(station_dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Resumen de calidad agregado por macrorregión IBGE.

    Args:
        station_dfs: ``{station_code: DataFrame}`` con índice temporal.

    Returns:
        DataFrame con columnas ``[region, n_stations, n_samples,
        missing_pct_mean, ts_min, ts_max]``.
    """
    from src.utils.regions import region_of

    rows = []
    for code, df in station_dfs.items():
        rows.append({
            "station": code,
            "region": region_of(code),
            "n_samples": int(len(df)),
            "missing_pct": float(df.isna().mean().mean() * 100),
            "ts_min": df.index.min(),
            "ts_max": df.index.max(),
        })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    grp = raw.groupby("region")
    return grp.agg(
        n_stations=("station", "nunique"),
        n_samples=("n_samples", "sum"),
        missing_pct_mean=("missing_pct", "mean"),
        ts_min=("ts_min", "min"),
        ts_max=("ts_max", "max"),
    ).reset_index()


def run(config: dict) -> None:
    in_dir = Path(config["paths"]["data_processed"]) / "resampled"
    fig_dir = Path(config["paths"]["figures"]) / "eda_general"
    out_dir = Path(config["paths"]["tables"]) / "eda_general"
    target = config["task"]["target"]

    station_dfs: dict[str, pd.DataFrame] = {}
    for path in sorted(in_dir.glob("*.parquet")):
        df = load_parquet(path)
        rep = quality_report(df)
        save_json(rep, out_dir / f"{path.stem}.json")
        target_distribution(df, target, fig_dir / path.stem)
        correlation_heatmap(df, fig_dir / path.stem)
        station_dfs[path.stem] = df
        log.info("EDA general %s — %d filas", path.stem, rep["n_rows"])

    if station_dfs:
        try:
            summary_by_region(station_dfs).to_csv(
                out_dir / "summary_by_region.csv", index=False
            )
        except KeyError as exc:
            log.warning("summary_by_region: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
