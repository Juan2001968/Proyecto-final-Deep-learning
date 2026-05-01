"""Reporte final: tabla agregada + plots para el Jupyter Book."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import get_logger, load_yaml

log = get_logger(__name__)


def _fmt_table(agg: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Tabla legible: una fila por modelo/estación con `media ± std (IC95%)`."""
    sub = agg.query("metric == @metric").copy()
    sub["display"] = sub.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]",
        axis=1,
    )
    return sub.pivot(index="model", columns="station", values="display")


def _bar_plot(agg: pd.DataFrame, metric: str, out_path: Path) -> None:
    sub = agg.query("metric == @metric")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sub, x="model", y="mean", hue="station", ax=ax,
                errorbar=None)
    for _, r in sub.iterrows():
        ax.errorbar(x=r["model"], y=r["mean"],
                    yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                    fmt="none", color="black", capsize=3, lw=0.8)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Comparativa modelos — {metric.upper()} (media + IC95% bootstrap)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _per_horizon_plot(ph: pd.DataFrame, out_path: Path) -> None:
    if ph.empty:
        return
    grp = ph.groupby(["model", "h"])["rmse_h"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=grp, x="h", y="rmse_h", hue="model", marker="o", ax=ax)
    ax.set_xlabel("paso del horizonte")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE por paso del horizonte")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_by_region(
    agg_region: pd.DataFrame, out_path: Path, *, metric: str = "rmse",
) -> None:
    """Compara modelos × región usando la paleta consistente de regions.yaml.

    Args:
        agg_region: tabla devuelta por ``aggregate(..., group_by='region')``.
        out_path: ruta del PNG a generar.
        metric: ``"rmse"`` | ``"mae"`` | ``"r2"`` | ``"mape"`` | ``"smape"``.
    """
    from src.utils.regions import all_regions, region_color_map

    sub = agg_region.query("metric == @metric").copy()
    if sub.empty:
        return
    palette = region_color_map()
    order = [r for r in all_regions() if r in sub["region"].unique()]

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=sub, x="model", y="mean", hue="region",
        hue_order=order, palette=palette, ax=ax, errorbar=None,
    )
    for _, r in sub.iterrows():
        ax.errorbar(
            x=r["model"], y=r["mean"],
            yerr=[[max(r["mean"] - r["ci_lo"], 0)], [max(r["ci_hi"] - r["mean"], 0)]],
            fmt="none", color="black", capsize=3, lw=0.7,
        )
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} por modelo × region (media + IC95% bootstrap)")
    ax.legend(title="region", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_horizon_by_region(
    ph_region: pd.DataFrame, out_path: Path, *, metric: str = "rmse_h",
) -> None:
    """Curvas RMSE-por-horizonte una por región, con la paleta consistente."""
    from src.utils.regions import all_regions, region_color_map

    if ph_region.empty:
        return
    palette = region_color_map()
    order = [r for r in all_regions() if r in ph_region["region"].unique()]

    g = sns.relplot(
        data=ph_region, x="h", y=metric, hue="region", col="model",
        kind="line", marker="o", palette=palette, hue_order=order,
        col_wrap=3, height=3.2, aspect=1.4, facet_kws={"sharey": True},
    )
    g.set_axis_labels("paso del horizonte", metric.replace("_h", "").upper())
    g.fig.suptitle(f"{metric.replace('_h', '').upper()} por horizonte y region", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def run(config: dict) -> None:
    tab_dir = Path(config["paths"]["tables"]) / "benchmark"
    fig_dir = Path(config["paths"]["figures"]) / "benchmark"

    agg_path = tab_dir / "agg_mean_std_ci.csv"
    if not agg_path.exists():
        log.warning("Falta %s — corre primero src.benchmark.compare", agg_path); return

    agg = pd.read_csv(agg_path)
    for metric in ["rmse", "mae", "r2"]:
        _fmt_table(agg, metric).to_csv(tab_dir / f"final_table_{metric}.csv")
        _bar_plot(agg, metric, fig_dir / f"bar_{metric}.png")

    ph_path = tab_dir / "per_horizon.csv"
    if ph_path.exists():
        _per_horizon_plot(pd.read_csv(ph_path), fig_dir / "per_horizon_rmse.png")

    region_path = tab_dir / "agg_mean_std_ci_by_region.csv"
    if region_path.exists():
        agg_region = pd.read_csv(region_path)
        for metric in ["rmse", "mae", "r2"]:
            plot_metrics_by_region(agg_region, fig_dir / f"by_region_{metric}.png", metric=metric)

    ph_region_path = tab_dir / "per_horizon_by_region.csv"
    if ph_region_path.exists():
        plot_per_horizon_by_region(
            pd.read_csv(ph_region_path), fig_dir / "per_horizon_by_region.png"
        )

    log.info("Reporte final escrito en %s y %s", tab_dir, fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
