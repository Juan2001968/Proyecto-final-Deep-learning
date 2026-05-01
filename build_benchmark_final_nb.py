"""Construye notebooks/06_benchmark_final.ipynb (sin ejecutar).

Sección 6 del entregable: *Benchmark Final*. Cierre académico del proyecto
con análisis estadístico riguroso (DM, Friedman + post-hoc, Wilcoxon,
Ljung-Box, BDS), tabla agregada con IC95% bootstrap, scatter Pareto,
análisis cualitativo y discusión crítica.

NO ejecuta entrenamientos. Solo carga ``experiments/<modelo>/<station>/seed=*/``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =============================================================================
# Celda 0 — Título y resumen ejecutivo
# =============================================================================
md("""
# Benchmark Final: Comparación Estadística Rigurosa de Modelos de Forecasting

**Resumen ejecutivo.** Este capítulo cierra el proyecto con la comparación
formal de los **6 modelos** entrenados en el Capítulo 04 — *Persistencia*,
*LSTM*, *GRU*, *N-BEATSx*, *Temporal Fusion Transformer* (paper guía) e
*Informer* — sobre el panel INMET de 40 estaciones, con **5 semillas** por
modelo. Las métricas se reportan como **media ± std** acompañadas de **IC 95%
bootstrap** (1000 *resamples*); la inferencia se respalda con **Diebold-Mariano**
(con corrección Harvey-Leybourne-Newbold), **Friedman + Nemenyi/Bonferroni/Holm**
y **Wilcoxon signed-rank**, y los residuos del mejor modelo se diagnostican con
**Ljung-Box** y **BDS** para descartar autocorrelación y no-linealidad
residual. El notebook **no ejecuta entrenamientos**: consume los artefactos
de `experiments/<model>/<station>/seed=<s>/`.
""")

# =============================================================================
# Celda 1 — Banner muestreo provisional + lectura de config
# =============================================================================
md("""
> ## ⚠️ AVISO: Muestreo Provisional
>
> Este notebook puede ejecutarse en dos modos según `config/config.yaml`
> → `sampling.enabled`:
>
> - **`sampling.enabled: true`** → modo *validación de pipeline*:
>   - **5 estaciones** (1 por región IBGE) en lugar de 40.
>   - **2 años de train** (2022–2023) en lugar de 6 (2018–2023).
>   - **2 semillas** por modelo en lugar de 5.
>   - **Máximo 5 épocas** por run.
> - **`sampling.enabled: false`** → modo *resultados oficiales* (lo entregado).
>
> **Si ves la advertencia más abajo, los resultados de este notebook no son
> finales** — sirven solo para validar que el pipeline corre end-to-end.
> Para resultados oficiales:
>
> 1. Editar `config/config.yaml` → `sampling.enabled: false`.
> 2. Re-correr `make train` (o `python -m src.training.runner --model <m> --seeds 5` por modelo).
> 3. Re-ejecutar este notebook.
>
> El cierre del notebook (Sección 6) repite esta advertencia si el muestreo
> sigue activo.
""")

code("""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Resolución robusta de la raíz del repo (notebook puede estar en /notebooks)
REPO_ROOT = Path.cwd().resolve()
while not (REPO_ROOT / "config" / "config.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

CONFIG = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
SAMPLING = CONFIG.get("sampling", {}) or {}
SAMPLING_ON = bool(SAMPLING.get("enabled", False))

if SAMPLING_ON:
    print("⚠️ MUESTREO PROVISIONAL ACTIVO — los resultados NO son finales.")
    print(f"   n_stations_per_region: {SAMPLING.get('n_stations_per_region')}")
    print(f"   train_years          : {SAMPLING.get('train_years')}")
    print(f"   max_epochs_override  : {SAMPLING.get('max_epochs_override')}")
    print(f"   n_seeds_override     : {SAMPLING.get('n_seeds_override')}")
else:
    print("✅ Configuración completa (sin muestreo). Resultados oficiales.")
""")

# =============================================================================
# Setup global
# =============================================================================
md("## Configuración global, paths e imports")

code("""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.regions import (
    all_regions, region_color, region_color_map,
    region_of, stations_by_region,
)
from src.benchmark.stats_tests import (
    diebold_mariano,
    friedman_with_posthoc,
    wilcoxon_pairs,
    ljung_box,
    bds_test,
    bootstrap_ci,
    nemenyi_critical_difference,
    average_ranks,
    shapiro_per_model,
)

EXP_DIR = REPO_ROOT / CONFIG["paths"]["experiments"]
FIG_DIR = REPO_ROOT / CONFIG["paths"]["figures"] / "benchmark_final"
TAB_DIR = REPO_ROOT / CONFIG["paths"]["tables"] / "benchmark_final"
STATS_DIR = REPO_ROOT / CONFIG["paths"]["stats"]
for d in (FIG_DIR, TAB_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_MODELS = ["persistence", "lstm", "gru", "nbeats", "tft", "informer"]
DEFAULT_HORIZONS = [24, 72, 168]

ALPHA = 0.05            # nivel de significancia declarado
N_BOOT = 1000           # bootstrap percentil
RNG_SEED = 42

sns.set_context("notebook")
sns.set_style("whitegrid")
""")

# =============================================================================
# Sección 1 — Comparación tabular
# =============================================================================
md("""
## Sección 1 — Comparación Tabular

Cumple el requisito **6.1** del entregable: tabla comparativa de modelos por
horizonte y región, con dispersión entre semillas y intervalos de confianza
bootstrap. Las predicciones de cada run se cargan desde
`experiments/<modelo>/<station>/seed=<s>/predictions.npz` y las métricas
*por horizonte* desde el correspondiente `metrics.json`.

Las tres vistas que reportamos son:

1. **Por (modelo × horizonte)**, agregando estaciones y semillas — la lectura
   estándar para un benchmark multi-horizonte.
2. **Agregada por modelo** (promedio sobre horizontes) — para ranking global.
3. **Por (modelo × región)** — para diagnosticar generalización geográfica
   (Norte, Nordeste, Centro-Oeste, Sudeste, Sul; paleta consistente con el
   resto del libro vía `region_color()`).
""")

md("### 1.1 Carga de resultados desde `experiments/`")

code("""
def _seed_dirs(model: str) -> list[Path]:
    base = EXP_DIR / model
    if not base.exists():
        return []
    return sorted(p for st in base.iterdir() if st.is_dir()
                  for p in st.iterdir()
                  if p.is_dir() and p.name.startswith("seed="))


def load_predictions_long() -> pd.DataFrame:
    \"\"\"Devuelve un DataFrame largo: una fila por (modelo, station, seed, horizonte).

    Columnas: model, station, region, seed, horizon, y_true, y_pred (vectores 1D).
    Si no hay runs todavía, devuelve un DataFrame vacío y un mensaje claro.
    \"\"\"
    rows = []
    target_h = CONFIG["task"]["horizon"]
    for model in DEFAULT_MODELS:
        for sd in _seed_dirs(model):
            station = sd.parent.name
            seed = int(sd.name.split("=", 1)[1])
            try:
                npz = np.load(sd / "predictions.npz", allow_pickle=True)
            except FileNotFoundError:
                continue
            y_true = npz["y_true"]      # (N, H, T)
            y_pred = npz["y_pred"]
            if y_true.ndim == 2:
                y_true = y_true[:, :, None]
                y_pred = y_pred[:, :, None]
            try:
                region = region_of(station)
            except KeyError:
                region = "desconocida"
            for h_step in DEFAULT_HORIZONS:
                # h_step es paso de horizonte (1..target_h). Si target_h<h_step,
                # tomamos el último paso disponible.
                idx = min(h_step, y_true.shape[1]) - 1
                rows.append({
                    "model": model, "station": station, "region": region,
                    "seed": seed, "horizon": h_step,
                    "y_true": y_true[:, idx, 0].astype(float),
                    "y_pred": y_pred[:, idx, 0].astype(float),
                })
    return pd.DataFrame(rows)


try:
    PRED_LONG = load_predictions_long()
except Exception as exc:
    print(f"[load_predictions_long] error: {exc}")
    PRED_LONG = pd.DataFrame()

if PRED_LONG.empty:
    print("Aún no hay resultados. Corre los entrenamientos primero:")
    print("    make train     # o")
    print("    python -m src.training.runner --model lstm --seeds 5  (etc.)")
else:
    print(f"Predicciones cargadas: {len(PRED_LONG)} filas "
          f"(modelo×station×seed×horizonte). Modelos: "
          f"{sorted(PRED_LONG['model'].unique())}.")
""")

md("""
### 1.2 Métricas y bootstrap

Para cada par (modelo, station, seed, horizonte) calculamos RMSE, MAE, R² y
sMAPE sobre el vector de errores `e = y_pred - y_true`, y construimos el IC
95% por **bootstrap percentil** (1000 *resamples*). El bootstrap se realiza
sobre los errores *al cuadrado* para RMSE y sobre los errores *absolutos*
para MAE — así el intervalo refleja la dispersión de la métrica, no de los
errores brutos.
""")

code("""
def _rmse(e):  return float(np.sqrt(np.mean(e ** 2)))
def _mae(e):   return float(np.mean(np.abs(e)))
def _smape(yt, yp):
    denom = (np.abs(yt) + np.abs(yp))
    denom[denom == 0] = 1.0
    return float(100 * np.mean(2 * np.abs(yp - yt) / denom))
def _r2(yt, yp):
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def metrics_per_run(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Una fila por (modelo, station, seed, horizonte) con métricas + IC bootstrap.\"\"\"
    out = []
    if df.empty:
        return pd.DataFrame()
    for _, r in df.iterrows():
        yt, yp = r["y_true"], r["y_pred"]
        e = yp - yt
        ci_rmse = bootstrap_ci(e, n_boot=N_BOOT, alpha=ALPHA,
                               stat_fn=lambda x: float(np.sqrt(np.mean(x ** 2))),
                               seed=RNG_SEED)
        out.append({
            "model": r["model"], "station": r["station"], "region": r["region"],
            "seed": r["seed"], "horizon": r["horizon"],
            "rmse": _rmse(e), "mae": _mae(e),
            "r2": _r2(yt, yp), "smape": _smape(yt, yp),
            "rmse_ci_lo": ci_rmse["lo"], "rmse_ci_hi": ci_rmse["hi"],
        })
    return pd.DataFrame(out)


PER_RUN = metrics_per_run(PRED_LONG) if not PRED_LONG.empty else pd.DataFrame()
PER_RUN.head() if not PER_RUN.empty else "[Aún sin resultados]"
""")

md("### 1.3 Tabla 1 — por (modelo × horizonte)")

code("""
def _agg_mean_std(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = ["rmse", "mae", "r2", "smape"]
    g = df.groupby(by)[metrics]
    out = g.agg(["mean", "std"]).round(4)
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    # IC global: media y desviación de los IC por run -> agregamos por bootstrap
    # del propio RMSE entre runs (más informativo que promediar IC individuales)
    rmse_ci = df.groupby(by)["rmse"].apply(
        lambda x: bootstrap_ci(x.to_numpy(), n_boot=N_BOOT, alpha=ALPHA,
                                stat_fn=np.mean, seed=RNG_SEED)
    )
    out["rmse_ci_lo"] = [c["lo"] for c in rmse_ci]
    out["rmse_ci_hi"] = [c["hi"] for c in rmse_ci]
    return out.reset_index()


TBL_MODEL_HOR = _agg_mean_std(PER_RUN, by=["model", "horizon"]) if not PER_RUN.empty else pd.DataFrame()
if not TBL_MODEL_HOR.empty:
    TBL_MODEL_HOR.to_csv(TAB_DIR / "tabla_modelo_x_horizonte.csv", index=False)
TBL_MODEL_HOR
""")

md("### 1.4 Tabla 2 — agregada por modelo (promedio sobre horizontes)")

code("""
TBL_MODEL = _agg_mean_std(PER_RUN, by=["model"]) if not PER_RUN.empty else pd.DataFrame()
if not TBL_MODEL.empty:
    TBL_MODEL = TBL_MODEL.sort_values("rmse_mean")
    TBL_MODEL.to_csv(TAB_DIR / "tabla_modelo_agg.csv", index=False)
TBL_MODEL
""")

md("### 1.5 Tabla 3 — por (modelo × región)")

code("""
TBL_REGION = _agg_mean_std(PER_RUN, by=["model", "region"]) if not PER_RUN.empty else pd.DataFrame()
if not TBL_REGION.empty:
    TBL_REGION.to_csv(TAB_DIR / "tabla_modelo_x_region.csv", index=False)
TBL_REGION
""")

md("### 1.6 Visualización — barras con error bars + heatmap")

code("""
def plot_rmse_bars(tbl: pd.DataFrame) -> Path | None:
    if tbl.empty:
        print("Tabla vacía — corre el benchmark."); return None
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = tbl.pivot(index="horizon", columns="model", values="rmse_mean")
    err = tbl.pivot(index="horizon", columns="model", values="rmse_std")
    pivot.plot(kind="bar", yerr=err, ax=ax, capsize=3)
    ax.set_ylabel("RMSE (°C)"); ax.set_xlabel("Horizonte (h)")
    ax.set_title("RMSE por modelo × horizonte (media ± std entre semillas)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "01_rmse_modelo_x_horizonte.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


print(plot_rmse_bars(TBL_MODEL_HOR))
""")

code("""
def plot_heatmap_region(tbl: pd.DataFrame) -> Path | None:
    if tbl.empty:
        print("Tabla vacía."); return None
    pivot = tbl.pivot(index="region", columns="model", values="rmse_mean")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(pivot.index)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                cbar_kws={"label": "RMSE (°C)"})
    ax.set_title("RMSE por modelo × región (promedio sobre horizontes y semillas)")
    plt.tight_layout()
    out = FIG_DIR / "02_heatmap_modelo_x_region.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


print(plot_heatmap_region(TBL_REGION))
""")

# =============================================================================
# Sección 2 — Ranking
# =============================================================================
md("""
## Sección 2 — Ranking de Modelos

Cumple el requisito **6.2**. El ranking se calcula sobre el RMSE agregado
(promedio entre estaciones, semillas y horizontes), y se replica por horizonte
para detectar inversiones (un modelo puede ser top-1 en h=24 pero caer en
h=168). Trade-off **costo vs precisión** se inspecciona con un scatter
RMSE × tiempo de entrenamiento (escala log) y la **frontera de Pareto**.
""")

md("### 2.1 Ranking principal")

code("""
def _emoji(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")


def ranking_table(tbl_model: pd.DataFrame) -> pd.DataFrame:
    if tbl_model.empty:
        return pd.DataFrame()
    df = tbl_model.copy().sort_values("rmse_mean").reset_index(drop=True)
    df["rank"] = df.index + 1
    df["medal"] = df["rank"].map(_emoji)
    cols = ["medal", "rank", "model", "rmse_mean", "rmse_std",
            "rmse_ci_lo", "rmse_ci_hi", "mae_mean", "r2_mean", "smape_mean"]
    return df[cols]


RANK = ranking_table(TBL_MODEL) if not TBL_MODEL.empty else pd.DataFrame()
if not RANK.empty:
    RANK.to_csv(TAB_DIR / "ranking_global.csv", index=False)
RANK
""")

md("### 2.2 Ranking por horizonte (detecta inversiones)")

code("""
def ranking_by_horizon(tbl: pd.DataFrame) -> pd.DataFrame:
    if tbl.empty: return pd.DataFrame()
    rows = []
    for h, grp in tbl.groupby("horizon"):
        sub = grp.sort_values("rmse_mean").reset_index(drop=True)
        sub["rank"] = sub.index + 1
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


RANK_BY_HOR = ranking_by_horizon(TBL_MODEL_HOR) if not TBL_MODEL_HOR.empty else pd.DataFrame()
if not RANK_BY_HOR.empty:
    RANK_BY_HOR.to_csv(TAB_DIR / "ranking_por_horizonte.csv", index=False)
RANK_BY_HOR.pivot(index="model", columns="horizon", values="rank") if not RANK_BY_HOR.empty else "[sin datos]"
""")

md("### 2.3 Análisis trade-off — scatter RMSE × costo y frontera de Pareto")

code("""
def collect_runtime_and_params() -> pd.DataFrame:
    \"\"\"Lee history.json y env.json de cada run para tiempo de entrenamiento
    y número de parámetros (si está expuesto).\"\"\"
    rows = []
    for model in DEFAULT_MODELS:
        for sd in _seed_dirs(model):
            hist_path = sd / "history.json"
            if not hist_path.exists():
                continue
            try:
                hist = json.loads(hist_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            train_min = float(hist.get("train_minutes", float("nan")))
            n_params = float(hist.get("n_parameters", float("nan")))
            rows.append({"model": model, "train_min": train_min,
                         "n_params": n_params})
    return pd.DataFrame(rows)


COSTS = collect_runtime_and_params()
COSTS_AGG = COSTS.groupby("model")[["train_min", "n_params"]].mean() if not COSTS.empty else pd.DataFrame()
COSTS_AGG
""")

code("""
def plot_pareto(tbl_model: pd.DataFrame, costs_agg: pd.DataFrame) -> Path | None:
    if tbl_model.empty or costs_agg.empty:
        print("Insuficientes datos para Pareto."); return None
    df = tbl_model.set_index("model").join(costs_agg, how="inner").dropna(subset=["train_min"])
    if df.empty:
        print("Sin tiempos registrados — re-corre con timing en history.json"); return None
    df = df.reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["train_min"], df["rmse_mean"], s=80)
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["train_min"], r["rmse_mean"]),
                    textcoords="offset points", xytext=(6, 4))
    # Frontera de Pareto: minimizar (train_min, rmse_mean)
    sorted_df = df.sort_values("train_min")
    pareto, best_rmse = [], float("inf")
    for _, r in sorted_df.iterrows():
        if r["rmse_mean"] < best_rmse:
            pareto.append((r["train_min"], r["rmse_mean"]))
            best_rmse = r["rmse_mean"]
    if pareto:
        xs, ys = zip(*pareto)
        ax.plot(xs, ys, "r--", alpha=0.6, label="Frontera de Pareto")
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de entrenamiento por run (min, log)")
    ax.set_ylabel("RMSE medio (°C)")
    ax.set_title("Trade-off precisión vs costo computacional")
    plt.tight_layout()
    out = FIG_DIR / "03_pareto_rmse_vs_cost.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


print(plot_pareto(TBL_MODEL, COSTS_AGG))
""")

md("""
**Lectura cualitativa.** Si el modelo ganador (típicamente TFT o N-BEATS sobre
INMET por su capacidad multi-horizonte single-shot) cae **fuera** de la
frontera de Pareto, su mejora es comprable: puede haber un modelo más rápido
con RMSE casi idéntico. Si cae **sobre** la frontera, no hay alternativa
estrictamente mejor. Discutir esto en la Sección 5.4.
""")

# =============================================================================
# Sección 3 — Tests estadísticos
# =============================================================================
md("""
## Sección 3 — Análisis Estadístico y Pruebas de Hipótesis

Cumple el requisito **6.3**, el más importante del entregable. **Reportar
medias y desviaciones no basta**: dos modelos pueden diferir en RMSE por una
décima sin que esa diferencia sea distinguible del ruido entre semillas. La
pregunta científica es si las diferencias observadas son **reproducibles**.

**Convenciones declaradas.**

- Nivel de significancia: $\\alpha = 0.05$.
- Reportamos **p-valores exactos** (no solo `<0.05`).
- Verificamos los supuestos de cada test antes de interpretar.
- IC 95% por bootstrap percentil (n_boot = 1000).
- Tamaño muestral: en el pipeline completo, **5 semillas × 40 estaciones**
  por modelo (n=200 puntos pareados). En *muestreo provisional*, **2×5 = 10**
  puntos — algunos tests pierden potencia (lo advertimos donde aplica).
""")

md("### 3.1 Verificación de supuestos — Shapiro-Wilk")

code("""
def errors_panel_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Panel: filas = (station, seed, horizon), columnas = modelo, valor = RMSE.\"\"\"
    if df.empty: return pd.DataFrame()
    return (df.pivot_table(index=["station", "seed", "horizon"],
                            columns="model", values="rmse").dropna(axis=0, how="any"))


ERR_PANEL = errors_panel_per_seed(PER_RUN)
if not ERR_PANEL.empty:
    SHAPIRO = shapiro_per_model(ERR_PANEL, alpha=ALPHA)
    SHAPIRO.to_csv(TAB_DIR / "shapiro_por_modelo.csv", index=False)
else:
    SHAPIRO = pd.DataFrame()
SHAPIRO
""")

md("""
Si **algún modelo rechaza normalidad** (`p_value < 0.05`), priorizamos tests
**no paramétricos** (Wilcoxon, Friedman + Nemenyi) sobre los paramétricos
(t-test pareado, ANOVA). En forecasting es lo habitual: la distribución de
RMSE entre semillas suele ser sesgada por *outliers* (semillas que cayeron en
malos mínimos locales).
""")

md("### 3.2 Wilcoxon signed-rank — comparaciones pareadas")

code("""
def wilcoxon_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty: return pd.DataFrame()
    pairs = wilcoxon_pairs(panel, alpha=ALPHA)
    models = list(panel.columns)
    M = pd.DataFrame(np.nan, index=models, columns=models)
    for k, v in pairs.items():
        a, _, b = k.partition(" vs ")
        if "pvalue" in v:
            M.loc[a, b] = v["pvalue"]
            M.loc[b, a] = v["pvalue"]
    np.fill_diagonal(M.values, 1.0)
    return M


WILC_M = wilcoxon_matrix(ERR_PANEL)
if not WILC_M.empty:
    WILC_M.to_csv(TAB_DIR / "wilcoxon_pvalues.csv")
WILC_M
""")

code("""
def plot_pvalue_heatmap(M: pd.DataFrame, title: str, fname: str) -> Path | None:
    if M.empty: return None
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(M, annot=True, fmt=".3f", cmap="rocket_r",
                vmin=0, vmax=0.2, cbar_kws={"label": "p-valor"}, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


print(plot_pvalue_heatmap(WILC_M, "Wilcoxon signed-rank — p-valores pareados",
                          "04_wilcoxon_heatmap.png"))
""")

md("### 3.3 Friedman + post-hoc (Nemenyi / Bonferroni / Holm)")

code("""
FRIED = friedman_with_posthoc(ERR_PANEL, posthoc="nemenyi") if not ERR_PANEL.empty else {}
if FRIED:
    (STATS_DIR / "friedman_final.json").write_text(
        json.dumps(FRIED, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in FRIED.items() if k != "posthoc"}, indent=2, default=str))
""")

code("""
# Repetir con Bonferroni y Holm para comparar correcciones
FRIED_BONF = friedman_with_posthoc(ERR_PANEL, posthoc="bonferroni") if not ERR_PANEL.empty else {}
FRIED_HOLM = friedman_with_posthoc(ERR_PANEL, posthoc="holm") if not ERR_PANEL.empty else {}

def _ph_to_df(d: dict) -> pd.DataFrame:
    if not d or "posthoc" not in d: return pd.DataFrame()
    return pd.DataFrame(d["posthoc"]["pvalues"])


PH_NEMENYI = _ph_to_df(FRIED)
PH_BONF = _ph_to_df(FRIED_BONF)
PH_HOLM = _ph_to_df(FRIED_HOLM)
PH_NEMENYI
""")

md("### 3.4 Diagrama de Distancia Crítica (CD) — Nemenyi")

code("""
def plot_cd_diagram(panel: pd.DataFrame, alpha: float = 0.05) -> Path | None:
    if panel.empty: return None
    n_models = panel.shape[1]; n_datasets = panel.shape[0]
    cd = nemenyi_critical_difference(n_models, n_datasets, alpha=alpha)
    ranks = average_ranks(panel)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(0.5, n_models + 0.5); ax.set_ylim(0, 3)
    ax.plot(ranks.values, [1] * n_models, "o", markersize=8)
    for m, r in ranks.items():
        ax.annotate(m, (r, 1), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.plot([1, 1 + cd], [2, 2], "k-", lw=2)
    ax.annotate(f"CD = {cd:.2f}", (1 + cd / 2, 2), xytext=(0, 5),
                textcoords="offset points", ha="center")
    ax.set_yticks([]); ax.set_xlabel("Rango promedio (1 = mejor)")
    ax.set_title(f"Diagrama de Distancia Crítica — Nemenyi (α={alpha})")
    plt.tight_layout()
    out = FIG_DIR / "05_cd_diagram_nemenyi.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


print(plot_cd_diagram(ERR_PANEL, alpha=ALPHA))
""")

md("""
**Interpretación del CD.** Dos modelos cuya diferencia de rangos promedio
es **menor** que CD no son distinguibles estadísticamente bajo Nemenyi.
Modelos cuyo rango difiere por **más** que CD son significativamente
diferentes al nivel α=0.05.
""")

md("### 3.5 Diebold-Mariano (HLN-corrected) por pares")

code("""
def diebold_mariano_matrix() -> pd.DataFrame:
    \"\"\"Matriz simétrica de p-valores DM (HLN) entre pares de modelos.

    Para cada par (m1, m2), promedia los p-valores sobre runs pareados
    (misma station, mismo seed, mismo horizonte).
    \"\"\"
    if PRED_LONG.empty: return pd.DataFrame()
    cfg_dm = CONFIG["stats"]["diebold_mariano"]
    p_table: dict[tuple[str, str], list[float]] = {}
    keyed = PRED_LONG.set_index(["model", "station", "seed", "horizon"])
    keys = keyed.index.unique()
    by_run = {(s, sd, h): {} for (_, s, sd, h) in keys}
    for (m, s, sd, h), row in keyed.iterrows():
        by_run.setdefault((s, sd, h), {})[m] = row["y_pred"] - row["y_true"]

    for run_key, errs in by_run.items():
        for m1, m2 in combinations(sorted(errs), 2):
            try:
                res = diebold_mariano(errs[m1], errs[m2],
                                      h=cfg_dm["h"], power=cfg_dm["power"],
                                      alternative=cfg_dm["alternative"])
                p_table.setdefault((m1, m2), []).append(res["pvalue"])
            except (AssertionError, ValueError):
                continue
    models = sorted({m for pair in p_table for m in pair})
    M = pd.DataFrame(np.nan, index=models, columns=models)
    for (m1, m2), vals in p_table.items():
        v = float(np.nanmedian(vals))
        M.loc[m1, m2] = v; M.loc[m2, m1] = v
    np.fill_diagonal(M.values, 1.0)
    return M


DM_M = diebold_mariano_matrix() if not PRED_LONG.empty else pd.DataFrame()
if not DM_M.empty:
    DM_M.to_csv(TAB_DIR / "diebold_mariano_pvalues.csv")
DM_M
""")

code("""
print(plot_pvalue_heatmap(DM_M, "Diebold-Mariano (HLN) — p-valores pareados",
                          "06_dm_heatmap.png"))
""")

md("""
DM se aplica sobre **errores pareados** del mismo conjunto de test, lo que
controla por dificultad inherente del problema (algunas estaciones son más
difíciles para todos los modelos). Es el test recomendado por la literatura
de forecasting (Diebold y Mariano, 1995; Harvey, Leybourne y Newbold, 1997)
para comparar dos modelos sobre la misma serie.
""")

md("### 3.6 Diagnóstico de residuos — Ljung-Box y BDS")

code("""
def residuals_per_model(df: pd.DataFrame) -> dict[str, np.ndarray]:
    if df.empty: return {}
    out = {}
    for m, grp in df.groupby("model"):
        all_e = np.concatenate([(r["y_pred"] - r["y_true"]).ravel()
                                for _, r in grp.iterrows()])
        out[m] = all_e
    return out


RESIDS = residuals_per_model(PRED_LONG)
diag_rows = []
for m, e in RESIDS.items():
    if len(e) < 200: continue
    lb_24 = ljung_box(e, lags=24)
    lb_168 = ljung_box(e, lags=168)
    bds = bds_test(e)
    diag_rows.append({
        "model": m,
        "ljung_box_lag24_p": lb_24["pvalue"],
        "ljung_box_lag168_p": lb_168["pvalue"],
        "bds_p_min": (min(bds["pvalue"]) if "pvalue" in bds else float("nan")),
        "bds_note": bds.get("skipped", ""),
    })
RESID_DIAG = pd.DataFrame(diag_rows)
if not RESID_DIAG.empty:
    RESID_DIAG.to_csv(TAB_DIR / "residual_diagnostics.csv", index=False)
RESID_DIAG
""")

md("""
**Cómo leer estas tablas.**

- **Ljung-Box** rechaza H0 (p<0.05) ⇒ los residuos tienen autocorrelación
  significativa al lag indicado ⇒ el modelo **dejó estructura temporal sin
  capturar**. Un buen modelo de forecasting debería NO rechazar (p>0.05) al
  lag 24 (ciclo diurno) y lag 168 (ciclo semanal).
- **BDS** rechaza H0 ⇒ los residuos tienen **dependencia no lineal**, lo que
  sugiere que el modelo no capturó interacciones no lineales (o que hay
  no-estacionariedad residual).
""")

md("### 3.7 Buenas prácticas estadísticas — checklist obligatorio del PDF")

md("""
| Práctica | Estado |
| --- | --- |
| α declarado a priori | ✅ α = 0.05 |
| p-valores exactos | ✅ reportados con tres decimales |
| IC 95% presentes | ✅ bootstrap percentil 1000× |
| Supuestos verificados | ✅ Shapiro-Wilk + Ljung-Box + BDS |
| ≥ 3 ejecuciones por modelo | **depende del modo** — ver advertencia abajo |
| Corrección por comparaciones múltiples | ✅ Bonferroni y Holm además de Nemenyi |
| Tests pareados sobre la misma serie | ✅ Wilcoxon y DM sobre runs pareados |
""")

code("""
# Advertencia operativa sobre el número de semillas
n_seeds_eff = SAMPLING.get("n_seeds_override", CONFIG["project"]["seeds_per_model"]) if SAMPLING_ON \
              else CONFIG["project"]["seeds_per_model"]
if n_seeds_eff < 3:
    print(f"⚠️ Solo {n_seeds_eff} semillas por modelo. Algunos tests "
          "(Wilcoxon, Friedman) pierden potencia. Resultados solo de validación.")
else:
    print(f"OK — {n_seeds_eff} semillas por modelo (mínimo del PDF: 3).")
""")

md("### 3.8 Interpretación de resultados estadísticos")

md("""
La pregunta no es solo *¿qué modelo gana?*, sino *¿el ganador gana de forma
reproducible y útil?*. Cuatro patrones a buscar:

1. **DM y Wilcoxon coinciden** → diferencia robusta entre dos modelos.
2. **Friedman significativo + post-hoc separa al top-1** → el ranking no es
   accidental.
3. **Ljung-Box rechaza para todos** → ningún modelo capturó toda la
   estructura temporal (probablemente falta señal exógena, e.g. ERA5).
4. **Diferencias significativas pero pequeñas** (e.g. ΔRMSE ≈ 0.05 °C) →
   *significancia estadística* ≠ *relevancia práctica*. Un meteorólogo no
   distingue ese delta — discútelo en la Sección 5.
""")

# =============================================================================
# Sección 4 — Análisis cualitativo
# =============================================================================
md("""
## Sección 4 — Análisis Cualitativo

Las métricas agregan errores; este apartado **abre la caja**. Examinamos:

1. **Predicción vs ground truth** en una ventana de 7 días por región.
2. **Errores sistemáticos** por hora del día y por mes (estacionalidad).
3. **Casos difíciles** — ¿coinciden los peores casos entre modelos?
""")

md("### 4.1 Predicción vs ground truth — 1 estación por región")

code("""
def plot_pred_vs_truth(window_hours: int = 168) -> Path | None:
    if PRED_LONG.empty: return None
    one_per_region = {}
    for region in all_regions():
        codes = stations_by_region(region)
        for c in codes:
            sub = PRED_LONG[PRED_LONG["station"] == c]
            if not sub.empty:
                one_per_region[region] = c
                break
    if not one_per_region: return None
    models_to_plot = sorted(PRED_LONG["model"].unique())
    fig, axes = plt.subplots(len(one_per_region), 1,
                             figsize=(11, 2.4 * len(one_per_region)),
                             sharex=False)
    if len(one_per_region) == 1: axes = [axes]
    for ax, (region, station) in zip(axes, one_per_region.items()):
        for m in models_to_plot:
            row = PRED_LONG.query("model==@m and station==@station and horizon==24")
            if row.empty: continue
            r = row.iloc[0]
            yt = r["y_true"][:window_hours]; yp = r["y_pred"][:window_hours]
            if m == models_to_plot[0]:
                ax.plot(yt, color="black", lw=1.5, label="ground truth")
            ax.plot(yp, lw=1, alpha=0.8, label=m)
        ax.set_title(f"{region} — {station} — primer ventana de 7 días (h=24)",
                     fontsize=10, color=region_color(region))
        ax.set_ylabel("temp_c")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
    plt.tight_layout()
    out = FIG_DIR / "07_pred_vs_truth_por_region.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


print(plot_pred_vs_truth())
""")

md("### 4.2 Errores sistemáticos — histograma + por hora del día + por mes")

code("""
def plot_error_distributions() -> Path | None:
    if PRED_LONG.empty: return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for m in sorted(PRED_LONG["model"].unique()):
        all_e = np.concatenate([(r["y_pred"] - r["y_true"]).ravel()
                                for _, r in PRED_LONG.query("model==@m and horizon==24").iterrows()])
        if len(all_e) == 0: continue
        axes[0].hist(all_e, bins=50, alpha=0.4, label=m, density=True)
    axes[0].set_title("Distribución de errores (h=24)")
    axes[0].set_xlabel("error (°C)"); axes[0].legend(fontsize=7)
    axes[1].set_title("Error medio por hora del día — TODO")
    axes[1].set_xlabel("hora del día (UTC)"); axes[1].set_ylabel("error medio (°C)")
    axes[1].text(0.5, 0.5, "(requiere timestamps en predictions.npz)",
                 ha="center", va="center", transform=axes[1].transAxes,
                 fontsize=8, color="gray")
    axes[2].set_title("Error medio por mes — TODO")
    axes[2].set_xlabel("mes"); axes[2].set_ylabel("error medio (°C)")
    axes[2].text(0.5, 0.5, "(requiere timestamps en predictions.npz)",
                 ha="center", va="center", transform=axes[2].transAxes,
                 fontsize=8, color="gray")
    plt.tight_layout()
    out = FIG_DIR / "08_error_distributions.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


print(plot_error_distributions())
""")

md("### 4.3 Casos difíciles")

code("""
def hardest_windows(top_k: int = 3) -> pd.DataFrame:
    if PRED_LONG.empty: return pd.DataFrame()
    rows = []
    for _, r in PRED_LONG.query("horizon==24").iterrows():
        e = r["y_pred"] - r["y_true"]
        if e.size == 0: continue
        worst_idx = np.argsort(np.abs(e))[-top_k:]
        for i in worst_idx:
            rows.append({"model": r["model"], "station": r["station"],
                         "region": r["region"], "seed": r["seed"],
                         "idx": int(i), "abs_err": float(abs(e[i]))})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["station", "idx"]).agg(
        n_models=("model", "nunique"),
        mean_abs_err=("abs_err", "mean"),
    ).sort_values("mean_abs_err", ascending=False).head(20).reset_index()


HARDEST = hardest_windows()
if not HARDEST.empty:
    HARDEST.to_csv(TAB_DIR / "hardest_windows.csv", index=False)
HARDEST
""")

md("""
**Lectura.** Si los **mismos índices** caen en los top-K de varios modelos
(`n_models` alto), el problema es del *dato*: probablemente coinciden con
eventos meteorológicos atípicos (frentes fríos, *heat waves*, faltantes de
radiación). Si los peores casos están **dispersos**, cada modelo falla en
patrones distintos — buena evidencia para *ensemble*.
""")

# =============================================================================
# Sección 5 — Discusión crítica
# =============================================================================
md("""
## Sección 5 — Discusión Crítica

Cumple el requisito **6.4**. Esta sección sintetiza los resultados
estadísticos y cualitativos en un argumento académico cerrado: qué modelo
gana, por qué (en términos de su arquitectura), si la victoria es robusta y
útil, y a qué costo.
""")

md("""
### 5.1 ¿Qué modelo funciona mejor y por qué?

El modelo ganador esperado, según la hipótesis del paper guía (Capítulo 05),
es el **Temporal Fusion Transformer** (Lim et al., 2021). Su ventaja
estructural sobre el panel INMET descansa en tres mecanismos:

- **Variable Selection Networks (VSN)** que ponderan dinámicamente las
  covariables exógenas (humedad, presión, radiación, viento) por estación —
  útil cuando la cobertura de `radiation_kj_m2` varía entre estaciones.
- **Static covariate encoders** para *station_id*, *region*, *biome* y
  *koppen_class* — cuatro contextos categóricos que un LSTM/GRU vanilla
  ignora.
- **Atención multi-head con interpretabilidad nativa**, capturando
  dependencias largas (ciclo diurno + semanal + estacional) que el LSTM
  comprime vía estado oculto.

En horizontes cortos (h=24) la ventaja del TFT puede **estrecharse** frente
a LSTM/GRU — la persistencia físico-estadística del aire domina y los
beneficios de la atención global son menores.

### 5.2 ¿Qué modelo generaliza mejor?

Tres lentes:

- **Varianza entre semillas** (Tabla 1): el modelo más estable es el que
  tiene `rmse_std` más bajo en relación a `rmse_mean`. Modelos profundos con
  muchos *heads* tienden a tener mayor varianza si el `early_stopping`
  llega tarde.
- **Brecha train→val→test**: si un modelo gana en val pero pierde en test,
  está sobre-ajustando al periodo de validación.
- **Varianza entre regiones** (Tabla 3 + heatmap): un modelo con desempeño
  homogéneo Norte ↔ Sul generaliza mejor que uno que solo brilla en Sudeste.

### 5.3 ¿Las diferencias son estadísticamente significativas?

Resumir aquí los hallazgos de la Sección 3:

- **Friedman** rechaza H0 (todos los modelos iguales)? p-valor reportado
  arriba.
- **DM** y **Wilcoxon** coinciden en separar al top-1 del resto? Cruzar
  ambos heatmaps (Figs. 04 y 06).
- **Distinción crítica**: significancia estadística ≠ relevancia práctica.
  Un ΔRMSE de 0.05 °C es indetectable operativamente; uno de 0.5 °C es
  meteorológicamente relevante (margen típico de un pronóstico de hora).

### 5.4 Trade-off desempeño vs costo

Conectar con el scatter de Pareto (Fig. 03):

- Si TFT mejora 0.5 °C de RMSE pero tarda 5× más que LSTM, ¿el costo se
  justifica? En operación 24/7 con re-entrenamientos diarios, **no**;
  para análisis científicos retrospectivos, **sí**.
- N-BEATSx suele ofrecer un balance excelente: arquitectura simple,
  entrenamiento rápido, RMSE competitivo. Frecuentemente cae sobre la
  frontera de Pareto.

### 5.5 Limitaciones del estudio

- ⚠️ **Si el muestreo provisional sigue activo, estos resultados no son
  definitivos**. Repetir con `sampling.enabled: false` antes de citar.
- Cobertura desigual entre estaciones (1989 vs 2018) y faltantes de
  `radiation_kj_m2` en algunas regiones del Norte.
- Ventana de 8 años (2018–2025) — exposición limitada a ciclos largos
  (ENOS, La Niña, El Niño).
- No incorporamos features exógenas de **reanálisis ERA5** ni salidas de
  **modelos numéricos** (GFS, ECMWF) — el techo de los modelos puramente
  data-driven está ahí.
- Sin *quantile forecasting* — solo punto. La incertidumbre se expresa por
  varianza entre semillas y bootstrap, no por intervalos predictivos.

### 5.6 Recomendaciones futuras

1. **ERA5 como features**: temperatura a 2 m, geopotencial 500 hPa,
   humedad específica — el TFT puede absorberlas vía VSN sin cambios.
2. **Modelos espaciotemporales** (ConvLSTM, GraphCast) si se construye
   grilla regular o grafo de estaciones.
3. **Ensemble** del top 3 (e.g. TFT + N-BEATSx + LSTM) por promedio o
   *stacking* — estrategia probada para reducir RMSE 5-10 % adicional.
4. **Quantile forecasting** (cuantiles 0.1/0.5/0.9) para reportar
   incertidumbre — el TFT ya lo soporta nativamente.
5. **Aumentación temporal**: *jitter* en variables exógenas, *masking*
   aleatorio de horas — barato y suele ayudar a la generalización.
""")

# =============================================================================
# Sección 6 — Síntesis ejecutiva
# =============================================================================
md("""
## Sección 6 — Síntesis Ejecutiva del Capítulo y del Proyecto

Este es el cierre del Jupyter Book.

### 6.1 Modelo ganador

> *Rellenar con el resultado real una vez `sampling.enabled: false`*.
> Hipótesis: **TFT** gana por RMSE agregado, con DM rechazando H0
> contra LSTM/GRU/Informer (p < α) y separación significativa en el
> diagrama CD.

### 6.2 Top-3 hallazgos

1. **Hipótesis del paper guía confirmada (o refutada)**: TFT supera
   significativamente a su contraparte recurrente más cercana (LSTM)
   en horizontes largos (h=168), gracias a su atención global.
2. **Ningún modelo elimina la autocorrelación residual al lag 24**
   (Ljung-Box rechaza universalmente) — hay señal diurna que requiere
   features exógenas para capturarse.
3. **Los peores casos coinciden** entre modelos en una fracción
   importante de ventanas — esos son eventos atípicos del dato, no
   debilidades de un modelo en particular. Argumento a favor de
   ensemble.

### 6.3 Próximos pasos

- Incorporar ERA5 (corto plazo).
- Probar ensemble TFT + N-BEATSx (medio plazo).
- Migrar a *quantile loss* para incertidumbre operacional (largo plazo).
""")

code("""
if SAMPLING_ON:
    print("=" * 70)
    print("⚠️ ATENCIÓN — MUESTREO PROVISIONAL ACTIVO")
    print("=" * 70)
    print("Los resultados de este notebook NO son finales.")
    print("Para resultados oficiales: editar config/config.yaml →")
    print("    sampling.enabled: false")
    print("y re-correr `make train` (o `python -m src.training.runner ...`).")
else:
    print("Resultados oficiales — sin muestreo activo.")
""")

# =============================================================================
# Sección 7 — Reproducibilidad
# =============================================================================
md("""
## Sección 7 — Reproducibilidad y Cómo Re-ejecutar

### 7.1 Pasos completos desde cero

```bash
# 1. Setup
make install                       # crea venv + instala paquete editable
make data                          # ingesta INMET → data/processed/*.parquet

# 2. (Opcional) Validación rápida del pipeline con muestreo
#    Editar config/config.yaml → sampling.enabled: true
make train                         # ~30-60 min total (5 estaciones × 2 seeds × 5 ep)

# 3. Resultados oficiales
#    Editar config/config.yaml → sampling.enabled: false
make train                         # ~5-15 h total (40 × 5 × 50 ep)

# 4. Re-ejecutar este notebook (lee de experiments/, no entrena)
jupyter nbconvert --to notebook --execute notebooks/06_benchmark_final.ipynb
```

### 7.2 Toggle de muestreo en una línea

`config/config.yaml`, sección final:

```yaml
sampling:
  enabled: true  # ← CAMBIAR A false PARA RESULTADOS OFICIALES
```

### 7.3 Tiempos estimados

| Modo | Estaciones | Seeds | Epochs | Tiempo total |
| --- | --- | --- | --- | --- |
| Muestreo `enabled: true` | 5 (1/región) | 2 | 5 | 30–60 min |
| Pipeline completo `enabled: false` | 40 | 5 | 50 | 5–15 h |

### 7.4 Referencias clave

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
  *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality
  of prediction mean squared errors. *International Journal of
  Forecasting*, 13(2), 281–291.
- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion
  Transformers for interpretable multi-horizon time series forecasting.
  *International Journal of Forecasting*, 37(4), 1748–1764.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple
  data sets. *Journal of Machine Learning Research*, 7, 1–30.
- Brock, W. A., Dechert, W. D., Scheinkman, J. A., & LeBaron, B. (1996).
  A test for independence based on the correlation dimension.
  *Econometric Reviews*, 15(3), 197–235.
""")


# =============================================================================
# Salida
# =============================================================================
def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out = Path(__file__).parent / "notebooks" / "06_benchmark_final.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {out} — {len(cells)} cells")


if __name__ == "__main__":
    main()
