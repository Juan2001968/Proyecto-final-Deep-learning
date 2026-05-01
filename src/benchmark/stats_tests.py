"""Tests estadísticos para comparar modelos:

- **Diebold-Mariano** (errores pareados sobre la misma serie de test).
- **Friedman + post-hoc** (Nemenyi / Bonferroni / Holm) sobre múltiples
  modelos y datasets/semillas.
- **Wilcoxon signed-rank** sobre errores pareados.
- **Ljung-Box** sobre residuos del mejor modelo.
- **BDS** para no-linealidad residual.

Carga las predicciones desde ``experiments/<model>/<station>/seed=*/predictions.npz``.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.utils import get_logger, load_yaml, save_json

log = get_logger(__name__)

try:  # opcional, requiere statsmodels >=0.14
    from statsmodels.tsa.stattools import bds as _bds
except Exception:                     # pragma: no cover
    _bds = None

try:
    import scikit_posthocs as sp       # post-hoc Nemenyi/Bonferroni/Holm
except Exception:                      # pragma: no cover
    sp = None


# --------------------------------------------------------------- Diebold-Mariano

def diebold_mariano(
    e1: np.ndarray, e2: np.ndarray, *, h: int = 1, power: int = 2,
    alternative: str = "two-sided",
) -> dict:
    """Test DM clásico con corrección de Harvey-Leybourne-Newbold para muestras finitas.

    ``e1``, ``e2`` son errores de pronóstico de los modelos 1 y 2 (pareados).
    """
    e1 = np.asarray(e1).ravel()
    e2 = np.asarray(e2).ravel()
    assert e1.shape == e2.shape and len(e1) > h, "Errores de tamaño incompatible"

    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    mean_d = d.mean()
    # autocovarianzas hasta lag h-1
    gamma = [np.cov(d[: n - k], d[k:], ddof=0)[0, 1] for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return {"stat": float("nan"), "pvalue": float("nan"), "n": int(n), "note": "var<=0"}

    dm = mean_d / np.sqrt(var_d)
    # Harvey, Leybourne, Newbold (1997)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    df = n - 1
    if alternative == "two-sided":
        pval = 2 * (1 - st.t.cdf(abs(dm_hln), df=df))
    elif alternative == "less":
        pval = st.t.cdf(dm_hln, df=df)
    else:
        pval = 1 - st.t.cdf(dm_hln, df=df)
    return {"stat": float(dm_hln), "pvalue": float(pval), "n": int(n), "h": int(h), "power": int(power)}


# --------------------------------------------------------------- Friedman + post-hoc

def friedman_with_posthoc(error_table: pd.DataFrame, posthoc: str = "nemenyi") -> dict:
    """``error_table``: filas = datasets/semillas, columnas = modelos, valores = error medio."""
    arrs = [error_table[c].to_numpy() for c in error_table.columns]
    stat, pval = st.friedmanchisquare(*arrs)
    out: dict = {"friedman": {"stat": float(stat), "pvalue": float(pval),
                              "n_datasets": int(error_table.shape[0]),
                              "n_models": int(error_table.shape[1])}}
    if sp is None:
        out["posthoc_warning"] = "scikit_posthocs no instalado — sólo Friedman."
        return out

    if posthoc == "nemenyi":
        ph = sp.posthoc_nemenyi_friedman(error_table)
    elif posthoc == "bonferroni":
        ph = sp.posthoc_conover_friedman(error_table, p_adjust="bonferroni")
    elif posthoc == "holm":
        ph = sp.posthoc_conover_friedman(error_table, p_adjust="holm")
    else:
        raise ValueError(posthoc)
    out["posthoc"] = {"method": posthoc, "pvalues": ph.to_dict()}
    return out


# --------------------------------------------------------------- Wilcoxon

def wilcoxon_pairs(error_table: pd.DataFrame, *, alpha: float = 0.05) -> dict:
    pairs = {}
    for a, b in combinations(error_table.columns, 2):
        try:
            res = st.wilcoxon(error_table[a], error_table[b], zero_method="wilcox", correction=False)
            pairs[f"{a} vs {b}"] = {"stat": float(res.statistic), "pvalue": float(res.pvalue),
                                    "significant": bool(res.pvalue < alpha)}
        except ValueError as exc:
            pairs[f"{a} vs {b}"] = {"error": str(exc)}
    return pairs


# --------------------------------------------------------------- Residuals

def ljung_box(residuals: np.ndarray, lags: int = 24) -> dict:
    res = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return {"lag": lags, "stat": float(res["lb_stat"].iloc[0]), "pvalue": float(res["lb_pvalue"].iloc[0])}


def bds_test(residuals: np.ndarray, *, max_dim: int = 3) -> dict:
    """BDS (Brock-Dechert-Scheinkman) para no-linealidad residual.

    Si statsmodels no expone ``bds`` se devuelve un payload con ``skipped``.
    """
    if _bds is None:
        return {"skipped": "bds no disponible en esta versión de statsmodels"}
    residuals = np.asarray(residuals).ravel()
    residuals = residuals[~np.isnan(residuals)]
    if residuals.std() == 0 or len(residuals) < 50:
        return {"skipped": "muestra demasiado corta o varianza nula"}
    stat, pval = _bds(residuals, max_dim=max_dim)
    return {"stat": [float(x) for x in np.atleast_1d(stat)],
            "pvalue": [float(x) for x in np.atleast_1d(pval)],
            "max_dim": int(max_dim)}


# --------------------------------------------------------------- Bootstrap CI

def bootstrap_ci(values: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05,
                 stat_fn=np.mean, seed: int = 42) -> dict:
    """IC ``(1-alpha)*100%`` por bootstrap percentil.

    ``stat_fn`` permite calcular IC para media, RMSE (``lambda e: sqrt(mean(e**2))``), etc.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values).ravel()
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(values[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": float(stat_fn(values)), "lo": lo, "hi": hi,
            "alpha": alpha, "n_boot": int(n_boot)}


# --------------------------------------------------------------- Critical-Difference

def nemenyi_critical_difference(n_models: int, n_datasets: int,
                                 alpha: float = 0.05) -> float:
    """Distancia crítica de Nemenyi para diagrama CD.

    Usa la tabla de q-alpha para alpha=0.05 hasta 10 modelos. Para más
    modelos cae a aproximación.
    """
    q_alpha_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q = q_alpha_05.get(n_models, 3.164 + 0.05 * (n_models - 10))
    return float(q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets)))


def average_ranks(error_table: pd.DataFrame) -> pd.Series:
    """Rangos promedio por modelo (1 = mejor) — útil para diagrama CD."""
    ranks = error_table.rank(axis=1, method="average", ascending=True)
    return ranks.mean(axis=0).sort_values()


def shapiro_per_model(error_table: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Test de Shapiro-Wilk por modelo. Devuelve tabla con ``p_value`` y ``normal``."""
    rows = []
    for col in error_table.columns:
        x = error_table[col].dropna().to_numpy()
        if len(x) < 3:
            rows.append({"model": col, "p_value": float("nan"), "normal": False,
                         "note": "n<3"})
            continue
        try:
            stat, p = st.shapiro(x)
            rows.append({"model": col, "stat": float(stat), "p_value": float(p),
                         "normal": bool(p > alpha)})
        except Exception as exc:
            rows.append({"model": col, "p_value": float("nan"), "normal": False,
                         "note": str(exc)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- Driver

def _gather_errors(experiments_root: Path) -> pd.DataFrame:
    """Tabla MEAN_ABS_ERROR por (modelo, station, seed) → filas: una por test sample.
    Cada celda es un vector de errores aplanado del set de test.
    """
    table: dict[tuple[str, str, int], np.ndarray] = {}
    for npz_path in experiments_root.rglob("predictions.npz"):
        seed_dir, station_dir, model_dir = npz_path.parents[0:3]
        seed = int(seed_dir.name.split("=")[1])
        d = np.load(npz_path, allow_pickle=True)
        err = (d["y_pred"] - d["y_true"]).reshape(-1)
        table[(model_dir.name, station_dir.name, seed)] = err
    return table


def run(config: dict) -> None:
    out_dir = Path(config["paths"]["stats"])
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_root = Path(config["paths"]["experiments"])

    errors = _gather_errors(exp_root)
    if not errors:
        log.warning("No hay predicciones — corre el benchmark antes."); return

    # ----------- DM por pares de modelos en la misma estación + seed
    dm_results = []
    keys = list(errors.keys())
    for (m1, st1, s1), (m2, st2, s2) in combinations(keys, 2):
        if m1 == m2 or st1 != st2 or s1 != s2:
            continue
        cfg_dm = config["stats"]["diebold_mariano"]
        try:
            res = diebold_mariano(errors[(m1, st1, s1)], errors[(m2, st2, s2)],
                                  h=cfg_dm["h"], power=cfg_dm["power"],
                                  alternative=cfg_dm["alternative"])
            dm_results.append({"model_a": m1, "model_b": m2, "station": st1, "seed": s1, **res})
        except AssertionError:
            continue
    pd.DataFrame(dm_results).to_csv(out_dir / "diebold_mariano.csv", index=False)

    # ----------- Friedman: tabla mean_error por (station, seed) × modelo
    rows = {}
    for (m, st_, s), err in errors.items():
        rows.setdefault((st_, s), {})[m] = float(np.mean(err ** 2))   # MSE como score
    error_df = pd.DataFrame(rows).T.dropna(axis=0, how="any")
    if error_df.shape[1] >= 3 and error_df.shape[0] >= 3:
        fried = friedman_with_posthoc(error_df, config["stats"]["friedman_posthoc"])
        save_json(fried, out_dir / "friedman.json")
        wilc = wilcoxon_pairs(error_df, alpha=config["stats"]["wilcoxon_alpha"])
        save_json(wilc, out_dir / "wilcoxon.json")
    else:
        log.warning("Insuficientes (modelos, datasets) para Friedman/Wilcoxon")

    # ----------- Ljung-Box / BDS sobre los residuos del mejor (RMSE más bajo)
    rmse_per_run = {k: float(np.sqrt(np.mean(v ** 2))) for k, v in errors.items()}
    best_key = min(rmse_per_run, key=rmse_per_run.get)
    save_json(
        {
            "best_run": {"model": best_key[0], "station": best_key[1], "seed": best_key[2],
                         "rmse": rmse_per_run[best_key]},
            "ljung_box": ljung_box(errors[best_key]),
            "bds": bds_test(errors[best_key]),
        },
        out_dir / "residual_diagnostics.json",
    )
    log.info("Tests estadísticos guardados en %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
