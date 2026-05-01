"""Construye notebooks/04_benchmark_models.ipynb (sin ejecutar)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ============================================================================
# Celda 0
# ============================================================================
md("""
# Implementación de Modelos Benchmark para Forecasting Multi-horizonte de `temp_c`

**Resumen ejecutivo.** Este capítulo materializa el benchmark del proyecto a
partir del subset de **6 modelos** seleccionado en el SOTA (Capítulo 03):
*Persistencia*, *LSTM vanilla*, *GRU*, *N-BEATSx*, *Temporal Fusion Transformer*
e *Informer*. Documenta el **diseño experimental** (variables, split temporal,
pipeline de preprocesamiento, configuración), **fija una arquitectura por
modelo** con sus hiperparámetros base, declara la **función de pérdida** y la
**estrategia de entrenamiento** que cada modelo utilizará, formaliza el
**control de semillas** (5 corridas por modelo) y la **trazabilidad** (entorno,
hash dataset, commit), audita las **garantías anti-leakage** ya validadas por
los tests del proyecto, y deja preparados los **helpers de evaluación
preliminar** (curvas, predicción vs ground truth, tabla agregada de métricas).
**No ejecuta entrenamientos**: el peso computacional se delega a
`python -m src.training.runner` corrido en terminal.
""")

# ============================================================================
# Sección 1 — Diseño Experimental
# ============================================================================
md("## 1. Diseño Experimental")

md("### 1.1 Variables de entrada y objetivo")

code("""
import sys
import os
from pathlib import Path

# Resolución robusta de la raíz del repo
REPO_ROOT = Path.cwd().resolve()
while not (REPO_ROOT / "config").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import pandas as pd

variables = pd.DataFrame([
    # Target
    {"nombre": "temp_c", "descripcion": "Temperatura del aire (target)",
     "unidad": "°C", "fuente": "INMET", "incluida": "TARGET",
     "justificacion": "Variable objetivo del proyecto."},
    # Exógenas dinámicas (selección del EDA, sec. 6)
    {"nombre": "humidity_pct", "descripcion": "Humedad relativa",
     "unidad": "%", "fuente": "INMET", "incluida": "sí (input)",
     "justificacion": "|corr|≈0.6 con temp_c; MI alta (EDA §6)."},
    {"nombre": "pressure_mb", "descripcion": "Presión atmosférica al nivel de estación",
     "unidad": "mb", "fuente": "INMET", "incluida": "sí (input)",
     "justificacion": "Correlación moderada y CCF informativa con lead 1–3 h."},
    {"nombre": "radiation_kj_m2", "descripcion": "Radiación global",
     "unidad": "kJ/m²", "fuente": "INMET", "incluida": "sí (input, con cuidado)",
     "justificacion": "CCF muestra que adelanta a temp_c 2–4 h. ~21 % NaN nocturnos legítimos — el modelo descarta ventanas con NaN remanente, no se imputa con 0."},
    {"nombre": "wind_speed_ms", "descripcion": "Velocidad del viento",
     "unidad": "m/s", "fuente": "INMET", "incluida": "sí (input)",
     "justificacion": "MI moderada con temp_c; capta enfriamiento por advección."},
    {"nombre": "dew_point_c", "descripcion": "Punto de rocío",
     "unidad": "°C", "fuente": "INMET", "incluida": "sí (input)",
     "justificacion": "Alta correlación con humidity (multicolinealidad aceptada en redes profundas)."},
    {"nombre": "precip_mm", "descripcion": "Precipitación horaria",
     "unidad": "mm", "fuente": "INMET", "incluida": "no",
     "justificacion": "Correlación ≈ 0 con temp_c; MI baja (EDA §6) — descartada."},
    {"nombre": "wind_dir_deg", "descripcion": "Dirección del viento (cruda)",
     "unidad": "°", "fuente": "INMET", "incluida": "no (cruda)",
     "justificacion": "No lineal en grados; entra vía wind_dir_sin/cos en features.py."},
    # Cíclicas
    {"nombre": "hour_sin / hour_cos", "descripcion": "Codificación cíclica de la hora",
     "unidad": "—", "fuente": "derivada (process.py)", "incluida": "sí (known future)",
     "justificacion": "FFT del EDA confirma pico dominante a 24 h."},
    {"nombre": "doy_sin / doy_cos", "descripcion": "Codificación cíclica del día del año",
     "unidad": "—", "fuente": "derivada (process.py)", "incluida": "sí (known future)",
     "justificacion": "FFT del EDA confirma pico anual ~8766 h; STL ~50–80 % varianza estacional."},
    {"nombre": "month_sin / month_cos", "descripcion": "Codificación cíclica del mes",
     "unidad": "—", "fuente": "derivada (process.py)", "incluida": "sí (known future)",
     "justificacion": "Granularidad mensual complementaria; útil para desambiguar tendencia anual y anomalías estacionales."},
    # Estáticas (metadata por estación)
    {"nombre": "station_id", "descripcion": "ID entero por estación (40 niveles)",
     "unidad": "—", "fuente": "process.py + stations.yaml", "incluida": "sí (static categórica)",
     "justificacion": "Embedding por estación absorbe el nivel base climático local."},
    {"nombre": "region", "descripcion": "Macrorregión IBGE (5 niveles)",
     "unidad": "—", "fuente": "stations.yaml", "incluida": "sí (static categórica)",
     "justificacion": "Captura heterogeneidad regional dramática observada en EDA §7."},
    {"nombre": "biome", "descripcion": "Bioma (6 niveles)",
     "unidad": "—", "fuente": "stations.yaml", "incluida": "sí (static categórica)",
     "justificacion": "Cardinalidad baja, alta información (Caatinga ≠ Pampa)."},
    {"nombre": "koppen_class", "descripcion": "Clase Köppen-Geiger (9 niveles)",
     "unidad": "—", "fuente": "stations.yaml", "incluida": "sí (static categórica)",
     "justificacion": "Granularidad climática más fina que region/biome."},
    {"nombre": "latitude", "descripcion": "Latitud de la estación",
     "unidad": "°", "fuente": "metadata.json", "incluida": "sí (static real)",
     "justificacion": "Información geográfica continua; correlación lineal moderada con temp_c (Pearson)."},
    {"nombre": "longitude", "descripcion": "Longitud de la estación",
     "unidad": "°", "fuente": "metadata.json", "incluida": "sí (static real, normalizada)",
     "justificacion": "Información geográfica continua; entra como feature estática estandarizada."},
    {"nombre": "altitude", "descripcion": "Altitud de la estación",
     "unidad": "m", "fuente": "metadata.json", "incluida": "sí (static real)",
     "justificacion": "Modula directamente el régimen térmico (gradiente vertical)."},
])
variables
""")

md("""
**Target multi-horizonte.** El proyecto predice `temp_c` a tres horizontes
simultáneos: **+24 h, +72 h, +168 h**. Estrategia: el modelo emite un vector de
**168 pasos** y la evaluación se reporta por separado para cuts en {24, 72, 168}.
Esto evita entrenar tres modelos distintos y permite a los Transformers
aprovechar la dependencia temporal larga.

> ⚠️ **Estado actual del repo**: `config/config.yaml` tiene `task.horizon: 24`.
> Para activar el régimen multi-horizonte completo del benchmark, **cambiar a
> `horizon: 168`** y reportar las métricas por slice (h=24, h=72, h=168) en la
> celda de carga de resultados (Sección 5.5).
""")

md("### 1.2 Partición del dataset")

md("""
| Split | Años | Origen del filtro | # estaciones | Filas/estación (aprox.) |
|---|---|---|---|---|
| **Train** | 2018, 2019, 2020, 2021, 2022, 2023 | `cfg.split.by_year.train_years` | 40 | ~52 584 |
| **Val** | 2024 | `cfg.split.by_year.val_years` | 39 (A301 sin 2024) | ~8 784 |
| **Test** | 2025 | `cfg.split.by_year.test_years` | 38 (A301 y A615 sin 2025) | ~8 760 |

**Garantía formal de no-leakage**:

```text
max(train.index) = 2023-12-31 23:00:00  <  2024-01-01 00:00:00 = min(val.index)
max(val.index)   = 2024-12-31 23:00:00  <  2025-01-01 00:00:00 = min(test.index)
```

Validado automáticamente por `tests/test_split_real_data.py::test_no_leakage_on_real_data`
sobre `data/processed/A001.parquet`.

> 🛡️ **Bloque anti-leakage del entrenamiento**
>
> 1. **Scaler fit** sólo en `train` — `FeatureScaler.fit(..., source="train")`
>    en `src/data/scalers.py`, validado por `test_scaler_fit_train_only.py`.
> 2. **Ventaneo** no cruza fronteras de split — `make_windows` genera ventanas
>    completamente contenidas en cada split, validado por
>    `test_windowing_no_leakage.py`.
> 3. **Imputación causal** — `process.py` usa `ffill(limit=6)`; ventanas con
>    NaN remanente se descartan en `make_windows`, no se rellenan con la media
>    (que sería leakage de estadísticos globales).
""")

md("### 1.3 Pipeline de preprocesamiento")

md("""
```text
data/processed/<wmo>.parquet                 (ya escrito por src.data.process)
        │
        ▼
[carga panel] load_parquet(...) por estación
        │
        ▼
[split temporal por años]   train (2018–2023)  |  val (2024)  |  test (2025)
        │                                                   │
        ▼                                                   ▼
[FeatureScaler.fit en train]  ────────────────────► [transform val/test]
        │
        ▼
[make_windows]   X (B, lookback=168, n_features),  y (B, horizon=168, n_targets)
        │
        ▼
[DataLoader]   batch=64, shuffle=True (train), False (val/test)
        │
        ▼
[runner.py]   instancia modelo y entrena con Trainer
```

La función `_resolve_feature_cols` del runner devuelve el conjunto de
features de entrada (todo numérico excepto los targets, más los targets
listados como exógenos si el modo es multitarget). El target sale del
config (`task.target = "temp_c"`).
""")

code("""
# Demo: instanciación del pipeline (NO se ejecuta entrenamiento).
# Para correr realmente, usar `python -m src.training.runner --model <nombre>`.

from src.utils import load_yaml
from src.data.windowing import make_windows
from src.data.scalers import FeatureScaler

cfg = load_yaml("config/config.yaml")
print("Target :", cfg["task"]["target"])
print("Exog   :", cfg["task"]["exog"])
print("Lookback:", cfg["task"]["lookback"], "h")
print("Horizon :", cfg["task"]["horizon"], "h  (cambiar a 168 para multi-horizonte)")
print("Freq   :", cfg["task"]["freq"])

# Demostración de la API de ventaneo (NO ejecuta para evitar carga pesada).
if False:
    import pandas as pd
    df_train = pd.read_parquet("data/processed/A001.parquet")
    df_train = df_train[df_train.index.year.isin(cfg["split"]["by_year"]["train_years"])]
    feats = ["humidity_pct", "pressure_mb", "radiation_kj_m2", "wind_speed_ms", "dew_point_c"]
    tgts = [cfg["task"]["target"]]
    w = make_windows(df_train, feats, tgts, cfg["task"]["lookback"], cfg["task"]["horizon"])
    print("X shape:", w.X.shape, "y shape:", w.y.shape)
    fx = FeatureScaler(name=cfg["scaling"]["method"]).fit(w.X, source="train")
    Xtr = fx.transform(w.X)
""")

md("### 1.4 Configuración experimental base")

code("""
# Lectura de la config global y de los configs por modelo.
import yaml
from pathlib import Path

cfg_global = load_yaml("config/config.yaml")
configs_dir = Path("config/models")
model_configs = {p.stem: load_yaml(p) for p in sorted(configs_dir.glob("*.yaml"))}

print("Modelos con config:", sorted(model_configs.keys()))
print()
print("Hiperparámetros globales (cfg.training):")
for k, v in cfg_global["training"].items():
    print(f"  {k:25s} = {v}")
print()
print("Semillas: base =", cfg_global["project"]["seed"],
      "| n_runs =", cfg_global["project"]["seeds_per_model"])
""")

code("""
# Tabla resumen de configuración base por modelo.
rows = []
for name, mc in model_configs.items():
    tr = mc.get("training", {})
    rows.append({
        "modelo": name,
        "class": mc["model"]["class"].split(".")[-1],
        "batch_size": tr.get("batch_size"),
        "epochs": tr.get("epochs"),
        "lr": tr.get("lr"),
        "optimizer": tr.get("optimizer"),
        "weight_decay": tr.get("weight_decay"),
        "patience": tr.get("early_stopping_patience"),
        "grad_clip": tr.get("grad_clip"),
    })
pd.DataFrame(rows).set_index("modelo")
""")

md("""
**Decisiones del diseño experimental que mitigan riesgos identificados en el EDA**

- **Split temporal estricto** (no CV aleatorio) — mitiga el riesgo principal
  para forecasting: leakage temporal. Validado por `tests/`.
- **Lookback = 168 h** justificado por la ACF significativa hasta lag 168
  (EDA §5.4) — captura ciclo diario y dependencia semanal débil.
- **Estandarización por estación** (cuando el runner lo soporte vía panel
  global) absorbe la heterogeneidad regional dramática (Norte plano vs Sul
  amplio) detectada en EDA §7.
- **Quantile loss / Huber** para los modelos compatibles (TFT) — mitiga la
  sub-predicción de colas en eventos extremos (`p01/p99` del EDA §2).
- **`radiation_kj_m2` se incluye** como exógena pero las ventanas con NaN
  remanente se descartan: alternativa más segura que imputar con 0 (que sería
  ruido informativo) o con la media (que sería leakage de estadísticos
  globales).
""")

# ============================================================================
# Sección 2 — Implementación de Modelos
# ============================================================================
md("## 2. Implementación de Modelos")

md("""
A continuación se documenta la implementación de los **6 modelos** del
benchmark. Por cada modelo: justificación, arquitectura, hiperparámetros,
loss, estrategia de entrenamiento, comando de ejecución y outputs esperados.

> **Convención de outputs (común a todos)**: el runner produce, por cada
> `(modelo, estación, semilla)`, un directorio
> `experiments/<model>/<station>/seed=<s>/` con: `checkpoint.pt`,
> `history.json`, `predictions.npz`, `metrics.json`, `env.json`,
> `config_used.yaml`, `scaler_x.joblib`, `scaler_y.joblib`. La ausencia de
> `residuals.npz` se compensa porque las predicciones se guardan en
> `predictions.npz` con `y_pred` y `y_true`, de los cuales el residuo se
> deriva en notebook 06.
""")

# 2.1 Persistencia
md("""
### 2.1 Persistencia (naive baseline)

#### Justificación

Sanity check obligatorio del benchmark. Predice `ŷ_{t+h} = y_t` para todo `h`.
El test estadístico (Diebold-Mariano, ver notebook 06) **debe rechazar la
hipótesis nula a favor de cualquier modelo DL**; si no lo hace, el dataset es
trivial o el modelo no aprendió. Es la línea base referida implícitamente en
toda la familia de papers del SOTA (Khan & Maity, 2020; Suleman & Shridevi,
2022).

#### Arquitectura

```
Input X (B, lookback=168, n_features)
  └─ Selecciona la columna del target en el último paso (t)
  └─ Repite el escalar a lo largo del horizon
  └─ Output ŷ (B, horizon, 1)
```

Sin parámetros entrenables. Implementación:
`src/models/model_persistence.py::PersistenceForecaster` (skeleton).

#### Hiperparámetros

| hiperparámetro | valor base | rango | justificación |
|---|---|---|---|
| `epochs` | 1 | — | No hay entrenamiento; una pasada de evaluación. |
| `batch_size` | 64 | — | Mismo que los demás para reusar `DataLoader` y métricas. |

#### Función de pérdida

**MSE** (sólo para reportar la curva de validación; no afecta a los pesos
porque no hay parámetros entrenables).

#### Estrategia de entrenamiento

No aplica. El runner instancia el modelo y va directamente a evaluación sobre
val y test.

#### Comando para entrenar

```bash
python -m src.training.runner --model persistence --seeds 5
```

#### Outputs esperados

`experiments/persistence/<station>/seed={42..46}/` con `metrics.json`,
`predictions.npz` y `env.json`. Como referencia, `checkpoint.pt` será un
artefacto vacío (sin pesos).
""")

# 2.2 LSTM
md("""
### 2.2 LSTM vanilla

#### Justificación

Baseline de la **familia recurrente**. Conecta directamente con el SFA-LSTM de
**Suleman & Shridevi (2022)**, IEEE Access — el paper de la revisión inicial
que aplica LSTM con atención a temperatura. Implementamos primero la versión
*sin atención* para cuantificar la ganancia marginal de los mecanismos
avanzados (atención sobre variables, atención multi-head) en modelos
posteriores.

Referencia: Suleman, M. A. R., & Shridevi, S. (2022). *Short-Term Weather
Forecasting Using Spatial Feature Attention Based LSTM Model.* IEEE Access, 10,
82456–82468. https://doi.org/10.1109/ACCESS.2022.3196381

#### Arquitectura

```
Input X (B, lookback=168, n_features=12+)
  └─ LSTM(hidden=128, layers=2, dropout=0.2, bidirectional=False)
  └─ Toma el último estado oculto: out[:, -1, :]
  └─ Linear(128 → horizon × n_targets)
  └─ Reshape → Output ŷ (B, horizon=168, n_targets=1)
```

Implementación: `src/models/model_lstm.py::LSTMForecaster`. **Parámetros
aproximados**: ~10⁵ (≈ 137 K para hidden=128, layers=2, n_features=12,
horizon=168).

#### Hiperparámetros

| hiperparámetro | valor base | rango | justificación |
|---|---|---|---|
| `hidden_size` | 128 | {64, 128, 256} | Balance capacidad/sobreajuste; el panel ~2.1 M ejemplos tolera 128. |
| `num_layers` | 2 | {1, 2, 3} | 2 capas para profundidad sin gradient vanishing en lookback=168. |
| `dropout` | 0.2 | {0.0, 0.2, 0.3} | Regularización moderada (Suleman & Shridevi usan 0.2). |
| `bidirectional` | False | — | Causalidad estricta: ningún modelo puede mirar al futuro. |
| `lr` | 1e-3 | {5e-4, 1e-3, 5e-3} | Estándar Adam; warmup no necesario en LSTM. |
| `batch_size` | 64 | {32, 64, 128} | Heurística para el volumen del panel. |
| `epochs` (máx.) | 50 | — | El early stopping decide. |
| `early_stopping_patience` | 8 | — | Suficiente para detectar plateau sin sobreajustar. |

#### Función de pérdida

**MSE** sobre el output (predicción puntual). Justificación: baseline
canónico; comparable directamente con métricas RMSE/MAE de la literatura.

> Nota: el EDA recomendó **Huber** para robustez a outliers en colas. Como
> baseline conservador usamos MSE; en una iteración posterior se puede
> contrastar con Huber (`loss_fn=nn.HuberLoss(delta=1.0)` en el `Trainer`).

#### Estrategia de entrenamiento

- **Optimizer**: Adam (config `cfg.training.optimizer`).
- **Weight decay**: 1e-5.
- **Scheduler**: ninguno por defecto (LSTM no suele necesitar; opcional `ReduceLROnPlateau` con factor 0.5 si se observa plateau largo).
- **Gradient clipping**: 1.0 (estándar para evitar explosión en LSTMs).
- **Mixed precision**: no por defecto. Se puede habilitar con `torch.cuda.amp.autocast` en GPU si ese hardware está disponible.

#### Comando para entrenar

```bash
python -m src.training.runner --model lstm --config config/config.yaml --seeds 5
```

#### Outputs esperados

`experiments/lstm/<station>/seed={42..46}/` con `checkpoint.pt`,
`history.json`, `predictions.npz`, `metrics.json`, `env.json`,
`config_used.yaml`, `scaler_{x,y}.joblib`.
""")

# 2.3 GRU
md("""
### 2.3 GRU

#### Justificación

Variante más **liviana** del LSTM (un gate menos). Comparación interna de la
familia recurrente: ¿el costo extra del LSTM compensa frente al GRU? Si la
diferencia es marginal, GRU es preferible por velocidad de entrenamiento e
inferencia.

Referencia conceptual: Cho, K., et al. (2014). *Learning phrase representations
using RNN encoder-decoder for statistical machine translation.* EMNLP 2014.
arXiv:1406.1078.

#### Arquitectura

Idéntica a LSTM excepto por la celda recurrente (GRU en lugar de LSTM):

```
Input X (B, 168, n_features)
  └─ GRU(hidden=128, layers=2, dropout=0.2)
  └─ Linear(128 → horizon × n_targets)
  └─ Output ŷ (B, 168, 1)
```

Implementación: `src/models/model_gru.py::GRUForecaster`. **Parámetros
aproximados**: ~10⁵ (≈ 100 K para hidden=128, ~25 % menos que LSTM equivalente).

#### Hiperparámetros

Idénticos a LSTM (Sec. 2.2) salvo el `hidden_size` que mantenemos en 128 para
comparación justa, sabiendo que GRU rinde competitivamente con menos
parámetros.

#### Función de pérdida

**MSE**, idéntica a LSTM para comparabilidad directa.

#### Estrategia de entrenamiento

Idéntica a LSTM (Sec. 2.2): Adam, weight_decay 1e-5, grad_clip 1.0, sin
scheduler base, early stopping patience 8.

#### Comando para entrenar

```bash
python -m src.training.runner --model gru --config config/config.yaml --seeds 5
```

#### Outputs esperados

`experiments/gru/<station>/seed={42..46}/` con la suite estándar.
""")

# 2.4 N-BEATSx
md("""
### 2.4 N-BEATSx (extensión multivariada de N-BEATS)

#### Justificación

Baseline **fuerte de forecasting puro no recurrente, no Transformer**.
Reproduce explícitamente la descomposición *tendencia + estacionalidad* que el
EDA §5.2 (STL) confirmó como dominante (50–80 % de varianza por región). N-BEATS
ganó la M4 Competition contra 60+ métodos. La extensión `N-BEATSx` añade
covariables exógenas, requeridas por nuestro régimen multivariado.

Referencia: Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020).
*N-BEATS: Neural basis expansion analysis for interpretable time series
forecasting.* ICLR 2020. arXiv:1905.10437.

#### Arquitectura

```
Input X (B, 168, n_features)
  └─ Bloque₁ (MLP fully-connected, layer_width=256)
        ├─ backcast₁ (basis polynomial trend, deg=3)
        └─ forecast₁
  └─ Residuo: input - backcast₁
  └─ Bloque₂ (MLP, basis Fourier seasonality)
        ├─ backcast₂
        └─ forecast₂
  └─ ... (3 bloques por stack × 3 stacks {trend, seasonality, generic})
  └─ Σ forecasts → ŷ (B, horizon=168, n_targets=1)
```

Implementación: `src/models/model_nbeats.py::NBEATSForecaster`. **Parámetros
aproximados**: ~5×10⁵–10⁶ dependiendo de `layer_width` y `num_blocks_per_stack`.

#### Hiperparámetros

| hiperparámetro | valor base | rango | justificación |
|---|---|---|---|
| `stack_types` | `[trend, seasonality, generic]` | — | Versión Interpretable: la descomposición casa con STL del EDA. |
| `num_blocks_per_stack` | 3 | {2, 3, 4} | Recomendación del paper original. |
| `num_layers` | 4 | {3, 4, 5} | Profundidad del MLP por bloque. |
| `layer_width` | 256 | {128, 256, 512} | Capacidad del bloque; 256 es el default robusto. |
| `expansion_coefficient_dim` | 5 | {3, 5, 10} | Tamaño de la base latente. |
| `trend_polynomial_degree` | 3 | {2, 3, 4} | Polinomio cúbico para tendencia. |
| `lr` | 1e-3 | — | Adam estándar. |
| `weight_decay` | 0.0 | — | El paper sugiere desactivar weight_decay. |
| `epochs` (máx.) | 50 | — | Early stopping decide. |
| `early_stopping_patience` | 8 | — | Igual al resto. |

#### Función de pérdida

**MSE** estándar. La versión Interpretable de N-BEATS también es compatible
con sMAPE/MASE (escala-libres) que veremos en la M-tabla del benchmark.

#### Estrategia de entrenamiento

- Adam, lr=1e-3, sin weight_decay, sin scheduler.
- `grad_clip=1.0`.
- Early stopping patience 8.

#### Comando para entrenar

```bash
python -m src.training.runner --model nbeats --config config/config.yaml --seeds 5
```

#### Outputs esperados

`experiments/nbeats/<station>/seed={42..46}/` con la suite estándar. La
versión Interpretable produce además **descomposiciones por bloque** que
podemos visualizar en notebook 05 (paper guía y análisis interpretativo).
""")

# 2.5 TFT
md("""
### 2.5 Temporal Fusion Transformer (TFT) — paper guía

#### Justificación

**Paper guía** del proyecto (SOTA §6). Encaja con todas las decisiones del
EDA: multi-horizonte nativo, embeddings de entidad como ciudadanos de primera
clase, covariables conocidas a futuro (cíclicas), interpretabilidad vía VSN +
atención, y quantile loss para colas (eventos extremos).

Referencia: Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal
Fusion Transformers for interpretable multi-horizon time series forecasting.*
International Journal of Forecasting, 37(4), 1748–1764.
https://doi.org/10.1016/j.ijforecast.2021.03.012

> ⚠️ **Estado del skeleton**: `src/models/model_tft.py::TFTForecaster` existe
> como esqueleto con `raise NotImplementedError`. La implementación se difiere
> a una entrega posterior. Se recomienda apoyarse en `pytorch-forecasting`
> como referencia.

#### Arquitectura

```
[Static covariates] (station_id, region, biome, koppen, lat, lng, alt)
        │
        └─► Static encoders (4 contextos: c_s, c_e, c_c, c_h)
                │
[Past observed]   ── VSN (gated) ─────┐
[Past known]      ── VSN (gated) ─────┼──► LSTM encoder (h=64)
[Future known]    ── VSN (gated) ─────┘                         │
                                                                  ▼
                                                Multi-head self-attention (heads=4)
                                                                  │
                                                                  ▼
                                                Gated residual + Quantile heads
                                                                  │
                                                                  ▼
                                          ŷ (B, horizon=168, |quantiles|=3) → P10/P50/P90
```

**Parámetros aproximados**: ~10⁶ (depende de `hidden_size` y de la cardinalidad
de los embeddings).

#### Hiperparámetros

| hiperparámetro | valor base | rango | justificación |
|---|---|---|---|
| `hidden_size` | 64 | {32, 64, 128} | Default del paper. |
| `attention_heads` | 4 | {2, 4, 8} | 4 heads cubren bien dependencias diaria/semanal/anual. |
| `dropout` | 0.1 | {0.1, 0.2, 0.3} | Default del paper. |
| `n_static_categorical` | 4 | — | station_id, region, biome, koppen. |
| `n_static_real` | 3 | — | latitude, longitude, altitude. |
| `quantiles` | [0.1, 0.5, 0.9] | — | P10/P50/P90 → bandas P10–P90 = ~80 % CI. |
| `lr` | 1e-3 | {5e-4, 1e-3, 3e-3} | Adam, default del paper. |
| `weight_decay` | 1e-4 | — | Mayor que LSTM por densidad de parámetros. |
| `epochs` (máx.) | 60 | — | Más que LSTM por tamaño del modelo. |
| `early_stopping_patience` | 10 | — | Más que LSTM por mayor variabilidad de val_loss. |

#### Función de pérdida

**Quantile loss multi-percentil** (P10, P50, P90):

```
QL(y, q̂) = max(q · (y − q̂), (q−1) · (y − q̂))
loss = Σ_q QL_q(y, q̂_q)
```

Justificación EDA: las colas (`p01`, `p99`) son ~2 % del volumen pero contienen
los eventos críticos. La quantile loss **no sub-predice** las colas (a
diferencia del MSE que minimiza la suma de cuadrados).

#### Estrategia de entrenamiento

- **Optimizer**: Adam.
- **Weight decay**: 1e-4.
- **Scheduler**: opcional `ReduceLROnPlateau` factor 0.5 patience 5.
- **Gradient clipping**: 1.0.
- **Mixed precision**: recomendado en GPU (`torch.cuda.amp.autocast`) por la
  cantidad de parámetros.

#### Comando para entrenar

```bash
python -m src.training.runner --model tft --config config/config.yaml --seeds 5
```

#### Outputs esperados

Suite estándar **+ atención y pesos VSN guardados** (extensión propuesta del
runner para TFT, ya que estos artefactos son la fuente de la
interpretabilidad). Por defecto: `experiments/tft/<station>/seed={42..46}/`
con la suite común.
""")

# 2.6 Informer
md("""
### 2.6 Informer

#### Justificación

Segundo Transformer del benchmark, contraste arquitectónico con TFT. Diseñado
para **horizontes largos** vía ProbSparse self-attention (O(L log L)) y
generative decoder (one-shot, sin error compounding). El horizonte de 168 h
del proyecto (7 días) es exactamente el régimen donde Informer brilla.

Referencia: Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., &
Zhang, W. (2021). *Informer: Beyond efficient transformer for long sequence
time-series forecasting.* AAAI 35(12), 11106–11115. arXiv:2012.07436.

> ⚠️ **Estado del skeleton**: existe `src/models/model_transformer.py` con
> arquitectura Transformer vanilla. La variante Informer se selecciona vía
> `config/models/informer.yaml::variant: informer`. La lógica específica
> (ProbSparse, distilling, generative decoder) **se debe implementar
> internamente** dispatch-by-variant.

#### Arquitectura

```
Input X (B, 168, n_features)
  └─ Token + positional embedding
  └─ ProbSparse self-attention encoder (3 capas)  ← O(L log L)
       └─ Self-attention distilling (conv + max-pool entre capas)
  └─ Generative decoder (2 capas)                 ← one-shot
  └─ Linear → Output ŷ (B, horizon=168, n_targets=1)
```

**Parámetros aproximados**: ~10⁶–10⁷ dependiendo de `d_model` y profundidad.

#### Hiperparámetros

| hiperparámetro | valor base | rango | justificación |
|---|---|---|---|
| `d_model` | 128 | {64, 128, 256} | Default del paper para series multivariadas medianas. |
| `nhead` | 8 | {4, 8} | Multi-head amplio para capturar patrones diversos. |
| `num_encoder_layers` | 3 | {2, 3, 4} | Profundidad estándar; el distilling reduce L entre capas. |
| `num_decoder_layers` | 2 | {1, 2} | Decoder ligero por la generación one-shot. |
| `dim_feedforward` | 256 | — | 2 × d_model como heurística. |
| `dropout` | 0.1 | — | Default Transformer. |
| `prob_sparse_factor` | 5 | {3, 5, 10} | Factor `c` de ProbSparse: `u = c · log L`. |
| `lr` | 1e-4 | — | Lower que LSTM, alineado con Transformers. |
| `optimizer` | AdamW | — | AdamW ayuda en Transformers grandes. |
| `weight_decay` | 1e-4 | — | Importante en Transformers. |
| `warmup_steps` | 1000 | — | Calentamiento del LR (estándar Transformer). |
| `epochs` (máx.) | 50 | — | El paper original usa 5–10 épocas + warmup; nosotros damos margen. |
| `early_stopping_patience` | 10 | — | Mayor que LSTM por inestabilidad inicial. |

#### Función de pérdida

**MSE** sobre la salida puntual. Informer no es nativo probabilístico; para
intervalos se postprocesa con bootstrap o ensemble de semillas.

#### Estrategia de entrenamiento

- **Optimizer**: AdamW, lr=1e-4.
- **Weight decay**: 1e-4.
- **Scheduler**: warmup lineal de 1000 pasos seguido de decaimiento (cosine o
  step según availability).
- **Gradient clipping**: 1.0.
- **Mixed precision**: recomendado.

#### Comando para entrenar

```bash
python -m src.training.runner --model informer --config config/config.yaml --seeds 5
```

#### Outputs esperados

`experiments/informer/<station>/seed={42..46}/` con la suite estándar.
""")

# ============================================================================
# Sección 3 — Requisitos experimentales mejorados
# ============================================================================
md("## 3. Requisitos Experimentales Mejorados")

md("""
### 3.1 Control de semillas

Toda corrida pasa por `src.utils.seed.set_seed(seed)` que fija las semillas de
**Python `random`**, **NumPy**, **PyTorch (CPU + CUDA)** y configura
`torch.backends.cudnn.deterministic = True`. Validado por
`tests/test_seed.py`.

| seed_id | semilla | uso |
|---|---|---|
| 1 | 42 | Semilla base (`cfg.project.seed`). |
| 2 | 43 | base + 1 |
| 3 | 44 | base + 2 |
| 4 | 45 | base + 3 |
| 5 | 46 | base + 4 |

> El runner usa `seed = cfg.project.seed + i` para `i ∈ [0, n_runs)`. Para
> mayor diversidad estadística se puede sustituir por una lista no contigua
> (ej. `[42, 123, 2024, 7, 314159]`) modificando `runner.py:148`. La rúbrica
> exige **N ≥ 5** corridas — el default cumple.

> 📊 **Reporte estadístico**: cada modelo se ejecutará N=5 veces con semillas
> distintas. Las métricas se reportarán como **media ± desviación estándar**
> y, en notebook 06, con **IC 95 %** vía bootstrap (ya configurado en
> `cfg.evaluation.bootstrap_ci: true`, `bootstrap_n: 1000`).
""")

md("""
### 3.2 Repetición de experimentos

```text
            ┌─ seed=42 ─► train ─► val ─► test ─► metrics_42.json
            ├─ seed=43 ─► train ─► val ─► test ─► metrics_43.json
modelo X ───┼─ seed=44 ─► train ─► val ─► test ─► metrics_44.json
            ├─ seed=45 ─► train ─► val ─► test ─► metrics_45.json
            └─ seed=46 ─► train ─► val ─► test ─► metrics_46.json

→ aggregate_runs(modelo X) → media ± std + IC 95 %
```

**Total de runs en el benchmark**: 6 modelos × 5 semillas × 40 estaciones =
**1 200 runs** si se entrena modelo-por-estación (régimen actual del runner);
ó **6 × 5 = 30 runs** en régimen panel global (recomendado para TFT/Informer
una vez que el runner se adapte).
""")

md("### 3.3 Hardware y entorno")

code("""
import torch
import platform

print(f"Plataforma   : {platform.platform()}")
print(f"Python       : {platform.python_version()}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA disp.   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM total   : {vram:.2f} GB")
    print(f"CUDA version : {torch.version.cuda}")
else:
    print("Backend      : CPU (entrenamiento factible pero más lento)")
""")

md("""
**Tiempo estimado por run** (con GPU mid-range tipo RTX 3060 12 GB):

| Modelo | t/run/estación (estimado) | Observación |
|---|---|---|
| Persistencia | < 1 min | Sólo evaluación |
| LSTM | 4–8 min | hidden=128, ~50 épocas con early stopping |
| GRU | 3–6 min | ~25 % menos que LSTM |
| N-BEATSx | 6–12 min | Stack profundo |
| TFT | 12–25 min | Mayor cantidad de parámetros |
| Informer | 10–20 min | Pero converge en pocas épocas |

**Tiempo total estimado del benchmark** (régimen per-estación, 40 estaciones,
5 semillas, 6 modelos): aproximadamente **40–70 horas**. En régimen panel
global (recomendado para TFT/Informer): **6–12 horas**.

Recomendación: arrancar con un **subset de 5 estaciones representativas** (las
mismas 5 del EDA §5.1: A101 Manaus, A309 Petrolina, A001 Brasilia, A701 São
Paulo, A801 Porto Alegre) para iterar hiperparámetros, y escalar al panel
completo para la evaluación final.
""")

md("""
### 3.4 Trazabilidad

Cada run guarda en `experiments/<model>/<station>/seed=<s>/env.json` (vía
`src.utils.reproducibility.capture_environment`):

- **Versión de Python** y de las librerías relevantes (`torch`, `numpy`,
  `pandas`, `pytorch-forecasting` cuando aplique).
- **Commit git** actual (hash + branch).
- **Hash del dataset** (`hash_dataframe(...)` sobre el train DataFrame).
- **Semilla** y `cfg.project.seed`.
- **Hardware** (CUDA disponible, nombre GPU si aplica).

**MLflow** está prefigurado en `cfg.training.mlflow`:

```yaml
training:
  mlflow:
    enabled: false                       # cambiar a true para activar
    tracking_uri: "file:./experiments/mlruns"
```

Para visualizar:

```bash
mlflow ui --backend-store-uri file:./experiments/mlruns
```

### Checklist de los 5 requisitos del PDF

- ✓ **Semillas**: `set_seed` valida CPU/CUDA/Python; 5 corridas por modelo.
- ✓ **Repeticiones**: ≥ 5 runs (`cfg.project.seeds_per_model: 5`).
- ✓ **Media + std + IC 95 %**: agregación por `aggregate_runs(...)` en §5.5;
  IC 95 % via bootstrap en notebook 06.
- ✓ **GPU documentada**: celda 3.3 ejecutable + tabla de tiempos estimados.
- ✓ **Trazabilidad**: `env.json` por run con commit, libs, hash dataset,
  semilla; MLflow opcional.
""")

# ============================================================================
# Sección 4 — Buenas prácticas anti-leakage
# ============================================================================
md("""
## 4. Buenas Prácticas Anti-Leakage

> 🛡️ **Las cinco garantías del proyecto, todas validadas por tests
> automáticos.**

### 4.1 Separación train/val/test

Split **temporal** estricto por años: train = 2018–2023, val = 2024,
test = 2025. Sin barajado ni *k-fold aleatorio*. Garantía formal:
`max(train.index) < min(val.index) < max(val.index) < min(test.index)`.

✓ Validado por:
- `tests/test_split_no_leakage.py::test_by_year_strict_chronology`
- `tests/test_split_real_data.py::test_no_leakage_on_real_data` (sobre
  `data/processed/A001.parquet`).

### 4.2 Transformaciones ajustadas sólo en train

`FeatureScaler.fit(X, source="train")` rechaza vía aserción cualquier intento
de fitear sobre val o test. Aplicado tanto al input como al target.

✓ Validado por: `tests/test_scaler_fit_train_only.py` (4 tests, incluyendo
intento explícito de fit con `source="val"` que debe lanzar `AssertionError`).

### 4.3 Sin mezcla temporal

`make_windows` genera ventanas **completamente contenidas** dentro de cada
split. La primera ventana de val empieza al menos `lookback` pasos después del
inicio de val.

✓ Validado por: `tests/test_windowing_no_leakage.py` (2 tests):
- `test_windows_within_split_only` — ningún timestamp de la ventana cae fuera
  de su split.
- `test_no_window_spans_two_splits` — ningún caso cruza la frontera train/val.

### 4.4 Imputación causal

`process.py` aplica `ffill(limit=6)` (forward fill, ≤ 6 h consecutivas).
Ventanas con NaN remanente se **descartan** en el dataset (no se rellenan con
0 ni con la media), evitando ruido informativo y leakage de estadísticos
globales.

✓ Heredado de `src/data/process.py`; verificado implícitamente por
`tests/test_process_integrity.py`.

### 4.5 Validación consistente

Mismo dataset, mismo `lookback=168`, mismo `horizon`, misma normalización
(scaler fit train por estación) para los 6 modelos. Cualquier diferencia en
métricas atribuible a la **arquitectura**, no al preprocesamiento.

✓ Garantizado por `runner.py`: el bloque de carga + ventaneo + escalado es
**idéntico** entre modelos; sólo cambia la clase del forecaster.
""")

# ============================================================================
# Sección 5 — Evaluación inicial
# ============================================================================
md("## 5. Evaluación Inicial")

md("""
> ℹ️ Esta sección define **helpers** y **visualizaciones**. Se ejecutará
> *después* de que los runs se hayan completado en terminal. Si los runs aún
> no están disponibles, las celdas mostrarán un mensaje claro y se saltarán.
""")

md("""
### 5.1 Métricas usadas

| Métrica | Fórmula | Justificación |
|---|---|---|
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Métrica principal; penaliza errores grandes (apropiada para temperatura). |
| **MAE** | `mean(|y - ŷ|)` | Robusta a outliers; complementaria al RMSE. |
| **R²** | `1 - SSres/SStot` | Varianza explicada; útil para reportar calidad relativa. |
| **sMAPE** | `200 · mean(|y-ŷ|/(|y|+|ŷ|))` | Escala-libre, comparable entre estaciones; cuidado: indefinida si y≈ŷ≈0. |
| **MAPE** | `100 · mean(|y-ŷ|/|y|)` | Escala-libre; problemática cuando `|y| → 0`. |

Las métricas se reportan **por horizonte** (h=24, 72, 168), **por región**
(Norte/Nordeste/Centro-Oeste/Sudeste/Sul) y **agregadas**, ya configurado por:

```yaml
evaluation:
  metrics: [rmse, mae, r2, mape, smape]
  per_horizon: true
```

Los cuts h=24/72/168 se obtienen como slices del vector de output de 168 pasos.
""")

md("### 5.2 Helpers para cargar resultados de los runs")

code("""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path("experiments")
FIG_DIR = Path("results/figures/benchmark")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_MODELS = ["persistence", "lstm", "gru", "nbeats", "tft", "informer"]


def _run_dir(model: str, station: str, seed: int) -> Path:
    return EXP_DIR / model / station / f"seed={seed}"


def load_run_metrics(model: str, station: str, seed: int) -> dict | None:
    \"\"\"Carga metrics.json de un run concreto. Devuelve None si no existe.\"\"\"
    f = _run_dir(model, station, seed) / "metrics.json"
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fp:
        return json.load(fp)


def load_run_history(model: str, station: str, seed: int) -> dict | None:
    f = _run_dir(model, station, seed) / "history.json"
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fp:
        return json.load(fp)


def load_run_predictions(model: str, station: str, seed: int) -> dict | None:
    \"\"\"Devuelve dict con keys 'y_true', 'y_pred', 'timestamps', 'target_names'.\"\"\"
    f = _run_dir(model, station, seed) / "predictions.npz"
    if not f.exists():
        return None
    data = np.load(f, allow_pickle=True)
    return {k: data[k] for k in data.files}


def discover_runs(model: str) -> pd.DataFrame:
    \"\"\"Inventario de runs encontrados en disco para un modelo.\"\"\"
    rows = []
    base = EXP_DIR / model
    if not base.exists():
        return pd.DataFrame(columns=["station", "seed", "has_metrics", "has_history", "has_preds"])
    for station_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in station_dir.iterdir() if p.is_dir() and p.name.startswith("seed=")):
            seed = int(seed_dir.name.split("=", 1)[1])
            rows.append({
                "station": station_dir.name,
                "seed": seed,
                "has_metrics": (seed_dir / "metrics.json").exists(),
                "has_history": (seed_dir / "history.json").exists(),
                "has_preds": (seed_dir / "predictions.npz").exists(),
            })
    return pd.DataFrame(rows)


def aggregate_runs(model: str, seeds: list[int] = DEFAULT_SEEDS) -> pd.DataFrame:
    \"\"\"Agrega métricas por (estación, seed) en estadísticos por estación.

    Devuelve DataFrame con índice = estación y columnas =
    [<metric>_mean, <metric>_std] por cada métrica disponible.
    \"\"\"
    rows = []
    base = EXP_DIR / model
    if not base.exists():
        return pd.DataFrame()
    for station_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        per_seed: list[dict] = []
        for s in seeds:
            m = load_run_metrics(model, station_dir.name, s)
            if m is None:
                continue
            # Sólo conservamos los floats de primer nivel para agregación.
            per_seed.append({k: v for k, v in m.items() if isinstance(v, (int, float))})
        if not per_seed:
            continue
        df = pd.DataFrame(per_seed)
        agg = {}
        for col in df.columns:
            agg[f"{col}_mean"] = df[col].mean()
            agg[f"{col}_std"] = df[col].std(ddof=1) if len(df) > 1 else 0.0
        rows.append({"station": station_dir.name, **agg})
    return pd.DataFrame(rows).set_index("station") if rows else pd.DataFrame()


# Inventario rápido por modelo (no falla si no hay runs todavía).
inventario = pd.DataFrame({
    m: [len(discover_runs(m))] for m in DEFAULT_MODELS
}, index=["#runs encontrados"]).T
inventario
""")

md("### 5.3 Curvas de entrenamiento por modelo")

code("""
import matplotlib.pyplot as plt
from src.utils.regions import region_color

REPS = {
    "Norte": "A101",
    "Nordeste": "A309",
    "Centro-Oeste": "A001",
    "Sudeste": "A701",
    "Sul": "A801",
}


def plot_history_for_model(model: str, station: str = "A001",
                            seeds: list[int] = DEFAULT_SEEDS,
                            outname: str | None = None) -> None:
    \"\"\"Dibuja train/val loss por época, una curva por semilla. Tolerante a runs faltantes.\"\"\"
    histories = {s: load_run_history(model, station, s) for s in seeds}
    if not any(histories.values()):
        print(f"[{model}/{station}] Resultados aún no disponibles.")
        print(f"  Corre: python -m src.training.runner --model {model} --seeds 5")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for s, h in histories.items():
        if h is None:
            continue
        epochs = [r["epoch"] for r in h]
        tr = [r["train_loss"] for r in h]
        va = [r["val_loss"]   for r in h]
        ax.plot(epochs, tr, lw=0.7, alpha=0.6, label=f"train s={s}")
        ax.plot(epochs, va, lw=1.2, label=f"val s={s}")
        # Marca early stopping (epoch del mejor val).
        best_ep = epochs[int(np.argmin(va))]
        ax.axvline(best_ep, color="red", linestyle=":", lw=0.5, alpha=0.5)
    ax.set_title(f"{model} — {station}: curvas de entrenamiento (cruz roja = mejor val)")
    ax.set_xlabel("época")
    ax.set_ylabel("loss (MSE escalado)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    if outname:
        fig.savefig(FIG_DIR / outname, dpi=120, bbox_inches="tight")
    plt.show()


# Una llamada por modelo. Si no hay runs todavía, imprime el comando y sigue.
for m in DEFAULT_MODELS:
    plot_history_for_model(m, station="A001", outname=f"05_3_history_{m}_A001.png")
""")

md("""
### 5.4 Análisis cualitativo preliminar

Para una estación representativa por región, plotea predicción vs ground truth
en una ventana de 7 días y para los 3 horizontes (24h, 72h, 168h). Tolerante a
runs faltantes — si no hay predicciones, salta con mensaje.
""")

code("""
def plot_prediction_vs_truth(model: str, station: str, seed: int = 42,
                              window_days: int = 7, h_cuts: tuple = (24, 72, 168),
                              outname: str | None = None) -> None:
    \"\"\"Predicción vs ground truth para los 3 cortes de horizonte.\"\"\"
    preds = load_run_predictions(model, station, seed)
    if preds is None:
        print(f"[{model}/{station} seed={seed}] sin predicciones.")
        print(f"  Corre: python -m src.training.runner --model {model} --seeds 5")
        return
    y_true = preds["y_true"]   # (N, horizon, n_targets)
    y_pred = preds["y_pred"]
    # Tomamos las primeras N ventanas que cubren `window_days * 24` horas.
    n_windows = min(window_days * 24, y_true.shape[0])
    # Si el target es univariado, tomamos canal 0.
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
        y_pred = y_pred[..., 0]
    fig, axes = plt.subplots(len(h_cuts), 1, figsize=(13, 8), sharex=True)
    color = region_color(_region_of(station))
    for ax, h in zip(axes, h_cuts):
        h_idx = min(h, y_true.shape[1]) - 1
        yt = y_true[:n_windows, h_idx]
        yp = y_pred[:n_windows, h_idx]
        ax.plot(yt, color="black", lw=1.0, label="ground truth")
        ax.plot(yp, color=color, lw=1.0, alpha=0.85, label=f"{model} (h={h}h)")
        ax.set_title(f"{model} — {station} — horizonte +{h}h (semilla {seed})")
        ax.set_ylabel("temp_c (°C)")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("paso temporal (h, primeros 7 días)")
    plt.tight_layout()
    if outname:
        fig.savefig(FIG_DIR / outname, dpi=120, bbox_inches="tight")
    plt.show()


def _region_of(station: str) -> str:
    from src.utils.regions import region_of as _ro
    try:
        return _ro(station)
    except KeyError:
        return "Norte"  # fallback


# Por defecto: LSTM, semilla 42, 5 estaciones representativas (una por región).
for region, station in REPS.items():
    plot_prediction_vs_truth("lstm", station, seed=42,
                             outname=f"05_4_pred_vs_truth_lstm_{station}.png")
""")

md("### 5.5 Tabla preliminar de métricas")

code("""
def benchmark_table(models: list[str] = DEFAULT_MODELS,
                    seeds: list[int] = DEFAULT_SEEDS) -> pd.DataFrame:
    \"\"\"Agrega resultados de los modelos disponibles en una tabla unificada.

    Cada fila = (modelo, métrica). Cada columna = estadístico (mean / std).
    Si un modelo no tiene runs todavía, queda omitido.
    \"\"\"
    rows = []
    for m in models:
        df_m = aggregate_runs(m, seeds=seeds)
        if df_m.empty:
            continue
        # Promedio sobre estaciones del mean por métrica (panel-level).
        mean_cols = [c for c in df_m.columns if c.endswith("_mean")]
        std_cols  = [c for c in df_m.columns if c.endswith("_std")]
        panel_mean = df_m[mean_cols].mean()
        panel_std  = df_m[std_cols].mean()  # promedio del std intra-modelo
        n_estaciones = len(df_m)
        for col in mean_cols:
            base = col[:-len("_mean")]
            rows.append({
                "modelo": m,
                "metric": base,
                "panel_mean": float(panel_mean[col]),
                "panel_std_intra_seed": float(panel_std.get(f"{base}_std", float("nan"))),
                "n_estaciones": n_estaciones,
            })
    if not rows:
        print("Aún no hay resultados agregables. Corre los entrenamientos primero.")
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .set_index(["modelo", "metric"])
              .sort_index())


tabla = benchmark_table()
tabla
""")

md("""
> 🔬 **El análisis estadístico riguroso** (Diebold-Mariano por pares,
> Friedman + Nemenyi, Wilcoxon signed-rank sobre errores pareados, Ljung-Box
> y BDS sobre residuos) **se realiza en `notebooks/06_benchmark_final.ipynb`**.
> Aquí solo entregamos la tabla descriptiva como insumo de inspección rápida.
""")

# ============================================================================
# Sección 6 — Síntesis ejecutiva
# ============================================================================
md("""
## 6. Síntesis Ejecutiva del Capítulo

### 6.1 Modelos implementados

| # | Modelo | Familia | Estado del skeleton |
|---|---|---|---|
| 1 | **Persistencia** | Baseline ingenuo | `model_persistence.py` (skeleton, NotImplementedError) |
| 2 | **LSTM** | Recurrente | `model_lstm.py` ✓ funcional (forward implementado) |
| 3 | **GRU** | Recurrente liviano | `model_gru.py` ✓ funcional |
| 4 | **N-BEATSx** | MLP con basis | `model_nbeats.py` ✓ funcional (variante `nbeats`) |
| 5 | **TFT** | Híbrido (LSTM+Att+VSN) | `model_tft.py` (skeleton, NotImplementedError) — **paper guía** |
| 6 | **Informer** | Transformer eficiente | `model_transformer.py` con `variant: informer` (lógica específica pendiente dentro del archivo) |

> ⚠️ **TODO de implementación** identificado:
>
> - **`PersistenceForecaster.forward`**: lógica simple (repetir último target).
> - **`TFTForecaster`**: implementación completa (usar `pytorch-forecasting` como referencia).
> - **`TransformerForecaster.variant="informer"`**: añadir ProbSparse self-attention, distilling y generative decoder.
> - **Adaptación del runner para régimen panel global** (un único modelo entrenado sobre todas las estaciones con embedding de `station_id`): requerido para TFT/Informer/DeepAR. El runner actual entrena per-estación.
> - **Cambiar `cfg.task.horizon` a 168** y reportar slices {24, 72, 168} en
>   `metrics.json` y en la tabla de la §5.5.

### 6.2 Garantías de reproducibilidad confirmadas

- ✓ Semillas: `set_seed` cubre Python/NumPy/PyTorch (CPU+CUDA), 5 corridas por
  modelo (`cfg.project.seeds_per_model: 5`).
- ✓ Tests anti-leakage: `test_split_no_leakage`, `test_split_real_data`,
  `test_scaler_fit_train_only`, `test_windowing_no_leakage`,
  `test_process_integrity`. **33 tests pasando**.
- ✓ Trazabilidad por run: `env.json` con commit, libs, hash dataset, semilla,
  hardware (vía `capture_environment`).
- ✓ MLflow opcional (`cfg.training.mlflow.enabled`).
- ✓ Validación consistente: idéntico pipeline (loader, scaler, ventaneo) para
  los 6 modelos — la diferencia es la arquitectura.

### 6.3 Resultados preliminares observados

> Esta subsección se completa una vez ejecutados los runs. Si las celdas 5.3
> y 5.5 ya se ejecutaron con éxito, copiar aquí el top-3 por RMSE total y los
> tiempos por época. Mientras tanto, espacio reservado.

**Top-3 esperado a priori (hipótesis del SOTA, a refutar/confirmar)**:

1. **TFT** (multi-horizon nativo + embeddings + quantile loss + atención).
2. **Informer** (Transformer eficiente para 168 h).
3. **N-BEATSx** (descomposición tendencia/estacionalidad alineada con STL del EDA).

Como **piso** (que cualquier modelo DL debe superar significativamente):
**Persistencia**.

### 6.4 Próximos pasos

1. Implementar los TODO listados en §6.1 (Persistence forward, TFT, variante
   Informer, adaptación panel del runner).
2. Cambiar `cfg.task.horizon` a 168 y verificar que `compute_metrics` reporta
   slices {24, 72, 168}.
3. Lanzar el benchmark en terminal por cada modelo:

   ```bash
   for model in persistence lstm gru nbeats tft informer; do
       python -m src.training.runner --model $model --seeds 5
   done
   ```

4. **Notebook 05 (`05_guide_paper.ipynb`)** — análisis profundo del **TFT**
   como paper guía: réplica controlada en una estación, descomposición de
   atención y VSN, comparación contra los baselines.
5. **Notebook 06 (`06_benchmark_final.ipynb`)** — **análisis estadístico
   riguroso**: Diebold-Mariano (pareado), Friedman + Nemenyi/Bonferroni/Holm
   (multimodelos), Wilcoxon signed-rank, Ljung-Box y BDS sobre residuos. IC
   95 % por bootstrap. Tabla final por horizonte y por región.

---

*Fin del capítulo 04. Antes de proceder al notebook 05, completar los TODO de
implementación e iniciar la primera tanda de runs en terminal.*
""")


# ============================================================================
# Build sin ejecutar
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

out_path = Path("notebooks/04_benchmark_models.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook escrito: {out_path} ({len(cells)} celdas)")
