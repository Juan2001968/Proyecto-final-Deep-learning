# Proyecto Final — Deep Learning para Forecasting Meteorológico (INMET)

Forecasting multistep de variables meteorológicas a partir de **8 años de datos
horarios del INMET (Brasil)** usando arquitecturas de Deep Learning, con
benchmark estadísticamente riguroso (RMSE / MAE / R² + Diebold-Mariano,
Friedman/Nemenyi, Wilcoxon).

> Plantilla académica reproducible. La lógica específica de cada modelo se
> rellena dentro de `src/models/`; el resto del pipeline (ingesta, splits,
> ventaneo, scaling, evaluación) ya está cableado para evitar **data leakage
> temporal**.

## Tabla de contenidos
- [Resumen](#resumen)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Inicio rápido](#inicio-rápido)
- [Datos INMET](#datos-inmet)
- [Reproducibilidad](#reproducibilidad)
- [Benchmark estadístico](#benchmark-estadístico)
- [Jupyter Book](#jupyter-book)

## Resumen

| Item                | Valor                                                       |
|---------------------|-------------------------------------------------------------|
| Tarea               | Regresión multistep sobre series de tiempo (forecasting)    |
| Frecuencia          | Horaria                                                     |
| Horizonte           | Configurable en `config/config.yaml` (`horizon`)            |
| Lookback            | Configurable en `config/config.yaml` (`lookback`)           |
| Variable objetivo   | Configurable (`target`); default: temperatura del aire      |
| Métrica principal   | RMSE (complementarias: MAE, R²; tests: Diebold-Mariano)     |
| Modelos base        | LSTM, GRU, TCN, Transformer (Informer/Autoformer), N-BEATS  |

## Estructura del repositorio

```
proyecto-final-deep/
├── config/        # YAML de proyecto, estaciones y modelos
├── data/          # raw / interim / processed (no versionado)
├── src/           # código modular: utils, data, eda, models, training, eval, benchmark
├── notebooks/     # 01..06 — narrativa académica
├── experiments/   # logs/checkpoints por run
├── results/       # figuras, tablas, tests estadísticos
├── jupyter_book/  # libro reproducible
└── tests/         # tests críticos de no-leakage
```

## Requisitos

- Python 3.11
- Ver `pyproject.toml` (o `requirements.txt`) para dependencias.
- Opcional: GPU (CUDA 12.x) para entrenamiento en PyTorch.

## Inicio rápido

```bash
make setup        # instala dependencias y prepara entorno
make ingest       # descomprime / parsea CSVs INMET → data/interim
make process      # crea splits temporales y parquet en data/processed
make eda          # EDA general + EDA de series temporales
make train MODEL=lstm
make benchmark    # corre todos los modelos con N semillas
make book         # build del Jupyter Book
make all          # pipeline completo
```

## Datos INMET

Los CSVs anuales por estación se descargan desde
<https://portal.inmet.gov.br/dadoshistoricos> y se colocan en `data/raw/`.
Detalles en [`data/raw/README.md`](data/raw/README.md). Atención a:

- separador `;`, decimal `,`, encoding `latin-1`/`ISO-8859-1`,
- valores faltantes codificados como `-9999`,
- encabezado con metadatos (`REGIÃO`, `UF`, `CODIGO WMO`, lat/lon/alt),
- columnas `Data` + `Hora UTC` que combinamos en un timestamp UTC tz-naive,
- los nombres de columna varían entre años (2018: `DATA (YYYY-MM-DD)` /
  `HORA (UTC)`; 2019+: `Data` / `Hora UTC`). El proceso resuelve estas
  variantes por *pattern matching* normalizado.

### Pipeline de datos

```
data/raw/<year>/*.CSV
       │
       ▼  make ingest  (src.data.ingest_inmet)
data/interim/<wmo>/<year>.csv + metadata.json
       │
       ▼  make process  (src.data.process)
data/processed/<wmo>.parquet      (8 años, índice horario tz-naive)
       │
       ▼  notebooks/02_eda.ipynb  ó  pipeline de entrenamiento
```

| Etapa | Módulo | Entrada | Salida |
|---|---|---|---|
| **Ingest** | `src.data.ingest_inmet` | `data/raw/<year>.zip` (o sus CSVs) | `data/interim/<wmo>/<year>.csv` + `metadata.json` por estación |
| **Process** | `src.data.process` | `data/interim/<wmo>/*.csv` | `data/processed/<wmo>.parquet` |

`make process` orquesta in-memory por estación:

1. lee y concatena los 8 CSV anuales,
2. canoniza nombres de columna (tolerante a variantes entre años),
3. parsea timestamp `datetime64[ns]` **tz-naive** (UTC implícito),
4. recorta outliers físicos (temp ∈ [-10, 50] °C, hum ∈ [0, 100] %, etc.),
5. reindexa a rejilla horaria regular y reporta gaps consecutivos > 6 h,
6. imputa causalmente los gaps cortos con `ffill(limit=6)`,
7. añade features cíclicas (`hour_{sin,cos}`, `doy_{sin,cos}`, `month_{sin,cos}`),
8. añade `station_id` (entero), `region`, `biome`, `koppen_class` desde
   `config/stations.yaml`.

Los lags y rolling features **no** se generan aquí — pertenecen a feature
engineering por modelo (`src/data/features.py`) durante el entrenamiento.

## Reproducibilidad

- Semillas globales (`numpy`, `torch`, `random`, CUDA) en `src/utils/seed.py`.
- Captura de versiones de librerías y hash del dataset en
  `src/utils/reproducibility.py`.
- Splits **temporales** (no aleatorios) verificados por
  `tests/test_split_no_leakage.py`.
- Ventaneo *lookback→horizon* sin cruzar límites de split
  (`tests/test_windowing_no_leakage.py`).
- Escalado **fit solo con train** (`tests/test_scaler_fit_train_only.py`).

## Benchmark estadístico

`src/benchmark/stats_tests.py` incluye:

- **Diebold-Mariano** para comparar dos modelos en errores de pronóstico.
- **Friedman + post-hoc Nemenyi / Bonferroni / Holm** para múltiples modelos.
- **Wilcoxon signed-rank** sobre errores pareados.
- **Ljung-Box** sobre residuos, **BDS** para no-linealidad residual.

## Jupyter Book

```bash
make book
# o:
jupyter-book build jupyter_book
```

El libro consume directamente los notebooks de `notebooks/` y los assets de
`results/`.
