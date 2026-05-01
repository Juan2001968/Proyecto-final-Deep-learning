"""Construye notebooks/05_guide_paper.ipynb (sin ejecutar)."""

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
# Paper Guía: Temporal Fusion Transformer — Análisis Técnico y Adaptación al Proyecto

**Resumen ejecutivo.** El paper guía del proyecto es el **Temporal Fusion
Transformer** de Lim et al. (2021), publicado en *International Journal of
Forecasting* (Q1, JCR). Resuelve un problema directamente alineado con el
nuestro: **forecasting multi-horizonte multivariado con entidades
heterogéneas, covariables conocidas a futuro y necesidad de
interpretabilidad**. Su arquitectura combina LSTM (encoder-decoder local),
multi-head self-attention (dependencias largas), Variable Selection Networks
(interpretabilidad por variable) y quantile loss (incertidumbre nativa). El
proyecto **mantiene fielmente** la espina arquitectónica (VSN + GRN + LSTM +
atención + cuantiles) y **adapta** los inputs al panel INMET (40 estaciones ×
5 macrorregiones), las covariables estáticas reales (lat/lng/alt/bioma/Köppen)
y el régimen de horizontes {24, 72, 168} h. Lo que **se difiere** son
detalles de scheduling y augmentation no críticos. Este capítulo justifica la
elección, describe el modelo en detalle y mapea explícitamente cada decisión
del EDA al mecanismo del TFT que la honra.
""")

# ============================================================================
# Sección 1 — Referencia
# ============================================================================
md("## 1. Referencia y Datos del Paper")

md("""
### 1.1 Referencia académica completa

> **Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).** *Temporal Fusion
> Transformers for interpretable multi-horizon time series forecasting.*
> **International Journal of Forecasting, 37**(4), 1748–1764.
> https://doi.org/10.1016/j.ijforecast.2021.03.012

| Campo | Valor |
|---|---|
| **Revista** | *International Journal of Forecasting* (Elsevier) |
| **Cuartil JCR** | **Q1** |
| **Factor de impacto** | ~7.0 (categoría Operations Research & Management Science) [verificar fecha exacta] |
| **Año de publicación** | 2021 |
| **Citas** | > 5 000 (Google Scholar, consulta orientativa 2024–2025) [verificar fecha exacta] |
| **DOI** | `10.1016/j.ijforecast.2021.03.012` |
| **arXiv** | `arXiv:1912.09363` (preprint 2019, versión final 2021) |
| **Código oficial** | `pytorch-forecasting` (mantenido por la comunidad) y referencia original en TensorFlow del Google Research GitHub |

### 1.2 Autores y contexto

| Autor | Filiación |
|---|---|
| **Bryan Lim** | Oxford-Man Institute & University of Oxford (en el momento de publicación) |
| **Sercan Ö. Arık** | Google Cloud AI Research |
| **Nicolas Loeff** | Google Cloud AI Research |
| **Tomas Pfister** | Google Cloud AI Research |

El TFT se inscribe en una **línea de investigación más amplia** en Google
Cloud AI Research sobre **forecasting estructural** y **modelos de atención
para series**, que también incluye trabajos posteriores como **TimeGPT** y
extensiones interpretables. Lim, antes y después del TFT, ha publicado sobre
forecasting financiero (volatility), forecasting médico y forecasting con
covariables exógenas — toda la maquinaria que vemos en el TFT madura a través
de esos antecedentes.

### 1.3 Resumen del paper

El TFT aborda el problema canónico del **forecasting multi-horizonte
multivariado** sobre paneles de series temporales heterogéneas — situaciones
en las que coexisten **covariables conocidas a futuro** (calendario, hora del
día, holidays), **covariables observadas pasadas** (variables exógenas con
historia) y **metadatos estáticos por entidad** (identificador, atributos
inmutables). El paper propone una arquitectura **híbrida** que combina cuatro
ideas: (i) *Variable Selection Networks* que aprenden por timestep qué
variables aportan al pronóstico, (ii) *Gated Residual Networks* como bloque
universal para regular el flujo de información, (iii) un **encoder-decoder
LSTM** para procesar dependencias locales más una **multi-head self-attention
masked** para dependencias largas, y (iv) **quantile loss** multi-percentil
que produce intervalos de confianza nativos. El modelo se evalúa sobre cuatro
datasets públicos (**Electricity, Traffic, Volatility, Retail**), donde
**supera a DeepAR, DeepState, MQ-RNN y métodos clásicos** con mejoras
promedio del 7–25 % en quantile loss según dataset. La aportación clave
respecto a la literatura previa es **unificar interpretabilidad,
multi-horizonte y heterogeneidad de entidades** en una arquitectura única
entrenable end-to-end, donde la interpretabilidad se obtiene "gratis" como
subproducto de los pesos de las VSN y de la atención multi-head.
""")

# ============================================================================
# Sección 2 — Justificación de selección
# ============================================================================
md("## 2. Justificación de Selección como Paper Guía")

md("### 2.1 Criterios de selección")

code("""
import sys, os
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
while not (REPO_ROOT / "config").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import pandas as pd

criterios = pd.DataFrame([
    {"criterio": "Relevancia al problema (forecasting temp_c multi-horizonte)",
     "cumple": "✓",
     "comentario": "Diseño nativo para multi-horizonte multivariado; benchmarks del paper incluyen variables continuas con estacionalidad fuerte (Electricity)."},
    {"criterio": "Impacto en la literatura (citas, Q1)",
     "cumple": "✓",
     "comentario": "Int. J. Forecasting (Q1); >5 000 citas; referencia obligada en cualquier benchmark de forecasting con DL."},
    {"criterio": "Recencia (publicado 2020–2025)",
     "cumple": "✓",
     "comentario": "Publicación final 2021; preprint 2019; sigue siendo SOTA en 2024 frente a Transformers puros."},
    {"criterio": "Adaptable a panel multi-estación (40 estaciones)",
     "cumple": "✓",
     "comentario": "Static covariates son ciudadanos de primera clase: station_id + region + biome + Köppen + lat/lng/alt encajan natívamente."},
    {"criterio": "Soporta covariables temporales conocidas a futuro",
     "cumple": "✓",
     "comentario": "Inputs separados como 'known future': hour_sin/cos, doy_sin/cos, month_sin/cos del EDA entran sin fricción."},
    {"criterio": "Soporta covariables observadas pasadas",
     "cumple": "✓",
     "comentario": "Inputs 'past observed': humidity, pressure, radiation, wind_speed, dew_point seleccionadas en EDA §6."},
    {"criterio": "Pronóstico probabilístico (intervalos)",
     "cumple": "✓",
     "comentario": "Quantile loss multi-percentil (P10/P50/P90) — captura colas para eventos extremos del EDA."},
    {"criterio": "Interpretabilidad (variables y tiempo)",
     "cumple": "✓",
     "comentario": "VSN dan pesos por variable; multi-head attention da pesos por paso temporal — defendible académicamente."},
    {"criterio": "Implementación abierta disponible",
     "cumple": "✓",
     "comentario": "`pytorch-forecasting` mantenido por la comunidad + ref original en TF del Google Research GitHub."},
    {"criterio": "Costo computacional viable en hardware mid-range",
     "cumple": "≈",
     "comentario": "Modelo de ~10⁶ parámetros; entrenable en GPU 8–12 GB; mayor que LSTM, menor que Informer en datasets equivalentes."},
])
criterios
""")

md("""
### 2.2 Por qué este paper y no otro

El SOTA del proyecto (Capítulo 03) identificó nueve modelos relevantes y
seleccionó **seis** para el benchmark. El TFT compite directamente con dos
finalistas serios como paper guía: **Informer** (Zhou et al., 2021) y
**N-BEATS** (Oreshkin et al., 2020).

**Frente a Informer**, ambos son Transformers para forecasting de larga
secuencia y ambos manejan los 168 h de horizonte sin compounding
autoregresivo. Pero Informer **carece de embeddings de entidad nativos** y
de un mecanismo elegante para *known future inputs*; las features estáticas
(`station_id`, `region`, `biome`, `koppen_class`, `lat/lng/alt`) se
concatenarían como inputs "aplanados" al encoder, perdiendo la inductive bias
que el TFT explota con sus *static covariate encoders* (cuatro contextos
distintos inyectados en cuatro etapas distintas del modelo). Para el panel
INMET, donde las **5 macrorregiones IBGE × 6 biomas × 9 clases Köppen**
generan una señal categórica masiva, el TFT capitaliza mejor esa información.
Adicionalmente, el TFT entrega **interpretabilidad nativa** vía VSN —
defendible en una entrega académica — mientras Informer es esencialmente una
caja negra en su variante estándar.

**Frente a N-BEATS**, ambos son multi-horizonte single-shot y ambos son
interpretables a su manera (N-BEATS via descomposición *trend + seasonality*
explícita, TFT vía VSN + atención). N-BEATS, sin embargo, es **originalmente
univariado**; las extensiones multivariadas (NBEATSx) y panel
(NeuralForecast) llegaron después y son menos maduras que la implementación
de TFT. Más importante: N-BEATS **no tiene mecanismo nativo para covariables
estáticas heterogéneas** — las features inmutables por estación se inyectan
como features adicionales sin la modulación contextual sofisticada que el
TFT realiza con sus cuatro static encoders.

**El cierre del argumento** lo da la coincidencia entre los hallazgos del EDA
y los mecanismos del TFT: estacionalidad fuerte ↔ known future inputs;
heterogeneidad regional dramática ↔ static covariates; lookback 168 h ↔
self-attention escalable; eventos extremos en colas ↔ quantile loss; necesidad
de interpretabilidad ↔ VSN + atención. **Ningún otro modelo del Top 9
satisface las cinco simultáneamente.**
""")

# ============================================================================
# Sección 3 — Descripción Técnica
# ============================================================================
md("## 3. Descripción Técnica del Modelo Original")

md("### 3.1 Visión general de la arquitectura")

md("""
```text
                        ─── INPUTS ───
   Static covariates           Past inputs                Future inputs
   (categórica + real,         (covariables observadas    (covariables conocidas
    invariantes en t)           + conocidas pasadas)       en horizonte)
            │                       │                            │
            ▼                       │                            │
   ┌──────────────────┐    ┌────────┴─────────┐         ┌────────┴─────────┐
   │ Static Covariate │    │   Variable       │         │   Variable       │
   │    Encoders      │    │   Selection      │         │   Selection      │
   │  (4 contextos)   │    │  Network (VSN)   │         │  Network (VSN)   │
   └──────────────────┘    └────────┬─────────┘         └────────┬─────────┘
            │ c_s,c_e,c_c,c_h        │                            │
            │   ┌────────────────────┘                            │
            │   ▼                                                  │
            │  ┌────────────────────────────────┐                  │
            │  │  LSTM Encoder (locality)       │                  │
            │  │  con estado inicial = c_h      │                  │
            │  └─────────────┬──────────────────┘                  │
            │                │ (states)                            │
            │                ▼                                     │
            │  ┌────────────────────────────────┐                  │
            │  │  LSTM Decoder (locality)       │◄─────────────────┘
            │  └─────────────┬──────────────────┘
            │                │ (h_t)
            │                ▼
            │  ┌────────────────────────────────┐
            └─►│  Static Enrichment (GRN(c_e))  │
               └─────────────┬──────────────────┘
                             ▼
               ┌────────────────────────────────┐
               │ Multi-head Interpretable Attn  │  ← causal mask
               │   Q,K,V sobre encoder+decoder   │
               └─────────────┬──────────────────┘
                             ▼
               ┌────────────────────────────────┐
               │ Position-wise FFN + Add&Norm  │
               └─────────────┬──────────────────┘
                             ▼
               ┌────────────────────────────────┐
               │ Quantile heads (P10, P50, P90) │
               └─────────────┬──────────────────┘
                             ▼
                  ŷ_{t+1..t+H}  (B, H, |Q|)
```

**Tres tipos de inputs explícitos** y un único output multi-horizonte con
cuantiles. Toda la red comparte un building block común — la **Gated Residual
Network (GRN)** — que aparece dentro de las VSN, dentro de los static
covariate encoders, dentro del static enrichment y en las cabezas finales.
""")

md("""
### 3.2 Componentes principales

#### 3.2.1 Variable Selection Networks (VSN)

Las **VSN** son módulos *gating* que determinan, **por timestep**, qué
variables exógenas son relevantes para el pronóstico en ese instante. La
intuición es que en climatología, por ejemplo, la radiación solar pesa más al
mediodía que a medianoche; las VSN aprenden esa dependencia temporal de
relevancia.

Para un conjunto de variables `{x¹_t, x², ..., x^m_t}` en el timestep `t`:

1. Cada variable se proyecta mediante una GRN propia: `h^j_t = GRN_j(x^j_t)`.
2. Un vector concatenado `Ξ_t = [x¹_t || ... || x^m_t]` pasa por una GRN +
   *softmax* para producir **pesos por variable** `v^χ_t ∈ ℝ^m`.
3. La salida combinada es la suma ponderada:
   `ξ_t = Σⱼ v^χ_{t,j} · h^j_t`.

**Aporte de interpretabilidad**: los pesos `v^χ_t` se pueden inspeccionar
post-hoc para reportar **ranking de importancia por variable** (validación
empírica del análisis de Mutual Information del EDA §6).

Existen **tres VSN** en el TFT: una para inputs estáticos, una para inputs
pasados (observados + conocidos pasados) y una para inputs futuros (conocidos
en horizonte).

#### 3.2.2 Gated Residual Networks (GRN)

La **GRN** es el bloque base reutilizado en todo el modelo. Su forma:

```
GRN(a, c=None):
    η₁ = ELU(W₁ a + W_c c + b₁)            # opcional: contexto estático c
    η₂ = W₂ η₁ + b₂
    η₃ = GLU(η₂)                            # gating: σ(W_g η₂) ⊙ (W_z η₂)
    return LayerNorm(a_skip + η₃)           # skip connection desde a
```

**Funciones clave**:

- **Gating (GLU)** decide cuánto del bloque contribuye realmente — útil para
  *bypass* selectivo cuando el módulo no aporta (ej. dataset pequeño donde
  más capacidad sobreajusta).
- **Skip connection** garantiza que la red puede comportarse como identidad
  si el módulo no es informativo (importante para evitar dependencia
  obligatoria de cada bloque).
- **Contexto estático opcional `c`**: la GRN puede ser modulada por un
  vector estático (proveniente de los static covariate encoders), permitiendo
  que metadatos por entidad afecten el procesamiento dinámico.

#### 3.2.3 LSTM Encoder-Decoder

Un **LSTM bidireccional opcional** procesa los inputs pasados (encoder) y un
**LSTM unidireccional** los inputs futuros (decoder). Su rol es capturar
**dependencias locales** — patrones de minutos a horas — antes de que la
atención multi-head capte las dependencias largas.

**Diferencia con un LSTM clásico**: aquí el LSTM **no produce el output
final**. Su salida `φ_t` se enriquece con contexto estático (Static
Enrichment, §3.2.4) y luego pasa por la atención. El LSTM es un módulo
*upstream* dentro del flujo, no la cabeza del modelo.

**Estado inicial inyectado**: el encoder LSTM inicia con `(c_h, c_c)`
provenientes de los static encoders, de modo que la entidad (estación)
modula la dinámica recurrente desde el primer paso.

#### 3.2.4 Static Enrichment Layer

Tras el LSTM, cada timestep `t` se enriquece con **otro** contexto estático
`c_e` distinto del usado para inicializar el LSTM:

```
θ_t = GRN(φ_t, c_e)
```

Esta etapa permite que la entidad (estación) module la representación
temporal en cada paso, sin diluirse en la recurrencia.

**Por qué cuatro contextos estáticos distintos** (`c_s`, `c_e`, `c_c`,
`c_h`): el paper diseña encoders separados para que cada parte del modelo
"vea" los metadatos por entidad de la forma más útil — uno modula la VSN
estática, otro inicializa el LSTM, otro enriquece la representación post-LSTM
y otro modula el último gating. Es una decisión empírica del paper que
mejora la performance medida.

#### 3.2.5 Multi-head Attention sobre la dimensión temporal

La atención multi-head **causal** (con máscara triangular) se aplica sobre
las representaciones enriquecidas `θ_t` para capturar **dependencias largas**
(más allá del alcance del LSTM). El paper introduce una variante
**interpretable**: en lugar de promediar los outputs de los heads
concatenados, **promedia los scores de atención** entre heads — permitiendo
una matriz `α_{i,j}` única e interpretable como "cuánto el paso `i` mira al
paso `j`".

**Aporte de interpretabilidad**: la matriz `α` es directamente reportable.
Para forecasting de temperatura podemos ver, ej., que la predicción del paso
+24 h presta atención fuerte a los pasos -24 y -168 (mismo periodo del día
anterior y misma hora de la semana anterior) — coincidiendo con los lags
relevantes detectados por la ACF en EDA §5.4.

#### 3.2.6 Quantile Loss

La cabeza final no produce un valor puntual sino **un vector de cuantiles**
`(q̂^{0.1}_t, q̂^{0.5}_t, q̂^{0.9}_t)` para cada paso del horizonte. Se
entrena con la loss:

```
QL_q(y, q̂_q) = max( q · (y − q̂_q), (q − 1) · (y − q̂_q) )
loss(y, q̂)   = Σ_q QL_q(y, q̂_q)  promediada sobre batch y horizonte
```

**Por qué quantile y no MSE**:

- MSE minimiza el **error medio cuadrático** y converge al **valor esperado**
  condicional. Para distribuciones asimétricas o con colas pesadas (eventos
  extremos del EDA §2), el valor esperado está sesgado hacia el centro —
  el modelo **sub-predice** las colas.
- Quantile loss converge al **cuantile condicional** exacto. P50 = mediana
  predicha; P10, P90 = bandas. Sin sub-predicción de colas.
- **Bandas P10–P90 ≈ intervalo de confianza del 80 %** sin necesidad de
  bootstrap o ensembles.
""")

md("""
### 3.3 Flujo de datos durante una predicción

Para una muestra (estación A001, ventana terminada en `t`):

1. **Inputs estáticos**: `[station_id=0, region="Centro-Oeste", biome="Cerrado",
   koppen_class="Aw", lat=-15.79, lng=-47.93, alt=1159.5]` → encoders generan
   los cuatro vectores de contexto `c_s, c_e, c_c, c_h`.
2. **Inputs pasados** (`t-167..t`): tensor `(168, n_past)` con
   `[temp_c, humidity_pct, pressure_mb, radiation_kj_m2, wind_speed_ms,
   dew_point_c, hour_sin, hour_cos, doy_sin, doy_cos, month_sin, month_cos]`.
3. **Inputs futuros** (`t+1..t+168`): tensor `(168, n_future)` con sólo las
   covariables conocidas a futuro: `[hour_sin, hour_cos, doy_sin, doy_cos,
   month_sin, month_cos]`. Las exógenas observadas no entran al decoder
   (porque no las conocemos a futuro).
4. **VSN sobre pasado** y **VSN sobre futuro** producen las representaciones
   `ξ^{past}_t` y `ξ^{fut}_t` con pesos por variable inspeccionables.
5. **LSTM encoder** procesa `ξ^{past}_{t-167..t}` partiendo de `c_h` →
   produce `(φ_{t-167}, ..., φ_t)`.
6. **LSTM decoder** procesa `ξ^{fut}_{t+1..t+168}` heredando el estado final
   del encoder → produce `(φ_{t+1}, ..., φ_{t+168})`.
7. **Static enrichment**: cada `φ_τ` se modula con `c_e` → `θ_τ = GRN(φ_τ, c_e)`.
8. **Multi-head self-attention** (causal) sobre `(θ_{t-167}, ..., θ_{t+168})`
   → produce `(β_{t+1}, ..., β_{t+168})` para los pasos del horizonte, junto
   con la matriz `α` de pesos.
9. **Position-wise FFN + Add & Norm** sobre cada `β_τ`.
10. **Quantile heads**: `(q̂^{0.1}_τ, q̂^{0.5}_τ, q̂^{0.9}_τ)` por cada paso
    `τ` del horizonte — output final `(168, 3)`.
""")

md("""
### 3.4 Supuestos del modelo

| Supuesto | Estado en este proyecto |
|---|---|
| Series con **frecuencia constante** | ✓ INMET es horario estricto, regularizado en `process.py`. |
| Existencia de covariables temporales **conocidas a futuro** | ✓ Las cíclicas (`hour_sin/cos`, `doy_sin/cos`, `month_sin/cos`) son determinísticas y conocidas a cualquier horizonte. |
| Existencia de covariables **estáticas por entidad** | ✓ `lat`, `lng`, `alt`, `region`, `biome`, `koppen_class`, `station_id` — siete features. |
| Suficientes ejemplos por entidad para entrenar embeddings significativos | **Parcial**: ~52 584 horas/estación en train; suficiente. Con la salvedad de A301 y A615 que tienen menos años cubiertos. |
| **Estacionariedad débil** dentro de la ventana de lookback | **Parcial**: el ADF del EDA §5.3 mostró que la serie original no es estacionaria, pero la diferenciación de orden 1 sí; el modelo absorbe la no-estacionariedad vía la normalización por estación + LSTM + atención. |
| **Independencia entre entidades** | **Aproximada**: en realidad las estaciones cercanas correlacionan; el TFT no modela esto explícitamente. Mitigación: el embedding de `region` captura algo de la correlación espacial agregada. |
| **Distribución gaussiana del residuo** | **No**: la quantile loss no asume distribución alguna; supera al MSE en colas asimétricas (eventos extremos). |
""")

md("### 3.5 Hiperparámetros del paper original vs proyecto")

code("""
hp = pd.DataFrame([
    {"hiperparámetro": "hidden_size",
     "valor_paper": "160 (Electricity, Traffic) / variable según dataset",
     "valor_proyecto": "64",
     "justificación": "Default del paper para datasets pequeños; el panel INMET ~2.1 M ejemplos tolera 64 sin sobreajustar; subir a 128 si val plateau."},
    {"hiperparámetro": "attention_head_size",
     "valor_paper": "4",
     "valor_proyecto": "4",
     "justificación": "Mantener; cubre dependencias diaria/semanal/anual con 4 heads independientes."},
    {"hiperparámetro": "dropout",
     "valor_paper": "0.1",
     "valor_proyecto": "0.1",
     "justificación": "Mantener; el panel grande no requiere dropout agresivo."},
    {"hiperparámetro": "learning_rate",
     "valor_paper": "0.001",
     "valor_proyecto": "0.001",
     "justificación": "Default Adam que el paper validó; mantener."},
    {"hiperparámetro": "max_encoder_length (lookback)",
     "valor_paper": "varies (168 en Electricity)",
     "valor_proyecto": "168",
     "justificación": "Justificado por ACF/PACF del EDA §5.4 (autocorr significativa hasta lag 168)."},
    {"hiperparámetro": "max_prediction_length (horizon)",
     "valor_paper": "varies (24 en Electricity)",
     "valor_proyecto": "168 (slices a 24/72/168)",
     "justificación": "Multi-horizonte ambicioso del proyecto; slices reportan métricas en cortes h=24, 72, 168."},
    {"hiperparámetro": "batch_size",
     "valor_paper": "64–256",
     "valor_proyecto": "64",
     "justificación": "64 es robusto; subir a 128 si la GPU lo permite."},
    {"hiperparámetro": "epochs (máx.)",
     "valor_paper": "100 con early stopping",
     "valor_proyecto": "60 con early stopping (patience 10)",
     "justificación": "Reducimos epochs máx por presupuesto computacional; early stopping decide."},
    {"hiperparámetro": "weight_decay",
     "valor_paper": "0.01–0.1 [verificar contra paper original]",
     "valor_proyecto": "1e-4",
     "justificación": "Más conservador; el paper recomienda regularización fuerte para datasets pequeños — el nuestro es grande."},
    {"hiperparámetro": "n_static_categorical",
     "valor_paper": "varies",
     "valor_proyecto": "4",
     "justificación": "station_id, region, biome, koppen_class."},
    {"hiperparámetro": "n_static_real",
     "valor_paper": "varies",
     "valor_proyecto": "3",
     "justificación": "latitude, longitude, altitude."},
    {"hiperparámetro": "quantiles",
     "valor_paper": "[0.1, 0.5, 0.9]",
     "valor_proyecto": "[0.1, 0.5, 0.9]",
     "justificación": "P10/P50/P90 estándar; bandas P10–P90 ≈ 80 % CI."},
])
hp
""")

md("""
### 3.6 Resultados originales reportados

El paper evalúa el TFT sobre **cuatro datasets públicos** reportando
*quantile loss* a P50 y P90 (menor es mejor). Los valores específicos
publicados [verificar contra paper original Tabla 4 / Tabla 5]:

| Dataset | Métrica | TFT | DeepAR | DeepState | MQ-RNN | Notas |
|---|---|---|---|---|---|---|
| **Electricity** | P50 quantile loss | ~0.027 | ~0.075 | ~0.083 | ~0.040 | Mejora ~7 % respecto al mejor competidor |
| **Electricity** | P90 quantile loss | ~0.059 | ~0.040 | ~0.056 | ~0.030 | TFT comparable; gana en P50 |
| **Traffic** | P50 quantile loss | ~0.095 | ~0.161 | ~0.167 | ~0.117 | Mejora ~19 % respecto al mejor competidor |
| **Traffic** | P90 quantile loss | ~0.144 | ~0.099 | ~0.113 | ~0.082 | TFT competitivo |
| **Volatility** | P50 quantile loss | ~0.033 | ~0.039 | ~0.040 | ~0.040 | Mejora ~15 % respecto al mejor competidor |
| **Retail** (M5-equivalent) | sMAPE / quantile loss | mejora consistente sobre baselines clásicos y DL | | | | |

> Los valores numéricos exactos deben **verificarse contra el paper
> original** (Tablas 4–5). El patrón general — TFT supera a baselines en P50
> con margen claro y compite en P90 — es el resultado replicable.

**Resultado adicional reportado**: análisis de **interpretabilidad** sobre
Electricity muestra que el modelo aprendió a darle peso alto a `hour` y a
`day_of_week` (covariables conocidas a futuro), y que la atención temporal se
concentra en lags `−24`, `−48`, `−168` — coherente con la estacionalidad
diaria/semanal del consumo eléctrico. Es exactamente el comportamiento que
esperamos en `temp_c` brasileño (la ACF del EDA §5.4 mostró picos en los
mismos lags).
""")

# ============================================================================
# Sección 4 — Adaptación al proyecto
# ============================================================================
md("## 4. Adaptación al Proyecto")

md("""
### 4.1 Qué se mantiene fiel al paper original

- **Estructura encoder-decoder** con LSTM local + multi-head attention global.
- **Variable Selection Networks** (las tres: estática, pasada, futura).
- **Gated Residual Networks** como bloque universal con skip connection y
  GLU.
- **Static Covariate Encoders** que producen los cuatro contextos `c_s, c_e,
  c_c, c_h` inyectados en cuatro etapas.
- **Static Enrichment Layer** con GRN modulada por contexto estático.
- **Multi-head Interpretable Attention** con promedio de scores entre heads
  (no de outputs concatenados).
- **Quantile Loss multi-percentil** con `Q = {0.1, 0.5, 0.9}`.
- **Causal masking** estricto en la atención.

### 4.2 Qué se adapta

- **Dataset**: el paper usa Electricity/Traffic/Volatility/Retail; aquí
  **INMET brasileño**, frecuencia horaria, 40 estaciones, 8 años, target
  meteorológico continuo (`temp_c`).
- **Covariables estáticas**: el paper usa identificadores opacos de entidad
  (ej. ID de cliente sin metadata externa); aquí enriquecemos con metadata
  **real y semánticamente cargada** — `latitude`, `longitude`, `altitude`,
  `region` (5 macrorregiones IBGE), `biome` (6 biomas), `koppen_class` (9
  clases climáticas), además del `station_id`. Esto debería dar al TFT
  ventaja sobre los baselines.
- **Covariables temporales conocidas a futuro**: cíclicas seno/coseno de hora,
  día del año y mes (ya generadas en `process.py`). El paper usa también
  *holidays* o eventos especiales del calendario; en meteorología no aplica.
- **Lookback / horizonte**: lookback = 168 h y horizon = 168 h con cortes a
  24/72/168; justificados por ACF/PACF y por la longitud máxima sostenible
  con quantile loss sin compounding.
- **Loss**: **MSE para baselines (LSTM/GRU/N-BEATSx/Informer) y quantile loss
  para TFT**. Decisión consciente de mantener la fortaleza del TFT
  (incertidumbre nativa) en lugar de degradarla a punto-estimación. Para
  comparar con los baselines en métricas tipo MSE/RMSE, reportamos **P50
  como la predicción puntual del TFT** y la usamos en RMSE/MAE/R² igual que
  los demás.
- **Hiperparámetros**: `hidden_size=64` (vs 160 del paper) por presupuesto
  GPU; `dropout=0.1` (igual al paper); `weight_decay=1e-4` (más conservador).
  Tabla completa en §3.5.
- **Régimen de entrenamiento**: paper usa **modelo global con un único
  conjunto de pesos** entrenado sobre todas las series; aquí hacemos lo mismo
  (40 estaciones, embeddings nativos del TFT diferencian entre ellas). Esto
  difiere del runner actual que es per-estación — **adaptación pendiente**
  documentada en notebook 04 §6.1.

### 4.3 Qué NO se incorpora del paper

- **Scheduled sampling y curriculum learning**: el paper menciona estrategias
  para estabilizar el entrenamiento del decoder; las omitimos en la primera
  iteración para mantener simplicidad.
- **Ensembles**: el paper en algunos benchmarks ensemble múltiples runs; aquí
  reportamos cada semilla individualmente y la agregación se hace en el
  benchmark (notebook 06) con bootstrap.
- **Data augmentation específica de cada dataset**: irrelevante en
  meteorología.
- **Tuning bayesiano de hiperparámetros**: el paper aplica búsqueda extensiva;
  nosotros usamos los valores recomendados como punto de partida y dejamos
  el tuning bayesiano como extensión futura.
- **Multi-target output**: el paper soporta múltiples targets simultáneos;
  aquí mantenemos univariado (`temp_c`) — la extensión a `[temp_c, humidity,
  pressure]` es una segunda fase.
""")

# ============================================================================
# Sección 5 — Relación con los modelos del benchmark
# ============================================================================
md("## 5. Relación con los Modelos del Benchmark")

md("### 5.1 Tabla comparativa")

code("""
relation = pd.DataFrame([
    {"modelo_benchmark": "Persistencia",
     "similitudes_con_TFT": "Ninguna a nivel de aprendizaje; sí comparten el output multi-horizonte (vector de 168).",
     "diferencias_clave": "Persistencia no aprende; TFT aprende embeddings, atención, VSN, quantile loss. Persistencia es el piso obligatorio."},
    {"modelo_benchmark": "LSTM vanilla",
     "similitudes_con_TFT": "Comparten el LSTM como bloque temporal local (encoder).",
     "diferencias_clave": "TFT añade VSN (selección de variables), GRN (gating), multi-head attention (long-range), embeddings de entidad y quantile loss. LSTM solo aplana al final con un Linear."},
    {"modelo_benchmark": "GRU",
     "similitudes_con_TFT": "Familia recurrente; estructura similar al LSTM dentro del TFT.",
     "diferencias_clave": "GRU usa una celda más simple (3 gates → 2 gates); TFT integra LSTM dentro de un sistema mucho más amplio. GRU no tiene atención ni embeddings."},
    {"modelo_benchmark": "N-BEATSx",
     "similitudes_con_TFT": "Multi-horizonte single-shot; ambos descomponen el pronóstico (TFT vía atención + VSN, N-BEATS vía bases trend/seasonality).",
     "diferencias_clave": "N-BEATS no usa atención ni embeddings nativos. TFT modula el pronóstico con metadatos por entidad; N-BEATS no nativamente."},
    {"modelo_benchmark": "Informer",
     "similitudes_con_TFT": "Ambos son arquitecturas con multi-head attention para forecasting de larga secuencia; ambos manejan 168 h.",
     "diferencias_clave": "Informer privilegia eficiencia (ProbSparse) y no tiene VSN ni embeddings nativos. TFT privilegia interpretabilidad y entidades."},
])
relation
""")

md("""
### 5.2 Discusión

**Frente a los baselines simples (LSTM/GRU)**, el TFT debería ganar
principalmente por dos vías. (i) **Multi-horizonte de mayor calidad**: la
atención multi-head escala a 168 pasos sin la degradación típica de LSTM
puro; el LSTM/GRU comprime toda la historia en un único vector latente y
luego una capa lineal proyecta a 168 — para horizontes largos esto se vuelve
sub-óptimo. (ii) **Aprovechamiento de la heterogeneidad regional**: los
embeddings de `region`/`biome`/`koppen_class` y las features estáticas
`lat/lng/alt` deberían dar al TFT información directa sobre el régimen
climático local; un LSTM/GRU con estas features simplemente concatenadas al
input las trata como cualquier otra variable y dispersa el aprendizaje. La
ganancia esperada es mayor en estaciones de regímenes "exóticos" (Pampa
templado en Sul, Caatinga semi-árida en Nordeste) y menor en Cerrado/Mata
Atlántica donde los baselines también pueden hacerlo bien.

**Frente a Informer**, ambos son Transformers para multi-horizonte largo y
ambos son competitivos en sus dominios. Informer puede **ganar en velocidad
de inferencia** (one-shot decoder más simple, ProbSparse subcuadrática) pero
**pierde en interpretabilidad** (no hay equivalente a las VSN del TFT) y en
manejo nativo de covariables conocidas a futuro. Para `temp_c` brasileño con
metadata rica por estación, esperamos que el TFT iguale o supere
ligeramente a Informer en RMSE y le **gane claramente en interpretabilidad**
— lo cual es importante para la defensa académica.

**Casos donde el TFT podría perder**: si las relaciones son muy locales y
poco contextuales (ej. estaciones sin diferenciación clara de régimen), un
LSTM bien tuneado podría empatar a costo computacional 5–10× menor. También
en estaciones con lookback efectivo corto (donde la dependencia temporal
relevante es < 24 h, lo cual no es nuestro caso según ACF/PACF), el aparato
de atención del TFT es overkill.
""")

# ============================================================================
# Sección 6 — Conexión con EDA
# ============================================================================
md("## 6. Conexión con las Decisiones del EDA")

code("""
eda_link = pd.DataFrame([
    {"decision_EDA": "Estacionalidad diaria + anual fuerte (FFT picos 24h, 12h, 8766h; STL ~50–80 % varianza estacional)",
     "como_TFT_la_honra": "Las cíclicas (`hour_sin/cos`, `doy_sin/cos`, `month_sin/cos`) entran como **'known future inputs'** y la VSN futura les asigna pesos automáticamente. El multi-head attention captura los lags 24/168 nativamente."},
    {"decision_EDA": "Heterogeneidad regional dramática (Norte 22-32 °C plano vs Sul 5-35 °C amplio)",
     "como_TFT_la_honra": "**Static covariates** (`region`, `biome`, `koppen_class`, `lat/lng/alt`) se inyectan vía los 4 contextos estáticos, modulando VSN, LSTM, static-enrichment y output."},
    {"decision_EDA": "Lookback de 168 h justificado por ACF/PACF significativa hasta lag 168",
     "como_TFT_la_honra": "**`max_encoder_length=168`** directamente; la atención causal cubre los 168 pasos sin degradación a diferencia de LSTM puro."},
    {"decision_EDA": "Multi-horizonte simultáneo {24, 72, 168}",
     "como_TFT_la_honra": "**`max_prediction_length=168`** con output single-shot; las métricas se reportan en slices h=24, h=72, h=168 sobre el mismo vector de salida."},
    {"decision_EDA": "Eventos extremos en colas (p01/p99) con flag `is_extreme`",
     "como_TFT_la_honra": "**Quantile loss multi-percentil** (P10/P50/P90) modela las colas explícitamente — no sub-predice como MSE. Las bandas P10–P90 ≈ 80 % CI."},
    {"decision_EDA": "~21 % faltantes en `radiation_kj_m2` (gaps nocturnos legítimos)",
     "como_TFT_la_honra": "La **VSN pasada** aprende a darle peso bajo en pasos nocturnos; ventanas con NaN remanente se descartan en el dataset (no se imputan con 0)."},
    {"decision_EDA": "Exógenas seleccionadas: humidity, pressure, radiation, wind_speed, dew_point",
     "como_TFT_la_honra": "Entran como **'past observed inputs'** a la VSN pasada — la VSN reportará empíricamente cuáles son más relevantes (validación del análisis MI del EDA §6)."},
    {"decision_EDA": "Estandarización por estación (scaler fit en train por wmo)",
     "como_TFT_la_honra": "Compatible: la normalización ocurre upstream, el TFT consume los inputs ya estandarizados; la inversión se hace post-hoc al inverse_transform las predicciones."},
    {"decision_EDA": "Anti-leakage temporal estricto (split por años + ventaneo no cruza fronteras)",
     "como_TFT_la_honra": "Compatible: el TFT consume las ventanas que entrega `make_windows`; la causal mask del attention impide adicionalmente cualquier filtración intra-ventana."},
    {"decision_EDA": "Sesgo de representación (Sudeste/Nordeste con más estaciones)",
     "como_TFT_la_honra": "Mitigable con **`WeightedRandomSampler`** ponderado por región — compatible con la implementación oficial. El embedding de `region` además permite al modelo compensar el sesgo internamente."},
])
eda_link
""")

# ============================================================================
# Sección 7 — Limitaciones y mitigaciones
# ============================================================================
md("""
## 7. Limitaciones del Paper Guía y Mitigaciones

### 7.1 Costo computacional

El TFT tiene ~10⁶ parámetros con `hidden_size=64`, repartidos entre VSN
(varias), GRN (varias), LSTM, multi-head attention y heads cuantílicas. El
entrenamiento es **5–10× más caro** que un LSTM equivalente.

**Mitigación**:
- `hidden_size=64` (vs 160 default del paper) — adecuado para la cantidad de
  datos del proyecto y ahorra GPU.
- `batch_size=64` — más alto si la GPU lo permite (mid-range 8–12 GB).
- Mixed precision (`torch.cuda.amp`) recomendado.
- Si el panel completo (40 estaciones × 5 seeds) es inviable, iterar primero
  en un subset de 5 estaciones representativas (las del EDA §5.1).

### 7.2 Sensibilidad a hiperparámetros

VSN + GRN + atención introducen muchos hiperparámetros (hidden, heads,
dropout en cada bloque). El paper aplicó búsqueda bayesiana extensiva en cada
dataset.

**Mitigación**:
- Usar los valores recomendados por el paper (`hidden=160`, `heads=4`,
  `dropout=0.1`) como punto de partida y sólo ajustar `hidden_size` por
  presupuesto.
- Reportar la **variabilidad entre semillas** (5 corridas) — si la varianza
  es alta, indica sensibilidad y motiva tuning posterior.

### 7.3 Requiere datos suficientes por entidad

Embeddings necesitan ejemplos para ser informativos. ~52 584 horas/estación
es bueno, pero estaciones con baja cobertura (< 90 %, A301 y A615 según EDA
§3) podrían diluir la calidad del embedding.

**Mitigación**:
- Filtrar estaciones con cobertura < 90 % de `config/exclude_stations.yaml`
  (lista declarativa; aún por crear).
- Pesos de muestreo inversamente proporcionales a la cobertura para
  compensar.

### 7.4 Quantile loss menos comparable con baselines MSE

Reportar quantile loss del TFT junto con MSE/RMSE de los baselines complica
la comparación.

**Mitigación**:
- Reportar **P50 (mediana) como predicción puntual del TFT** y calcular
  RMSE/MAE/R² sobre P50 — directamente comparable con los baselines.
- Adicionalmente reportar quantile loss del TFT (P10/P50/P90) como métrica
  exclusiva del modelo probabilístico.

### 7.5 Interpretabilidad ≠ causalidad

Los pesos de las VSN y la atención son **explicaciones del modelo**, no
afirmaciones causales. Un peso alto sobre `humidity_pct` significa que el
modelo lo usa, no que la humedad cause la temperatura.

**Mitigación**:
- Documentar explícitamente esta limitación en la discusión de resultados.
- Cruzar los pesos del TFT con el análisis de **Mutual Information** del
  EDA §6: si VSN y MI coinciden, gana credibilidad la asociación; si no
  coinciden, hay una hipótesis interesante que perseguir.
- No usar el TFT como herramienta causal.

### 7.6 Implementación oficial inestable

El TFT existe en TensorFlow (Google Research) y en `pytorch-forecasting`.
La API de `pytorch-forecasting` ha cambiado entre versiones, y la integración
con `lightning` introduce dependencias adicionales.

**Mitigación**:
- Fijar versión de `pytorch-forecasting` en `requirements.txt`.
- Aislar la integración detrás del wrapper `TFTForecaster` en
  `src/models/model_tft.py` para que el resto del proyecto no se acople.
- Si la dependencia da problemas, portar la arquitectura propia siguiendo
  el paper (ya hay implementaciones en GitHub que se pueden adaptar).
""")

# ============================================================================
# Sección 8 — Síntesis
# ============================================================================
md("""
## 8. Síntesis Ejecutiva del Capítulo

**Paper guía**: Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
*Temporal Fusion Transformers for interpretable multi-horizon time series
forecasting.* International Journal of Forecasting, 37(4), 1748–1764.

**Razón principal**: El TFT es la única arquitectura del Top 9 del SOTA que
satisface simultáneamente las cinco demandas críticas del proyecto —
multi-horizonte nativo, embeddings de entidad, covariables conocidas a
futuro, interpretabilidad y pronóstico probabilístico — todas alineadas con
las decisiones del EDA.

**Aporte arquitectónico**:
- **Variable Selection Networks** que pesan variables por timestep de forma
  interpretable.
- **Static Covariate Encoders** que inyectan metadatos por entidad en cuatro
  etapas distintas del modelo.
- **Multi-head Interpretable Attention** que captura dependencias largas
  produciendo una matriz de atención reportable.

**Adaptaciones al proyecto**:
- Inputs estáticos enriquecidos con metadata real (`region`, `biome`,
  `koppen_class`, `lat/lng/alt`) en lugar de IDs opacos del paper.
- Lookback y horizonte fijados por el EDA (168 h cada uno).
- `hidden_size=64` por presupuesto GPU; `dropout=0.1` y `quantiles=[0.1,
  0.5, 0.9]` se mantienen del paper.

**Expectativa de desempeño relativo**: Esperamos que TFT supere a LSTM/GRU
en horizontes largos (h=72 y h=168) y en estaciones con clima atípico
(Pampa, Caatinga), con paridad o ligera ventaja frente a Informer en
RMSE/MAE y **ventaja clara en interpretabilidad**. N-BEATSx queda como
tercer competidor fuerte; un escenario plausible es que se aproxime al TFT
en RMSE puntual pero pierda en colas (P10/P90).

**Próximo paso**: notebook 06 (`06_benchmark_final.ipynb`) — análisis
estadístico riguroso del benchmark final con Diebold-Mariano por pares,
Friedman + Nemenyi/Bonferroni/Holm para múltiples modelos, Wilcoxon
signed-rank sobre errores pareados, IC 95 % vía bootstrap, y reporte por
horizonte y por región.

---

*Fin del capítulo 05. La implementación específica del TFT se difiere a la
fase de modelado posterior; este capítulo documenta el contrato técnico
que esa implementación debe respetar.*
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

out_path = Path("notebooks/05_guide_paper.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook escrito: {out_path} ({len(cells)} celdas)")
