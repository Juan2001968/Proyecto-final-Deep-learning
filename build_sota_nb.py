"""Construye notebooks/03_state_of_the_art.ipynb (sin ejecutar)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ============================================================================
# Celda 0 — Título y resumen
# ============================================================================
md("""
# Estado del Arte: Modelos de Deep Learning para Forecasting Meteorológico Multi-horizonte

**Resumen ejecutivo.** Este capítulo revisa el estado del arte en modelos de
*deep learning* aplicados al pronóstico meteorológico multi-horizonte sobre
paneles de estaciones, con énfasis en la predicción de **temperatura del aire
(`temp_c`)** a 24 h, 72 h y 168 h sobre el panel nacional brasileño (40 estaciones
INMET, 5 macrorregiones IBGE, 2018–2025). Partiendo de los **5 papers IEEE Q1**
ya documentados por los autores, ampliamos la revisión con **4 trabajos
adicionales** que cubren huecos arquitectónicos críticos para este proyecto:
*Temporal Fusion Transformer*, *Informer*, *N-BEATS* y *DeepAR*. El objetivo es
identificar un *Top 9* representativo, justificar técnicamente la selección
final de **5–6 modelos a implementar** en el benchmark, y elegir un **paper
guía** que vertebre las decisiones arquitectónicas del proyecto.
""")

# ============================================================================
# Sección 1 — Estrategia de búsqueda
# ============================================================================
md("""
## 1. Estrategia de Búsqueda

### 1.1 Palabras clave

| Categoría | Términos retenidos | Términos descartados (con motivo) |
|---|---|---|
| Tarea | "weather forecasting", "temperature prediction", "multi-horizon time series forecasting", "panel forecasting" | "climate projection" — escala de décadas, fuera de alcance horario |
| Arquitectura | "deep learning", "LSTM", "Transformer time series", "ConvLSTM", "attention mechanism", "N-BEATS", "Temporal Fusion Transformer" | "GAN" — generación, no forecasting puntual |
| Datos | "automatic weather station", "meteorological station network", "INMET", "ASOS" | "satellite imagery", "radar" — modalidades fuera del enfoque tabular del proyecto |
| Métricas | "RMSE", "MAE", "Diebold-Mariano" | — |

Las cadenas finales de búsqueda más productivas combinaron:

- `("deep learning" OR "LSTM" OR "Transformer") AND ("weather" OR "temperature") AND ("multi-step" OR "multi-horizon")`
- `("attention" AND "LSTM") AND "meteorological"`
- `("Temporal Fusion Transformer" OR "Informer" OR "N-BEATS") AND "forecasting"`

### 1.2 Bases consultadas

1. **IEEE Xplore** — fuente primaria; los 5 papers de la revisión inicial provienen de aquí.
2. **Web of Science** — confirmación de Q1 / cuartiles JCR para los papers IEEE.
3. **Scopus** — verificación de citas y métricas de impacto.
4. **Google Scholar** — complemento para identificar trabajos seminales no IEEE
   (TFT, Informer, N-BEATS, DeepAR) y para tracking de citas.
5. **arXiv** — preprints citables de venues como NeurIPS, ICLR y AAAI cuando
   el publisher no asigna DOI tradicional (ej. proceedings de NeurIPS).

### 1.3 Periodo de búsqueda

**2019–2025** (últimos 7 años). Esto cubre:

- el auge post-Transformer (2017) en series de tiempo, con los primeros papers
  serios de aplicación meteorológica desde 2019;
- el período donde aparecen las arquitecturas específicas para forecasting
  (TFT 2021, Informer 2021, Autoformer 2021, PatchTST 2023);
- los trabajos del PDF original cubren 2020–2024, todos dentro del rango.

### 1.4 Criterios de inclusión

1. Aplicación directa a forecasting meteorológico **multivariado** (temperatura,
   precipitación, humedad o equivalentes).
2. Uso de **deep learning** como núcleo del modelo (no como simple postproceso).
3. Métricas cuantitativas reportadas (RMSE, MAE, R², CRPS o equivalentes).
4. Preferencia por **revistas Q1** (JCR) o conferencias *tier-1* (NeurIPS, ICLR,
   AAAI, ICML) para los trabajos arquitectónicos generales.
5. Disponibilidad pública del código o pseudocódigo suficiente para reproducción.

### 1.5 Criterios de exclusión

1. Papers de **remote sensing** puro (imágenes satelitales) sin componente
   temporal en serie de estaciones.
2. Papers de **NWP físico** (modelos numéricos basados en ecuaciones primitivas)
   sin componente DL.
3. Trabajos de **modelado climático** a escala anual o decadal (fuera del
   horizonte horario-diario del proyecto).
4. Papers que reportan únicamente *case studies* sin baseline ni split temporal
   honesto.
""")

# ============================================================================
# Sección 2 — Identificación del Top
# ============================================================================
md("""
## 2. Identificación del Top de Modelos

A los **5 papers IEEE Q1** de la revisión inicial sumamos **4 trabajos**
adicionales que llenan huecos arquitectónicos del PDF original:

1. **Khan & Maity (2020)** — Conv1D-MLP híbrido para precipitación diaria.
2. **Tekin, Fazla & Kozat (2024)** — ConvLSTM + atención + context matcher.
3. **Suleman & Shridevi (2022)** — SFA-LSTM con atención sobre variables.
4. **Sharma et al. (2023)** — DL para lluvia intensa en terreno complejo.
5. **Trivedi, Sharma & Pattnaik (2024)** — DL para minimización de error de
   pronóstico de lluvia fuerte.

**Adiciones** (justificación abajo):

6. **Lim et al. (2021), Temporal Fusion Transformer (TFT)** — Multi-horizonte
   nativo con embeddings de entidad. Encaja con el panel de 40 estaciones y
   cubre el hueco de "Transformers específicos para forecasting" que el PDF no
   tenía.
7. **Zhou et al. (2021), Informer** — Atención dispersa (`ProbSparse`) para
   secuencias largas. Cubre el horizonte de 168 h (7 días) que es ambicioso
   para LSTM/ConvLSTM clásicos.
8. **Oreshkin et al. (2020), N-BEATS** — Bloques de basis interpretables;
   *baseline fuerte* en forecasting puro. Cubre el hueco de modelos no
   recurrentes y no-Transformer competitivos.
9. **Salinas et al. (2020), DeepAR** — Modelo global probabilístico con
   embeddings de entidad. Cubre el hueco de pronóstico **probabilístico** sobre
   panel — útil para reportar incertidumbre en eventos extremos.

### Por qué estos 4 y no los del PDF únicamente

El PDF del usuario se centra en **precipitación** y **modelos espacio-temporales
con grilla regular** (ConvLSTM). Para nuestro proyecto (temperatura horaria
multi-horizonte sobre panel disperso de estaciones), los huecos son:

- ❌ **Sin Transformer puro** para forecasting → cubierto con TFT, Informer.
- ❌ **Sin modelo global con embeddings de entidad** → cubierto con DeepAR, TFT.
- ❌ **Sin baseline no-recurrente competitivo** → cubierto con N-BEATS.
- ❌ **Sin pronóstico probabilístico** → cubierto con DeepAR (gaussiana) y TFT
  (quantile loss).

### Tabla resumen del Top 9
""")

code("""
import pandas as pd

top_models = pd.DataFrame([
    {
        "#": 1, "año": 2020, "autores": "Khan & Maity",
        "arquitectura": "Conv1D + MLP (híbrido)",
        "tipo_problema": "Forecasting precipitación diaria multi-step",
        "dataset": "GCM simulations (Khordha, India)",
        "metrica_clave": "RMSE, NSE",
        "Q1": "sí (IEEE Access)",
        "en_PDF": "sí",
    },
    {
        "#": 2, "año": 2024, "autores": "Tekin, Fazla & Kozat",
        "arquitectura": "ConvLSTM + atención + context matcher",
        "tipo_problema": "Forecasting espacio-temporal NWF",
        "dataset": "ERA5 reanálisis (grilla)",
        "metrica_clave": "MSE, MAE",
        "Q1": "sí (IEEE T-GRS)",
        "en_PDF": "sí",
    },
    {
        "#": 3, "año": 2022, "autores": "Suleman & Shridevi",
        "arquitectura": "SFA-LSTM (encoder-decoder + atención)",
        "tipo_problema": "Forecasting temperatura corto plazo",
        "dataset": "Estación única (multivariado)",
        "metrica_clave": "RMSE, MAE",
        "Q1": "sí (IEEE Access)",
        "en_PDF": "sí",
    },
    {
        "#": 4, "año": 2023, "autores": "Sharma et al.",
        "arquitectura": "DL distrital (CNN/RNN)",
        "tipo_problema": "Forecasting lluvia intensa terreno complejo",
        "dataset": "NE India, escala distrital",
        "metrica_clave": "POD, FAR, CSI",
        "Q1": "sí (IEEE T-GRS)",
        "en_PDF": "sí",
    },
    {
        "#": 5, "año": 2024, "autores": "Trivedi, Sharma & Pattnaik",
        "arquitectura": "DL como post-procesador",
        "tipo_problema": "Reducción de error en pronóstico real-time",
        "dataset": "Assam, eventos lluvia fuerte",
        "metrica_clave": "RMSE, bias",
        "Q1": "sí (IEEE GRSL)",
        "en_PDF": "sí",
    },
    {
        "#": 6, "año": 2021, "autores": "Lim, Arık, Loeff & Pfister",
        "arquitectura": "Temporal Fusion Transformer",
        "tipo_problema": "Forecasting multi-horizonte multivariado",
        "dataset": "Electricity, Traffic, Volatility, Retail",
        "metrica_clave": "Quantile loss (P50/P90)",
        "Q1": "sí (Int. J. Forecasting)",
        "en_PDF": "no",
    },
    {
        "#": 7, "año": 2021, "autores": "Zhou et al. (Informer)",
        "arquitectura": "Transformer + ProbSparse self-attention",
        "tipo_problema": "Long sequence time-series forecasting",
        "dataset": "ETT, ECL, Weather, ILI",
        "metrica_clave": "MSE, MAE",
        "Q1": "AAAI 2021 (best paper)",
        "en_PDF": "no",
    },
    {
        "#": 8, "año": 2020, "autores": "Oreshkin et al. (N-BEATS)",
        "arquitectura": "Stack de bloques fully-connected con basis",
        "tipo_problema": "Forecasting univariado y multi-step",
        "dataset": "M3, M4, Tourism",
        "metrica_clave": "sMAPE, MASE",
        "Q1": "ICLR 2020",
        "en_PDF": "no",
    },
    {
        "#": 9, "año": 2020, "autores": "Salinas et al. (DeepAR)",
        "arquitectura": "RNN autoregresivo probabilístico",
        "tipo_problema": "Forecasting global probabilístico (panel)",
        "dataset": "Electricity, Traffic, Wikipedia, M4",
        "metrica_clave": "ND, NRMSE, P50/P90 quantile loss",
        "Q1": "sí (Int. J. Forecasting)",
        "en_PDF": "no",
    },
])
top_models
""")

# ============================================================================
# Sección 3 — Análisis detallado por modelo
# ============================================================================
md("## 3. Análisis Detallado por Modelo")

# 3.1 Khan & Maity
md("""
### 3.1 Khan & Maity (2020) — Conv1D-MLP híbrido

#### 3.1.1 Referencia académica

Khan, M. I., & Maity, R. (2020). *Hybrid Deep Learning Approach for Multi-Step-Ahead
Daily Rainfall Prediction Using GCM Simulations.* **IEEE Access, 8**, 52774–52784.
https://doi.org/10.1109/ACCESS.2020.2980977

#### 3.1.2 Tipo de arquitectura

Híbrido **Conv1D + MLP** — convolución 1D extrae patrones locales sobre la ventana
de variables exógenas, luego un MLP profundo proyecta la representación a un
horizonte multi-paso (1–5 días).

```
[ventana 9 vars × T pasos] → Conv1D(filtros) → flatten → MLP(hidden) → ŷ_{t+1..t+5}
```

#### 3.1.3 Descripción técnica

- **Componentes**: bloque convolucional 1D para capturar correlaciones cruzadas
  intra-ventana entre las 9 variables meteorológicas; cabeza MLP densa que
  aprende la regresión multi-paso.
- **Flujo de datos**: input 9 features × T → conv1D → activación → flatten → MLP
  con varias capas → vector de 5 predicciones (un día por salida).
- **Innovación**: combinación específica para precipitación diaria sobre
  simulaciones GCM, demostrando ganancia frente a MLP profundo y SVR como
  baselines clásicos.

#### 3.1.4 Problema que aborda

Forecasting **diario** de **precipitación** a horizonte **1–5 días**, alimentado
con simulaciones de modelos de circulación general (GCM).

#### 3.1.5 Datasets utilizados en el paper original

Salidas de un GCM sobre **Khordha (India)**, frecuencia diaria, periodo histórico
multi-decadal, 9 variables meteorológicas.

#### 3.1.6 Métricas reportadas

- **RMSE** y **NSE** (Nash-Sutcliffe Efficiency).
- Mejora reportada del híbrido frente a MLP profundo y SVR; el desempeño
  decrece con el horizonte, como es esperado.

#### 3.1.7 Fortalezas

- Arquitectura simple, **rápida de entrenar**, baja huella computacional.
- Conv1D captura interacciones de corto plazo entre variables sin necesidad de
  recurrencia.
- Sirve como **baseline competitivo** frente a modelos no-DL.

#### 3.1.8 Limitaciones

- **Frecuencia diaria** (no horaria) ⇒ ventanas cortas; pierde la riqueza del
  ciclo diurno presente en datos horarios.
- **Variable target precipitación**, no temperatura → distribución asimétrica
  que no se traduce directamente al regimen de `temp_c`.
- **Una sola estación**, no panel; sin embeddings de entidad.
- Sin manejo explícito de la estacionalidad anual.

#### 3.1.9 Complejidad computacional

Modelo pequeño (~10⁴–10⁵ parámetros). Entrenamiento factible en CPU.

#### 3.1.10 Aplicabilidad a este proyecto

- **Encaje con el panel de 40 estaciones**: bajo. Requiere replicar el modelo
  por estación o reformularlo con embeddings — no es nativo.
- **Multi-horizonte 24/72/168**: la cabeza MLP escala bien al cambio de
  horizonte, pero a 168 pasos la salida densa se vuelve subóptima.
- **Heterogeneidad regional**: no la maneja nativamente; sin embeddings de
  región/bioma/Köppen quedaría sub-óptimo en regiones contrastantes (Norte vs Sul).
- **Hallazgos del EDA**: la fuerte estacionalidad diaria/anual detectada por
  FFT y STL no la capturaría sin codificación cíclica explícita (que sí está
  en `process.py`). El paper no aporta tratamiento explícito de outliers ni de
  eventos extremos.

**Conclusión**: útil como **baseline conceptual de modelo híbrido**, pero
**no se implementará** porque su frecuencia diaria y su falta de embeddings
lo dejan fuera del régimen del proyecto.
""")

# 3.2 Tekin et al.
md("""
### 3.2 Tekin, Fazla & Kozat (2024) — ConvLSTM + atención + context matcher

#### 3.2.1 Referencia académica

Tekin, S. F., Fazla, A., & Kozat, S. S. (2024). *Numerical Weather Forecasting
Using Convolutional-LSTM With Attention and Context Matcher Mechanisms.*
**IEEE Transactions on Geoscience and Remote Sensing, 62**.
https://doi.org/10.1109/TGRS.2024.3409084

#### 3.2.2 Tipo de arquitectura

**ConvLSTM** (LSTM con celdas convolucionales) + **mecanismo de atención** +
**context matcher** (módulo que alinea contextos espacio-temporales heterogéneos).

```
input grid (T, H, W, C) → ConvLSTM stack → attention → context matcher → output grid
```

#### 3.2.3 Descripción técnica

- **ConvLSTM** reemplaza los productos matriciales de un LSTM clásico por
  convoluciones 2D, preservando la estructura espacial en cada paso temporal.
- **Atención** sobre los estados ocultos de la ConvLSTM para enfatizar pasos
  o regiones más informativos.
- **Context matcher** aprende a alinear correlaciones cruzadas entre dominios
  espacio-temporales con resoluciones distintas (innovación principal del
  paper).

#### 3.2.4 Problema que aborda

Pronóstico meteorológico numérico (NWF) de variables espacio-temporales sobre
**grillas regulares de alta resolución**.

#### 3.2.5 Datasets utilizados en el paper original

Reanálisis tipo **ERA5** (grilla 0.25°) con múltiples variables atmosféricas.

#### 3.2.6 Métricas reportadas

**MSE** y **MAE** sobre la grilla; mejora frente a baselines ConvLSTM puros y
modelos U-Net temporales.

#### 3.2.7 Fortalezas

- Captura **explícitamente** la estructura espacial (vital cuando los datos
  vienen en grilla regular).
- Atención da un grado de interpretabilidad sobre regiones/tiempos relevantes.
- Robusto a horizontes medios.

#### 3.2.8 Limitaciones

- **Requiere grilla regular**: incompatible con un panel de **estaciones
  dispersas** sin pre-interpolación (que introduciría sesgo de smoothing).
- Costo computacional alto: convolución 2D × T pasos.
- Sin embeddings de entidad (no fue diseñado para panel disperso).

#### 3.2.9 Complejidad computacional

Alta — del orden de 10⁶–10⁷ parámetros, requiere GPU para entrenamiento.

#### 3.2.10 Aplicabilidad a este proyecto

- **Encaje con el panel de 40 estaciones**: **bajo**. El INMET son estaciones
  puntuales irregulares, no una grilla. Una pre-interpolación a grilla
  introduciría artefactos en regiones donde la densidad de estaciones es muy
  baja (Norte, Pantanal).
- **Multi-horizonte 24/72/168**: posible pero costoso a 168 pasos.
- **Heterogeneidad regional**: gestionada implícitamente vía la convolución
  espacial, pero pierde semántica (región/bioma/Köppen no se representan).
- **Hallazgos del EDA**: la heterogeneidad regional **dramática** detectada
  (Norte ~22-32 °C plano vs Sul 5-35 °C amplio) se captura mejor con
  embeddings de entidad que con convolución en grilla.

**Conclusión**: **no se implementará** — la naturaleza del panel disperso es
incompatible con la inductive bias de ConvLSTM. El paper queda como antecedente
teórico para el componente **espacio-temporal** del estudio.
""")

# 3.3 Suleman & Shridevi
md("""
### 3.3 Suleman & Shridevi (2022) — SFA-LSTM

#### 3.3.1 Referencia académica

Suleman, M. A. R., & Shridevi, S. (2022). *Short-Term Weather Forecasting Using
Spatial Feature Attention Based LSTM Model.* **IEEE Access, 10**, 82456–82468.
https://doi.org/10.1109/ACCESS.2022.3196381

#### 3.3.2 Tipo de arquitectura

**SFA-LSTM** — *Spatial Feature Attention LSTM* en arquitectura
encoder-decoder con atención sobre el eje de variables.

```
Encoder LSTM → context con atención por variable → Decoder LSTM → ŷ_{t+1..t+H}
```

#### 3.3.3 Descripción técnica

- **Encoder**: LSTM que comprime la ventana histórica multivariada en una
  representación latente.
- **Atención sobre variables (feature attention)**: aprende qué variables
  meteorológicas son más relevantes para predecir `temp_c` en cada paso.
- **Decoder**: LSTM que genera la secuencia de predicciones a horizonte H.
- **Innovación**: la atención no es temporal (sobre pasos pasados) sino
  **sobre variables**, dando interpretabilidad respecto a qué exógenas
  conducen el pronóstico.

#### 3.3.4 Problema que aborda

Forecasting de **temperatura** corto plazo (horario) en una estación,
multivariado.

#### 3.3.5 Datasets utilizados en el paper original

Estación meteorológica única, frecuencia horaria, variables meteorológicas
estándar (temp, hum, presión, viento, radiación).

#### 3.3.6 Métricas reportadas

- **RMSE** y **MAE** sobre temperatura.
- Mejora frente a LSTM vanilla y arquitecturas seq2seq sin atención.

#### 3.3.7 Fortalezas

- **Interpretable** — el peso de atención por variable revela la importancia
  relativa (humedad, presión, etc.) para predecir temperatura.
- Diseñado **específicamente para temperatura** corto plazo, alineado con
  nuestro target.
- Costo moderado, entrenable en hardware estándar.

#### 3.3.8 Limitaciones

- **Una sola estación**: el paper no demuestra generalización a panel.
- Atención solo sobre variables, **no sobre el eje temporal** (un Transformer
  o TFT sí cubren ambos ejes).
- Sin embeddings de entidad → para usarlo en panel hay que entrenar 40 modelos
  o adaptarlo.

#### 3.3.9 Complejidad computacional

Modelo mediano, ~10⁵–10⁶ parámetros, entrenable en GPU consumer.

#### 3.3.10 Aplicabilidad a este proyecto

- **Encaje con el panel**: medio. Funciona como modelo **por estación**, pero
  perdemos la transferencia entre estaciones similares.
- **Multi-horizonte 24/72/168**: el decoder seq2seq escala razonablemente al
  horizonte, aunque con degradación a 168 pasos típica de los LSTM.
- **Heterogeneidad regional**: si se entrena una vez por estación,
  intrínsecamente respeta la heterogeneidad; si se entrena global, requiere
  añadir embeddings.
- **Hallazgos del EDA**: el análisis de correlaciones (Sección 6 del EDA)
  identificó humedad/presión/radiación/viento como exógenas clave — la
  atención sobre variables del SFA-LSTM **se prestaría perfectamente** para
  validar empíricamente el ranking de importancia que predijo MI.

**Conclusión**: **se implementará una variante** (LSTM con atención sobre
variables) como representante de la familia recurrente con interpretabilidad,
adaptada para usar embeddings de estación en lugar de un modelo por estación.
""")

# 3.4 Sharma et al.
md("""
### 3.4 Sharma et al. (2023) — DL distrital para lluvia intensa

#### 3.4.1 Referencia académica

Sharma, O., Trivedi, D., Pattnaik, S., Hazra, V., & Puhan, N. B. (2023).
*Improvement in District Scale Heavy Rainfall Prediction Over Complex Terrain
of North East India Using Deep Learning.* **IEEE Transactions on Geoscience and
Remote Sensing, 61**, 1–8.
https://doi.org/10.1109/TGRS.2023.3322676

#### 3.4.2 Tipo de arquitectura

DL específico (CNN/RNN combinadas) para predicción de **eventos extremos**
(lluvia intensa) a escala distrital.

#### 3.4.3 Descripción técnica

- Combinación de extractores espaciales (CNN) con módulo temporal recurrente.
- Énfasis en la **identificación de eventos extremos** (no en la regresión
  promedio).
- Métricas categóricas además de regresión continua.

#### 3.4.4 Problema que aborda

Detección y cuantificación de **lluvia intensa** (>50 mm/día u umbral
equivalente) a escala distrital en el noreste de la India.

#### 3.4.5 Datasets utilizados en el paper original

Datos meteorológicos de NE India (terreno complejo: Himalaya), escala
distrital, periodo plurianual.

#### 3.4.6 Métricas reportadas

- **POD** (Probability of Detection), **FAR** (False Alarm Rate), **CSI**
  (Critical Success Index).
- RMSE y bias para la regresión continua.

#### 3.4.7 Fortalezas

- Foco específico en **eventos extremos**, alineado con la necesidad del
  proyecto de detectar olas de calor / heladas (`temp_c < p01` o `> p99`).
- Demuestra que DL **mejora la detección de eventos raros** frente a
  baselines numéricos.

#### 3.4.8 Limitaciones

- **Variable target precipitación**, no temperatura.
- Foco en clasificación de evento + regresión, no multi-horizonte puro.
- Geografía y régimen climático (monzónico, terreno alpino) muy distintos
  del contexto brasileño.

#### 3.4.9 Complejidad computacional

Alta — modelo combinado entrenado en GPU, dataset extenso.

#### 3.4.10 Aplicabilidad a este proyecto

- **Encaje con el panel**: bajo en arquitectura, **alto en filosofía
  metodológica** sobre eventos extremos.
- **Multi-horizonte 24/72/168**: no aplicable directamente — el paper se centra
  en horizonte de un día con foco en evento sí/no.
- **Hallazgos del EDA**: la idea de **flag `is_extreme`** y de evaluación
  condicionada a colas que propusimos en el EDA está alineada con esta
  filosofía. Los **POD/FAR/CSI** son adaptables al régimen de temperatura
  como métricas auxiliares de detección de heladas/olas de calor.

**Conclusión**: **no se implementa la arquitectura**, pero **se adoptan sus
métricas POD/FAR/CSI** para reportar desempeño en eventos extremos de `temp_c`
condicionados al flag `is_extreme`.
""")

# 3.5 Trivedi et al.
md("""
### 3.5 Trivedi, Sharma & Pattnaik (2024) — DL como post-procesador

#### 3.5.1 Referencia académica

Trivedi, D., Sharma, O., & Pattnaik, S. (2024). *Minimization of Forecast Error
Using Deep Learning for Real-Time Heavy Rainfall Events Over Assam.* **IEEE
Geoscience and Remote Sensing Letters, 21**, 1–4.
https://doi.org/10.1109/LGRS.2024.3378517

#### 3.5.2 Tipo de arquitectura

**DL como post-procesador / corrector de bias** sobre salidas de modelos
meteorológicos numéricos operativos (NWP).

#### 3.5.3 Descripción técnica

- Toma salidas crudas de un sistema NWP operativo y aprende un mapping
  correctivo a observaciones reales.
- DL aprende el sesgo y los errores sistemáticos del NWP físico.

#### 3.5.4 Problema que aborda

Reducción del error de pronóstico de **lluvias fuertes en tiempo real** sobre
Assam (NE India).

#### 3.5.5 Datasets utilizados en el paper original

Salidas de NWP operativo + observaciones de estaciones, eventos de lluvia
fuerte.

#### 3.5.6 Métricas reportadas

**RMSE**, **bias**, mejora en localización/cantidad/severidad.

#### 3.5.7 Fortalezas

- Demuestra que DL **complementa** y no necesariamente reemplaza al NWP.
- Bajo costo de inferencia (modelo correctivo es pequeño).

#### 3.5.8 Limitaciones

- Depende de tener un **NWP operativo** disponible — no aplica si no hay
  fuente externa.
- Problema lluvia, no temperatura.

#### 3.5.9 Complejidad computacional

Baja — el corrector se entrena rápido al ser un modelo pequeño.

#### 3.5.10 Aplicabilidad a este proyecto

- **Encaje con el panel**: bajo en este momento — no estamos integrando
  salidas NWP externas.
- **Hallazgos del EDA**: el paper sugiere una **fase 2 del proyecto**:
  integrar salidas de modelos numéricos operativos (ej. GFS, ECMWF) como
  features exógenas adicionales, donde DL aprenda el residuo. Esto es
  consistente con la decisión del EDA de mantener el panel observacional puro
  como baseline y dejar la integración multi-fuente como extensión.

**Conclusión**: **no se implementa** en esta entrega; se cita como **dirección
de extensión futura** (post-procesado de NWP operativo brasileño).
""")

# 3.6 TFT
md("""
### 3.6 Lim, Arık, Loeff & Pfister (2021) — Temporal Fusion Transformer (TFT)

#### 3.6.1 Referencia académica

Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion
Transformers for interpretable multi-horizon time series forecasting.*
**International Journal of Forecasting, 37**(4), 1748–1764.
https://doi.org/10.1016/j.ijforecast.2021.03.012

#### 3.6.2 Tipo de arquitectura

**Temporal Fusion Transformer**: arquitectura híbrida que combina LSTM (encoder
local), self-attention multi-head (encoder global) y *gating networks* para
selección de features. Soporta nativamente multi-horizonte y múltiples tipos
de inputs (estáticos, conocidos a futuro, observados pasados).

```
[Static covariates] -> Static encoder -> context vectors
[Known future + past observed + past known] -> VSN -> LSTM encoder -> Self-attention
                                                              -> Gated residual + Quantile heads -> ŷ_{t+1..t+H} (P10/P50/P90)
```

#### 3.6.3 Descripción técnica

- **Variable Selection Networks (VSN)**: gating que aprende qué variables son
  relevantes en cada timestep — interpretabilidad nativa.
- **Static covariate encoders**: integran metadatos invariantes en el tiempo
  (en nuestro caso: `station_id`, `region`, `biome`, `koppen_class`, `lat/lng/alt`).
- **LSTM encoder/decoder**: procesa secuencias locales con atención de corto
  plazo.
- **Multi-head self-attention**: captura dependencias largas (cubre nuestros
  168 h de lookback).
- **Quantile loss multi-percentil** (P10/P50/P90): genera intervalos de
  confianza nativos.
- **Innovación**: primer modelo que unifica de forma elegante *static + known
  future + past known + past observed*, con interpretabilidad y multi-horizon
  nativo.

#### 3.6.4 Problema que aborda

**Forecasting multi-horizonte multivariado** con múltiples entidades, con
soporte explícito para covariables conocidas a futuro (ej. hora del día,
día de la semana — cíclicas que ya tenemos).

#### 3.6.5 Datasets utilizados en el paper original

Electricity, Traffic, Volatility, Retail (M5). Demuestra superioridad sobre
DeepAR, DeepState, MQ-RNN y métodos clásicos en *quantile loss*.

#### 3.6.6 Métricas reportadas

- **Quantile loss** P50 y P90.
- Mejora promedio del 7–25 % frente a DeepAR según dataset.

#### 3.6.7 Fortalezas

- **Multi-horizonte nativo** (vector de output con cuantiles).
- **Embeddings de entidad nativos** (los static covariates son ciudadanos de
  primera clase).
- **Interpretabilidad**: VSN da pesos por variable, atención da pesos por
  paso temporal.
- **Pronóstico probabilístico** (cuantiles).
- Maneja inputs heterogéneos (estáticos / conocidos futuros / pasados).

#### 3.6.8 Limitaciones

- **Costo de entrenamiento**: alto, requiere GPU; hiperparámetros sensibles
  (heads, hidden size, dropout).
- Implementación oficial (`pytorch-forecasting`) puede tener APIs cambiantes.
- Demanda más datos para no sobreajustar el componente atención.

#### 3.6.9 Complejidad computacional

~10⁶ parámetros típico; entrenamiento en horas con GPU mid-range para datasets
de tamaño similar al nuestro (~2 M ejemplos crudos).

#### 3.6.10 Aplicabilidad a este proyecto

- **Encaje con el panel de 40 estaciones**: **óptimo**. Los `static covariates`
  encajan perfecto con `station_id`, `region`, `biome`, `koppen_class`,
  `lat/lng/alt`.
- **Multi-horizonte 24/72/168**: **nativo** — produce los 168 pasos en una
  sola pasada y permite evaluar cortes en 24/72/168.
- **Heterogeneidad regional**: gestionada vía embeddings + atención que
  aprende a ponderar regiones distintas.
- **Hallazgos del EDA**:
  - estacionalidad fuerte → known-future inputs cíclicos (ya generados en
    `process.py`) entran nativamente;
  - exógenas seleccionadas → past observed inputs;
  - eventos extremos → quantile loss P10/P90 captura las colas; el desempeño
    en colas se reporta directamente.
- **Lookback 168 h**: la atención multi-head escala bien a 168 pasos sin la
  caída de performance típica de LSTM puro.

**Conclusión**: **se implementa** y se elige como **paper guía** (Sección 6).
""")

# 3.7 Informer
md("""
### 3.7 Zhou et al. (2021) — Informer

#### 3.7.1 Referencia académica

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021).
*Informer: Beyond efficient transformer for long sequence time-series
forecasting.* **Proceedings of the AAAI Conference on Artificial Intelligence,
35**(12), 11106–11115. arXiv:2012.07436. (AAAI 2021 Best Paper Award.)

#### 3.7.2 Tipo de arquitectura

**Transformer encoder-decoder** con tres innovaciones específicas:
**ProbSparse self-attention** (atención dispersa), **self-attention distilling**
(reducción de la longitud de la secuencia entre capas), y **generative-style
decoder** (predice la salida completa en una pasada, no autoregresivamente).

```
input (T_in pasos) -> encoder (ProbSparse + distilling) -> decoder (one-shot) -> ŷ_{t+1..t+H}
```

#### 3.7.3 Descripción técnica

- **ProbSparse self-attention**: O(L log L) en lugar de O(L²) del Transformer
  vanilla — viable para secuencias muy largas.
- **Distilling**: entre capas del encoder, reduce la longitud por convolución
  + max-pooling, amortiza el costo cuadrático residual.
- **Generative decoder**: predice todo el horizonte en una pasada (sin
  autoregresión paso a paso) ⇒ inferencia rápida y sin error compounding.
- **Innovación**: primer Transformer práctico para horizontes largos
  (cientos de pasos).

#### 3.7.4 Problema que aborda

**Long sequence time-series forecasting** (LSTF) — horizontes de 96 a 720
pasos sobre series multivariadas.

#### 3.7.5 Datasets utilizados en el paper original

ETT (Electricity Transformer Temperature, **¡muy similar a nuestro target!**),
ECL (Electricity Consuming Load), Weather, ILI (Influenza-Like Illness).

#### 3.7.6 Métricas reportadas

**MSE** y **MAE**. Mejora ~30–50 % frente a Transformer vanilla, LSTM y
LogTrans en horizontes largos.

#### 3.7.7 Fortalezas

- **Diseñado para horizontes largos** (168 h y más).
- Inferencia **one-shot** evita error compounding.
- Eficiencia computacional (sub-cuadrática en la longitud).

#### 3.7.8 Limitaciones

- Sin embeddings de entidad nativos — hay que añadirlos como features
  estáticas concatenadas (no es ciudadano de primera clase como en TFT).
- Sensibilidad a hiperparámetros (factor de muestreo en ProbSparse).
- Sin pronóstico probabilístico nativo (sólo punto).

#### 3.7.9 Complejidad computacional

~10⁶–10⁷ parámetros; entrenamiento en GPU.

#### 3.7.10 Aplicabilidad a este proyecto

- **Encaje con el panel**: medio-alto. Soporta multi-variable; los embeddings
  de estación se concatenan a la entrada.
- **Multi-horizonte 24/72/168**: **óptimo** — fue diseñado para esto. El
  generative decoder predice 168 pasos en una pasada.
- **Heterogeneidad regional**: gestionada vía features estáticas concatenadas
  (sub-óptimo vs TFT pero funcional).
- **Hallazgos del EDA**: el dataset ETT del paper es **literalmente
  temperatura horaria** — hay reportes públicos de Informer sobre forecasting
  de temperatura que sirven de referencia para hyperparametrización.

**Conclusión**: **se implementa** como segundo Transformer del benchmark, para
contrastar contra TFT en eficiencia y calidad multi-horizonte.
""")

# 3.8 N-BEATS
md("""
### 3.8 Oreshkin et al. (2020) — N-BEATS

#### 3.8.1 Referencia académica

Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). *N-BEATS:
Neural basis expansion analysis for interpretable time series forecasting.*
**International Conference on Learning Representations (ICLR) 2020**.
arXiv:1905.10437.

#### 3.8.2 Tipo de arquitectura

Stack profundo de bloques **fully-connected** que producen **forecast** y
**backcast** simultáneamente, con conexiones residuales tipo "doble residuo".
Existen dos sabores: **N-BEATS-Generic** (basis aprendidos) y
**N-BEATS-Interpretable** (basis explícitos: tendencia polinomial +
estacionalidad de Fourier).

```
input (T pasos) -> bloque₁ -> backcast₁ + forecast₁
              -> input - backcast₁ -> bloque₂ -> ...
                                              -> Σ forecasts -> ŷ_{t+1..t+H}
```

#### 3.8.3 Descripción técnica

- Cada bloque es un MLP con dos cabezas: una predice el "backcast" (qué del
  input explica) y otra el "forecast" (predicción).
- El input al siguiente bloque es el residuo del backcast del bloque anterior.
- **Versión Interpretable**: los basis del forecast/backcast son **polinomios**
  (tendencia) y **base de Fourier** (estacionalidad), reproduciendo
  descomposición clásica.
- **Innovación**: arquitectura **sin recurrencia ni atención** que ganó
  competencias M4 frente a estadísticos clásicos y DeepAR.

#### 3.8.4 Problema que aborda

Forecasting **univariado** o multivariado simple, multi-step, donde la
descomposición tendencia + estacionalidad es informativa.

#### 3.8.5 Datasets utilizados en el paper original

**M3, M4, Tourism** — competencias estándar de forecasting con miles de series.

#### 3.8.6 Métricas reportadas

**sMAPE**, **MASE**, **OWA**. Ganador de la M4 Competition (frente a 60+
métodos incluyendo modelos hibridos estadísticos-DL).

#### 3.8.7 Fortalezas

- **Sin recurrencia, sin atención** ⇒ entrenamiento rápido y muy paralelo.
- **Versión Interpretable** descompone la predicción en tendencia +
  estacionalidad — alineado con lo observado en STL del EDA.
- **Estado del arte** en forecasting univariado puro.
- Robusto, pocos hiperparámetros.

#### 3.8.8 Limitaciones

- **Originalmente univariado**; las versiones multivariadas son extensiones
  posteriores (NBEATSx).
- Sin embeddings de entidad nativos (panel se entrena por serie o con
  identificadores como features estáticas).
- Sin pronóstico probabilístico (sólo punto).

#### 3.8.9 Complejidad computacional

~10⁵–10⁶ parámetros; entrenamiento factible en CPU/GPU consumer.

#### 3.8.10 Aplicabilidad a este proyecto

- **Encaje con el panel**: medio. Para multivariado es necesario usar
  **NBEATSx** (extensión con covariables) o procesar cada estación.
- **Multi-horizonte 24/72/168**: **nativo** (la cabeza forecast tiene H
  unidades de salida).
- **Heterogeneidad regional**: requiere `station_id` como feature estática
  o entrenamiento por estación.
- **Hallazgos del EDA**:
  - **descomposición STL** mostró ~50–80 % de varianza estacional → la
    versión Interpretable de N-BEATS reproduce esa descomposición de forma
    *learned*;
  - **lookback 168 h**: bien soportado;
  - **outliers / eventos extremos**: el modelo tolera Huber loss adaptado.

**Conclusión**: **se implementa** (probablemente la versión `NBEATSx` con
covariables exógenas) como baseline fuerte de forecasting puro **no
recurrente, no Transformer**.
""")

# 3.9 DeepAR
md("""
### 3.9 Salinas, Flunkert, Gasthaus & Januschowski (2020) — DeepAR

#### 3.9.1 Referencia académica

Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). *DeepAR:
Probabilistic forecasting with autoregressive recurrent networks.*
**International Journal of Forecasting, 36**(3), 1181–1191.
https://doi.org/10.1016/j.ijforecast.2019.07.001

#### 3.9.2 Tipo de arquitectura

**RNN autoregresivo** (LSTM/GRU) que predice los **parámetros de una
distribución** (gaussiana o negativa-binomial) en cada paso, en lugar de
un valor puntual. Modelo **global** (un único modelo entrenado sobre todas
las series del panel).

```
input + embedding(entity) -> LSTM -> (μ_t, σ_t) -> sample/quantiles
                              ↑      (autoregresivo en inferencia)
                              └── y_{t-1}
```

#### 3.9.3 Descripción técnica

- **Modelo global**: un único conjunto de pesos para **todas** las series,
  con un **embedding de identidad** por serie que captura el "nivel base".
- **Output probabilístico**: la red predice los parámetros de una
  distribución (gaussiana para `temp_c`); en inferencia se samplea para
  obtener intervalos.
- **Autoregresivo en inferencia**: realimenta y_{t-1} sampleado para predecir
  y_t (con error compounding pero quantiles más fieles a la incertidumbre).
- **Innovación**: primer modelo de DL global con embeddings de entidad y
  pronóstico probabilístico que demuestra ganancia consistente sobre métodos
  clásicos en miles de series.

#### 3.9.4 Problema que aborda

Forecasting **probabilístico global** sobre **paneles de series similares**
(productos, regiones, estaciones).

#### 3.9.5 Datasets utilizados en el paper original

**Electricity, Traffic, Wikipedia, M4-hourly**, parts (Amazon retail).
Demuestra ganancia consistente sobre ARIMA, ETS, MQ-RNN.

#### 3.9.6 Métricas reportadas

**ND** (Normalized Deviation), **NRMSE**, **P50/P90 quantile loss**.

#### 3.9.7 Fortalezas

- **Embeddings de entidad nativos** (cada estación tendría su embedding).
- **Pronóstico probabilístico** (intervalos de confianza nativos vía
  sampleo).
- **Modelo global**: aprovecha información compartida entre estaciones
  similares (Cerrado vs Mata Atlántica) — alineado con la conclusión del EDA
  de que un modelo global con embeddings supera a un modelo por estación.
- Implementación robusta en `gluonts` y `pytorch-forecasting`.

#### 3.9.8 Limitaciones

- **LSTM-based** ⇒ misma limitación que LSTM puro para horizontes muy largos
  (168 h ya empuja el límite).
- Distribución gaussiana asume colas livianas — los **eventos extremos** de
  `temp_c` pueden caer fuera y requerir distribuciones alternativas
  (Student-t, asimétricas).
- Inferencia autoregresiva → error compounding.

#### 3.9.9 Complejidad computacional

~10⁵–10⁶ parámetros; entrenamiento moderado en GPU.

#### 3.9.10 Aplicabilidad a este proyecto

- **Encaje con el panel**: **óptimo en filosofía** — fue diseñado para
  paneles de series similares.
- **Multi-horizonte 24/72/168**: aceptable a 24/72 h, degradación notable
  a 168 h por compounding autoregresivo.
- **Heterogeneidad regional**: capturada por el embedding de entidad +
  features estáticas (region/biome/koppen).
- **Hallazgos del EDA**:
  - heterogeneidad regional dramática → embeddings nativos;
  - eventos extremos → quantiles P10/P90 dan intervalos asimétricos;
  - panel balanceado por `WeightedRandomSampler` → encaja con el ImputableSampler
    de gluonts.

**Conclusión**: **se evaluará como alternativa probabilística** de la familia
recurrente. Si el desempeño es competitivo, se incluye en el benchmark final;
si LSTM con atención (variante Suleman & Shridevi) ya es suficiente, se
mantiene DeepAR como referencia bibliográfica.
""")

# ============================================================================
# Sección 4 — Comparación crítica
# ============================================================================
md("""
## 4. Comparación Crítica

### 4.1 Tabla comparativa estructurada
""")

code("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

comparison = pd.DataFrame([
    {"modelo": "Conv1D-MLP (Khan)",     "año": 2020,
     "arquitectura": "CNN+MLP híbrido",
     "multi_horizonte": "vector denso (limitado)",
     "panel_embeddings": "no",
     "estacionalidad": "no explícita (necesita features cíclicas)",
     "costo": "bajo",
     "interpretabilidad": "baja",
     "idoneidad_temp_brasil": 2},
    {"modelo": "ConvLSTM+Att (Tekin)",  "año": 2024,
     "arquitectura": "ConvLSTM+attention",
     "multi_horizonte": "rolling encoder-decoder",
     "panel_embeddings": "no (asume grilla)",
     "estacionalidad": "implícita en features 2D",
     "costo": "alto",
     "interpretabilidad": "media (atención)",
     "idoneidad_temp_brasil": 2},
    {"modelo": "SFA-LSTM (Suleman)",    "año": 2022,
     "arquitectura": "LSTM enc-dec + attn variables",
     "multi_horizonte": "decoder seq2seq",
     "panel_embeddings": "no nativo",
     "estacionalidad": "vía features cíclicas",
     "costo": "medio",
     "interpretabilidad": "media-alta (attn variables)",
     "idoneidad_temp_brasil": 4},
    {"modelo": "DL distrital (Sharma)", "año": 2023,
     "arquitectura": "CNN+RNN focalizado en extremos",
     "multi_horizonte": "limitado (~1 día)",
     "panel_embeddings": "no",
     "estacionalidad": "implícita",
     "costo": "alto",
     "interpretabilidad": "media",
     "idoneidad_temp_brasil": 2},
    {"modelo": "DL post-proc (Trivedi)","año": 2024,
     "arquitectura": "DL corrector NWP",
     "multi_horizonte": "depende del NWP",
     "panel_embeddings": "no",
     "estacionalidad": "heredada del NWP",
     "costo": "bajo",
     "interpretabilidad": "baja",
     "idoneidad_temp_brasil": 1},
    {"modelo": "TFT (Lim)",             "año": 2021,
     "arquitectura": "Transformer+LSTM+VSN",
     "multi_horizonte": "nativo (vector quantile)",
     "panel_embeddings": "sí (static covariates)",
     "estacionalidad": "vía known-future inputs",
     "costo": "alto",
     "interpretabilidad": "alta (VSN+attn)",
     "idoneidad_temp_brasil": 5},
    {"modelo": "Informer (Zhou)",       "año": 2021,
     "arquitectura": "Transformer ProbSparse",
     "multi_horizonte": "nativo (one-shot)",
     "panel_embeddings": "vía features estáticas",
     "estacionalidad": "vía features cíclicas",
     "costo": "alto",
     "interpretabilidad": "media",
     "idoneidad_temp_brasil": 5},
    {"modelo": "N-BEATS (Oreshkin)",    "año": 2020,
     "arquitectura": "MLP stack con basis",
     "multi_horizonte": "nativo (vector forecast)",
     "panel_embeddings": "vía NBEATSx",
     "estacionalidad": "nativa (basis Fourier)",
     "costo": "medio",
     "interpretabilidad": "alta (versión interpretable)",
     "idoneidad_temp_brasil": 4},
    {"modelo": "DeepAR (Salinas)",      "año": 2020,
     "arquitectura": "LSTM autoregresivo prob.",
     "multi_horizonte": "autoregresivo (compounding)",
     "panel_embeddings": "sí (entity embedding)",
     "estacionalidad": "vía features cíclicas",
     "costo": "medio",
     "interpretabilidad": "media",
     "idoneidad_temp_brasil": 4},
])
comparison
""")

md("### 4.2 Heatmap de idoneidad por criterio (1–5)")

code("""
# Heatmap visual de idoneidad por criterio para este proyecto.
# Escalas 1 (pobre) - 5 (excelente).
criteria = pd.DataFrame([
    # modelo,                     mh,  panel,  est,   cost(inv), interp, datos_temp_brasil
    ["Conv1D-MLP (Khan)",         2,   1,      2,     5,         2,      2],
    ["ConvLSTM+Att (Tekin)",      3,   1,      3,     2,         3,      2],
    ["SFA-LSTM (Suleman)",        4,   2,      3,     4,         4,      4],
    ["DL distrital (Sharma)",     1,   1,      2,     2,         3,      2],
    ["DL post-proc (Trivedi)",    1,   1,      2,     5,         2,      1],
    ["TFT (Lim)",                 5,   5,      5,     2,         5,      5],
    ["Informer (Zhou)",           5,   3,      4,     2,         3,      5],
    ["N-BEATS (Oreshkin)",        5,   3,      5,     4,         5,      4],
    ["DeepAR (Salinas)",          3,   5,      4,     4,         3,      4],
], columns=["modelo", "multi-horizonte", "panel/embeddings",
            "estacionalidad", "costo (inv)", "interpretabilidad",
            "idoneidad temp_c BR"]).set_index("modelo")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(criteria, annot=True, cmap="YlGnBu", vmin=1, vmax=5, cbar_kws={"label": "puntaje (1–5)"}, ax=ax)
ax.set_title("Idoneidad por criterio para forecasting de temp_c sobre panel INMET")
plt.tight_layout()
out = Path("../results/figures/sota") if Path.cwd().name == "notebooks" else Path("results/figures/sota")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "04_heatmap_idoneidad.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
### 4.3 Análisis cualitativo de tendencias

#### 4.3.1 Transición arquitectónica RNN → Attention → Transformer

- **2019–2020**: dominio claro de **LSTM** y variantes (DeepAR, encoder-decoder
  con atención). Los papers del PDF de **Khan** y **Suleman** son representativos.
- **2021**: salto cualitativo con **Transformers específicos** para series:
  TFT (multi-horizon + interpretabilidad) e Informer (long sequence). Ambos
  resuelven limitaciones específicas de Transformer vanilla en forecasting.
- **2022–2024**: consolidación de Transformers + emergencia de variantes
  livianas (Autoformer, FEDformer, PatchTST). Los híbridos
  ConvLSTM+attention (Tekin 2024) **siguen siendo competitivos** en problemas
  con componente espacial (grillas).

#### 4.3.2 Modelos híbridos vs forecasting puro

- **CNN+RNN / ConvLSTM**: dominantes cuando hay **componente espacial fuerte**
  (radar, satélite, grilla regular). En **panel disperso** de estaciones
  pierden ventaja.
- **Modelos puros de forecasting** (N-BEATS, DeepAR, TFT, Informer):
  diseñados para series, agnosticos a la modalidad espacial. Mejor encaje
  con el panel INMET.

#### 4.3.3 Modelos globales con embeddings vs locales

- **Globales con embeddings** (DeepAR, TFT): **consistentemente mejores**
  cuando hay 100+ series similares — aprovechan transferencia entre series.
- **Locales por entidad**: razonables si hay pocas series y mucha
  heterogeneidad regional. En nuestro caso (40 estaciones, heterogeneidad
  *dramática* pero con grupos por bioma/Köppen), la **estrategia híbrida
  (un modelo global + embeddings) gana**.

#### 4.3.4 Descomposición explícita

- **N-BEATS-Interpretable** y **Autoformer** descomponen explícitamente
  tendencia + estacionalidad. Ventaja clara cuando la estacionalidad domina,
  como en nuestro EDA (50–80 % de varianza explicada por estacional según STL).

### 4.4 Gap identificado en la literatura

Tras la revisión, identificamos **tres huecos** que este proyecto contribuye a
llenar:

1. **Pocos papers cubren panel nacional brasileño con DL.** La mayoría de
   trabajos sobre forecasting meteorológico en LATAM usan métodos clásicos
   (ARIMA, regresión). Los pocos que aplican DL se centran en una región
   específica (Amazonia, Cerrado, Pampa) y no cruzan la heterogeneidad
   continental.
2. **Pocos comparan Transformers puros vs LSTM con tests estadísticos
   formales (Diebold-Mariano, Friedman/Nemenyi).** La literatura suele
   reportar ganancias en RMSE/MAE sin evaluar significancia estadística —
   este proyecto adopta el protocolo riguroso del benchmark.
3. **Falta estudio sistemático de transferencia entre regímenes climáticos
   diversos.** Comparaciones entre Amazonia (clima ecuatorial) y Pampa
   (subtropical templado) son raras; nuestro panel cubre los **5 climas
   IBGE** y **6 biomas** simultáneamente.
""")

# ============================================================================
# Sección 5 — Selección final
# ============================================================================
md("""
## 5. Selección Final del Top de Modelos a Implementar

A partir del Top 9, seleccionamos **6 modelos** para el benchmark final.
La justificación adopta la lógica del entregable: cubrir las **4 familias
arquitectónicas** (baseline ingenuo, recurrente, basado en bloques, basado
en attention) y reportar al menos **un modelo probabilístico** y **un modelo
interpretable**.
""")

code("""
selected = pd.DataFrame([
    {"#": 1, "modelo": "Persistencia (naive)",
     "familia": "Baseline ingenuo",
     "razón_de_inclusión": "Sanity check obligatorio: ŷ_{t+h} = y_t. Cualquier modelo DL debe superarlo significativamente (Diebold-Mariano).",
     "complejidad": "trivial",
     "implementación": "src/models/baselines.py"},
    {"#": 2, "modelo": "LSTM vanilla",
     "familia": "Recurrente",
     "razón_de_inclusión": "Baseline de la familia recurrente. Conecta con Suleman & Shridevi (2022). Permite cuantificar la ganancia marginal de mecanismos avanzados.",
     "complejidad": "media",
     "implementación": "src/models/lstm.py"},
    {"#": 3, "modelo": "GRU",
     "familia": "Recurrente liviana",
     "razón_de_inclusión": "Alternativa más liviana al LSTM. Comparación interna de la familia recurrente: ¿el costo extra del LSTM compensa?",
     "complejidad": "media-baja",
     "implementación": "src/models/gru.py"},
    {"#": 4, "modelo": "N-BEATSx",
     "familia": "MLP con basis",
     "razón_de_inclusión": "Baseline fuerte de forecasting puro no recurrente, no Transformer. La descomposición tendencia+estacionalidad encaja con lo observado en STL del EDA.",
     "complejidad": "media",
     "implementación": "src/models/nbeats.py"},
    {"#": 5, "modelo": "Temporal Fusion Transformer (TFT)",
     "familia": "Híbrida (LSTM+Attention+VSN)",
     "razón_de_inclusión": "Multi-horizonte nativo + embeddings de estación + quantile loss. Encaje óptimo con todas las decisiones del EDA. Paper guía del proyecto.",
     "complejidad": "alta",
     "implementación": "src/models/tft.py"},
    {"#": 6, "modelo": "Informer",
     "familia": "Transformer eficiente",
     "razón_de_inclusión": "Diseñado para horizontes largos (168 h); generative decoder evita compounding. Contraste arquitectónico contra TFT.",
     "complejidad": "alta",
     "implementación": "src/models/informer.py"},
])
selected
""")

md("""
### Modelos del Top 9 que **no se implementan** (con justificación)

| Modelo | Razón de exclusión |
|---|---|
| **Conv1D-MLP (Khan & Maity, 2020)** | Frecuencia diaria (vs horaria del proyecto), variable lluvia (vs temp_c), sin embeddings. Una versión adaptada quedaría redundante con el LSTM vanilla como baseline simple. |
| **ConvLSTM+Att (Tekin et al., 2024)** | Requiere **grilla regular**; el panel INMET son estaciones puntuales dispersas. Pre-interpolar a grilla introduciría smoothing en regiones de baja densidad (Norte, Pantanal). |
| **DL distrital (Sharma et al., 2023)** | Foco en lluvia intensa con clasificación POD/FAR/CSI; arquitectura no extrapolable directamente a temperatura horaria multi-step. **Adoptamos sus métricas** (POD/FAR/CSI sobre `is_extreme`) sin implementar la arquitectura. |
| **DL post-proc (Trivedi et al., 2024)** | Requiere salidas de NWP operativo externo. El proyecto, en esta entrega, es puramente observacional. Queda como **fase 2** (integración GFS/ECMWF). |
| **DeepAR (Salinas et al., 2020)** | **Candidato condicional**: si el LSTM vanilla + atención sobre variables (variante Suleman & Shridevi) ya cubre el régimen probabilístico, DeepAR queda como referencia bibliográfica. Si en la fase de validación se requiere reportar intervalos de confianza vía sampleo, se incorpora. |

> Nota: la frontera entre el "subset implementado" (6) y el "Top 9 revisado" es
> deliberadamente conservadora. Implementar bien y comparar rigurosamente 6
> modelos con tests estadísticos es preferible a implementar 9 superficialmente.
""")

# ============================================================================
# Sección 6 — Paper guía
# ============================================================================
md("""
## 6. Paper Guía del Proyecto

### 6.1 Selección

El **paper guía** elegido es:

> **Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion
> Transformers for interpretable multi-horizon time series forecasting.*
> International Journal of Forecasting, 37(4), 1748–1764.
> https://doi.org/10.1016/j.ijforecast.2021.03.012**

### 6.2 Descripción detallada del modelo

El **Temporal Fusion Transformer (TFT)** es una arquitectura especializada para
forecasting multi-horizonte multivariado que integra cinco innovaciones
arquitectónicas en una única red entrenable end-to-end:

1. **Variable Selection Networks (VSN)**: módulos *gating* que aprenden,
   *por timestep*, qué variables exógenas contribuyen al pronóstico. Producen
   pesos interpretables que reemplazan la "caja negra" típica de las redes
   profundas.
2. **Static covariate encoders**: cuatro contextos vectoriales que se inyectan
   en distintas etapas (selección de variables, LSTM, atención, gating final),
   permitiendo que metadatos invariantes en el tiempo (en nuestro caso:
   `station_id`, `region`, `biome`, `koppen_class`, `latitude`, `longitude`,
   `altitude`) modulen todo el procesamiento.
3. **LSTM encoder-decoder local**: captura dependencias de corto plazo y
   procesa la secuencia de inputs (pasados observados + pasados conocidos +
   futuros conocidos como las features cíclicas).
4. **Multi-head self-attention**: captura dependencias de largo plazo,
   crucial para el lookback de 168 h y la estacionalidad anual remanente.
5. **Quantile loss multi-percentil**: la cabeza de salida produce los cuantiles
   P10, P50, P90 (configurable), entregando **intervalos de confianza
   nativos** sin necesidad de ensembles.

#### Supuestos clave

- Las variables se separan en **estáticas** (constantes por entidad), **conocidas
  a futuro** (cíclicas, calendario), **observadas pasadas** (exógenas con
  historia) y **target** (observado pasado, predicho futuro). Esta taxonomía
  encaja perfecto con nuestros datos.
- El target se modela como **regresión** con loss cuantílica; soporta múltiples
  targets simultáneos (no necesario en nuestro proyecto, que es univariado).

### 6.3 Justificación de selección

| Criterio | Por qué TFT |
|---|---|
| **Multi-horizonte nativo** | Genera 168 pasos en una pasada con quantile loss; nuestros horizontes 24/72/168 se evalúan como cortes del mismo output. |
| **Embeddings de entidad** | Static covariates son ciudadanos de primera clase. Ideal para 40 estaciones × 5 regiones × 6 biomas. |
| **Covariables cíclicas conocidas a futuro** | El EDA confirmó dominancia de ciclos diario (24 h) y anual; TFT las consume nativamente como "known future inputs". |
| **Interpretabilidad** | VSN + atención producen pesos por variable y por paso; permite **validar empíricamente** las hipótesis del EDA (importancia de humedad/presión/radiación). |
| **Pronóstico probabilístico** | Quantile loss da bandas P10/P90 — captura incertidumbre en eventos extremos detectados en el EDA. |
| **Q1 + impacto** | International Journal of Forecasting (Q1, JCR), >5000 citas; implementación oficial mantenida en `pytorch-forecasting`. |
| **Encaje con la literatura del PDF** | Conecta con la línea de **atención sobre variables** de Suleman & Shridevi (2022), pero la lleva al régimen multi-horizonte y panel. |

### 6.4 Relación con los demás modelos del benchmark

- **vs Persistencia**: TFT debe superarla con margen *significativo*
  (Diebold-Mariano p < 0.05); si no lo hace, el dataset es trivial o el modelo
  está mal entrenado.
- **vs LSTM/GRU**: comparte la familia (LSTM como bloque interno), pero añade
  atención multi-head y VSN. La pregunta de investigación es: ¿el costo extra
  de TFT compensa frente a LSTM con atención sobre variables (Suleman-style)?
- **vs N-BEATSx**: ambos producen multi-horizonte de una pasada, pero N-BEATS
  no tiene atención ni interpretabilidad por variable. Comparación
  arquitectura-libre: ¿qué importa más, basis explícitos o atención?
- **vs Informer**: ambos son Transformers, pero TFT prioriza
  interpretabilidad+entidades, Informer prioriza eficiencia en horizontes muy
  largos. Comparación pertinente para evaluar **sub-horizon** (24h, 72h)
  vs **long-horizon** (168h).
""")

# ============================================================================
# Sección 7 — Conexión con el EDA
# ============================================================================
md("""
## 7. Conexión Explícita con el EDA

Cada decisión del EDA se mapea al modelo del Top que mejor responde:

| Decisión / hallazgo del EDA | Implicación para el modelo | Modelo(s) más adecuado(s) |
|---|---|---|
| **Estacionalidad diaria + anual fuerte** (FFT picos en 24 h, 12 h, 8766 h; STL ~50–80 % varianza) | Necesita arquitectura que capture ciclos de corto y largo plazo | **N-BEATS-Interpretable** (basis Fourier explícitos), **TFT** (known-future cíclicas + atención multi-head), **Autoformer** (descomposición series-decomp interna) |
| **Lookback `168 h` recomendado** (ACF/PACF significativa hasta 168) | Define la longitud de la ventana de entrada | **Todos los modelos del subset lo soportan**; **Informer** y **TFT** ganan en horizontes largos por atención escalable; **LSTM/GRU** sufren en 168 |
| **Heterogeneidad regional dramática** (Norte 22-32 plano vs Sul 5-35 amplio) | Modelo global con embeddings supera a uno por estación | **TFT** (static covariates), **DeepAR** (entity embeddings), **N-BEATSx** (covariables estáticas) |
| **Eventos extremos en colas (p01/p99)** + flag `is_extreme` | Loss robusta o cuantílica para no sub-predecir colas | **TFT** (quantile loss nativo P10/P50/P90), **DeepAR** (distribución gaussiana o Student-t), **Huber loss** sobre LSTM/GRU/N-BEATS |
| **Faltantes ~21 % en `radiation_kj_m2`** (gaps nocturnos legítimos) | Excluir esa feature o imputar con cuidado | Todos toleran si se preprocesa (ya hecho con `ffill(6)`); **TFT/N-BEATSx/Informer** filtran ventanas con NaN antes de entrenamiento |
| **Panel: 40 estaciones × ~52 584 horas (train) ≈ 2.1 M ejemplos** | Volumen moderado-alto: viable para Transformers | **TFT** y **Informer** entrenables en GPU mid-range en horas; **LSTM** puede ser lento por la recurrencia |
| **Exógenas seleccionadas** (humidity, pressure, radiation, wind_speed, dew_point) | Inputs multivariados al modelo | **TFT**: past observed inputs; **N-BEATSx**: covariables exógenas; **DeepAR/LSTM/GRU**: concat al input |
| **Estandarización por estación** (decisión del EDA) | El scaler fitea en train por wmo | **Todos compatibles**; ya implementado en `src/data/scalers.py` |
| **Sesgo de representación regional** (Sudeste/Nordeste sobre-representados) | Muestreo balanceado por región | `WeightedRandomSampler` durante entrenamiento — compatible con **todos los modelos** del subset |
| **Multi-horizonte 24/72/168 h** | Output multi-step | **TFT, Informer, N-BEATSx**: nativos. **LSTM/GRU** con cabeza densa: vector de 168. **DeepAR**: autoregresivo (compounding) |
| **Anti-leakage temporal** (split por años, ventaneo no cruza fronteras) | El benchmark debe respetar el split | **Todos los modelos** consumen los mismos splits ya validados por `tests/test_split_real_data.py` y `test_windowing_no_leakage.py` |
""")

# ============================================================================
# Sección 8 — Síntesis ejecutiva
# ============================================================================
md("""
## 8. Síntesis Ejecutiva del SOTA

### 8.1 Top 9 identificado

1. Khan & Maity (2020) — Conv1D-MLP híbrido (precipitación diaria).
2. Tekin, Fazla & Kozat (2024) — ConvLSTM + atención + context matcher.
3. Suleman & Shridevi (2022) — SFA-LSTM (atención sobre variables).
4. Sharma et al. (2023) — DL distrital para lluvia intensa.
5. Trivedi, Sharma & Pattnaik (2024) — DL como post-procesador NWP.
6. Lim et al. (2021) — Temporal Fusion Transformer.
7. Zhou et al. (2021) — Informer.
8. Oreshkin et al. (2020) — N-BEATS.
9. Salinas et al. (2020) — DeepAR.

### 8.2 Subset a implementar (6)

1. **Persistencia** — sanity check obligatorio (Diebold-Mariano).
2. **LSTM vanilla** — baseline familia recurrente.
3. **GRU** — variante liviana de la familia recurrente.
4. **N-BEATSx** — baseline fuerte no recurrente, no Transformer.
5. **Temporal Fusion Transformer (TFT)** — multi-horizon + embeddings + quantile (paper guía).
6. **Informer** — Transformer eficiente para horizonte largo (168 h).

### 8.3 Paper guía

**Temporal Fusion Transformer** (Lim et al., 2021): encaja con todas las
decisiones del EDA — multi-horizonte nativo, embeddings de estación,
covariables cíclicas conocidas a futuro, interpretabilidad y quantile loss.
Vertebra la arquitectura del proyecto y sirve como referencia para los demás
modelos.

### 8.4 Gap llenado por este proyecto

(i) Aplicación sistemática de DL para forecasting de temperatura sobre **panel
nacional brasileño** (40 estaciones × 5 macrorregiones), comparando
**Transformers puros vs LSTM** con **tests estadísticos formales**
(Diebold-Mariano, Friedman/Nemenyi); (ii) estudio de **transferencia entre
regímenes climáticos diversos** (Amazonia ↔ Pampa) vía embeddings de entidad;
(iii) evaluación condicional sobre **eventos extremos** con métricas POD/FAR/CSI
adaptadas de Sharma et al. (2023).

### 8.5 Próximos pasos

- **Notebook 04 — `04_benchmark_models.ipynb`**: implementación de los 6
  modelos en `src/models/`, entrenamiento con N semillas (config: 5),
  evaluación cuantitativa por horizonte y por región, tests estadísticos
  pareados (Diebold-Mariano, Friedman/Nemenyi/Wilcoxon).
- **Notebook 05 — `05_guide_paper.ipynb`**: análisis profundo del TFT como
  paper guía, con réplica controlada sobre 1 estación y discusión de
  interpretabilidad.
- **Notebook 06 — `06_benchmark_final.ipynb`**: tabla comparativa final,
  métricas por región, intervalos de confianza bootstrap, conclusiones del
  benchmark.

---

*Fin del capítulo 03. El siguiente notebook (`04_benchmark_models.ipynb`)
inicia la fase de implementación.*
""")


# ============================================================================
# Construir notebook (sin ejecutar)
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

out_path = Path("notebooks/03_state_of_the_art.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook escrito: {out_path} ({len(cells)} celdas)")
