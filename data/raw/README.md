# `data/raw/` — datos crudos INMET

Esta carpeta contiene los CSV originales descargados del portal INMET. **No
se modifican**. Cualquier transformación se hace aguas abajo en
`data/interim/` y `data/processed/`.

## Cómo descargar

1. Ir a <https://portal.inmet.gov.br/dadoshistoricos>.
2. Descargar los ZIP anuales (un ZIP por año), p.ej. `2016.zip` … `2023.zip`.
3. Colocar los ZIPs **directamente** en `data/raw/` (sin extraer):

   ```
   data/raw/
   ├── 2016.zip
   ├── 2017.zip
   ├── ...
   └── 2023.zip
   ```

4. Ejecutar:

   ```bash
   make ingest
   ```

   El script `src/data/ingest_inmet.py` descomprime los ZIPs en
   subcarpetas por año, filtra las estaciones listadas en
   `config/stations.yaml` (códigos WMO) y deja el árbol limpio en
   `data/interim/<station_code>/<year>.csv`.

## Convención de nombres de archivo INMET

Cada CSV anual tiene un nombre del estilo:

```
INMET_<REGION>_<UF>_<WMO>_<NOMBRE>_<DATA_INICIAL>_<DATA_FINAL>.CSV
```

Ejemplo:

```
INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023.CSV
```

El código WMO (4 caracteres alfanuméricos, p.ej. `A001`) se usa como
identificador estable de estación.

## Verificación rápida

```bash
ls data/raw/*.zip | wc -l   # debería listar los años descargados
```
