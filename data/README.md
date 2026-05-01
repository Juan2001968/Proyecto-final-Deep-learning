# Datos

Esta carpeta es el "lago de datos" del proyecto y se versiona **vacía**: los
archivos reales de datos están bajo `.gitignore` (excepto los README).

Subdirectorios:

| Carpeta            | Contenido                                                        |
|--------------------|------------------------------------------------------------------|
| `raw/`             | CSV anuales por estación tal como vienen de INMET (no se modifica) |
| `interim/`         | CSV concatenados por estación, con timestamps en UTC              |
| `processed/`       | Parquet final particionado en `train` / `val` / `test` temporales |

## Origen y licencia

- **Fuente**: [INMET — Banco de Dados Meteorológicos](https://portal.inmet.gov.br/dadoshistoricos)
- **Licencia**: dato público de divulgación oficial. Para uso académico citar
  la fuente. Verificar la sección de "Termos de uso" del portal antes de
  redistribuir.
- **Cobertura**: ~600 estaciones automáticas en Brasil, frecuencia horaria.
- **Resolución temporal**: 1 hora (UTC).
- **Tamaño aproximado**: ~30–80 MB/año por estación tras compresión, ~700 MB
  para 8 años × 3 estaciones de ejemplo.

## Caveats importantes (INMET)

1. **Encoding**: los CSV vienen en `latin-1` / `ISO-8859-1`; **no** UTF-8.
2. **Decimal coma**: usar `decimal=","` en `pandas.read_csv`.
3. **Separador `;`** no coma.
4. **Header de 8 líneas** con metadatos (REGIÃO, UF, ESTAÇÃO, CODIGO WMO,
   LAT, LON, ALT, DATA DE FUNDAÇÃO). Saltarlo con `skiprows`.
5. **Faltantes**: `-9999` (a veces `-9999.0`) en columnas numéricas.
6. **Timestamps**: viene `Data` (YYYY/MM/DD) + `Hora UTC` (HHMM UTC). En
   archivos antiguos puede aparecer como `00:00 UTC` con espacio.
7. **Cambios de instrumentación**: estaciones automáticas pueden cambiar
   sensores → saltos / cambios de varianza no estacionarios.
8. **Gaps**: huecos prolongados (días enteros) son comunes; `src/data/clean.py`
   los detecta y `src/data/resample.py` decide la política de imputación.
9. **Outliers físicos**: precipitación negativa, humedad >100 %, etc. — el
   pipeline las recorta a rangos físicos plausibles.
