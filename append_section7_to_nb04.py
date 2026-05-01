"""Apéndice idempotente al notebook 04: agrega la sección 7
'Ejecución de Entrenamientos desde el Notebook' al final, sin tocar
las secciones 1-6 ni ejecutar ninguna celda.

Si la sección 7 ya existe (detectada por marcador), se reemplaza para
mantener la idempotencia.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path("notebooks/04_benchmark_models.ipynb")
MARKER = "<!-- SECCION_7_TRAIN_FROM_NB -->"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# Celdas nuevas (11 en total)
# ---------------------------------------------------------------------------
new_cells: list = []

# 7.0 — Markdown: título de sección + introducción (combinados)
new_cells.append(md(f"""{MARKER}
## 7. Ejecución de Entrenamientos desde el Notebook

Estas celdas reemplazan al loop de bash/PowerShell. Cada modelo tiene su
propia celda — se pueden correr **independientemente** y en cualquier orden.
El logging va al output de la celda **en tiempo real** (gracias a
`subprocess.Popen` + stream línea por línea).

Cada celda:

1. Imprime el nombre del modelo y el comando equivalente que correría en
   terminal.
2. Hace stream del output del runner mientras entrena.
3. Al terminar, imprime un resumen de los runs generados en
   `experiments/<model>/<station>/seed=<N>/`.

Notas operativas:

- Las celdas son **idempotentes**: correr `train_model("lstm")` dos veces
  simplemente sobreescribe los artefactos.
- Se pueden interrumpir con el botón **Detener** del notebook.
- Si `config/config.yaml` → `sampling.enabled: true`, los runs son rápidos
  (~2–5 min por modelo). Para resultados oficiales, poner `false` y
  re-ejecutar.
- El error de un modelo no detiene a los demás — cada celda es independiente.
"""))

# 7.1 — Setup
new_cells.append(code('''import subprocess
import sys
import os
import json
from pathlib import Path
from IPython.display import display, Markdown

# Asegurar que estamos en la raíz del proyecto, no en notebooks/
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
assert (PROJECT_ROOT / "src" / "training" / "runner.py").exists(), \\
    f"No encuentro runner.py — ¿estás en la raíz del proyecto?"


def train_model(model_name: str) -> int:
    """Lanza el entrenamiento del modelo y stream-ea logs al notebook.

    Returns el returncode (0 = OK, !=0 = error).
    """
    cmd = [sys.executable, "-m", "src.training.runner", "--model", model_name]
    print(f"\\n{'='*60}")
    print(f"🚀 Entrenando: {model_name}")
    print(f"Comando: {' '.join(cmd)}")
    print(f"{'='*60}\\n", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    print(f"\\n{'='*60}")
    if proc.returncode == 0:
        print(f"✅ {model_name} completado.")
    else:
        print(f"❌ {model_name} falló con código {proc.returncode}.")
    print(f"{'='*60}")

    # Listar artefactos generados
    exp_dir = PROJECT_ROOT / "experiments" / model_name
    if exp_dir.exists():
        runs = list(exp_dir.glob("*/*/metrics.json"))
        print(f"\\nArtefactos generados ({len(runs)} runs):")
        for r in sorted(runs)[:20]:
            print(f"  - {r.relative_to(PROJECT_ROOT)}")

    return proc.returncode


print("Función train_model lista. Usa train_model('<nombre>') en las siguientes celdas.")
'''))

# 7.2 — Persistence
new_cells.append(code('''# Persistence: baseline naive sin parámetros entrenables.
# Tiempo estimado con sampling: ~2-3 min.
train_model("persistence")
'''))

# 7.3 — LSTM
new_cells.append(code('''# LSTM: baseline recurrente. Familia RNN clásica.
# Tiempo estimado con sampling: ~5-10 min.
train_model("lstm")
'''))

# 7.4 — GRU
new_cells.append(code('''# GRU: alternativa más liviana al LSTM.
# Tiempo estimado con sampling: ~5-8 min.
train_model("gru")
'''))

# 7.5 — TCN
new_cells.append(code('''# TCN: convolucional dilatada para series temporales.
# Tiempo estimado con sampling: ~5-12 min.
train_model("tcn")
'''))

# 7.6 — N-BEATS
new_cells.append(code('''# N-BEATS: arquitectura basada en bloques de basis para forecasting.
# Tiempo estimado con sampling: ~7-15 min.
train_model("nbeats")
'''))

# 7.7 — Transformer
new_cells.append(code('''# Transformer: atención multi-head sobre toda la secuencia.
# Tiempo estimado con sampling: ~10-15 min.
train_model("transformer")
'''))

# 7.8 — TFT
new_cells.append(code('''# TFT: Temporal Fusion Transformer - el paper guía.
# Tiempo estimado con sampling: ~12-20 min.
train_model("tft")
'''))

# 7.9 — Verificación final
new_cells.append(code('''# Resumen del estado del benchmark
exp_dir = PROJECT_ROOT / "experiments"
modelos = ["persistence", "lstm", "gru", "tcn", "nbeats", "transformer", "tft"]
total = 0
print(f"{'Modelo':<15} {'Runs':>6}")
print("-" * 22)
for m in modelos:
    path = exp_dir / m
    n = len(list(path.glob("*/*/metrics.json"))) if path.exists() else 0
    total += n
    flag = "✅" if n >= 8 else ("⚠️" if n > 0 else "⏳")
    print(f"{flag} {m:<13} {n:>6}")
print("-" * 22)
print(f"{'TOTAL':<15} {total:>6}")
print(f"\\nEsperado con sampling: 7 modelos × 4 estaciones × 2 seeds = 56 runs.")
'''))

# 7.10 — Cierre
new_cells.append(md("""### Próximo paso

Cuando todas las celdas de arriba hayan ejecutado y la verificación muestre
**56 runs totales** (con sampling activo), abre
`notebooks/06_benchmark_final.ipynb` y haz **Ejecutar todo** para ver el
análisis estadístico completo.

Para resultados oficiales (sin muestreo), edita `config/config.yaml` →
`sampling.enabled: false` y vuelve a ejecutar todas las celdas de esta
sección. Tiempo estimado del benchmark completo: **5–15 horas**.
"""))


def main() -> None:
    nb = nbf.read(NB_PATH, as_version=4)
    # Idempotencia: si ya existe la marca, recortamos desde ahí.
    keep = []
    for cell in nb.cells:
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if MARKER in src:
            break
        keep.append(cell)
    n_old = len(nb.cells)
    n_kept = len(keep)
    nb.cells = keep + new_cells
    nbf.write(nb, NB_PATH)
    print(f"Notebook actualizado: {NB_PATH}")
    print(f"  Celdas previas: {n_old}")
    print(f"  Celdas conservadas (antes de la marca): {n_kept}")
    print(f"  Celdas nuevas agregadas: {len(new_cells)}")
    print(f"  Total final: {len(nb.cells)}")


if __name__ == "__main__":
    main()
