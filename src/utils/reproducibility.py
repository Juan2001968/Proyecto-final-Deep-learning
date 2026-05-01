"""Captura del entorno (versiones de libs, hash del dataset) para reproducir."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


_PIN_LIBRARIES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "torch",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "yaml",
)


def _safe_version(modname: str) -> str | None:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def capture_environment() -> dict[str, Any]:
    """Snapshot de python/OS/git/libs para guardar junto a cada run."""
    env: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    try:
        env["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        env["git_commit"] = None

    env["libraries"] = {lib: _safe_version(lib) for lib in _PIN_LIBRARIES}
    return env


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash determinístico de un DataFrame (orden de columnas + filas)."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
