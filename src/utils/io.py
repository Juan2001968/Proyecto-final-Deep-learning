"""Lectura/escritura consistente: YAML, Parquet, JSON y artefactos sklearn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------- YAML

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


# -------------------------------------------------------------------- Parquet

def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


# ----------------------------------------------------------------------- JSON

def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)
