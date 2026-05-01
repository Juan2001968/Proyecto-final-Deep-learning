"""Modelos de DL para forecasting multistep. Lógica concreta a rellenar."""

from .base import BaseForecaster

# TODO: implementar Informer (src/models/model_informer.py). El config
# `config/models/informer.yaml` ya existe pero el módulo Python falta —
# el runner romperá si se intenta entrenar antes de que se implemente.

__all__ = ["BaseForecaster"]
