"""Control determinístico de semillas para numpy / random / torch / CUDA."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Fija las semillas de todas las fuentes de aleatoriedad usadas.

    Si ``deterministic=True`` también fuerza algoritmos determinísticos en
    cuDNN (más lento pero reproducible). Llamar **antes** de construir
    DataLoaders, modelos y optimizadores.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass
