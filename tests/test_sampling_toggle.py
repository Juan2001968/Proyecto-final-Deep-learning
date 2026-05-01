"""Tests del toggle de muestreo provisional.

Verifican que ``config.sampling`` recorta volumen y compute SIN tocar la
lógica de modelado, escalado, ventaneo o tests estadísticos.

Reglas que se prueban:

1. Con ``sampling.enabled: false`` el split usa los 6 años de train y todas
   las estaciones del ``stations.yaml`` (comportamiento original).
2. Con ``sampling.enabled: true`` el split usa solo los ``train_years`` del
   bloque ``sampling`` y filtra a ``n_stations_per_region`` por región.
3. Con ``sampling.enabled: true`` los overrides de ``epochs`` y ``seeds``
   se aplican sobre el ``model_cfg`` y el contador del runner.
"""

from __future__ import annotations

from src.data.split import _apply_sampling
from src.training.runner import _apply_sampling_overrides
from src.utils.regions import all_regions


def _full_config(stations_yaml: dict | None = None) -> dict:
    return {
        "split": {
            "mode": "by_year",
            "by_year": {
                "train_years": [2018, 2019, 2020, 2021, 2022, 2023],
                "val_years": [2024],
                "test_years": [2025],
            },
        },
        "sampling": {"enabled": False},
    }


def test_sampling_disabled_keeps_full_pipeline():
    cfg = _full_config()
    split_cfg, sampled = _apply_sampling(cfg)
    assert sampled is None
    assert split_cfg["by_year"]["train_years"] == [2018, 2019, 2020, 2021, 2022, 2023]


def test_sampling_enabled_shrinks_split_and_stations():
    cfg = _full_config()
    cfg["sampling"] = {
        "enabled": True,
        "n_stations_per_region": 1,
        "train_years": [2022, 2023],
    }
    split_cfg, sampled = _apply_sampling(cfg)
    assert split_cfg["by_year"]["train_years"] == [2022, 2023]
    assert sampled is not None
    assert len(sampled) == len(all_regions())  # 1 por región


def test_sampling_overrides_epochs_and_seeds():
    cfg = _full_config()
    cfg["sampling"] = {
        "enabled": True,
        "n_stations_per_region": 1,
        "train_years": [2022, 2023],
        "max_epochs_override": 5,
        "n_seeds_override": 2,
    }
    model_cfg = {
        "model": {"name": "lstm"},
        "training": {"epochs": 50, "lr": 1e-3, "batch_size": 64,
                     "early_stopping_patience": 8},
    }
    stations = ["A1", "A2", "B3"]  # arbitrarias
    new_model_cfg, new_seeds, new_stations = _apply_sampling_overrides(
        cfg, model_cfg, seeds=5, stations=stations,
    )
    assert new_model_cfg["training"]["epochs"] == 5
    assert new_seeds == 2
    # No mutación del original
    assert model_cfg["training"]["epochs"] == 50


def test_sampling_disabled_passthrough_in_runner():
    cfg = _full_config()
    model_cfg = {"model": {"name": "lstm"},
                 "training": {"epochs": 50, "lr": 1e-3, "batch_size": 64,
                              "early_stopping_patience": 8}}
    stations = ["A1", "A2"]
    new_model_cfg, new_seeds, new_stations = _apply_sampling_overrides(
        cfg, model_cfg, seeds=5, stations=stations,
    )
    assert new_model_cfg is model_cfg
    assert new_seeds == 5
    assert new_stations == stations
