"""Callbacks ligeros: early stopping, checkpoint, log de curvas y tiempo."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class EarlyStopping:
    patience: int = 8
    min_delta: float = 0.0
    best: float = field(default=float("inf"))
    counter: int = 0
    should_stop: bool = False

    def step(self, metric: float) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


@dataclass
class Checkpoint:
    path: Path
    best: float = field(default=float("inf"))

    def maybe_save(self, model: torch.nn.Module, metric: float) -> bool:
        if metric < self.best:
            self.best = metric
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
            return True
        return False


@dataclass
class HistoryLogger:
    history: dict[str, list[float]] = field(default_factory=dict)

    def log(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self.history.setdefault(k, []).append(float(v))


@dataclass
class Timer:
    t0: float = field(default_factory=time.time)
    epoch_t0: float | None = None
    epoch_times: list[float] = field(default_factory=list)

    def epoch_start(self) -> None:
        self.epoch_t0 = time.time()

    def epoch_end(self) -> float:
        assert self.epoch_t0 is not None
        dt = time.time() - self.epoch_t0
        self.epoch_times.append(dt)
        return dt

    @property
    def total(self) -> float:
        return time.time() - self.t0
