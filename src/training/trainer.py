"""Loop de entrenamiento PyTorch con early stopping, checkpoint y MLflow opcional."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import get_logger

from .callbacks import Checkpoint, EarlyStopping, HistoryLogger, Timer

log = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "auto",
        grad_clip: float | None = 1.0,
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = (
            model.configure_optimizers(lr=lr, weight_decay=weight_decay)
            if hasattr(model, "configure_optimizers")
            else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        )
        self.grad_clip = grad_clip
        self.loss_fn = loss_fn or nn.MSELoss()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ------------------------------------------------------------ epoch step

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        losses = []
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self._step(batch)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        losses = []
        for batch in loader:
            loss, _ = self._step(batch)
            losses.append(loss.item())
        return float(np.mean(losses))

    # ------------------------------------------------------------ public API

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 8,
        checkpoint_path: str | Path | None = None,
    ) -> dict:
        history = HistoryLogger()
        timer = Timer()
        es = EarlyStopping(patience=patience)
        ckpt = Checkpoint(Path(checkpoint_path)) if checkpoint_path else None

        for epoch in range(1, epochs + 1):
            timer.epoch_start()
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate(val_loader)
            dt = timer.epoch_end()

            history.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss, epoch_time_s=dt)
            saved = ckpt.maybe_save(self.model, val_loss) if ckpt else False

            log.info(
                "epoch %02d/%02d — train=%.4f val=%.4f (%.1fs)%s",
                epoch, epochs, train_loss, val_loss, dt, " *" if saved else "",
            )
            if es.step(val_loss):
                log.info("Early stopping en epoch %d (best val=%.4f)", epoch, es.best)
                break

        return {"history": history.history, "best_val_loss": es.best, "total_s": timer.total}

    @torch.no_grad()
    def predict(self, loader: Iterable) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        ys, yh = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            ys.append(y.cpu().numpy())
            yh.append(self.model(x).cpu().numpy())
        return np.concatenate(yh), np.concatenate(ys)
