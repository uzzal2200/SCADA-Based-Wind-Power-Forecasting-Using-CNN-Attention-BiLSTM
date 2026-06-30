"""
Training and inference loop (Algorithm 1, Steps 2-4; Sec. III-F).

Implements, per mini-batch:

    H_cnn <- CNN(X_b)
    H_bi  <- BiLSTM(H_cnn)
    c     <- Attn(H_bi)
    y_b   <- FC(c)

and, per epoch:

    * Huber loss (delta=1.0) backpropagation
    * gradient-norm clipping (max_norm=1.0)
    * Adam (or AdamW) update, weight_decay=1e-5, lr=1e-3
    * validation-loss-driven ReduceLROnPlateau (patience=20, factor=0.5)
    * early stopping (patience=15) with best-checkpoint restoration

`Trainer.predict` performs Step 4 (held-out inference + inverse min-max
transform back to physical units, kW).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wtpf.training.callbacks import EarlyStopping


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    epoch_time_s: list = field(default_factory=list)
    best_epoch: int = -1
    total_train_time_s: float = 0.0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer_name: str = "adam",
        huber_delta: float = 1.0,
        grad_clip_max_norm: float = 1.0,
        scheduler_patience: int = 20,
        scheduler_factor: float = 0.5,
        early_stopping_patience: int = 15,
        seed: int = 42,
        log_every: int = 1,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.huber_delta = huber_delta
        self.grad_clip_max_norm = grad_clip_max_norm
        self.log_every = log_every
        self.log = logger or (lambda msg: print(msg))

        self.criterion = nn.HuberLoss(delta=huber_delta)

        opt_cls = torch.optim.AdamW if optimizer_name.lower() == "adamw" else torch.optim.Adam
        self.optimizer = opt_cls(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(mode=train)
        total_loss, n_samples = 0.0, 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                    self.optimizer.step()

                batch_size = X_batch.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size

        return total_loss / max(n_samples, 1)

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        checkpoint_path: Optional[str | Path] = None,
    ) -> TrainingHistory:
        history = TrainingHistory()
        t_start = time.time()

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            epoch_time = time.time() - t0

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.lr.append(current_lr)
            history.epoch_time_s.append(epoch_time)

            improved = self.early_stopping.step(val_loss, self.model)
            if improved:
                history.best_epoch = epoch
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)

            if epoch % self.log_every == 0 or improved:
                flag = " *" if improved else ""
                self.log(
                    f"Epoch {epoch:3d}/{max_epochs} | "
                    f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
                    f"lr={current_lr:.2e} | {epoch_time:.1f}s{flag}"
                )

            if self.early_stopping.should_stop:
                self.log(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best val_loss={self.early_stopping.best_loss:.5f} "
                    f"at epoch {history.best_epoch})."
                )
                break

        history.total_train_time_s = time.time() - t_start
        self.early_stopping.restore_best(self.model)
        return history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a held-out loader.

        Returns scaled (y_true, y_pred) arrays — apply
        `target_scaler.inverse_transform` to obtain physical units (kW),
        matching Algorithm 1 Step 4.
        """
        self.model.eval()
        y_true_all, y_pred_all = [], []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_pred = self.model(X_batch).cpu().numpy()
            y_true_all.append(y_batch.numpy())
            y_pred_all.append(y_pred)
        return np.concatenate(y_true_all), np.concatenate(y_pred_all)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
