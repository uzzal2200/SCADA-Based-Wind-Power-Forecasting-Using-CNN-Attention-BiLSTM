"""
Early stopping callback (Sec. III-F).

Training halts when the validation Huber loss fails to improve for
`patience` consecutive epochs (15 in the paper), restoring the best
checkpoint seen so far.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_state: Optional[dict] = None
        self.should_stop: bool = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Call once per epoch with the current validation loss.

        Returns True if this is a new best (checkpoint was updated).
        """
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def restore_best(self, model: nn.Module) -> nn.Module:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
