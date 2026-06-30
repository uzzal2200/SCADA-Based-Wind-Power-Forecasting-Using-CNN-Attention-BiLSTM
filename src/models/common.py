"""
Shared fully-connected output head used by every architecture in this
study (Sec. III-D.4, Eq. 20):

    P_hat = W2 * ReLU(W1 * c + b1) + b2

A two-layer MLP with an intermediate ReLU and dropout, mapping a
fixed-length context vector down to a single scalar power prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FCOutputHead(nn.Module):
    """Two-layer FC head: Linear -> ReLU -> Dropout -> Linear(->1).

    Used by the CNN baseline and the proposed CNN-Attention-BiLSTM model
    (Table 3, layers 9-10), reproducing their exact reported parameter
    counts (Table 6: CNN = 18,433; CNN-Attention-BiLSTM = 624,898).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SingleLinearHead(nn.Module):
    """Single linear output layer: in_dim -> 1.

    Used by the LSTM, BiLSTM, CNN-BiLSTM and Attention-BiLSTM baselines
    (Sec. III-E: "the final hidden state is directly passed to the linear
    output layer" / "decoded using a linear head"), reproducing their
    exact reported parameter counts (Table 6).
    """

    def __init__(self, in_dim: int, **_unused):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
