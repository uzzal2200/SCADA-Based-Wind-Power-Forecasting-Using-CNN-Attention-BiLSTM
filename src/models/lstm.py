"""
Unidirectional LSTM baseline (Baseline 2, Sec. III-E).

Two stacked unidirectional LSTM layers (H=128), no attention, no
convolutional preprocessing. The final hidden state h_T is decoded
directly by the linear output head — the simplest recurrent baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wtpf.models.common import SingleLinearHead


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_features: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        rnn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )
        self.head = SingleLinearHead(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (B, H) — final layer's final hidden state
        return self.head(last_hidden)
