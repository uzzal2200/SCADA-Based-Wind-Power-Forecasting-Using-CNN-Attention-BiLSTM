"""
Bidirectional LSTM baseline (Baseline 3, Sec. III-E, Eq. 8-16).

A two-layer BiLSTM (H=128 per direction, 2H=256 concatenated) without
attention. The concatenated final forward/backward hidden state
[h_T_fwd || h_T_bwd] is decoded by the linear output head — captures
bidirectional context but lacks selective time-step weighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wtpf.models.common import SingleLinearHead


class BiLSTMForecaster(nn.Module):
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
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )
        self.head = SingleLinearHead(hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, B, H) -> take final layer's fwd & bwd states
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        context = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2H)
        return self.head(context)
