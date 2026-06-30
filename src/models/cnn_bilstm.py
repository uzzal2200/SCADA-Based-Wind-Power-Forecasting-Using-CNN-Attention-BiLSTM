"""
CNN-BiLSTM baseline (Baseline 4, Sec. III-E).

A two-layer 1D CNN (C=64) followed by a two-layer BiLSTM (H=128 per
direction). The BiLSTM's final concatenated hidden state is decoded by
the linear head. Isolates the combined contribution of convolutional
preprocessing + bidirectional recurrence *without* attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wtpf.models.common import SingleLinearHead


class CNNBiLSTMForecaster(nn.Module):
    def __init__(
        self,
        input_features: int = 8,
        cnn_channels: int = 64,
        kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        hidden_size: int = 128,
        bilstm_layers: int = 2,
        rnn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_features, cnn_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if bilstm_layers > 1 else 0.0,
        )
        self.head = SingleLinearHead(hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                       # (B, F, T)
        h = torch.relu(self.bn1(self.conv1(h)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.cnn_dropout(h)
        h = h.transpose(1, 2)                        # (B, T, C)

        _, (h_n, _) = self.lstm(h)
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        context = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.head(context)
