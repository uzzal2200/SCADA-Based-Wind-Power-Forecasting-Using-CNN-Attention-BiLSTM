"""
CNN baseline (Baseline 1, Sec. III-E).

A two-layer 1D convolutional network (C=64 filters, k=3) with batch
normalisation and ReLU, followed by global average pooling and a
two-layer fully-connected head. No recurrent or attention component —
the weakest sequential baseline in the comparative study.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wtpf.models.common import FCOutputHead


class CNNForecaster(nn.Module):
    def __init__(
        self,
        input_features: int = 8,
        cnn_channels: int = 64,
        kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_features, cnn_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.dropout = nn.Dropout(cnn_dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = FCOutputHead(cnn_channels, fc_hidden_dim, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for Conv1d
        h = x.transpose(1, 2)
        h = torch.relu(self.bn1(self.conv1(h)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        h = self.pool(h).squeeze(-1)  # (B, C)
        return self.head(h)
