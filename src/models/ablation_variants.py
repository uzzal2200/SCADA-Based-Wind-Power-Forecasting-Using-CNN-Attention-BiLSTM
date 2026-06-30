"""
Component ablation variants of the proposed CNN-Attention-BiLSTM
(Sec. IV-I, Table 10). Each variant removes or replaces exactly one
architectural component, all else held fixed, to isolate its individual
contribution to predictive accuracy:

    w/o CNN        : Input -> BiLSTM -> Attn -> FC
    w/o Attention  : CNN -> BiLSTM -> Mean-Pool -> FC
    Uni-LSTM       : CNN -> UniLSTM -> Attn -> FC
    w/o Soft-Pool  : CNN -> BiLSTM -> last hidden state -> FC

Reported hierarchy of component importance (Table 10, Fig. 22):
Attention > BiLSTM directionality > CNN > Soft-Pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wtpf.models.common import FCOutputHead
from wtpf.models.layers import AdditiveSelfAttention, LastTimestepReadout, MeanPool


class _CNNBlock(nn.Module):
    """Reusable two-layer 1D CNN block, identical to the proposed model."""

    def __init__(self, input_features: int, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_features, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = torch.relu(self.bn1(self.conv1(h)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        return h.transpose(1, 2)


class WithoutCNN(nn.Module):
    """Ablation: Input -> BiLSTM -> Attention -> FC (no CNN preprocessing)."""

    def __init__(
        self,
        input_features: int = 8,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        rnn_dropout: float = 0.2,
        attn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
        **_unused,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_features,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if bilstm_layers > 1 else 0.0,
        )
        attn_dim = bilstm_hidden * 2
        self.attention = AdditiveSelfAttention(attn_dim, dropout=attn_dropout)
        self.head = FCOutputHead(attn_dim, fc_hidden_dim, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_bi, _ = self.bilstm(x)
        context, _ = self.attention(h_bi)
        return self.head(context)


class WithoutAttention(nn.Module):
    """Ablation: CNN -> BiLSTM -> Mean-Pool -> FC (no attention)."""

    def __init__(
        self,
        input_features: int = 8,
        cnn_channels: int = 64,
        kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        rnn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
        **_unused,
    ):
        super().__init__()
        self.cnn = _CNNBlock(input_features, cnn_channels, kernel_size, cnn_dropout)
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if bilstm_layers > 1 else 0.0,
        )
        self.pool = MeanPool(dropout=fc_dropout)
        self.head = FCOutputHead(bilstm_hidden * 2, fc_hidden_dim, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_cnn = self.cnn(x)
        h_bi, _ = self.bilstm(h_cnn)
        context = self.pool(h_bi)
        return self.head(context)


class UniLSTMVariant(nn.Module):
    """Ablation: CNN -> Unidirectional LSTM -> Attention -> FC."""

    def __init__(
        self,
        input_features: int = 8,
        cnn_channels: int = 64,
        kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        rnn_dropout: float = 0.2,
        attn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
        **_unused,
    ):
        super().__init__()
        self.cnn = _CNNBlock(input_features, cnn_channels, kernel_size, cnn_dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=rnn_dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = AdditiveSelfAttention(lstm_hidden, dropout=attn_dropout)
        self.head = FCOutputHead(lstm_hidden, fc_hidden_dim, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_cnn = self.cnn(x)
        h_seq, _ = self.lstm(h_cnn)
        context, _ = self.attention(h_seq)
        return self.head(context)


class WithoutSoftPool(nn.Module):
    """Ablation: CNN -> BiLSTM -> last-hidden-state readout -> FC
    (replaces learned soft-attention pooling with conventional readout)."""

    def __init__(
        self,
        input_features: int = 8,
        cnn_channels: int = 64,
        kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        rnn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
        **_unused,
    ):
        super().__init__()
        self.cnn = _CNNBlock(input_features, cnn_channels, kernel_size, cnn_dropout)
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if bilstm_layers > 1 else 0.0,
        )
        self.readout = LastTimestepReadout(dropout=fc_dropout)
        self.head = FCOutputHead(bilstm_hidden * 2, fc_hidden_dim, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_cnn = self.cnn(x)
        h_bi, _ = self.bilstm(h_cnn)
        context = self.readout(h_bi)
        return self.head(context)


ABLATION_REGISTRY = {
    "no_cnn": WithoutCNN,
    "no_attention": WithoutAttention,
    "uni_lstm": UniLSTMVariant,
    "no_soft_pool": WithoutSoftPool,
}
