"""
Attention-BiLSTM baseline (Baseline 5, Sec. III-E).

A two-layer BiLSTM (H=128 per direction) augmented with the single-head
additive (Bahdanau-style) self-attention mechanism, producing a context
vector c = sum_t alpha_t h_t from the *complete* BiLSTM output sequence
(not just the final hidden state). Isolates the incremental contribution
of attention *without* a convolutional component.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wtpf.models.common import SingleLinearHead
from wtpf.models.layers import AdditiveSelfAttention


class AttentionBiLSTMForecaster(nn.Module):
    def __init__(
        self,
        input_features: int = 8,
        hidden_size: int = 128,
        bilstm_layers: int = 2,
        rnn_dropout: float = 0.2,
        attn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if bilstm_layers > 1 else 0.0,
        )
        self.attention = AdditiveSelfAttention(hidden_size * 2, dropout=attn_dropout)
        self.head = SingleLinearHead(hidden_size * 2)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        h_seq, _ = self.lstm(x)               # (B, T, 2H) — full sequence
        context, weights = self.attention(h_seq)
        out = self.head(context)
        if return_attention:
            return out, weights
        return out
