"""
Proposed CNN-Attention-BiLSTM hybrid architecture (Sec. III-D, Table 3).

Data flow (Eq. 6):

    X --CNN--> H_cnn --BiLSTM--> H_bi --Attn--> c --FC--> P_hat

Layer-by-layer (Table 3), for T=144, F=8:

     1  Input                          (B,144,8)  -> (B,144,8)
     2  Conv1D(64,k=3,p=1)+BN+ReLU     (B,8,144)  -> (B,64,144)
     3  Conv1D(64,k=3,p=1)+BN+ReLU     (B,64,144) -> (B,64,144)
     4  Dropout(0.2) + transpose back  (B,64,144) -> (B,144,64)
     5  BiLSTM layer 1 (H=128x2=256)   (B,144,64) -> (B,144,256)
     6  BiLSTM layer 2 (H=128x2=256)   (B,144,256)-> (B,144,256)
     7  Additive self-attention        (B,144,256)-> (B,256)
     8  Dropout(0.2)                   (B,256)    -> (B,256)
     9  FC(256->64)+ReLU+Dropout(0.2)  (B,256)    -> (B,64)
    10  FC(64->1)                      (B,64)     -> (B,1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wtpf.models.common import FCOutputHead
from wtpf.models.layers import AdditiveSelfAttention


class CNNAttentionBiLSTM(nn.Module):
    """The proposed hybrid model (best-performing architecture in the paper:
    R^2=0.9716, RMSE=35.42 kW on the held-out DkIT test partition)."""

    def __init__(
        self,
        input_features: int = 8,
        cnn_channels: int = 64,
        kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        rnn_dropout: float = 0.2,
        attn_dropout: float = 0.2,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
    ):
        super().__init__()

        # --- CNN block (local temporal feature extraction, Eq. 7) ---
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_features, cnn_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        # --- BiLSTM block (bidirectional temporal modelling, Eq. 8-16) ---
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if bilstm_layers > 1 else 0.0,
        )

        # --- Single-head additive self-attention (Eq. 17-19) ---
        attn_dim = bilstm_hidden * 2
        self.attention = AdditiveSelfAttention(attn_dim, dropout=attn_dropout)

        # --- Output head (Eq. 20) ---
        self.head = FCOutputHead(attn_dim, fc_hidden_dim, fc_dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Parameters
        ----------
        x:
            Input SCADA window, shape (B, T, F) with T=144, F=8.
        return_attention:
            If True, also return the (B, T) attention weight tensor for
            interpretability analysis (Sec. IV-B).
        """
        # CNN block: (B, T, F) -> (B, F, T) -> conv -> (B, T, C)
        h = x.transpose(1, 2)
        h = torch.relu(self.bn1(self.conv1(h)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.cnn_dropout(h)
        h_cnn = h.transpose(1, 2)  # (B, T, C=64)

        # BiLSTM block: (B, T, C) -> (B, T, 2H=256)
        h_bi, _ = self.bilstm(h_cnn)

        # Attention pooling: (B, T, 2H) -> (B, 2H)
        context, attn_weights = self.attention(h_bi)

        # Output head: (B, 2H) -> (B,)
        out = self.head(context)

        if return_attention:
            return out, attn_weights
        return out
