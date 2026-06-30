"""
Single-head additive (Bahdanau-style) self-attention pooling layer
(Sec. III-D.3, Eq. 17-19).

Given a sequence of hidden states H in R^(B x T x D), computes a scalar
score per timestep via a learned projection, normalises with softmax,
and returns the attention-weighted context vector c in R^(B x D) together
with the attention weights (useful for interpretability, e.g. Sec. IV-B).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class AdditiveSelfAttention(nn.Module):
    """Additive self-attention pooling over the time axis.

        e_t     = w^T h_t                      (Eq. 17)
        alpha_t = softmax_t(e_t)                (Eq. 18)
        c       = sum_t alpha_t * h_t           (Eq. 19)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h: Tensor of shape (B, T, D) — e.g. the full BiLSTM output sequence.

        Returns
        -------
        context: Tensor of shape (B, D)
        weights: Tensor of shape (B, T) — attention weights, sum to 1 over T.
        """
        scores = self.score_proj(h).squeeze(-1)              # (B, T)
        weights = torch.softmax(scores, dim=-1)                # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, D)
        context = self.dropout(context)
        return context, weights


class MeanPool(nn.Module):
    """Unweighted mean-pooling over the time axis.

    Used as the "w/o Attention" ablation variant (Table 10): replaces the
    learned additive-attention context vector with a simple average.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        context = h.mean(dim=1)
        return self.dropout(context)


class LastTimestepReadout(nn.Module):
    """Readout using only the final timestep's hidden state.

    Used as the "w/o Soft-Pool" ablation variant (Table 10): replaces the
    learned attention pooling with the conventional last-hidden-state
    decoding strategy used by the plain recurrent baselines.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        context = h[:, -1, :]
        return self.dropout(context)
