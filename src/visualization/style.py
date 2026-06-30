"""Shared matplotlib/seaborn styling so every figure in the project has a
consistent, publication-quality look (matching the 300-600 DPI figures
used in the paper)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "CNN": "#FF7F0E",
    "LSTM": "#1F77B4",
    "BiLSTM": "#9467BD",
    "CNN-BiLSTM": "#2CA02C",
    "Attention-BiLSTM": "#8C564B",
    "CNN-Attention-BiLSTM (Ours)": "#D62728",
}

MODEL_ORDER = ["CNN", "LSTM", "BiLSTM", "CNN-BiLSTM", "Attention-BiLSTM", "CNN-Attention-BiLSTM (Ours)"]


def set_style(dpi: int = 300, font_size: int = 10) -> None:
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["font.size"] = font_size
    sns.set_style("whitegrid")


def savefig(fig, path: str, dpi: int = 300) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
