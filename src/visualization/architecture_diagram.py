"""
Programmatic architecture-diagram generator.

Renders the two schematic diagrams used in the paper (Fig. 1 — overall
five-stage framework; Fig. 11 — proposed hybrid architecture) as clean,
reproducible matplotlib figures, so the README's diagrams are generated
by code rather than hand-drawn images.

Run directly:  python -m wtpf.visualization.architecture_diagram
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def _box(ax, xy, w, h, text, facecolor="#E8F0FE", edgecolor="#1A56DB", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return rect


def _arrow(ax, p1, p2, color="#444444"):
    arrow = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, color=color, linewidth=1.4)
    ax.add_patch(arrow)


def plot_overall_framework() -> plt.Figure:
    """Fig. 1 — five-stage overall forecasting framework."""
    stages = [
        ("1. Data Acquisition\n& Source", "Vestas V52 SCADA\n14 yr, 10-min, 22 channels"),
        ("2. Preprocessing &\nFeature Engineering", "Filter \u2022 sin/cos encode\nMin-Max scale"),
        ("3. Sequence\nConstruction & Split", "Sliding window T=144\n70/15/15 chronological"),
        ("4. Modelling &\nTraining", "CNN-Attn-BiLSTM\n+ 5 baselines"),
        ("5. Evaluation &\nAnalysis", "RMSE/MAE/R2/MAPE\nsMAPE/PSNR + tests"),
    ]

    fig, ax = plt.subplots(figsize=(16, 4))
    w, h, gap = 2.6, 1.6, 0.6
    x = 0.2
    for title, subtitle in stages:
        _box(ax, (x, 0.5), w, h, f"{title}\n\n{subtitle}", fontsize=8.5)
        if x + w + gap < 0.2 + len(stages) * (w + gap):
            _arrow(ax, (x + w, 0.5 + h / 2), (x + w + gap, 0.5 + h / 2))
        x += w + gap

    ax.set_xlim(0, x)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title("Overall SCADA-Based Wind Turbine Power Forecasting Framework", fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def plot_hybrid_architecture() -> plt.Figure:
    """Fig. 11 — proposed CNN-Attention-BiLSTM architecture, layer by layer."""
    layers = [
        ("Input Sequence", "T=144, F=8\n(B,144,8)"),
        ("CNN Feature\nExtractor", "2x Conv1D(k=3,C=64)\n+BN+ReLU+Dropout"),
        ("BiLSTM\n(2 layers)", "H=128/direction\n(B,144,256)"),
        ("Additive Self-\nAttention", "Linear 256->1\nsoftmax + weighted sum"),
        ("FC Output\nHead", "256->64->1\nReLU+Dropout"),
        ("Predicted\nPower", "P_hat(t+1)\n[kW]"),
    ]
    colors = ["#FDEBD0", "#D6EAF8", "#D5F5E3", "#FCF3CF", "#F5D7E0", "#E8DAEF"]

    fig, ax = plt.subplots(figsize=(18, 3.6))
    w, h, gap = 2.7, 1.7, 0.55
    x = 0.2
    for (title, subtitle), color in zip(layers, colors):
        _box(ax, (x, 0.4), w, h, f"{title}\n\n{subtitle}", facecolor=color, fontsize=8.5)
        if x + w + gap < 0.2 + len(layers) * (w + gap):
            _arrow(ax, (x + w, 0.4 + h / 2), (x + w + gap, 0.4 + h / 2))
        x += w + gap

    ax.set_xlim(0, x)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.set_title("Proposed CNN-Attention-BiLSTM Architecture", fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def render_all(output_dir: str | Path = "assets") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_overall_framework()
    fig1.savefig(output_dir / "overall_framework.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plot_hybrid_architecture()
    fig2.savefig(output_dir / "hybrid_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved diagrams to {output_dir}/overall_framework.png and {output_dir}/hybrid_architecture.png")


if __name__ == "__main__":
    render_all()
