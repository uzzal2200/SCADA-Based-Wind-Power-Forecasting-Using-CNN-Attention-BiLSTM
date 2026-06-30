#!/usr/bin/env python3
"""
Component ablation study (Sec. IV-I, Table 10, Fig. 22): trains the four
reduced variants (w/o CNN, w/o Attention, Uni-LSTM, w/o Soft-Pool) plus
the full proposed model on identical data splits, and saves the
comparison table and figure.

Usage:
    python scripts/04_run_ablation_study.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from wtpf.config import load_config  # noqa: E402
from wtpf.experiment import run_experiment  # noqa: E402
from wtpf.pipeline import build_datasets  # noqa: E402
from wtpf.utils.device import get_device  # noqa: E402
from wtpf.utils.logger import get_logger  # noqa: E402
from wtpf.utils.seed import set_global_seed  # noqa: E402
from wtpf.visualization import savefig
from wtpf.visualization.result_plots import plot_component_ablation

logger = get_logger("wtpf.scripts.ablation")

DISPLAY_NAMES = {
    "no_cnn": "w/o CNN",
    "no_attention": "w/o Attention",
    "uni_lstm": "Uni-LSTM",
    "no_soft_pool": "w/o Soft-Pool",
    "cnn_attention_bilstm": "Full Proposed",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(cfg.training.checkpoint_dir) / "ablation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prepared = build_datasets(cfg)

    variants = list(cfg.ablation)
    # "full" maps to the proposed model's registry key
    variant_keys = ["cnn_attention_bilstm" if v == "full" else v for v in variants]

    rows = {}
    for key in variant_keys:
        logger.info("=" * 70)
        logger.info("Ablation variant: %s", DISPLAY_NAMES.get(key, key))
        logger.info("=" * 70)
        set_global_seed(cfg.project.seed)
        result = run_experiment(
            key, cfg, prepared, device, checkpoint_dir=checkpoint_dir, logger=logger.info
        )
        rows[DISPLAY_NAMES.get(key, key)] = {"RMSE": result.metrics.rmse, "R2": result.metrics.r2}

    ablation_df = pd.DataFrame(rows).T
    ablation_df.index.name = "Variant"

    full_rmse = ablation_df.loc["Full Proposed", "RMSE"]
    ablation_df["Delta RMSE (%)"] = 100 * (ablation_df["RMSE"] - full_rmse) / full_rmse

    out_csv = results_dir / "table10_ablation_results.csv"
    ablation_df.to_csv(out_csv)
    logger.info("Saved ablation results (Table 10 style) to %s", out_csv)
    print(ablation_df.round(4))

    fig = plot_component_ablation(ablation_df[["RMSE", "R2"]])
    fig_path = Path(args.figures_dir) / "fig22_component_ablation.png"
    savefig(fig, str(fig_path))
    logger.info("Saved ablation figure to %s", fig_path)


if __name__ == "__main__":
    main()
