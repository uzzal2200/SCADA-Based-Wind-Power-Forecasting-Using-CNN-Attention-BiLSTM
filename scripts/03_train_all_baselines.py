#!/usr/bin/env python3
"""
Train all six comparative architectures (CNN, LSTM, BiLSTM, CNN-BiLSTM,
Attention-BiLSTM, CNN-Attention-BiLSTM) on identical chronological data
splits, and save the Table 5 (performance) and Table 6 (complexity)
style comparison results, plus per-model raw predictions for downstream
statistical testing / seasonal / power-band analysis.

Usage:
    python scripts/03_train_all_baselines.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wtpf.config import load_config  # noqa: E402
from wtpf.experiment import run_experiment  # noqa: E402
from wtpf.pipeline import build_datasets, load_and_clean  # noqa: E402
from wtpf.utils.device import get_device  # noqa: E402
from wtpf.utils.logger import get_logger  # noqa: E402
from wtpf.utils.seed import set_global_seed  # noqa: E402

logger = get_logger("wtpf.scripts.train_all")

DISPLAY_NAMES = {
    "cnn": "CNN",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn_bilstm": "CNN-BiLSTM",
    "attention_bilstm": "Attention-BiLSTM",
    "cnn_attention_bilstm": "CNN-Attention-BiLSTM (Ours)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.project.seed)
    device = get_device(cfg.project.device)
    logger.info("Using device: %s", device)

    results_dir = Path(args.results_dir)
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing shared dataset splits (identical across all models)...")
    df = load_and_clean(cfg)
    prepared = build_datasets(cfg, df)
    np.save(results_dir / "test_timestamps.npy", prepared.test_index.to_numpy())
    if "WindSpeed" in df.columns:
        np.save(
            results_dir / "test_wind_speed.npy",
            df.loc[prepared.test_index, "WindSpeed"].to_numpy(),
        )

    performance_rows = {}
    complexity_rows = {}

    for model_name in cfg.baselines:
        logger.info("=" * 70)
        logger.info("Training: %s", model_name)
        logger.info("=" * 70)
        set_global_seed(cfg.project.seed)  # re-seed for a fair comparison per model

        result = run_experiment(
            model_name, cfg, prepared, device, checkpoint_dir=checkpoint_dir, logger=logger.info
        )

        display = DISPLAY_NAMES.get(model_name, model_name)
        performance_rows[display] = result.metrics.to_dict()
        complexity_rows[display] = {
            "Trainable Params": result.n_params,
            "Size (MB)": round(result.size_mb, 3),
            "Train Time (s)": round(result.history.total_train_time_s, 1),
            "Infer. Time (ms)": round(result.inference_latency_ms, 2),
        }

        np.savez(
            pred_dir / f"{model_name}.npz",
            y_true=result.y_true_kw,
            y_pred=result.y_pred_kw,
        )

    perf_df = pd.DataFrame(performance_rows).T
    perf_df.index.name = "Model"
    perf_path = results_dir / "table5_performance_comparison.csv"
    perf_df.to_csv(perf_path)
    logger.info("Saved performance comparison (Table 5 style) to %s", perf_path)
    print(perf_df.round(4))

    complexity_df = pd.DataFrame(complexity_rows).T
    complexity_df.index.name = "Model"
    complexity_path = results_dir / "table6_model_complexity.csv"
    complexity_df.to_csv(complexity_path)
    logger.info("Saved model complexity comparison (Table 6 style) to %s", complexity_path)
    print(complexity_df)


if __name__ == "__main__":
    main()
