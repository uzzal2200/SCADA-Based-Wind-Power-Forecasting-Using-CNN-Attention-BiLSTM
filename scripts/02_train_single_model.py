#!/usr/bin/env python3
"""
Train and evaluate a single architecture (any key in
`wtpf.models.MODEL_REGISTRY`, e.g. the proposed "cnn_attention_bilstm" or
any baseline / ablation variant).

Usage:
    python scripts/02_train_single_model.py --model cnn_attention_bilstm
    python scripts/02_train_single_model.py --model lstm --config configs/config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wtpf.config import load_config  # noqa: E402
from wtpf.experiment import run_experiment  # noqa: E402
from wtpf.models.factory import MODEL_REGISTRY  # noqa: E402
from wtpf.pipeline import build_datasets  # noqa: E402
from wtpf.utils.device import get_device  # noqa: E402
from wtpf.utils.logger import get_logger  # noqa: E402
from wtpf.utils.seed import set_global_seed  # noqa: E402

logger = get_logger("wtpf.scripts.train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--model", default="cnn_attention_bilstm", choices=sorted(MODEL_REGISTRY.keys())
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.project.seed)
    device = get_device(cfg.project.device)
    logger.info("Using device: %s", device)

    prepared = build_datasets(cfg)

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(
        args.model, cfg, prepared, device, checkpoint_dir=checkpoint_dir, logger=logger.info
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model,
        "n_params": result.n_params,
        "size_mb": result.size_mb,
        "train_time_s": result.history.total_train_time_s,
        "inference_latency_ms": result.inference_latency_ms,
        "best_epoch": result.history.best_epoch,
        "metrics": result.metrics.to_dict(),
    }
    out_path = results_dir / f"{args.model}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary to %s", out_path)
    logger.info("Checkpoint saved to %s", checkpoint_dir / f"{args.model}.pt")


if __name__ == "__main__":
    main()
