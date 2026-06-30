#!/usr/bin/env python3
"""
Bonferroni-corrected pairwise Wilcoxon Signed-Rank significance testing
(Sec. IV-E, Table 8) of the proposed model vs. every baseline, using the
predictions saved by `scripts/03_train_all_baselines.py`.

Usage:
    python scripts/05_statistical_significance.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wtpf.config import load_config  # noqa: E402
from wtpf.evaluation.statistical_tests import run_bonferroni_wilcoxon_suite  # noqa: E402
from wtpf.utils.logger import get_logger  # noqa: E402

logger = get_logger("wtpf.scripts.stats")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(args.results_dir)
    pred_dir = results_dir / "predictions"

    proposed_key = "cnn_attention_bilstm"
    abs_errors = {}
    for model_name in cfg.baselines:
        npz_path = pred_dir / f"{model_name}.npz"
        if not npz_path.exists():
            logger.warning("Missing predictions for %s (%s) — run scripts/03 first. Skipping.", model_name, npz_path)
            continue
        data = np.load(npz_path)
        abs_errors[model_name] = np.abs(data["y_true"] - data["y_pred"])

    if proposed_key not in abs_errors:
        raise RuntimeError(
            f"Predictions for the proposed model ('{proposed_key}') not found in {pred_dir}. "
            f"Run scripts/03_train_all_baselines.py first."
        )

    results = run_bonferroni_wilcoxon_suite(
        abs_errors,
        proposed_key=proposed_key,
        family_alpha=cfg.evaluation.statistical_test.bonferroni_alpha,
    )

    rows = {}
    for name, r in results.items():
        rows[r.comparison] = {
            "W": r.statistic,
            "Z": r.z,
            "p_value": r.p_value,
            "r (effect size)": r.effect_size_r,
            "effect": r.effect_label,
            "significant": r.significant,
        }
    table = pd.DataFrame(rows).T

    out_path = results_dir / "table8_statistical_tests.csv"
    table.to_csv(out_path)
    logger.info("Saved Wilcoxon test results (Table 8 style) to %s", out_path)
    print(table)


if __name__ == "__main__":
    main()
