"""
Statistical significance testing (Sec. IV-E).

Pairwise Wilcoxon Signed-Rank Tests on absolute prediction errors of the
proposed model vs. each baseline, with Bonferroni correction for multiple
comparisons (alpha* = 0.05 / 5 = 0.01) and rank-biserial effect size
r = |Z| / sqrt(n), following Cohen's conventions:

    r >= 0.1  -> small
    r >= 0.3  -> medium
    r >= 0.5  -> large
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import norm


def _effect_label(r: float) -> str:
    if r >= 0.5:
        return "Large"
    if r >= 0.3:
        return "Medium"
    if r >= 0.1:
        return "Small"
    return "Negligible"


@dataclass
class WilcoxonResult:
    comparison: str
    statistic: float
    z: float
    p_value: float
    effect_size_r: float
    effect_label: str
    significant: bool


def _wilcoxon_z_statistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Run the Wilcoxon signed-rank test and additionally derive the
    normal-approximation Z statistic (scipy does not expose Z directly
    for large samples using the exact/auto method)."""
    diff = x - y
    diff = diff[diff != 0]
    n = len(diff)
    stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

    # Normal approximation Z, consistent with how SciPy derives p for
    # large n (continuity-corrected).
    mean_w = n * (n + 1) / 4.0
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (stat - mean_w) / std_w if std_w > 0 else 0.0
    return float(stat), float(z)


def wilcoxon_test(
    abs_err_proposed: np.ndarray,
    abs_err_baseline: np.ndarray,
    comparison_label: str,
    alpha: float = 0.01,
) -> WilcoxonResult:
    """Pairwise Wilcoxon Signed-Rank test between the proposed model's
    absolute errors and a baseline's absolute errors.
    """
    stat, z = _wilcoxon_z_statistic(abs_err_proposed, abs_err_baseline)
    n = len(abs_err_proposed)
    p_value = float(2 * norm.sf(abs(z)))
    r = abs(z) / np.sqrt(n)

    return WilcoxonResult(
        comparison=comparison_label,
        statistic=stat,
        z=z,
        p_value=p_value,
        effect_size_r=float(r),
        effect_label=_effect_label(r),
        significant=p_value < alpha,
    )


def run_bonferroni_wilcoxon_suite(
    abs_errors_by_model: Dict[str, np.ndarray],
    proposed_key: str = "cnn_attention_bilstm",
    family_alpha: float = 0.05,
) -> Dict[str, WilcoxonResult]:
    """Run the full Bonferroni-corrected pairwise comparison suite
    (Sec. IV-E, Table 8): proposed model vs. every other model in the
    dict, correcting for the number of comparisons performed.
    """
    baselines = {k: v for k, v in abs_errors_by_model.items() if k != proposed_key}
    n_comparisons = len(baselines)
    alpha_star = family_alpha / max(n_comparisons, 1)

    results: Dict[str, WilcoxonResult] = {}
    proposed_errors = abs_errors_by_model[proposed_key]
    for name, baseline_errors in baselines.items():
        n = min(len(proposed_errors), len(baseline_errors))
        results[name] = wilcoxon_test(
            proposed_errors[:n], baseline_errors[:n], comparison_label=f"Proposed vs {name}", alpha=alpha_star
        )
    return results
