"""
Power-band performance disaggregation (Sec. IV-F, Fig. 19).

Disaggregates test-set performance into four operational power bands —
Low (0-100 kW), Medium (100-300 kW), High (300-600 kW) and Rated
(600-900 kW) — computing all four error metrics (RMSE, MAE, MAPE,
sMAPE) within each band. MAPE/sMAPE are characteristically inflated in
the Low band owing to near-zero power denominators (see Sec. IV-F).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from wtpf.evaluation.metrics import compute_all_metrics, MetricBundle

DEFAULT_POWER_BANDS: Dict[str, Tuple[float, float]] = {
    "Low": (0, 100),
    "Medium": (100, 300),
    "High": (300, 600),
    "Rated": (600, 900),
}


def power_band_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bands: Dict[str, Tuple[float, float]] = DEFAULT_POWER_BANDS,
    power_threshold_kw: float = 5.0,
) -> Dict[str, MetricBundle]:
    """Compute per-power-band MetricBundle, binned by the *actual*
    (ground-truth) power output.
    """
    results: Dict[str, MetricBundle] = {}
    for name, (lo, hi) in bands.items():
        mask = (y_true >= lo) & (y_true < hi)
        if not np.any(mask):
            continue
        results[name] = compute_all_metrics(
            y_true[mask], y_pred[mask], power_threshold_kw=power_threshold_kw
        )
    return results
