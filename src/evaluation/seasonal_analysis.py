"""
Seasonal performance decomposition (Sec. IV-D, Table 7).

Splits the test set into the four meteorological seasons (DJF / MAM /
JJA / SON) using the timestamp index aligned with the test predictions,
then computes per-season metrics. Note (as in the paper) that seasonal
R^2 is computed relative to the *within-season* conditional mean, which
is why it differs from the aggregate R^2 in Table 5.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from wtpf.evaluation.metrics import compute_all_metrics, MetricBundle

DEFAULT_SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def seasonal_breakdown(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    power_threshold_kw: float = 5.0,
) -> Dict[str, MetricBundle]:
    """Compute per-season MetricBundle for DJF/MAM/JJA/SON.

    Parameters
    ----------
    timestamps:
        DatetimeIndex aligned 1:1 with y_true/y_pred (e.g. `test_index`
        returned by `prepare_datasets`).
    y_true, y_pred:
        Inverse-transformed (kW) ground truth and predictions.
    """
    season = pd.Series(timestamps.month, index=range(len(timestamps))).map(DEFAULT_SEASON_MAP)

    results: Dict[str, MetricBundle] = {}
    for s in SEASON_ORDER:
        mask = (season == s).to_numpy()
        if not np.any(mask):
            continue
        results[s] = compute_all_metrics(
            y_true[mask], y_pred[mask], power_threshold_kw=power_threshold_kw
        )
    return results


def relative_rmse_improvement(
    proposed_rmse: float, reference_rmse: float
) -> float:
    """Percentage RMSE change of the proposed model vs. a reference
    (e.g. the CNN baseline), matching the ``Delta RMSE`` column of
    Table 7 — negative values denote an improvement."""
    return 100.0 * (proposed_rmse - reference_rmse) / reference_rmse
