"""
Residual diagnostics (Sec. IV-C, Fig. 16-17).

Utility functions supporting the four-panel residual analysis (signed
residual vs. actual power, residual histogram with Gaussian fit, residual
by wind-speed bin, temporal residual sequence) and the cross-model
absolute-error boxplot/CDF comparison.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def signed_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """y_hat - y (Sec. IV-C convention)."""
    return y_pred - y_true


def residual_gaussian_fit(residuals: np.ndarray) -> Tuple[float, float]:
    """(mu, sigma) of the residual distribution."""
    return float(np.mean(residuals)), float(np.std(residuals))


def residuals_by_wind_speed_bin(
    residuals: np.ndarray,
    wind_speed: np.ndarray,
    bins: Tuple[float, ...] = (0, 5, 10, 15, 20, 25),
) -> Dict[str, np.ndarray]:
    """Bucket residuals by wind speed bin for the boxplot panel of Fig. 16c."""
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]
    out: Dict[str, np.ndarray] = {}
    for i, label in enumerate(labels):
        mask = (wind_speed >= bins[i]) & (wind_speed < bins[i + 1])
        out[label] = residuals[mask]
    return out


def absolute_error_cdf(abs_errors: np.ndarray, thresholds: Tuple[float, ...] = (50, 100, 150)) -> Dict[float, float]:
    """Cumulative proportion of predictions within each error threshold
    (kW), matching the CDF reference lines in Fig. 17."""
    return {t: float(np.mean(abs_errors <= t)) for t in thresholds}
