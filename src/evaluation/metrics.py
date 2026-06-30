"""
Evaluation metrics (Sec. III-G, Eq. 22-27).

All six complementary metrics used in the comparative study, computed on
inverse-transformed predictions (physical units, kW):

    RMSE   - Eq. 22  - root mean square error
    MAE    - Eq. 23  - mean absolute error
    R^2    - Eq. 24  - coefficient of determination
    MAPE   - Eq. 25  - mean absolute percentage error, evaluated only
                        where y >= 5 kW to avoid near-zero-denominator
                        instability
    sMAPE  - Eq. 26  - symmetric MAPE, bounded & symmetric
    PSNR   - Eq. 27  - peak signal-to-noise ratio (dB), borrowed from
                        signal processing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. 22."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. 23."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. 24."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, power_threshold_kw: float = 5.0) -> float:
    """Eq. 25. Evaluated only where y_true >= power_threshold_kw."""
    mask = y_true >= power_threshold_kw
    if not np.any(mask):
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Eq. 26."""
    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return float(100.0 * np.mean(numerator / denominator))


def psnr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Eq. 27. y_max is the maximum *observed* power in the evaluated set."""
    mse = np.mean((y_true - y_pred) ** 2)
    y_max = np.max(y_true)
    return float(20.0 * np.log10(y_max / (np.sqrt(mse) + eps)))


@dataclass
class MetricBundle:
    r2: float
    rmse: float
    mae: float
    mape: float
    smape: float
    psnr: float

    def to_dict(self) -> dict:
        return {
            "R2": self.r2,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "MAPE": self.mape,
            "sMAPE": self.smape,
            "PSNR": self.psnr,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"R2={self.r2:.4f}  RMSE={self.rmse:.2f}kW  MAE={self.mae:.2f}kW  "
            f"MAPE={self.mape:.2f}%  sMAPE={self.smape:.2f}%  PSNR={self.psnr:.2f}dB"
        )


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    power_threshold_kw: float = 5.0,
    smape_eps: float = 1e-8,
    psnr_eps: float = 1e-8,
) -> MetricBundle:
    """Compute all six metrics in one call, on physical-unit (kW) arrays."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return MetricBundle(
        r2=r2(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred, power_threshold_kw),
        smape=smape(y_true, y_pred, smape_eps),
        psnr=psnr(y_true, y_pred, psnr_eps),
    )
