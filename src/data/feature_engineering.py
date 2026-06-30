"""
Feature engineering (Sec. III-C.2 / III-C.3).

Implements the three engineered-feature families described in the paper:

1. Polar (sine/cosine) encoding of wind direction, resolving the
   0deg/360deg circular discontinuity (Eq. 1):

        dir_sin = sin(pi * theta / 180)
        dir_cos = cos(pi * theta / 180)

2. Cyclic time-of-day encoding (Eq. 2) — engineered but ultimately
   *excluded* from the final feature set (negligible |r| < 0.05 with the
   target, no measurable reduction in validation Huber loss):

        hour_sin = sin(2*pi*h / 24)
        hour_cos = cos(2*pi*h / 24)

3. IEC-style theoretical power curve (binned mean active power vs. wind
   speed) — likewise engineered but excluded from the final feature set.

`select_model_features` then returns the final eight-feature input
matrix x = [v_ws, dir_sin, dir_cos, sigma_v, omega_r, beta, T_n, T_g]
(Eq. 3) plus the target column.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def add_wind_direction_encoding(
    df: pd.DataFrame, wind_dir_col: str = "WindDirAbs"
) -> pd.DataFrame:
    """Add `dir_sin` / `dir_cos` columns (Eq. 1)."""
    out = df.copy()
    theta_rad = np.deg2rad(out[wind_dir_col])
    out["dir_sin"] = np.sin(theta_rad)
    out["dir_cos"] = np.cos(theta_rad)
    return out


def add_cyclic_time_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add `hour_sin` / `hour_cos` columns from the DatetimeIndex (Eq. 2).

    Note: excluded from the final 8-feature model input (see paper
    Sec. III-C.3) but retained here for completeness / ablation use.
    """
    out = df.copy()
    hour = out.index.hour.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def add_theoretical_power_curve(
    df: pd.DataFrame,
    wind_speed_col: str = "WindSpeed",
    power_col: str = "Power",
    bin_width: float = 0.5,
) -> pd.DataFrame:
    """Map each record to a binned mean (IEC-style) theoretical power curve.

    Bins wind speed into `bin_width` m/s intervals across the full
    dataset, averages active power within each bin, and maps that
    bin-average back to every record as `Theoretical_Power`. Excluded
    from the final feature set (Sec. III-C.3) but useful as a diagnostic
    / physically grounded reference feature.
    """
    out = df.copy()
    max_ws = float(np.ceil(out[wind_speed_col].max())) + bin_width
    bins = np.arange(0, max_ws, bin_width)
    bin_labels = (bins[:-1] + bins[1:]) / 2

    ws_bin = pd.cut(out[wind_speed_col], bins=bins, labels=bin_labels)
    curve = out.groupby(ws_bin, observed=True)[power_col].mean()

    out["Theoretical_Power"] = ws_bin.map(curve).astype(float)
    # Records that fall outside the binned range fall back to 0.
    out["Theoretical_Power"] = out["Theoretical_Power"].fillna(0.0)
    return out


# Final 8-feature input vector x (Eq. 3): v_ws, dir_sin, dir_cos, sigma_v,
# omega_r, beta, T_n, T_g
DEFAULT_SELECTED_FEATURES: tuple[str, ...] = (
    "WindSpeed",
    "dir_sin",
    "dir_cos",
    "StdDevWindSpeed",
    "RotorRPM",
    "Pitch",
    "NacelTemp",
    "GearOilTemp",
)


def select_model_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_SELECTED_FEATURES,
    target_col: str = "Power",
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice the engineered DataFrame down to the final model inputs.

    Returns
    -------
    (X, y):
        X is a DataFrame with exactly `feature_cols` columns (in order),
        y is the target Series (active power, kW).
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required feature column(s) {missing}. Did you run "
            f"add_wind_direction_encoding() first?"
        )
    X = df[list(feature_cols)].copy()
    y = df[target_col].copy()
    return X, y
