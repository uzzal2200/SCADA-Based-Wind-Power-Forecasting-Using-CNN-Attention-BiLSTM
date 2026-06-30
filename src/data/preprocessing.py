"""
Data filtering and cleaning (Sec. III-C.1).

Raw SCADA records are filtered to remove operational anomalies, sensor
faults and physically implausible data:

    * wind speed retained only within [0, 25] m/s
    * active power retained only where P >= 5 kW (removes start-up
      transients, curtailment periods and idling states that would
      otherwise destabilise MAPE)
    * missing values are removed via row-wise deletion (no imputation,
      to preserve temporal continuity and avoid synthetic patterns)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("wtpf.data.preprocessing")


def filter_operational_range(
    df: pd.DataFrame,
    wind_speed_col: str = "WindSpeed",
    power_col: str = "Power",
    wind_speed_min: float = 0.0,
    wind_speed_max: float = 25.0,
    power_min_kw: float = 5.0,
) -> pd.DataFrame:
    """Retain only physically plausible operational records.

    v_ws in [wind_speed_min, wind_speed_max] and P >= power_min_kw.
    """
    before = len(df)
    mask = (
        (df[wind_speed_col] >= wind_speed_min)
        & (df[wind_speed_col] <= wind_speed_max)
        & (df[power_col] >= power_min_kw)
    )
    out = df.loc[mask].copy()
    logger.info(
        "Operational-range filter: %d -> %d rows (%.1f%% retained)",
        before,
        len(out),
        100.0 * len(out) / max(before, 1),
    )
    return out


def drop_missing_rows(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Row-wise deletion of any record containing a missing value.

    No imputation is applied (Sec. III-C.1) so as not to introduce
    artificial temporal patterns into a strictly chronological dataset.
    """
    before = len(df)
    out = df.dropna(subset=subset).copy()
    dropped = before - len(out)
    if dropped:
        logger.info("Dropped %d rows containing missing values.", dropped)
    return out
