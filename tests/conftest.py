"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_scada_df() -> pd.DataFrame:
    """A small synthetic SCADA-like DataFrame for fast, deterministic tests."""
    rng = np.random.default_rng(42)
    n = 1000
    idx = pd.date_range("2012-01-01", periods=n, freq="10min")
    wind_speed = np.clip(rng.normal(7, 3, n), 0, 25)
    power = np.clip(wind_speed**3 * 0.5 + rng.normal(0, 20, n), -10, 850)

    df = pd.DataFrame(
        {
            "WindSpeed": wind_speed,
            "Power": power,
            "WindDirAbs": rng.uniform(0, 360, n),
            "StdDevWindSpeed": np.abs(rng.normal(1, 0.3, n)),
            "RotorRPM": np.abs(rng.normal(20, 5, n)),
            "Pitch": np.abs(rng.normal(2, 3, n)),
            "NacelTemp": rng.normal(20, 5, n),
            "GearOilTemp": rng.normal(40, 5, n),
            "GenRPM": np.abs(rng.normal(1200, 300, n)),
        },
        index=idx,
    )
    df.index.name = "Timestamps"
    return df
