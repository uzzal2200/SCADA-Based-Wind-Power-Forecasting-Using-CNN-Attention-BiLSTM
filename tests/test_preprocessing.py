import numpy as np
import pandas as pd

from wtpf.data.preprocessing import drop_missing_rows, filter_operational_range


def test_filter_operational_range_bounds(synthetic_scada_df):
    df = synthetic_scada_df.copy()
    df.loc[df.index[0], "WindSpeed"] = -5.0   # below range
    df.loc[df.index[1], "WindSpeed"] = 100.0  # above range
    df.loc[df.index[2], "Power"] = 0.0        # below power threshold

    out = filter_operational_range(
        df, wind_speed_min=0.0, wind_speed_max=25.0, power_min_kw=5.0
    )

    assert (out["WindSpeed"] >= 0.0).all()
    assert (out["WindSpeed"] <= 25.0).all()
    assert (out["Power"] >= 5.0).all()
    assert len(out) <= len(df)


def test_drop_missing_rows_removes_nan(synthetic_scada_df):
    df = synthetic_scada_df.copy()
    df.loc[df.index[5], "WindSpeed"] = np.nan
    df.loc[df.index[10], "Power"] = np.nan

    out = drop_missing_rows(df)
    assert out.isnull().sum().sum() == 0
    assert len(out) == len(df) - 2


def test_drop_missing_rows_no_imputation(synthetic_scada_df):
    """Row-wise deletion only — no values should be filled in."""
    df = synthetic_scada_df.copy()
    original_mean = df["WindSpeed"].mean()
    df.loc[df.index[0], "WindSpeed"] = np.nan

    out = drop_missing_rows(df)
    # the remaining values should be untouched (not interpolated)
    assert np.isclose(out["WindSpeed"].mean(), df["WindSpeed"].dropna().mean())
    assert not np.isclose(out["WindSpeed"].mean(), original_mean)
