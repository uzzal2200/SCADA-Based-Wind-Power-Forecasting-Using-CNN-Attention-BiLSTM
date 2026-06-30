import numpy as np
import pandas as pd
import pytest

from wtpf.data.feature_engineering import (
    DEFAULT_SELECTED_FEATURES,
    add_cyclic_time_encoding,
    add_theoretical_power_curve,
    add_wind_direction_encoding,
    select_model_features,
)


def test_wind_direction_encoding_bijective_at_boundary():
    """sin/cos encoding must resolve the 0deg/360deg discontinuity:
    values near 0 and near 360 should map to nearly identical (sin, cos)."""
    df = pd.DataFrame({"WindDirAbs": [0.0, 359.9999, 180.0, 90.0]})
    out = add_wind_direction_encoding(df)

    assert np.isclose(out.loc[0, "dir_sin"], out.loc[1, "dir_sin"], atol=1e-3)
    assert np.isclose(out.loc[0, "dir_cos"], out.loc[1, "dir_cos"], atol=1e-3)

    # unit-circle property: sin^2 + cos^2 == 1
    radius = out["dir_sin"] ** 2 + out["dir_cos"] ** 2
    assert np.allclose(radius, 1.0, atol=1e-9)

    # known values
    assert np.isclose(out.loc[2, "dir_sin"], 0.0, atol=1e-9)   # 180deg -> sin=0
    assert np.isclose(out.loc[2, "dir_cos"], -1.0, atol=1e-9)  # 180deg -> cos=-1
    assert np.isclose(out.loc[3, "dir_sin"], 1.0, atol=1e-9)   # 90deg  -> sin=1


def test_cyclic_time_encoding_range(synthetic_scada_df):
    out = add_cyclic_time_encoding(synthetic_scada_df)
    assert out["hour_sin"].between(-1.0, 1.0).all()
    assert out["hour_cos"].between(-1.0, 1.0).all()


def test_theoretical_power_curve_monotonic_trend(synthetic_scada_df):
    out = add_theoretical_power_curve(synthetic_scada_df)
    assert "Theoretical_Power" in out.columns
    assert out["Theoretical_Power"].notna().all()
    # higher wind speed bins should generally have higher theoretical power
    low_ws_power = out.loc[out["WindSpeed"] < 5, "Theoretical_Power"].mean()
    high_ws_power = out.loc[out["WindSpeed"] > 12, "Theoretical_Power"].mean()
    assert high_ws_power > low_ws_power


def test_select_model_features_returns_exactly_eight(synthetic_scada_df):
    df = add_wind_direction_encoding(synthetic_scada_df)
    X, y = select_model_features(df)

    assert list(X.columns) == list(DEFAULT_SELECTED_FEATURES)
    assert len(X.columns) == 8
    assert len(X) == len(y) == len(df)
    assert y.name == "Power"


def test_select_model_features_missing_column_raises():
    df = pd.DataFrame({"WindSpeed": [1, 2, 3], "Power": [10, 20, 30]})
    with pytest.raises(KeyError):
        select_model_features(df)
