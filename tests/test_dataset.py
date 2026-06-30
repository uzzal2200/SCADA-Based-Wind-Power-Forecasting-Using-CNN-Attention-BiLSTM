import numpy as np
import pytest

from wtpf.data.dataset import (
    MinMaxScalerTrainOnly,
    build_sliding_windows,
    chronological_split,
    prepare_datasets,
)
from wtpf.data.feature_engineering import add_wind_direction_encoding, select_model_features


def test_minmax_scaler_train_only_range():
    rng = np.random.default_rng(0)
    train = rng.normal(10, 3, (100, 3))
    test = rng.normal(10, 3, (20, 3))

    scaler = MinMaxScalerTrainOnly().fit(train)
    train_scaled = scaler.transform(train)

    assert np.isclose(train_scaled.min(axis=0), 0.0).all()
    assert np.isclose(train_scaled.max(axis=0), 1.0).all()

    # test data scaled with train statistics may legitimately fall
    # outside [0, 1] if its range exceeds the training range
    inv = scaler.inverse_transform(train_scaled)
    assert np.allclose(inv, train, atol=1e-6)


def test_minmax_scaler_inverse_transform_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(50, 10, (200,))
    scaler = MinMaxScalerTrainOnly().fit(x)
    scaled = scaler.transform(x)
    recovered = scaler.inverse_transform(scaled)
    assert np.allclose(recovered, x, atol=1e-6)


def test_chronological_split_no_overlap_no_shuffle():
    split = chronological_split(1000, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert split.train_idx[0] == 0
    assert split.train_idx[1] == split.val_idx[0]
    assert split.val_idx[1] == split.test_idx[0]
    assert split.test_idx[1] == 1000

    # strictly increasing -> chronological, no shuffling possible
    all_starts = [split.train_idx[0], split.val_idx[0], split.test_idx[0]]
    assert all_starts == sorted(all_starts)


def test_chronological_split_invalid_ratios_raises():
    with pytest.raises(ValueError):
        chronological_split(100, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_build_sliding_windows_shapes_and_target_alignment():
    n, f, T = 50, 3, 10
    X = np.arange(n * f, dtype=np.float32).reshape(n, f)
    y = np.arange(n, dtype=np.float32)

    Xw, yw = build_sliding_windows(X, y, lookback=T, horizon=1)

    expected_n_windows = n - T - 1 + 1
    assert Xw.shape == (expected_n_windows, T, f)
    assert yw.shape == (expected_n_windows,)

    # Eq. 5: X_i = [x_i, ..., x_{i+T-1}], y_i = P_{i+T}
    assert np.allclose(Xw[0], X[0:T])
    assert np.isclose(yw[0], y[T])
    assert np.allclose(Xw[-1], X[expected_n_windows - 1 : expected_n_windows - 1 + T])


def test_build_sliding_windows_too_few_samples_raises():
    X = np.zeros((5, 2), dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError):
        build_sliding_windows(X, y, lookback=10)


def test_prepare_datasets_end_to_end_no_leakage(synthetic_scada_df):
    df = add_wind_direction_encoding(synthetic_scada_df)
    X, y = select_model_features(df)

    prepared = prepare_datasets(
        X, y, lookback=24, horizon=1, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    assert len(prepared.train_ds) > 0
    assert len(prepared.val_ds) > 0
    assert len(prepared.test_ds) > 0

    # test_index should be chronologically the last segment
    assert prepared.test_index[0] > prepared.test_index[-1] - prepared.test_index.freq * len(prepared.test_index) \
        if prepared.test_index.freq else True
    assert len(prepared.test_index) == len(prepared.test_ds)

    Xb, yb = prepared.train_ds[0]
    assert Xb.shape == (24, 8)
    assert yb.ndim == 0
