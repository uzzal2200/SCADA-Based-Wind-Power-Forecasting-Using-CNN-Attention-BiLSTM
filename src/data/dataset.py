"""
Normalisation, chronological splitting and sliding-window sequence
construction (Sec. III-C.4, Eq. 4-5).

Pipeline order (matches Algorithm 1, Step 1, and avoids any temporal
leakage):

    1. Split the cleaned, feature-engineered records chronologically
       (no shuffling) into train (70%) / val (15%) / test (15%).
    2. Fit min-max scalers *exclusively* on the training partition (one
       scaler for the 8 input features, one separate scaler for the
       target), then transform all three partitions without refitting.
    3. Build fixed-length sliding windows (T = 144, i.e. 24h of 10-min
       history) independently *within* each partition so no window ever
       spans a partition boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------
# Min-max scaling (Eq. 4), fit on training partition only
# -----------------------------------------------------------------------
class MinMaxScalerTrainOnly:
    """Min-max scaler fit exclusively on a training partition.

    x_hat = (x - x_min) / (x_max - x_min)

    Mirrors sklearn's MinMaxScaler API (`fit`, `transform`,
    `inverse_transform`) but is intentionally minimal and dependency-free
    for 1D/2D NumPy arrays, with explicit guarding against fitting on
    anything but the training split.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScalerTrainOnly":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None:
            raise RuntimeError("Scaler must be fit() before transform().")
        X = np.asarray(X, dtype=np.float64)
        orig_ndim = X.ndim
        if orig_ndim == 1:
            X = X.reshape(-1, 1)
        denom = np.where(self.data_max_ - self.data_min_ == 0, 1.0, self.data_max_ - self.data_min_)
        lo, hi = self.feature_range
        scaled = lo + (X - self.data_min_) / denom * (hi - lo)
        return scaled.ravel() if orig_ndim == 1 else scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.data_min_ is None:
            raise RuntimeError("Scaler must be fit() before inverse_transform().")
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        orig_ndim = X_scaled.ndim
        if orig_ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        lo, hi = self.feature_range
        X = (X_scaled - lo) / (hi - lo) * (self.data_max_ - self.data_min_) + self.data_min_
        return X.ravel() if orig_ndim == 1 else X


# -----------------------------------------------------------------------
# Chronological split (no shuffling)
# -----------------------------------------------------------------------
@dataclass
class ChronoSplit:
    train_idx: Tuple[int, int]
    val_idx: Tuple[int, int]
    test_idx: Tuple[int, int]


def chronological_split(
    n_samples: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> ChronoSplit:
    """Compute chronological (row-order) split boundaries.

    Returns half-open index ranges [start, end) for train/val/test that
    partition `n_samples` records in temporal order, with no shuffling
    and no overlap.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    n_train = int(round(n_samples * train_ratio))
    n_val = int(round(n_samples * val_ratio))
    n_test = n_samples - n_train - n_val

    train_range = (0, n_train)
    val_range = (n_train, n_train + n_val)
    test_range = (n_train + n_val, n_train + n_val + n_test)
    return ChronoSplit(train_range, val_range, test_range)


# -----------------------------------------------------------------------
# Sliding-window sequence construction (Eq. 5)
# -----------------------------------------------------------------------
def build_sliding_windows(
    X: np.ndarray, y: np.ndarray, lookback: int = 144, horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Build fixed-length input windows and single-step targets.

    X_i = [x_i, x_{i+1}, ..., x_{i+T-1}] in R^(T x F)
    y_i = P_{i+T}                          (Eq. 5, horizon=1)

    Parameters
    ----------
    X:
        Scaled feature matrix, shape (N, F).
    y:
        Scaled target vector, shape (N,).
    lookback:
        T, the lookback window length (144 = 24h of 10-min records).
    horizon:
        Steps ahead to predict (1 = single-step-ahead, as in the paper).

    Returns
    -------
    (X_windows, y_windows):
        X_windows has shape (N - T - horizon + 1, T, F);
        y_windows has shape (N - T - horizon + 1,).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = len(X)
    n_windows = n - lookback - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough samples ({n}) to build a single window of length "
            f"{lookback} + horizon {horizon}."
        )

    n_features = X.shape[1]
    X_windows = np.lib.stride_tricks.sliding_window_view(X, (lookback, n_features))[:, 0, :, :]
    X_windows = X_windows[: n_windows]
    y_windows = y[lookback + horizon - 1 : lookback + horizon - 1 + n_windows]
    return X_windows, y_windows


# -----------------------------------------------------------------------
# PyTorch Dataset / DataLoader
# -----------------------------------------------------------------------
class SCADASequenceDataset(Dataset):
    """Wraps windowed (X, y) NumPy arrays as a PyTorch Dataset."""

    def __init__(self, X_windows: np.ndarray, y_windows: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(X_windows)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y_windows)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 256,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders.

    Only the training loader is shuffled at the *mini-batch* level (this
    does not break the chronological partitioning used to build windows
    — see Sec. III-C.4 — it only randomises the order in which already
    non-overlapping windows are fed to the optimiser, standard practice
    for sequence-to-one regression).
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------
# End-to-end orchestration helper
# -----------------------------------------------------------------------
@dataclass
class PreparedData:
    train_ds: SCADASequenceDataset
    val_ds: SCADASequenceDataset
    test_ds: SCADASequenceDataset
    feature_scaler: MinMaxScalerTrainOnly
    target_scaler: MinMaxScalerTrainOnly
    test_index: pd.DatetimeIndex  # for seasonal / time-aligned analysis


def prepare_datasets(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int = 144,
    horizon: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> PreparedData:
    """Full pipeline: chronological split -> scale (train-fit only) ->
    sliding windows -> PyTorch datasets.

    `X` and `y` must already be cleaned and feature-engineered (i.e. the
    output of `select_model_features`), indexed by a sorted
    DatetimeIndex.
    """
    n = len(X)
    split = chronological_split(n, train_ratio, val_ratio, test_ratio)

    X_arr = X.to_numpy()
    y_arr = y.to_numpy()

    # --- fit scalers on the training partition only ---
    feature_scaler = MinMaxScalerTrainOnly().fit(X_arr[split.train_idx[0] : split.train_idx[1]])
    target_scaler = MinMaxScalerTrainOnly().fit(y_arr[split.train_idx[0] : split.train_idx[1]])

    X_scaled = feature_scaler.transform(X_arr)
    y_scaled = target_scaler.transform(y_arr)

    def _slice(rng: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        s, e = rng
        return X_scaled[s:e], y_scaled[s:e]

    X_train, y_train = _slice(split.train_idx)
    X_val, y_val = _slice(split.val_idx)
    X_test, y_test = _slice(split.test_idx)

    Xw_train, yw_train = build_sliding_windows(X_train, y_train, lookback, horizon)
    Xw_val, yw_val = build_sliding_windows(X_val, y_val, lookback, horizon)
    Xw_test, yw_test = build_sliding_windows(X_test, y_test, lookback, horizon)

    test_start, test_end = split.test_idx
    test_index = X.index[test_start + lookback + horizon - 1 : test_end]

    return PreparedData(
        train_ds=SCADASequenceDataset(Xw_train, yw_train),
        val_ds=SCADASequenceDataset(Xw_val, yw_val),
        test_ds=SCADASequenceDataset(Xw_test, yw_test),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        test_index=test_index,
    )
