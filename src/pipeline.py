"""
End-to-end data pipeline orchestrator.

Chains together every step described in Sec. III-A through III-C.4 so
scripts have a single call to go from the raw CSV path to ready-to-train
PyTorch datasets:

    raw CSV -> load -> filter -> wind-direction encode -> select 8
    features -> chronological split -> scale (train-fit only) ->
    sliding windows -> SCADASequenceDataset (train/val/test)
"""

from __future__ import annotations

import logging

import pandas as pd

from wtpf.config import Config
from wtpf.data.dataset import PreparedData, prepare_datasets
from wtpf.data.feature_engineering import add_wind_direction_encoding, select_model_features
from wtpf.data.loader import load_raw_scada
from wtpf.data.preprocessing import drop_missing_rows, filter_operational_range

logger = logging.getLogger("wtpf.pipeline")


def load_and_clean(cfg: Config) -> pd.DataFrame:
    """Steps 1-2 of Algorithm 1: load raw SCADA, filter, encode wind
    direction. Returns the cleaned, feature-engineered DataFrame (still
    containing all original columns plus dir_sin/dir_cos)."""
    df = load_raw_scada(cfg.data.raw_csv_path, timestamp_col=cfg.data.timestamp_col)
    logger.info("Loaded raw SCADA: %d rows, %d columns", *df.shape)

    df = filter_operational_range(
        df,
        wind_speed_col="WindSpeed",
        power_col=cfg.data.target_col,
        wind_speed_min=cfg.data.wind_speed_min,
        wind_speed_max=cfg.data.wind_speed_max,
        power_min_kw=cfg.data.power_min_kw,
    )
    df = drop_missing_rows(df)
    df = add_wind_direction_encoding(df, wind_dir_col=cfg.features.wind_dir_col)
    return df


def build_datasets(cfg: Config, df: pd.DataFrame | None = None) -> PreparedData:
    """Steps 3-4 of Algorithm 1: select the 8 final features, chronological
    split, scale, build sliding windows -> PreparedData (train/val/test
    torch Datasets + fitted scalers + test DatetimeIndex)."""
    if df is None:
        df = load_and_clean(cfg)

    X, y = select_model_features(
        df,
        feature_cols=cfg.features.selected_features,
        target_col=cfg.data.target_col,
    )

    prepared = prepare_datasets(
        X,
        y,
        lookback=cfg.sequence.lookback_window,
        horizon=cfg.sequence.horizon,
        train_ratio=cfg.sequence.train_ratio,
        val_ratio=cfg.sequence.val_ratio,
        test_ratio=cfg.sequence.test_ratio,
    )
    logger.info(
        "Sequence windows -> train=%d, val=%d, test=%d (T=%d, F=%d)",
        len(prepared.train_ds), len(prepared.val_ds), len(prepared.test_ds),
        cfg.sequence.lookback_window, len(cfg.features.selected_features),
    )
    return prepared
