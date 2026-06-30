"""
Raw SCADA data loading (Sec. III-A).

The dataset consists of 10-minute Vestas V52 SCADA records from Dundalk
Institute of Technology (DkIT), Ireland, spanning January 2006 to March
2020 (653,103 raw records, 22 sensor channels). The raw CSV is *not*
redistributed with this repository — see `data/README.md` for the public
source (Byrne & MacArtain, Mendeley Data, doi: 10.17632/tm988rs48k.2).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_scada(
    csv_path: str | Path,
    timestamp_col: str = "Timestamps",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Load the raw SCADA CSV and set a sorted DatetimeIndex.

    Parameters
    ----------
    csv_path:
        Path to the raw SCADA CSV file.
    timestamp_col:
        Name of the timestamp column in the raw file.
    dayfirst:
        Whether the timestamp strings are day-first formatted (as in the
        original DkIT export).

    Returns
    -------
    pd.DataFrame
        Raw SCADA records indexed by timestamp, sorted chronologically.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw SCADA CSV not found at '{csv_path}'.\n"
            f"Download the Vestas V52 DkIT dataset (Byrne & MacArtain, 2022, "
            f"doi: 10.17632/tm988rs48k.2) and place it at this path, or "
            f"update `data.raw_csv_path` in configs/config.yaml. See "
            f"data/README.md for details."
        )

    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], dayfirst=dayfirst, format="mixed")
    df = df.set_index(timestamp_col).sort_index()
    return df
