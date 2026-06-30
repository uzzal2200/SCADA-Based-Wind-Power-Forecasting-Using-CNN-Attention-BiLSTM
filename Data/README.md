# Dataset

This codebase trains and evaluates on the **Vestas V52 10-minute raw SCADA
dataset** collected at Dundalk Institute of Technology (DkIT), Ireland.

The raw CSV is **not redistributed** in this repository (large file, public
dataset with its own citation requirements). Download it directly from the
original source and place it at the path configured in `configs/config.yaml`
(`data.raw_csv_path`, default `data/raw/VestasV52_10_min_raw_SCADA_DkIT.csv`).

## Source

> R. Byrne and P. MacArtain, "Vestas V52 wind turbine, 10-minute SCADA data,
> 2006–2020 — Dundalk Institute of Technology, Ireland," *Mendeley Data*,
> vol. 2, 2022. doi: [10.17632/tm988rs48k.2](https://doi.org/10.17632/tm988rs48k.2)

## Directory layout

```
data/
├── README.md              <- this file
├── raw/                   <- place the downloaded CSV here (gitignored)
│   └── VestasV52_10_min_raw_SCADA_DkIT.csv
└── processed/              <- optional cache for cleaned/feature-engineered
                              intermediates (gitignored)
```

## Expected schema

| Column            | Description                          | Unit  |
|-------------------|---------------------------------------|-------|
| `Timestamps`      | 10-minute record timestamp            | —     |
| `WindSpeed`       | Hub-height wind speed                 | m/s   |
| `WindDirAbs`      | Absolute wind direction               | °     |
| `WindDirRel`      | Nacelle-relative wind direction (yaw) | °     |
| `StdDevWindSpeed` | Wind speed standard deviation         | m/s   |
| `Power`           | Active power output                   | kW    |
| `MaxPower` / `MinPower` / `StdDevPower` | Interval power statistics | kW |
| `RotorRPM`        | Rotor rotational speed                | RPM   |
| `GenRPM`          | Generator rotational speed            | RPM   |
| `Pitch`           | Blade pitch angle                     | °     |
| `EnvirTemp`       | Ambient temperature                   | °C    |
| `NacelTemp`       | Nacelle temperature                   | °C    |
| `GearOilTemp` / `GearBearTemp` / `GenTemp` / `GenBearTemp` | Drivetrain temperatures | °C |
| `AvgRPow`         | Average reactive power                | kVAr  |

Only 8 of these columns (plus the two engineered `dir_sin`/`dir_cos`
columns) are used as model inputs — see `configs/config.yaml ->
features.selected_features` and Eq. (3) of the paper.

## Quick sanity check after downloading

```bash
python -c "
from wtpf.config import load_config
from wtpf.pipeline import load_and_clean
cfg = load_config('configs/config.yaml')
df = load_and_clean(cfg)
print(df.shape, df.index.min(), df.index.max())
"
```

You should see roughly 600k+ operational records spanning January 2006 to
March 2020 after filtering.
