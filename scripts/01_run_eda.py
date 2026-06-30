#!/usr/bin/env python3
"""
Reproduce the dataset-analysis figures (Sec. III-B, Fig. 2-10) from the
cleaned, operational SCADA records and save them to `figures/eda/`.

Usage:
    python scripts/01_run_eda.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from wtpf.config import load_config  # noqa: E402
from wtpf.pipeline import load_and_clean  # noqa: E402
from wtpf.utils.logger import get_logger  # noqa: E402
from wtpf.visualization import eda_plots, savefig  # noqa: E402

logger = get_logger("wtpf.scripts.eda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="figures/eda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and cleaning raw SCADA records...")
    df = load_and_clean(cfg)
    logger.info("Operational dataset: %d rows spanning %s to %s", len(df), df.index.min(), df.index.max())

    figure_jobs = [
        ("fig02_polar_scatter", lambda: eda_plots.plot_polar_scatter(df)),
        ("fig03_3d_cylindrical", lambda: eda_plots.plot_3d_cylindrical(df)),
        ("fig04_power_curve", lambda: eda_plots.plot_power_curve(df)),
        ("fig06_seasonal_diurnal", lambda: eda_plots.plot_seasonal_diurnal_distributions(df)),
        ("fig07_power_vs_rpm", lambda: eda_plots.plot_power_vs_rpm(df)),
        ("fig08_marginal_histograms", lambda: eda_plots.plot_marginal_histograms(df)),
        ("fig09_daily_monthly_annual", lambda: eda_plots.plot_daily_monthly_annual_power(df)),
        ("fig10_wind_rose", lambda: eda_plots.plot_wind_rose(df)),
    ]

    for name, job in figure_jobs:
        logger.info("Rendering %s ...", name)
        fig = job()
        savefig(fig, str(out_dir / f"{name}.png"))

    logger.info("Saved %d EDA figures to %s", len(figure_jobs), out_dir)


if __name__ == "__main__":
    main()
