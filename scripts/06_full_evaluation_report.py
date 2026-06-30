#!/usr/bin/env python3
"""
Full post-hoc evaluation report (Sec. IV-A, IV-C, IV-D, IV-F, IV-G):
seasonal decomposition (Table 7), power-band disaggregation (Fig. 19),
residual diagnostics (Fig. 16-17) for the proposed model, and the
headline comparison figures (metric bar chart, radar chart, boxplot/CDF).

Reads the predictions saved by `scripts/03_train_all_baselines.py`.

Usage:
    python scripts/06_full_evaluation_report.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wtpf.config import load_config  # noqa: E402
from wtpf.evaluation.power_band_analysis import power_band_breakdown  # noqa: E402
from wtpf.evaluation.residual_analysis import (  # noqa: E402
    absolute_error_cdf, residual_gaussian_fit, signed_residuals,
)
from wtpf.evaluation.seasonal_analysis import seasonal_breakdown  # noqa: E402
from wtpf.utils.logger import get_logger  # noqa: E402
from wtpf.visualization import savefig  # noqa: E402
from wtpf.visualization.result_plots import (  # noqa: E402
    plot_boxplot_cdf_comparison, plot_metric_bar_comparison,
    plot_power_band_performance, plot_radar_chart, plot_residual_four_panel,
    plot_seasonal_performance,
)

logger = get_logger("wtpf.scripts.evaluate")

DISPLAY_NAMES = {
    "cnn": "CNN",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn_bilstm": "CNN-BiLSTM",
    "attention_bilstm": "Attention-BiLSTM",
    "cnn_attention_bilstm": "CNN-Attention-BiLSTM (Ours)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(args.results_dir)
    pred_dir = results_dir / "predictions"
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    timestamps_path = results_dir / "test_timestamps.npy"
    if not timestamps_path.exists():
        raise RuntimeError(f"{timestamps_path} not found — run scripts/03_train_all_baselines.py first.")
    test_index = pd.DatetimeIndex(np.load(timestamps_path, allow_pickle=True))

    predictions, abs_errors = {}, {}
    for model_name in cfg.baselines:
        npz_path = pred_dir / f"{model_name}.npz"
        if not npz_path.exists():
            logger.warning("Missing predictions for %s — skipping.", model_name)
            continue
        data = np.load(npz_path)
        display = DISPLAY_NAMES.get(model_name, model_name)
        predictions[display] = (data["y_true"], data["y_pred"])
        abs_errors[display] = np.abs(data["y_true"] - data["y_pred"])

    if not predictions:
        raise RuntimeError(f"No predictions found in {pred_dir} — run scripts/03 first.")

    # ---- Table 5-style metric comparison + bar chart + radar -----------
    perf_csv = results_dir / "table5_performance_comparison.csv"
    if perf_csv.exists():
        perf_df = pd.read_csv(perf_csv, index_col=0)
        fig = plot_metric_bar_comparison(perf_df)
        savefig(fig, str(figures_dir / "fig12_metric_bar_comparison.png"))
        fig = plot_radar_chart(perf_df)
        savefig(fig, str(figures_dir / "fig20_radar_chart.png"))
        logger.info("Saved metric bar chart and radar chart.")

    # ---- Boxplot + CDF of absolute error --------------------------------
    fig = plot_boxplot_cdf_comparison(abs_errors)
    savefig(fig, str(figures_dir / "fig17_boxplot_cdf_comparison.png"))

    # ---- Seasonal decomposition (Table 7) -------------------------------
    seasonal_results = {}
    for display, (y_true, y_pred) in predictions.items():
        seasonal_results[display] = seasonal_breakdown(
            test_index, y_true, y_pred, power_threshold_kw=cfg.evaluation.mape_power_threshold_kw
        )
    seasonal_rows = {
        (display, season): bundle.to_dict()
        for display, seasons in seasonal_results.items()
        for season, bundle in seasons.items()
    }
    seasonal_df = pd.DataFrame(seasonal_rows).T
    seasonal_df.index.names = ["Model", "Season"]
    seasonal_df.to_csv(results_dir / "table7_seasonal_performance.csv")
    fig = plot_seasonal_performance(seasonal_results)
    savefig(fig, str(figures_dir / "fig18_seasonal_performance.png"))
    logger.info("Saved seasonal performance table & figure.")

    # ---- Power-band disaggregation (Fig. 19) -----------------------------
    band_results = {}
    for display, (y_true, y_pred) in predictions.items():
        band_results[display] = power_band_breakdown(
            y_true, y_pred, bands=cfg.evaluation.power_bands_kw, power_threshold_kw=cfg.evaluation.mape_power_threshold_kw
        )
    band_rows = {
        (display, band): bundle.to_dict()
        for display, bands in band_results.items()
        for band, bundle in bands.items()
    }
    band_df = pd.DataFrame(band_rows).T
    band_df.index.names = ["Model", "Band"]
    band_df.to_csv(results_dir / "power_band_performance.csv")
    fig = plot_power_band_performance(band_results)
    savefig(fig, str(figures_dir / "fig19_power_band_performance.png"))
    logger.info("Saved power-band performance table & figure.")

    # ---- Residual diagnostics for the proposed model (Fig. 16) ----------
    proposed_display = DISPLAY_NAMES["cnn_attention_bilstm"]
    if proposed_display in predictions:
        y_true, y_pred = predictions[proposed_display]
        residuals = signed_residuals(y_true, y_pred)
        mu, sigma = residual_gaussian_fit(residuals)
        logger.info("Proposed model residuals: mu=%.3f kW, sigma=%.3f kW", mu, sigma)

        cdf = absolute_error_cdf(np.abs(residuals))
        logger.info("Absolute-error CDF (proposed model): %s", cdf)

        ws_path = results_dir / "test_wind_speed.npy"
        if ws_path.exists():
            wind_speed = np.load(ws_path)
        else:
            logger.warning(
                "%s not found — residual-by-wind-speed-bin panel (c) will be empty. "
                "Re-run scripts/03_train_all_baselines.py to generate it.",
                ws_path,
            )
            wind_speed = np.full_like(y_true, fill_value=np.nan)
        try:
            fig = plot_residual_four_panel(residuals, y_true, wind_speed)
            savefig(fig, str(figures_dir / "fig16_residual_analysis.png"))
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipped residual-by-wind-speed panel: %s", exc)

    logger.info("Full evaluation report complete. See %s and %s.", results_dir, figures_dir)


if __name__ == "__main__":
    main()
