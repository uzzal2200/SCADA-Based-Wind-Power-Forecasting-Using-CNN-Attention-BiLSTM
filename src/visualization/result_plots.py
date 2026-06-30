"""
Results / comparative-performance plotting utilities (Sec. IV, Fig. 12-22).

Each function takes plain Python/NumPy/pandas structures (metric
dictionaries, prediction arrays) produced by `wtpf.evaluation`, decoupling
visualisation from any specific training run.
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wtpf.visualization.style import MODEL_ORDER, PALETTE, set_style


def plot_metric_bar_comparison(results: pd.DataFrame) -> plt.Figure:
    """Bar charts of RMSE/MAE/MAPE/sMAPE across all models, with a trend
    line highlighting monotonic error reduction (Fig. 12).

    `results` must be a DataFrame indexed by model name with columns
    RMSE, MAE, MAPE, sMAPE (kW / kW / % / %).
    """
    set_style()
    metrics = ["RMSE", "MAE", "MAPE", "sMAPE"]
    models = [m for m in MODEL_ORDER if m in results.index]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, metric in zip(axes, metrics):
        values = results.loc[models, metric]
        colors = [PALETTE.get(m, "#888888") for m in models]
        ax.bar(models, values, color=colors)
        ax.plot(models, values, color="black", marker="o", ms=3, lw=1, linestyle="--", alpha=0.6)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_timeseries_overlay(
    timestamps: pd.DatetimeIndex,
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_steps: int = 300,
    start: int = 0,
) -> plt.Figure:
    """Multi-model predicted vs. actual power overlay (Fig. 13)."""
    set_style()
    end = start + n_steps
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(range(n_steps), actual[start:end], color="black", lw=1.5, label="Actual")
    for name, pred in predictions.items():
        ax.plot(range(n_steps), pred[start:end], lw=1, linestyle="--",
                 color=PALETTE.get(name, None), label=name, alpha=0.85)
    ax.set_xlabel("Time Steps (10-min)")
    ax.set_ylabel("Active Power (kW)")
    ax.set_title("Multi-Model Time-Series Overlay")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_single_day_forecast(
    timestamps: pd.DatetimeIndex, actual: np.ndarray, predicted: np.ndarray
) -> plt.Figure:
    """24-hour single-day forecast with shaded absolute error (Fig. 14)."""
    set_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps, actual, color="black", lw=1.3, label="Actual Power")
    ax.plot(timestamps, predicted, color="red", lw=1.3, linestyle="--", label="Predicted Power")
    ax.fill_between(timestamps, actual, predicted, color="pink", alpha=0.5, label="Absolute Error")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Active Power (kW)")
    ax.set_title("24-Hour Forecast Case Study")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_scatter_grid_pred_vs_actual(predictions: Dict[str, np.ndarray], actual: np.ndarray) -> plt.Figure:
    """Density-weighted predicted-vs-actual scatter grid for all models (Fig. 15)."""
    from scipy.stats import gaussian_kde

    set_style()
    models = [m for m in predictions if predictions[m] is not None]
    n = len(models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    lims = (min(0, actual.min()), actual.max() * 1.05)
    for ax, name in zip(axes, models):
        pred = predictions[name]
        # subsample for KDE tractability
        idx = np.random.default_rng(0).choice(len(actual), size=min(20000, len(actual)), replace=False)
        a, p = actual[idx], pred[idx]
        try:
            xy = np.vstack([a, p])
            density = gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones_like(a)
        order = density.argsort()
        sc = ax.scatter(a[order], p[order], c=density[order], cmap="viridis", s=4, alpha=0.6)
        ax.plot(lims, lims, color="black", lw=1, label="y = x")
        slope, intercept = np.polyfit(a, p, 1)
        ax.plot(lims, [slope * x + intercept for x in lims], color="red", linestyle="--", lw=1,
                 label=f"Fit: {slope:.3f}x+{intercept:.1f}")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Power (kW)"); ax.set_ylabel("Predicted Power (kW)")
        ax.set_title(name)
        ax.legend(fontsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_residual_four_panel(
    residuals: np.ndarray, actual: np.ndarray, wind_speed: np.ndarray
) -> plt.Figure:
    """Four-panel residual diagnostic: signed residual vs. power, residual
    histogram + Gaussian fit, residual by wind-speed bin, temporal residual
    sequence (Fig. 16)."""
    set_style()
    mu, sigma = residuals.mean(), residuals.std()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(actual, residuals, s=3, alpha=0.3, color="indianred")
    axes[0, 0].axhline(0, color="black", lw=1)
    axes[0, 0].set_xlabel("Actual Power (kW)"); axes[0, 0].set_ylabel("Residual (kW)")
    axes[0, 0].set_title(f"(a) Signed Residual vs Actual Power  (mean={mu:.2f} kW)")

    axes[0, 1].hist(residuals, bins=80, density=True, color="indianred", alpha=0.7)
    x = np.linspace(residuals.min(), residuals.max(), 200)
    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    axes[0, 1].plot(x, gauss, color="black", lw=1.5)
    axes[0, 1].set_title(f"(b) Residual Histogram (mu={mu:.2f}, sigma={sigma:.2f})")

    bins = np.arange(0, 30, 5)
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    box_data = []
    for i in range(len(bins) - 1):
        mask = (wind_speed >= bins[i]) & (wind_speed < bins[i + 1])
        box_data.append(residuals[mask] if mask.any() else np.array([0.0]))
    axes[1, 0].boxplot(box_data, labels=labels, showfliers=False)
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].set_xlabel("Wind Speed Bin (m/s)"); axes[1, 0].set_ylabel("Residual (kW)")
    axes[1, 0].set_title("(c) Residual by Wind Speed Bin")

    n_show = min(300, len(residuals))
    colors = np.where(residuals[:n_show] >= 0, "pink", "lightblue")
    axes[1, 1].bar(range(n_show), residuals[:n_show], color=colors, width=1.0)
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set_xlabel("Time Steps"); axes[1, 1].set_ylabel("Residual (kW)")
    axes[1, 1].set_title("(d) Temporal Residual Sequence")

    fig.tight_layout()
    return fig


def plot_boxplot_cdf_comparison(abs_errors: Dict[str, np.ndarray]) -> plt.Figure:
    """Cross-model absolute-error boxplot + empirical CDF (Fig. 17)."""
    set_style()
    models = [m for m in MODEL_ORDER if m in abs_errors]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    data = [abs_errors[m] for m in models]
    colors = [PALETTE.get(m, "#888888") for m in models]
    bp = axes[0].boxplot(data, labels=models, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, m in enumerate(models, start=1):
        axes[0].text(i, np.median(abs_errors[m]), f"{np.median(abs_errors[m]):.1f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Absolute Error (kW)")
    axes[0].set_title("Absolute Error Distribution")
    axes[0].tick_params(axis="x", rotation=30)

    for m in models:
        sorted_err = np.sort(abs_errors[m])
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        axes[1].plot(sorted_err, cdf, label=m, color=PALETTE.get(m, None), lw=1.5)
    for ref in (50, 100, 150):
        axes[1].axvline(ref, color="gray", linestyle=":", lw=0.8)
    axes[1].set_xlabel("Absolute Error (kW)"); axes[1].set_ylabel("Cumulative Proportion")
    axes[1].set_title("Empirical CDF of Absolute Error")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_seasonal_performance(seasonal_results: Dict[str, Dict[str, "MetricBundle"]]) -> plt.Figure:
    """Seasonal RMSE/MAE/R2 grouped bar charts across all models (Fig. 18).

    `seasonal_results` is {model_name: {season: MetricBundle}}.
    """
    set_style()
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    models = [m for m in MODEL_ORDER if m in seasonal_results]
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_attrs = ["rmse", "mae", "r2"]
    titles = ["RMSE (kW)", "MAE (kW)", "R2"]

    for ax, attr, title in zip(axes, metric_attrs, titles):
        x = np.arange(len(seasons))
        for i, m in enumerate(models):
            values = [
                getattr(seasonal_results[m][s], attr) if s in seasonal_results[m] else np.nan
                for s in seasons
            ]
            ax.bar(x + i * width, values, width=width, label=m, color=PALETTE.get(m, None))
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(seasons)
        ax.set_title(title)
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.3), ncol=3)
    fig.tight_layout()
    return fig


def plot_power_band_performance(band_results: Dict[str, Dict[str, "MetricBundle"]]) -> plt.Figure:
    """Power-band RMSE/MAE/MAPE/sMAPE grouped bar charts (Fig. 19).

    `band_results` is {model_name: {band: MetricBundle}}.
    """
    set_style()
    bands = ["Low", "Medium", "High", "Rated"]
    models = [m for m in MODEL_ORDER if m in band_results]
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_attrs = ["rmse", "mae", "mape", "smape"]
    titles = ["(a) RMSE (kW)", "(b) MAE (kW)", "(c) MAPE (%)", "(d) sMAPE (%)"]

    for ax, attr, title in zip(axes.flatten(), metric_attrs, titles):
        x = np.arange(len(bands))
        for i, m in enumerate(models):
            values = [
                getattr(band_results[m][b], attr) if b in band_results[m] else np.nan
                for b in bands
            ]
            ax.bar(x + i * width, values, width=width, label=m, color=PALETTE.get(m, None))
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(bands)
        ax.set_title(title)

    axes[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.35), ncol=3)
    fig.tight_layout()
    return fig


def plot_radar_chart(results: pd.DataFrame) -> plt.Figure:
    """Normalised hexagonal radar chart across all six metrics (Fig. 20).

    `results` indexed by model, columns RMSE, R2, PSNR, sMAPE, MAPE, MAE.
    Lower-is-better metrics are inverted before normalisation so that a
    larger enclosed polygon always denotes better performance.
    """
    set_style()
    metrics = ["RMSE", "R2", "PSNR", "sMAPE", "MAPE", "MAE"]
    lower_is_better = {"RMSE", "sMAPE", "MAPE", "MAE"}

    norm = results[metrics].copy()
    for col in metrics:
        vals = norm[col]
        if col in lower_is_better:
            vals = vals.max() - vals  # invert so larger = better
        rng = vals.max() - vals.min()
        norm[col] = (vals - vals.min()) / rng if rng > 0 else 0.5

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for model in [m for m in MODEL_ORDER if m in norm.index]:
        values = norm.loc[model, metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=model, color=PALETTE.get(model, None), lw=2)
        ax.fill(angles, values, color=PALETTE.get(model, None), alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_title("Normalised Multi-Metric Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def plot_taylor_diagram(model_stats: Dict[str, Dict[str, float]], obs_std: float) -> plt.Figure:
    """Simplified Taylor diagram (Fig. 21).

    `model_stats` is {model_name: {"std": float, "corr": float}} —
    standard deviation (kW) and Pearson correlation of each model's
    predictions relative to the observed series. `obs_std` is the
    standard deviation of the observed (actual) power.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    max_std = max([obs_std] + [s["std"] for s in model_stats.values()]) * 1.2
    corr_ticks = [0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.0]
    ax.set_xticks([np.arccos(c) for c in corr_ticks])
    ax.set_xticklabels([str(c) for c in corr_ticks])
    ax.set_rlim(0, max_std)

    # observation reference point
    ax.plot(0, obs_std, marker="*", color="black", ms=16, label="Observations (Ref)")

    for name, stats in model_stats.items():
        theta = np.arccos(np.clip(stats["corr"], -1, 1))
        ax.plot(theta, stats["std"], marker="o", ms=10, label=name, color=PALETTE.get(name, None))

    ax.set_title("Taylor Diagram")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def plot_component_ablation(ablation_df: pd.DataFrame) -> plt.Figure:
    """RMSE / R2 bars for each ablation variant vs. the full proposed
    model, with relative-change annotations (Fig. 22).

    `ablation_df` indexed by variant name, columns RMSE, R2.
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    full_rmse = ablation_df.loc["Full Proposed", "RMSE"]
    full_r2 = ablation_df.loc["Full Proposed", "R2"]

    colors = plt.cm.tab10(np.linspace(0, 1, len(ablation_df)))

    axes[0].bar(ablation_df.index, ablation_df["RMSE"], color=colors)
    axes[0].axhline(full_rmse, color="red", linestyle="--", lw=1)
    for i, (name, row) in enumerate(ablation_df.iterrows()):
        delta = 100 * (row["RMSE"] - full_rmse) / full_rmse
        axes[0].text(i, row["RMSE"], f"{delta:+.1f}%" if name != "Full Proposed" else "", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("(a) RMSE (kW) — lower is better")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(ablation_df.index, ablation_df["R2"], color=colors)
    axes[1].axhline(full_r2, color="red", linestyle="--", lw=1)
    for i, (name, row) in enumerate(ablation_df.iterrows()):
        delta = row["R2"] - full_r2
        axes[1].text(i, row["R2"], f"{delta:+.4f}" if name != "Full Proposed" else "", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("(b) R2 — higher is better")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return fig
