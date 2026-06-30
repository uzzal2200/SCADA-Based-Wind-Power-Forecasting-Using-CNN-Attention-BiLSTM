"""
Exploratory data analysis plotting utilities (Sec. III-B, Fig. 2-10).

These functions reproduce the key SCADA data-analysis visualisations
described in the paper from a cleaned/operational DataFrame. They mirror
(in modular, reusable form) the plots originally prototyped in
`notebooks/wind_turbine_eda.ipynb`.

All functions take a DataFrame indexed by a sorted DatetimeIndex with at
minimum the columns: WindSpeed, Power, WindDirAbs (plus RotorRPM, GenRPM,
Pitch, NacelTemp, EnvirTemp where relevant) and return the created
matplotlib Figure so callers can further customise / save it.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wtpf.visualization.style import set_style

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


def plot_polar_scatter(
    df: pd.DataFrame,
    wind_speed_col: str = "WindSpeed",
    power_col: str = "Power",
    wind_dir_col: str = "WindDirAbs",
    sample_n: int = 5000,
    random_state: int = 42,
) -> plt.Figure:
    """2D polar scatter of wind speed & active power vs. wind direction (Fig. 2)."""
    set_style()
    sample = df.sample(min(sample_n, len(df)), random_state=random_state)
    theta = np.deg2rad(sample[wind_dir_col])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": "polar"})
    sc1 = axes[0].scatter(theta, sample[wind_speed_col], c=sample[wind_speed_col], cmap="YlOrRd", s=8, alpha=0.6)
    axes[0].set_title("Wind Speed in Polar Coordinates")
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label="Wind Speed (m/s)")

    sc2 = axes[1].scatter(theta, sample[power_col], c=sample[power_col], cmap="viridis", s=8, alpha=0.6)
    axes[1].set_title("Active Power in Polar Coordinates")
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label="Power (kW)")

    fig.tight_layout()
    return fig


def plot_3d_cylindrical(
    df: pd.DataFrame,
    wind_speed_col: str = "WindSpeed",
    power_col: str = "Power",
    wind_dir_col: str = "WindDirAbs",
    sample_n: int = 8000,
    random_state: int = 42,
) -> plt.Figure:
    """3D cylindrical scatter: direction (azimuth) x speed (radius) x power (z), coloured by month (Fig. 3)."""
    set_style()
    sample = df.sample(min(sample_n, len(df)), random_state=random_state).copy()
    theta = np.deg2rad(sample[wind_dir_col].to_numpy())
    r = sample[wind_speed_col].to_numpy()
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Z = sample[power_col].to_numpy()
    months = sample.index.month

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X, Y, Z, c=months, cmap="twilight", s=6, alpha=0.6)
    ax.set_xlabel("Wind Direction (East comp.)")
    ax.set_ylabel("Wind Direction (North comp.)")
    ax.set_zlabel("Active Power (kW)")
    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("Month")
    fig.tight_layout()
    return fig


def plot_power_curve(
    df: pd.DataFrame,
    wind_speed_col: str = "WindSpeed",
    power_col: str = "Power",
    wind_dir_col: str = "WindDirAbs",
    sample_n: int = 15000,
    random_state: int = 42,
) -> plt.Figure:
    """Operational power curve scatter, coloured by wind direction (Fig. 4)."""
    set_style()
    sample = df.sample(min(sample_n, len(df)), random_state=random_state)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        sample[wind_speed_col], sample[power_col], c=sample[wind_dir_col],
        cmap="hsv", s=6, alpha=0.4,
    )
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Active Power (kW)")
    ax.set_title("Operational Power Curve")
    plt.colorbar(sc, ax=ax, label="Wind Direction (deg)")
    fig.tight_layout()
    return fig


def plot_marginal_histograms(
    df: pd.DataFrame,
    cols: Sequence[str] = (
        "WindSpeed", "Power", "WindDirAbs", "RotorRPM",
        "Pitch", "NacelTemp", "EnvirTemp", "GenRPM",
    ),
) -> plt.Figure:
    """Marginal frequency distributions with mean/median annotations (Fig. 8)."""
    set_style()
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cols):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        data = df[col].dropna()
        ax.hist(data, bins=50, color="#4C72B0", alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", lw=1, label=f"Mean={data.mean():.1f}")
        ax.axvline(data.median(), color="red", linestyle=":", lw=1, label=f"Median={data.median():.1f}")
        ax.set_title(col)
        ax.legend(fontsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_daily_monthly_annual_power(df: pd.DataFrame, power_col: str = "Power") -> plt.Figure:
    """Daily, monthly and annual average active power output (Fig. 9)."""
    set_style()
    daily = df[power_col].resample("D").mean()
    monthly = df[power_col].resample("ME").mean()
    annual = df[power_col].resample("YE").mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(daily.index, daily.values, color="#1f77b4", lw=0.6)
    axes[0].set_title("Daily Average Active Power (kW)")

    axes[1].plot(monthly.index, monthly.values, color="#ff7f0e", marker="o", ms=2, lw=1)
    axes[1].fill_between(monthly.index, monthly.values, alpha=0.2, color="#ff7f0e")
    axes[1].set_title("Monthly Average Active Power (kW)")

    years = annual.index.year
    axes[2].bar(years, annual.values, color="#2ca02c")
    for x, v in zip(years, annual.values):
        axes[2].text(x, v + 5, f"{v:.0f}", ha="center", fontsize=8)
    axes[2].set_title("Annual Average Active Power (kW)")

    fig.tight_layout()
    return fig


def plot_wind_rose(
    df: pd.DataFrame,
    wind_dir_col: str = "WindDirAbs",
    wind_speed_col: str = "WindSpeed",
    n_dir_bins: int = 16,
    speed_bins: Sequence[float] = (0, 3, 6, 9, 12, 15, 30),
    speed_labels: Sequence[str] = ("0-3", "3-6", "6-9", "9-12", "12-15", ">15"),
) -> plt.Figure:
    """Wind rose: joint distribution of direction frequency & speed magnitude (Fig. 10)."""
    set_style()
    dir_bins = np.linspace(0, 360, n_dir_bins + 1)
    dir_centers = (dir_bins[:-1] + dir_bins[1:]) / 2

    dir_bin_idx = pd.cut(df[wind_dir_col] % 360, bins=dir_bins, labels=False, include_lowest=True)
    speed_bin_idx = pd.cut(df[wind_speed_col], bins=speed_bins, labels=speed_labels)

    table = pd.crosstab(dir_bin_idx, speed_bin_idx, normalize=True) * 100

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    theta = np.deg2rad(dir_centers)
    width = 2 * np.pi / n_dir_bins * 0.9
    bottom = np.zeros(n_dir_bins)
    colors = plt.cm.YlOrRd(np.linspace(0.25, 1.0, len(speed_labels)))

    for label, color in zip(speed_labels, colors):
        values = table[label].reindex(range(n_dir_bins), fill_value=0).to_numpy() if label in table else np.zeros(n_dir_bins)
        ax.bar(theta, values, width=width, bottom=bottom, color=color, label=f"{label} m/s")
        bottom += values

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Wind Rose — Direction vs. Speed Distribution")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def plot_seasonal_diurnal_distributions(
    df: pd.DataFrame, power_col: str = "Power", wind_speed_col: str = "WindSpeed"
) -> plt.Figure:
    """Seasonal power boxplot, diurnal wind-speed profile, power-by-wind-speed-bin (Fig. 6)."""
    import seaborn as sns

    set_style()
    work = df.copy()
    work["Season"] = work.index.month.map(SEASON_MAP)
    work["Hour"] = work.index.hour
    ws_bins = np.arange(0, 31, 2)
    work["ws_bin"] = pd.cut(work[wind_speed_col], bins=ws_bins)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    season_order = [s for s in ["Spring", "Summer", "Autumn", "Winter"] if s in work["Season"].unique()]

    sns.boxplot(data=work, x="Season", y=power_col, order=season_order, ax=axes[0], showfliers=False)
    axes[0].set_title("Active Power by Season")

    sns.boxplot(data=work, x="Hour", y=wind_speed_col, ax=axes[1], showfliers=False)
    axes[1].set_title("Wind Speed by Hour of Day")

    bin_order = [c for c in work["ws_bin"].cat.categories if (work["ws_bin"] == c).any()]
    sns.boxplot(data=work, x="ws_bin", y=power_col, order=bin_order, ax=axes[2], showfliers=False)
    axes[2].set_title("Active Power by Wind Speed Bin")
    axes[2].tick_params(axis="x", rotation=90)

    fig.tight_layout()
    return fig


def plot_power_vs_rpm(
    df: pd.DataFrame,
    power_col: str = "Power",
    rotor_rpm_col: str = "RotorRPM",
    gen_rpm_col: str = "GenRPM",
    wind_speed_col: str = "WindSpeed",
    sample_n: int = 12000,
    random_state: int = 42,
) -> plt.Figure:
    """Power vs. rotor RPM & generator RPM, coloured by wind speed (Fig. 7)."""
    set_style()
    sample = df.sample(min(sample_n, len(df)), random_state=random_state)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(sample[rotor_rpm_col], sample[power_col], c=sample[wind_speed_col], cmap="plasma", s=8, alpha=0.5)
    axes[0].set_xlabel("Rotor RPM"); axes[0].set_ylabel("Active Power (kW)")
    axes[0].set_title("Power vs. Rotor RPM")
    plt.colorbar(sc1, ax=axes[0], label="Wind Speed (m/s)")

    sc2 = axes[1].scatter(sample[gen_rpm_col], sample[power_col], c=sample[wind_speed_col], cmap="plasma", s=8, alpha=0.5)
    axes[1].set_xlabel("Generator RPM"); axes[1].set_ylabel("Active Power (kW)")
    axes[1].set_title("Power vs. Generator RPM")
    plt.colorbar(sc2, ax=axes[1], label="Wind Speed (m/s)")

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: Sequence[str], sample_n: int = 50000, random_state: int = 42) -> plt.Figure:
    """Pairwise correlation heatmap of SCADA channels."""
    import seaborn as sns

    set_style()
    sample = df[list(cols)].sample(min(sample_n, len(df)), random_state=random_state)
    corr = sample.corr()

    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 3, 0.5 * len(cols) + 2))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("SCADA Feature Correlation Heatmap")
    fig.tight_layout()
    return fig
