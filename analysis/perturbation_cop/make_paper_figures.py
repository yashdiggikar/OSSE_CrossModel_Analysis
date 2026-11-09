#!/usr/bin/env python3
"""
Publication-quality figures for the Correlated OSSE Perturbation (COP)
experiments, built from each model's already-computed analysis CSVs/.npy
bias maps (no raw NetCDF re-read needed). Covers FCN + Aurora now; will
pick up Prithvi automatically once its analysis CSVs land (same
graceful-skip convention as compute_cross_model_comparison.py).

Produces, per model x experiment, into paper_figures/<Model>/:
    global_rmse_timeseries_<experiment>.png
    latband_rmse_timeseries_<experiment>.png
    spatial_bias_map_<experiment>.png
And, per model x experiment x variable, from the correlation-length CSVs:
    corr_length_summary_<experiment>.png

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python make_paper_figures.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

BASE = Path("/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS")
OUT = BASE / "paper_figures"

INJECT_LEAD_HOUR = 720  # day 30 -- matches the project-wide INJECT_STEP=120 convention

# Categorical palette (validated CVD-safe order, see dataviz skill reference palette)
BLUE   = "#2a78d6"
ORANGE = "#eb6834"
GRAY   = "#898781"

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#e1e0d9",
    "axes.edgecolor": "#c3c2b7",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# temp_vars/wind_vars: the directly-inserted family, now at all 3 representative
# levels (300/500/1000hPa) instead of the old single 500hPa panel, matching this
# project's 2026-07-25 expansion of the underlying CSVs. Deliberately NOT
# widened to the full cross-variable grid (T/U/V/Z/humidity are now computed
# for every experiment in the CSVs, per that same expansion) -- a per-variable
# time-series/map figure with 18 panels stops being a readable "paper figure";
# cross-variable coupling (e.g. does T-insertion improve Z/wind) is exactly
# what compute_cross_model_comparison.py's effectiveness plot is for (extended
# the same day to include geopotential_upper_air/humidity_upper_air categories)
# and what the results-analysis doc reports in tabular form -- no need to
# duplicate that here. corr_vars mirrors the same family-only choice using
# compute_correlation_length_corr.py's own bare (non-level-suffixed) label
# convention.
MODELS = {
    "FCN": {
        "dir": "FCN",
        "experiments": {"temp": "t_perturbed_corr", "uv": "uv_perturbed_corr"},
        "lats": np.linspace(90.0, -90.0, 721),
        "lons": np.arange(0.0, 360.0, 0.25),
        "temp_vars": ["T300", "T500", "T1000", "T2M"],
        "wind_vars": ["U300", "U500", "U1000", "V300", "V500", "V1000", "U10", "V10"],
        "corr_vars": {"temp": ["T", "T2M"], "uv": ["U", "U10", "V", "V10"]},
    },
    "Aurora": {
        "dir": "Aurora",
        "experiments": {"temp": "T_PERTURBED_CORR", "uv": "UV_PERTURBED_CORR"},
        "lats": np.arange(90.0, -90.0, -0.25),
        "lons": np.arange(0.0, 360.0, 0.25),
        "temp_vars": ["T300", "T500", "T1000", "T2M"],
        "wind_vars": ["U300", "U500", "U1000", "V300", "V500", "V1000", "U10", "V10"],
        "corr_vars": {"temp": ["T", "T2M"], "uv": ["U", "U10", "V", "V10"]},
    },
    "Prithvi": {
        "dir": "Prithvi",
        "experiments": {"temp": "perturbed_corr_temp_ins", "uv": "perturbed_corr_uv_ins"},
        "lats": np.linspace(-89.5, 90.0, 360),
        "lons": np.linspace(-180.0, 179.375, 576),
        "temp_vars": ["T300", "T500", "T1000", "T2M"],
        "wind_vars": ["U300", "U500", "U1000", "V300", "V500", "V1000", "U10M", "V10M"],
        "corr_vars": {"temp": ["T", "T2M"], "uv": ["U", "U10M", "V", "V10M"]},
    },
}


def dropna_plot(ax, x, y, **kw):
    """Plot after dropping NaNs -- required, not cosmetic: control files commonly have a
    coarser cadence than the experiment (e.g. 12-hourly control vs 6-hourly experiment),
    so every other point is NaN. A line broken into many isolated 2-point segments gets
    silently dropped by matplotlib's default path simplification and renders blank, even
    though the legend still shows an entry (found + fixed this same bug in
    compute_cross_model_comparison.py this session -- see SESSION_LOG)."""
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(y.astype(float))
    ax.plot(x[mask], y[mask], **kw)


# ---------------------------------------------------------------------------
# A) Global RMSE time series
# ---------------------------------------------------------------------------
def plot_global_rmse(model, cfg, exp_type):
    exp_name = cfg["experiments"][exp_type]
    csv_path = BASE / cfg["dir"] / exp_name / f"global_rmse_{exp_name}.csv"
    if not csv_path.exists():
        print(f"  [global_rmse] {model}/{exp_type}: SKIP -- {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    variables = cfg["temp_vars"] if exp_type == "temp" else cfg["wind_vars"]
    variables = [v for v in variables if f"control_rmse_{v}" in df.columns]

    fig, axes = plt.subplots(1, len(variables), figsize=(4.6 * len(variables), 3.8), squeeze=False)
    days = df["lead_hour"] / 24.0
    for ax, var in zip(axes[0], variables):
        dropna_plot(ax, days, df[f"control_rmse_{var}"], color=BLUE, lw=1.6, label="Control")
        dropna_plot(ax, days, df[f"experiment_rmse_{var}"], color=ORANGE, lw=1.6, label="Experiment")
        ax.axvline(INJECT_LEAD_HOUR / 24.0, color=GRAY, lw=1.0, ls="--", zorder=0)
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("forecast day")
        ax.set_ylabel("RMSE" + (" (K)" if var.startswith("T") else " (m/s)"))
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{model} — global RMSE vs. real truth — {exp_name}\n"
                 f"(dashed line = injection start, day 30)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_dir = OUT / model; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"global_rmse_timeseries_{exp_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# B) Latitude-band RMSE time series -- 4 single-latitude slices (2N/14N/40N/60N),
#    NOT averaged bands (fixed 2026-07-25: the original convention averaged
#    2N-14N and 40N-60N into 2 bands; the user actually wants these 4 exact
#    latitudes reported individually, no averaging -- the underlying CSV
#    columns changed from _2N14N/_40N60N to _2N/_14N/_40N/_60N to match).
# ---------------------------------------------------------------------------
def plot_latband_rmse(model, cfg, exp_type):
    exp_name = cfg["experiments"][exp_type]
    csv_path = BASE / cfg["dir"] / exp_name / f"latband_rmse_{exp_name}.csv"
    if not csv_path.exists():
        print(f"  [latband_rmse] {model}/{exp_type}: SKIP -- {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    variables = cfg["temp_vars"] if exp_type == "temp" else cfg["wind_vars"]
    variables = [v for v in variables if f"control_rmse_{v}_2N" in df.columns]

    slice_colors = {"2N": BLUE, "14N": "#3fa34d", "40N": ORANGE, "60N": "#9b3fd4"}
    fig, axes = plt.subplots(1, len(variables), figsize=(4.6 * len(variables), 3.8), squeeze=False)
    days = df["lead_hour"] / 24.0
    for ax, var in zip(axes[0], variables):
        for lat_key, color in slice_colors.items():
            dropna_plot(ax, days, df[f"control_rmse_{var}_{lat_key}"],
                        color=color, lw=1.4, ls="-", label=f"{lat_key}, control")
            dropna_plot(ax, days, df[f"experiment_rmse_{var}_{lat_key}"],
                        color=color, lw=1.4, ls="--", label=f"{lat_key}, experiment")
        ax.axvline(INJECT_LEAD_HOUR / 24.0, color=GRAY, lw=1.0, ls="--", zorder=0)
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("forecast day")
        ax.set_ylabel("RMSE" + (" (K)" if var.startswith("T") else " (m/s)"))
        ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle(f"{model} — latitude-slice RMSE vs. real truth — {exp_name}\n"
                 f"(2N/14N/40N/60N single latitudes, not averaged bands; "
                 f"dashed vertical line = injection start, day 30; dashed curve = experiment)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_dir = OUT / model; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"latband_rmse_timeseries_{exp_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# C) Spatial bias map -- 3 columns (control, experiment, diff) x 1 row per variable.
#    day60=False: post-injection mean, day 30-60 (original convention).
#    day60=True: day-60-only snapshot (added 2026-07-25 alongside the day30-60
#    mean, per user request -- reads the "_day60"-suffixed .npy files the
#    spatial_bias scripts now also save).
# ---------------------------------------------------------------------------
def plot_spatial_bias_map(model, cfg, exp_type, day60=False):
    exp_name = cfg["experiments"][exp_type]
    exp_dir = BASE / cfg["dir"] / exp_name
    suffix = "_day60" if day60 else ""
    variables = cfg["temp_vars"] if exp_type == "temp" else cfg["wind_vars"]
    variables = [v for v in variables if (exp_dir / f"bias_control_{v}{suffix}.npy").exists()]
    if not variables:
        print(f"  [spatial_bias{suffix}] {model}/{exp_type}: SKIP -- no bias{suffix} .npy files found")
        return
    lats, lons = cfg["lats"], cfg["lons"]

    fig, axes = plt.subplots(len(variables), 3, figsize=(13, 3.6 * len(variables)),
                              subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    for row, var in enumerate(variables):
        ctrl = np.load(exp_dir / f"bias_control_{var}{suffix}.npy")
        exp = np.load(exp_dir / f"bias_experiment_{var}{suffix}.npy")
        diff = exp - ctrl
        shared_vmax = np.nanpercentile(np.abs(np.concatenate([ctrl.ravel(), exp.ravel()])), 98)
        diff_vmax = np.nanpercentile(np.abs(diff), 98)
        units = "K" if var.startswith("T") else "m/s"

        for col, (field, title, vmax) in enumerate([
            (ctrl, f"{var} control bias", shared_vmax),
            (exp, f"{var} experiment bias", shared_vmax),
            (diff, f"{var} experiment − control", diff_vmax),
        ]):
            ax = axes[row][col]
            im = ax.pcolormesh(lons, lats, field, transform=ccrs.PlateCarree(),
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.coastlines(linewidth=0.5)
            ax.set_title(title, fontsize=9)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85, aspect=25)
            cb.set_label(units, fontsize=8)
            cb.ax.tick_params(labelsize=7)

    period_label = "day 60 snapshot" if day60 else "post-injection mean, day 30–60"
    fig.suptitle(f"{model} — {period_label} bias (forecast − truth) — {exp_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir = OUT / model; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"spatial_bias_map_{exp_name}{suffix}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# D) Correlation-length summary -- empirical vs. target L, by band and direction
# ---------------------------------------------------------------------------
def plot_corr_length_summary(model, cfg, exp_type):
    exp_name = cfg["experiments"][exp_type]
    csv_path = BASE / cfg["dir"] / exp_name / f"corr_length_horizontal_{exp_name}.csv"
    if not csv_path.exists():
        print(f"  [corr_length] {model}/{exp_type}: SKIP -- {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    variables = [v for v in cfg["corr_vars"][exp_type] if v in df["variable"].unique()]
    if not variables:
        print(f"  [corr_length] {model}/{exp_type}: SKIP -- no matching variables in CSV")
        return

    band_order = ["equator_5N", "midlat_45N", "highlat_70N"]
    band_labels = {"equator_5N": "5N\n(equator)", "midlat_45N": "45N\n(midlat)", "highlat_70N": "70N\n(high-lat)"}
    x_pos = {b: i for i, b in enumerate(band_order)}

    fig, axes = plt.subplots(1, len(variables), figsize=(4.0 * len(variables), 4.2), squeeze=False)
    target_L = df["target_L_km"].iloc[0]
    # 2026-07-25: corr_length_horizontal CSVs now carry a level_hpa column
    # (300/500/1000 for upper-air rows, NaN for surface rows) since the
    # horizontal check was extended from one representative level (500hPa)
    # to all 3 -- pick 500hPa here to keep this quick-look summary a single
    # point per band/direction as originally designed, not 3 overlapping ones.
    has_level_col = "level_hpa" in df.columns
    for ax, var in zip(axes[0], variables):
        sub = df[df["variable"] == var]
        if has_level_col:
            sub = sub[sub["level_hpa"].isna() | (sub["level_hpa"] == 500)]
        for direction, color, marker, dx in [("zonal", BLUE, "o", -0.08), ("meridional", ORANGE, "s", 0.08)]:
            d = sub[sub["direction"] == direction]
            xs = [x_pos[b] + dx for b in d["band"]]
            ax.scatter(xs, d["empirical_L_km"], color=color, marker=marker, s=55,
                       label=direction, zorder=3, edgecolors="white", linewidths=0.6)
        ax.axhline(target_L, color=GRAY, lw=1.2, ls="--", zorder=1, label=f"target ({target_L:.0f} km)")
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels([band_labels[b] for b in band_order], fontsize=8)
        ax.set_ylabel("empirical correlation length (km)")
        ax.set_title(var, fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{model} — horizontal correlation-length recovery vs. latitude — {exp_name}\n"
                 f"(pole convergence of the fixed-degree grid distorts zonal spacing at high latitude)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_dir = OUT / model; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"corr_length_summary_{exp_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    for model, cfg in MODELS.items():
        for exp_type in ["temp", "uv"]:
            print(f"\n=== {model} / {exp_type} ===")
            plot_global_rmse(model, cfg, exp_type)
            plot_latband_rmse(model, cfg, exp_type)
            if cfg["lats"] is not None:
                plot_spatial_bias_map(model, cfg, exp_type, day60=False)
                plot_spatial_bias_map(model, cfg, exp_type, day60=True)
            else:
                print(f"  [spatial_bias] {model}/{exp_type}: SKIP -- grid not wired up for this model yet")
            plot_corr_length_summary(model, cfg, exp_type)
    print("\nDone.")
