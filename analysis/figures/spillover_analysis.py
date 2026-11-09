#!/usr/bin/env python3
"""
Spillover analysis for T_LAND / T_OCEAN insertion experiments (Aurora, FCNv2, Prithvi WxC).

Option 1: Spillover-distance decay curves.
    For each model, restrict to the NOT-inserted region (ocean for T_LAND, land for
    T_OCEAN), bin every grid cell by its great-circle distance to the nearest
    inserted (source) grid cell, and compute the T500 RMSE-improvement (%) of the
    experiment over the control within each distance bin, using the validated
    last-20-day (days 41-60) averaging window. Produces 2 figures (one per
    experiment type), each with 3 lines (one per model).

Option 2: Small-multiples spreading maps.
    For each model x 6 snapshot days (31/35/40/45/50/60), map the T500
    error-reduction field (|control error| - |experiment error|) to visualise how
    the truth-insertion signal spreads from the source region into the
    not-inserted region over time. Produces 2 figures (one per experiment type),
    each a 3 (models) x 6 (days) grid of maps.
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4 as nc4
import h5py
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, "/umbc/rs/pi_halem/users/yashd1")
from cross_var_rmse_grouped import (
    AURORA_IC, FCN_IC, PRITHVI_IC,
    AURORA_CTRL, FCN_CTRL, PRITHVI_CTRL,
    AURORA_T_FILES, FCN_T_FILES, PRITHVI_T_FILES,
    AURORA_EXP_IDX, FCN_EXP_IDX, PRITHVI_EXP_IDX,
    AURORA_CTRL_IDX, FCN_CTRL_IDX,
    ERA5Cache, era5_idx_nearest, merra2_pres_field, PRI_LI_500,
)

BASE = Path("/umbc/rs/pi_halem/users/yashd1")
OUT_DIR = BASE / "final_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EARTH_R_KM = 6371.0

# ─── DISTANCE BINS (km), finer near the insertion boundary ───────────────────
DIST_BIN_EDGES = np.array([0, 100, 200, 350, 500, 700, 900, 1200, 1500,
                            1900, 2400, 3000, 4000, 6000, 10000, 20100], dtype=float)
N_BINS = len(DIST_BIN_EDGES) - 1
BIN_CENTERS = 0.5 * (DIST_BIN_EDGES[:-1] + DIST_BIN_EDGES[1:])
BIN_CENTERS[-1] = DIST_BIN_EDGES[-2] + 1500.0  # representative point for the open-ended tail bin

# ─── GEODESIC DISTANCE FIELD (cKDTree on unit-sphere Cartesian coords) ───────

def latlon_to_xyz(lat_deg, lon_deg):
    latr = np.deg2rad(lat_deg)
    lonr = np.deg2rad(lon_deg)
    x = np.cos(latr) * np.cos(lonr)
    y = np.cos(latr) * np.sin(lonr)
    z = np.sin(latr)
    return x, y, z

def dist_to_nearest_source(lat1d, lon1d, source_mask):
    """Great-circle distance (km) from every grid cell to the nearest True cell
    in source_mask. Cells inside source_mask itself get distance 0."""
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    x, y, z = latlon_to_xyz(lat2d, lon2d)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    src_idx = np.where(source_mask.ravel())[0]
    tree = cKDTree(pts[src_idx])
    chord, _ = tree.query(pts, k=1)
    chord = np.clip(chord, 0.0, 2.0)
    theta = 2.0 * np.arcsin(chord / 2.0)
    return (theta * EARTH_R_KM).reshape(source_mask.shape)

# ─── PER-MODEL GRID + LAND MASK ───────────────────────────────────────────────

def aurora_grid_and_mask():
    with h5py.File(str(AURORA_T_FILES["ocean"]), "r") as f:
        lat = np.array(f["lat"])
        lon = np.array(f["lon"])
    with h5py.File(str(BASE / "AURORA_DATA" / "static.nc"), "r") as f:
        lsm = np.array(f["lsm"])[0]          # (721, 1440), 90 -> -90 descending
    lsm = lsm[:-1, :]                         # drop -90 row -> (720, 1440), matches exp grid
    land_mask = lsm > 0.5
    return lat, lon, land_mask

def fcn_grid_and_mask():
    with h5py.File(str(FCN_T_FILES["ocean"]), "r") as f:
        lat = np.array(f["lat"])
        lon = np.array(f["lon"])
    with h5py.File(str(BASE / "FCN3" / "makani" / "stat-files" / "land_mask.nc"), "r") as f:
        lsm = np.array(f["LSM"])[0]          # (721, 1440), already matches FCN/ERA5 grid exactly
    land_mask = lsm > 0.5
    return lat, lon, land_mask

def prithvi_grid_and_mask():
    with h5py.File(str(PRITHVI_T_FILES["ocean"]), "r") as f:
        lat = np.array(f["lat"])
        lon = np.array(f["lon"])
    with h5py.File(str(BASE / "prithvi" / "data" / "merra2" / "MERRA2_sfc_20220101.nc"), "r") as f:
        frland = np.array(f["FRLAND"])[0] if np.array(f["FRLAND"]).ndim == 3 else np.array(f["FRLAND"])
    frland = frland[1:, :]                    # drop -90 row -> (360, 576), ascending, matches exp grid
    land_mask = frland > 0.5
    return lat, lon, land_mask

# ─── FIELD ACCESSORS (T500 only) ─────────────────────────────────────────────

def aurora_fc_field(handle, idx):
    return np.array(handle["t500"][idx])

def aurora_ctrl_field(handle, idx):
    return np.array(handle["t500"][idx])

def fcn_fc_field(handle, idx):
    return np.array(handle["prediction"][idx, 54, :, :])

def fcn_ctrl_field(handle, idx):
    return np.array(handle["prediction"][idx, 54, :, :])

def prithvi_fc_field(handle, idx):
    return np.array(handle["T"][idx, PRI_LI_500, :, :])

# ─── BINNED ACCUMULATION ──────────────────────────────────────────────────────

def accumulate_bins(idx_list, get_fc, get_truth, weight2d, eval_mask, dist_field):
    flat_mask = eval_mask.ravel()
    bin_idx_all = np.clip(np.digitize(dist_field.ravel(), DIST_BIN_EDGES) - 1, 0, N_BINS - 1)
    bin_idx_eval = bin_idx_all[flat_mask]
    w_eval = weight2d.ravel()[flat_mask]
    sumsq = np.zeros(N_BINS)
    sumw = np.zeros(N_BINS)
    for idx in idx_list:
        fc = get_fc(idx).ravel()[flat_mask]
        truth = get_truth(idx).ravel()[flat_mask]
        sq = w_eval * (fc - truth) ** 2
        sumsq += np.bincount(bin_idx_eval, weights=sq, minlength=N_BINS)
        sumw += np.bincount(bin_idx_eval, weights=w_eval, minlength=N_BINS)
    return sumsq, sumw

def bins_to_rmse(sumsq, sumw):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(sumsq / sumw)

# ─── OPTION 1: PER-MODEL DECAY CURVES ────────────────────────────────────────

def aurora_decay(era5_cache, exp_type):
    """exp_type: 'land' or 'ocean' (the INSERTED region)."""
    lat, lon, land_mask = aurora_grid_and_mask()
    weight2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    eval_mask = ~land_mask if exp_type == "land" else land_mask   # NOT-inserted region
    source_mask = land_mask if exp_type == "land" else ~land_mask
    dist_field = dist_to_nearest_source(lat, lon, source_mask)

    era5_times, era5_data, era5_lat, _ = era5_cache.get("t", 500)
    era5_t500 = era5_data[:, :-1, :] if era5_lat[0] > era5_lat[-1] else era5_data[:, 1:, :]

    f_exp = h5py.File(str(AURORA_T_FILES[exp_type]), "r")
    truth_cache_exp = {}
    def get_truth_exp(idx):
        if idx not in truth_cache_exp:
            vt = AURORA_IC + timedelta(hours=idx * 6)
            ei = era5_idx_nearest(era5_times, vt)
            truth_cache_exp[idx] = era5_t500[ei]
        return truth_cache_exp[idx]
    sumsq_e, sumw_e = accumulate_bins(AURORA_EXP_IDX, lambda i: aurora_fc_field(f_exp, i),
                                       get_truth_exp, weight2d, eval_mask, dist_field)
    f_exp.close()

    f_ctrl = nc4.Dataset(str(AURORA_CTRL))
    truth_cache_ctrl = {}
    def get_truth_ctrl(idx):
        if idx not in truth_cache_ctrl:
            vt = AURORA_IC + timedelta(hours=int(f_ctrl["lead_time"][idx]))
            ei = era5_idx_nearest(era5_times, vt)
            truth_cache_ctrl[idx] = era5_t500[ei]
        return truth_cache_ctrl[idx]
    sumsq_c, sumw_c = accumulate_bins(AURORA_CTRL_IDX, lambda i: aurora_ctrl_field(f_ctrl, i),
                                       get_truth_ctrl, weight2d, eval_mask, dist_field)
    f_ctrl.close()

    return bins_to_rmse(sumsq_c, sumw_c), bins_to_rmse(sumsq_e, sumw_e)

def fcn_decay(era5_cache, exp_type):
    lat, lon, land_mask = fcn_grid_and_mask()
    weight2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    eval_mask = ~land_mask if exp_type == "land" else land_mask
    source_mask = land_mask if exp_type == "land" else ~land_mask
    dist_field = dist_to_nearest_source(lat, lon, source_mask)

    era5_times, era5_data, _, _ = era5_cache.get("t", 500)

    f_exp = h5py.File(str(FCN_T_FILES[exp_type]), "r")
    truth_cache_exp = {}
    def get_truth_exp(idx):
        if idx not in truth_cache_exp:
            vt = FCN_IC + timedelta(hours=idx * 6)
            ei = era5_idx_nearest(era5_times, vt)
            truth_cache_exp[idx] = era5_data[ei]
        return truth_cache_exp[idx]
    sumsq_e, sumw_e = accumulate_bins(FCN_EXP_IDX, lambda i: fcn_fc_field(f_exp, i),
                                       get_truth_exp, weight2d, eval_mask, dist_field)
    f_exp.close()

    f_ctrl = nc4.Dataset(str(FCN_CTRL))
    truth_cache_ctrl = {}
    def get_truth_ctrl(idx):
        if idx not in truth_cache_ctrl:
            vt = FCN_IC + timedelta(hours=float(f_ctrl["time"][idx]))
            ei = era5_idx_nearest(era5_times, vt)
            truth_cache_ctrl[idx] = era5_data[ei]
        return truth_cache_ctrl[idx]
    sumsq_c, sumw_c = accumulate_bins(FCN_CTRL_IDX, lambda i: fcn_ctrl_field(f_ctrl, i),
                                       get_truth_ctrl, weight2d, eval_mask, dist_field)
    f_ctrl.close()

    return bins_to_rmse(sumsq_c, sumw_c), bins_to_rmse(sumsq_e, sumw_e)

def prithvi_decay(exp_type):
    lat, lon, land_mask = prithvi_grid_and_mask()
    weight2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    eval_mask = ~land_mask if exp_type == "land" else land_mask
    source_mask = land_mask if exp_type == "land" else ~land_mask
    dist_field = dist_to_nearest_source(lat, lon, source_mask)

    f_exp = h5py.File(str(PRITHVI_T_FILES[exp_type]), "r")
    truth_cache = {}
    def get_truth(idx):
        if idx not in truth_cache:
            vt = PRITHVI_IC + timedelta(hours=(idx + 1) * 6)
            truth_cache[idx] = merra2_pres_field("T", 51, vt)[1:, :]
        return truth_cache[idx]
    sumsq_e, sumw_e = accumulate_bins(PRITHVI_EXP_IDX, lambda i: prithvi_fc_field(f_exp, i),
                                       get_truth, weight2d, eval_mask, dist_field)
    f_exp.close()

    f_ctrl = h5py.File(str(PRITHVI_CTRL), "r")
    sumsq_c, sumw_c = accumulate_bins(PRITHVI_EXP_IDX, lambda i: prithvi_fc_field(f_ctrl, i),
                                       get_truth, weight2d, eval_mask, dist_field)
    f_ctrl.close()

    return bins_to_rmse(sumsq_c, sumw_c), bins_to_rmse(sumsq_e, sumw_e)

# ─── OPTION 1: PLOTTING ──────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})
MODEL_COLORS = {"Aurora": "#4C72B0", "FCNv2": "#DD8452", "Prithvi WxC": "#55A868"}
MODEL_MARKERS = {"Aurora": "o", "FCNv2": "s", "Prithvi WxC": "^"}

def plot_decay(curves, exp_type, out_path):
    """curves: dict[model_name] -> (ctrl_rmse_bins, exp_rmse_bins)"""
    fig, ax = plt.subplots(figsize=(9.2, 6.4), constrained_layout=True)
    x = BIN_CENTERS
    for model, (ctrl, exp) in curves.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_imp = 100.0 * (ctrl - exp) / ctrl
        valid = np.isfinite(pct_imp)
        ax.plot(x[valid], pct_imp[valid], marker=MODEL_MARKERS[model], markersize=6.5,
                linewidth=2.2, color=MODEL_COLORS[model], label=model,
                markeredgecolor="white", markeredgewidth=0.6)

    ax.axhline(0, color="black", lw=1.0, ls="-", alpha=0.6)
    ax.set_xscale("symlog", linthresh=200)
    ax.set_xlabel(f"Great-circle distance from nearest inserted "
                   f"({'land' if exp_type == 'land' else 'ocean'}) cell (km)", fontweight="bold")
    ax.set_ylabel("T500 RMSE Improvement vs. Control (%)", fontweight="bold")
    region_word = "Ocean" if exp_type == "land" else "Land"
    src_word = "Land" if exp_type == "land" else "Ocean"
    ax.set_title(f"Spillover of Temperature-Insertion Skill from {src_word} into {region_word}\n"
                 f"T500, Last-20-Day Average (Days 41–60) vs. Control", fontweight="bold")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.45, color="#999999")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#555555")
    ax.legend(title="Model", loc="best", frameon=True, framealpha=0.92, edgecolor="#bbbbbb")
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ─── OPTION 2: SNAPSHOT MAPS ─────────────────────────────────────────────────

SNAPSHOT_DAYS = [31, 35, 40, 45, 50, 60]

def make_truth_lookup_aurora_fcn(era5_cache):
    times, data, lat, _ = era5_cache.get("t", 500)
    return times, data, lat

def option2_aurora(era5_cache, exp_type):
    lat, lon, land_mask = aurora_grid_and_mask()
    times, data, era5_lat = make_truth_lookup_aurora_fcn(era5_cache)
    era5_t500 = data[:, :-1, :] if era5_lat[0] > era5_lat[-1] else data[:, 1:, :]
    f_exp = h5py.File(str(AURORA_T_FILES[exp_type]), "r")
    f_ctrl = nc4.Dataset(str(AURORA_CTRL))
    maps = []
    for day in SNAPSHOT_DAYS:
        lead_h = day * 24
        exp_idx = min(lead_h // 6, 239)
        ctrl_idx = min(lead_h // 12, 120)
        vt_exp = AURORA_IC + timedelta(hours=exp_idx * 6)
        vt_ctrl = AURORA_IC + timedelta(hours=int(f_ctrl["lead_time"][ctrl_idx]))
        truth_exp = era5_t500[era5_idx_nearest(times, vt_exp)]
        truth_ctrl = era5_t500[era5_idx_nearest(times, vt_ctrl)]
        err_exp = np.abs(aurora_fc_field(f_exp, exp_idx) - truth_exp)
        err_ctrl = np.abs(aurora_ctrl_field(f_ctrl, ctrl_idx) - truth_ctrl)
        maps.append(err_ctrl - err_exp)
    f_exp.close(); f_ctrl.close()
    return lat, lon, land_mask, maps

def option2_fcn(era5_cache, exp_type):
    lat, lon, land_mask = fcn_grid_and_mask()
    times, data, _ = make_truth_lookup_aurora_fcn(era5_cache)
    f_exp = h5py.File(str(FCN_T_FILES[exp_type]), "r")
    f_ctrl = nc4.Dataset(str(FCN_CTRL))
    n_ctrl = f_ctrl["time"].shape[0]
    maps = []
    for day in SNAPSHOT_DAYS:
        lead_h = day * 24
        exp_idx = min(lead_h // 6, 239)
        ctrl_idx = min(lead_h // 12, n_ctrl - 1)
        vt_exp = FCN_IC + timedelta(hours=exp_idx * 6)
        vt_ctrl = FCN_IC + timedelta(hours=float(f_ctrl["time"][ctrl_idx]))
        truth_exp = data[era5_idx_nearest(times, vt_exp)]
        truth_ctrl = data[era5_idx_nearest(times, vt_ctrl)]
        err_exp = np.abs(fcn_fc_field(f_exp, exp_idx) - truth_exp)
        err_ctrl = np.abs(fcn_ctrl_field(f_ctrl, ctrl_idx) - truth_ctrl)
        maps.append(err_ctrl - err_exp)
    f_exp.close(); f_ctrl.close()
    return lat, lon, land_mask, maps

def option2_prithvi(exp_type):
    lat, lon, land_mask = prithvi_grid_and_mask()
    f_exp = h5py.File(str(PRITHVI_T_FILES[exp_type]), "r")
    f_ctrl = h5py.File(str(PRITHVI_CTRL), "r")
    maps = []
    for day in SNAPSHOT_DAYS:
        lead_h = day * 24
        idx = min(lead_h // 6 - 1, 239)
        vt = PRITHVI_IC + timedelta(hours=(idx + 1) * 6)
        truth = merra2_pres_field("T", 51, vt)[1:, :]
        err_exp = np.abs(prithvi_fc_field(f_exp, idx) - truth)
        err_ctrl = np.abs(prithvi_fc_field(f_ctrl, idx) - truth)
        maps.append(err_ctrl - err_exp)
    f_exp.close(); f_ctrl.close()
    return lat, lon, land_mask, maps

def plot_option2(model_data, exp_type, out_path):
    """model_data: list of (model_name, lat, lon, land_mask, maps[6])"""
    n_models = len(model_data)
    n_days = len(SNAPSHOT_DAYS)
    fig, axes = plt.subplots(n_models, n_days, figsize=(3.1 * n_days, 2.7 * n_models),
                              constrained_layout=True)
    vmax = 0.0
    for _, _, _, _, maps in model_data:
        for m in maps:
            vmax = max(vmax, np.nanpercentile(np.abs(m), 98))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = None
    for r, (model, lat, lon, land_mask, maps) in enumerate(model_data):
        for c, day in enumerate(SNAPSHOT_DAYS):
            ax = axes[r, c]
            im = ax.imshow(maps[c], extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                            origin="upper" if lat[0] > lat[-1] else "lower",
                            cmap="RdBu", norm=norm, aspect="auto")
            ax.contour(lon, lat, land_mask.astype(float), levels=[0.5],
                       colors="black", linewidths=0.5, alpha=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Day {day}", fontsize=12.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(model, fontsize=12.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.012, aspect=35)
    cbar.set_label("T500 Error Reduction vs. Control (K)\n(Control |error| − Experiment |error|)",
                   fontsize=12, fontweight="bold")

    src_word = "Land" if exp_type == "land" else "Ocean"
    region_word = "Ocean" if exp_type == "land" else "Land"
    fig.suptitle(f"Spatial Spreading of Temperature-Insertion Skill: {src_word} → {region_word}\n"
                 f"Black contour = land/ocean boundary  |  Blue = improvement, Red = degradation",
                 fontsize=15, fontweight="bold")
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ─── METHODOLOGY TEXT FILE (OPTION 1) ────────────────────────────────────────

METHODOLOGY_TEXT = """\
SPILLOVER-DISTANCE DECAY CURVES (Option 1) — Methodology
==========================================================

WHAT THIS SHOWS
----------------
For the T_LAND and T_OCEAN truth-insertion experiments, these figures quantify
how the forecast-skill benefit of inserting ERA5/MERRA-2 temperature truth into
one geographic region (land or ocean) "spills over" into the complementary,
NOT-inserted region (ocean or land) as a function of great-circle distance from
the insertion boundary. Two figures are produced:

  fig_spillover_decay_T_LAND.png   (truth inserted over LAND -> measured over OCEAN)
  fig_spillover_decay_T_OCEAN.png  (truth inserted over OCEAN -> measured over LAND)

Each figure overlays one decay curve per model (Aurora, FCNv2, Prithvi WxC),
plotting T500 RMSE improvement (%) over the control forecast as a function of
distance (km, symmetric-log x-axis) from the nearest grid cell that actually
received truth insertion.

HOW IT WAS CALCULATED
----------------------
1. Land/sea mask per model (boolean, True = land):
     Aurora    : AURORA_DATA/static.nc, variable "lsm" > 0.5
                 (721x1440 grid, cropped to 720 rows to match the model's
                 90 -> -89.75 deg latitude grid by dropping the -90 deg row)
     FCNv2     : FCN3/makani/stat-files/land_mask.nc, variable "LSM" > 0.5
                 (721x1440 grid, identical to the model's own output grid)
     Prithvi WxC: prithvi/data/merra2/MERRA2_sfc_20220101.nc, variable
                 "FRLAND" > 0.5 (361x576 grid, cropped to 360 rows to match the
                 model's -89.5 -> 90 deg latitude grid by dropping the -90 deg row)

2. Great-circle distance field:
   Every grid cell's latitude/longitude is converted to a unit-sphere Cartesian
   coordinate (x,y,z) = (cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)). A
   scipy.spatial.cKDTree is built from the Cartesian coordinates of every cell
   belonging to the SOURCE (inserted) class (land cells for T_LAND, ocean cells
   for T_OCEAN). Every cell in the grid is then queried against this tree for
   its nearest source-class neighbour; the resulting unit-sphere chord distance
   c is converted to a great-circle distance via d = 2 * R_Earth * arcsin(c/2),
   with R_Earth = 6371 km. Source-class cells themselves get distance 0.

3. Evaluation domain:
   Only grid cells belonging to the NOT-inserted class are scored (ocean cells
   for T_LAND, land cells for T_OCEAN) — i.e. the analysis measures how far the
   benefit of insertion propagates beyond the region where truth was directly
   written into the forecast state, not the directly-inserted region itself
   (which trivially has near-zero error after insertion).

4. Distance binning:
   Evaluation-domain cells are assigned to one of 15 great-circle-distance bins
   with edges (km): 0, 100, 200, 350, 500, 700, 900, 1200, 1500, 1900, 2400,
   3000, 4000, 6000, 10000, 20100. Bins are finer near the insertion boundary
   (where the spillover signal changes fastest) and coarser farther away. The
   final bin is open-ended and captures all remaining far-field cells (e.g.
   mid-ocean / continental-interior points).

5. RMSE per bin:
   For every 6-hourly step in the validated last-20-day analysis window (days
   41-60 of the 60-day rollout, i.e. the window already used throughout this
   thesis chapter for % RMSE-improvement statistics), the squared error between
   the model's T500 field and the ERA5 (Aurora/FCNv2) or MERRA-2 (Prithvi WxC)
   truth field is computed, weighted by cos(latitude) to correct for the
   convergence of meridians at the poles. These weighted squared errors are
   accumulated per distance bin, summed jointly over time AND space within each
   bin, and a single RMSE is taken as the square root of (summed weighted
   squared error / summed weight) per bin — i.e. RMSE = sqrt(mean(SE)), not the
   average of per-timestep RMSE values. This matches the joint time-space
   averaging convention used everywhere else in this chapter's RMSE analysis.
   This is done independently for the control forecast and the experiment
   (insertion) forecast, each evaluated at its own native output cadence
   (control: 12-hourly; experiment: 6-hourly) against the nearest available
   truth timestamp.

6. % RMSE improvement per bin:
     improvement(%) = 100 * (RMSE_control - RMSE_experiment) / RMSE_control

   This is the quantity plotted on the y-axis against the bin's representative
   distance (bin midpoint; the open final bin uses a representative distance of
   ~11500 km) on the x-axis.

INTERPRETATION
--------------
A curve that stays high (close to the directly-inserted region's improvement
level) out to large distances indicates strong large-scale dynamical coupling
that propagates the inserted information far beyond the insertion boundary
(e.g. via large-scale wave/advective transport encoded by the model). A curve
that decays quickly toward 0% close to the boundary indicates that the model's
benefit from insertion is geographically localized to the directly-inserted
region, with little cross-boundary spillover.
"""

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print(f"Start: {datetime.now()}\n")
    era5_cache = ERA5Cache()

    # ---------- OPTION 1 ----------
    print("=== OPTION 1: spillover-distance decay curves ===")
    for exp_type in ["land", "ocean"]:
        print(f"-- T_{exp_type.upper()} --")
        ctrl_a, exp_a = aurora_decay(era5_cache, exp_type)
        print("  Aurora done")
        ctrl_f, exp_f = fcn_decay(era5_cache, exp_type)
        print("  FCNv2 done")
        ctrl_p, exp_p = prithvi_decay(exp_type)
        print("  Prithvi done")

        curves = {"Aurora": (ctrl_a, exp_a), "FCNv2": (ctrl_f, exp_f),
                  "Prithvi WxC": (ctrl_p, exp_p)}

        rows = []
        for model, (ctrl, exp) in curves.items():
            for i in range(N_BINS):
                rows.append({"model": model, "bin_low_km": DIST_BIN_EDGES[i],
                             "bin_high_km": DIST_BIN_EDGES[i + 1],
                             "bin_center_km": BIN_CENTERS[i],
                             "control_rmse": ctrl[i], "experiment_rmse": exp[i],
                             "pct_improvement": 100 * (ctrl[i] - exp[i]) / ctrl[i] if ctrl[i] else np.nan})
        pd.DataFrame(rows).to_csv(OUT_DIR / f"spillover_decay_T_{exp_type.upper()}.csv", index=False)

        plot_decay(curves, exp_type, OUT_DIR / f"fig_spillover_decay_T_{exp_type.upper()}.png")

    (OUT_DIR / "spillover_decay_methodology.txt").write_text(METHODOLOGY_TEXT)
    print(f"  Saved: {OUT_DIR / 'spillover_decay_methodology.txt'}")

    # ---------- OPTION 2 ----------
    print("\n=== OPTION 2: small-multiples spreading maps ===")
    for exp_type in ["land", "ocean"]:
        print(f"-- T_{exp_type.upper()} --")
        lat_a, lon_a, mask_a, maps_a = option2_aurora(era5_cache, exp_type)
        print("  Aurora done")
        lat_f, lon_f, mask_f, maps_f = option2_fcn(era5_cache, exp_type)
        print("  FCNv2 done")
        lat_p, lon_p, mask_p, maps_p = option2_prithvi(exp_type)
        print("  Prithvi done")

        model_data = [("Aurora", lat_a, lon_a, mask_a, maps_a),
                      ("FCNv2", lat_f, lon_f, mask_f, maps_f),
                      ("Prithvi WxC", lat_p, lon_p, mask_p, maps_p)]
        plot_option2(model_data, exp_type, OUT_DIR / f"fig_spillover_maps_T_{exp_type.upper()}.png")

    print(f"\nDone: {datetime.now()}")

if __name__ == "__main__":
    main()
