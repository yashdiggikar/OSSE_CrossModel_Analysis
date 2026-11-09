#!/usr/bin/env python3
"""
Spatial bias (mean signed error, forecast - truth) for the Prithvi WxC
Correlated OSSE Perturbation experiments (perturbed_corr_temp_ins,
perturbed_corr_uv_ins).

Computed over the POST-INJECTION period only (step >= 120, day 30-60), for
both the control forecast and the experiment forecast — see
FCN/compute_spatial_bias_corr.py's docstring for the full rationale.

Uses the CORRECT control file per Organized_results/file_paths.md
("prithviOSSE/run2/output/control_forecast_Jan2022.nc") — see
compute_global_rmse_corr.py's docstring for the history of the stale
control-file bug in prithviOSSE/analyze_osse.py that this avoids.

Output per experiment (into PERTURBATION_EXPERIMENTS/Prithvi/<experiment>/):
    bias_control_<VAR>.npy, bias_experiment_<VAR>.npy   (lat,lon) mean signed error
    spatial_bias_<experiment>.csv                        summary table (global + 3 lat bands)
    spatial_bias_map_<experiment>_<VAR>.png              experiment-minus-control bias difference map

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_spatial_bias_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
import time
from pathlib import Path
from datetime import datetime, timedelta

import netCDF4 as nc4
import h5py
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

BASE       = Path("/umbc/rs/pi_halem/users/yashd1")
PRITHVI    = BASE / "prithviOSSE"
MERRA2_DIR = BASE / "prithvi" / "data" / "merra2"
CTRL_FILE  = PRITHVI / "run2" / "output" / "control_forecast_Jan2022.nc"   # Root cause of the "OSError: NetCDF: HDF error" seen throughout SESSION_LOG_2026-07-22.md 10.2 was import order: h5py imported before netCDF4 makes netCDF4.Dataset() fail on these files -- fixed by importing netCDF4 first (see imports above).
OUT_DIR    = BASE / "PERTURBATION_EXPERIMENTS" / "Prithvi"

EXPERIMENTS = [
    ("perturbed_corr_temp_ins", PRITHVI / "perturbed_corr_temp_ins" / "output" / "perturbed_corr_temp_osse_forecast_Jan2022.nc", "temp"),
    ("perturbed_corr_uv_ins",   PRITHVI / "perturbed_corr_uv_ins"   / "output" / "perturbed_corr_uv_osse_forecast_Jan2022.nc",   "uv"),
]

IC_T1       = datetime(2022, 1, 1, 6, 0)
DT_H        = 6
N_STEPS     = 240
INJECT_STEP = 120

LEV_VALS = [34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72]
LEVEL_APPROX_HPA = [52, 118, 164, 226, 265, 310, 424, 532, 604, 693, 832, 903, 946, 961]
PLOT_LEV = 51    # ~532 hPa, same choice as compute_global_rmse_corr.py / analyze_osse.py
PLOT_LEV_IDX = LEV_VALS.index(PLOT_LEV)

TARGET_HPA = [300, 500, 1000]
LEVEL_IDX_FOR_HPA = {}
for target in TARGET_HPA:
    nearest_i = min(range(len(LEVEL_APPROX_HPA)), key=lambda i: abs(LEVEL_APPROX_HPA[i] - target))
    LEVEL_IDX_FOR_HPA[target] = nearest_i
    print(f"  target {target}hPa -> nearest native lev {LEV_VALS[nearest_i]} (~{LEVEL_APPROX_HPA[nearest_i]}hPa)")

UPPER_VARS = {"T": "T", "U": "U", "V": "V", "Z": "H", "Q": "QV"}
SFC_VARS = [("T2M", "T2M"), ("U10M", "U10M"), ("V10M", "V10M")]

PRITHVI_LATS = np.linspace(-89.5, 90.0, 360)
PRITHVI_LONS = np.linspace(-180.0, 179.375, 576)

POLAR_MASK  = np.abs(PRITHVI_LATS) > 60
MIDLAT_MASK = (np.abs(PRITHVI_LATS) <= 60) & (np.abs(PRITHVI_LATS) >= 30)
TROP_MASK   = np.abs(PRITHVI_LATS) < 30


def merra2_tidx(valid_dt):
    return valid_dt.hour // 3


def load_merra2_sfc(date_str, varname, t_idx):
    fpath = MERRA2_DIR / f"MERRA2_sfc_{date_str}.nc"
    if not fpath.exists():
        return None
    with h5py.File(str(fpath), "r") as f:
        if varname not in f:
            return None
        return f[varname][t_idx, 1:, :].astype(np.float32)


def load_merra2_pres(date_str, varname, lev_idx, t_idx):
    fpath = MERRA2_DIR / f"MERRA_pres_{date_str}.nc"
    if not fpath.exists():
        return None
    with h5py.File(str(fpath), "r") as f:
        if varname not in f:
            return None
        merra_lev = list(f["lev"][()])
        lv = LEV_VALS[lev_idx]
        if int(lv) not in merra_lev:
            return None
        m_idx = merra_lev.index(int(lv))
        return f[varname][t_idx, m_idx, 1:, :].astype(np.float32)


def load_nc_var(ds, varname, n_steps, lev_idx=None):
    if lev_idx is None:
        return ds.variables[varname][:n_steps].astype(np.float32)
    return ds.variables[varname][:n_steps, lev_idx].astype(np.float32)


def region_stats(bias_map):
    return {
        "global_mean_bias": float(np.nanmean(bias_map)),
        "global_mean_abs_bias": float(np.nanmean(np.abs(bias_map))),
        "polar_mean_bias": float(np.nanmean(bias_map[POLAR_MASK, :])),
        "midlat_mean_bias": float(np.nanmean(bias_map[MIDLAT_MASK, :])),
        "tropics_mean_bias": float(np.nanmean(bias_map[TROP_MASK, :])),
    }


print("Verifying control file grid ...")
# Opened once and kept open for the whole script -- repeatedly open/closing
# the same multi-GB file per-variable was found to segfault on this NFS
# filesystem in the equivalent FCN analysis scripts; same fix applied here
# proactively.
ctrl_ds = None
for _attempt in range(10):
    try:
        ctrl_ds = nc4.Dataset(str(CTRL_FILE))
        assert ctrl_ds.dimensions["lat"].size == 360, (
            f"CTRL_FILE has {ctrl_ds.dimensions['lat'].size} lats, expected 360 — wrong control file."
        )
        break
    except OSError as e:
        print(f"  transient open error ({e!r}), retrying in 30s ...")
        ctrl_ds = None
        time.sleep(30)
else:
    raise RuntimeError(f"Could not open {CTRL_FILE} after 10 attempts.")
print(f"  CTRL_FILE OK: {CTRL_FILE}")

summary_rows = []

for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    valid_times = [IC_T1 + timedelta(hours=(s + 1) * DT_H) for s in range(N_STEPS)]
    post_steps = [s for s in range(N_STEPS) if s >= INJECT_STEP]
    exp_ds = nc4.Dataset(str(nc_file))

    # Every variable/level is evaluated for every experiment (cross-variable
    # coupling included), not just the directly-inserted family.
    var_defs = [(f"{label}{hpa}", varname, LEVEL_IDX_FOR_HPA[hpa])
                for label, varname in UPPER_VARS.items() for hpa in TARGET_HPA]
    var_defs += [(label, varname, None) for label, varname in SFC_VARS]
    last_step = N_STEPS - 1  # day-60-only snapshot, alongside the day-30-60 mean below

    for label, varname, lev_idx in var_defs:
        try:
            fc_data = load_nc_var(exp_ds, varname, N_STEPS, lev_idx)
            ctrl_data = load_nc_var(ctrl_ds, varname, N_STEPS, lev_idx)
        except (KeyError, IndexError) as e:
            print(f"  [{exp_name}] {label}: variable '{varname}' unavailable ({e}), skipping.")
            continue

        exp_diffs, ctrl_diffs = [], []
        for s in post_steps:
            vt = valid_times[s]
            date_str = vt.strftime("%Y%m%d")
            t_idx = merra2_tidx(vt)
            truth = (load_merra2_sfc(date_str, varname, t_idx) if lev_idx is None
                     else load_merra2_pres(date_str, varname, lev_idx, t_idx))
            if truth is None:
                continue
            exp_diffs.append(fc_data[s] - truth)
            ctrl_diffs.append(ctrl_data[s] - truth)

        if not exp_diffs:
            print(f"  [{exp_name}] {label}: no matched timesteps, skipping.")
            continue

        exp_bias = np.nanmean(np.stack(exp_diffs), axis=0)
        ctrl_bias = np.nanmean(np.stack(ctrl_diffs), axis=0)
        # netCDF4's default auto-masking makes fc_data/ctrl_data MaskedArrays, which
        # propagate through the subtraction/stack/nanmean above; np.save can't write
        # a MaskedArray (MaskedArray.tofile() is unimplemented) -- convert masked
        # entries to NaN (not just .data, which would keep the underlying garbage
        # fill value) before saving.
        exp_bias = np.ma.filled(exp_bias, np.nan) if np.ma.isMaskedArray(exp_bias) else exp_bias
        ctrl_bias = np.ma.filled(ctrl_bias, np.nan) if np.ma.isMaskedArray(ctrl_bias) else ctrl_bias

        np.save(out_dir / f"bias_experiment_{label}.npy", exp_bias)
        np.save(out_dir / f"bias_control_{label}.npy", ctrl_bias)

        # Day-60-only snapshot (single step), alongside the day-30-60 mean above.
        vt60 = valid_times[last_step]
        date_str60 = vt60.strftime("%Y%m%d")
        t_idx60 = merra2_tidx(vt60)
        truth60 = (load_merra2_sfc(date_str60, varname, t_idx60) if lev_idx is None
                   else load_merra2_pres(date_str60, varname, lev_idx, t_idx60))
        if truth60 is not None:
            exp_bias_day60 = fc_data[last_step] - truth60
            ctrl_bias_day60 = ctrl_data[last_step] - truth60
            exp_bias_day60 = np.ma.filled(exp_bias_day60, np.nan) if np.ma.isMaskedArray(exp_bias_day60) else exp_bias_day60
            ctrl_bias_day60 = np.ma.filled(ctrl_bias_day60, np.nan) if np.ma.isMaskedArray(ctrl_bias_day60) else ctrl_bias_day60
        else:
            exp_bias_day60 = np.full_like(exp_bias, np.nan)
            ctrl_bias_day60 = np.full_like(exp_bias, np.nan)
        np.save(out_dir / f"bias_experiment_{label}_day60.npy", exp_bias_day60)
        np.save(out_dir / f"bias_control_{label}_day60.npy", ctrl_bias_day60)

        exp_stats  = region_stats(exp_bias)
        ctrl_stats = region_stats(ctrl_bias)
        exp_stats_day60  = region_stats(exp_bias_day60)
        ctrl_stats_day60 = region_stats(ctrl_bias_day60)
        row = {"experiment": exp_name, "variable": label}
        row.update({f"control_{k}": v for k, v in ctrl_stats.items()})
        row.update({f"experiment_{k}": v for k, v in exp_stats.items()})
        row.update({f"control_day60_{k}": v for k, v in ctrl_stats_day60.items()})
        row.update({f"experiment_day60_{k}": v for k, v in exp_stats_day60.items()})
        summary_rows.append(row)
        print(f"  [{exp_name}] {label}: control global_mean_bias={ctrl_stats['global_mean_bias']:+.4f}  "
              f"experiment global_mean_bias={exp_stats['global_mean_bias']:+.4f}  "
              f"(polar: ctrl={ctrl_stats['polar_mean_bias']:+.4f} exp={exp_stats['polar_mean_bias']:+.4f})")

        # Write the CSV after every variable (not just once at the end of the
        # experiment) so the numeric results survive even if the plotting
        # step below crashes.
        df = pd.DataFrame([r for r in summary_rows if r["experiment"] == exp_name])
        out_csv = out_dir / f"spatial_bias_{exp_name}.csv"
        df.to_csv(out_csv, index=False)

        diff_map = exp_bias - ctrl_bias
        vmax = np.nanpercentile(np.abs(diff_map), 98)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(PRITHVI_LONS, PRITHVI_LATS, diff_map, transform=ccrs.PlateCarree(),
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{exp_name}  {label}  bias(experiment) - bias(control)\n"
                      f"post-injection mean (day 30-60)", fontsize=10)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label=label)
        fig.savefig(out_dir / f"spatial_bias_map_{exp_name}_{label}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        diff_map_day60 = exp_bias_day60 - ctrl_bias_day60
        vmax60 = np.nanpercentile(np.abs(diff_map_day60), 98)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(PRITHVI_LONS, PRITHVI_LATS, diff_map_day60, transform=ccrs.PlateCarree(),
                            cmap="RdBu_r", vmin=-vmax60, vmax=vmax60, shading="auto")
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{exp_name}  {label}  bias(experiment) - bias(control)\n"
                      f"day 60 only", fontsize=10)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label=label)
        fig.savefig(out_dir / f"spatial_bias_map_{exp_name}_{label}_day60.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {out_dir / f'spatial_bias_{exp_name}.csv'}")
    exp_ds.close()

ctrl_ds.close()
print("\nDone.")
