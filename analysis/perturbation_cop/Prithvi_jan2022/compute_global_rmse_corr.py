#!/usr/bin/env python3
"""
Global (cosine-lat-weighted) RMSE vs MERRA-2 for the Prithvi WxC Correlated
OSSE Perturbation experiments (perturbed_corr_temp_ins, perturbed_corr_uv_ins).

Reuses the exact same area-weighting/lookup methodology as
prithviOSSE/analyze_osse.py (global_rmse(), merra2_tidx(),
load_merra2_sfc/pres()) applied to the two new global-insertion experiments.

IMPORTANT: this script uses the CORRECT, already-fixed control forecast per
Organized_results/file_paths.md ("prithviOSSE/run2/output/
control_forecast_Jan2022.nc", 360-lat grid) — NOT the stale
"prithvi/output/forecast_60day_Jan2022.nc" (361-lat, pre-OSSE stand-in) that
prithviOSSE/analyze_osse.py itself still references; see file_paths.md
Section 1 "Two other files exist and are not the control" for the full
history of that bug.

Full variable/level grid (per user request): T, U, V, Z (as H, geopotential
height), Q (as QV, specific humidity) at the native levels nearest 300/500/
1000 hPa (Prithvi has no exact 1000 hPa level -- nearest is lev 72, ~961
hPa), plus surface T2M/U10M/V10M -- computed for BOTH experiments regardless
of which family was directly inserted, so this also reports cross-variable
coupling.

Output: one CSV per experiment, columns:
    lead_hour, valid_time, control_rmse_<VAR>, experiment_rmse_<VAR>, ...
written to this directory (PERTURBATION_EXPERIMENTS/Prithvi/<experiment>/).

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_global_rmse_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
import time
from pathlib import Path
from datetime import datetime, timedelta

import netCDF4 as nc4
import h5py
import numpy as np
import pandas as pd

BASE       = Path("/umbc/rs/pi_halem/users/yashd1")
PRITHVI    = BASE / "prithviOSSE"
MERRA2_DIR = BASE / "prithvi" / "data" / "merra2"
CTRL_FILE  = PRITHVI / "run2" / "output" / "control_forecast_Jan2022.nc"   # CORRECT control, see docstring. Root cause of the "OSError: NetCDF: HDF error" seen throughout SESSION_LOG_2026-07-22.md 10.2 was import order: h5py imported before netCDF4 makes netCDF4.Dataset() fail on these files -- fixed by importing netCDF4 first (see imports above).
OUT_DIR    = BASE / "PERTURBATION_EXPERIMENTS" / "Prithvi"

EXPERIMENTS = [
    ("perturbed_corr_temp_ins", PRITHVI / "perturbed_corr_temp_ins" / "output" / "perturbed_corr_temp_osse_forecast_Jan2022.nc", "temp"),
    ("perturbed_corr_uv_ins",   PRITHVI / "perturbed_corr_uv_ins"   / "output" / "perturbed_corr_uv_osse_forecast_Jan2022.nc",   "uv"),
]

IC_T1   = datetime(2022, 1, 1, 6, 0)
DT_H    = 6
N_STEPS = 240

LEV_VALS = [34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72]
LEVEL_APPROX_HPA = [52, 118, 164, 226, 265, 310, 424, 532, 604, 693, 832, 903, 946, 961]
PLOT_LEV = 51    # ~532 hPa, closest native level to 500 hPa (same choice as analyze_osse.py)
PLOT_LEV_IDX = LEV_VALS.index(PLOT_LEV)

# Nearest native level to each target hPa (Prithvi has no exact 300/500/1000hPa level)
TARGET_HPA = [300, 500, 1000]
LEVEL_IDX_FOR_HPA = {}
for target in TARGET_HPA:
    nearest_i = min(range(len(LEVEL_APPROX_HPA)), key=lambda i: abs(LEVEL_APPROX_HPA[i] - target))
    LEVEL_IDX_FOR_HPA[target] = nearest_i
    print(f"  target {target}hPa -> nearest native lev {LEV_VALS[nearest_i]} (~{LEVEL_APPROX_HPA[nearest_i]}hPa)")

UPPER_VARS = {"T": "T", "U": "U", "V": "V", "Z": "H", "Q": "QV"}  # label -> Prithvi/MERRA2 var name (identical both sides)
SFC_VARS = [("T2M", "T2M"), ("U10M", "U10M"), ("V10M", "V10M")]  # label, var name (identical both sides)

PRITHVI_LATS = np.linspace(-89.5, 90.0, 360)
COS_W = np.cos(np.deg2rad(PRITHVI_LATS))[:, np.newaxis]   # (360,1)


def global_rmse(fc, truth, w2d):
    w_total = float(w2d.sum()) * fc.shape[1]
    return float(np.sqrt(np.sum(w2d * (fc - truth) ** 2) / w_total))


def merra2_tidx(valid_dt):
    return valid_dt.hour // 3


def load_merra2_sfc(date_str, varname, t_idx):
    fpath = MERRA2_DIR / f"MERRA2_sfc_{date_str}.nc"
    if not fpath.exists():
        return None
    with h5py.File(str(fpath), "r") as f:
        if varname not in f:
            return None
        arr = f[varname][t_idx, 1:, :]   # skip -90 row: 361 -> 360
    return arr.astype(np.float32)


def load_merra2_pres(date_str, varname, lev_idx, t_idx):
    fpath = MERRA2_DIR / f"MERRA_pres_{date_str}.nc"
    if not fpath.exists():
        return None
    with h5py.File(str(fpath), "r") as f:
        if varname not in f:
            return None
        merra_lev = list(f["lev"][()])   # descending, e.g. [72,71,...,34]
        lv = LEV_VALS[lev_idx]
        if int(lv) not in merra_lev:
            return None
        m_idx = merra_lev.index(int(lv))
        arr = f[varname][t_idx, m_idx, 1:, :]
    return arr.astype(np.float32)


def load_nc_var(ds, varname, n_steps, lev_idx=None):
    """Load a variable's full time series (and optional single level) from an
    already-open experiment/control NetCDF handle, 360-lat grid, no cropping needed."""
    if lev_idx is None:
        return ds.variables[varname][:n_steps].astype(np.float32)
    return ds.variables[varname][:n_steps, lev_idx].astype(np.float32)


print("Verifying control file grid ...")
# Opened once and kept open for the whole script -- repeatedly open/closing
# the same multi-GB file per-variable was found to segfault on this NFS
# filesystem in the equivalent FCN analysis scripts; same fix applied here
# proactively. Retries with an actual delay: this file has failed to open
# with a transient "NetCDF: HDF error" on multiple prior job attempts, and
# 3 back-to-back retries with no sleep between them never cleared it --
# giving the NFS client time between attempts is the actual fix.
ctrl_ds = None
for _attempt in range(10):
    try:
        ctrl_ds = nc4.Dataset(str(CTRL_FILE))
        assert ctrl_ds.dimensions["lat"].size == 360, (
            f"CTRL_FILE has {ctrl_ds.dimensions['lat'].size} lats, expected 360 — "
            f"wrong control file (see docstring re: the stale 361-lat file)."
        )
        break
    except OSError as e:
        print(f"  transient open error ({e!r}), retrying in 30s ...")
        ctrl_ds = None
        time.sleep(30)
else:
    raise RuntimeError(f"Could not open {CTRL_FILE} after 10 attempts.")
print(f"  CTRL_FILE OK: {CTRL_FILE} (lat=360, time units={ctrl_ds.variables['time'].units})")

for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    valid_times = [IC_T1 + timedelta(hours=(s + 1) * DT_H) for s in range(N_STEPS)]
    lead_hours = [(vt - IC_T1).total_seconds() / 3600 for vt in valid_times]
    exp_ds = nc4.Dataset(str(nc_file))

    # Every variable/level is evaluated for every experiment (cross-variable
    # coupling included), not just the directly-inserted family.
    var_defs = [(f"{label}{hpa}", varname, LEVEL_IDX_FOR_HPA[hpa])
                for label, varname in UPPER_VARS.items() for hpa in TARGET_HPA]
    var_defs += [(label, varname, None) for label, varname in SFC_VARS]

    result = {"lead_hour": lead_hours, "valid_time": [str(v) for v in valid_times]}
    for label, varname, lev_idx in var_defs:
        try:
            fc_data = load_nc_var(exp_ds, varname, N_STEPS, lev_idx)
            ctrl_data = load_nc_var(ctrl_ds, varname, N_STEPS, lev_idx)
        except (KeyError, IndexError) as e:
            print(f"  [{exp_name}] {label}: variable '{varname}' unavailable ({e}), skipping.")
            continue

        exp_rmse = np.full(N_STEPS, np.nan, dtype=np.float64)
        ctrl_rmse = np.full(N_STEPS, np.nan, dtype=np.float64)
        for s, vt in enumerate(valid_times):
            date_str = vt.strftime("%Y%m%d")
            t_idx = merra2_tidx(vt)
            if lev_idx is None:
                truth = load_merra2_sfc(date_str, varname, t_idx)
            else:
                truth = load_merra2_pres(date_str, varname, lev_idx, t_idx)
            if truth is None:
                continue
            exp_rmse[s] = global_rmse(fc_data[s], truth, COS_W)
            ctrl_rmse[s] = global_rmse(ctrl_data[s], truth, COS_W)
        result[f"control_rmse_{label}"] = ctrl_rmse
        result[f"experiment_rmse_{label}"] = exp_rmse
        print(f"  [{exp_name}] {label}: mean control_rmse={np.nanmean(ctrl_rmse):.4f}  "
              f"mean experiment_rmse={np.nanmean(exp_rmse):.4f}")

        # Write after every variable, not just once at the end, so results
        # already computed survive even if a later variable/step crashes.
        out_csv = out_dir / f"global_rmse_{exp_name}.csv"
        pd.DataFrame(result).to_csv(out_csv, index=False)

    print(f"  Saved: {out_dir / f'global_rmse_{exp_name}.csv'}")
    exp_ds.close()

ctrl_ds.close()
print("\nDone.")
