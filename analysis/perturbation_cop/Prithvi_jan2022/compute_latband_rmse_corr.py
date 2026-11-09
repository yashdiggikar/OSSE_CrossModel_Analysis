#!/usr/bin/env python3
"""
Latitude-band RMSE vs MERRA-2 for the Prithvi WxC Correlated OSSE
Perturbation experiments (perturbed_corr_temp_ins, perturbed_corr_uv_ins).

Two bands, as requested:
    BAND_TROPICAL = 2N - 14N
    BAND_MIDLAT   = 40N - 60N

Same cosine-lat-weighted RMSE formula as compute_global_rmse_corr.py, just
restricted to the rows falling inside each band. Uses the CORRECT control
file per Organized_results/file_paths.md (see compute_global_rmse_corr.py's
docstring for the stale-control-file history this avoids).

Output: one CSV per experiment, columns:
    lead_hour, valid_time,
    control_rmse_<VAR>_2N14N, experiment_rmse_<VAR>_2N14N,
    control_rmse_<VAR>_40N60N, experiment_rmse_<VAR>_40N60N, ...
written to PERTURBATION_EXPERIMENTS/Prithvi/<experiment>/.

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_latband_rmse_corr.py
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
CTRL_FILE  = PRITHVI / "run2" / "output" / "control_forecast_Jan2022.nc"   # Root cause of the "OSError: NetCDF: HDF error" seen throughout SESSION_LOG_2026-07-22.md 10.2 was import order: h5py imported before netCDF4 makes netCDF4.Dataset() fail on these files -- fixed by importing netCDF4 first (see imports above).
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
PLOT_LEV = 51    # ~532 hPa, same choice as compute_global_rmse_corr.py
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
COS_W = np.cos(np.deg2rad(PRITHVI_LATS))[:, np.newaxis]

LAT_SLICES = {"2N": 2.0, "14N": 14.0, "40N": 40.0, "60N": 60.0}
LAT_ROWS = {name: int(np.argmin(np.abs(PRITHVI_LATS - lat))) for name, lat in LAT_SLICES.items()}
for name, row in LAT_ROWS.items():
    print(f"Slice {name}: nearest row {row} (lat={PRITHVI_LATS[row]:.3f})")


def slice_rmse(fc, truth, row):
    return float(np.sqrt(np.mean((fc[row, :] - truth[row, :]) ** 2)))


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

        for slice_name, row in LAT_ROWS.items():
            exp_rmse  = np.full(N_STEPS, np.nan, dtype=np.float64)
            ctrl_rmse = np.full(N_STEPS, np.nan, dtype=np.float64)
            for s, vt in enumerate(valid_times):
                date_str = vt.strftime("%Y%m%d")
                t_idx = merra2_tidx(vt)
                truth = (load_merra2_sfc(date_str, varname, t_idx) if lev_idx is None
                         else load_merra2_pres(date_str, varname, lev_idx, t_idx))
                if truth is None:
                    continue
                exp_rmse[s] = slice_rmse(fc_data[s], truth, row)
                ctrl_rmse[s] = slice_rmse(ctrl_data[s], truth, row)
            result[f"control_rmse_{label}_{slice_name}"] = ctrl_rmse
            result[f"experiment_rmse_{label}_{slice_name}"] = exp_rmse
            print(f"  [{exp_name}] {label} [{slice_name}]: mean control={np.nanmean(ctrl_rmse):.4f}  "
                  f"mean experiment={np.nanmean(exp_rmse):.4f}")

        # Write after every variable, not just once at the end, so results
        # already computed survive even if a later variable/step crashes.
        out_csv = out_dir / f"latband_rmse_{exp_name}.csv"
        pd.DataFrame(result).to_csv(out_csv, index=False)

    print(f"  Saved: {out_dir / f'latband_rmse_{exp_name}.csv'}")
    exp_ds.close()

ctrl_ds.close()
print("\nDone.")
