#!/usr/bin/env python3
"""
Global (cosine-lat-weighted, no lat breakdown) RMSE of Temperature (T) across
ALL pressure levels, at 3 snapshot days (Day 30 = first injected step, Day 45,
Day 60), for the lower-3-level T-insertion experiment, across Aurora, FCNv2,
and Prithvi WxC.

One journal-quality figure per model: 3 panels (Day 30 / 45 / 60), each panel
a 2-column (Control, Experiment) x N-level heatmap of global RMSE (K).
"""

import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.font_manager as fm
import netCDF4 as nc4
import h5py
import xarray as xr

# ─── PATHS ─────────────────────────────────────────────────────────────────
BASE = Path("/umbc/rs/pi_halem/users/yashd1")
ERA5_DIR = BASE / "AURORA_DATA" / "rmse_data"
MERRA2_DIR = BASE / "prithvi" / "data" / "merra2"

AURORA_CTRL = BASE / "auroraoutput_1year_1stfeb23.nc"
AURORA_EXP  = BASE / "CC_aurora" / "T_LOWER" / "output" / "aurora_t_lower.nc"
AURORA_OUT  = BASE / "CC_aurora" / "T_LOWER"

FCN_CTRL = BASE / "FourCastNet" / "forecast_normal_60days_12hr.nc"
FCN_EXP  = BASE / "fcn_inference" / "t_lower" / "forecast_t_lower3_global_day30_60.nc"
FCN_OUT  = BASE / "fcn_inference" / "t_lower"

PRITHVI_CTRL = BASE / "prithviOSSE" / "run2" / "output" / "control_forecast_Jan2022.nc"
PRITHVI_EXP  = BASE / "prithviOSSE" / "lower_temp_ins" / "output" / "lower_temp_osse_forecast_Jan2022.nc"
PRITHVI_OUT  = BASE / "prithviOSSE" / "lower_temp_ins"

AURORA_IC  = datetime(2023, 2, 1, 12, 0)
FCN_IC     = datetime(2023, 2, 1, 0, 0)
PRITHVI_IC = datetime(2022, 1, 1, 6, 0)

# 0-based array index, TRUE valid lead-hours (verified empirically per model — each
# model's saved file has a DIFFERENT step<->time convention, see investigation notes):
#   Aurora:  array_index i  <-> true valid = IC + i*6h.        Injection starts at i=121.
#   FCN:     array_index i  <-> true valid = IC + i*6h.        Injection starts at i=120.
#   Prithvi: array_index i  <-> true valid = IC + (i+1)*6h.    Injection starts at i=120.
# "Day 30" panel = each model's actual FIRST INJECTED step (not necessarily exactly day 30.0).
DAY_STEPS_AURORA  = [("Day 30\n(1st Injected Step)", 121, 726),
                     ("Day 45", 180, 1080),
                     ("Day 60", 239, 1434)]
DAY_STEPS_FCN     = [("Day 30\n(1st Injected Step)", 120, 720),
                     ("Day 45", 180, 1080),
                     ("Day 60", 239, 1434)]
DAY_STEPS_PRITHVI = [("Day 30\n(1st Injected Step)", 120, 726),
                     ("Day 45", 179, 1080),
                     ("Day 60", 239, 1440)]

AF_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PRI_LEV_VALS = [34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72]
PRI_LEV_HPA  = [52, 118, 164, 226, 265, 310, 424, 532, 604, 693, 832, 903, 946, 960]
MERRA_LEV_DESC = [72, 71, 68, 63, 56, 53, 51, 48, 45, 44, 43, 41, 39, 34]

FCN_T_CH = {  # channel index of t{level} in the 73-channel FCN prediction array
    50: 47, 100: 48, 150: 49, 200: 50, 250: 51, 300: 52, 400: 53,
    500: 54, 600: 55, 700: 56, 850: 57, 925: 58, 1000: 59,
}

INSERTED_HPA = {"Aurora": [1000, 925, 850], "FCN": [1000, 925, 850], "Prithvi": [960, 946, 903]}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def global_rmse(fc, truth, lat_1d):
    cw = np.cos(np.deg2rad(lat_1d))[:, None]
    diff2 = (fc - truth) ** 2
    return float(np.sqrt(np.sum(cw * diff2) / (diff2.shape[1] * np.sum(cw))))

def era5_idx_nearest(era5_times, valid_dt):
    t64 = np.datetime64(valid_dt.strftime("%Y-%m-%dT%H:%M"), "m")
    diff = np.abs(era5_times.astype("datetime64[m]") - t64)
    return int(np.argmin(diff))

def load_era5_pressure_level_month(month_str, var_shortname, level_hpa):
    fp = ERA5_DIR / f"era5_pressure_{month_str}.grib"
    ds = xr.open_dataset(fp, engine="cfgrib",
                          backend_kwargs={"filter_by_keys": {
                              "shortName": var_shortname, "typeOfLevel": "isobaricInhPa",
                              "level": int(level_hpa)}})
    var = list(ds.data_vars)[0]
    times = ds.time.values
    data = ds[var].values
    lat = ds.latitude.values
    lon = ds.longitude.values
    ds.close()
    return times, data, lat, lon

def month_str_for(dt):
    return f"2023_{dt.month:02d}"

def era5_t_at(level_hpa, valid_dt):
    """Nearest ERA5 T snapshot (lat,lon) + lat array for a given level & datetime."""
    m = month_str_for(valid_dt)
    times, data, lat, lon = load_era5_pressure_level_month(m, "t", level_hpa)
    i = era5_idx_nearest(times, valid_dt)
    return data[i], lat, lon

def merra2_pres_at(level_val, valid_dt):
    fp = MERRA2_DIR / f"MERRA_pres_{valid_dt.strftime('%Y%m%d')}.nc"
    ds = nc4.Dataset(str(fp))
    ti = valid_dt.hour // 3
    lev_list = [int(v) for v in ds.variables["lev"][:]]
    li = lev_list.index(int(level_val))
    val = np.array(ds.variables["T"][ti, li, :, :])
    ds.close()
    return val

# ─── AURORA ──────────────────────────────────────────────────────────────────

def compute_aurora():
    print("=== AURORA ===")
    ds_exp = nc4.Dataset(str(AURORA_EXP))
    aurora_lat = ds_exp["lat"][:]
    aurora_lon = ds_exp["lon"][:]
    n_lev = len(AF_LEVELS)
    n_day = len(DAY_STEPS_AURORA)
    exp_rmse = np.zeros((n_lev, n_day))
    ctrl_rmse = np.zeros((n_lev, n_day))

    ds_ctrl = nc4.Dataset(str(AURORA_CTRL))
    ctrl_lt = ds_ctrl["lead_time"][:]  # 0,12,24,...
    ctrl_valid = np.array([AURORA_IC + timedelta(hours=int(h)) for h in ctrl_lt])

    for di, (day_label, step_idx, lead_h) in enumerate(DAY_STEPS_AURORA):
        valid_dt = AURORA_IC + timedelta(hours=lead_h)
        # nearest control index (12h cadence)
        cidx = int(np.argmin(np.abs(np.array([(v - valid_dt).total_seconds() for v in ctrl_valid]))))
        print(f"  {day_label.strip()}: valid={valid_dt}  ctrl_idx={cidx} (ctrl valid={ctrl_valid[cidx]})")
        for li, lev in enumerate(AF_LEVELS):
            print(f"    level {lev} hPa ...", end="\r")
            truth, era5_lat, era5_lon = era5_t_at(lev, valid_dt)
            # ERA5 lat: 721 rows north-first; Aurora: 720 rows, drop south-pole row
            if era5_lat[0] > era5_lat[-1]:
                truth_a = truth[:-1, :]
            else:
                truth_a = truth[1:, :]
            var_key = f"t{lev}"
            fc_exp = np.array(ds_exp[var_key][step_idx])
            fc_ctrl = np.array(ds_ctrl[var_key][cidx])
            exp_rmse[li, di] = global_rmse(fc_exp, truth_a, aurora_lat)
            ctrl_rmse[li, di] = global_rmse(fc_ctrl, truth_a, aurora_lat)
            del truth, truth_a
        print()
    ds_exp.close(); ds_ctrl.close()
    gc.collect()
    return exp_rmse, ctrl_rmse

# ─── FCN ─────────────────────────────────────────────────────────────────────

def compute_fcn():
    print("=== FCN ===")
    ds_exp = nc4.Dataset(str(FCN_EXP))
    n_lev = len(AF_LEVELS)
    n_day = len(DAY_STEPS_FCN)
    exp_rmse = np.zeros((n_lev, n_day))
    ctrl_rmse = np.zeros((n_lev, n_day))

    ds_ctrl = nc4.Dataset(str(FCN_CTRL))
    ctrl_t_h = ds_ctrl["time"][:]  # raw units: "hours since 2023-02-01 00:00:00" (verified)
    ctrl_valid = np.array([FCN_IC + timedelta(hours=float(h)) for h in ctrl_t_h])

    fcn_lat = None
    for di, (day_label, step_idx, lead_h) in enumerate(DAY_STEPS_FCN):
        valid_dt = FCN_IC + timedelta(hours=lead_h)
        cidx = int(np.argmin(np.abs(np.array([(v - valid_dt).total_seconds() for v in ctrl_valid]))))
        print(f"  {day_label.strip()}: valid={valid_dt}  ctrl_idx={cidx} (ctrl valid={ctrl_valid[cidx]})")
        for li, lev in enumerate(AF_LEVELS):
            print(f"    level {lev} hPa ...", end="\r")
            truth, era5_lat, era5_lon = era5_t_at(lev, valid_dt)
            if fcn_lat is None:
                fcn_lat = era5_lat  # FCN grid = 721 lat, same as ERA5, no drop needed
            ch = FCN_T_CH[lev]
            fc_exp = np.array(ds_exp["prediction"][step_idx, ch, :, :])
            fc_ctrl = np.array(ds_ctrl["prediction"][cidx, ch, :, :])
            exp_rmse[li, di] = global_rmse(fc_exp, truth, era5_lat)
            ctrl_rmse[li, di] = global_rmse(fc_ctrl, truth, era5_lat)
            del truth
        print()
    ds_exp.close(); ds_ctrl.close()
    gc.collect()
    return exp_rmse, ctrl_rmse

# ─── PRITHVI ─────────────────────────────────────────────────────────────────

def compute_prithvi():
    print("=== PRITHVI ===")
    f_exp = h5py.File(str(PRITHVI_EXP), "r")
    pri_lat = np.array(f_exp["lat"])  # 360, -89.5..90
    f_ctrl = h5py.File(str(PRITHVI_CTRL), "r")  # 361 lat incl -90 -> skip first row

    n_lev = len(PRI_LEV_VALS)
    n_day = len(DAY_STEPS_PRITHVI)
    exp_rmse = np.zeros((n_lev, n_day))
    ctrl_rmse = np.zeros((n_lev, n_day))

    for di, (day_label, step_idx, lead_h) in enumerate(DAY_STEPS_PRITHVI):
        valid_dt = PRITHVI_IC + timedelta(hours=lead_h)
        print(f"  {day_label.strip()}: valid={valid_dt}  ctrl uses same step_idx={step_idx}")
        for li, lev_val in enumerate(PRI_LEV_VALS):
            print(f"    lev {lev_val} (~{PRI_LEV_HPA[li]} hPa) ...", end="\r")
            truth = merra2_pres_at(lev_val, valid_dt)  # (361,576)
            truth_p = truth[1:, :]  # skip -90 row -> (360,576)
            fc_exp = np.array(f_exp["T"][step_idx, li, :, :])
            fc_ctrl = np.array(f_ctrl["T"][step_idx, li, :, :])  # ctrl already 360 lat, matches exp grid
            exp_rmse[li, di] = global_rmse(fc_exp, truth_p, pri_lat)
            ctrl_rmse[li, di] = global_rmse(fc_ctrl, truth_p, pri_lat)
            del truth, truth_p
        print()
    f_exp.close(); f_ctrl.close()
    gc.collect()
    return exp_rmse, ctrl_rmse

# ─── PLOTTING ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

def plot_model_figure(model_name, levels_hpa, exp_rmse, ctrl_rmse, out_paths, day_steps):
    """exp_rmse/ctrl_rmse: (n_lev, n_day). One figure, 3 panels (one per day)."""
    n_lev, n_day = exp_rmse.shape
    vmax = max(exp_rmse.max(), ctrl_rmse.max())
    vmin = 0.0
    cmap = plt.cm.get_cmap("inferno_r")
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, n_day, figsize=(4.6 * n_day + 1.0, 0.42 * n_lev + 3.2),
                              constrained_layout=True)
    fig.suptitle(f"{model_name} — Global RMSE of Temperature Across All Pressure Levels\n"
                 f"Lower-3-Level Insertion Experiment (Control vs. Experiment)",
                 fontsize=16, fontweight="bold")

    inserted = INSERTED_HPA[model_name]
    cols = ["Control\n(Free-Running)", "Experiment\n(Insertion)"]

    for di, (day_label, _step_idx, lead_h) in enumerate(day_steps):
        ax = axes[di]
        mat = np.stack([ctrl_rmse[:, di], exp_rmse[:, di]], axis=1)  # (n_lev, 2)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(cols, fontsize=12, fontweight="bold")
        ax.set_yticks(range(n_lev))
        ytick_labels = [f"{lv} hPa" + ("  ●" if lv in inserted else "") for lv in levels_hpa]
        ax.set_yticklabels(ytick_labels, fontsize=11.5)
        ax.set_title(f"{day_label}\n(t = {lead_h/24:.2f} d)", fontsize=15, fontweight="bold", pad=10)

        for r in range(n_lev):
            for c in range(2):
                val = mat[r, c]
                txt_color = "white" if norm(val) > 0.55 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                         fontsize=10.5, color=txt_color, fontweight="bold")

        for r in range(n_lev):
            if levels_hpa[r] in inserted:
                ax.add_patch(plt.Rectangle((-0.5, r - 0.5), 2, 1, fill=False,
                                            edgecolor="cyan", lw=2.4))

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(n_lev - 0.5, -0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                       location="right", shrink=0.92, pad=0.015)
    cb.set_label("Temperature RMSE (K)", fontsize=13, fontweight="bold")

    fig.text(0.5, -0.02,
              "Cyan box = directly-inserted lower levels   |   Global, cosine-latitude-weighted RMSE vs. truth",
              ha="center", fontsize=10.5, style="italic")

    for p in out_paths:
        fig.savefig(p, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"  Saved: {out_paths}")

def save_csv(model_name, levels_hpa, exp_rmse, ctrl_rmse, out_dir, day_steps):
    day_labels = [d[0].split("\n")[0] for d in day_steps]
    lead_hs    = [d[2] for d in day_steps]
    rows = []
    for li, lev in enumerate(levels_hpa):
        for di, dl in enumerate(day_labels):
            rows.append({"level_hpa": lev, "day": dl, "true_lead_hours": lead_hs[di],
                         "control_rmse_K": ctrl_rmse[li, di],
                         "experiment_rmse_K": exp_rmse[li, di]})
    df = pd.DataFrame(rows)
    fp = out_dir / f"global_rmse_alllevels_3day_{model_name.lower()}.csv"
    df.to_csv(fp, index=False)
    print(f"  Saved CSV: {fp}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print(f"Start: {datetime.now()}\n")

    exp_a, ctrl_a = compute_aurora()
    save_csv("Aurora", AF_LEVELS, exp_a, ctrl_a, AURORA_OUT, DAY_STEPS_AURORA)
    plot_model_figure("Aurora", AF_LEVELS, exp_a, ctrl_a,
                       [AURORA_OUT / "fig_global_rmse_alllevels_3day_aurora.png",
                        BASE / "fig_global_rmse_alllevels_3day_aurora.png"], DAY_STEPS_AURORA)

    exp_f, ctrl_f = compute_fcn()
    save_csv("FCN", AF_LEVELS, exp_f, ctrl_f, FCN_OUT, DAY_STEPS_FCN)
    plot_model_figure("FCN", AF_LEVELS, exp_f, ctrl_f,
                       [FCN_OUT / "fig_global_rmse_alllevels_3day_fcn.png",
                        BASE / "fig_global_rmse_alllevels_3day_fcn.png"], DAY_STEPS_FCN)

    exp_p, ctrl_p = compute_prithvi()
    save_csv("Prithvi", PRI_LEV_HPA, exp_p, ctrl_p, PRITHVI_OUT, DAY_STEPS_PRITHVI)
    plot_model_figure("Prithvi", PRI_LEV_HPA, exp_p, ctrl_p,
                       [PRITHVI_OUT / "fig_global_rmse_alllevels_3day_prithvi.png",
                        BASE / "fig_global_rmse_alllevels_3day_prithvi.png"], DAY_STEPS_PRITHVI)

    print(f"\nDone: {datetime.now()}")

if __name__ == "__main__":
    main()
