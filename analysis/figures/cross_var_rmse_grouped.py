#!/usr/bin/env python3
"""
Cross-variable RMSE improvement, regional (land/ocean/coastal) insertion experiments.

Figure 1 (per model): T-insertion (T_LAND/T_OCEAN/T_COASTAL) -> effect on U500, V500, Z500
Figure 2 (per model): UV-insertion (UV_LAND/UV_OCEAN/UV_COASTAL) -> effect on T500, Z500

% RMSE improvement = (ctrl_rmse - exp_rmse) / ctrl_rmse * 100, last-20-day average (days 41-60),
global cosine-latitude-weighted RMSE (MSE averaged over time+space jointly, then sqrt).
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
import netCDF4 as nc4
import h5py
import xarray as xr

BASE = Path("/umbc/rs/pi_halem/users/yashd1")
ERA5_DIR = BASE / "AURORA_DATA" / "rmse_data"
MERRA2_DIR = BASE / "prithvi" / "data" / "merra2"
OUT_DIR = BASE / "final_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MMSTUDY = BASE / "Multimodel-study"

AURORA_IC  = datetime(2023, 2, 1, 12, 0)
FCN_IC     = datetime(2023, 2, 1, 0, 0)
PRITHVI_IC = datetime(2022, 1, 1, 6, 0)

AURORA_CTRL  = BASE / "auroraoutput_1year_1stfeb23.nc"
FCN_CTRL     = BASE / "FourCastNet" / "forecast_normal_60days_12hr.nc"
PRITHVI_CTRL = BASE / "prithviOSSE" / "run2" / "output" / "control_forecast_Jan2022.nc"

AURORA_T_FILES = {
    "land":    BASE / "CC_aurora/T_LAND/output/aurora_t_land.nc",
    "ocean":   BASE / "CC_aurora/T_OCEAN/output/aurora_t_ocean.nc",
    "coastal": BASE / "CC_aurora/T_COASTAL/output/aurora_t_coastal.nc",
}
AURORA_UV_FILES = {
    "land":    BASE / "CC_aurora/UV_LAND/output/aurora_uv_land.nc",
    "ocean":   BASE / "CC_aurora/UV_OCEAN/output/aurora_uv_ocean.nc",
    "coastal": BASE / "CC_aurora/UV_COASTAL/output/aurora_uv_coastal.nc",
}
FCN_T_FILES = {
    "land":    BASE / "fcn_inference/forecast_t_landonly_day30_60.nc",
    "ocean":   BASE / "fcn_inference/t_ocean/forecast_t_ocean_day30_60.nc",
    "coastal": BASE / "fcn_inference/t_coastal/forecast_t_coastal_day30_60.nc",
}
FCN_UV_FILES = {
    "land":    BASE / "fcn_inference/uv_land/forecast_uv_landonly_day30_60.nc",
    "ocean":   BASE / "fcn_inference/uv_ocean/forecast_uv_ocean_day30_60.nc",
    "coastal": BASE / "fcn_inference/uv_coastal/forecast_uv_coastal_day30_60.nc",
}
PRITHVI_T_FILES = {
    "land":    BASE / "prithviOSSE/land_temp_ins/output/land_temp_osse_forecast_Jan2022.nc",
    "ocean":   BASE / "prithviOSSE/ocean_temp_ins/output/ocean_temp_osse_forecast_Jan2022.nc",
    "coastal": BASE / "prithviOSSE/coastal_temp_ins/output/coastal_temp_osse_forecast_Jan2022.nc",
}
PRITHVI_UV_FILES = {
    "land":    BASE / "prithviOSSE/land_uv_ins/output/land_uv_osse_forecast_Jan2022.nc",
    "ocean":   BASE / "prithviOSSE/ocean_uv_ins/output/ocean_uv_osse_forecast_Jan2022.nc",
    "coastal": BASE / "prithviOSSE/coastal_uv_ins/output/coastal_uv_osse_forecast_Jan2022.nc",
}

REGIONS = ["land", "ocean", "coastal"]            # used by the regional compute functions
PLOT_BARS = ["land", "ocean", "coastal", "global"]  # used by the plotting function (adds Global)
REGION_LABELS = {"land": "Land", "ocean": "Ocean", "coastal": "Coastal", "global": "Global"}

FCN_CH = {"u500": 15, "v500": 28, "z500": 41, "t500": 54}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def global_mse_step(fc, truth, lat_1d):
    """Cosine-weighted mean-squared-error for one timestep (no sqrt)."""
    cw = np.cos(np.deg2rad(lat_1d))[:, None]
    return float(np.sum(cw * (fc - truth) ** 2) / (truth.shape[1] * np.sum(cw)))

def window_rmse(mse_list):
    return float(np.sqrt(np.mean(mse_list)))

def era5_idx_nearest(times, valid_dt):
    t64 = np.datetime64(valid_dt.strftime("%Y-%m-%dT%H:%M"), "m")
    diff = np.abs(times.astype("datetime64[m]") - t64)
    return int(np.argmin(diff))

def load_era5_month_var(month_str, shortname, level_hpa):
    fp = ERA5_DIR / f"era5_pressure_{month_str}.grib"
    ds = xr.open_dataset(fp, engine="cfgrib",
                          backend_kwargs={"filter_by_keys": {
                              "shortName": shortname, "typeOfLevel": "isobaricInhPa",
                              "level": int(level_hpa)}})
    var = list(ds.data_vars)[0]
    times = ds.time.values
    data = ds[var].values
    lat = ds.latitude.values
    lon = ds.longitude.values
    ds.close()
    return times, data, lat, lon

class ERA5Cache:
    """Loads era5_pressure_2023_03 + _04 once per (var,level), caches by key."""
    def __init__(self):
        self.cache = {}

    def get(self, shortname, level_hpa):
        key = (shortname, level_hpa)
        if key not in self.cache:
            times_all, data_all = [], []
            lat = lon = None
            for m in ["2023_03", "2023_04"]:
                t, d, la, lo = load_era5_month_var(m, shortname, level_hpa)
                times_all.append(t); data_all.append(d)
                lat, lon = la, lo
            times = np.concatenate(times_all)
            data = np.concatenate(data_all, axis=0)
            order = np.argsort(times)
            self.cache[key] = (times[order], data[order], lat, lon)
        return self.cache[key]

def merra2_pres_field(varname, lev_val, valid_dt):
    fp = MERRA2_DIR / f"MERRA_pres_{valid_dt.strftime('%Y%m%d')}.nc"
    ds = nc4.Dataset(str(fp))
    ti = valid_dt.hour // 3
    lev_list = [int(v) for v in ds.variables["lev"][:]]
    li = lev_list.index(int(lev_val))
    val = np.array(ds.variables[varname][ti, li, :, :])
    ds.close()
    return val

# ─── GLOBAL-INSERTION BAR (from Multimodel-study) ────────────────────────────
# Same last-20-day (days 41-60) averaging window, simple mean of the precomputed
# per-step RMSE rows/columns (raw squared-error fields aren't available for these).

def _aurora_fcn_global_bar(csv_exp, csv_ctrl, varnames, is_aurora):
    df_exp = pd.read_csv(csv_exp, index_col=0)
    df_ctrl = pd.read_csv(csv_ctrl, index_col=0)
    cols = pd.to_datetime(df_exp.columns)
    first = cols[0]
    if is_aurora:
        lead_days = (cols - first).total_seconds() / 86400 + 0.5
    else:
        lead_days = (cols - first).total_seconds() / 86400
    mask = (lead_days >= 41) & (lead_days <= 60)
    out = {}
    for v in varnames:
        out[v] = (float(df_ctrl.loc[v].values[mask].mean()),
                   float(df_exp.loc[v].values[mask].mean()))
    return out

def _prithvi_global_bar(csv_exp, csv_ctrl, var_lev_map):
    df_exp = pd.read_csv(csv_exp)
    df_ctrl = pd.read_csv(csv_ctrl)
    lead_days = df_exp["lead_hour"].values / 24.0
    mask = (lead_days >= 41) & (lead_days <= 60)
    out = {}
    for v, col in var_lev_map.items():
        out[v] = (float(df_ctrl[col].values[mask].mean()),
                   float(df_exp[col].values[mask].mean()))
    return out

def compute_global_bars():
    """Returns dict[(model, exp_type, var)] -> (ctrl_rmse, exp_rmse) for the GLOBAL
    temp_ins/wind_ins experiments already computed in Multimodel-study."""
    out = {}

    aurora_t = _aurora_fcn_global_bar(
        MMSTUDY / "Aurora/temp_ins/aurora_rmse_all_vars.csv",
        MMSTUDY / "Aurora/control/aurora_rmse_all_vars.csv",
        ["u500", "v500", "z500"], is_aurora=True)
    aurora_uv = _aurora_fcn_global_bar(
        MMSTUDY / "Aurora/wind_ins/aurora_rmse_all_vars.csv",
        MMSTUDY / "Aurora/control/aurora_rmse_all_vars.csv",
        ["t500", "z500"], is_aurora=True)
    for v, vals in aurora_t.items(): out[("Aurora", "T", v)] = vals
    for v, vals in aurora_uv.items(): out[("Aurora", "UV", v)] = vals

    fcn_t = _aurora_fcn_global_bar(
        MMSTUDY / "FCNv2/temp_ins/rmse_all_vars.csv",
        MMSTUDY / "FCNv2/Control/rmse_all_vars.csv",
        ["u500", "v500", "z500"], is_aurora=False)
    fcn_uv = _aurora_fcn_global_bar(
        MMSTUDY / "FCNv2/wind_ins/rmse_all_vars.csv",
        MMSTUDY / "FCNv2/Control/rmse_all_vars.csv",
        ["t500", "z500"], is_aurora=False)
    for v, vals in fcn_t.items(): out[("FCN", "T", v)] = vals
    for v, vals in fcn_uv.items(): out[("FCN", "UV", v)] = vals

    pri_t = _prithvi_global_bar(
        MMSTUDY / "PrithviWxC/temp_ins/rmse_upper.csv",
        MMSTUDY / "PrithviWxC/control/rmse_upper.csv",
        {"u500": "U_lev51", "v500": "V_lev51", "z500": "H_lev51"})
    pri_uv = _prithvi_global_bar(
        MMSTUDY / "PrithviWxC/wind_ins/rmse_upper.csv",
        MMSTUDY / "PrithviWxC/control/rmse_upper.csv",
        {"t500": "T_lev51", "z500": "H_lev51"})
    for v, vals in pri_t.items(): out[("Prithvi", "T", v)] = vals
    for v, vals in pri_uv.items(): out[("Prithvi", "UV", v)] = vals

    return out

# ─── WINDOW DEFINITIONS (last-20-day avg, days 41-60), verified per model ────
# Aurora/FCN: array_index i <-> true valid = IC + i*6h
# Prithvi:    array_index i <-> true valid = IC + (i+1)*6h
AURORA_EXP_IDX  = list(range(164, 240))   # day41(984h, idx164) .. day~59.75(idx239)
FCN_EXP_IDX     = list(range(164, 240))
PRITHVI_EXP_IDX = list(range(163, 240))   # day41(984h via (163+1)*6) .. day60(idx239)
AURORA_CTRL_IDX  = list(range(82, 121))   # 12h cadence: 82*12=984h .. 120*12=1440h
FCN_CTRL_IDX     = list(range(82, 120))   # FCN control only has 120 steps (idx 0..119, max lead=1428h=59.5d)

# ─── MAIN AURORA COMPUTATION ──────────────────────────────────────────────────

def compute_aurora_cross(era5_cache):
    print("=== AURORA cross-variable RMSE ===")
    results = {}  # results[(exp_type, region, var)] = (ctrl_rmse, exp_rmse)

    ds_ctrl = nc4.Dataset(str(AURORA_CTRL))
    aurora_lat = ds_ctrl["lat"][:]  # control has same 720-lat grid

    # Pre-load ERA5 truth + control MSE list per variable (shared across regions)
    for exp_type, files, varlist in [("T", AURORA_T_FILES, ["u500", "v500", "z500"]),
                                       ("UV", AURORA_UV_FILES, ["t500", "z500"])]:
        for var in varlist:
            shortname = var[0]
            era5_times, era5_data, era5_lat, era5_lon = era5_cache.get(shortname, 500)
            era5_lat_trim = era5_data[:, :-1, :] if era5_lat[0] > era5_lat[-1] else era5_data[:, 1:, :]

            # control MSE list for this var (same across all regions for a given exp_type/var)
            ctrl_mse = []
            for ci in ctrl_idx_for_aurora():
                vt = AURORA_IC + timedelta(hours=int(ds_ctrl["lead_time"][ci]))
                ei = era5_idx_nearest(era5_times, vt)
                fc = np.array(ds_ctrl[var][ci])
                ctrl_mse.append(global_mse_step(fc, era5_lat_trim[ei], aurora_lat))
            ctrl_rmse = window_rmse(ctrl_mse)

            for region in REGIONS:
                ds_exp = nc4.Dataset(str(files[region]))
                exp_mse = []
                for idx in AURORA_EXP_IDX:
                    vt = AURORA_IC + timedelta(hours=idx * 6)
                    ei = era5_idx_nearest(era5_times, vt)
                    fc = np.array(ds_exp[var][idx])
                    exp_mse.append(global_mse_step(fc, era5_lat_trim[ei], aurora_lat))
                ds_exp.close()
                exp_rmse = window_rmse(exp_mse)
                results[(exp_type, region, var)] = (ctrl_rmse, exp_rmse)
                print(f"  {exp_type}-{region:8s} {var:5s}: ctrl={ctrl_rmse:.4f}  exp={exp_rmse:.4f}  "
                      f"imp={100*(ctrl_rmse-exp_rmse)/ctrl_rmse:.1f}%")
            del era5_lat_trim; gc.collect()
    ds_ctrl.close()
    return results

def ctrl_idx_for_aurora():
    return AURORA_CTRL_IDX

# ─── MAIN FCN COMPUTATION ─────────────────────────────────────────────────────

def compute_fcn_cross(era5_cache):
    print("=== FCN cross-variable RMSE ===")
    results = {}

    ds_ctrl = nc4.Dataset(str(FCN_CTRL))
    fcn_lat = None

    for exp_type, files, varlist in [("T", FCN_T_FILES, ["u500", "v500", "z500"]),
                                       ("UV", FCN_UV_FILES, ["t500", "z500"])]:
        for var in varlist:
            shortname = var[0]
            ch = FCN_CH[var]
            era5_times, era5_data, era5_lat, era5_lon = era5_cache.get(shortname, 500)
            if fcn_lat is None:
                fcn_lat = era5_lat  # FCN grid = 721 lat, matches ERA5 exactly

            ctrl_mse = []
            for ci in FCN_CTRL_IDX:
                vt = FCN_IC + timedelta(hours=float(ds_ctrl["time"][ci]))
                ei = era5_idx_nearest(era5_times, vt)
                fc = np.array(ds_ctrl["prediction"][ci, ch, :, :])
                ctrl_mse.append(global_mse_step(fc, era5_data[ei], fcn_lat))
            ctrl_rmse = window_rmse(ctrl_mse)

            for region in REGIONS:
                ds_exp = nc4.Dataset(str(files[region]))
                exp_mse = []
                for idx in FCN_EXP_IDX:
                    vt = FCN_IC + timedelta(hours=idx * 6)
                    ei = era5_idx_nearest(era5_times, vt)
                    fc = np.array(ds_exp["prediction"][idx, ch, :, :])
                    exp_mse.append(global_mse_step(fc, era5_data[ei], fcn_lat))
                ds_exp.close()
                exp_rmse = window_rmse(exp_mse)
                results[(exp_type, region, var)] = (ctrl_rmse, exp_rmse)
                print(f"  {exp_type}-{region:8s} {var:5s}: ctrl={ctrl_rmse:.4f}  exp={exp_rmse:.4f}  "
                      f"imp={100*(ctrl_rmse-exp_rmse)/ctrl_rmse:.1f}%")
    ds_ctrl.close()
    return results

# ─── MAIN PRITHVI COMPUTATION ─────────────────────────────────────────────────

PRI_LI_500 = 7   # index of lev=51 (~532 hPa) within PRI_LEV_VALS

def compute_prithvi_cross():
    print("=== PRITHVI cross-variable RMSE ===")
    results = {}

    f_ctrl = h5py.File(str(PRITHVI_CTRL), "r")
    pri_lat = np.array(f_ctrl["lat"])  # 360, matches experiment grid exactly

    for exp_type, files, varlist in [("T", PRITHVI_T_FILES, ["U", "V", "H"]),
                                       ("UV", PRITHVI_UV_FILES, ["T", "H"])]:
        for var in varlist:
            ctrl_mse = []
            for idx in PRITHVI_EXP_IDX:
                vt = PRITHVI_IC + timedelta(hours=(idx + 1) * 6)
                truth = merra2_pres_field(var, 51, vt)[1:, :]  # skip -90 row -> (360,576)
                fc = np.array(f_ctrl[var][idx, PRI_LI_500, :, :])
                ctrl_mse.append(global_mse_step(fc, truth, pri_lat))
            ctrl_rmse = window_rmse(ctrl_mse)

            for region in REGIONS:
                f_exp = h5py.File(str(files[region]), "r")
                exp_mse = []
                for idx in PRITHVI_EXP_IDX:
                    vt = PRITHVI_IC + timedelta(hours=(idx + 1) * 6)
                    truth = merra2_pres_field(var, 51, vt)[1:, :]
                    fc = np.array(f_exp[var][idx, PRI_LI_500, :, :])
                    exp_mse.append(global_mse_step(fc, truth, pri_lat))
                f_exp.close()
                exp_rmse = window_rmse(exp_mse)
                varlabel = {"U": "u500", "V": "v500", "H": "z500", "T": "t500"}[var]
                results[(exp_type, region, varlabel)] = (ctrl_rmse, exp_rmse)
                print(f"  {exp_type}-{region:8s} {varlabel:5s}: ctrl={ctrl_rmse:.4f}  exp={exp_rmse:.4f}  "
                      f"imp={100*(ctrl_rmse-exp_rmse)/ctrl_rmse:.1f}%")
    f_ctrl.close()
    return results

# ─── PLOTTING ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

# Muted, colorblind-friendly palette (seaborn "deep"-style), softer than primary RGB
REGION_COLORS = {"land": "#4C72B0", "ocean": "#55A868", "coastal": "#C44E52", "global": "#8172B2"}
VAR_DISPLAY = {"u500": "U500", "v500": "V500", "z500": "Z500", "t500": "T500"}
EDGE_COLOR = "#333333"

def plot_grouped(model_name, exp_type, groups, results, global_bars, out_path, title_suffix):
    """groups: list of variable keys, each a group on the x-axis; bars = land/ocean/coastal/global."""
    n_groups = len(groups)
    n_bars = len(PLOT_BARS)
    group_gap = 1.35                       # extra spacing between groups (less "zoomed"/thick look)
    cluster_w = 0.62                       # total width used by the 4 bars within a group
    bar_w = cluster_w / n_bars
    x = np.arange(n_groups) * group_gap

    fig, ax = plt.subplots(figsize=(2.7 * n_groups + 3.2, 6.3), constrained_layout=True)

    all_vals = []
    bar_handles = []
    for bi, region in enumerate(PLOT_BARS):
        vals = []
        for g in groups:
            if region == "global":
                ctrl, exp = global_bars[g]
            else:
                ctrl, exp = results[(exp_type, region, g)]
            vals.append(100 * (ctrl - exp) / ctrl)
        all_vals.extend(vals)
        offs = (bi - (n_bars - 1) / 2) * bar_w
        bars = ax.bar(x + offs, vals, bar_w * 0.92, label=REGION_LABELS[region],
                      color=REGION_COLORS[region], edgecolor=EDGE_COLOR, linewidth=0.7)
        bar_handles.append((bars, vals))

    vmax, vmin = max(all_vals), min(all_vals)
    data_range = max(vmax - vmin, 1.0)
    pad = 0.08 * data_range
    ax.set_ylim(min(0, vmin) - pad - (0.05 * data_range if vmin < 0 else 0),
                max(0, vmax) + pad + (0.05 * data_range if vmax > 0 else 0))

    for bars, vals in bar_handles:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (pad * 0.45 if v >= 0 else -pad * 0.45),
                    f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([VAR_DISPLAY[g] for g in groups], fontsize=14, fontweight="bold")
    ax.set_xlim(x[0] - group_gap / 2, x[-1] + group_gap / 2)
    ax.set_ylabel("RMSE Improvement (%)", fontsize=14, fontweight="bold")
    ax.axhline(0, color="black", lw=1.1)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.5, color="#999999")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#333333")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#555555")

    insertion_label = "Temperature (T)" if exp_type == "T" else "Wind (U, V)"
    fig.suptitle(f"{model_name} — Cross-Variable RMSE Improvement from {insertion_label} Insertion\n"
                 f"{title_suffix}  |  Last-20-Day Average (Days 41–60) vs. Control",
                 fontsize=14.5, fontweight="bold")
    ax.legend(title="Insertion Extent", loc="best", frameon=True, fontsize=11.5,
              title_fontsize=12, framealpha=0.9, edgecolor="#bbbbbb")

    fig.savefig(out_path, bbox_inches="tight", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def save_csv(model_name, results, out_path):
    rows = []
    for (exp_type, region, var), (ctrl, exp) in results.items():
        rows.append({"exp_type": exp_type, "region": region, "var": var,
                     "control_rmse": ctrl, "experiment_rmse": exp,
                     "pct_improvement": 100 * (ctrl - exp) / ctrl})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved CSV: {out_path}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print(f"Start: {datetime.now()}\n")
    era5_cache = ERA5Cache()

    print("Loading global-insertion bars from Multimodel-study ...")
    gbars = compute_global_bars()

    aurora_res = compute_aurora_cross(era5_cache)
    save_csv("Aurora", aurora_res, OUT_DIR / "cross_var_rmse_aurora.csv")
    plot_grouped("Aurora", "T", ["u500", "v500", "z500"], aurora_res,
                 {v: gbars[("Aurora", "T", v)] for v in ["u500", "v500", "z500"]},
                 OUT_DIR / "fig1_aurora_T_insertion_cross_var.png",
                 "Bars: T_LAND / T_OCEAN / T_COASTAL / Global T insertion")
    plot_grouped("Aurora", "UV", ["t500", "z500"], aurora_res,
                 {v: gbars[("Aurora", "UV", v)] for v in ["t500", "z500"]},
                 OUT_DIR / "fig2_aurora_UV_insertion_cross_var.png",
                 "Bars: UV_LAND / UV_OCEAN / UV_COASTAL / Global UV insertion")

    fcn_res = compute_fcn_cross(era5_cache)
    save_csv("FCN", fcn_res, OUT_DIR / "cross_var_rmse_fcn.csv")
    plot_grouped("FCNv2", "T", ["u500", "v500", "z500"], fcn_res,
                 {v: gbars[("FCN", "T", v)] for v in ["u500", "v500", "z500"]},
                 OUT_DIR / "fig1_fcn_T_insertion_cross_var.png",
                 "Bars: t_land / t_ocean / t_coastal / Global T insertion")
    plot_grouped("FCNv2", "UV", ["t500", "z500"], fcn_res,
                 {v: gbars[("FCN", "UV", v)] for v in ["t500", "z500"]},
                 OUT_DIR / "fig2_fcn_UV_insertion_cross_var.png",
                 "Bars: uv_land / uv_ocean / uv_coastal / Global UV insertion")

    pri_res = compute_prithvi_cross()
    save_csv("Prithvi", pri_res, OUT_DIR / "cross_var_rmse_prithvi.csv")
    plot_grouped("Prithvi WxC", "T", ["u500", "v500", "z500"], pri_res,
                 {v: gbars[("Prithvi", "T", v)] for v in ["u500", "v500", "z500"]},
                 OUT_DIR / "fig1_prithvi_T_insertion_cross_var.png",
                 "Bars: land / ocean / coastal / Global T insertion")
    plot_grouped("Prithvi WxC", "UV", ["t500", "z500"], pri_res,
                 {v: gbars[("Prithvi", "UV", v)] for v in ["t500", "z500"]},
                 OUT_DIR / "fig2_prithvi_UV_insertion_cross_var.png",
                 "Bars: land / ocean / coastal / Global UV insertion")

    print(f"\nDone: {datetime.now()}")

if __name__ == "__main__":
    main()
