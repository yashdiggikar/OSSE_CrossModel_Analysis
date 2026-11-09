#!/usr/bin/env python3
"""
Section 5.12.1 – Temperature Insertion by Cloud Regime  (cloud/tcc)
Figures 5.75 (grouped bar) & 5.76 (time series)

Cross-variable targets of T insertion:
  U500, V500 : wind at 500 hPa
  Z500       : geopotential at 500 hPa
  Q700       : moisture at 700 hPa
               [sp. humidity Aurora/Prithvi/GC | RH% FCN]

Cloud regimes (ERA5 TCC):
  CLR  TCC < 0.10          Clear sky
  CIR  0.10 ≤ TCC < 0.40  Cirrus / thin
  MIX  0.40 ≤ TCC < 0.70  Mixed
  OPQ  TCC ≥ 0.70          Opaque / overcast

Models:
  Aurora     (Feb 2023, 12-hourly, 120 steps)
  FourCastNet(Feb 2023, 12-hourly, 120 steps)
  Prithvi-WxC(Jan 2022, 6-hourly,  240 steps)
  GraphCast  (Jan 2022, 6-hourly,  240 steps)   ← NEW
"""

import os, gc, warnings
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _regime_terrain_plot_common import apply_style, build_bar_figure, build_timeseries_figure

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
OUT_BASE  = BASE / "common/tcc"
ERA5_DIR  = BASE / "AURORA_DATA/rmse_data"          # 2023 GRIB for Aurora/FCN
ERA5_2022 = BASE / "era5_vs_cams/era5_2022"          # 2022 GRIB for GC
MERRA_DIR = BASE / "prithvi/data/merra2"
TCC_DIR   = BASE / "ALL_ERA5/tcc"
GC_DIR    = BASE / "gc_pred_files"

OUT = {m: OUT_BASE/m for m in ["aurora","fcn","prithvi","graphcast"]}
for d in OUT.values():
    (d/"intermediates").mkdir(parents=True, exist_ok=True)

AUR_CTRL  = BASE / "auroraoutput_1year_1stfeb23.nc"
AUR_T_INS = BASE / "Aurora_Pred_Files/auroraoutput_temp_inserted.nc"
FCN_CTRL  = BASE / "FourCastNet/forecast_normal_60days_12hr.nc"
FCN_T_INS = BASE / "FourCastNet/Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc"
PRI_CTRL  = BASE / "prithviOSSE/run2/output/control_forecast_Jan2022.nc"
PRI_T_INS = BASE / "prithviOSSE/run2/output/osse_forecast_Jan2022.nc"
GC_CTRL   = GC_DIR / "gc_sm_240steps_STRICT.nc"
GC_T_INS  = GC_DIR / "gc_sm_240steps_temp_inj.nc"

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
REGIME_BOUNDS = [0.0, 0.10, 0.40, 0.70, 1.01]
REGIME_NAMES  = ["Clear\n(TCC<10%)", "Cirrus\n(10–40%)", "Mixed\n(40–70%)", "Opaque\n(>70%)"]
REGIME_SHORT  = ["Clear", "Cirrus", "Mixed", "Opaque"]
REGIME_COLORS = ["#D4A24E", "#6F9FB8", "#8C7AA9", "#3B4252"]
N_REG = 4

AURORA_IC  = datetime(2023, 2, 1, 12, 0)
FCN_IC     = datetime(2023, 2, 1,  0, 0)
PRITHVI_IC = datetime(2022, 1, 1, 12, 0)
GC_IC      = datetime(2022, 1, 1,  6, 0)

N_AUR_FCN = 120; N_PRI = 240; N_GC = 240

AUR_FCN_INJ = 60;  AUR_FCN_ADJ = slice(60, 80);   AUR_FCN_MAT = slice(80, 120)
PRI_INJ     = 120; PRI_ADJ     = slice(120, 160);  PRI_MAT     = slice(160, 240)
GC_INJ      = 120; GC_ADJ      = slice(120, 160);  GC_MAT      = slice(160, 240)

FCN_CH = {"u500": 15, "v500": 28, "z500": 41, "r700": 69}

# GC pressure levels: [50,100,150,200,250,300,400,500,600,700,850,925,1000]
GC_500_IDX = 7; GC_700_IDX = 9

# Prithvi / MERRA2 level indices
PRI_500_IDX  = 7;  PRI_700_IDX  = 8   # approximate; MERRA ascending lev
MERRA_500_IDX= 6;  MERRA_700_IDX= 5

CROSS_VARS = ["U500", "V500", "Z500", "Q700"]
VAR_LABELS = {
    "U500": "U-wind 500 hPa (m/s)",
    "V500": "V-wind 500 hPa (m/s)",
    "Z500": "Geopotential 500 hPa (m2/s2)",
    "Q700": "Moisture 700 hPa  [Sp. Humidity (Aurora/Prithvi/GC) | RH% (FCN)]",
}
VAR_SHORT = {"U500": "U500", "V500": "V500", "Z500": "Z500", "Q700": "Q/R 700"}

MODEL_LABELS = {"aurora": "Aurora", "fcn": "FourCastNet",
                "prithvi": "Prithvi-WxC", "graphcast": "GraphCast"}
MODEL_ORDER  = ["aurora", "fcn", "prithvi", "graphcast"]

apply_style()

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def cos_weights_2d(nlat, lat0, lat1, nlon):
    w = np.cos(np.deg2rad(np.linspace(lat0, lat1, nlat))).clip(0)
    return w[:, None] * np.ones((1, nlon))

def nearest_time_idx(ref_times, target_dt):
    return int(np.argmin([abs((t - target_dt).total_seconds()) for t in ref_times]))

def classify_regime(tcc):
    reg = np.zeros(tcc.shape, dtype=np.int8)
    for r, (lo, hi) in enumerate(zip(REGIME_BOUNDS[:-1], REGIME_BOUNDS[1:])):
        reg[(tcc >= lo) & (tcc < hi)] = r
    return reg

def rmse_by_regime(fc, truth, regime_mask, cos_w2d):
    err2 = (fc - truth) ** 2
    out  = np.full(N_REG, np.nan)
    for r in range(N_REG):
        mask = regime_mask == r
        w    = cos_w2d[mask]
        if w.sum() < 1e-8: continue
        out[r] = np.sqrt(np.sum(w * err2[mask]) / w.sum())
    return out

def compute_rmse_series(fc_arr, truth_arr, regime_arr, cos2d, truth_idx=None):
    N   = fc_arr.shape[0]
    out = np.full((N, N_REG), np.nan)
    for s in range(N):
        tr     = truth_arr[truth_idx[s]] if truth_idx is not None else truth_arr[s]
        out[s] = rmse_by_regime(fc_arr[s], tr, regime_arr[s], cos2d)
    return out

def improvement_pct(ctrl_base, ins_phase_mean, eps=1e-12):
    return np.where(ctrl_base > eps,
                    100.0 * (ctrl_base - ins_phase_mean) / ctrl_base, np.nan)

def phase_improvement(rmse_dict, var, inj, adj, mat):
    cb = rmse_dict[var]["ctrl"][inj]
    return (improvement_pct(cb, np.nanmean(rmse_dict[var]["ins"][adj], axis=0)),
            improvement_pct(cb, np.nanmean(rmse_dict[var]["ins"][mat], axis=0)))

def era5_to_rh(q, t, p_hpa):
    e_s = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    q_s = 0.622 * e_s / (p_hpa - e_s)
    return np.clip(q / (q_s + 1e-12), 0, 1.5) * 100.0

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 – TCC REGIME MASKS
# ──────────────────────────────────────────────────────────────────────────────
log("=== Building TCC regime masks ===")

tcc23_ds    = xr.open_dataset(TCC_DIR/"era5_total_cloud_cover_2023_Feb_Apr_6hourly.nc", engine="h5netcdf")
tcc23_data  = tcc23_ds["tcc"].values
tcc23_times = [pd.Timestamp(str(t)).to_pydatetime() for t in tcc23_ds["valid_time"].values]
tcc23_ds.close()

tcc22_ds    = xr.open_dataset(TCC_DIR/"era5_total_cloud_cover_2022_jan_feb_6hourly.nc", engine="h5netcdf")
tcc22_data  = tcc22_ds["tcc"].values
tcc22_times = [pd.Timestamp(str(t)).to_pydatetime() for t in tcc22_ds["valid_time"].values]
tcc22_ds.close()
log(f"  TCC 2023: {tcc23_data.shape}   TCC 2022: {tcc22_data.shape}")

aur_vtimes = [AURORA_IC  + timedelta(hours=12*s) for s in range(N_AUR_FCN)]
fcn_vtimes = [FCN_IC     + timedelta(hours=12*s) for s in range(N_AUR_FCN)]
pri_vtimes = [PRITHVI_IC + timedelta(hours= 6*s) for s in range(N_PRI)]
gc_vtimes  = [GC_IC      + timedelta(hours= 6*(s+1)) for s in range(N_GC)]

def build_regime_masks_af(tcc_data, tcc_times, vtimes, nlat):
    masks = np.zeros((len(vtimes), nlat, 1440), dtype=np.int8)
    for s, vt in enumerate(vtimes):
        ti       = nearest_time_idx(tcc_times, vt)
        masks[s] = classify_regime(tcc_data[ti, :nlat, :])
    return masks

log("  Aurora …");  aur_regime = build_regime_masks_af(tcc23_data, tcc23_times, aur_vtimes, 720)
log("  FCN   …");   fcn_regime = build_regime_masks_af(tcc23_data, tcc23_times, fcn_vtimes, 721)

# Prithvi → 360×576 via nearest-neighbour from TCC (720×1440)
log("  Prithvi …")
merra_lats = np.linspace(-89.5, 90.0, 360)
merra_lons = np.arange(576) * (360.0/576.0)
era5_lat_i = np.clip(np.round((90.0 - merra_lats) / 0.25).astype(int), 0, 719)
era5_lon_i = (np.round(merra_lons / 0.25).astype(int)) % 1440
lat_g, lon_g = np.meshgrid(era5_lat_i, era5_lon_i, indexing="ij")
pri_regime = np.zeros((N_PRI, 360, 576), dtype=np.int8)
for s, vt in enumerate(pri_vtimes):
    ti = nearest_time_idx(tcc22_times, vt)
    pri_regime[s] = classify_regime(tcc22_data[ti][lat_g, lon_g])

# GraphCast → 181×360 (1°, lat -90 to 90 ascending)
log("  GraphCast …")
gc_lats_1d = np.linspace(-90.0, 90.0, 181)
gc_lons_1d = np.linspace(0.0, 359.0, 360)
tcc_lat_gc = np.clip(np.round((90.0 - gc_lats_1d) / 0.25).astype(int), 0, 719)
tcc_lon_gc = (np.round(gc_lons_1d / 0.25).astype(int)) % 1440
lg_gc, lg2_gc = np.meshgrid(tcc_lat_gc, tcc_lon_gc, indexing="ij")
gc_regime = np.zeros((N_GC, 181, 360), dtype=np.int8)
for s, vt in enumerate(gc_vtimes):
    ti = nearest_time_idx(tcc22_times, vt)
    gc_regime[s] = classify_regime(tcc22_data[ti][lg_gc, lg2_gc])

del tcc23_data, tcc22_data; gc.collect()
log("  Regime masks done.")

aur_cos2d = cos_weights_2d(720,  90.0, -89.75, 1440)
fcn_cos2d = cos_weights_2d(721,  90.0, -90.0,  1440)
pri_cos2d = cos_weights_2d(360, -89.5,  90.0,   576)
gc_cos2d  = cos_weights_2d(181, -90.0,  90.0,   360)

aur_lead = np.arange(N_AUR_FCN) * 0.5
fcn_lead = np.arange(N_AUR_FCN) * 0.5
pri_lead = np.arange(N_PRI)     * 0.25
gc_lead  = np.arange(N_GC)      * 0.25

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 – ERA5 2023 TRUTH (Aurora / FCN)
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== ERA5 2023 truth (Aurora/FCN) ===")
ERA5_2023_FILES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2,3,4]]

def load_era5_pres_23(sn, lev):
    dsets    = [xr.open_dataset(str(f), engine="cfgrib",
                filter_by_keys={"shortName": sn, "typeOfLevel": "isobaricInhPa",
                                "level": int(lev)}) for f in ERA5_2023_FILES]
    combined = xr.concat(dsets, dim="time").sortby("time")
    times    = [pd.Timestamp(str(t)).to_pydatetime() for t in combined.time.values]
    data     = combined[sn].values.astype(np.float32)
    for d in dsets: d.close()
    return times, data

log("  U500 …"); era5_u_times, era5_u500 = load_era5_pres_23("u", 500)
log("  V500 …"); _,            era5_v500 = load_era5_pres_23("v", 500)
log("  Z500 …"); _,            era5_z500 = load_era5_pres_23("z", 500)
log("  Q700 …"); era5_q_times, era5_q700 = load_era5_pres_23("q", 700)
log("  T700 …"); _,            era5_t700 = load_era5_pres_23("t", 700)
log(f"  shape: {era5_u500.shape}")

log("  RH700 …")
era5_r700 = era5_to_rh(era5_q700, era5_t700, 700.0).astype(np.float32)
del era5_t700; gc.collect()

aur_era5_idx = [nearest_time_idx(era5_u_times, vt) for vt in aur_vtimes]
fcn_era5_idx = [nearest_time_idx(era5_u_times, vt) for vt in fcn_vtimes]

era5_u500_720 = era5_u500[:, :-1, :]; era5_v500_720 = era5_v500[:, :-1, :]
era5_z500_720 = era5_z500[:, :-1, :]; era5_q700_720 = era5_q700[:, :-1, :]
era5_r700_720 = era5_r700[:, :-1, :]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 – ERA5 2022 TRUTH → bilinear regrid to GC 1° grid
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== ERA5 2022 truth → 1° (GraphCast) ===")
ERA5_2022_FILES = [ERA5_2022 / f"era5_pressure_2022_{m:02d}.grib" for m in [1,2,3]]

ERA5_LATS_ORIG = np.linspace(90.0, -90.0, 721)
ERA5_LONS_ORIG = np.linspace(0.0, 359.75, 1440)
ERA5_LATS_ASC  = ERA5_LATS_ORIG[::-1]
_gc_mesh       = np.array(np.meshgrid(gc_lats_1d, gc_lons_1d, indexing="ij")).reshape(2,-1).T

def bilinear_to_gc(arr2d_721x1440):
    arr_asc = arr2d_721x1440[::-1, :]
    interp  = RegularGridInterpolator((ERA5_LATS_ASC, ERA5_LONS_ORIG), arr_asc,
                                       method="linear", bounds_error=False, fill_value=None)
    return interp(_gc_mesh).reshape(181, 360).astype(np.float32)

def load_era5_pres_22_gc(sn, lev):
    dsets    = [xr.open_dataset(str(f), engine="cfgrib",
                filter_by_keys={"shortName": sn, "typeOfLevel": "isobaricInhPa",
                                "level": int(lev)}) for f in ERA5_2022_FILES]
    combined = xr.concat(dsets, dim="time").sortby("time")
    times    = [pd.Timestamp(str(t)).to_pydatetime() for t in combined.time.values]
    arr      = combined[sn].values.astype(np.float32)  # (T, 721, 1440)
    for d in dsets: d.close()
    # bilinear regrid every time step
    out = np.empty((arr.shape[0], 181, 360), dtype=np.float32)
    for t in range(arr.shape[0]):
        out[t] = bilinear_to_gc(arr[t])
    return times, out

log("  U500 …"); gc_u_times, gc_era5_u500 = load_era5_pres_22_gc("u", 500)
log("  V500 …"); _,          gc_era5_v500 = load_era5_pres_22_gc("v", 500)
log("  Z500 …"); _,          gc_era5_z500 = load_era5_pres_22_gc("z", 500)
log("  Q700 …"); _,          gc_era5_q700 = load_era5_pres_22_gc("q", 700)
log(f"  shape: {gc_era5_u500.shape}")

gc_era5_idx = [nearest_time_idx(gc_u_times, vt) for vt in gc_vtimes]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 – AURORA
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== AURORA ===")

def load_aur(fpath, key, n):
    with h5py.File(str(fpath), "r") as f:
        return f[key][:n].astype(np.float32)

aur_vmap = {
    "U500": ("u500", era5_u500_720, aur_era5_idx),
    "V500": ("v500", era5_v500_720, aur_era5_idx),
    "Z500": ("z500", era5_z500_720, aur_era5_idx),
    "Q700": ("q700", era5_q700_720, aur_era5_idx),
}
aur_rmse = {}
for var, (key, truth, tidx) in aur_vmap.items():
    log(f"  {var} …")
    ctrl = load_aur(AUR_CTRL,  key, N_AUR_FCN)
    ins  = load_aur(AUR_T_INS, key, N_AUR_FCN)
    aur_rmse[var] = {
        "ctrl": compute_rmse_series(ctrl, truth, aur_regime, aur_cos2d, tidx),
        "ins":  compute_rmse_series(ins,  truth, aur_regime, aur_cos2d, tidx),
    }
    del ctrl, ins; gc.collect()

del era5_u500_720, era5_v500_720, era5_z500_720, era5_q700_720, era5_r700_720; gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 – FCN
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== FCN ===")

def load_fcn(fpath, ch, n):
    with h5py.File(str(fpath), "r") as f:
        return f["prediction"][:n, ch, :, :].astype(np.float32)

fcn_vmap = {
    "U500": (FCN_CH["u500"], era5_u500, fcn_era5_idx),
    "V500": (FCN_CH["v500"], era5_v500, fcn_era5_idx),
    "Z500": (FCN_CH["z500"], era5_z500, fcn_era5_idx),
    "Q700": (FCN_CH["r700"], era5_r700, fcn_era5_idx),
}
fcn_rmse = {}
for var, (ch, truth, tidx) in fcn_vmap.items():
    log(f"  {var} (ch={ch}) …")
    ctrl = load_fcn(FCN_CTRL,  ch, N_AUR_FCN)
    ins  = load_fcn(FCN_T_INS, ch, N_AUR_FCN)
    fcn_rmse[var] = {
        "ctrl": compute_rmse_series(ctrl, truth, fcn_regime, fcn_cos2d, tidx),
        "ins":  compute_rmse_series(ins,  truth, fcn_regime, fcn_cos2d, tidx),
    }
    del ctrl, ins; gc.collect()

del era5_u500, era5_v500, era5_z500, era5_q700, era5_r700; gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 – PRITHVI
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== PRITHVI ===")

def build_merra2_truth(var_name, lev_idx, n_steps, vtimes):
    out = np.full((n_steps, 360, 576), np.nan, dtype=np.float32)
    prev_date = None; pres_ds = None
    for s, vt in enumerate(vtimes):
        date_str = vt.strftime("%Y%m%d"); ti_3h = vt.hour // 3
        if date_str != prev_date:
            if pres_ds is not None: pres_ds.close()
            ppath = MERRA_DIR / f"MERRA_pres_{date_str}.nc"
            if not ppath.exists():
                log(f"  WARN: missing {date_str}"); prev_date = date_str; continue
            pres_ds = xr.open_dataset(str(ppath), engine="h5netcdf")
            prev_date = date_str
        out[s] = pres_ds[var_name].values[ti_3h, lev_idx, 1:, :]
        if s % 40 == 0: log(f"    step {s}/{n_steps}  {vt}")
    if pres_ds is not None: pres_ds.close()
    return out

def load_pri(fpath, var_name, lev_idx):
    ds  = xr.open_dataset(str(fpath), engine="h5netcdf")
    arr = ds[var_name].values[:, lev_idx, :, :]
    ds.close()
    if arr.shape[1] == 361:   # control has south-pole row; crop to 360 like OSSE files
        arr = arr[:, 1:, :]
    return arr.astype(np.float32)

pri_vmap = {
    "U500": ("U",  MERRA_500_IDX, PRI_500_IDX),
    "V500": ("V",  MERRA_500_IDX, PRI_500_IDX),
    "Z500": ("H",  MERRA_500_IDX, PRI_500_IDX),
    "Q700": ("QV", MERRA_700_IDX, PRI_700_IDX),
}
pri_rmse = {}
for var, (mname, m_li, p_li) in pri_vmap.items():
    log(f"  {var} (MERRA2 '{mname}' lev_idx={m_li}) …")
    truth = build_merra2_truth(mname, m_li, N_PRI, pri_vtimes)
    ctrl  = load_pri(PRI_CTRL,  mname, p_li)
    ins   = load_pri(PRI_T_INS, mname, p_li)
    pri_rmse[var] = {
        "ctrl": compute_rmse_series(ctrl, truth, pri_regime, pri_cos2d),
        "ins":  compute_rmse_series(ins,  truth, pri_regime, pri_cos2d),
    }
    del truth, ctrl, ins; gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 – GRAPHCAST
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== GRAPHCAST ===")

def load_gc(fpath, varname, lev_idx):
    ds  = xr.open_dataset(str(fpath), engine="h5netcdf", decode_timedelta=False)
    arr = ds[varname].values[:N_GC, 0, lev_idx, :, :]   # (N_GC, 181, 360)
    ds.close()
    return arr.astype(np.float32)

gc_vmap = {
    "U500": ("u_component_of_wind", GC_500_IDX, gc_era5_u500, gc_era5_idx),
    "V500": ("v_component_of_wind", GC_500_IDX, gc_era5_v500, gc_era5_idx),
    "Z500": ("geopotential",        GC_500_IDX, gc_era5_z500, gc_era5_idx),
    "Q700": ("specific_humidity",   GC_700_IDX, gc_era5_q700, gc_era5_idx),
}
gc_rmse = {}
for var, (gcvar, li, truth, tidx) in gc_vmap.items():
    log(f"  {var} ('{gcvar}' lev_idx={li}) …")
    ctrl = load_gc(GC_CTRL,  gcvar, li)
    ins  = load_gc(GC_T_INS, gcvar, li)
    gc_rmse[var] = {
        "ctrl": compute_rmse_series(ctrl, truth, gc_regime, gc_cos2d, tidx),
        "ins":  compute_rmse_series(ins,  truth, gc_regime, gc_cos2d, tidx),
    }
    del ctrl, ins; gc.collect()

del gc_era5_u500, gc_era5_v500, gc_era5_z500, gc_era5_q700; gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 – CSVs
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== Saving CSVs ===")

def save_rmse_csv(rmse_dict, lead_arr, out_dir, tag):
    rows = []
    for var, d in rmse_dict.items():
        for s in range(len(lead_arr)):
            for r in range(N_REG):
                rows.append({"step": s, "lead_day": round(float(lead_arr[s]),3),
                             "variable": var, "regime": REGIME_SHORT[r],
                             "ctrl_rmse": float(d["ctrl"][s,r]),
                             "ins_rmse":  float(d["ins"][s,r])})
    pd.DataFrame(rows).to_csv(out_dir/"intermediates"/f"{tag}_t_ins_regime_rmse.csv", index=False)
    log(f"  saved → {tag}")

save_rmse_csv(aur_rmse, aur_lead, OUT["aurora"],    "aurora")
save_rmse_csv(fcn_rmse, fcn_lead, OUT["fcn"],       "fcn")
save_rmse_csv(pri_rmse, pri_lead, OUT["prithvi"],   "prithvi")
save_rmse_csv(gc_rmse,  gc_lead,  OUT["graphcast"], "graphcast")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 – IMPROVEMENTS
# ──────────────────────────────────────────────────────────────────────────────
ALL_RMSE = {
    "aurora":     (aur_rmse, AUR_FCN_INJ, AUR_FCN_ADJ, AUR_FCN_MAT, aur_lead),
    "fcn":        (fcn_rmse, AUR_FCN_INJ, AUR_FCN_ADJ, AUR_FCN_MAT, fcn_lead),
    "prithvi":    (pri_rmse, PRI_INJ,     PRI_ADJ,     PRI_MAT,     pri_lead),
    "graphcast":  (gc_rmse,  GC_INJ,      GC_ADJ,      GC_MAT,      gc_lead),
}

improvements = {}
for model, (rdict, inj, adj, mat, lead) in ALL_RMSE.items():
    improvements[model] = {}
    for var in CROSS_VARS:
        if var not in rdict: continue
        a, m = phase_improvement(rdict, var, inj, adj, mat)
        improvements[model][var] = {"adj": a, "mat": m}

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 – FIGURE 5.75: GROUPED BAR CHART (4 models × 4 vars)
# ──────────────────────────────────────────────────────────────────────────────
log("\n--- Figure 5.75 ---")

fig75 = build_bar_figure(
    "Figure 5.75 – Temperature Insertion: RMSE Improvement (%) by Cloud Regime\n"
    "Baseline = Day-30 Control RMSE  |  Positive = improvement  |  4 Models",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_SHORT, improvements,
    REGIME_NAMES, N_REG, "Cloud Regime", figsize=(24, 18))
for model, out_dir in OUT.items():
    fig75.savefig(out_dir/"fig5_75_t_ins_rmse_improvement_by_regime.png", bbox_inches="tight", dpi=300)
    log(f"  saved → {out_dir.name}")
plt.close(fig75)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 – FIGURE 5.76: TIME SERIES (4 vars × 4 models)
# ──────────────────────────────────────────────────────────────────────────────
log("\n--- Figure 5.76 ---")

fig76 = build_timeseries_figure(
    "Figure 5.76 – Temperature Insertion: RMSE Time Series by Cloud Regime\n"
    "Dashed = Control  |  Solid = T Insertion  |  Blue = Adj. phase  |  Pink = Mature phase",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_LABELS, ALL_RMSE,
    REGIME_COLORS, REGIME_SHORT, figsize=(28, 20), hspace=0.5, wspace=0.28)
for model, out_dir in OUT.items():
    fig76.savefig(out_dir/"fig5_76_t_ins_rmse_timeseries_by_regime.png", bbox_inches="tight", dpi=300)
    log(f"  saved → {out_dir.name}")
plt.close(fig76)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12 – SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== Summary table ===")
rows = []
for model, (rdict, inj, adj, mat, lead) in ALL_RMSE.items():
    for var in CROSS_VARS:
        if var not in rdict: continue
        ai, mi = phase_improvement(rdict, var, inj, adj, mat)
        for r in range(N_REG):
            rows.append({"Model": MODEL_LABELS[model], "Variable": var,
                         "Regime": REGIME_SHORT[r],
                         "Ctrl_RMSE_Day30": round(float(rdict[var]["ctrl"][inj,r]),6),
                         "Adj_Imp_%": round(float(ai[r]),3),
                         "Mat_Imp_%": round(float(mi[r]),3)})
df = pd.DataFrame(rows)
df.to_csv(OUT_BASE/"table_t_ins_cloud_regime_summary.csv", index=False)
for model, out_dir in OUT.items():
    df[df["Model"]==MODEL_LABELS[model]].to_csv(out_dir/"t_ins_cloud_regime_summary.csv", index=False)

log("\n=== cloud/tcc complete ===")
