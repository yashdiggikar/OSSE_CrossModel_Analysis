#!/usr/bin/env python3
"""
Section 5.12 – Cloud-Fraction Regime Analysis
Figures 5.74–5.82, Table 5.14

Experiments (3 main perfect-insertion runs per model):
  Aurora  : auroraoutput_temp_inserted  / _uv_inserted / _geopotential_inserted
  FCN     : forecast_truth_injected     / _uv_injected  / _z_injected
  Prithvi : t_sp_ins                    / uv_ins_run    / geo_ins_run

Cloud regimes (ERA5 TCC):
  CLR  = TCC < 0.10       (Clear)
  CIR  = 0.10 ≤ TCC < 0.40  (Cirrus / thin)
  MIX  = 0.40 ≤ TCC < 0.70  (Mixed / partly cloudy)
  OPQ  = TCC ≥ 0.70       (Opaque / overcast)

Outputs:  ALL_ERA5/tcc/{aurora,fcn,prithvi}/
          ├─ fig5_74 … fig5_82 .png
          ├─ table5_14_cloud_regime_summary.csv
          └─ intermediates/*.csv
"""

import warnings; warnings.filterwarnings("ignore")
import gc, sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
TCC_DIR   = BASE / "ALL_ERA5/tcc"
ERA5_DIR  = BASE / "AURORA_DATA/rmse_data"
MERRA_DIR = BASE / "prithviOSSE/data/merra2"
AUR_DIR   = BASE / "Aurora_Pred_Files"
FCN_DIR   = BASE / "FourCastNet"
PRI_DIR   = BASE / "prithviOSSE"
CTRL_1YR  = BASE / "auroraoutput_1year_1stfeb23.nc"

OUT = {
    "aurora":  TCC_DIR / "aurora",
    "fcn":     TCC_DIR / "fcn",
    "prithvi": TCC_DIR / "prithvi",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
REGIME_BOUNDS = [0.0, 0.10, 0.40, 0.70, 1.01]
REGIME_NAMES  = ["Clear", "Cirrus", "Mixed", "Opaque"]
REGIME_COLORS = ["#4393c3", "#92c5de", "#f4a582", "#d6604d"]
N_REG = 4

# Injection starts at step 60 for Aurora/FCN (12h steps → day 30)
# Injection starts at step 120 for Prithvi (6h steps → day 30)
AUR_FCN_INJ  = 60
PRI_INJ      = 120
AUR_FCN_ADJ  = slice(60,  90)
AUR_FCN_MAT  = slice(90,  120)
PRI_ADJ      = slice(120, 180)
PRI_MAT      = slice(180, 240)

AURORA_IC    = datetime(2023, 2, 1, 12, 0)
FCN_IC       = datetime(2023, 2, 1,  0, 0)
PRITHVI_IC   = datetime(2022, 1, 1, 12, 0)
N_AUR_FCN    = 120
N_PRI        = 240

# FCN channel indices
FCN_CH = {
    "u10m": 0, "v10m": 1, "t2m": 4, "msl": 6, "tcwv": 7,
    "u500": 15, "v500": 28, "z500": 41,
    "t500": 54, "r700": 68,
}

# Prithvi level indices  (ascending lev=[34,39,41,43,44,45,48,51,53,56,63,68,71,72])
# hPa ≈                   [52,118,164,226,265,310,424,532,604,693,832,903,946,960]
PRI_500_IDX = 7   # lev=51, ~532 hPa
PRI_700_IDX = 9   # lev=56, ~693 hPa

# MERRA2 level indices (descending lev=[72,71,68,63,56,53,51,48,45,44,43,41,39,34])
MERRA_500_IDX = 6  # lev=51, ~532 hPa
MERRA_700_IDX = 4  # lev=56, ~693 hPa

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9,
})

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)


def cos_weights(n_lat, lat_start, lat_end):
    """1D cosine-lat weights (ascending or descending)."""
    lats = np.linspace(lat_start, lat_end, n_lat)
    return np.cos(np.deg2rad(lats)).clip(0)


def rmse_by_regime(fc_step, truth_step, regime_mask_step, cos_w2d):
    """
    Compute RMSE in each regime for ONE timestep.
    fc_step, truth_step : (nlat, nlon)
    regime_mask_step    : (nlat, nlon) integer 0-3
    cos_w2d             : (nlat, nlon)
    Returns array of shape (N_REG,).
    """
    err2 = (fc_step - truth_step) ** 2
    out  = np.full(N_REG, np.nan)
    for r in range(N_REG):
        mask = regime_mask_step == r
        w    = cos_w2d[mask]
        if w.sum() < 1e-6:
            continue
        out[r] = np.sqrt(np.sum(w * err2[mask]) / w.sum())
    return out


def bias_by_regime(fc_step, truth_step, regime_mask_step, cos_w2d):
    """Cosine-lat weighted mean bias per regime (signed)."""
    diff = fc_step - truth_step
    out  = np.full(N_REG, np.nan)
    for r in range(N_REG):
        mask = regime_mask_step == r
        w    = cos_w2d[mask]
        if w.sum() < 1e-6:
            continue
        out[r] = np.sum(w * diff[mask]) / w.sum()
    return out


def classify_regime(tcc_2d):
    """Map TCC values to regime index 0-3."""
    reg = np.zeros(tcc_2d.shape, dtype=np.int8)
    for r, (lo, hi) in enumerate(zip(REGIME_BOUNDS[:-1], REGIME_BOUNDS[1:])):
        reg[(tcc_2d >= lo) & (tcc_2d < hi)] = r
    return reg


def nearest_era5_time_idx(era5_times_dt, target_dt):
    """Return index of nearest ERA5 timestamp."""
    deltas = [abs((t - target_dt).total_seconds()) for t in era5_times_dt]
    return int(np.argmin(deltas))


def improvement_pct(ctrl, ins, eps=1e-10):
    """% RMSE improvement; positive = better."""
    return np.where(ctrl > eps, 100. * (ctrl - ins) / ctrl, np.nan)


def save_csv(df, path):
    df.to_csv(path, index=False)
    log(f"  saved → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOAD ERA5 TCC AND BUILD REGIME MASKS
# ─────────────────────────────────────────────────────────────────────────────

log("=== Loading ERA5 TCC ===")

# 2023 file for Aurora / FCN
tcc23_ds    = xr.open_dataset(TCC_DIR / "era5_total_cloud_cover_2023_Feb_Apr_6hourly.nc")
tcc23_data  = tcc23_ds["tcc"].values           # (356, 721, 1440)
tcc23_times = [pd.Timestamp(str(t)).to_pydatetime()
               for t in tcc23_ds["valid_time"].values]
tcc23_ds.close()
log(f"  TCC 2023: {tcc23_data.shape}  {tcc23_times[0]} … {tcc23_times[-1]}")

# 2022 file for Prithvi
tcc22_ds    = xr.open_dataset(TCC_DIR / "era5_total_cloud_cover_2022_jan_feb_6hourly.nc")
tcc22_data  = tcc22_ds["tcc"].values           # (360, 721, 1440)
tcc22_times = [pd.Timestamp(str(t)).to_pydatetime()
               for t in tcc22_ds["valid_time"].values]
tcc22_ds.close()
log(f"  TCC 2022: {tcc22_data.shape}  {tcc22_times[0]} … {tcc22_times[-1]}")

# Build Aurora/FCN valid times
aur_valid_times = [AURORA_IC + timedelta(hours=12 * s) for s in range(N_AUR_FCN)]
fcn_valid_times = [FCN_IC    + timedelta(hours=12 * s) for s in range(N_AUR_FCN)]
pri_valid_times = [PRITHVI_IC + timedelta(hours=6 * s) for s in range(N_PRI)]

# Match each model step to nearest TCC timestamp
def build_tcc_regime_masks_aurora_fcn(tcc_data, tcc_times, valid_times, nlat=720):
    """
    Build regime masks on the Aurora/FCN 720×1440 or 721×1440 grid.
    tcc_data is 721 lats; we drop the South Pole row for Aurora (720).
    """
    masks = np.zeros((len(valid_times), nlat, 1440), dtype=np.int8)
    for s, vt in enumerate(valid_times):
        ti   = nearest_era5_time_idx(tcc_times, vt)
        tcc2 = tcc_data[ti, :nlat, :]          # clip to nlat
        masks[s] = classify_regime(tcc2)
    return masks

log("  Building Aurora regime masks (720 lats)…")
aur_regime = build_tcc_regime_masks_aurora_fcn(
    tcc23_data, tcc23_times, aur_valid_times, nlat=720)

log("  Building FCN regime masks (721 lats)…")
fcn_regime = build_tcc_regime_masks_aurora_fcn(
    tcc23_data, tcc23_times, fcn_valid_times, nlat=721)

# Prithvi: regrid ERA5 TCC (721×1440) → MERRA2 grid (360×576) nearest-neighbour
log("  Building Prithvi regime masks (360×576) with TCC 2022…")
merra_lats  = np.linspace(-89.5, 90.0, 360)        # ascending
merra_lons  = np.arange(576) * (360.0 / 576.0)     # 0 … 359.375
era5_lat_i  = np.clip(np.round((90.0 - merra_lats) / 0.25).astype(int), 0, 720)
era5_lon_i  = (np.round(merra_lons / 0.25).astype(int)) % 1440
lat_grid, lon_grid = np.meshgrid(era5_lat_i, era5_lon_i, indexing="ij")  # (360,576)

pri_regime = np.zeros((N_PRI, 360, 576), dtype=np.int8)
for s, vt in enumerate(pri_valid_times):
    ti = nearest_era5_time_idx(tcc22_times, vt)
    tcc2 = tcc22_data[ti][lat_grid, lon_grid]      # (360, 576) regridded
    pri_regime[s] = classify_regime(tcc2)

log("  Regime masks done.")

# Regime occurrence frequencies (for Fig 5.74)
def regime_freq(masks):
    """Frequency (0-1) of each regime at each grid point, averaged over time."""
    freq = np.zeros((N_REG,) + masks.shape[1:], dtype=np.float32)
    for r in range(N_REG):
        freq[r] = (masks == r).mean(axis=0)
    return freq

aur_regime_freq = regime_freq(aur_regime)   # (4, 720, 1440)
fcn_regime_freq = regime_freq(fcn_regime)   # (4, 721, 1440)
pri_regime_freq = regime_freq(pri_regime)   # (4, 360, 576)

# Global occurrence (area-weighted, for bar chart)
def global_occ(freq_4d, cos_w1d):
    w = cos_w1d / cos_w1d.sum()
    return np.array([(freq_4d[r] * w[:, None]).sum() for r in range(N_REG)])

aur_occ = global_occ(aur_regime_freq, cos_weights(720,  90.0, -89.75))
fcn_occ = global_occ(fcn_regime_freq, cos_weights(721,  90.0, -90.0))
pri_occ = global_occ(pri_regime_freq, cos_weights(360, -89.5,  90.0))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – LOAD ERA5 TRUTH FOR AURORA / FCN
# ─────────────────────────────────────────────────────────────────────────────
log("\n=== Loading ERA5 pressure-level truth (T500, U500, V500, Z500, Q700) ===")

ERA5_PRES_FILES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF_FILES = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

def load_era5_pres(shortname, level_hpa):
    dsets = []
    for f in ERA5_PRES_FILES:
        ds = xr.open_dataset(str(f), engine="cfgrib",
                             filter_by_keys={"shortName": shortname,
                                             "typeOfLevel": "isobaricInhPa",
                                             "level": int(level_hpa)})
        dsets.append(ds)
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = [pd.Timestamp(str(t)).to_pydatetime() for t in combined.time.values]
    data  = combined[shortname].values.astype(np.float32)  # (N, 721, 1440)
    for ds in dsets: ds.close()
    return times, data

_SFC_GRIB_SN = {"t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl", "tcwv": "tcwv"}
def load_era5_sfc(xr_name):
    grib_sn = _SFC_GRIB_SN.get(xr_name, xr_name)
    dsets = []
    for f in ERA5_SURF_FILES:
        ds = xr.open_dataset(str(f), engine="cfgrib",
                             filter_by_keys={"shortName": grib_sn})
        dsets.append(ds)
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = [pd.Timestamp(str(t)).to_pydatetime() for t in combined.time.values]
    data  = combined[xr_name].values.astype(np.float32)
    for ds in dsets: ds.close()
    return times, data

log("  T500…");  era5_t_times,  era5_t500  = load_era5_pres("t", 500)
log("  U500…");  _,             era5_u500  = load_era5_pres("u", 500)
log("  V500…");  _,             era5_v500  = load_era5_pres("v", 500)
log("  Z500…");  _,             era5_z500  = load_era5_pres("z", 500)
log("  Q700…");  era5_q_times,  era5_q700  = load_era5_pres("q", 700)
log("  T2M…");   era5_s_times,  era5_t2m   = load_era5_sfc("t2m")
try:
    log("  TCWV…"); _, era5_tcwv = load_era5_sfc("tcwv")
    HAS_TCWV = True
    log(f"  TCWV shape: {era5_tcwv.shape}")
except Exception as e:
    log(f"  TCWV not available ({e}) — moisture proxy skipped for FCN")
    era5_tcwv = None
    HAS_TCWV = False
log(f"  ERA5 pressure shape: {era5_t500.shape}")

# Index ERA5 times for each model step
def era5_step_indices(model_times, era5_times):
    return [nearest_era5_time_idx(era5_times, vt) for vt in model_times]

aur_t_idx = era5_step_indices(aur_valid_times, era5_t_times)
aur_s_idx = era5_step_indices(aur_valid_times, era5_s_times)
fcn_t_idx = era5_step_indices(fcn_valid_times, era5_t_times)
fcn_s_idx = era5_step_indices(fcn_valid_times, era5_s_times)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – LOAD MERRA2 TRUTH FOR PRITHVI
# ─────────────────────────────────────────────────────────────────────────────
log("\n=== Loading MERRA2 truth for Prithvi ===")

def build_merra2_truth_arrays():
    """
    For each of the 240 Prithvi steps (6h from IC 2022-01-01 12:00),
    load the nearest MERRA2 3h slot and accumulate into (240, 360, 576) arrays.
    MERRA2 lats are 361 (−90 to 90); skip index 0 (−90°) to get 360 rows.
    MERRA2 pressure lev is DESCENDING [72,71,68,...,34].
    """
    n = N_PRI
    T500  = np.full((n, 360, 576), np.nan, dtype=np.float32)
    U500  = np.full((n, 360, 576), np.nan, dtype=np.float32)
    V500  = np.full((n, 360, 576), np.nan, dtype=np.float32)
    Z500  = np.full((n, 360, 576), np.nan, dtype=np.float32)
    Q700  = np.full((n, 360, 576), np.nan, dtype=np.float32)
    T2M   = np.full((n, 360, 576), np.nan, dtype=np.float32)
    TQL   = np.full((n, 360, 576), np.nan, dtype=np.float32)
    TQI   = np.full((n, 360, 576), np.nan, dtype=np.float32)
    TQV   = np.full((n, 360, 576), np.nan, dtype=np.float32)
    CLOUD = np.full((n, 360, 576), np.nan, dtype=np.float32)  # at 500 hPa proxy

    prev_date = None
    pres_ds = sfc_ds = None

    for s, vt in enumerate(pri_valid_times):
        date_str = vt.strftime("%Y%m%d")
        ti       = vt.hour // 3   # 3h slot within day (0=00Z,1=03Z,...,7=21Z)

        if date_str != prev_date:
            if pres_ds is not None: pres_ds.close()
            if sfc_ds  is not None: sfc_ds.close()
            ppath = MERRA_DIR / f"MERRA_pres_{date_str}.nc"
            spath = MERRA_DIR / f"MERRA2_sfc_{date_str}.nc"
            if not ppath.exists() or not spath.exists():
                log(f"  WARN: MERRA2 missing for {date_str}, skipping step {s}")
                prev_date = date_str
                continue
            pres_ds = xr.open_dataset(str(ppath), engine="netcdf4")
            sfc_ds  = xr.open_dataset(str(spath),  engine="netcdf4")
            prev_date = date_str

        # Pressure vars: lev descending [72,71,68,...], lat 361 (skip row 0 = -90°)
        T500[s]  = pres_ds["T"].values[ti,  MERRA_500_IDX, 1:, :]
        U500[s]  = pres_ds["U"].values[ti,  MERRA_500_IDX, 1:, :]
        V500[s]  = pres_ds["V"].values[ti,  MERRA_500_IDX, 1:, :]
        Z500[s]  = pres_ds["H"].values[ti,  MERRA_500_IDX, 1:, :]  # geopotential height (m)
        Q700[s]  = pres_ds["QV"].values[ti, MERRA_700_IDX, 1:, :]
        CLOUD[s] = pres_ds["CLOUD"].values[ti, MERRA_500_IDX, 1:, :]
        # Surface vars
        T2M[s]  = sfc_ds["T2M"].values[ti, 1:, :]
        TQL[s]  = sfc_ds["TQL"].values[ti, 1:, :]
        TQI[s]  = sfc_ds["TQI"].values[ti, 1:, :]
        TQV[s]  = sfc_ds["TQV"].values[ti, 1:, :]

        if s % 40 == 0:
            log(f"  MERRA2 step {s}/{n}  {vt}")

    if pres_ds is not None: pres_ds.close()
    if sfc_ds  is not None: sfc_ds.close()
    return T500, U500, V500, Z500, Q700, T2M, TQL, TQI, TQV, CLOUD

(m2_T500, m2_U500, m2_V500, m2_Z500, m2_Q700,
 m2_T2M, m2_TQL, m2_TQI, m2_TQV, m2_CLOUD) = build_merra2_truth_arrays()
log(f"  MERRA2 truth shape: {m2_T500.shape}")

# Prithvi cosine weights (360 lats, ascending −89.5→90)
pri_cos1d  = cos_weights(360, -89.5, 90.0).clip(0)
pri_cos2d  = pri_cos1d[:, None] * np.ones((1, 576))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – COMPUTE REGIME RMSE  (generic engine)
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_rmse_aurora(var_key, ctrl_arr, ins_arr, truth_arr, truth_idx,
                                regime_masks, cos2d):
    """
    All arrays: ctrl/ins (N_steps, 720, 1440), truth (N_era5, 720, 1440).
    truth_idx: list of N_steps ERA5 indices.
    Returns (ctrl_rmse, ins_rmse) each shape (N_steps, N_REG).
    """
    N = ctrl_arr.shape[0]
    ctrl_r = np.full((N, N_REG), np.nan)
    ins_r  = np.full((N, N_REG), np.nan)
    for s in range(N):
        tr  = truth_arr[truth_idx[s]]
        reg = regime_masks[s]
        ctrl_r[s] = rmse_by_regime(ctrl_arr[s], tr, reg, cos2d)
        ins_r[s]  = rmse_by_regime(ins_arr[s],  tr, reg, cos2d)
    return ctrl_r, ins_r


def compute_regime_bias_aurora(var_key, ctrl_arr, ins_arr, truth_arr, truth_idx,
                                regime_masks, cos2d):
    N = ctrl_arr.shape[0]
    ctrl_b = np.full((N, N_REG), np.nan)
    ins_b  = np.full((N, N_REG), np.nan)
    for s in range(N):
        tr  = truth_arr[truth_idx[s]]
        reg = regime_masks[s]
        ctrl_b[s] = bias_by_regime(ctrl_arr[s], tr, reg, cos2d)
        ins_b[s]  = bias_by_regime(ins_arr[s],  tr, reg, cos2d)
    return ctrl_b, ins_b


def compute_regime_rmse_prithvi(ctrl_arr, ins_arr, truth_arr, regime_masks, cos2d):
    """All arrays: (N_steps, 360, 576)."""
    N = ctrl_arr.shape[0]
    ctrl_r = np.full((N, N_REG), np.nan)
    ins_r  = np.full((N, N_REG), np.nan)
    for s in range(N):
        reg = regime_masks[s]
        ctrl_r[s] = rmse_by_regime(ctrl_arr[s], truth_arr[s], reg, cos2d)
        ins_r[s]  = rmse_by_regime(ins_arr[s],  truth_arr[s], reg, cos2d)
    return ctrl_r, ins_r


def compute_regime_bias_prithvi(ctrl_arr, ins_arr, truth_arr, regime_masks, cos2d):
    N = ctrl_arr.shape[0]
    ctrl_b = np.full((N, N_REG), np.nan)
    ins_b  = np.full((N, N_REG), np.nan)
    for s in range(N):
        reg = regime_masks[s]
        ctrl_b[s] = bias_by_regime(ctrl_arr[s], truth_arr[s], reg, cos2d)
        ins_b[s]  = bias_by_regime(ins_arr[s],  truth_arr[s], reg, cos2d)
    return ctrl_b, ins_b


def rmse_df(ctrl_r, ins_r, lead_days, exp_label, var_label):
    """Pack regime RMSE into a long-form DataFrame."""
    rows = []
    N = ctrl_r.shape[0]
    for s in range(N):
        for r in range(N_REG):
            rows.append({
                "step": s, "lead_day": lead_days[s],
                "regime": REGIME_NAMES[r],
                "exp": exp_label, "var": var_label,
                "ctrl_rmse": ctrl_r[s, r],
                "ins_rmse":  ins_r[s, r],
                "improvement_pct": improvement_pct(ctrl_r[s, r], ins_r[s, r]),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – AURORA
# ─────────────────────────────────────────────────────────────────────────────
log("\n=== AURORA ===")

aur_cos1d = cos_weights(720, 90.0, -89.75)
aur_cos2d = aur_cos1d[:, None] * np.ones((1, 1440))
aur_lead  = np.arange(N_AUR_FCN) * 0.5  # lead days

# Load Aurora ctrl (first 120 steps from 1yr file, h5py)
log("  Loading Aurora ctrl (h5py)…")
aur_ctrl = {}
with h5py.File(str(CTRL_1YR), "r") as f:
    for var in ["t500", "u500", "v500", "z500", "q700", "2t", "10u", "10v", "msl"]:
        if var in f:
            aur_ctrl[var] = f[var][:120].astype(np.float32)  # (120, 720, 1440)
log(f"  ctrl vars: {list(aur_ctrl.keys())}")

# ERA5 truth for Aurora: drop South Pole row (720 lats)
era5_t500_720 = era5_t500[:, :-1, :]  # (N_era5, 720, 1440)
era5_u500_720 = era5_u500[:, :-1, :]
era5_v500_720 = era5_v500[:, :-1, :]
era5_z500_720 = era5_z500[:, :-1, :]
era5_q700_720 = era5_q700[:, :-1, :]
era5_t2m_720  = era5_t2m[:,  :-1, :]
era5_tcwv_720 = era5_tcwv[:, :-1, :] if HAS_TCWV and era5_tcwv is not None else None

AUR_EXPS = {
    "t_ins":  AUR_DIR / "auroraoutput_temp_inserted.nc",
    "uv_ins": AUR_DIR / "auroraoutput_uv_inserted.nc",
    "geo_ins": AUR_DIR / "auroraoutput_geopotential_inserted.nc",
}

aur_results = {}  # exp → var → {ctrl_r, ins_r, ctrl_b, ins_b}
for exp, fpath in AUR_EXPS.items():
    log(f"  Aurora {exp}…")
    aur_results[exp] = {}
    with h5py.File(str(fpath), "r") as f:
        ins_vars = {}
        for var in ["t500", "u500", "v500", "z500", "q700", "2t", "10u", "10v"]:
            if var in f:
                ins_vars[var] = f[var][:120].astype(np.float32)

    aur_vmap = {"T500": "t500", "U500": "u500", "V500": "v500",
                "Z500": "z500", "Q700": "q700", "T2M": "2t"}
    aur_truth_pairs = [
        ("T500", era5_t500_720, aur_t_idx),
        ("U500", era5_u500_720, aur_t_idx),
        ("V500", era5_v500_720, aur_t_idx),
        ("Z500", era5_z500_720, aur_t_idx),
        ("Q700", era5_q700_720, aur_t_idx),
        ("T2M",  era5_t2m_720,  aur_s_idx),
    ]
    for var, truth_arr, tidx in aur_truth_pairs:
        ak = aur_vmap.get(var)
        if ak is None or ak not in aur_ctrl or ak not in ins_vars:
            continue
        cr, ir = compute_regime_rmse_aurora(
            var, aur_ctrl[ak], ins_vars[ak], truth_arr, tidx, aur_regime, aur_cos2d)
        cb, ib = compute_regime_bias_aurora(
            var, aur_ctrl[ak], ins_vars[ak], truth_arr, tidx, aur_regime, aur_cos2d)
        aur_results[exp][var] = {"ctrl_r": cr, "ins_r": ir, "ctrl_b": cb, "ins_b": ib}

    del ins_vars; gc.collect()

# UV500 = sqrt(U²+V²)
for exp in AUR_EXPS:
    if "U500" in aur_results[exp] and "V500" in aur_results[exp]:
        # Approximate: UV500 error ≈ sqrt of combined for regime masks (full grid needed)
        # Use already computed U and V RMSE as proxies, or load jointly
        # For simplicity, average U500 and V500 RMSE
        ur = aur_results[exp]["U500"]["ctrl_r"]
        vr = aur_results[exp]["V500"]["ctrl_r"]
        aur_results[exp]["UV500"] = {
            "ctrl_r": np.sqrt((ur**2 + vr**2) / 2),
            "ins_r":  np.sqrt((aur_results[exp]["U500"]["ins_r"]**2 +
                               aur_results[exp]["V500"]["ins_r"]**2) / 2),
            "ctrl_b": (aur_results[exp]["U500"]["ctrl_b"] +
                       aur_results[exp]["V500"]["ctrl_b"]) / 2,
            "ins_b":  (aur_results[exp]["U500"]["ins_b"] +
                       aur_results[exp]["V500"]["ins_b"]) / 2,
        }

# Save Aurora CSVs
for exp in AUR_EXPS:
    frames = []
    for var in aur_results[exp]:
        d = aur_results[exp][var]
        df = rmse_df(d["ctrl_r"], d["ins_r"], aur_lead, exp, var)
        frames.append(df)
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        save_csv(df_all, OUT["aurora"] / "intermediates" / f"aurora_{exp}_regime_rmse.csv")
        # Bias CSV
        bias_rows = []
        for var in aur_results[exp]:
            d = aur_results[exp][var]
            for s in range(N_AUR_FCN):
                for r in range(N_REG):
                    bias_rows.append({
                        "step": s, "lead_day": aur_lead[s], "regime": REGIME_NAMES[r],
                        "exp": exp, "var": var,
                        "ctrl_bias": d["ctrl_b"][s, r], "ins_bias": d["ins_b"][s, r],
                    })
        save_csv(pd.DataFrame(bias_rows),
                 OUT["aurora"] / "intermediates" / f"aurora_{exp}_regime_bias.csv")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – FCN
# ─────────────────────────────────────────────────────────────────────────────
log("\n=== FCN ===")

fcn_cos1d = cos_weights(721, 90.0, -90.0)
fcn_cos2d = fcn_cos1d[:, None] * np.ones((1, 1440))
fcn_lead  = np.arange(N_AUR_FCN) * 0.5

FCN_CTRL_FILE = FCN_DIR / "forecast_normal_60days_12hr.nc"
FCN_EXP_FILES = {
    "t_ins":   FCN_DIR / "Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc",
    "uv_ins":  FCN_DIR / "Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc",
    "geo_ins": FCN_DIR / "Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc",
}

# Map variable → (ch_idx, truth_array, truth_idx_list)
FCN_VAR_MAP = {
    "T500": (FCN_CH["t500"], era5_t500, fcn_t_idx),
    "U500": (FCN_CH["u500"], era5_u500, fcn_t_idx),
    "V500": (FCN_CH["v500"], era5_v500, fcn_t_idx),
    "Z500": (FCN_CH["z500"], era5_z500, fcn_t_idx),
    "T2M":  (FCN_CH["t2m"],  era5_t2m,  fcn_s_idx),
}
if HAS_TCWV and era5_tcwv is not None:
    FCN_VAR_MAP["TCWV"] = (FCN_CH["tcwv"], era5_tcwv, fcn_s_idx)

fcn_results = {}
for exp, fpath in FCN_EXP_FILES.items():
    log(f"  FCN {exp}…")
    fcn_results[exp] = {}
    ctrl_ds = xr.open_dataset(str(FCN_CTRL_FILE), engine="netcdf4")
    ins_ds  = xr.open_dataset(str(fpath),          engine="netcdf4")

    for var, (ch, truth_arr, tidx) in FCN_VAR_MAP.items():
        # Lazy-slice FIRST, then .values — avoids loading full 36 GB array
        ctrl_ch = ctrl_ds["prediction"][:, ch, :, :].values.astype(np.float32)
        ins_ch  = ins_ds["prediction"][:,  ch, :, :].values.astype(np.float32)
        cr, ir = compute_regime_rmse_aurora(
            var, ctrl_ch, ins_ch, truth_arr, tidx, fcn_regime, fcn_cos2d)
        cb, ib = compute_regime_bias_aurora(
            var, ctrl_ch, ins_ch, truth_arr, tidx, fcn_regime, fcn_cos2d)
        fcn_results[exp][var] = {"ctrl_r": cr, "ins_r": ir, "ctrl_b": cb, "ins_b": ib}
        del ctrl_ch, ins_ch; gc.collect()

    ctrl_ds.close(); ins_ds.close()

    # UV500
    if "U500" in fcn_results[exp] and "V500" in fcn_results[exp]:
        ur = fcn_results[exp]["U500"]["ctrl_r"]
        vr = fcn_results[exp]["V500"]["ctrl_r"]
        fcn_results[exp]["UV500"] = {
            "ctrl_r": np.sqrt((ur**2 + vr**2) / 2),
            "ins_r":  np.sqrt((fcn_results[exp]["U500"]["ins_r"]**2 +
                               fcn_results[exp]["V500"]["ins_r"]**2) / 2),
            "ctrl_b": (fcn_results[exp]["U500"]["ctrl_b"] +
                       fcn_results[exp]["V500"]["ctrl_b"]) / 2,
            "ins_b":  (fcn_results[exp]["U500"]["ins_b"] +
                       fcn_results[exp]["V500"]["ins_b"]) / 2,
        }

# Save FCN CSVs
for exp in FCN_EXP_FILES:
    frames = []
    for var in fcn_results[exp]:
        d = fcn_results[exp][var]
        df = rmse_df(d["ctrl_r"], d["ins_r"], fcn_lead, exp, var)
        frames.append(df)
    if frames:
        save_csv(pd.concat(frames), OUT["fcn"] / "intermediates" / f"fcn_{exp}_regime_rmse.csv")

# Free ERA5 truth now (no longer needed after FCN)
del era5_t500, era5_u500, era5_v500, era5_z500, era5_q700, era5_t2m
if HAS_TCWV and era5_tcwv is not None:
    del era5_tcwv
del era5_t500_720, era5_u500_720, era5_v500_720, era5_z500_720
del era5_q700_720, era5_t2m_720, era5_tcwv_720
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – PRITHVI
# ─────────────────────────────────────────────────────────────────────────────
log("\n=== PRITHVI ===")

pri_lead = np.arange(N_PRI) * 0.25   # lead days at 6h steps

PRI_CTRL_FILE = PRI_DIR / "run2/output/osse_forecast_Jan2022.nc"
PRI_EXP_FILES = {
    "t_ins":   PRI_DIR / "t_sp_ins/output/t_sp_osse_forecast_Jan2022.nc",
    "uv_ins":  PRI_DIR / "uv_ins_run/output/uv_osse_forecast_Jan2022.nc",
    "geo_ins": PRI_DIR / "geo_ins_run/output/geo_osse_forecast_Jan2022.nc",
}

def load_pri_var(fpath, var, lev_idx=None):
    """Load Prithvi variable; lev_idx selects pressure level (ascending lev)."""
    ds = xr.open_dataset(str(fpath), engine="netcdf4")
    if lev_idx is not None:
        arr = ds[var].values[:, lev_idx, :, :]   # (240, 360, 576)
    else:
        arr = ds[var].values                      # (240, 360, 576) surface
    ds.close()
    return arr.astype(np.float32)

pri_ctrl = PRI_CTRL_FILE

pri_results = {}
for exp, fpath in PRI_EXP_FILES.items():
    log(f"  Prithvi {exp}…")
    pri_results[exp] = {}

    for var, ctrl_arr, ins_arr, truth_arr in [
        ("T500",  load_pri_var(pri_ctrl, "T", PRI_500_IDX),
                  load_pri_var(fpath,    "T", PRI_500_IDX), m2_T500),
        ("U500",  load_pri_var(pri_ctrl, "U", PRI_500_IDX),
                  load_pri_var(fpath,    "U", PRI_500_IDX), m2_U500),
        ("V500",  load_pri_var(pri_ctrl, "V", PRI_500_IDX),
                  load_pri_var(fpath,    "V", PRI_500_IDX), m2_V500),
        ("Z500",  load_pri_var(pri_ctrl, "H", PRI_500_IDX),
                  load_pri_var(fpath,    "H", PRI_500_IDX), m2_Z500),
        ("Q700",  load_pri_var(pri_ctrl, "QV", PRI_700_IDX),
                  load_pri_var(fpath,    "QV", PRI_700_IDX), m2_Q700),
        ("T2M",   load_pri_var(pri_ctrl, "T2M"),
                  load_pri_var(fpath,    "T2M"), m2_T2M),
        ("TQL",   load_pri_var(pri_ctrl, "TQL"),
                  load_pri_var(fpath,    "TQL"), m2_TQL),
        ("TQI",   load_pri_var(pri_ctrl, "TQI"),
                  load_pri_var(fpath,    "TQI"), m2_TQI),
        ("TQV",   load_pri_var(pri_ctrl, "TQV"),
                  load_pri_var(fpath,    "TQV"), m2_TQV),
        ("CLOUD", load_pri_var(pri_ctrl, "CLOUD", PRI_500_IDX),
                  load_pri_var(fpath,    "CLOUD", PRI_500_IDX), m2_CLOUD),
    ]:
        cr, ir = compute_regime_rmse_prithvi(ctrl_arr, ins_arr, truth_arr,
                                              pri_regime, pri_cos2d)
        cb, ib = compute_regime_bias_prithvi(ctrl_arr, ins_arr, truth_arr,
                                              pri_regime, pri_cos2d)
        pri_results[exp][var] = {"ctrl_r": cr, "ins_r": ir, "ctrl_b": cb, "ins_b": ib}
        del ctrl_arr, ins_arr; gc.collect()

    # UV500
    if "U500" in pri_results[exp] and "V500" in pri_results[exp]:
        ur = pri_results[exp]["U500"]["ctrl_r"]
        vr = pri_results[exp]["V500"]["ctrl_r"]
        pri_results[exp]["UV500"] = {
            "ctrl_r": np.sqrt((ur**2 + vr**2) / 2),
            "ins_r":  np.sqrt((pri_results[exp]["U500"]["ins_r"]**2 +
                               pri_results[exp]["V500"]["ins_r"]**2) / 2),
            "ctrl_b": (pri_results[exp]["U500"]["ctrl_b"] +
                       pri_results[exp]["V500"]["ctrl_b"]) / 2,
            "ins_b":  (pri_results[exp]["U500"]["ins_b"] +
                       pri_results[exp]["V500"]["ins_b"]) / 2,
        }

for exp in PRI_EXP_FILES:
    frames = [rmse_df(pri_results[exp][v]["ctrl_r"],
                      pri_results[exp][v]["ins_r"],
                      pri_lead, exp, v)
              for v in pri_results[exp]]
    if frames:
        save_csv(pd.concat(frames),
                 OUT["prithvi"] / "intermediates" / f"prithvi_{exp}_regime_rmse.csv")

del m2_T500, m2_U500, m2_V500, m2_Z500, m2_Q700
del m2_T2M, m2_TQL, m2_TQI, m2_TQV, m2_CLOUD
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_LABELS = {"aurora": "Aurora", "fcn": "FourCastNet", "prithvi": "Prithvi-WxC"}
EXP_LABELS   = {"t_ins": "Temp Ins.", "uv_ins": "Wind Ins.", "geo_ins": "Geo Ins."}

def phase_improvement(results, exp, var, adj_sl, mat_sl):
    """Return (adj_imp, mat_imp) each shape (N_REG,) as % improvement."""
    if exp not in results or var not in results[exp]:
        return np.full(N_REG, np.nan), np.full(N_REG, np.nan)
    d = results[exp][var]
    ctrl_adj = np.nanmean(d["ctrl_r"][adj_sl], axis=0)
    ins_adj  = np.nanmean(d["ins_r"][adj_sl],  axis=0)
    ctrl_mat = np.nanmean(d["ctrl_r"][mat_sl], axis=0)
    ins_mat  = np.nanmean(d["ins_r"][mat_sl],  axis=0)
    return improvement_pct(ctrl_adj, ins_adj), improvement_pct(ctrl_mat, ins_mat)


def phase_bias(results, exp, var, phase_sl):
    if exp not in results or var not in results[exp]:
        return np.full(N_REG, np.nan), np.full(N_REG, np.nan)
    d = results[exp][var]
    cb = np.nanmean(d["ctrl_b"][phase_sl], axis=0)
    ib = np.nanmean(d["ins_b"][phase_sl],  axis=0)
    return cb, ib


def bar_group_ax(ax, adj_imp, mat_imp, title, ylabel=True):
    """Plot grouped bars (Adj / Mature) for 4 regimes."""
    x   = np.arange(N_REG)
    w   = 0.35
    b1  = ax.bar(x - w/2, adj_imp, w, label="Adj. Phase",
                 color="#2171b5", edgecolor="white", linewidth=0.5)
    b2  = ax.bar(x + w/2, mat_imp, w, label="Mature Phase",
                 color="#cb181d", edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(REGIME_NAMES, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ylabel: ax.set_ylabel("RMSE Improvement (%)")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5.74 – Cloud Regime Masks + Occurrence Frequencies
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Fig 5.74 ---")

PROJ  = ccrs.Robinson()
TRANS = ccrs.PlateCarree()

def fig74_model(model, freq4d, occ, lats, lons, out_dir, period_label):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Fig 5.74 – Cloud-Regime Occurrence Frequency\n"
                 f"{MODEL_LABELS[model]}  |  {period_label}",
                 fontsize=14, fontweight="bold", y=0.98)

    cmaps  = ["Blues", "Greens", "Oranges", "Reds"]
    titles = ["Clear  (TCC < 10%)", "Cirrus  (10–40%)",
              "Mixed  (40–70%)",    "Opaque (>70%)"]

    axes_map = []
    for i in range(N_REG):
        ax = fig.add_subplot(3, 2, i + 1, projection=PROJ)
        axes_map.append(ax)
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="k")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, color="gray")
        im = ax.pcolormesh(lons, lats, freq4d[i] * 100,
                           transform=TRANS, cmap=cmaps[i],
                           vmin=0, vmax=80 if i > 0 else 100)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04,
                     shrink=0.7, label="Occurrence (%)")
        ax.set_title(titles[i], fontsize=11)

    # Bar chart of global occurrence
    ax_bar = fig.add_subplot(3, 2, 5)
    bars = ax_bar.barh(REGIME_NAMES, occ * 100, color=REGIME_COLORS, edgecolor="k", lw=0.5)
    for bar, val in zip(bars, occ * 100):
        ax_bar.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=9)
    ax_bar.set_xlabel("Global Occurrence (%)")
    ax_bar.set_title("Global Regime Frequency", fontsize=11)
    ax_bar.set_xlim(0, 55)
    ax_bar.grid(axis="x", alpha=0.3)

    # Pie chart
    ax_pie = fig.add_subplot(3, 2, 6)
    ax_pie.pie(occ, labels=REGIME_NAMES, colors=REGIME_COLORS,
               autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 9})
    ax_pie.set_title("Regime Composition", fontsize=11)

    plt.tight_layout()
    out_path = out_dir / "fig5_74_cloud_regime_masks.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log(f"  saved → {out_path.name}")


aur_lats = np.linspace(90.0, -89.75, 720)
aur_lons = np.linspace(0, 359.75, 1440)
fcn_lats = np.linspace(90.0, -90.0,  721)
fcn_lons = np.linspace(0, 359.75, 1440)
pri_lats = np.linspace(-89.5, 90.0, 360)
pri_lons = np.linspace(0, 359.375, 576)

fig74_model("aurora",  aur_regime_freq, aur_occ,
            aur_lats, aur_lons, OUT["aurora"],  "Feb–Apr 2023")
fig74_model("fcn",     fcn_regime_freq, fcn_occ,
            fcn_lats, fcn_lons, OUT["fcn"],     "Feb–Apr 2023")
fig74_model("prithvi", pri_regime_freq, pri_occ,
            pri_lats, pri_lons, OUT["prithvi"], "Jan–Mar 2022")

# Free regime frequency arrays
del aur_regime_freq, fcn_regime_freq, pri_regime_freq
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5.75 – T Insertion: RMSE Improvement by Regime (Adj vs Mature)
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Fig 5.75 ---")

def fig75_or_77(exp_key, primary_var, cross_var, fig_num, title_prefix,
                all_results, adj_slices, mat_slices, out_dirs):
    """
    Grouped bar chart: RMSE improvement (%) per cloud regime.
    Primary and cross variables shown side by side.
    Rows = models, Cols = primary / cross / UV500.
    """
    model_keys = ["aurora", "fcn", "prithvi"]
    col_vars   = [primary_var, cross_var, "UV500"]
    col_labels = [primary_var, cross_var, "UV500"]

    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharey=False)
    fig.suptitle(
        f"Fig {fig_num} – {title_prefix}: RMSE Improvement (%) by Cloud Regime",
        fontsize=14, fontweight="bold")

    for row, model in enumerate(model_keys):
        res   = all_results[model]
        adj_s = adj_slices[model]
        mat_s = mat_slices[model]
        for col, var in enumerate(col_vars):
            ax  = axes[row, col]
            adj_imp, mat_imp = phase_improvement(res, exp_key, var, adj_s, mat_s)
            title = f"{MODEL_LABELS[model]} – {var}"
            bar_group_ax(ax, adj_imp, mat_imp, title, ylabel=(col == 0))

    for col, lbl in enumerate(col_labels):
        axes[0, col].set_title(f"{lbl}\n{axes[0,col].get_title()}", fontsize=9)

    plt.tight_layout()
    # Save in all 3 model dirs
    for model, out_dir in out_dirs.items():
        path = out_dir / f"fig{fig_num.replace('.','_')}_rmse_improvement_{exp_key}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        log(f"  saved → {out_dir.name}/{path.name}")
    plt.close(fig)


ALL_RESULTS = {
    "aurora":  aur_results,
    "fcn":     fcn_results,
    "prithvi": pri_results,
}
ADJ_SLICES = {"aurora": AUR_FCN_ADJ, "fcn": AUR_FCN_ADJ, "prithvi": PRI_ADJ}
MAT_SLICES = {"aurora": AUR_FCN_MAT, "fcn": AUR_FCN_MAT, "prithvi": PRI_MAT}

fig75_or_77("t_ins",  "T500", "UV500", "5.75", "Temperature Insertion",
            ALL_RESULTS, ADJ_SLICES, MAT_SLICES, OUT)
fig75_or_77("uv_ins", "UV500", "T500", "5.77", "Wind Insertion",
            ALL_RESULTS, ADJ_SLICES, MAT_SLICES, OUT)
fig75_or_77("geo_ins","Z500", "T500",  "5.79", "Geopotential Insertion",
            ALL_RESULTS, ADJ_SLICES, MAT_SLICES, OUT)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5.76 / 5.78 / 5.80 – RMSE Time Series by Regime
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Figs 5.76, 5.78, 5.80 ---")

def fig_timeseries(exp_key, vars_list, fig_num, title_prefix,
                   all_results, lead_days_map, inj_steps_map, out_dirs):
    """
    4 regimes × 3 models × len(vars_list) panels.
    Layout: rows=regimes, cols=models.  One figure per variable.
    """
    model_keys = ["aurora", "fcn", "prithvi"]

    for var in vars_list:
        fig, axes = plt.subplots(N_REG, 3, figsize=(20, 14), sharex=False, sharey=False)
        fig.suptitle(
            f"Fig {fig_num} – {title_prefix}: {var} RMSE Time Series by Cloud Regime",
            fontsize=13, fontweight="bold")

        for row, regime in enumerate(REGIME_NAMES):
            for col, model in enumerate(model_keys):
                ax   = axes[row, col]
                res  = all_results[model]
                ld   = lead_days_map[model]
                inj  = inj_steps_map[model]

                if exp_key not in res or var not in res[exp_key]:
                    ax.set_visible(False)
                    continue

                d  = res[exp_key][var]
                cr = d["ctrl_r"][:, row]    # (N_steps,)
                ir = d["ins_r"][:, row]     # (N_steps,)

                ax.plot(ld, cr, color="gray",            lw=1.5,
                        label="Control", linestyle="--")
                ax.plot(ld, ir, color=REGIME_COLORS[row], lw=2.0,
                        label="Insertion")
                ax.axvline(ld[inj], color="k", lw=1.0, linestyle=":",
                           label="Injection start")
                ax.fill_betweenx(
                    [ax.get_ylim()[0] if ax.get_ylim()[0] < np.nanmin(ir)*0.9
                     else 0, np.nanmax(cr)*1.1],
                    ld[inj], ld[min(inj + 30, len(ld)-1)],
                    alpha=0.08, color="#2171b5", label="Adj. Phase")

                if row == 0:
                    ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"{regime}\nRMSE", fontsize=9)
                ax.set_xlabel("Lead Day" if row == N_REG-1 else "")
                ax.legend(fontsize=7, framealpha=0.6)
                ax.grid(alpha=0.25)

        plt.tight_layout()
        var_safe = var.lower()
        for model, out_dir in out_dirs.items():
            path = out_dir / f"fig{fig_num.replace('.','_')}_timeseries_{exp_key}_{var_safe}.png"
            fig.savefig(path, bbox_inches="tight", dpi=300)
            log(f"  saved → {path.name}")
        plt.close(fig)


LEAD_MAP = {"aurora": np.arange(N_AUR_FCN)*0.5,
            "fcn":    np.arange(N_AUR_FCN)*0.5,
            "prithvi": np.arange(N_PRI)*0.25}
INJ_MAP  = {"aurora": AUR_FCN_INJ, "fcn": AUR_FCN_INJ, "prithvi": PRI_INJ}

fig_timeseries("t_ins",   ["T500", "UV500", "TCWV"],
               "5.76", "Temp Insertion",
               ALL_RESULTS, LEAD_MAP, INJ_MAP, OUT)

fig_timeseries("uv_ins",  ["UV500", "T500", "TCWV"],
               "5.78", "Wind Insertion",
               ALL_RESULTS, LEAD_MAP, INJ_MAP, OUT)

fig_timeseries("geo_ins", ["Z500", "T500", "TCWV"],
               "5.80", "Geo Insertion",
               ALL_RESULTS, LEAD_MAP, INJ_MAP, OUT)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5.81 – Mean Bias by Regime (T, moisture, cloud)
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Fig 5.81 ---")

def fig81_bias(all_results, adj_slices, mat_slices, out_dirs):
    exps = ["t_ins", "uv_ins", "geo_ins"]
    vars_bias = ["T500", "Q700", "T2M"]
    model_keys = ["aurora", "fcn", "prithvi"]
    # Use Mature phase bias
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle("Fig 5.81 – Mean Bias by Cloud Regime (Mature Phase)\n"
                 "T500 | Q700 | T2M  ×  Temp / Wind / Geo Insertion",
                 fontsize=13, fontweight="bold")

    for row, exp in enumerate(exps):
        for col, var in enumerate(vars_bias):
            ax = axes[row, col]
            x  = np.arange(N_REG)
            w  = 0.22

            for mi, model in enumerate(model_keys):
                res   = all_results[model]
                mat_s = mat_slices[model]
                cb, ib = phase_bias(res, exp, var, mat_s)
                offset = (mi - 1) * w
                ax.bar(x + offset, ib - cb, w,
                       label=MODEL_LABELS[model] if row == 0 else "_nolegend_",
                       color=["#2166ac", "#d6604d", "#4dac26"][mi],
                       edgecolor="white", lw=0.4, alpha=0.85)

            ax.axhline(0, color="k", lw=0.8, linestyle="--")
            ax.set_xticks(x); ax.set_xticklabels(REGIME_NAMES, fontsize=8)
            if col == 0: ax.set_ylabel(f"{EXP_LABELS[exp]}\nBias Δ (Ins–Ctrl)")
            if row == 0: ax.set_title(var, fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=c, label=MODEL_LABELS[m])
               for m, c in zip(model_keys, ["#2166ac","#d6604d","#4dac26"])]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    plt.tight_layout()
    for model, out_dir in out_dirs.items():
        path = out_dir / "fig5_81_mean_bias_by_regime.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        log(f"  saved → {path.name}")
    plt.close(fig)

fig81_bias(ALL_RESULTS, ADJ_SLICES, MAT_SLICES, OUT)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5.82 – Ice & Liquid Cloud Response by Regime
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Fig 5.82 ---")

def fig82_cloud_response(all_results, adj_slices, mat_slices, out_dirs):
    """
    For Prithvi: TQL and TQI anomaly (Ins – Ctrl) per regime.
    For Aurora/FCN: Q700 anomaly (proxy).
    """
    exps       = ["t_ins", "uv_ins", "geo_ins"]
    model_keys = ["aurora", "fcn", "prithvi"]

    # Variable to use per model for cloud condensate proxy
    cloud_vars = {"aurora": "Q700", "fcn": "Q700", "prithvi": "TQL"}
    ice_vars   = {"aurora": None,   "fcn": None,    "prithvi": "TQI"}

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle("Fig 5.82 – Cloud Condensate Response by Regime (Mature Phase)\n"
                 "Liquid / Q700 proxy (solid) | Ice TQI (hatched, Prithvi only)",
                 fontsize=13, fontweight="bold")

    for row, exp in enumerate(exps):
        for col, model in enumerate(model_keys):
            ax  = axes[row, col]
            res = all_results[model]
            mat = mat_slices[model]
            x   = np.arange(N_REG)
            w   = 0.35

            # Liquid/proxy
            cv = cloud_vars[model]
            cb_liq, ib_liq = phase_bias(res, exp, cv, mat)
            delta_liq = ib_liq - cb_liq
            ax.bar(x - w/2, delta_liq, w, color="#4393c3",
                   edgecolor="white", lw=0.5, label=f"{cv} Δ bias")

            # Ice (Prithvi only)
            iv = ice_vars[model]
            if iv is not None:
                cb_ice, ib_ice = phase_bias(res, exp, iv, mat)
                delta_ice = ib_ice - cb_ice
                ax.bar(x + w/2, delta_ice, w, color="#d6604d",
                       edgecolor="white", lw=0.5, hatch="///",
                       label="TQI Δ bias")

            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xticks(x); ax.set_xticklabels(REGIME_NAMES, fontsize=8)
            if col == 0: ax.set_ylabel(f"{EXP_LABELS[exp]}\nBias Δ (Ins–Ctrl)")
            if row == 0: ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for model, out_dir in out_dirs.items():
        path = out_dir / "fig5_82_cloud_response_by_regime.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        log(f"  saved → {path.name}")
    plt.close(fig)

fig82_cloud_response(ALL_RESULTS, ADJ_SLICES, MAT_SLICES, OUT)

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5.14 – Cloud-Regime Summary
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Table 5.14 ---")

exp_var_map = {
    "t_ins":   [("T500", "Direct"), ("UV500", "Cross"), ("Q700", "Moisture"), ("TQL", "Cloud-liq")],
    "uv_ins":  [("UV500","Direct"), ("T500", "Cross"),  ("Q700", "Moisture"), ("TQI", "Cloud-ice")],
    "geo_ins": [("Z500", "Direct"), ("T500", "Cross"),  ("Q700", "Moisture"), ("CLOUD","Cloud-frac")],
}

rows = []
for model_key in ["aurora", "fcn", "prithvi"]:
    res    = ALL_RESULTS[model_key]
    adj_s  = ADJ_SLICES[model_key]
    mat_s  = MAT_SLICES[model_key]
    for exp, var_list in exp_var_map.items():
        for var, role in var_list:
            if exp not in res or var not in res[exp]:
                continue
            adj_imp, mat_imp = phase_improvement(res, exp, var, adj_s, mat_s)
            for r in range(N_REG):
                best_phase = "Mature" if mat_imp[r] > adj_imp[r] else "Adjustment"
                rows.append({
                    "Model":            MODEL_LABELS[model_key],
                    "Experiment":       EXP_LABELS[exp],
                    "Variable":         var,
                    "Role":             role,
                    "Regime":           REGIME_NAMES[r],
                    "Adj_Improvement%": round(float(adj_imp[r]), 3),
                    "Mat_Improvement%": round(float(mat_imp[r]), 3),
                    "Peak_Phase":       best_phase,
                    "Max_Improvement%": round(float(max(adj_imp[r], mat_imp[r])), 3),
                })

table_df = pd.DataFrame(rows)
for model, out_dir in OUT.items():
    save_csv(table_df[table_df["Model"] == MODEL_LABELS[model]],
             out_dir / "table5_14_cloud_regime_summary.csv")
save_csv(table_df, TCC_DIR / "table5_14_ALL_MODELS.csv")

log("\n=== All figures and tables complete ===")
log(f"Outputs in {TCC_DIR}/{{aurora,fcn,prithvi}}/")
