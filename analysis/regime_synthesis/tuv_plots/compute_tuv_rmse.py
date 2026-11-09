#!/usr/bin/env python3
"""
Compute global RMSE timeseries + spatial synergy fields for T+UV insertion.
Saves cache files to tuv_plots/cache/ for the plotting script.
Models: FCN and Aurora (ERA5 truth), Prithvi (MERRA-2 truth).
"""
import warnings; warnings.filterwarnings("ignore")
import os, gc
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
from datetime import datetime, timedelta

import h5py, numpy as np, pandas as pd, xarray as xr

CACHE = Path("/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/cache")
CACHE.mkdir(parents=True, exist_ok=True)
def L(msg): print(msg, flush=True)

# ── Grid weights ───────────────────────────────────────────────────────────────
def make_weights(lats_1d, nlon):
    w = np.cos(np.deg2rad(lats_1d))
    return (w[:, None] * np.ones((1, nlon))).astype(np.float32)

def global_rmse_ts(pred, truth, W):
    """(T,lat,lon) arrays → (T,) global cos-lat-weighted RMSE."""
    sq   = (pred.astype(np.float32) - truth.astype(np.float32)) ** 2
    return np.sqrt((sq * W).sum(axis=(-2, -1)) / W.sum()).astype(np.float64)

def spatial_impr_pct(ctrl_arr, exp_arr, truth_arr, time_mask):
    """Mature-phase spatial improvement % at each grid point (lat,lon)."""
    ctrl_mse = np.nanmean((ctrl_arr[time_mask] - truth_arr[time_mask])**2, axis=0)
    exp_mse  = np.nanmean((exp_arr[time_mask]  - truth_arr[time_mask])**2, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ctrl_mse > 0, (ctrl_mse - exp_mse) / ctrl_mse * 100, np.nan)

# ── ERA5 loader ────────────────────────────────────────────────────────────────
def load_era5(grib_files, shortname, level, target_times, nlat_out):
    N = len(target_times)
    result = np.full((N, nlat_out, 1440), np.nan, dtype=np.float32)
    for fpath in grib_files:
        if not Path(fpath).exists(): continue
        try:
            ds = xr.open_dataset(str(fpath), engine="cfgrib",
                                 filter_by_keys={"shortName": shortname,
                                                 "typeOfLevel": "isobaricInhPa",
                                                 "level": int(level)},
                                 backend_kwargs={"indexpath": ""})
            ds = ds.drop_vars(["step","isobaricInhPa","valid_time"], errors="ignore")
            da = ds[list(ds.data_vars)[0]]
        except Exception as e:
            L(f"  SKIP {Path(fpath).name} {shortname}@{level}: {e}"); continue
        era5_ts = pd.to_datetime(da.time.values)
        if da.latitude.values[0] < da.latitude.values[-1]:
            da = da.isel(latitude=slice(None, None, -1))
        arr = da.values.astype(np.float32)
        da.close(); gc.collect()
        for k, tgt in enumerate(target_times):
            if not np.all(np.isnan(result[k])): continue
            deltas = np.abs(era5_ts - pd.Timestamp(tgt))
            idx = int(deltas.argmin())
            if deltas[idx].total_seconds() <= 3601:
                row = arr[idx]
                if row.shape[0] == nlat_out + 1: row = row[1:]   # 721→720
                result[k] = row
        del arr; gc.collect()
    matched = int(np.sum(~np.all(np.isnan(result), axis=(-2, -1))))
    L(f"    ERA5 {shortname}@{level}: {matched}/{N} matched")
    return result

def era5_rh(grib_files, level, target_times, nlat_out):
    """Derive relative humidity from ERA5 q and T."""
    q = load_era5(grib_files, "q", level, target_times, nlat_out)
    T = load_era5(grib_files, "t", level, target_times, nlat_out)
    Tc    = T - 273.15
    e_sat = 6.1078 * np.exp(17.2694 * Tc / (Tc + 237.29))
    e_act = q * level / (0.622 + 0.378 * q)
    return np.clip(100.0 * e_act / e_sat, 0, 100).astype(np.float32)

# ── MERRA-2 loader ─────────────────────────────────────────────────────────────
MERRA2_DIR = Path("/umbc/rs/pi_halem/users/yashd1/prithviOSSE/data/merra2")

# Prithvi stores its 14 levels ASCENDING by level-ID; MERRA-2 pressure files
# store the same 14 levels DESCENDING. A Prithvi array position must be mapped
# through the level ID to find the matching MERRA array position -- passing
# the Prithvi position straight into MERRA's array (as done previously) reads
# the wrong pressure level (see section_5_7_prithvi_latband_analysis.py for
# the reference implementation of this mapping).
LEVELS_ASC     = [34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72]
MERRA_LEV_DESC = [72, 71, 68, 63, 56, 53, 51, 48, 45, 44, 43, 41, 39, 34]

def pri_idx_to_merra_idx(pri_lev_idx):
    return MERRA_LEV_DESC.index(LEVELS_ASC[pri_lev_idx])

def load_merra2(mvar, lev_idx, target_times, nlat=360, nlon=576):
    N = len(target_times)
    result = np.full((N, nlat, nlon), np.nan, dtype=np.float32)
    for k, tgt in enumerate(target_times):
        fpath = MERRA2_DIR / f"MERRA_pres_{tgt.strftime('%Y%m%d')}.nc"
        if not fpath.exists(): continue
        try:
            with h5py.File(str(fpath), "r") as f:
                if mvar not in f: continue
                begin_date = int(f["time"].attrs["begin_date"][0])
                epoch      = pd.Timestamp(str(begin_date))
                times_day  = epoch + pd.to_timedelta(f["time"][:].astype(int), unit="min")
                deltas     = np.abs(times_day - pd.Timestamp(tgt))
                idx = int(deltas.argmin())
                if deltas[idx].total_seconds() > 3601: continue
                result[k] = f[mvar][idx, lev_idx, -nlat:, :nlon].astype(np.float32)
        except Exception as e:
            pass
    L(f"    MERRA2 {mvar} lev={lev_idx}: {int(np.sum(~np.all(np.isnan(result),axis=(-2,-1))))}/{N}")
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# FCN
# ═══════════════════════════════════════════════════════════════════════════════
L("\n" + "="*70 + "\nFCN TUV\n" + "="*70)

FCN_TUV  = "/umbc/rs/pi_halem/users/yashd1/fcn_inference/tuv_global/forecast_tuv_global_day30_60.nc"
FCN_CTRL = "/umbc/rs/pi_halem/users/yashd1/FourCastNet/forecast_normal_60days_12hr.nc"
FCN_TINS = "/umbc/rs/pi_halem/users/yashd1/FourCastNet/Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc"
FCN_UINS = "/umbc/rs/pi_halem/users/yashd1/FourCastNet/Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc"
ERA5_23  = [f"/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data/era5_pressure_2023_{m:02d}.grib"
            for m in [2, 3, 4]]

# Channel indices
FCN_CH = {"t500":54,"u500":15,"v500":28,"z300":39,"z500":41,"z1000":46,"r850":70,"r500":67}
NLAT_F, NLON_F = 721, 1440
W_F = make_weights(np.linspace(90,-90,NLAT_F), NLON_F)

# Time axes
with h5py.File(FCN_TUV, "r") as _f:
    _t  = _f["time"][:]
    _un = _f["time"].attrs.get("units", b"hours since 1900-01-01 00:00:00")
    if isinstance(_un, bytes): _un = _un.decode()
    _epoch = pd.Timestamp(_un.split("since")[-1].strip())
    _unit  = _un.split("since")[0].strip()
    _td    = pd.to_timedelta(_t, unit=_unit[0])
fcn_tuv_times = [(_epoch + d).to_pydatetime() for d in _td]
fcn_tuv_lead  = np.array([(t - fcn_tuv_times[0]).total_seconds()/86400 for t in fcn_tuv_times])

FCN_IC12 = datetime(2023,2,1,0,0)
fcn_ctrl_times = [FCN_IC12 + timedelta(hours=12*k) for k in range(120)]
fcn_ctrl_lead  = np.array([0.5*k for k in range(120)])

# ERA5 variable specs: var → (shortname, level) or None for RH
FCN_ERA5_SPEC = {
    "t500":  ("t",500),  "u500": ("u",500), "v500": ("v",500),
    "z300":  ("z",300),  "z500": ("z",500), "z1000":("z",1000),
    "r850":  None,       "r500": None,  # derived
}

rows = {"tuv":{"lead_day":fcn_tuv_lead}, "ctrl":{"lead_day":fcn_ctrl_lead},
        "tins":{"lead_day":fcn_ctrl_lead}, "uins":{"lead_day":fcn_ctrl_lead}}

for var, ch in FCN_CH.items():
    spec = FCN_ERA5_SPEC[var]
    L(f"\nFCN {var}")
    if spec is None:
        lev = int(var[1:])
        e_tuv  = era5_rh(ERA5_23, lev, fcn_tuv_times,  NLAT_F)
        e_ctrl = era5_rh(ERA5_23, lev, fcn_ctrl_times, NLAT_F)
    else:
        sn, lev = spec
        e_tuv  = load_era5(ERA5_23, sn, lev, fcn_tuv_times,  NLAT_F)
        e_ctrl = load_era5(ERA5_23, sn, lev, fcn_ctrl_times, NLAT_F)

    with h5py.File(FCN_TUV,  "r") as f: tuv  = f["prediction"][:, ch, :, :].astype(np.float32)
    with h5py.File(FCN_CTRL, "r") as f: ctrl = f["prediction"][:, ch, :, :].astype(np.float32)
    with h5py.File(FCN_TINS, "r") as f: tins = f["prediction"][:, ch, :, :].astype(np.float32)
    with h5py.File(FCN_UINS, "r") as f: uins = f["prediction"][:, ch, :, :].astype(np.float32)

    rows["tuv"][var]  = global_rmse_ts(tuv,  e_tuv,  W_F)
    rows["ctrl"][var] = global_rmse_ts(ctrl, e_ctrl, W_F)
    rows["tins"][var] = global_rmse_ts(tins, e_ctrl, W_F)
    rows["uins"][var] = global_rmse_ts(uins, e_ctrl, W_F)

    if var in ("z500","r850"):
        mat_t = (fcn_tuv_lead  > 40) & (fcn_tuv_lead  <= 60)
        mat_c = (fcn_ctrl_lead > 40) & (fcn_ctrl_lead <= 60)
        ctrl_mse = np.nanmean((ctrl[mat_c]-e_ctrl[mat_c])**2, axis=0)
        tuv_mse  = np.nanmean((tuv[mat_t] -e_tuv[mat_t]  )**2, axis=0)
        tins_mse = np.nanmean((tins[mat_c]-e_ctrl[mat_c])**2, axis=0)
        uins_mse = np.nanmean((uins[mat_c]-e_ctrl[mat_c])**2, axis=0)
        with np.errstate(invalid='ignore',divide='ignore'):
            i_t   = np.where(ctrl_mse>0,(ctrl_mse-tins_mse)/ctrl_mse*100, np.nan)
            i_uv  = np.where(ctrl_mse>0,(ctrl_mse-uins_mse)/ctrl_mse*100, np.nan)
            i_tuv = np.where(ctrl_mse>0,(ctrl_mse-tuv_mse )/ctrl_mse*100, np.nan)
        syn = i_tuv - np.maximum(i_t, i_uv)
        np.save(CACHE/f"fcn_synergy_{var}.npy", syn)
        np.save(CACHE/f"fcn_impr_tuv_{var}.npy", i_tuv)
        L(f"  FCN {var} synergy mean={np.nanmean(syn):.2f}%  impr_tuv mean={np.nanmean(i_tuv):.2f}%")

    del tuv, ctrl, tins, uins, e_tuv, e_ctrl; gc.collect()

for key in rows:
    pd.DataFrame(rows[key]).to_csv(CACHE/f"fcn_{key}_rmse.csv", index=False)
L("FCN CSVs saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# AURORA
# ═══════════════════════════════════════════════════════════════════════════════
L("\n" + "="*70 + "\nAURORA TUV\n" + "="*70)

BASE_AUR = Path("/umbc/rs/pi_halem/users/yashd1")
AUR_TUV  = BASE_AUR/"CC_aurora/TUV_GLOBAL/output/aurora_tuv_global.nc"
AUR_CTRL = BASE_AUR/"auroraoutput_1year_1stfeb23.nc"
AUR_TINS = BASE_AUR/"Aurora_Pred_Files/auroraoutput_temp_inserted.nc"
AUR_UINS = BASE_AUR/"Aurora_Pred_Files/auroraoutput_uv_inserted.nc"

NLAT_A, NLON_A = 720, 1440
W_A = make_weights(np.linspace(89.875,-89.875,NLAT_A), NLON_A)

with h5py.File(str(AUR_TUV), "r") as f:
    base_sec  = int(f["base_time"][()])
    lead_hrs  = f["lead_time"][:]
aur_ic       = pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=base_sec)
aur_tuv_times= [(aur_ic + pd.Timedelta(hours=int(h))).to_pydatetime() for h in lead_hrs]
aur_tuv_lead = np.array(lead_hrs/24, dtype=np.float64)

AUR_IC12 = datetime(2023,2,1,12,0)
aur_ctrl_times = [AUR_IC12 + timedelta(hours=12*k) for k in range(120)]
aur_ctrl_lead  = np.array([0.5*k for k in range(120)])

AUR_VARS = {"t500":("t",500),"u500":("u",500),"v500":("v",500),
            "z300":("z",300),"z500":("z",500),"z1000":("z",1000),
            "q850":("q",850),"q500":("q",500)}

rows_a = {"tuv":{"lead_day":aur_tuv_lead},"ctrl":{"lead_day":aur_ctrl_lead},
          "tins":{"lead_day":aur_ctrl_lead},"uins":{"lead_day":aur_ctrl_lead}}

def load_aur(fpath, var, sl=None):
    with h5py.File(str(fpath),"r") as f:
        return (f[var][:] if sl is None else f[var][sl]).astype(np.float32)

for var,(sn,lev) in AUR_VARS.items():
    L(f"\nAurora {var}")
    e_tuv  = load_era5(ERA5_23, sn, lev, aur_tuv_times,  NLAT_A)
    e_ctrl = load_era5(ERA5_23, sn, lev, aur_ctrl_times, NLAT_A)
    tuv  = load_aur(AUR_TUV,  var)
    ctrl = load_aur(AUR_CTRL, var, slice(0,120))
    tins = load_aur(AUR_TINS, var)
    uins = load_aur(AUR_UINS, var)

    rows_a["tuv"][var]  = global_rmse_ts(tuv,  e_tuv,  W_A)
    rows_a["ctrl"][var] = global_rmse_ts(ctrl, e_ctrl, W_A)
    rows_a["tins"][var] = global_rmse_ts(tins, e_ctrl, W_A)
    rows_a["uins"][var] = global_rmse_ts(uins, e_ctrl, W_A)

    if var in ("z500","q850"):
        mat_t = (aur_tuv_lead  > 40) & (aur_tuv_lead  <= 60)
        mat_c = (aur_ctrl_lead > 40) & (aur_ctrl_lead <= 60)
        ctrl_mse = np.nanmean((ctrl[mat_c]-e_ctrl[mat_c])**2, axis=0)
        tuv_mse  = np.nanmean((tuv[mat_t] -e_tuv[mat_t]  )**2, axis=0)
        tins_mse = np.nanmean((tins[mat_c]-e_ctrl[mat_c])**2, axis=0)
        uins_mse = np.nanmean((uins[mat_c]-e_ctrl[mat_c])**2, axis=0)
        with np.errstate(invalid='ignore',divide='ignore'):
            i_t   = np.where(ctrl_mse>0,(ctrl_mse-tins_mse)/ctrl_mse*100,np.nan)
            i_uv  = np.where(ctrl_mse>0,(ctrl_mse-uins_mse)/ctrl_mse*100,np.nan)
            i_tuv = np.where(ctrl_mse>0,(ctrl_mse-tuv_mse )/ctrl_mse*100,np.nan)
        syn = i_tuv - np.maximum(i_t, i_uv)
        np.save(CACHE/f"aurora_synergy_{var}.npy", syn)
        np.save(CACHE/f"aurora_impr_tuv_{var}.npy", i_tuv)
        L(f"  Aurora {var} synergy mean={np.nanmean(syn):.2f}%")

    del tuv, ctrl, tins, uins, e_tuv, e_ctrl; gc.collect()

for key in rows_a:
    pd.DataFrame(rows_a[key]).to_csv(CACHE/f"aurora_{key}_rmse.csv", index=False)
L("Aurora CSVs saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# PRITHVI
# ═══════════════════════════════════════════════════════════════════════════════
L("\n" + "="*70 + "\nPRITHVI TUV\n" + "="*70)

BASE_PRI = Path("/umbc/rs/pi_halem/users/yashd1/prithviOSSE")
PRI_TUV  = BASE_PRI/"t_and_uv/output/t_and_uv_osse_forecast_Jan2022.nc"
PRI_CTRL = Path("/umbc/rs/pi_halem/users/yashd1/prithviOSSE/run2/output/control_forecast_Jan2022.nc")
# NOTE: prithvi/output/forecast_60day_Jan2022.nc is the OLD baseline, documented
# as corrupted (361 lat rows instead of 360 -- see run2/run_forecast.py docstring,
# "job 149657"). That off-by-one grid silently misaligns every comparison against
# the 360-row PRI_TINS/PRI_UINS/PRI_TUV files. Use the run2 control instead.
PRI_TINS = BASE_PRI/"run2/output/osse_forecast_Jan2022.nc"
PRI_UINS = BASE_PRI/"uv_ins_run/output/uv_osse_forecast_Jan2022.nc"

NLAT_P, NLON_P = 360, 576
W_P = make_weights(np.linspace(89.5,-89.5,NLAT_P), NLON_P)

PRI_IC     = datetime(2022,1,1,0,0)
pri_times  = [PRI_IC + timedelta(hours=6*k) for k in range(240)]
pri_lead   = np.array([6*k/24 for k in range(240)])

# Prithvi variable → (MERRA2_var, lev_idx): approximate pressure mappings
PRI_VARS = {
    "T_lv7" : ("T",  7),   # ~554 hPa ≈ 500 hPa
    "U_lv7" : ("U",  7),
    "V_lv7" : ("V",  7),
    "H_lv5" : ("H",  5),   # ~315 hPa ≈ 300 hPa
    "H_lv7" : ("H",  7),   # ~500 hPa
    "H_lv12": ("H", 12),   # ~1000 hPa
    "QV_lv7": ("QV", 7),   # ~500 hPa moisture
    "QV_lv10":("QV",10),   # ~850 hPa moisture
}
# Prithvi nc var → lev_idx mapping (same key structure)
PRI_NC_VAR = {k: k.split("_lv")[0] for k in PRI_VARS}

def load_pri(nc_file, nc_var, lev_idx, nlat=NLAT_P):
    with h5py.File(str(nc_file), "r") as f:
        arr = f[nc_var][:, lev_idx, -nlat:, :]
    return arr.astype(np.float32)

rows_p = {"tuv":{"lead_day":pri_lead},"ctrl":{"lead_day":pri_lead},
          "tins":{"lead_day":pri_lead},"uins":{"lead_day":pri_lead}}

for vkey,(mvar,lidx) in PRI_VARS.items():
    L(f"\nPrithvi {vkey}")
    truth = load_merra2(mvar, pri_idx_to_merra_idx(lidx), pri_times)
    nc_var = mvar
    tuv  = load_pri(PRI_TUV,  nc_var, lidx)
    ctrl = load_pri(PRI_CTRL, nc_var, lidx)
    tins = load_pri(PRI_TINS, nc_var, lidx)
    uins = load_pri(PRI_UINS, nc_var, lidx)

    rows_p["tuv"][vkey]  = global_rmse_ts(tuv,  truth, W_P)
    rows_p["ctrl"][vkey] = global_rmse_ts(ctrl, truth, W_P)
    rows_p["tins"][vkey] = global_rmse_ts(tins, truth, W_P)
    rows_p["uins"][vkey] = global_rmse_ts(uins, truth, W_P)

    if vkey in ("H_lv7","QV_lv10"):
        mat = (pri_lead > 40) & (pri_lead <= 60)
        ctrl_mse = np.nanmean((ctrl[mat]-truth[mat])**2, axis=0)
        tuv_mse  = np.nanmean((tuv[mat] -truth[mat])**2, axis=0)
        tins_mse = np.nanmean((tins[mat]-truth[mat])**2, axis=0)
        uins_mse = np.nanmean((uins[mat]-truth[mat])**2, axis=0)
        with np.errstate(invalid='ignore',divide='ignore'):
            i_t   = np.where(ctrl_mse>0,(ctrl_mse-tins_mse)/ctrl_mse*100,np.nan)
            i_uv  = np.where(ctrl_mse>0,(ctrl_mse-uins_mse)/ctrl_mse*100,np.nan)
            i_tuv = np.where(ctrl_mse>0,(ctrl_mse-tuv_mse )/ctrl_mse*100,np.nan)
        syn = i_tuv - np.maximum(i_t, i_uv)
        np.save(CACHE/f"prithvi_synergy_{vkey}.npy", syn)
        np.save(CACHE/f"prithvi_impr_tuv_{vkey}.npy", i_tuv)
        L(f"  Prithvi {vkey} synergy mean={np.nanmean(syn):.2f}%")

    del tuv, ctrl, tins, uins, truth; gc.collect()

for key in rows_p:
    pd.DataFrame(rows_p[key]).to_csv(CACHE/f"prithvi_{key}_rmse.csv", index=False)
L("Prithvi CSVs saved.")

L("\n" + "="*70)
L("ALL COMPUTATIONS COMPLETE → " + str(CACHE))
L("="*70)
