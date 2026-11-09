#!/usr/bin/env python3
"""
Recompute Prithvi global RMSE timeseries for TUV study with the corrected
Prithvi-position -> MERRA-position level mapping (see compute_tuv_rmse.py
for the bug this fixes). Only touches the prithvi_*.csv / prithvi_*.npy
cache files -- FCN and Aurora caches are untouched.
"""
import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta

import h5py, numpy as np, pandas as pd

CACHE = Path("/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/cache")
def L(msg): print(msg, flush=True)

def make_weights(lats_1d, nlon):
    w = np.cos(np.deg2rad(lats_1d))
    return (w[:, None] * np.ones((1, nlon))).astype(np.float32)

def global_rmse_ts(pred, truth, W):
    sq = (pred.astype(np.float32) - truth.astype(np.float32)) ** 2
    return np.sqrt((sq * W).sum(axis=(-2, -1)) / W.sum()).astype(np.float64)

MERRA2_DIR = Path("/umbc/rs/pi_halem/users/yashd1/prithviOSSE/data/merra2")

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
        except Exception:
            pass
    L(f"    MERRA2 {mvar} lev_idx={lev_idx}: {int(np.sum(~np.all(np.isnan(result),axis=(-2,-1))))}/{N}")
    return result

BASE_PRI = Path("/umbc/rs/pi_halem/users/yashd1/prithviOSSE")
PRI_TUV  = BASE_PRI/"t_and_uv/output/t_and_uv_osse_forecast_Jan2022.nc"
PRI_CTRL = Path("/umbc/rs/pi_halem/users/yashd1/prithviOSSE/run2/output/control_forecast_Jan2022.nc")
PRI_TINS = BASE_PRI/"run2/output/osse_forecast_Jan2022.nc"
PRI_UINS = BASE_PRI/"uv_ins_run/output/uv_osse_forecast_Jan2022.nc"

NLAT_P, NLON_P = 360, 576
W_P = make_weights(np.linspace(89.5,-89.5,NLAT_P), NLON_P)

PRI_IC     = datetime(2022,1,1,0,0)
pri_times  = [PRI_IC + timedelta(hours=6*k) for k in range(240)]
pri_lead   = np.array([6*k/24 for k in range(240)])

PRI_VARS = {
    "T_lv7" : ("T",  7),
    "U_lv7" : ("U",  7),
    "V_lv7" : ("V",  7),
    "H_lv5" : ("H",  5),
    "H_lv7" : ("H",  7),
    "H_lv12": ("H", 12),
    "QV_lv7": ("QV", 7),
    "QV_lv10":("QV",10),
}

def load_pri(nc_file, nc_var, lev_idx, nlat=NLAT_P):
    with h5py.File(str(nc_file), "r") as f:
        arr = f[nc_var][:, lev_idx, -nlat:, :]
    return arr.astype(np.float32)

rows_p = {"tuv":{"lead_day":pri_lead},"ctrl":{"lead_day":pri_lead},
          "tins":{"lead_day":pri_lead},"uins":{"lead_day":pri_lead}}

L("="*70 + "\nPRITHVI TUV (corrected MERRA level mapping)\n" + "="*70)

for vkey,(mvar,lidx) in PRI_VARS.items():
    L(f"\nPrithvi {vkey}  (Prithvi lev_idx={lidx} -> level {LEVELS_ASC[lidx]} -> "
      f"MERRA lev_idx={pri_idx_to_merra_idx(lidx)})")
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
L("\nPrithvi CSVs re-saved with corrected MERRA level mapping.")
