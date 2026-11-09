#!/usr/bin/env python3
"""
Global cosine-lat-weighted RMSE time series for the 3 perturbed-insertion runs.
Per pressure level, per forecast step — no averaging over levels or time.

Outputs:
  CC_aurora/T_PERTURBED/rmse_perturbed_temp.csv       cols: lead_hour, t50..t1000
  fcn_inference/t_perturbed/rmse_perturbed_temp.csv   cols: lead_hour, t50..t1000
  prithviOSSE/perturbed_temp_ins/rmse_perturbed_temp.csv  cols: lead_hour, t52hPa..t960hPa
"""

import warnings; warnings.filterwarnings("ignore")
import gc
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
import cftime as cft

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
ERA5_DIR  = BASE / "AURORA_DATA" / "rmse_data"
MERRA_DIR = BASE / "prithvi" / "data" / "merra2"

# Aurora / FCN pressure levels (hPa) and their ERA5 filter level values
AF_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Prithvi MERRA-2 level indices and their approximate hPa values
PRI_LEV_VALS = [34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72]
PRI_LEV_HPA  = [52, 118, 164, 226, 265, 310, 424, 532, 604, 693, 832, 903, 946, 960]


def _cos_w1d(lats):
    return np.cos(np.deg2rad(lats)).clip(min=0)


def global_rmse(fc, truth, cos_w1d):
    """cos_w1d: (n_lat,). fc/truth: (n_lat, n_lon)."""
    w2d = cos_w1d[:, None]
    return float(np.sqrt(np.sum(w2d * (fc - truth) ** 2)
                         / (float(np.sum(cos_w1d)) * fc.shape[1])))


def load_era5_level(lev_hpa, months=("2023_02", "2023_03", "2023_04")):
    """Load ERA5 T at one pressure level across all months.
    Returns (times_np64, data_721_1440) where data has shape (N, 721, 1440).
    Latitude order: descending (90 → −90)."""
    dsets = []
    for m in months:
        fp = ERA5_DIR / f"era5_pressure_{m}.grib"
        ds = xr.open_dataset(str(fp), engine="cfgrib",
                             filter_by_keys={"shortName": "t",
                                             "typeOfLevel": "isobaricInhPa",
                                             "level": int(lev_hpa)})
        dsets.append(ds)
    combined = xr.concat(dsets, dim="time").sortby("time")
    lat_name = "latitude" if "latitude" in combined.dims else "lat"
    lat_vals = combined[lat_name].values
    if lat_vals[0] < lat_vals[-1]:                    # flip to descending if needed
        combined = combined.isel({lat_name: slice(None, None, -1)})
    times = combined["time"].values                   # np.datetime64
    data  = combined["t"].values.astype(np.float32)  # (N, 721, 1440)
    for d in dsets:
        d.close()
    del combined; gc.collect()
    return times, data


def era5_idx(era5_times, valid_dt):
    target = np.datetime64(valid_dt, "ns")
    return int(np.argmin(np.abs(era5_times - target)))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AURORA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_aurora():
    print("\n=== AURORA PERTURBED RMSE ===")
    nc_file = BASE / "CC_aurora/T_PERTURBED/output/aurora_t_perturbed.nc"
    out_csv = BASE / "CC_aurora/T_PERTURBED/rmse_perturbed_temp.csv"

    with nc4.Dataset(str(nc_file)) as ds:
        lead_times  = ds["lead_time"][:].data.astype(int)      # hours, (240,)
        aurora_lats = ds["lat"][:].data                         # (720,) descending

    print(f"  Forecast: {len(lead_times)} steps, "
          f"lead {lead_times[0]}–{lead_times[-1]} h")

    AURORA_IC = datetime(2023, 2, 1, 12, 0)
    valid_dts = [AURORA_IC + timedelta(hours=int(lt)) for lt in lead_times]
    cos_w     = _cos_w1d(aurora_lats)                           # (720,)

    rows = [{"lead_hour": int(lt)} for lt in lead_times]

    for lev in AF_LEVELS:
        print(f"  T{lev} hPa: loading ERA5 ...")
        era5_times, era5_data = load_era5_level(lev)            # (N, 721, 1440)
        print(f"  T{lev} hPa: loading Aurora forecast ...")
        with nc4.Dataset(str(nc_file)) as ds:
            fc_lev = ds[f"t{lev}"][:].data.astype(np.float32)  # (240, 720, 1440)

        for s, vt in enumerate(valid_dts):
            ei    = era5_idx(era5_times, vt)
            truth = era5_data[ei, :-1, :]                       # drop south-pole → 720
            rows[s][f"t{lev}"] = global_rmse(fc_lev[s], truth, cos_w)

        del fc_lev, era5_data; gc.collect()
        print(f"    done T{lev}")

    df = pd.DataFrame(rows)[["lead_hour"] + [f"t{l}" for l in AF_LEVELS]]
    df.to_csv(str(out_csv), index=False)
    print(f"  Saved: {out_csv}  ({len(df)} rows)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FCN
# ═══════════════════════════════════════════════════════════════════════════════

def build_ch_idx(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        n = ds.dimensions["channel"].size
        return {str(ds.variables["channel"][i]): i for i in range(n)}


def compute_fcn():
    print("\n=== FCN PERTURBED RMSE ===")
    nc_file = BASE / "fcn_inference/t_perturbed/forecast_t_perturbed_global_day30_60.nc"
    out_csv = BASE / "fcn_inference/t_perturbed/rmse_perturbed_temp.csv"

    # build channel map and read times
    ch_idx = build_ch_idx(nc_file)
    t_ch   = {lev: ch_idx[f"t{lev}"] for lev in AF_LEVELS}
    print(f"  Channel sample: {list(ch_idx.keys())[:6]}")

    with nc4.Dataset(str(nc_file)) as ds:
        tv       = ds.variables["time"]
        nums     = tv[:].data
        units, cal = tv.units, tv.calendar
        n_steps  = ds.dimensions["time"].size

    valid_dts = [cft.num2date(n, units, cal, only_use_cftime_datetimes=False)
                 for n in nums]
    valid_dts = [datetime(t.year, t.month, t.day, t.hour) for t in valid_dts]

    FCN_IC   = datetime(2023, 2, 1, 0, 0)
    lead_hrs = [int((vt - FCN_IC).total_seconds() / 3600) for vt in valid_dts]

    FCN_LATS = np.linspace(90.0, -90.0, 721)
    cos_w    = _cos_w1d(FCN_LATS)

    rows = [{"lead_hour": lh} for lh in lead_hrs]

    for lev in AF_LEVELS:
        print(f"  Loading ERA5 T{lev} hPa ...")
        era5_times, era5_data = load_era5_level(lev)            # (N, 721, 1440)

        # load FCN channel for this level
        with nc4.Dataset(str(nc_file)) as ds:
            fc_lev = ds["prediction"][:, t_ch[lev], :, :].data.astype(np.float32)

        for s, vt in enumerate(valid_dts):
            ei = era5_idx(era5_times, vt)
            rows[s][f"t{lev}"] = global_rmse(fc_lev[s], era5_data[ei], cos_w)

        del fc_lev, era5_data; gc.collect()
        print(f"    done T{lev}")

    df = pd.DataFrame(rows)[["lead_hour"] + [f"t{l}" for l in AF_LEVELS]]
    df.to_csv(str(out_csv), index=False)
    print(f"  Saved: {out_csv}  ({len(df)} rows)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PRITHVI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_prithvi():
    print("\n=== PRITHVI PERTURBED RMSE ===")
    nc_file = BASE / "prithviOSSE/perturbed_temp_ins/output/perturbed_temp_osse_forecast_Jan2022.nc"
    out_csv = BASE / "prithviOSSE/perturbed_temp_ins/rmse_perturbed_temp.csv"

    fcast   = xr.open_dataset(str(nc_file))
    pri_lat = fcast["lat"].values        # (360,)
    lev_arr = fcast["lev"].values        # level indices, e.g. [34,39,...]
    n_steps = fcast.sizes["time"]
    cos_w   = _cos_w1d(pri_lat)         # (360,)

    # Reconstruct valid times from IC_T1 = 2022-01-01 06:00
    IC_T1     = pd.Timestamp("2022-01-01T06:00")
    valid_dts = [IC_T1 + pd.Timedelta(hours=(s + 1) * 6) for s in range(n_steps)]
    lead_hrs  = [int((vt - IC_T1).total_seconds() / 3600) for vt in valid_dts]

    # pre-load forecast T
    T_all = fcast["T"].values.astype(np.float32)   # (240, 14, 360, 576)
    fcast.close()
    print(f"  Loaded T_all {T_all.shape}")

    col_names = [f"t{hpa}hPa" for hpa in PRI_LEV_HPA]

    # Group steps by MERRA-2 date
    date_to_steps = defaultdict(list)
    for s, vt in enumerate(valid_dts):
        date_to_steps[vt.strftime("%Y%m%d")].append((s, vt.hour // 3))

    rows   = [{"lead_hour": lh} for lh in lead_hrs]
    n_days = len(date_to_steps)

    for di, (date_str, step_tidx_list) in enumerate(sorted(date_to_steps.items())):
        pres_fp = MERRA_DIR / f"MERRA_pres_{date_str}.nc"
        if not pres_fp.exists():
            print(f"    MISSING: {pres_fp.name}")
            continue

        t_indices = sorted({ti for _, ti in step_tidx_list})

        # load MERRA-2 T for needed time indices, all 14 levels
        with h5py.File(str(pres_fp), "r") as f:
            merra_lev   = [int(v) for v in f["lev"][:]]
            lev_file_i  = [merra_lev.index(lv) for lv in PRI_LEV_VALS]
            n_lat_merra = len(f["lat"][:])
            lat_start   = n_lat_merra - len(pri_lat)   # skip north-pole rows
            # shape: (len(t_indices), 14, 360, 576)
            merra_T = f["T"][t_indices][:, lev_file_i, lat_start:, :].astype(np.float32)

        t_idx_row = {ti: row for row, ti in enumerate(t_indices)}

        for s, t_idx in step_tidx_list:
            row_m = t_idx_row[t_idx]
            for li in range(len(PRI_LEV_VALS)):
                rows[s][col_names[li]] = global_rmse(
                    T_all[s, li], merra_T[row_m, li], cos_w)

        del merra_T
        if (di + 1) % 10 == 0 or di == 0:
            print(f"  [{di+1}/{n_days}] {date_str}")

    df = pd.DataFrame(rows)[["lead_hour"] + col_names]
    df.to_csv(str(out_csv), index=False)
    print(f"  Saved: {out_csv}  ({len(df)} rows)")
    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compute_aurora()
    compute_fcn()
    compute_prithvi()
    print("\nAll done.")
