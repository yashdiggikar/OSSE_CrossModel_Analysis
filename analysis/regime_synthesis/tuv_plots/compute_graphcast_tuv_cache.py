#!/usr/bin/env python3
"""
Build the GraphCast entry for the tri-model TUV synergy figure
(common/tuv_final/plot_synergy_z500_q850.py), using the already-completed
GC-DETER-2023 study (Feb 2023 IC, 0.25deg/37lvl full-res) instead of the
older GraphCast-Small runs the rest of this study used.

Unlike compute_tuv_rmse.py (FCN/Aurora/Prithvi, which load truth on the fly
per model since those three predate this project's shared truth-loading
scripts), this reuses GC-DETER-2023's own compute_rmse.py/compute_rmse_q.py
output directly for the time-series CSVs (z500, q850 already scored for
every experiment), and only computes new per-gridpoint spatial fields (mean
squared error, mature phase = lead day 40-60, matching this figure's own
convention -- NOT the 30-60 convention compute_bias_maps.py uses elsewhere
in GC-DETER-2023) for the synergy maps, since those were never computed.

Experiments (GC-DETER-2023 naming -> this figure's naming):
    control          -> ctrl
    osse_t_global    -> tins  (T-only insertion)
    osse_uv_global   -> uins  (UV-only insertion)
    combined_tuv_ins -> tuv   (T+UV dual insertion)

Outputs (into common/tuv_plots/cache/, alongside the fcn_/aurora_/prithvi_
files compute_tuv_rmse.py already wrote):
    graphcast_{ctrl,tins,uins,tuv}_rmse.csv   (lead_day, z500, q850)
    graphcast_synergy_z500.npy / graphcast_impr_tuv_z500.npy
    graphcast_synergy_q850.npy / graphcast_impr_tuv_q850.npy
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray

GD = "/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023"
CACHE = "/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/cache"
os.makedirs(CACHE, exist_ok=True)

SAMPLE_NC = f"{GD}/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc"
ERA5_DIR = f"{GD}/era5_feb2023"
ERA5_Q_FILE = f"{GD}/era5_feb2023/era5_q_2023_febapr.grib"

EXPERIMENTS = {
    "ctrl": ("control", f"{GD}/control/full_predictions/rollout_60day_2023-02-01"),
    "tins": ("osse_t_global", f"{GD}/osse_t_global/full_predictions/osse_t_global_2023-02-01"),
    "uins": ("osse_uv_global", f"{GD}/osse_uv_global/full_predictions/osse_uv_global_2023-02-01"),
    "tuv":  ("combined_tuv_ins", f"{GD}/combined_tuv_ins/full_predictions/combined_tuv_ins_2023-02-01"),
}
RMSE_CSV = {
    "ctrl": f"{GD}/control/results/csv/rmse_60day_2023-02-01.csv",
    "tins": f"{GD}/osse_t_global/results/csv/rmse_osse_t_global_2023-02-01.csv",
    "uins": f"{GD}/osse_uv_global/results/csv/rmse_osse_uv_global_2023-02-01.csv",
    "tuv":  f"{GD}/combined_tuv_ins/results/csv/rmse_combined_tuv_ins_2023-02-01.csv",
}
RMSE_Q_CSV = {
    "ctrl": f"{GD}/control/results/csv/rmse_q_2023-02-01.csv",
    "tins": f"{GD}/osse_t_global/results/csv/rmse_q_osse_t_global_2023-02-01.csv",
    "uins": f"{GD}/osse_uv_global/results/csv/rmse_q_osse_uv_global_2023-02-01.csv",
    "tuv":  f"{GD}/combined_tuv_ins/results/csv/rmse_q_combined_tuv_ins_2023-02-01.csv",
}

print("Resolved main RMSE csv paths:")
for k, v in RMSE_CSV.items():
    print(f"  {k}: {v}  exists={os.path.exists(v)}")

# ── 1) Time-series CSVs (z500, q850) for the bar-chart's synergy_scalar/CI ──
for key in EXPERIMENTS:
    df_main = pd.read_csv(RMSE_CSV[key])
    z500 = df_main[(df_main["variable"] == "geopotential") & (df_main["level_hpa"] == 500)] \
        .sort_values("lead_days")[["lead_days", "rmse"]].rename(columns={"lead_days": "lead_day", "rmse": "z500"})

    df_q = pd.read_csv(RMSE_Q_CSV[key])
    q850 = df_q[(df_q["variable"] == "specific_humidity") & (df_q["level_hpa"] == 850)] \
        .sort_values("lead_days")[["lead_days", "rmse"]].rename(columns={"lead_days": "lead_day", "rmse": "q850"})

    out = pd.merge(z500, q850, on="lead_day", how="inner")
    out_path = os.path.join(CACHE, f"graphcast_{key}_rmse.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(out)} rows")

# ── 2) Spatial per-gridpoint MSE (mature phase, lead day 40-60) -> synergy ──
def mature_mse_field(pred_dir, var_gc, era5_var, level, era5_kind):
    """Returns (lat, lon) mean squared error over lead_day in (40, 60]."""
    ic_ds = xarray.open_dataset(SAMPLE_NC, decode_timedelta=True)
    ic_last_datetime = ic_ds.coords["datetime"].isel(time=1).values
    ic_ds.close()

    if era5_kind == "pressure":
        era5_files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_pressure_2023_*.grib")))
        truth_by_month = {f: xarray.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
                           for f in era5_files}

        def get_truth(valid_times):
            slices = []
            for vt in valid_times:
                for f, ds in truth_by_month.items():
                    tmin, tmax = ds.coords["time"].min().values, ds.coords["time"].max().values
                    if tmin <= vt <= tmax:
                        slices.append(ds[era5_var].sel(time=vt, isobaricInhPa=level).load())
                        break
            return xarray.concat(slices, dim="time")
    else:
        truth_ds = xarray.open_dataset(ERA5_Q_FILE, engine="cfgrib", backend_kwargs={"indexpath": ""})

        def get_truth(valid_times):
            return truth_ds["q"].sel(time=valid_times, isobaricInhPa=level).load()

    sum_sq = None
    n = 0
    lat_out = lon_out = None
    chunk_paths = sorted(glob.glob(os.path.join(pred_dir, "chunk_*.nc")))
    for ci, chunk_path in enumerate(chunk_paths):
        pred = xarray.open_dataset(chunk_path, decode_timedelta=True).load()
        lead_times = pred.coords["time"].values
        valid_times = ic_last_datetime + lead_times
        lead_days = lead_times / np.timedelta64(1, "D")
        mask = (lead_days > 40.0) & (lead_days <= 60.0)
        if not np.any(mask):
            pred.close()
            continue
        idx = np.nonzero(mask)[0]

        if lat_out is None:
            lat_out = pred.coords["lat"].values.astype(np.float64)
            lon_out = pred.coords["lon"].values.astype(np.float64)
            sum_sq = np.zeros((len(lat_out), len(lon_out)), dtype=np.float64)

        truth = get_truth(valid_times[idx])
        truth = truth.sortby("latitude").rename({"latitude": "lat", "longitude": "lon"})
        truth = truth.assign_coords(lat=pred.coords["lat"].values, lon=pred.coords["lon"].values)

        pred_var = pred[var_gc]
        if "batch" in pred_var.dims:
            pred_var = pred_var.squeeze("batch", drop=True)
        pred_var = pred_var.sel(level=level).isel(time=idx)

        diff2 = (pred_var.values - truth.values) ** 2
        sum_sq += diff2.sum(axis=0)
        n += len(idx)
        pred.close()

    if era5_kind == "pressure":
        for ds in truth_by_month.values():
            ds.close()
    else:
        truth_ds.close()

    return sum_sq / n, lat_out, lon_out, n


print("\nComputing spatial mature-phase (day 40-60) MSE fields for z500, q850 ...")
mse = {"z500": {}, "q850": {}}
for key, (_, pred_dir) in EXPERIMENTS.items():
    print(f"  z500 [{key}] ...", flush=True)
    mse_z, lat_out, lon_out, n_z = mature_mse_field(pred_dir, "geopotential", "z", 500, "pressure")
    mse["z500"][key] = mse_z
    print(f"    n={n_z} mature steps, mean MSE={np.nanmean(mse_z):.2f}")

    print(f"  q850 [{key}] ...", flush=True)
    mse_q, _, _, n_q = mature_mse_field(pred_dir, "specific_humidity", "q", 850, "q")
    mse["q850"][key] = mse_q
    print(f"    n={n_q} mature steps, mean MSE={np.nanmean(mse_q):.3e}")

np.save(os.path.join(CACHE, "graphcast_lat.npy"), lat_out)
np.save(os.path.join(CACHE, "graphcast_lon.npy"), lon_out)

for var in ["z500", "q850"]:
    ctrl_mse, tins_mse, uins_mse, tuv_mse = (mse[var]["ctrl"], mse[var]["tins"],
                                              mse[var]["uins"], mse[var]["tuv"])
    with np.errstate(invalid="ignore", divide="ignore"):
        i_t = np.where(ctrl_mse > 0, (ctrl_mse - tins_mse) / ctrl_mse * 100, np.nan)
        i_uv = np.where(ctrl_mse > 0, (ctrl_mse - uins_mse) / ctrl_mse * 100, np.nan)
        i_tuv = np.where(ctrl_mse > 0, (ctrl_mse - tuv_mse) / ctrl_mse * 100, np.nan)
    syn = i_tuv - np.maximum(i_t, i_uv)
    np.save(os.path.join(CACHE, f"graphcast_synergy_{var}.npy"), syn)
    np.save(os.path.join(CACHE, f"graphcast_impr_tuv_{var}.npy"), i_tuv)
    print(f"GraphCast {var}: synergy mean={np.nanmean(syn):.2f}%  impr_tuv mean={np.nanmean(i_tuv):.2f}%")

print("\nDONE -- GraphCast TUV cache written to", CACHE)
