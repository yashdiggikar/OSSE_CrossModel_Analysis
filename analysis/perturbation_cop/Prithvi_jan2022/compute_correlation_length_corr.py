#!/usr/bin/env python3
"""
Empirical correlation-length verification for the Prithvi WxC Correlated
OSSE Perturbation experiments (perturbed_corr_temp_ins, perturbed_corr_uv_ins).

See ../CORRELATION_LENGTH_VERIFICATION.md for the full method description
and rationale — same method as FCN/Aurora's versions of this script, adapted
to Prithvi's non-square grid (0.5 deg lat x 0.625 deg lon), its 14 native
MERRA-2 levels (LEVEL_APPROX_HPA, not the 13 standard pressure levels
FCN/Aurora use), and MERRA-2 truth (h5py daily files, not ERA5 grib).

Full variable grid (matching this project's other 9 "vs truth" scripts, per
user request): T, U, V, Z (as H, geopotential height), Q (as QV, specific
humidity) -- computed for BOTH experiments regardless of which family was
directly inserted (cross-variable coupling included), at 3 representative
native levels nearest 300/500/1000 hPa (Prithvi has no exact 1000hPa level --
nearest is lev 72, ~961hPa) for the horizontal check, all 14 native levels
for the vertical check.

Output per experiment (into PERTURBATION_EXPERIMENTS/Prithvi/<experiment>/):
    corr_length_horizontal_<experiment>.csv
    corr_length_vertical_<experiment>.csv
    corr_length_horizontal_map_<experiment>.png
    corr_length_vertical_map_<experiment>.png

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_correlation_length_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
import time
from pathlib import Path
from datetime import datetime, timedelta

import netCDF4 as nc4
import h5py
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE       = Path("/umbc/rs/pi_halem/users/yashd1")
PRITHVI    = BASE / "prithviOSSE"
MERRA2_DIR = BASE / "prithvi" / "data" / "merra2"
OUT_DIR    = BASE / "PERTURBATION_EXPERIMENTS" / "Prithvi"

EXPERIMENTS = [
    ("perturbed_corr_temp_ins", PRITHVI / "perturbed_corr_temp_ins" / "output" / "perturbed_corr_temp_osse_forecast_Jan2022.nc", "temp"),
    ("perturbed_corr_uv_ins",   PRITHVI / "perturbed_corr_uv_ins"   / "output" / "perturbed_corr_uv_osse_forecast_Jan2022.nc",   "uv"),
]

IC_T1   = datetime(2022, 1, 1, 6, 0)
DT_H    = 6
N_STEPS = 240
INJECT_STEP = 120
MAX_LAG_PX  = 40
VERT_L_LOGP = 0.5

LEV_VALS = [34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72]   # native MERRA-2 level codes, TOA->surface
LEVEL_APPROX_HPA = [52, 118, 164, 226, 265, 310, 424, 532, 604, 693, 832, 903, 946, 961]
TARGET_HPA = [300, 500, 1000]
REPR_IDX = {}
for target in TARGET_HPA:
    nearest_i = min(range(len(LEVEL_APPROX_HPA)), key=lambda i: abs(LEVEL_APPROX_HPA[i] - target))
    REPR_IDX[target] = nearest_i
    print(f"  target {target}hPa -> nearest native lev {LEV_VALS[nearest_i]} (~{LEVEL_APPROX_HPA[nearest_i]}hPa)")

PRITHVI_LATS = np.linspace(-89.5, 90.0, 360)
PRITHVI_LONS = np.linspace(-180.0, 179.375, 576)
GRID_DEG_LAT = 0.5
GRID_DEG_LON = 0.625
KM_PER_DEG_LAT = 111.0

BANDS = {"equator_5N": 5.0, "midlat_45N": 45.0, "highlat_70N": 70.0}
BAND_ROWS = {name: int(np.argmin(np.abs(PRITHVI_LATS - lat))) for name, lat in BANDS.items()}

# Full variable grid: label -> (Prithvi/MERRA-2 var name, derived-non-divergent flag)
UPPER_VARS = {"T": ("T", False), "U": ("U", True), "V": ("V", True), "Z": ("H", False), "Q": ("QV", False)}
SFC_VARS = {"T2M": "T2M", "U10M": "U10M", "V10M": "V10M"}   # identical name both sides


def merra2_tidx(valid_dt):
    return valid_dt.hour // 3


def load_merra2_pres_all_levels(date_str, varname, t_idx):
    """Returns (14, 360, 576) cube in LEV_VALS order, or None if file/var missing."""
    fpath = MERRA2_DIR / f"MERRA_pres_{date_str}.nc"
    if not fpath.exists():
        return None
    with h5py.File(str(fpath), "r") as f:
        if varname not in f:
            return None
        merra_lev = list(f["lev"][()])
        cube = np.empty((len(LEV_VALS), 360, 576), dtype=np.float32)
        for i, lv in enumerate(LEV_VALS):
            if int(lv) not in merra_lev:
                cube[i] = np.nan
                continue
            m_idx = merra_lev.index(int(lv))
            cube[i] = f[varname][t_idx, m_idx, 1:, :]   # skip -90 row: 361 -> 360
    return cube


def load_merra2_sfc(date_str, varname, t_idx):
    fpath = MERRA2_DIR / f"MERRA2_sfc_{date_str}.nc"
    if not fpath.exists():
        return None
    with h5py.File(str(fpath), "r") as f:
        if varname not in f:
            return None
        return f[varname][t_idx, 1:, :].astype(np.float32)


def lag_correlation_1d(line, max_lag, wrap):
    n = len(line)
    line = line - line.mean()
    denom = np.sum(line ** 2)
    corrs = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        if wrap:
            shifted = np.roll(line, -k)
            corrs[k] = np.sum(line * shifted) / denom if denom > 0 else np.nan
        else:
            a, b = (line, line) if k == 0 else (line[:n - k], line[k:])
            corrs[k] = np.sum(a * b) / denom if denom > 0 else np.nan
    return corrs


def efold_length_km(lags_km, corrs):
    target = 1.0 / np.e
    below = np.where(corrs <= target)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return 0.0
    x0, x1 = lags_km[i - 1], lags_km[i]
    y0, y1 = corrs[i - 1], corrs[i]
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def horizontal_check(field2d, label, level_hpa, derived_field, out_rows, ax, plot_it):
    n_lat = field2d.shape[0]
    for band_name, row in BAND_ROWS.items():
        lat = PRITHVI_LATS[row]
        zonal_line = field2d[row, :]
        zonal_corr = lag_correlation_1d(zonal_line, MAX_LAG_PX, wrap=True)
        km_per_px_zonal = KM_PER_DEG_LAT * np.cos(np.deg2rad(lat)) * GRID_DEG_LON
        zonal_lags_km = np.arange(MAX_LAG_PX + 1) * km_per_px_zonal
        L_zonal = efold_length_km(zonal_lags_km, zonal_corr)

        col = 0
        merid_col = field2d[max(0, row - MAX_LAG_PX):min(n_lat, row + MAX_LAG_PX + 1), col]
        merid_corr = lag_correlation_1d(merid_col[len(merid_col) // 2:], MAX_LAG_PX, wrap=False)
        km_per_px_merid = KM_PER_DEG_LAT * GRID_DEG_LAT
        merid_lags_km = np.arange(len(merid_corr)) * km_per_px_merid
        L_merid = efold_length_km(merid_lags_km, merid_corr)

        for direction, lags_km, corr, L in [
            ("zonal", zonal_lags_km, zonal_corr, L_zonal),
            ("meridional", merid_lags_km, merid_corr, L_merid),
        ]:
            out_rows.append({
                "variable": label, "level_hpa": level_hpa, "band": band_name, "latitude_deg": lat,
                "direction": direction, "derived_field": derived_field,
                "empirical_L_km": L, "target_L_km": 250.0,
                "ratio_empirical_over_target": L / 250.0 if not np.isnan(L) else np.nan,
            })
            if ax is not None and plot_it and band_name == "midlat_45N":
                ax.plot(lags_km, corr, marker="o", ms=3,
                        label=f"{label} {direction} (L≈{L:.0f} km)" if not np.isnan(L) else f"{label} {direction}")


def vertical_check(cube, levels_hpa, label, out_rows):
    n = len(levels_hpa)
    flat = cube.reshape(n, -1)
    emp_corr = np.corrcoef(flat)
    logp = np.log(levels_hpa)
    theo_corr = np.exp(-np.abs(logp[:, None] - logp[None, :]) / VERT_L_LOGP)
    for i in range(n):
        for j in range(n):
            out_rows.append({
                "variable": label, "level_i": levels_hpa[i], "level_j": levels_hpa[j],
                "empirical_corr": emp_corr[i, j], "theoretical_corr": theo_corr[i, j],
                "diff": emp_corr[i, j] - theo_corr[i, j],
            })


for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    valid_times = [IC_T1 + timedelta(hours=(s + 1) * DT_H) for s in range(N_STEPS)]
    inject_dt = valid_times[INJECT_STEP]
    date_str = inject_dt.strftime("%Y%m%d")
    t_idx = merra2_tidx(inject_dt)

    # Open once, retry-with-delay for the same NFS flakiness class already
    # documented for this project's Prithvi control-file access.
    exp_ds = None
    for _attempt in range(5):
        try:
            exp_ds = nc4.Dataset(str(nc_file))
            break
        except OSError as e:
            print(f"  transient open error ({e!r}), retrying in 15s ...")
            exp_ds = None
            time.sleep(15)
    if exp_ds is None:
        print(f"[{exp_name}] SKIP — could not open {nc_file} after 5 attempts.")
        continue

    horiz_rows, vert_rows = [], []
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    theta = np.linspace(0, MAX_LAG_PX * KM_PER_DEG_LAT * GRID_DEG_LON, 200)
    ax_h.plot(theta, np.exp(-theta / 250.0), "k--", lw=1, label="theoretical exp(-d/250km)")
    ax_h.plot(theta, np.exp(-theta / (250.0 * np.sqrt(2))), "k:", lw=1,
              label="theoretical exp(-d/(250*sqrt(2))km)")

    # Every upper-air variable/level is evaluated for every experiment (not
    # just the directly-inserted family) -- cross-variable coupling.
    for label, (varname, derived) in UPPER_VARS.items():
        truth_cube = load_merra2_pres_all_levels(date_str, varname, t_idx)
        if truth_cube is None:
            print(f"  [{exp_name}] {label}: no MERRA-2 truth for {date_str}, skipping.")
            continue
        if varname not in exp_ds.variables:
            print(f"  [{exp_name}] {label}: '{varname}' not in experiment output, skipping.")
            continue
        exp_cube = np.asarray(exp_ds.variables[varname][INJECT_STEP, :, :, :])   # (14, 360, 576)
        pert_cube = exp_cube - truth_cube

        for hpa in TARGET_HPA:
            lvl_idx = REPR_IDX[hpa]
            horizontal_check(pert_cube[lvl_idx], label, hpa, derived, horiz_rows, ax_h, plot_it=(hpa == 500))
        vertical_check(pert_cube, LEVEL_APPROX_HPA, label, vert_rows)
        print(f"  [{exp_name}] {label}: horizontal ({TARGET_HPA}hPa nearest-native) "
              f"+ vertical ({len(LEVEL_APPROX_HPA)} levels) checks done")

        pd.DataFrame(horiz_rows).to_csv(out_dir / f"corr_length_horizontal_{exp_name}.csv", index=False)
        pd.DataFrame(vert_rows).to_csv(out_dir / f"corr_length_vertical_{exp_name}.csv", index=False)

    for sfc_label, sfc_varname in SFC_VARS.items():
        sfc_truth = load_merra2_sfc(date_str, sfc_varname, t_idx)
        if sfc_truth is None:
            print(f"  [{exp_name}] {sfc_label} (surface): no MERRA-2 truth for {date_str}, skipping.")
            continue
        if sfc_varname not in exp_ds.variables:
            print(f"  [{exp_name}] {sfc_label} (surface): '{sfc_varname}' not in experiment output, skipping.")
            continue
        sfc_exp = np.asarray(exp_ds.variables[sfc_varname][INJECT_STEP, :, :])
        sfc_pert = sfc_exp - sfc_truth
        horizontal_check(sfc_pert, sfc_label, None, False, horiz_rows, None, plot_it=False)
        print(f"  [{exp_name}] {sfc_label} (surface): horizontal check done")

    pd.DataFrame(horiz_rows).to_csv(out_dir / f"corr_length_horizontal_{exp_name}.csv", index=False)
    pd.DataFrame(vert_rows).to_csv(out_dir / f"corr_length_vertical_{exp_name}.csv", index=False)

    ax_h.set_xlabel("lag distance (km)"); ax_h.set_ylabel("correlation")
    ax_h.set_title(f"{exp_name}: horizontal lag-correlation at 45N, level≈500hPa")
    ax_h.legend(fontsize=7); ax_h.grid(alpha=0.3)
    fig_h.savefig(out_dir / f"corr_length_horizontal_map_{exp_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig_h)

    if vert_rows:
        df_v = pd.DataFrame(vert_rows)
        first_var = df_v["variable"].iloc[0]
        sub = df_v[df_v["variable"] == first_var]
        levels_sorted = sorted(sub["level_i"].unique())
        n = len(levels_sorted)
        emp_m = sub.pivot(index="level_i", columns="level_j", values="empirical_corr").loc[levels_sorted, levels_sorted].values
        theo_m = sub.pivot(index="level_i", columns="level_j", values="theoretical_corr").loc[levels_sorted, levels_sorted].values
        fig_v, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for a, m, title in zip(axes, [emp_m, theo_m, emp_m - theo_m],
                                [f"empirical ({first_var})", "theoretical", "difference"]):
            im = a.imshow(m, vmin=-1 if title != "difference" else -0.3,
                          vmax=1 if title != "difference" else 0.3, cmap="RdBu_r")
            a.set_title(title, fontsize=10)
            a.set_xticks(range(n)); a.set_xticklabels(levels_sorted, rotation=90, fontsize=6)
            a.set_yticks(range(n)); a.set_yticklabels(levels_sorted, fontsize=6)
            plt.colorbar(im, ax=a, shrink=0.8)
        fig_v.suptitle(f"{exp_name}: vertical correlation matrix, levels in approx hPa")
        fig_v.savefig(out_dir / f"corr_length_vertical_map_{exp_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig_v)

    print(f"  Saved: {out_dir / f'corr_length_horizontal_{exp_name}.csv'}")
    print(f"  Saved: {out_dir / f'corr_length_vertical_{exp_name}.csv'}")
    exp_ds.close()

print("\nDone.")
