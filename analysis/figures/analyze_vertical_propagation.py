#!/usr/bin/env python3
"""
Vertical propagation analysis — Section 5.10
Figures 5.56–5.61 and Table 5.12 for T_LOWER and UV_LOWER insertion experiments
across Aurora, FCN, and Prithvi models.
"""

import os, gc, warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import LogFormatter, LogLocator
import matplotlib.ticker as mticker
import pandas as pd
import netCDF4 as nc4
import h5py
import xarray as xr

warnings.filterwarnings('ignore')

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE = Path('/umbc/rs/pi_halem/users/yashd1')
ERA5_DIR   = BASE / 'AURORA_DATA' / 'rmse_data'
MERRA2_DIR = BASE / 'prithvi' / 'data' / 'merra2'

AURORA_CTRL     = BASE / 'auroraoutput_1year_1stfeb23.nc'
AURORA_TLOWER   = BASE / 'CC_aurora' / 'T_LOWER'  / 'output' / 'aurora_t_lower.nc'
AURORA_UVLOWER  = BASE / 'CC_aurora' / 'UV_LOWER' / 'output' / 'aurora_uv_lower.nc'
AURORA_TDIR     = BASE / 'CC_aurora' / 'T_LOWER'
AURORA_UVDIR    = BASE / 'CC_aurora' / 'UV_LOWER'
AURORA_ROOT     = BASE / 'CC_aurora'

FCN_CTRL    = BASE / 'FourCastNet' / 'forecast_normal_60days_12hr.nc'
FCN_TLOWER  = BASE / 'fcn_inference' / 't_lower'  / 'forecast_t_lower3_global_day30_60.nc'
FCN_UVLOWER = BASE / 'fcn_inference' / 'uv_lower' / 'forecast_uv_lower3_global_day30_60.nc'
FCN_TDIR    = BASE / 'fcn_inference' / 't_lower'
FCN_UVDIR   = BASE / 'fcn_inference' / 'uv_lower'
FCN_ROOT    = BASE / 'fcn_inference'

PRITHVI_CTRL   = BASE / 'prithvi' / 'output' / 'forecast_60day_Jan2022.nc'
PRITHVI_TLOWER = BASE / 'prithviOSSE' / 'lower_temp_ins' / 'output' / 'lower_temp_osse_forecast_Jan2022.nc'
PRITHVI_UVLOWER= BASE / 'prithviOSSE' / 'lower_uv_ins'  / 'output' / 'lower_uv_osse_forecast_Jan2022.nc'
PRITHVI_TDIR   = BASE / 'prithviOSSE' / 'lower_temp_ins'
PRITHVI_UVDIR  = BASE / 'prithviOSSE' / 'lower_uv_ins'
PRITHVI_ROOT   = BASE / 'prithviOSSE'

for d in [AURORA_TDIR, AURORA_UVDIR, FCN_TDIR, FCN_UVDIR, PRITHVI_TDIR, PRITHVI_UVDIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
AURORA_IC  = datetime(2023, 2, 1, 12, 0)   # AURORA_CTRL's own reference time -- correct as-is
FCN_IC     = datetime(1900, 1, 1, 0, 0)       # epoch; FCN time var is hours since this
PRITHVI_IC = datetime(2022, 1, 1, 6, 0)


def read_base_time(nc_file):
    # AURORA_TLOWER/AURORA_UVLOWER (T_LOWER/UV_LOWER's own output) had a +6h
    # base_time mislabeling bug (base_time was captured from the first
    # forecast's own valid time instead of the true pre-rollout reference
    # time); patched in place, so read per-file rather than assuming
    # AURORA_IC applies to these experiment files too -- AURORA_CTRL never
    # had this bug and keeps using AURORA_IC directly, unchanged.
    with nc4.Dataset(str(nc_file)) as ds:
        return datetime.utcfromtimestamp(int(ds.variables["base_time"][...]))

N_STEPS   = 240
INJ_START = 120                       # 0-indexed first injection step
ADJ_SL    = slice(120, 180)           # adjustment phase (days 30–45)
MAT_SL    = slice(180, 240)           # mature phase (days 45–60)

# Aurora/FCN control: 12-h steps, lead_time[n] = 12*n h
ACTRL_ADJ = slice(60, 90)            # 720–1068 h
ACTRL_MAT = slice(90, 121)           # 1080–1440 h
FCTRL_ADJ = slice(60, 90)
FCTRL_MAT = slice(90, 120)

# Aurora & FCN pressure levels (hPa)
AF_LEVELS = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])

# Prithvi / MERRA2
PRI_LEV_VALS = np.array([34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72])
PRI_LEV_HPA  = np.array([52, 118, 164, 226, 265, 310, 424, 532, 604, 693, 832, 903, 946, 960])
# MERRA_pres files store levels in descending order:
MERRA_LEV_DESC = [72, 71, 68, 63, 56, 53, 51, 48, 45, 44, 43, 41, 39, 34]

# Latitude bins (5°)
LAT_EDGES   = np.arange(-90, 91, 5)          # 37 edges → 36 bands
LAT_CENTERS = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2   # -87.5 … 87.5

# Inserted pressure groups (for propagation efficiency)
def level_group_af(hpa):
    if hpa >= 850: return 0   # Inserted
    elif hpa >= 400: return 1  # Mid
    else: return 2              # Upper

def level_group_pri(hpa):
    if hpa >= 850: return 0
    elif hpa >= 400: return 1
    else: return 2

GROUP_LABELS = ['Inserted\n(850–1000 hPa)', 'Mid\n(400–700 hPa)', 'Upper\n(50–300 hPa)']
PRI_GRPLABELS = ['Inserted\n(~850–960 hPa)', 'Mid\n(400–700 hPa)', 'Upper\n(50–300 hPa)']

# ─── STYLE ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def heatmap_norm(vmin=-25, vmax=75):
    bounds = np.array([-25, -15, -5, 0, 5, 10, 20, 30, 40, 50, 60, 75])
    cmap   = plt.cm.get_cmap('RdYlGn', len(bounds) - 1)
    norm   = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds

def schematic_norm_T(vmin, vmax, n=14):
    bounds = np.linspace(vmin, vmax, n + 1)
    cmap   = plt.cm.get_cmap('plasma', n)
    norm   = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def schematic_norm_wind(vmax, n=14):
    bounds = np.linspace(0, vmax, n + 1)
    cmap   = plt.cm.get_cmap('YlOrRd', n)
    norm   = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def cos_weights(lat_1d):
    """Cosine-latitude weights, shape (n_lat,)."""
    return np.cos(np.deg2rad(lat_1d))

def global_rmse(fc, truth, lat_1d):
    """Global cos-lat weighted RMSE. fc/truth shape: (lat, lon)."""
    cw = cos_weights(lat_1d)[:, None]
    diff2 = (fc - truth) ** 2
    return float(np.sqrt(np.sum(cw * diff2) / (diff2.shape[1] * np.sum(cw))))

def band_rmse_series(fc_ts, truth_ts, lat_1d, band_lo, band_hi):
    """RMSE per step within a lat band. fc_ts: (n_steps, lat, lon)."""
    mask = (lat_1d >= band_lo) & (lat_1d < band_hi)
    if not mask.any():
        return np.zeros(len(fc_ts))
    cw = cos_weights(lat_1d[mask])
    cw /= cw.sum()
    diff2 = (fc_ts[:, mask, :] - truth_ts[:, mask, :]) ** 2  # (t, l, n)
    return np.sqrt(np.einsum('i,tij->t', cw, diff2).mean(axis=-1))   # per step

def lat_band_rmse(fc_ts, truth_ts, lat_1d):
    """RMSE per lat band averaged over all steps. Returns shape (n_bands,)."""
    n_bands = len(LAT_CENTERS)
    out = np.zeros(n_bands)
    for bi in range(n_bands):
        lo, hi = LAT_EDGES[bi], LAT_EDGES[bi + 1]
        mask = (lat_1d >= lo) & (lat_1d < hi)
        if not mask.any():
            continue
        cw = cos_weights(lat_1d[mask])
        cw /= cw.sum()
        diff2 = (fc_ts[:, mask, :] - truth_ts[:, mask, :]) ** 2
        out[bi] = np.sqrt(np.mean(np.einsum('i,ti->t', cw, diff2.mean(axis=-1))))
    return out

def pct_improvement(ctrl_rmse, exp_rmse):
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = np.where(ctrl_rmse > 0,
                       (ctrl_rmse - exp_rmse) / ctrl_rmse * 100.0, 0.0)
    return imp

# ─── ERA5 LOADING ─────────────────────────────────────────────────────────────

def load_era5_times():
    """Load all ERA5 timestamps (Feb–Apr 2023) as numpy datetime64."""
    months = ['2023_02', '2023_03', '2023_04']
    times_all = []
    for m in months:
        ds = xr.open_dataset(ERA5_DIR / f'era5_surface_{m}.grib',
                             engine='cfgrib',
                             backend_kwargs={'filter_by_keys': {'shortName': '2t'}},
                             )
        times_all.append(ds.time.values)
        ds.close()
    return np.concatenate(times_all)

def era5_idx_for(era5_times, valid_dt):
    """Nearest ERA5 time index for a Python datetime."""
    t64 = np.datetime64(valid_dt.strftime('%Y-%m-%dT%H:%M'), 'm')
    diff = np.abs(era5_times.astype('datetime64[m]') - t64)
    return int(np.argmin(diff))

def era5_valid_times_for_steps(ic_dt, lead_times_h):
    """Valid datetimes for experiment steps given IC and lead times (hours)."""
    return [ic_dt + timedelta(hours=int(lt)) for lt in lead_times_h]

def load_era5_surface(var_shortname, type_of_level=None, level_val=None):
    """Load concatenated ERA5 surface variable for Feb–Apr 2023. Returns (T, lat, lon)."""
    months = ['2023_02', '2023_03', '2023_04']
    fk = {'shortName': var_shortname}
    if type_of_level is not None:
        fk['typeOfLevel'] = type_of_level
    if level_val is not None:
        fk['level'] = level_val
    arrays = []
    for m in months:
        fp = ERA5_DIR / f'era5_surface_{m}.grib'
        ds = xr.open_dataset(fp, engine='cfgrib', backend_kwargs={'filter_by_keys': fk},
                             )
        var = list(ds.data_vars)[0]
        arrays.append(ds[var].values)
        ds.close()
    return np.concatenate(arrays, axis=0)  # (T_total, lat, lon)

def load_era5_pressure_level(var_shortname, level_hpa):
    """Load ERA5 pressure-level variable at one level for Feb–Apr 2023. Returns (T, lat, lon)."""
    months = ['2023_02', '2023_03', '2023_04']
    fk = {'shortName': var_shortname, 'typeOfLevel': 'isobaricInhPa', 'level': int(level_hpa)}
    arrays = []
    for m in months:
        fp = ERA5_DIR / f'era5_pressure_{m}.grib'
        ds = xr.open_dataset(fp, engine='cfgrib', backend_kwargs={'filter_by_keys': fk},
                             )
        var = list(ds.data_vars)[0]
        arrays.append(ds[var].values)
        ds.close()
    return np.concatenate(arrays, axis=0)

# ─── MERRA2 LOADING ───────────────────────────────────────────────────────────

def merra2_file(date, kind='sfc'):
    if kind == 'sfc':
        return MERRA2_DIR / f'MERRA2_sfc_{date.strftime("%Y%m%d")}.nc'
    else:
        return MERRA2_DIR / f'MERRA_pres_{date.strftime("%Y%m%d")}.nc'

def merra2_time_idx(hour):
    """3-hourly MERRA2 time index for a given hour (0–23)."""
    return hour // 3

def load_merra2_sfc_step(valid_dt, var):
    """Load one MERRA2 surface variable at a specific valid datetime. Returns (lat, lon)."""
    fp = merra2_file(valid_dt, 'sfc')
    ds = nc4.Dataset(str(fp))
    ti = merra2_time_idx(valid_dt.hour)
    val = ds.variables[var][ti, :, :]
    ds.close()
    return np.array(val)

def load_merra2_pres_step(valid_dt, var, lev_val):
    """Load one MERRA2 3D variable at a specific level and valid time. Returns (lat, lon)."""
    fp = merra2_file(valid_dt, 'pres')
    ds = nc4.Dataset(str(fp))
    ti = merra2_time_idx(valid_dt.hour)
    lev_list = [int(v) for v in ds.variables['lev'][:]]  # descending [72,71,...,34]
    li = lev_list.index(int(lev_val))
    val = ds.variables[var][ti, li, :, :]
    ds.close()
    return np.array(val)

# ─── FIGURE HELPERS ───────────────────────────────────────────────────────────

def add_colorbar(fig, ax, im, label, bounds=None, ticks=None, shrink=0.85):
    cb = fig.colorbar(im, ax=ax, shrink=shrink, pad=0.02)
    cb.set_label(label, fontsize=11)
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb

def plot_global_map(ax, data, lat_1d, lon_1d, cmap, norm, title, unit):
    """Equirectangular global map using imshow."""
    # Ensure lat is ascending for imshow origin='lower'
    if lat_1d[0] > lat_1d[-1]:
        data = data[::-1, :]
        lat_1d = lat_1d[::-1]
    im = ax.imshow(data, origin='lower',
                   extent=[float(lon_1d[0]), float(lon_1d[-1]),
                           float(lat_1d[0]),  float(lat_1d[-1])],
                   aspect='auto', cmap=cmap, norm=norm)
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    return im

def plot_heatmap_panel(ax, data_2d, pres_levels, title):
    """Lat-pressure heatmap, data_2d: (n_lev, n_lat_bands)."""
    cmap, norm, bounds = heatmap_norm()
    X = LAT_CENTERS
    Y = pres_levels
    im = ax.pcolormesh(X, Y, data_2d, cmap=cmap, norm=norm, shading='nearest')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_locator(mticker.FixedLocator(pres_levels))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Latitude (°N)', fontsize=11)
    ax.set_ylabel('Pressure (hPa)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.grid(True, which='major', ls='--', lw=0.4, alpha=0.5)
    return im, cmap, norm, bounds

# ─── FIG 5.56: INSERTION SCHEMATIC ────────────────────────────────────────────

def make_fig_556(era5_times, model_name, out_dirs, ic_dt, sample_step=119):
    """Global map showing ERA5 T at 850 hPa and wind speed at 1000 hPa at day 30."""
    print(f"  Fig 5.56 schematic for {model_name}...")

    # Valid time at ~day 30 (last free-running step)
    if model_name == 'Aurora':
        _ds = nc4.Dataset(str(AURORA_TLOWER))
        lt = _ds['lead_time'][sample_step]
        lat_1d = _ds['lat'][:]
        lon_1d = _ds['lon'][:]
        _ds.close()
        vt = read_base_time(AURORA_TLOWER) + timedelta(hours=int(lt))
    elif model_name == 'FCN':
        ds = nc4.Dataset(str(FCN_TLOWER))
        t_h = float(ds['time'][sample_step])
        vt = datetime(1900,1,1) + timedelta(hours=t_h)
        lat_1d = ds['lat'][:]
        lon_1d = ds['lon'][:]
        ds.close()
    else:
        vt = PRITHVI_IC + timedelta(hours=(sample_step+1)*6)
        lat_1d = h5py.File(str(PRITHVI_TLOWER))['lat'][:]
        lon_1d = h5py.File(str(PRITHVI_TLOWER))['lon'][:]

    era5_i = era5_idx_for(era5_times, vt)

    # Load ERA5 T at 850 hPa
    t850_all = load_era5_pressure_level('t', 850)  # (T_all, lat, lon)
    t850 = t850_all[era5_i, :, :]         # (721, 1440) with north-to-south lat order?
    era5_lat = None
    # Get ERA5 lat/lon from surface file
    ds_tmp = xr.open_dataset(ERA5_DIR / 'era5_surface_2023_02.grib', engine='cfgrib',
                              backend_kwargs={'filter_by_keys': {'shortName': 'msl'}},
                              )
    era5_lat = ds_tmp.latitude.values
    era5_lon = ds_tmp.longitude.values
    ds_tmp.close()
    del t850_all; gc.collect()

    # Load ERA5 U/V at 1000 hPa
    u1000_all = load_era5_pressure_level('u', 1000)
    v1000_all = load_era5_pressure_level('v', 1000)
    ws1000 = np.sqrt(u1000_all[era5_i]**2 + v1000_all[era5_i]**2)
    del u1000_all, v1000_all; gc.collect()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Figure 5.56 — Lower-Level Insertion Schematic ({model_name})\n'
                 f'ERA5 fields at {vt.strftime("%Y-%m-%d %H:%M UTC")} (Day 30)',
                 fontsize=14, fontweight='bold', y=1.01)

    # Panel 1: T at 850 hPa
    t_anom = t850 - 273.15
    vmin_T, vmax_T = np.percentile(t_anom, [2, 98])
    cmap_T, norm_T, bounds_T = schematic_norm_T(np.floor(vmin_T), np.ceil(vmax_T), n=16)
    im0 = plot_global_map(axes[0], t_anom, era5_lat, era5_lon, cmap_T, norm_T,
                          'ERA5 Temperature at 850 hPa — T_LOWER Insertion Level', '°C')
    cb0 = add_colorbar(fig, axes[0], im0, 'Temperature (°C)',
                       ticks=np.linspace(np.floor(vmin_T), np.ceil(vmax_T), 9))
    axes[0].text(0.02, 0.95, '↓ Global insertion at 1000, 925, 850 hPa',
                 transform=axes[0].transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: Wind speed at 1000 hPa
    ws_max = np.percentile(ws1000, 98)
    cmap_W, norm_W, bounds_W = schematic_norm_wind(np.ceil(ws_max), n=16)
    im1 = plot_global_map(axes[1], ws1000, era5_lat, era5_lon, cmap_W, norm_W,
                          'ERA5 Wind Speed at 1000 hPa — UV_LOWER Insertion Level', 'm/s')
    add_colorbar(fig, axes[1], im1, 'Wind Speed (m/s)',
                 ticks=np.linspace(0, np.ceil(ws_max), 9))
    axes[1].text(0.02, 0.95, '↓ Global insertion at 1000, 925, 850 hPa',
                 transform=axes[1].transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    for d in out_dirs:
        fig.savefig(d / 'fig5_56_insertion_schematic.png', bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved Fig 5.56 to {len(out_dirs)} directories.")

# ─── AURORA ANALYSIS ──────────────────────────────────────────────────────────

def analyze_aurora(era5_times):
    print("\n=== AURORA ANALYSIS ===")

    # Load Aurora grids
    ds_tl = nc4.Dataset(str(AURORA_TLOWER))
    aurora_lat = ds_tl['lat'][:]           # (720,) descending 90→-89.75
    aurora_lon = ds_tl['lon'][:]
    aurora_lt  = ds_tl['lead_time'][:]     # hours, 12-step increments... wait, 6h
    ds_tl.close()

    # Valid times for experiment steps
    exp_valid = [read_base_time(AURORA_TLOWER) + timedelta(hours=int(lt)) for lt in aurora_lt]

    # ERA5 index for each experiment step
    era5_idx_exp = [era5_idx_for(era5_times, vt) for vt in exp_valid]

    # Aurora control valid times (12h steps: lt[n] = 12*n)
    ds_ctrl = nc4.Dataset(str(AURORA_CTRL))
    ctrl_lt_all = ds_ctrl['lead_time'][:]
    ctrl_adj_vt = [AURORA_IC + timedelta(hours=int(lt)) for lt in ctrl_lt_all[ACTRL_ADJ]]
    ctrl_mat_vt = [AURORA_IC + timedelta(hours=int(lt)) for lt in ctrl_lt_all[ACTRL_MAT]]
    era5_idx_cadj = [era5_idx_for(era5_times, vt) for vt in ctrl_adj_vt]
    era5_idx_cmat = [era5_idx_for(era5_times, vt) for vt in ctrl_mat_vt]
    ds_ctrl.close()

    # ERA5 lat (721, north-first) → align to Aurora lat (720, north-first, no -90)
    # ERA5 drops south-pole row when matching Aurora: era5_data[:-1, :]
    # Actually ERA5 lat order depends on GRIB; confirm later — we slice carefully.

    print("  Computing surface RMSE time series...")
    results = {}
    for model_name, exp_file, out_dir, var_label in [
        ('Aurora_T_LOWER',  AURORA_TLOWER,  AURORA_TDIR,  't'),
        ('Aurora_UV_LOWER', AURORA_UVLOWER, AURORA_UVDIR, 'uv'),
    ]:
        surf_exp = {}
        surf_ctrl = {}

        # ── T2M ──
        era5_t2m_all = load_era5_surface('2t')
        # ERA5 lat: 721 (north first), Aurora: 720 (north first, no -90)
        # Check ERA5 lat direction
        ds_tmp = xr.open_dataset(ERA5_DIR / 'era5_surface_2023_02.grib', engine='cfgrib',
                                  backend_kwargs={'filter_by_keys': {'shortName': '2t'}},
                                  )
        era5_lat_full = ds_tmp.latitude.values
        ds_tmp.close()
        # ERA5 lat descending (90→-90, 721 pts); Aurora has 720 pts (no -90° row) → drop last
        if era5_lat_full[0] > era5_lat_full[-1]:
            era5_t2m_matched = era5_t2m_all[:, :-1, :]  # (T, 720, 1440)
            era5_lat_aur = era5_lat_full[:-1]
        else:
            era5_t2m_matched = era5_t2m_all[:, 1:, :]
            era5_lat_aur = era5_lat_full[1:]
        del era5_t2m_all; gc.collect()

        ds_exp = nc4.Dataset(str(exp_file))
        rmse_t2m_exp  = np.zeros(N_STEPS)
        for s in range(N_STEPS):
            fc  = np.array(ds_exp['2t'][s])
            tr  = era5_t2m_matched[era5_idx_exp[s]]
            rmse_t2m_exp[s] = global_rmse(fc, tr, aurora_lat)
        ds_exp.close()

        rmse_t2m_ctrl = np.zeros(len(ctrl_lt_all))
        ds_ctrl = nc4.Dataset(str(AURORA_CTRL))
        for ci in range(len(ctrl_lt_all)):
            vt = AURORA_IC + timedelta(hours=int(ctrl_lt_all[ci]))
            ei = era5_idx_for(era5_times, vt)
            fc  = np.array(ds_ctrl['2t'][ci])        # (lead_time, lat, lon) — no time dim
            tr  = era5_t2m_matched[ei]
            rmse_t2m_ctrl[ci] = global_rmse(fc, tr, aurora_lat)
        ds_ctrl.close()

        surf_exp['T2M']  = rmse_t2m_exp
        surf_ctrl['T2M'] = rmse_t2m_ctrl
        del era5_t2m_matched; gc.collect()

        # ── 10m Wind Speed ──
        era5_u10_all = load_era5_surface('10u')
        era5_v10_all = load_era5_surface('10v')
        if era5_lat_full[0] > era5_lat_full[-1]:
            era5_u10 = era5_u10_all[:, :-1, :]; era5_v10 = era5_v10_all[:, :-1, :]
        else:
            era5_u10 = era5_u10_all[:, 1:, :]; era5_v10 = era5_v10_all[:, 1:, :]
        del era5_u10_all, era5_v10_all; gc.collect()

        era5_ws10 = np.sqrt(era5_u10**2 + era5_v10**2)
        del era5_u10, era5_v10; gc.collect()

        ds_exp = nc4.Dataset(str(exp_file))
        rmse_ws_exp = np.zeros(N_STEPS)
        for s in range(N_STEPS):
            fc_ws = np.sqrt(np.array(ds_exp['10u'][s])**2 + np.array(ds_exp['10v'][s])**2)
            rmse_ws_exp[s] = global_rmse(fc_ws, era5_ws10[era5_idx_exp[s]], aurora_lat)
        ds_exp.close()

        rmse_ws_ctrl = np.zeros(len(ctrl_lt_all))
        ds_ctrl = nc4.Dataset(str(AURORA_CTRL))
        for ci in range(len(ctrl_lt_all)):
            vt = AURORA_IC + timedelta(hours=int(ctrl_lt_all[ci]))
            ei = era5_idx_for(era5_times, vt)
            fc_ws = np.sqrt(np.array(ds_ctrl['10u'][ci])**2 + np.array(ds_ctrl['10v'][ci])**2)
            rmse_ws_ctrl[ci] = global_rmse(fc_ws, era5_ws10[ei], aurora_lat)
        ds_ctrl.close()

        surf_exp['WS10']  = rmse_ws_exp
        surf_ctrl['WS10'] = rmse_ws_ctrl
        del era5_ws10; gc.collect()

        # ── MSL Pressure ──
        era5_msl_all = load_era5_surface('msl')
        if era5_lat_full[0] > era5_lat_full[-1]:
            era5_msl = era5_msl_all[:, :-1, :] / 100.0  # Pa→hPa
        else:
            era5_msl = era5_msl_all[:, 1:, :] / 100.0
        del era5_msl_all; gc.collect()

        ds_exp = nc4.Dataset(str(exp_file))
        rmse_msl_exp = np.zeros(N_STEPS)
        for s in range(N_STEPS):
            fc_msl = np.array(ds_exp['msl'][s]) / 100.0
            rmse_msl_exp[s] = global_rmse(fc_msl, era5_msl[era5_idx_exp[s]], aurora_lat)
        ds_exp.close()

        rmse_msl_ctrl = np.zeros(len(ctrl_lt_all))
        ds_ctrl = nc4.Dataset(str(AURORA_CTRL))
        for ci in range(len(ctrl_lt_all)):
            vt = AURORA_IC + timedelta(hours=int(ctrl_lt_all[ci]))
            ei = era5_idx_for(era5_times, vt)
            fc_msl = np.array(ds_ctrl['msl'][ci]) / 100.0
            rmse_msl_ctrl[ci] = global_rmse(fc_msl, era5_msl[ei], aurora_lat)
        ds_ctrl.close()

        surf_exp['MSL']  = rmse_msl_exp
        surf_ctrl['MSL'] = rmse_msl_ctrl
        del era5_msl; gc.collect()

        # Save surface RMSE CSVs
        exp_days = [lt / 24.0 for lt in aurora_lt]
        ctrl_days_all = [lt / 24.0 for lt in ctrl_lt_all]
        df_exp = pd.DataFrame({
            'lead_day': exp_days, 'T2M_rmse': surf_exp['T2M'],
            'WS10_rmse': surf_exp['WS10'], 'MSL_rmse': surf_exp['MSL']
        })
        df_ctrl = pd.DataFrame({
            'lead_day': ctrl_days_all, 'T2M_rmse': surf_ctrl['T2M'],
            'WS10_rmse': surf_ctrl['WS10'], 'MSL_rmse': surf_ctrl['MSL']
        })
        df_exp.to_csv(out_dir / 'surface_rmse_experiment.csv', index=False)
        df_ctrl.to_csv(out_dir / 'surface_rmse_control.csv', index=False)

        results[model_name] = {
            'surf_exp': surf_exp, 'surf_ctrl': surf_ctrl,
            'exp_days': exp_days, 'ctrl_days': ctrl_days_all,
            'out_dir': out_dir, 'var_label': var_label,
        }

    print("  Computing vertical lat-pressure heatmaps (Aurora)...")
    # Per-model per-experiment heatmaps
    for model_name, exp_file, out_dir, var_label, pvar in [
        ('Aurora_T_LOWER',  AURORA_TLOWER,  AURORA_TDIR,  't', 't'),
        ('Aurora_UV_LOWER', AURORA_UVLOWER, AURORA_UVDIR, 'uv', 'u'),
    ]:
        n_lev = len(AF_LEVELS)
        hmap_adj_exp  = np.zeros((n_lev, len(LAT_CENTERS)))
        hmap_mat_exp  = np.zeros_like(hmap_adj_exp)
        hmap_adj_ctrl = np.zeros_like(hmap_adj_exp)
        hmap_mat_ctrl = np.zeros_like(hmap_adj_exp)

        for li, lev_hpa in enumerate(AF_LEVELS):
            print(f"    {model_name} level {lev_hpa} hPa...")
            era5_var_all = load_era5_pressure_level(pvar, lev_hpa)
            if era5_lat_full[0] > era5_lat_full[-1]:
                era5_var = era5_var_all[:, :-1, :]
            else:
                era5_var = era5_var_all[:, 1:, :]
            del era5_var_all; gc.collect()

            ds_exp  = nc4.Dataset(str(exp_file))
            ds_ctrl = nc4.Dataset(str(AURORA_CTRL))
            var_key_exp  = f'{pvar}{lev_hpa}'
            var_key_ctrl = f'{pvar}{lev_hpa}'

            # Experiment: adj & mat
            fc_adj  = np.array(ds_exp[var_key_exp][ADJ_SL])
            tr_adj  = era5_var[[era5_idx_exp[s] for s in range(ADJ_SL.start, ADJ_SL.stop)]]
            fc_mat  = np.array(ds_exp[var_key_exp][MAT_SL])
            tr_mat  = era5_var[[era5_idx_exp[s] for s in range(MAT_SL.start, MAT_SL.stop)]]

            # Control: adj & mat
            ctrl_adj_ei = era5_idx_cadj
            ctrl_mat_ei = era5_idx_cmat
            fc_cadj = np.array(ds_ctrl[var_key_ctrl][ACTRL_ADJ])
            tr_cadj = era5_var[ctrl_adj_ei]
            fc_cmat = np.array(ds_ctrl[var_key_ctrl][ACTRL_MAT])
            tr_cmat = era5_var[ctrl_mat_ei]

            ds_exp.close(); ds_ctrl.close()

            hmap_adj_exp[li]  = lat_band_rmse(fc_adj,  tr_adj,  aurora_lat)
            hmap_mat_exp[li]  = lat_band_rmse(fc_mat,  tr_mat,  aurora_lat)
            hmap_adj_ctrl[li] = lat_band_rmse(fc_cadj, tr_cadj, aurora_lat)
            hmap_mat_ctrl[li] = lat_band_rmse(fc_cmat, tr_cmat, aurora_lat)

            del era5_var, fc_adj, tr_adj, fc_mat, tr_mat, fc_cadj, tr_cadj, fc_cmat, tr_cmat
            gc.collect()

        # % improvement
        imp_adj = pct_improvement(hmap_adj_ctrl, hmap_adj_exp)
        imp_mat = pct_improvement(hmap_mat_ctrl, hmap_mat_exp)

        # Save CSVs
        for phase, imp, ctrl_h, exp_h in [('adj', imp_adj, hmap_adj_ctrl, hmap_adj_exp),
                                           ('mat', imp_mat, hmap_mat_ctrl, hmap_mat_exp)]:
            df = pd.DataFrame(imp, index=AF_LEVELS, columns=LAT_CENTERS)
            df.index.name = 'pressure_hPa'
            df.to_csv(out_dir / f'heatmap_pct_improvement_{phase}.csv')
            df2 = pd.DataFrame(ctrl_h, index=AF_LEVELS, columns=LAT_CENTERS)
            df2.to_csv(out_dir / f'heatmap_ctrl_rmse_{phase}.csv')
            df3 = pd.DataFrame(exp_h, index=AF_LEVELS, columns=LAT_CENTERS)
            df3.to_csv(out_dir / f'heatmap_exp_rmse_{phase}.csv')

        results[model_name]['imp_adj'] = imp_adj
        results[model_name]['imp_mat'] = imp_mat
        results[model_name]['pres_levels'] = AF_LEVELS

    return results


# ─── FCN ANALYSIS ─────────────────────────────────────────────────────────────

# FCN channel index lookup
FCN_CH = {
    'u10m': 0, 'v10m': 1, 't2m': 4, 'msl': 6,
    't50': 47, 't100': 48, 't150': 49, 't200': 50, 't250': 51,
    't300': 52, 't400': 53, 't500': 54, 't600': 55, 't700': 56,
    't850': 57, 't925': 58, 't1000': 59,
    'u50': 8,  'u100': 9,  'u150': 10, 'u200': 11, 'u250': 12,
    'u300': 13,'u400': 14, 'u500': 15, 'u600': 16, 'u700': 17,
    'u850': 18,'u925': 19, 'u1000': 20,
    'v50': 21, 'v100': 22, 'v150': 23, 'v200': 24, 'v250': 25,
    'v300': 26,'v400': 27, 'v500': 28, 'v600': 29, 'v700': 30,
    'v850': 31,'v925': 32, 'v1000': 33,
}

def fcn_valid_times(ds_fcn):
    """Return list of Python datetimes for all FCN experiment steps."""
    t_h = ds_fcn['time'][:]
    epoch = datetime(1900, 1, 1, 0, 0)
    return [epoch + timedelta(hours=float(h)) for h in t_h]

def fcn_ctrl_valid_times(ds_ctrl):
    """FCN control times: hours since IC 2023-02-01 00:00."""
    t_h = ds_ctrl['time'][:]
    fcn_ic = datetime(2023, 2, 1, 0, 0)
    return [fcn_ic + timedelta(hours=float(h)) for h in t_h]

def fcn_ch(ds, channel_name, step_slice):
    return np.array(ds['prediction'][step_slice, FCN_CH[channel_name], :, :])

def analyze_fcn(era5_times):
    print("\n=== FCN ANALYSIS ===")

    ds_tl = nc4.Dataset(str(FCN_TLOWER))
    fcn_lat = ds_tl['lat'][:]    # (721,) descending 90→-90
    fcn_lon = ds_tl['lon'][:]
    exp_valid_tl = fcn_valid_times(ds_tl)
    era5_idx_exp_tl = [era5_idx_for(era5_times, vt) for vt in exp_valid_tl]
    ds_tl.close()

    ds_uvl = nc4.Dataset(str(FCN_UVLOWER))
    exp_valid_uvl = fcn_valid_times(ds_uvl)
    era5_idx_exp_uvl = [era5_idx_for(era5_times, vt) for vt in exp_valid_uvl]
    ds_uvl.close()

    ds_ctrl = nc4.Dataset(str(FCN_CTRL))
    ctrl_vt = fcn_ctrl_valid_times(ds_ctrl)
    era5_idx_ctrl = [era5_idx_for(era5_times, vt) for vt in ctrl_vt]
    ds_ctrl.close()

    era5_idx_cadj = era5_idx_ctrl[FCTRL_ADJ.start:FCTRL_ADJ.stop]
    era5_idx_cmat = era5_idx_ctrl[FCTRL_MAT.start:FCTRL_MAT.stop]

    # ERA5 lat/lon for FCN (721 lats, north-first)
    ds_tmp = xr.open_dataset(ERA5_DIR / 'era5_surface_2023_02.grib', engine='cfgrib',
                              backend_kwargs={'filter_by_keys': {'shortName': '2t'}},
                              )
    era5_lat_fcn = ds_tmp.latitude.values   # 721, north-first
    ds_tmp.close()

    print("  Computing FCN surface RMSE...")
    results = {}
    for model_name, exp_file, out_dir, era5_idx_exp, var_label in [
        ('FCN_T_LOWER',  FCN_TLOWER,  FCN_TDIR,  era5_idx_exp_tl,  't'),
        ('FCN_UV_LOWER', FCN_UVLOWER, FCN_UVDIR, era5_idx_exp_uvl, 'uv'),
    ]:
        surf_exp = {}; surf_ctrl = {}

        # T2M
        era5_t2m = load_era5_surface('2t')
        ds_exp  = nc4.Dataset(str(exp_file))
        ds_ctrl = nc4.Dataset(str(FCN_CTRL))
        rmse_t2m_exp  = np.array([global_rmse(np.array(ds_exp['prediction'][s, FCN_CH['t2m']]),
                                               era5_t2m[era5_idx_exp[s]], fcn_lat) for s in range(N_STEPS)])
        rmse_t2m_ctrl = np.array([global_rmse(np.array(ds_ctrl['prediction'][c, FCN_CH['t2m']]),
                                               era5_t2m[era5_idx_ctrl[c]], fcn_lat) for c in range(len(ctrl_vt))])
        ds_exp.close(); ds_ctrl.close()
        surf_exp['T2M'] = rmse_t2m_exp; surf_ctrl['T2M'] = rmse_t2m_ctrl
        del era5_t2m; gc.collect()

        # WS10
        era5_u10 = load_era5_surface('10u')
        era5_v10 = load_era5_surface('10v')
        era5_ws10 = np.sqrt(era5_u10**2 + era5_v10**2)
        del era5_u10, era5_v10; gc.collect()
        ds_exp  = nc4.Dataset(str(exp_file)); ds_ctrl = nc4.Dataset(str(FCN_CTRL))
        rmse_ws_exp  = np.array([global_rmse(
            np.sqrt(np.array(ds_exp['prediction'][s, FCN_CH['u10m']])**2 +
                    np.array(ds_exp['prediction'][s, FCN_CH['v10m']])**2),
            era5_ws10[era5_idx_exp[s]], fcn_lat) for s in range(N_STEPS)])
        rmse_ws_ctrl = np.array([global_rmse(
            np.sqrt(np.array(ds_ctrl['prediction'][c, FCN_CH['u10m']])**2 +
                    np.array(ds_ctrl['prediction'][c, FCN_CH['v10m']])**2),
            era5_ws10[era5_idx_ctrl[c]], fcn_lat) for c in range(len(ctrl_vt))])
        ds_exp.close(); ds_ctrl.close()
        surf_exp['WS10'] = rmse_ws_exp; surf_ctrl['WS10'] = rmse_ws_ctrl
        del era5_ws10; gc.collect()

        # MSL
        era5_msl = load_era5_surface('msl') / 100.0
        ds_exp  = nc4.Dataset(str(exp_file)); ds_ctrl = nc4.Dataset(str(FCN_CTRL))
        rmse_msl_exp  = np.array([global_rmse(np.array(ds_exp['prediction'][s, FCN_CH['msl']]) / 100.0,
                                               era5_msl[era5_idx_exp[s]], fcn_lat) for s in range(N_STEPS)])
        rmse_msl_ctrl = np.array([global_rmse(np.array(ds_ctrl['prediction'][c, FCN_CH['msl']]) / 100.0,
                                               era5_msl[era5_idx_ctrl[c]], fcn_lat) for c in range(len(ctrl_vt))])
        ds_exp.close(); ds_ctrl.close()
        surf_exp['MSL'] = rmse_msl_exp; surf_ctrl['MSL'] = rmse_msl_ctrl
        del era5_msl; gc.collect()

        _fcn_ic_dt = datetime(2023, 2, 1, 0, 0)
        _evt = exp_valid_tl if 't' in var_label else exp_valid_uvl
        df_exp = pd.DataFrame({
            'lead_day': [(vt - _fcn_ic_dt).total_seconds() / 86400 for vt in _evt],
            'T2M_rmse': surf_exp['T2M'], 'WS10_rmse': surf_exp['WS10'], 'MSL_rmse': surf_exp['MSL']})
        df_ctrl = pd.DataFrame({
            'lead_day': [(vt - _fcn_ic_dt).total_seconds() / 86400 for vt in ctrl_vt],
            'T2M_rmse': surf_ctrl['T2M'], 'WS10_rmse': surf_ctrl['WS10'], 'MSL_rmse': surf_ctrl['MSL']})
        df_exp.to_csv(out_dir / 'surface_rmse_experiment.csv', index=False)
        df_ctrl.to_csv(out_dir / 'surface_rmse_control.csv', index=False)

        results[model_name] = {'surf_exp': surf_exp, 'surf_ctrl': surf_ctrl,
                                'exp_days': df_exp['lead_day'].tolist(),
                                'out_dir': out_dir, 'var_label': var_label}

    print("  Computing FCN vertical heatmaps...")
    for model_name, exp_file, out_dir, era5_idx_exp_steps, var_label, pvar_prefix in [
        ('FCN_T_LOWER',  FCN_TLOWER,  FCN_TDIR,  era5_idx_exp_tl,  't',  't'),
        ('FCN_UV_LOWER', FCN_UVLOWER, FCN_UVDIR, era5_idx_exp_uvl, 'uv', 'u'),
    ]:
        n_lev = len(AF_LEVELS)
        hmap_adj_exp  = np.zeros((n_lev, len(LAT_CENTERS)))
        hmap_mat_exp  = np.zeros_like(hmap_adj_exp)
        hmap_adj_ctrl = np.zeros_like(hmap_adj_exp)
        hmap_mat_ctrl = np.zeros_like(hmap_adj_exp)

        for li, lev_hpa in enumerate(AF_LEVELS):
            print(f"    {model_name} level {lev_hpa} hPa...")
            era5_var = load_era5_pressure_level(pvar_prefix, lev_hpa)  # (T_all, 721, 1440)

            ds_exp  = nc4.Dataset(str(exp_file))
            ds_ctrl = nc4.Dataset(str(FCN_CTRL))
            ch_key = f'{pvar_prefix}{lev_hpa}'
            ch_idx = FCN_CH[ch_key]

            adj_steps_idx = list(range(ADJ_SL.start, ADJ_SL.stop))
            mat_steps_idx = list(range(MAT_SL.start, MAT_SL.stop))
            cadj_steps    = list(range(FCTRL_ADJ.start, FCTRL_ADJ.stop))
            cmat_steps    = list(range(FCTRL_MAT.start, FCTRL_MAT.stop))

            fc_adj  = np.array(ds_exp['prediction'][ADJ_SL, ch_idx, :, :])
            tr_adj  = era5_var[[era5_idx_exp_steps[s] for s in adj_steps_idx]]
            fc_mat  = np.array(ds_exp['prediction'][MAT_SL, ch_idx, :, :])
            tr_mat  = era5_var[[era5_idx_exp_steps[s] for s in mat_steps_idx]]

            fc_cadj = np.array(ds_ctrl['prediction'][FCTRL_ADJ, ch_idx, :, :])
            tr_cadj = era5_var[era5_idx_cadj]
            fc_cmat = np.array(ds_ctrl['prediction'][FCTRL_MAT, ch_idx, :, :])
            tr_cmat = era5_var[era5_idx_cmat]

            ds_exp.close(); ds_ctrl.close()

            hmap_adj_exp[li]  = lat_band_rmse(fc_adj,  tr_adj,  fcn_lat)
            hmap_mat_exp[li]  = lat_band_rmse(fc_mat,  tr_mat,  fcn_lat)
            hmap_adj_ctrl[li] = lat_band_rmse(fc_cadj, tr_cadj, fcn_lat)
            hmap_mat_ctrl[li] = lat_band_rmse(fc_cmat, tr_cmat, fcn_lat)

            del era5_var, fc_adj, tr_adj, fc_mat, tr_mat, fc_cadj, tr_cadj, fc_cmat, tr_cmat
            gc.collect()

        imp_adj = pct_improvement(hmap_adj_ctrl, hmap_adj_exp)
        imp_mat = pct_improvement(hmap_mat_ctrl, hmap_mat_exp)

        for phase, imp, ctrl_h, exp_h in [('adj', imp_adj, hmap_adj_ctrl, hmap_adj_exp),
                                           ('mat', imp_mat, hmap_mat_ctrl, hmap_mat_exp)]:
            pd.DataFrame(imp, index=AF_LEVELS, columns=LAT_CENTERS).to_csv(
                out_dir / f'heatmap_pct_improvement_{phase}.csv')
            pd.DataFrame(ctrl_h, index=AF_LEVELS, columns=LAT_CENTERS).to_csv(
                out_dir / f'heatmap_ctrl_rmse_{phase}.csv')
            pd.DataFrame(exp_h, index=AF_LEVELS, columns=LAT_CENTERS).to_csv(
                out_dir / f'heatmap_exp_rmse_{phase}.csv')

        results[model_name]['imp_adj']    = imp_adj
        results[model_name]['imp_mat']    = imp_mat
        results[model_name]['pres_levels'] = AF_LEVELS

    return results


# ─── PRITHVI ANALYSIS ─────────────────────────────────────────────────────────

def prithvi_valid_times():
    """Valid times for all 240 Prithvi experiment steps (6h from IC)."""
    return [PRITHVI_IC + timedelta(hours=(s + 1) * 6) for s in range(N_STEPS)]

def analyze_prithvi():
    print("\n=== PRITHVI ANALYSIS ===")

    f_tl  = h5py.File(str(PRITHVI_TLOWER), 'r')
    pri_lat_exp = np.array(f_tl['lat'])   # (360,) -89.5 → 90, ascending
    pri_lon_exp = np.array(f_tl['lon'])
    f_tl.close()

    f_ctrl  = h5py.File(str(PRITHVI_CTRL), 'r')
    pri_lat_ctrl = np.array(f_ctrl['lat'])  # (361,) -90 → 90, ascending (includes -90)
    f_ctrl.close()

    exp_valid = prithvi_valid_times()
    # ctrl same 6h steps
    ctrl_valid = [PRITHVI_IC + timedelta(hours=(s + 1) * 6) for s in range(N_STEPS)]

    # For ctrl, skip lat index 0 (-90°) → [1:, :]
    n_lev_pri = len(PRI_LEV_VALS)

    print("  Computing Prithvi surface RMSE...")
    results = {}
    for model_name, exp_file, out_dir, var_label in [
        ('Prithvi_T_LOWER',  PRITHVI_TLOWER,  PRITHVI_TDIR,  't'),
        ('Prithvi_UV_LOWER', PRITHVI_UVLOWER, PRITHVI_UVDIR, 'uv'),
    ]:
        surf_exp = {}; surf_ctrl = {}

        # T2M
        rmse_t2m_exp  = np.zeros(N_STEPS)
        rmse_t2m_ctrl = np.zeros(N_STEPS)
        f_exp  = h5py.File(str(exp_file), 'r')
        f_ctrl_ = h5py.File(str(PRITHVI_CTRL), 'r')
        for s in range(N_STEPS):
            vt = exp_valid[s]
            merra_t2m = load_merra2_sfc_step(vt, 'T2M')   # (361, 576)
            tr = merra_t2m[1:, :]                          # (360, 576) skip -90
            fc_exp  = np.array(f_exp['T2M'][s])            # (360, 576)
            fc_ctrl = np.array(f_ctrl_['T2M'][s])[1:, :]   # (361→360, 576)
            rmse_t2m_exp[s]  = global_rmse(fc_exp,  tr, pri_lat_exp)
            rmse_t2m_ctrl[s] = global_rmse(fc_ctrl, tr, pri_lat_exp)
        f_exp.close(); f_ctrl_.close()
        surf_exp['T2M'] = rmse_t2m_exp; surf_ctrl['T2M'] = rmse_t2m_ctrl

        # WS10
        rmse_ws_exp  = np.zeros(N_STEPS)
        rmse_ws_ctrl = np.zeros(N_STEPS)
        f_exp  = h5py.File(str(exp_file), 'r')
        f_ctrl_ = h5py.File(str(PRITHVI_CTRL), 'r')
        for s in range(N_STEPS):
            vt = exp_valid[s]
            u10m_tr = load_merra2_sfc_step(vt, 'U10M')[1:]  # (360, 576)
            v10m_tr = load_merra2_sfc_step(vt, 'V10M')[1:]
            ws_tr = np.sqrt(u10m_tr**2 + v10m_tr**2)
            ws_exp  = np.sqrt(np.array(f_exp['U10M'][s])**2  + np.array(f_exp['V10M'][s])**2)
            ws_ctrl = np.sqrt(np.array(f_ctrl_['U10M'][s])[1:]**2 + np.array(f_ctrl_['V10M'][s])[1:]**2)
            rmse_ws_exp[s]  = global_rmse(ws_exp,  ws_tr, pri_lat_exp)
            rmse_ws_ctrl[s] = global_rmse(ws_ctrl, ws_tr, pri_lat_exp)
        f_exp.close(); f_ctrl_.close()
        surf_exp['WS10'] = rmse_ws_exp; surf_ctrl['WS10'] = rmse_ws_ctrl

        # MSL (SLP in Pa → hPa)
        rmse_msl_exp  = np.zeros(N_STEPS)
        rmse_msl_ctrl = np.zeros(N_STEPS)
        f_exp  = h5py.File(str(exp_file), 'r')
        f_ctrl_ = h5py.File(str(PRITHVI_CTRL), 'r')
        for s in range(N_STEPS):
            vt = exp_valid[s]
            merra_slp = load_merra2_sfc_step(vt, 'SLP')[1:] / 100.0   # (360, 576) hPa
            fc_exp  = np.array(f_exp['SLP'][s]) / 100.0
            fc_ctrl = np.array(f_ctrl_['SLP'][s])[1:] / 100.0
            rmse_msl_exp[s]  = global_rmse(fc_exp,  merra_slp, pri_lat_exp)
            rmse_msl_ctrl[s] = global_rmse(fc_ctrl, merra_slp, pri_lat_exp)
        f_exp.close(); f_ctrl_.close()
        surf_exp['MSL'] = rmse_msl_exp; surf_ctrl['MSL'] = rmse_msl_ctrl

        exp_days = [(s + 1) * 6 / 24.0 for s in range(N_STEPS)]
        pd.DataFrame({'lead_day': exp_days, **{k: surf_exp[k]  for k in surf_exp}}).to_csv(
            out_dir / 'surface_rmse_experiment.csv', index=False)
        pd.DataFrame({'lead_day': exp_days, **{k: surf_ctrl[k] for k in surf_ctrl}}).to_csv(
            out_dir / 'surface_rmse_control.csv', index=False)

        results[model_name] = {'surf_exp': surf_exp, 'surf_ctrl': surf_ctrl,
                                'exp_days': exp_days, 'out_dir': out_dir, 'var_label': var_label}

    print("  Computing Prithvi vertical heatmaps...")
    for model_name, exp_file, out_dir, var_label, pvar_merra in [
        ('Prithvi_T_LOWER',  PRITHVI_TLOWER,  PRITHVI_TDIR,  't',  'T'),
        ('Prithvi_UV_LOWER', PRITHVI_UVLOWER, PRITHVI_UVDIR, 'uv', 'U'),
    ]:
        hmap_adj_exp  = np.zeros((n_lev_pri, len(LAT_CENTERS)))
        hmap_mat_exp  = np.zeros_like(hmap_adj_exp)
        hmap_adj_ctrl = np.zeros_like(hmap_adj_exp)
        hmap_mat_ctrl = np.zeros_like(hmap_adj_exp)

        # Pre-load step indices for each phase
        adj_steps = list(range(ADJ_SL.start, ADJ_SL.stop))
        mat_steps = list(range(MAT_SL.start, MAT_SL.stop))

        for li, lev_val in enumerate(PRI_LEV_VALS):
            print(f"    {model_name} lev {lev_val} (~{PRI_LEV_HPA[li]} hPa)...")
            # MERRA lev index (descending in file)
            merra_li = MERRA_LEV_DESC.index(int(lev_val))

            f_exp   = h5py.File(str(exp_file), 'r')
            f_ctrl_ = h5py.File(str(PRITHVI_CTRL), 'r')

            fc_adj_e  = np.array(f_exp[pvar_merra][ADJ_SL, li,  :, :])   # (60, 360, 576)
            fc_mat_e  = np.array(f_exp[pvar_merra][MAT_SL, li,  :, :])
            fc_adj_c  = np.array(f_ctrl_[pvar_merra][ADJ_SL, li, 1:, :]) # (60, 360, 576)
            fc_mat_c  = np.array(f_ctrl_[pvar_merra][MAT_SL, li, 1:, :])
            f_exp.close(); f_ctrl_.close()

            # Build truth arrays from MERRA step by step
            tr_adj_list = []; tr_mat_list = []
            for s in adj_steps:
                vt = exp_valid[s]
                merra_slice = load_merra2_pres_step(vt, pvar_merra, lev_val)  # (361, 576)
                tr_adj_list.append(merra_slice[1:, :])   # skip -90
            for s in mat_steps:
                vt = exp_valid[s]
                merra_slice = load_merra2_pres_step(vt, pvar_merra, lev_val)
                tr_mat_list.append(merra_slice[1:, :])

            tr_adj = np.array(tr_adj_list)   # (60, 360, 576)
            tr_mat = np.array(tr_mat_list)

            hmap_adj_exp[li]  = lat_band_rmse(fc_adj_e, tr_adj, pri_lat_exp)
            hmap_mat_exp[li]  = lat_band_rmse(fc_mat_e, tr_mat, pri_lat_exp)
            hmap_adj_ctrl[li] = lat_band_rmse(fc_adj_c, tr_adj, pri_lat_exp)
            hmap_mat_ctrl[li] = lat_band_rmse(fc_mat_c, tr_mat, pri_lat_exp)

            del fc_adj_e, fc_mat_e, fc_adj_c, fc_mat_c, tr_adj, tr_mat, tr_adj_list, tr_mat_list
            gc.collect()

        imp_adj = pct_improvement(hmap_adj_ctrl, hmap_adj_exp)
        imp_mat = pct_improvement(hmap_mat_ctrl, hmap_mat_exp)

        for phase, imp, ctrl_h, exp_h in [('adj', imp_adj, hmap_adj_ctrl, hmap_adj_exp),
                                           ('mat', imp_mat, hmap_mat_ctrl, hmap_mat_exp)]:
            pd.DataFrame(imp, index=PRI_LEV_HPA, columns=LAT_CENTERS).to_csv(
                out_dir / f'heatmap_pct_improvement_{phase}.csv')
            pd.DataFrame(ctrl_h, index=PRI_LEV_HPA, columns=LAT_CENTERS).to_csv(
                out_dir / f'heatmap_ctrl_rmse_{phase}.csv')
            pd.DataFrame(exp_h, index=PRI_LEV_HPA, columns=LAT_CENTERS).to_csv(
                out_dir / f'heatmap_exp_rmse_{phase}.csv')

        results[model_name]['imp_adj']    = imp_adj
        results[model_name]['imp_mat']    = imp_mat
        results[model_name]['pres_levels'] = PRI_LEV_HPA

    return results


# ─── FIG 5.57/5.58: VERTICAL HEATMAPS ────────────────────────────────────────

def make_heatmaps(results_dict):
    """Generate Figs 5.57 (T lower) and 5.58 (UV lower) per model."""
    fig_map = {
        'Aurora_T_LOWER':  ('5.57', 'Aurora', AURORA_TDIR),
        'FCN_T_LOWER':     ('5.57', 'FCN',    FCN_TDIR),
        'Prithvi_T_LOWER': ('5.57', 'Prithvi',PRITHVI_TDIR),
        'Aurora_UV_LOWER': ('5.58', 'Aurora', AURORA_UVDIR),
        'FCN_UV_LOWER':    ('5.58', 'FCN',    FCN_UVDIR),
        'Prithvi_UV_LOWER':('5.58', 'Prithvi',PRITHVI_UVDIR),
    }
    exp_type_T_label  = 'Temperature (T) — Lower-Level Insertion'
    exp_type_UV_label = 'Wind (UV) — Lower-Level Insertion'

    for exp_key, (fignum, model, out_dir) in fig_map.items():
        if exp_key not in results_dict:
            continue
        r = results_dict[exp_key]
        imp_adj    = r['imp_adj']     # (n_lev, n_lat)
        imp_mat    = r['imp_mat']
        pres_levels = r['pres_levels']
        var_label   = r['var_label']
        exp_label   = exp_type_T_label if var_label == 't' else exp_type_UV_label
        insert_hpa  = '1000/925/850 hPa' if model in ('Aurora', 'FCN') else '~960/946/903 hPa'

        cmap, norm, bounds = heatmap_norm()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        fig.suptitle(f'Figure {fignum} — Vertical Response: {model} — {exp_label}\n'
                     f'Inserted levels: {insert_hpa} (global)',
                     fontsize=13, fontweight='bold')

        im_ref = None
        for ax, data, title in [
            (axes[0], imp_adj, 'Adjustment Phase (Days 30–45)\n% RMSE Improvement'),
            (axes[1], imp_mat, 'Mature Phase (Days 45–60)\n% RMSE Improvement'),
        ]:
            im, _, _, _ = plot_heatmap_panel(ax, data, pres_levels, title)
            if im_ref is None:
                im_ref = im
            # Mark inserted levels
            insert_hpas_h = [1000, 925, 850] if model in ('Aurora', 'FCN') else [960, 946, 903]
            for k, il_hpa in enumerate(insert_hpas_h):
                nearest = pres_levels[np.argmin(np.abs(pres_levels - il_hpa))]
                ax.axhline(nearest, color='cyan', lw=1.5, ls='--', alpha=0.85,
                           label='Inserted levels' if k == 0 else '_')
            ax.legend(loc='upper right', fontsize=9)

        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          ax=axes, shrink=0.9, pad=0.02, location='right')
        cb.set_label('RMSE Improvement (%)', fontsize=12)
        cb.set_ticks(bounds[::2])

        figname = f'fig{fignum.replace(".","_")}_{var_label}_lower_vertical_heatmap_{model.lower()}.png'
        fig.savefig(out_dir / figname, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {figname} → {out_dir}")


# ─── FIG 5.59/5.60: SURFACE IMPACT ───────────────────────────────────────────

def make_surface_impact(results_dict):
    """Generate Figs 5.59 (T lower) and 5.60 (UV lower) per model."""
    panel_cfg = {
        'T2M': ('T2M RMSE', 'RMSE (K)', (0, None)),
        'WS10':('10m Wind Speed RMSE', 'RMSE (m/s)', (0, None)),
        'MSL': ('MSLP RMSE', 'RMSE (hPa)', (0, None)),
    }
    surf_vars = ['T2M', 'WS10', 'MSL']

    for fignum, var_label, keys in [
        ('5.59', 't',  ['Aurora_T_LOWER', 'FCN_T_LOWER', 'Prithvi_T_LOWER']),
        ('5.60', 'uv', ['Aurora_UV_LOWER','FCN_UV_LOWER','Prithvi_UV_LOWER']),
    ]:
        exp_name = 'T Lower-Level Insertion' if var_label == 't' else 'UV Lower-Level Insertion'
        for key in keys:
            if key not in results_dict:
                continue
            r = results_dict[key]
            model = key.split('_')[0]
            out_dir = r['out_dir']

            surf_exp  = r['surf_exp']
            surf_ctrl = r['surf_ctrl']
            exp_days  = np.array(r['exp_days'])

            # Build ctrl days
            if model == 'Prithvi':
                ctrl_days = exp_days  # same 6h steps
            else:
                # Aurora/FCN ctrl at 12h steps
                ctrl_lt_key = AURORA_CTRL if model == 'Aurora' else FCN_CTRL
                ds_c = nc4.Dataset(str(ctrl_lt_key))
                if model == 'Aurora':
                    ctrl_days = np.array(ds_c['lead_time'][:]) / 24.0
                else:
                    ctrl_days = np.array(ds_c['time'][:]) / 24.0
                ds_c.close()

            fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
            fig.suptitle(f'Figure {fignum} — Surface Impact: {model} — {exp_name}\n'
                         f'Global Cosine-Weighted RMSE vs Lead Time',
                         fontsize=13, fontweight='bold')

            colors = {'exp': '#d62728', 'ctrl': '#1f77b4'}
            for ax, sv in zip(axes, surf_vars):
                label_name, ylabel, ylim = panel_cfg[sv]
                ax.plot(exp_days, surf_exp[sv],  color=colors['exp'],  lw=2,
                        label='Experiment (Lower Insertion)')
                ax.plot(ctrl_days, surf_ctrl[sv], color=colors['ctrl'], lw=2,
                        ls='--', label='Control (Free Running)')
                ax.axvline(30, color='k', lw=1.2, ls=':', alpha=0.7, label='Insertion Start (Day 30)')
                ax.set_xlabel('Lead Time (Days)', fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(label_name, fontsize=12, fontweight='bold')
                ax.set_xlim(0, 60)
                ax.grid(True, ls='--', lw=0.4, alpha=0.6)
                ax.legend(fontsize=9)

                # Shade injection period
                ax.axvspan(30, 60, alpha=0.06, color='red', label='_')

            figname = f'fig{fignum.replace(".","_")}_{var_label}_lower_surface_impact_{model.lower()}.png'
            fig.savefig(out_dir / figname, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {figname} → {out_dir}")


# ─── FIG 5.61: PROPAGATION EFFICIENCY BAR CHART ───────────────────────────────

def compute_propagation_efficiency(imp_2d, pres_levels, is_prithvi=False):
    """Mean % improvement per level group (inserted/mid/upper). Returns dict."""
    group_fn = level_group_pri if is_prithvi else level_group_af
    groups   = {0: [], 1: [], 2: []}
    for li, hpa in enumerate(pres_levels):
        g = group_fn(hpa)
        # global mean over lat bands
        val = float(np.nanmean(imp_2d[li]))
        groups[g].append(val)
    return {k: float(np.nanmean(v)) if v else 0.0 for k, v in groups.items()}

def make_propagation_efficiency(all_results, combined_dir):
    """Figure 5.61: grouped bar chart of propagation efficiency."""
    print("\n  Generating Fig 5.61 — Propagation Efficiency Bar Chart...")

    order = [
        ('Aurora_T_LOWER',  'Aurora\nT Lower',  False),
        ('Aurora_UV_LOWER', 'Aurora\nUV Lower', False),
        ('FCN_T_LOWER',     'FCN\nT Lower',     False),
        ('FCN_UV_LOWER',    'FCN\nUV Lower',    False),
        ('Prithvi_T_LOWER', 'Prithvi\nT Lower', True),
        ('Prithvi_UV_LOWER','Prithvi\nUV Lower',True),
    ]

    labels = [o[1] for o in order if o[0] in all_results]
    grp_colors = ['#2ecc71', '#f39c12', '#e74c3c']  # inserted/mid/upper

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle('Figure 5.61 — Vertical Propagation Efficiency\n'
                 '% RMSE Improvement at Each Level Group (Adjustment and Mature Phases)',
                 fontsize=14, fontweight='bold')

    for ax, phase_key, phase_label in [(axes[0], 'imp_adj', 'Adjustment Phase (Days 30–45)'),
                                        (axes[1], 'imp_mat', 'Mature Phase (Days 45–60)')]:
        n_exp = sum(1 for o in order if o[0] in all_results)
        bar_width = 0.25
        x = np.arange(n_exp)

        for gi, (glabel, color) in enumerate(zip(GROUP_LABELS, grp_colors)):
            vals = []
            for key, _, is_pri in order:
                if key not in all_results:
                    continue
                r   = all_results[key]
                eff = compute_propagation_efficiency(r[phase_key], r['pres_levels'], is_pri)
                vals.append(eff[gi])
            bars = ax.bar(x + gi * bar_width, vals, bar_width,
                          label=glabel, color=color, edgecolor='k', lw=0.6)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Mean % RMSE Improvement', fontsize=12)
        ax.set_title(phase_label, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.axhline(0, color='k', lw=0.8)
        ax.grid(True, axis='y', ls='--', lw=0.4, alpha=0.6)
        ax.set_ylim(bottom=min(-5, ax.get_ylim()[0] - 2))

    for root in [AURORA_ROOT, FCN_ROOT, PRITHVI_ROOT]:
        fig.savefig(root / 'fig5_61_propagation_efficiency.png', bbox_inches='tight')
    fig.savefig(BASE / 'fig5_61_propagation_efficiency_all_models.png', bbox_inches='tight')
    plt.close(fig)
    print("  Saved Fig 5.61 to model root dirs and yashd1/.")


# ─── TABLE 5.12 ───────────────────────────────────────────────────────────────

def make_table_512(all_results, combined_dir):
    """Table 5.12: vertical propagation summary."""
    print("\n  Generating Table 5.12...")
    rows = []

    exp_info = {
        'Aurora_T_LOWER':  ('Aurora', 'T_LOWER', '1000/925/850 hPa'),
        'Aurora_UV_LOWER': ('Aurora', 'UV_LOWER','1000/925/850 hPa'),
        'FCN_T_LOWER':     ('FCN',    'T_LOWER', '1000/925/850 hPa'),
        'FCN_UV_LOWER':    ('FCN',    'UV_LOWER','1000/925/850 hPa'),
        'Prithvi_T_LOWER': ('Prithvi','T_LOWER', '~960/946/903 hPa'),
        'Prithvi_UV_LOWER':('Prithvi','UV_LOWER','~960/946/903 hPa'),
    }

    for key, (model, exp, inserted) in exp_info.items():
        if key not in all_results:
            continue
        r = all_results[key]
        pres = r['pres_levels']
        is_pri = (model == 'Prithvi')

        for phase_key, phase_label in [('imp_adj', 'Adjustment'), ('imp_mat', 'Mature')]:
            imp_global = np.nanmean(r[phase_key], axis=1)  # (n_lev,)

            # Max response level (non-inserted levels only)
            group_fn = level_group_pri if is_pri else level_group_af
            non_ins_mask = np.array([group_fn(h) != 0 for h in pres])
            if non_ins_mask.any():
                max_resp_idx = np.argmax(imp_global * non_ins_mask)
                max_resp_hpa = int(pres[max_resp_idx])
                max_resp_imp = float(imp_global[max_resp_idx])
            else:
                max_resp_hpa = 0; max_resp_imp = 0.0

            # Propagation efficiency: upper-level improvement / inserted-level improvement
            ins_imp  = np.nanmean(imp_global[[group_fn(h) == 0 for h in pres]])
            up_imp   = np.nanmean(imp_global[[group_fn(h) == 2 for h in pres]])
            prop_eff = float(up_imp / ins_imp * 100) if ins_imp > 0 else 0.0

            rows.append({
                'Model': model, 'Experiment': exp, 'Phase': phase_label,
                'Inserted_Levels_hPa': inserted,
                'Max_Response_Level_hPa': max_resp_hpa,
                'Max_Response_Improvement_pct': round(max_resp_imp, 2),
                'Propagation_Efficiency_pct': round(prop_eff, 2),
                'Mean_Inserted_Level_Improvement_pct': round(float(ins_imp), 2),
                'Mean_Upper_Level_Improvement_pct': round(float(up_imp), 2),
            })

    df = pd.DataFrame(rows)
    for root in [AURORA_ROOT, FCN_ROOT, PRITHVI_ROOT]:
        df.to_csv(root / 'table5_12_vertical_propagation_summary.csv', index=False)
    df.to_csv(BASE / 'table5_12_vertical_propagation_summary.csv', index=False)
    print(df.to_string(index=False))
    print("  Saved Table 5.12.")
    return df


# ─── COMBINED 3-MODEL HEATMAP FIGURES ─────────────────────────────────────────

def make_combined_heatmap(all_results, var_label, fignum, fig_title):
    """One figure with 3 model rows × 2 phase columns."""
    model_keys = [f'{m}_{var_label.upper()}_LOWER' for m in ['Aurora', 'FCN', 'Prithvi']]
    model_keys_alt = {
        't':  ['Aurora_T_LOWER', 'FCN_T_LOWER', 'Prithvi_T_LOWER'],
        'uv': ['Aurora_UV_LOWER','FCN_UV_LOWER','Prithvi_UV_LOWER'],
    }[var_label]

    valid = [(k, k.split('_')[0]) for k in model_keys_alt if k in all_results]
    if not valid:
        return

    cmap, norm, bounds = heatmap_norm()
    n_models = len(valid)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 5 * n_models), constrained_layout=True)
    if n_models == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(fig_title, fontsize=14, fontweight='bold')

    for row_i, (key, model) in enumerate(valid):
        r  = all_results[key]
        pl = r['pres_levels']
        for col_i, (phase_key, phase_label) in enumerate([
            ('imp_adj', 'Adjustment Phase (Days 30–45)'),
            ('imp_mat', 'Mature Phase (Days 45–60)'),
        ]):
            ax = axes[row_i, col_i]
            im, _, _, _ = plot_heatmap_panel(ax, r[phase_key], pl,
                                              f'{model} — {phase_label}')
            insert_hpas = [1000, 925, 850] if model in ('Aurora', 'FCN') else [960, 946, 903]
            for il in insert_hpas:
                nearest = pl[np.argmin(np.abs(pl - il))]
                ax.axhline(nearest, color='cyan', lw=1.2, ls='--', alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, location='right', shrink=0.95, pad=0.01)
    cb.set_label('RMSE Improvement (%)', fontsize=12)
    cb.set_ticks(bounds[::2])

    out_name = f'fig{fignum.replace(".","_")}_{var_label}_lower_vertical_heatmap_combined.png'
    for root in [AURORA_ROOT, FCN_ROOT, PRITHVI_ROOT]:
        fig.savefig(root / out_name, bbox_inches='tight')
    fig.savefig(BASE / out_name, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined heatmap: {out_name}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=== VERTICAL PROPAGATION ANALYSIS — Section 5.10 ===")
    print(f"Start: {datetime.now()}")

    all_results = {}

    # ── ERA5 time array (shared for Aurora & FCN) ──
    print("\nLoading ERA5 time array...")
    era5_times = load_era5_times()
    print(f"  ERA5 times: {len(era5_times)} entries, "
          f"{str(era5_times[0])[:16]} → {str(era5_times[-1])[:16]}")

    # ── Fig 5.56: Insertion Schematic ──
    print("\nGenerating Fig 5.56 schematics...")
    make_fig_556(era5_times, 'Aurora',
                 [AURORA_TDIR, AURORA_UVDIR], AURORA_IC, sample_step=119)
    make_fig_556(era5_times, 'FCN',
                 [FCN_TDIR, FCN_UVDIR], datetime(2023,2,1,0,0), sample_step=119)
    # For Prithvi, use MERRA2 fields — generate separately
    print("  (Prithvi schematic will use MERRA2 fields at day 30)")
    # Simple Prithvi schematic: load MERRA T/UV at day 30
    vt_pri_d30 = PRITHVI_IC + timedelta(hours=120 * 6)  # step 119+1
    merra_T850_d30 = load_merra2_pres_step(vt_pri_d30, 'T', 68)   # lev 68 ≈ 903 hPa (nearest lower)
    merra_U1000_d30 = load_merra2_pres_step(vt_pri_d30, 'U', 72)  # lev 72 ≈ 960 hPa
    merra_V1000_d30 = load_merra2_pres_step(vt_pri_d30, 'V', 72)
    merra_WS_d30 = np.sqrt(merra_U1000_d30**2 + merra_V1000_d30**2)
    f_pri = h5py.File(str(PRITHVI_TLOWER), 'r')
    pri_lat = np.array(f_pri['lat']); pri_lon = np.array(f_pri['lon']); f_pri.close()
    # MERRA2 lat is 361 (-90→90); trim to 360 (skip -90)
    merra_lat_all = np.linspace(-90, 90, 361)
    pri_lat_m     = merra_lat_all[1:]   # 360 pts

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Figure 5.56 — Lower-Level Insertion Schematic (Prithvi)\n'
                 f'MERRA-2 fields at {vt_pri_d30.strftime("%Y-%m-%d %H:%M UTC")} (Day 30)',
                 fontsize=14, fontweight='bold', y=1.01)
    T_anom = merra_T850_d30[1:] - 273.15   # skip -90 row
    vT_min, vT_max = np.percentile(T_anom, [2, 98])
    cmap_T, norm_T, _ = schematic_norm_T(np.floor(vT_min), np.ceil(vT_max), n=16)
    im0 = plot_global_map(axes[0], T_anom, pri_lat_m, pri_lon,
                          cmap_T, norm_T, 'MERRA-2 Temperature at ~903 hPa — T_LOWER Level', '°C')
    add_colorbar(fig, axes[0], im0, 'Temperature (°C)')
    ws_max = np.percentile(merra_WS_d30[1:], 98)
    cmap_W, norm_W, _ = schematic_norm_wind(np.ceil(ws_max), n=16)
    im1 = plot_global_map(axes[1], merra_WS_d30[1:], pri_lat_m, pri_lon,
                          cmap_W, norm_W, 'MERRA-2 Wind Speed at ~960 hPa — UV_LOWER Level', 'm/s')
    add_colorbar(fig, axes[1], im1, 'Wind Speed (m/s)')
    plt.tight_layout()
    for d in [PRITHVI_TDIR, PRITHVI_UVDIR]:
        fig.savefig(d / 'fig5_56_insertion_schematic.png', bbox_inches='tight')
    plt.close(fig)

    # ── Aurora Analysis ──
    aurora_results = analyze_aurora(era5_times)
    all_results.update(aurora_results)

    # ── FCN Analysis ──
    fcn_results = analyze_fcn(era5_times)
    all_results.update(fcn_results)

    # ── Prithvi Analysis ──
    prithvi_results = analyze_prithvi()
    all_results.update(prithvi_results)

    # ── Figures 5.57/5.58 — per-model vertical heatmaps ──
    print("\nGenerating per-model vertical heatmaps (Figs 5.57, 5.58)...")
    make_heatmaps(all_results)

    # ── Combined 3-model heatmaps ──
    print("\nGenerating combined 3-model heatmaps...")
    make_combined_heatmap(all_results, 't',  '5.57',
                          'Figure 5.57 — Vertical Response: T Lower Insertion\n'
                          'Lat–Pressure RMSE Improvement (%) — All Models')
    make_combined_heatmap(all_results, 'uv', '5.58',
                          'Figure 5.58 — Vertical Response: UV Lower Insertion\n'
                          'Lat–Pressure RMSE Improvement (%) — All Models')

    # ── Figures 5.59/5.60 — surface impact ──
    print("\nGenerating surface impact plots (Figs 5.59, 5.60)...")
    make_surface_impact(all_results)

    # ── Figure 5.61 — propagation efficiency ──
    make_propagation_efficiency(all_results, BASE)

    # ── Table 5.12 ──
    table = make_table_512(all_results, BASE)

    print(f"\n=== DONE: {datetime.now()} ===")
    print("Output directories:")
    for d in [AURORA_TDIR, AURORA_UVDIR, FCN_TDIR, FCN_UVDIR, PRITHVI_TDIR, PRITHVI_UVDIR]:
        print(f"  {d}")


if __name__ == '__main__':
    main()
