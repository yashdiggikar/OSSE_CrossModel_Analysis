#!/usr/bin/env python3
"""
Section 5.12.2 – Temperature Insertion by Terrain Class  (terrain)
Grouped bar chart + RMSE time series for all 4 models.

Cross-variable targets of T insertion:
  U500, V500, Z500, Q700

Terrain classes (5):
  Land Low   frland>0.5, elev<300m, non-polar
  Land Mod   frland>0.5, 300<=elev<1500m, non-polar
  Land High  frland>0.5, elev>=1500m, non-polar
  Ocean      frocean>0.5, non-polar
  Polar      |lat|>60 deg

Models: Aurora, FourCastNet, Prithvi-WxC, GraphCast
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
OUT_BASE  = BASE / "common/terrain"
ERA5_DIR  = BASE / "AURORA_DATA/rmse_data"
ERA5_2022 = BASE / "era5_vs_cams/era5_2022"
MERRA_DIR = BASE / "prithvi/data/merra2"
MERRA_SFC = BASE / "prithvi/data/merra2/MERRA2_sfc_20220103.nc"
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
N_TERR = 5
TERR_NAMES  = ["Land Low\n(<300 m)", "Land Mod\n(300-1500 m)", "Land High\n(>1500 m)",
               "Ocean", "Polar\n(|phi|>60)"]
TERR_SHORT  = ["Land Low", "Land Mod", "Land High", "Ocean", "Polar"]
TERR_COLORS = ["#9CAF6E", "#6B8E4E", "#5C4A3D", "#4A7A96", "#A9BFC9"]

AURORA_IC  = datetime(2023, 2, 1, 12, 0); FCN_IC     = datetime(2023, 2, 1,  0, 0)
PRITHVI_IC = datetime(2022, 1, 1, 12, 0); GC_IC      = datetime(2022, 1, 1,  6, 0)

N_AUR_FCN = 120; N_PRI = 240; N_GC = 240

AUR_FCN_INJ = 60;  AUR_FCN_ADJ = slice(60, 80);   AUR_FCN_MAT = slice(80, 120)
PRI_INJ     = 120; PRI_ADJ     = slice(120, 160);  PRI_MAT     = slice(160, 240)
GC_INJ      = 120; GC_ADJ      = slice(120, 160);  GC_MAT      = slice(160, 240)

FCN_CH = {"u500": 15, "v500": 28, "z500": 41, "r700": 69}
GC_500_IDX = 7; GC_700_IDX = 9
PRI_500_IDX = 7; PRI_700_IDX = 8
MERRA_500_IDX = 6; MERRA_700_IDX = 5

CROSS_VARS = ["U500", "V500", "Z500", "Q700"]
VAR_LABELS = {
    "U500": "U-wind 500 hPa (m/s)", "V500": "V-wind 500 hPa (m/s)",
    "Z500": "Geopotential 500 hPa (m2/s2)",
    "Q700": "Moisture 700 hPa [Sp. Humidity (Aurora/Prithvi/GC) | RH% (FCN)]",
}
VAR_SHORT = {"U500":"U500","V500":"V500","Z500":"Z500","Q700":"Q/R 700"}
MODEL_LABELS = {"aurora":"Aurora","fcn":"FourCastNet","prithvi":"Prithvi-WxC","graphcast":"GraphCast"}
MODEL_ORDER  = ["aurora","fcn","prithvi","graphcast"]

apply_style()

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def cos_weights_2d(nlat, lat0, lat1, nlon):
    return np.cos(np.deg2rad(np.linspace(lat0,lat1,nlat))).clip(0)[:,None]*np.ones((1,nlon))

def nearest_time_idx(ref_times, target_dt):
    return int(np.argmin([abs((t-target_dt).total_seconds()) for t in ref_times]))

def classify_terrain(elev, frland, frocean, lats_2d):
    polar = np.abs(lats_2d) > 60.0
    mask  = np.full(elev.shape, 3, dtype=np.int8)
    mask[(~polar)&(frland>0.5)&(elev<300)]                  = 0
    mask[(~polar)&(frland>0.5)&(elev>=300)&(elev<1500)]     = 1
    mask[(~polar)&(frland>0.5)&(elev>=1500)]                = 2
    mask[polar]                                              = 4
    return mask

def rmse_by_class(fc, truth, mask, cos_w2d):
    err2 = (fc-truth)**2; out = np.full(N_TERR, np.nan)
    for c in range(N_TERR):
        m = mask==c; w = cos_w2d[m]
        if w.sum()<1e-8: continue
        out[c] = np.sqrt(np.sum(w*err2[m])/w.sum())
    return out

def compute_rmse_series(fc_arr, truth_arr, terrain, cos2d, truth_idx=None):
    N = fc_arr.shape[0]; out = np.full((N,N_TERR),np.nan)
    for s in range(N):
        tr = truth_arr[truth_idx[s]] if truth_idx is not None else truth_arr[s]
        out[s] = rmse_by_class(fc_arr[s], tr, terrain, cos2d)
    return out

def improvement_pct(cb, ins_mean, eps=1e-12):
    return np.where(cb>eps, 100.0*(cb-ins_mean)/cb, np.nan)

def era5_to_rh(q, t, p_hpa):
    e_s = 6.112*np.exp(17.67*(t-273.15)/(t-29.65))
    q_s = 0.622*e_s/(p_hpa-e_s)
    return np.clip(q/(q_s+1e-12),0,1.5)*100.0

def regrid_merra_to_era5(merra_360, nlat_era5, lat0, lat1):
    ml=np.linspace(-89.5,90.0,360); el=np.linspace(lat0,lat1,nlat_era5)
    en=np.linspace(0,359.75,1440)
    li=np.array([np.argmin(np.abs(ml-la)) for la in el])
    ni=np.round(en/(360.0/576.0)).astype(int)%576
    lg,lg2=np.meshgrid(li,ni,indexing="ij")
    return merra_360[lg,lg2]

def regrid_merra_to_gc(merra_360):
    ml=np.linspace(-89.5,90.0,360); gl=np.linspace(-90.0,90.0,181)
    gn=np.linspace(0.0,359.0,360)
    li=np.array([np.argmin(np.abs(ml-la)) for la in gl])
    ni=np.round(gn/(360.0/576.0)).astype(int)%576
    lg,lg2=np.meshgrid(li,ni,indexing="ij")
    return merra_360[lg,lg2]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 – TERRAIN MASKS
# ──────────────────────────────────────────────────────────────────────────────
log("=== MERRA2 terrain ===")
sfc_ds  = xr.open_dataset(str(MERRA_SFC), engine="h5netcdf")
elev_360= sfc_ds["PHIS"].values[1:,:]/9.80665
frland  = sfc_ds["FRLAND"].values[1:,:]
frocean = sfc_ds["FROCEAN"].values[1:,:]
merra_lats_1d = sfc_ds["lat"].values[1:]; merra_lons_1d = sfc_ds["lon"].values
sfc_ds.close()

merra_lat2d,_ = np.meshgrid(merra_lats_1d,merra_lons_1d,indexing="ij")
terr_merra = classify_terrain(elev_360,frland,frocean,merra_lat2d)

elev_aur=regrid_merra_to_era5(elev_360,720,90.0,-89.75); frl_aur=regrid_merra_to_era5(frland,720,90.0,-89.75); fro_aur=regrid_merra_to_era5(frocean,720,90.0,-89.75)
aur_lats=np.linspace(90.0,-89.75,720); aur_lat2d=np.broadcast_to(aur_lats[:,None],(720,1440))
terr_aur=classify_terrain(elev_aur,frl_aur,fro_aur,aur_lat2d)

elev_fcn=regrid_merra_to_era5(elev_360,721,90.0,-90.0); frl_fcn=regrid_merra_to_era5(frland,721,90.0,-90.0); fro_fcn=regrid_merra_to_era5(frocean,721,90.0,-90.0)
fcn_lats=np.linspace(90.0,-90.0,721); fcn_lat2d=np.broadcast_to(fcn_lats[:,None],(721,1440))
terr_fcn=classify_terrain(elev_fcn,frl_fcn,fro_fcn,fcn_lat2d)

elev_gc=regrid_merra_to_gc(elev_360); frl_gc=regrid_merra_to_gc(frland); fro_gc=regrid_merra_to_gc(frocean)
gc_lats_1d=np.linspace(-90.0,90.0,181); gc_lat2d=np.broadcast_to(gc_lats_1d[:,None],(181,360))
terr_gc=classify_terrain(elev_gc,frl_gc,fro_gc,gc_lat2d)

for c,nm in enumerate(TERR_SHORT):
    log(f"  GC {nm}: {(terr_gc==c).sum():6d}  ({100*(terr_gc==c).mean():.1f}%)")

aur_cos2d=cos_weights_2d(720,90.0,-89.75,1440); fcn_cos2d=cos_weights_2d(721,90.0,-90.0,1440)
pri_cos2d=cos_weights_2d(360,-89.5,90.0,576);   gc_cos2d =cos_weights_2d(181,-90.0,90.0,360)

aur_vtimes=[AURORA_IC +timedelta(hours=12*s) for s in range(N_AUR_FCN)]
fcn_vtimes=[FCN_IC    +timedelta(hours=12*s) for s in range(N_AUR_FCN)]
pri_vtimes=[PRITHVI_IC+timedelta(hours= 6*s) for s in range(N_PRI)]
gc_vtimes =[GC_IC     +timedelta(hours= 6*(s+1)) for s in range(N_GC)]
aur_lead=np.arange(N_AUR_FCN)*0.5; fcn_lead=np.arange(N_AUR_FCN)*0.5
pri_lead=np.arange(N_PRI)*0.25;    gc_lead =np.arange(N_GC)*0.25

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 – ERA5 2023 TRUTH
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== ERA5 2023 truth (Aurora/FCN) ===")
ERA5_2023_FILES=[ERA5_DIR/f"era5_pressure_2023_{m:02d}.grib" for m in [2,3,4]]
def load_era5_23(sn,lev):
    dsets=[xr.open_dataset(str(f),engine="cfgrib",filter_by_keys={"shortName":sn,"typeOfLevel":"isobaricInhPa","level":int(lev)}) for f in ERA5_2023_FILES]
    combined=xr.concat(dsets,dim="time").sortby("time")
    times=[pd.Timestamp(str(t)).to_pydatetime() for t in combined.time.values]
    data=combined[sn].values.astype(np.float32)
    for d in dsets: d.close()
    return times,data

log("  U500 …"); era5_u_times,era5_u500=load_era5_23("u",500)
log("  V500 …"); _,           era5_v500=load_era5_23("v",500)
log("  Z500 …"); _,           era5_z500=load_era5_23("z",500)
log("  Q700 …"); era5_q_times,era5_q700=load_era5_23("q",700)
log("  T700 …"); _,           era5_t700=load_era5_23("t",700)
era5_r700=era5_to_rh(era5_q700,era5_t700,700.0).astype(np.float32)
del era5_t700; gc.collect()

aur_era5_idx=[nearest_time_idx(era5_u_times,vt) for vt in aur_vtimes]
fcn_era5_idx=[nearest_time_idx(era5_u_times,vt) for vt in fcn_vtimes]
era5_u500_720=era5_u500[:,:-1,:]; era5_v500_720=era5_v500[:,:-1,:]
era5_z500_720=era5_z500[:,:-1,:]; era5_q700_720=era5_q700[:,:-1,:]
era5_r700_720=era5_r700[:,:-1,:]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 – ERA5 2022 TRUTH → 1° (GC)
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== ERA5 2022 truth -> 1 deg (GC) ===")
ERA5_2022_FILES=[ERA5_2022/f"era5_pressure_2022_{m:02d}.grib" for m in [1,2,3]]
gc_lons_1d=np.linspace(0.0,359.0,360)
ERA5_LATS_ASC=np.linspace(-90.0,90.0,721); ERA5_LONS=np.linspace(0.0,359.75,1440)
_gc_mesh=np.array(np.meshgrid(gc_lats_1d,gc_lons_1d,indexing="ij")).reshape(2,-1).T

def bilinear_to_gc(arr2d):
    arr_asc=arr2d[::-1,:]
    interp=RegularGridInterpolator((ERA5_LATS_ASC,ERA5_LONS),arr_asc,method="linear",bounds_error=False,fill_value=None)
    return interp(_gc_mesh).reshape(181,360).astype(np.float32)

def load_era5_22_gc(sn,lev):
    dsets=[xr.open_dataset(str(f),engine="cfgrib",filter_by_keys={"shortName":sn,"typeOfLevel":"isobaricInhPa","level":int(lev)}) for f in ERA5_2022_FILES]
    combined=xr.concat(dsets,dim="time").sortby("time"); times=[pd.Timestamp(str(t)).to_pydatetime() for t in combined.time.values]
    arr=combined[sn].values.astype(np.float32)
    for d in dsets: d.close()
    out=np.empty((arr.shape[0],181,360),dtype=np.float32)
    for t in range(arr.shape[0]): out[t]=bilinear_to_gc(arr[t])
    return times,out

log("  U500 …"); gc_u_times,gc_era5_u500=load_era5_22_gc("u",500)
log("  V500 …"); _,         gc_era5_v500=load_era5_22_gc("v",500)
log("  Z500 …"); _,         gc_era5_z500=load_era5_22_gc("z",500)
log("  Q700 …"); _,         gc_era5_q700=load_era5_22_gc("q",700)
gc_era5_idx=[nearest_time_idx(gc_u_times,vt) for vt in gc_vtimes]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 – AURORA
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== AURORA ===")
def load_aur(fpath,key,n):
    with h5py.File(str(fpath),"r") as f: return f[key][:n].astype(np.float32)

aur_vmap={"U500":("u500",era5_u500_720,aur_era5_idx),"V500":("v500",era5_v500_720,aur_era5_idx),
          "Z500":("z500",era5_z500_720,aur_era5_idx),"Q700":("q700",era5_q700_720,aur_era5_idx)}
aur_rmse={}
for var,(key,truth,tidx) in aur_vmap.items():
    log(f"  {var} …")
    ctrl=load_aur(AUR_CTRL,key,N_AUR_FCN); ins=load_aur(AUR_T_INS,key,N_AUR_FCN)
    aur_rmse[var]={"ctrl":compute_rmse_series(ctrl,truth,terr_aur,aur_cos2d,tidx),
                   "ins": compute_rmse_series(ins, truth,terr_aur,aur_cos2d,tidx)}
    del ctrl,ins; gc.collect()
del era5_u500_720,era5_v500_720,era5_z500_720,era5_q700_720,era5_r700_720; gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 – FCN
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== FCN ===")
def load_fcn(fpath,ch,n):
    with h5py.File(str(fpath),"r") as f: return f["prediction"][:n,ch,:,:].astype(np.float32)

fcn_vmap={"U500":(FCN_CH["u500"],era5_u500,fcn_era5_idx),"V500":(FCN_CH["v500"],era5_v500,fcn_era5_idx),
          "Z500":(FCN_CH["z500"],era5_z500,fcn_era5_idx),"Q700":(FCN_CH["r700"],era5_r700,fcn_era5_idx)}
fcn_rmse={}
for var,(ch,truth,tidx) in fcn_vmap.items():
    log(f"  {var} (ch={ch}) …")
    ctrl=load_fcn(FCN_CTRL,ch,N_AUR_FCN); ins=load_fcn(FCN_T_INS,ch,N_AUR_FCN)
    fcn_rmse[var]={"ctrl":compute_rmse_series(ctrl,truth,terr_fcn,fcn_cos2d,tidx),
                   "ins": compute_rmse_series(ins, truth,terr_fcn,fcn_cos2d,tidx)}
    del ctrl,ins; gc.collect()
del era5_u500,era5_v500,era5_z500,era5_q700,era5_r700; gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 – PRITHVI
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== PRITHVI ===")
def build_merra2_truth(var_name,lev_idx,n_steps,vtimes):
    out=np.full((n_steps,360,576),np.nan,dtype=np.float32)
    prev_date=None; pres_ds=None
    for s,vt in enumerate(vtimes):
        ds_str=vt.strftime("%Y%m%d"); ti=vt.hour//3
        if ds_str!=prev_date:
            if pres_ds is not None: pres_ds.close()
            pp=MERRA_DIR/f"MERRA_pres_{ds_str}.nc"
            if not pp.exists(): log(f"  WARN: missing {ds_str}"); prev_date=ds_str; continue
            pres_ds=xr.open_dataset(str(pp),engine="h5netcdf"); prev_date=ds_str
        out[s]=pres_ds[var_name].values[ti,lev_idx,1:,:]
        if s%40==0: log(f"    step {s}/{n_steps}  {vt}")
    if pres_ds is not None: pres_ds.close()
    return out

def load_pri(fpath,var_name,lev_idx):
    ds=xr.open_dataset(str(fpath),engine="h5netcdf"); arr=ds[var_name].values[:,lev_idx,:,:]
    ds.close()
    if arr.shape[1]==361: arr=arr[:,1:,:]  # crop south-pole row in 361-lat control
    return arr.astype(np.float32)

pri_vmap={"U500":("U",MERRA_500_IDX,PRI_500_IDX),"V500":("V",MERRA_500_IDX,PRI_500_IDX),
          "Z500":("H",MERRA_500_IDX,PRI_500_IDX),"Q700":("QV",MERRA_700_IDX,PRI_700_IDX)}
pri_rmse={}
for var,(mname,m_li,p_li) in pri_vmap.items():
    log(f"  {var} …")
    truth=build_merra2_truth(mname,m_li,N_PRI,pri_vtimes)
    ctrl=load_pri(PRI_CTRL,mname,p_li); ins=load_pri(PRI_T_INS,mname,p_li)
    pri_rmse[var]={"ctrl":compute_rmse_series(ctrl,truth,terr_merra,pri_cos2d),
                   "ins": compute_rmse_series(ins, truth,terr_merra,pri_cos2d)}
    del truth,ctrl,ins; gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 – GRAPHCAST
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== GRAPHCAST ===")
def load_gc_var(fpath,varname,lev_idx):
    ds=xr.open_dataset(str(fpath),engine="h5netcdf",decode_timedelta=False)
    arr=ds[varname].values[:N_GC,0,lev_idx,:,:]; ds.close()
    return arr.astype(np.float32)

gc_vmap={"U500":("u_component_of_wind",GC_500_IDX,gc_era5_u500,gc_era5_idx),
         "V500":("v_component_of_wind",GC_500_IDX,gc_era5_v500,gc_era5_idx),
         "Z500":("geopotential",       GC_500_IDX,gc_era5_z500,gc_era5_idx),
         "Q700":("specific_humidity",  GC_700_IDX,gc_era5_q700,gc_era5_idx)}
gc_rmse={}
for var,(gcvar,li,truth,tidx) in gc_vmap.items():
    log(f"  {var} ('{gcvar}' lev={li}) …")
    ctrl=load_gc_var(GC_CTRL,gcvar,li); ins=load_gc_var(GC_T_INS,gcvar,li)
    gc_rmse[var]={"ctrl":compute_rmse_series(ctrl,truth,terr_gc,gc_cos2d,tidx),
                  "ins": compute_rmse_series(ins, truth,terr_gc,gc_cos2d,tidx)}
    del ctrl,ins; gc.collect()
del gc_era5_u500,gc_era5_v500,gc_era5_z500,gc_era5_q700; gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 – CSVs
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== CSVs ===")
def save_csv(rmse_dict,lead_arr,out_dir,tag):
    rows=[]
    for var,d in rmse_dict.items():
        for s in range(len(lead_arr)):
            for c in range(N_TERR):
                rows.append({"step":s,"lead_day":round(float(lead_arr[s]),3),"variable":var,
                             "terrain":TERR_SHORT[c],"ctrl_rmse":float(d["ctrl"][s,c]),
                             "ins_rmse":float(d["ins"][s,c])})
    pd.DataFrame(rows).to_csv(out_dir/"intermediates"/f"{tag}_t_ins_terrain_rmse.csv",index=False)
    log(f"  saved -> {tag}")

save_csv(aur_rmse,aur_lead,OUT["aurora"],   "aurora")
save_csv(fcn_rmse,fcn_lead,OUT["fcn"],      "fcn")
save_csv(pri_rmse,pri_lead,OUT["prithvi"],  "prithvi")
save_csv(gc_rmse, gc_lead, OUT["graphcast"],"graphcast")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 – IMPROVEMENTS
# ──────────────────────────────────────────────────────────────────────────────
ALL_RMSE={
    "aurora":   (aur_rmse,AUR_FCN_INJ,AUR_FCN_ADJ,AUR_FCN_MAT,aur_lead),
    "fcn":      (fcn_rmse,AUR_FCN_INJ,AUR_FCN_ADJ,AUR_FCN_MAT,fcn_lead),
    "prithvi":  (pri_rmse,PRI_INJ,    PRI_ADJ,    PRI_MAT,    pri_lead),
    "graphcast":(gc_rmse, GC_INJ,     GC_ADJ,     GC_MAT,     gc_lead),
}
improvements={}
for model,(rdict,inj,adj,mat,lead) in ALL_RMSE.items():
    improvements[model]={}
    for var in CROSS_VARS:
        if var not in rdict: continue
        cb=rdict[var]["ctrl"][inj]
        improvements[model][var]={
            "adj":np.where(cb>1e-12,100*(cb-np.nanmean(rdict[var]["ins"][adj],axis=0))/cb,np.nan),
            "mat":np.where(cb>1e-12,100*(cb-np.nanmean(rdict[var]["ins"][mat],axis=0))/cb,np.nan),
        }

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 – BAR CHART (4 models x 4 vars)
# ──────────────────────────────────────────────────────────────────────────────
log("\n--- Bar chart ---")
fig_bar = build_bar_figure(
    "T Insertion: RMSE Improvement (%) by Terrain Class\n"
    "Baseline = Day-30 Control RMSE  |  Positive = improvement  |  4 Models",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_SHORT, improvements,
    TERR_NAMES, N_TERR, "Terrain Class", figsize=(26, 18))
for model,out_dir in OUT.items():
    fig_bar.savefig(out_dir/"fig_terrain_t_ins_rmse_improvement_bar.png",bbox_inches="tight",dpi=300)
    log(f"  saved -> {out_dir.name}")
plt.close(fig_bar)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 – TIME SERIES (4 vars x 4 models)
# ──────────────────────────────────────────────────────────────────────────────
log("\n--- Time series ---")
fig_ts = build_timeseries_figure(
    "T Insertion: RMSE Time Series by Terrain Class\n"
    "Dashed = Control  |  Solid = T Insertion  |  Blue = Adj. phase  |  Pink = Mature phase",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_LABELS, ALL_RMSE,
    TERR_COLORS, TERR_SHORT, figsize=(28, 20), hspace=0.52, wspace=0.28)
for model,out_dir in OUT.items():
    fig_ts.savefig(out_dir/"fig_terrain_t_ins_rmse_timeseries.png",bbox_inches="tight",dpi=300)
    log(f"  saved -> {out_dir.name}")
plt.close(fig_ts)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12 – SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────
log("\n=== Summary table ===")
rows=[]
for model,(rdict,inj,adj,mat,lead) in ALL_RMSE.items():
    for var in CROSS_VARS:
        if var not in rdict: continue
        ai=improvements[model][var]["adj"]; mi=improvements[model][var]["mat"]
        for c in range(N_TERR):
            rows.append({"Model":MODEL_LABELS[model],"Variable":var,"Terrain":TERR_SHORT[c],
                         "Ctrl_RMSE_Day30":round(float(rdict[var]["ctrl"][inj,c]),6),
                         "Adj_Imp_%":round(float(ai[c]),3),"Mat_Imp_%":round(float(mi[c]),3)})
df=pd.DataFrame(rows)
df.to_csv(OUT_BASE/"table_t_ins_terrain_summary.csv",index=False)
for model,out_dir in OUT.items():
    df[df["Model"]==MODEL_LABELS[model]].to_csv(out_dir/"t_ins_terrain_summary.csv",index=False)

log("\n=== terrain complete ===")
