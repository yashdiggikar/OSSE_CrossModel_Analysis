#!/usr/bin/env python3
"""
cloud2 – UV Insertion by Cloud Regime – 4 Models
Cross-vars: T500, Z500, Q850 (sp.hum Aurora/Prithvi/GC | RH% FCN)
"""
import os, gc, warnings
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from pathlib import Path
import h5py, numpy as np, pandas as pd, xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import matplotlib.lines as mlines, matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _regime_terrain_plot_common import apply_style, build_bar_figure, build_timeseries_figure

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
OUT_BASE  = BASE / "common/cloud2"
ERA5_DIR  = BASE / "AURORA_DATA/rmse_data"
ERA5_2022 = BASE / "era5_vs_cams/era5_2022"
MERRA_DIR = BASE / "prithvi/data/merra2"
TCC_DIR   = BASE / "ALL_ERA5/tcc"
GC_DIR    = BASE / "gc_pred_files"

OUT = {m: OUT_BASE/m for m in ["aurora","fcn","prithvi","graphcast"]}
for d in OUT.values(): (d/"intermediates").mkdir(parents=True, exist_ok=True)

AUR_CTRL  = BASE/"auroraoutput_1year_1stfeb23.nc"
AUR_UV    = BASE/"Aurora_Pred_Files/auroraoutput_uv_inserted.nc"
FCN_CTRL  = BASE/"FourCastNet/forecast_normal_60days_12hr.nc"
FCN_UV    = BASE/"FourCastNet/Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc"
PRI_CTRL  = BASE/"prithviOSSE/run2/output/control_forecast_Jan2022.nc"
PRI_UV    = BASE/"prithviOSSE/uv_ins_run/output/uv_osse_forecast_Jan2022.nc"
GC_CTRL   = GC_DIR/"gc_sm_240steps_STRICT.nc"
GC_UV     = GC_DIR/"gc_sm_240steps_uv_inj.nc"

REGIME_BOUNDS=[0.0,0.10,0.40,0.70,1.01]
REGIME_NAMES=["Clear\n(<10%)","Cirrus\n(10-40%)","Mixed\n(40-70%)","Opaque\n(>70%)"]
REGIME_SHORT=["Clear","Cirrus","Mixed","Opaque"]
REGIME_COLORS=["#D4A24E","#6F9FB8","#8C7AA9","#3B4252"]
N_REG=4

AURORA_IC=datetime(2023,2,1,12,0); FCN_IC=datetime(2023,2,1,0,0)
PRITHVI_IC=datetime(2022,1,1,12,0); GC_IC=datetime(2022,1,1,6,0)
N_AUR_FCN=120; N_PRI=240; N_GC=240
AUR_FCN_INJ=60; AUR_FCN_ADJ=slice(60,80);  AUR_FCN_MAT=slice(80,120)
PRI_INJ=120;   PRI_ADJ=slice(120,160);     PRI_MAT=slice(160,240)
GC_INJ=120;    GC_ADJ=slice(120,160);      GC_MAT=slice(160,240)

FCN_CH={"t500":54,"z500":41,"r850":70}
GC_500_IDX=7; GC_850_IDX=10
PRI_500_IDX=7; PRI_850_IDX=10
MERRA_500_IDX=6; MERRA_850_IDX=3

CROSS_VARS=["T500","Z500","Q850"]
VAR_LABELS={"T500":"Temperature 500 hPa (K)","Z500":"Geopotential 500 hPa (m2/s2)",
            "Q850":"Moisture 850 hPa [Sp.Hum Aurora/Prithvi/GC | RH% FCN]"}
MODEL_LABELS={"aurora":"Aurora","fcn":"FourCastNet","prithvi":"Prithvi-WxC","graphcast":"GraphCast"}
MODEL_ORDER=["aurora","fcn","prithvi","graphcast"]
apply_style()

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}",flush=True)
def cos2d(nlat,la0,la1,nlon): return np.cos(np.deg2rad(np.linspace(la0,la1,nlat))).clip(0)[:,None]*np.ones((1,nlon))
def nti(rts,dt): return int(np.argmin([abs((t-dt).total_seconds()) for t in rts]))
def cls_reg(tcc):
    r=np.zeros(tcc.shape,dtype=np.int8)
    for i,(lo,hi) in enumerate(zip(REGIME_BOUNDS[:-1],REGIME_BOUNDS[1:])): r[(tcc>=lo)&(tcc<hi)]=i
    return r
def rmse_reg(fc,tr,rm,cw):
    e2=(fc-tr)**2; out=np.full(N_REG,np.nan)
    for r in range(N_REG):
        m=rm==r; w=cw[m]
        if w.sum()<1e-8: continue
        out[r]=np.sqrt(np.sum(w*e2[m])/w.sum())
    return out
def rmse_series(fc_arr,tr_arr,reg_arr,cw,tidx=None):
    N=fc_arr.shape[0]; out=np.full((N,N_REG),np.nan)
    for s in range(N):
        tr=tr_arr[tidx[s]] if tidx is not None else tr_arr[s]
        out[s]=rmse_reg(fc_arr[s],tr,reg_arr[s],cw)
    return out
def imp(cb,im,eps=1e-12): return np.where(cb>eps,100*(cb-im)/cb,np.nan)
def era5_rh(q,t,p): e_s=6.112*np.exp(17.67*(t-273.15)/(t-29.65)); qs=0.622*e_s/(p-e_s); return np.clip(q/(qs+1e-12),0,1.5)*100

# ── TCC masks ──
log("=== TCC masks ===")
ds=xr.open_dataset(TCC_DIR/"era5_total_cloud_cover_2023_Feb_Apr_6hourly.nc",engine="h5netcdf")
tcc23=ds["tcc"].values; t23=[pd.Timestamp(str(t)).to_pydatetime() for t in ds["valid_time"].values]; ds.close()
ds=xr.open_dataset(TCC_DIR/"era5_total_cloud_cover_2022_jan_feb_6hourly.nc",engine="h5netcdf")
tcc22=ds["tcc"].values; t22=[pd.Timestamp(str(t)).to_pydatetime() for t in ds["valid_time"].values]; ds.close()

av=[AURORA_IC +timedelta(hours=12*s) for s in range(N_AUR_FCN)]
fv=[FCN_IC    +timedelta(hours=12*s) for s in range(N_AUR_FCN)]
pv=[PRITHVI_IC+timedelta(hours= 6*s) for s in range(N_PRI)]
gv=[GC_IC     +timedelta(hours= 6*(s+1)) for s in range(N_GC)]

def mask_af(td,tt,vts,nl):
    m=np.zeros((len(vts),nl,1440),dtype=np.int8)
    for s,vt in enumerate(vts): m[s]=cls_reg(td[nti(tt,vt),:nl,:])
    return m
log("  Aurora …"); arm=mask_af(tcc23,t23,av,720)
log("  FCN …");    fcm=mask_af(tcc23,t23,fv,721)
log("  Prithvi …")
ml=np.linspace(-89.5,90.0,360); mn=np.arange(576)*(360.0/576.0)
ei=np.clip(np.round((90.0-ml)/0.25).astype(int),0,719); ej=np.round(mn/0.25).astype(int)%1440
lg,lg2=np.meshgrid(ei,ej,indexing="ij")
prm=np.zeros((N_PRI,360,576),dtype=np.int8)
for s,vt in enumerate(pv): prm[s]=cls_reg(tcc22[nti(t22,vt)][lg,lg2])
log("  GraphCast …")
gl1=np.linspace(-90.0,90.0,181); gn1=np.linspace(0.0,359.0,360)
tli=np.clip(np.round((90.0-gl1)/0.25).astype(int),0,719); tni=np.round(gn1/0.25).astype(int)%1440
lgg,lg2g=np.meshgrid(tli,tni,indexing="ij")
gcm=np.zeros((N_GC,181,360),dtype=np.int8)
for s,vt in enumerate(gv): gcm[s]=cls_reg(tcc22[nti(t22,vt)][lgg,lg2g])
del tcc23,tcc22; gc.collect()

acw=cos2d(720,90.0,-89.75,1440); fcw=cos2d(721,90.0,-90.0,1440)
pcw=cos2d(360,-89.5,90.0,576);   gcw=cos2d(181,-90.0,90.0,360)
al=np.arange(N_AUR_FCN)*0.5; fl=np.arange(N_AUR_FCN)*0.5; pl=np.arange(N_PRI)*0.25; gcl=np.arange(N_GC)*0.25

# ── ERA5 2023 truth (Aurora/FCN) ──
log("\n=== ERA5 2023 ===")
E23=[ERA5_DIR/f"era5_pressure_2023_{m:02d}.grib" for m in [2,3,4]]
def le23(sn,lev):
    ds=[xr.open_dataset(str(f),engine="cfgrib",filter_by_keys={"shortName":sn,"typeOfLevel":"isobaricInhPa","level":int(lev)}) for f in E23]
    c=xr.concat(ds,dim="time").sortby("time"); ts=[pd.Timestamp(str(t)).to_pydatetime() for t in c.time.values]; d=c[sn].values.astype(np.float32)
    for x in ds: x.close()
    return ts,d
log("  T500 …"); et,et5=le23("t",500); log("  Z500 …"); _,ez5=le23("z",500)
log("  Q850 …"); _,eq8=le23("q",850); log("  T850 …"); _,et8=le23("t",850)
er8=era5_rh(eq8,et8,850.0).astype(np.float32); del et8; gc.collect()
ai=[nti(et,vt) for vt in av]; fi=[nti(et,vt) for vt in fv]
et5_720=et5[:,:-1,:]; ez5_720=ez5[:,:-1,:]; eq8_720=eq8[:,:-1,:]; er8_720=er8[:,:-1,:]

# ── ERA5 2022 truth → 1° (GC) ──
log("\n=== ERA5 2022 -> 1 deg (GC) ===")
E22=[ERA5_2022/f"era5_pressure_2022_{m:02d}.grib" for m in [1,2,3]]
ela=np.linspace(-90.0,90.0,721); elo=np.linspace(0.0,359.75,1440)
gm=np.array(np.meshgrid(gl1,gn1,indexing="ij")).reshape(2,-1).T
def btgc(a2d):
    aa=a2d[::-1,:]; i=RegularGridInterpolator((ela,elo),aa,method="linear",bounds_error=False,fill_value=None)
    return i(gm).reshape(181,360).astype(np.float32)
def le22gc(sn,lev):
    ds=[xr.open_dataset(str(f),engine="cfgrib",filter_by_keys={"shortName":sn,"typeOfLevel":"isobaricInhPa","level":int(lev)}) for f in E22]
    c=xr.concat(ds,dim="time").sortby("time"); ts=[pd.Timestamp(str(t)).to_pydatetime() for t in c.time.values]
    a=c[sn].values.astype(np.float32)
    for x in ds: x.close()
    out=np.empty((a.shape[0],181,360),dtype=np.float32)
    for t in range(a.shape[0]): out[t]=btgc(a[t])
    return ts,out
log("  T500 …"); gtt,get5=le22gc("t",500); log("  Z500 …"); _,gez5=le22gc("z",500); log("  Q850 …"); _,geq8=le22gc("q",850)
gi=[nti(gtt,vt) for vt in gv]

# ── Aurora ──
log("\n=== Aurora ===")
def laur(fp,k,n):
    with h5py.File(str(fp),"r") as f: return f[k][:n].astype(np.float32)
arm_d={}
for var,k,tr,tidx in [("T500","t500",et5_720,ai),("Z500","z500",ez5_720,ai),("Q850","q850",eq8_720,ai)]:
    log(f"  {var} …"); c=laur(AUR_CTRL,k,N_AUR_FCN); i=laur(AUR_UV,k,N_AUR_FCN)
    arm_d[var]={"ctrl":rmse_series(c,tr,arm,acw,tidx),"ins":rmse_series(i,tr,arm,acw,tidx)}; del c,i; gc.collect()
del et5_720,ez5_720,eq8_720,er8_720; gc.collect()

# ── FCN ──
log("\n=== FCN ===")
def lfcn(fp,ch,n):
    with h5py.File(str(fp),"r") as f: return f["prediction"][:n,ch,:,:].astype(np.float32)
fcm_d={}
for var,ch,tr,tidx in [("T500",FCN_CH["t500"],et5,fi),("Z500",FCN_CH["z500"],ez5,fi),("Q850",FCN_CH["r850"],er8,fi)]:
    log(f"  {var} ch={ch}…"); c=lfcn(FCN_CTRL,ch,N_AUR_FCN); i=lfcn(FCN_UV,ch,N_AUR_FCN)
    fcm_d[var]={"ctrl":rmse_series(c,tr,fcm,fcw,tidx),"ins":rmse_series(i,tr,fcm,fcw,tidx)}; del c,i; gc.collect()
del et5,ez5,eq8,er8; gc.collect()

# ── Prithvi ──
log("\n=== Prithvi ===")
def m2truth(vn,li,ns,vts):
    out=np.full((ns,360,576),np.nan,dtype=np.float32); pd2=None; pds=None
    for s,vt in enumerate(vts):
        dstr=vt.strftime("%Y%m%d"); ti=vt.hour//3
        if dstr!=pd2:
            if pds is not None: pds.close()
            pp=MERRA_DIR/f"MERRA_pres_{dstr}.nc"
            if not pp.exists(): log(f"  WARN {dstr}"); pd2=dstr; continue
            pds=xr.open_dataset(str(pp),engine="h5netcdf"); pd2=dstr
        out[s]=pds[vn].values[ti,li,1:,:]
        if s%40==0: log(f"    step {s}/{ns}  {vt}")
    if pds is not None: pds.close()
    return out
def lpri(fp,vn,li): ds=xr.open_dataset(str(fp),engine="h5netcdf"); a=ds[vn].values[:,li,:,:]; ds.close(); return a.astype(np.float32)
prm_d={}
for var,mn,ml_i,pl_i in [("T500","T",MERRA_500_IDX,PRI_500_IDX),("Z500","H",MERRA_500_IDX,PRI_500_IDX),("Q850","QV",MERRA_850_IDX,PRI_850_IDX)]:
    log(f"  {var} …"); tr=m2truth(mn,ml_i,N_PRI,pv); c=lpri(PRI_CTRL,mn,pl_i); i=lpri(PRI_UV,mn,pl_i)
    prm_d[var]={"ctrl":rmse_series(c,tr,prm,pcw),"ins":rmse_series(i,tr,prm,pcw)}; del tr,c,i; gc.collect()

# ── GraphCast ──
log("\n=== GraphCast ===")
def lgc(fp,vn,li): ds=xr.open_dataset(str(fp),engine="h5netcdf",decode_timedelta=False); a=ds[vn].values[:N_GC,0,li,:,:]; ds.close(); return a.astype(np.float32)
gcm_d={}
for var,gcv,li,tr,tidx in [("T500","temperature",GC_500_IDX,get5,gi),("Z500","geopotential",GC_500_IDX,gez5,gi),("Q850","specific_humidity",GC_850_IDX,geq8,gi)]:
    log(f"  {var} '{gcv}' lev={li} …"); c=lgc(GC_CTRL,gcv,li); i=lgc(GC_UV,gcv,li)
    gcm_d[var]={"ctrl":rmse_series(c,tr,gcm,gcw,tidx),"ins":rmse_series(i,tr,gcm,gcw,tidx)}; del c,i; gc.collect()
del get5,gez5,geq8; gc.collect()

# ── CSVs ──
log("\n=== CSVs ===")
def sv(d,la,od,tag):
    rows=[]
    for var,dd in d.items():
        for s in range(len(la)):
            for r in range(N_REG):
                rows.append({"step":s,"lead_day":round(float(la[s]),3),"variable":var,"regime":REGIME_SHORT[r],"ctrl_rmse":float(dd["ctrl"][s,r]),"ins_rmse":float(dd["ins"][s,r])})
    pd.DataFrame(rows).to_csv(od/"intermediates"/f"{tag}_uv_ins_regime_rmse.csv",index=False)
    log(f"  saved -> {tag}")
sv(arm_d,al,OUT["aurora"],"aurora"); sv(fcm_d,fl,OUT["fcn"],"fcn"); sv(prm_d,pl,OUT["prithvi"],"prithvi"); sv(gcm_d,gcl,OUT["graphcast"],"graphcast")

ALL_RMSE={"aurora":(arm_d,AUR_FCN_INJ,AUR_FCN_ADJ,AUR_FCN_MAT,al),"fcn":(fcm_d,AUR_FCN_INJ,AUR_FCN_ADJ,AUR_FCN_MAT,fl),
           "prithvi":(prm_d,PRI_INJ,PRI_ADJ,PRI_MAT,pl),"graphcast":(gcm_d,GC_INJ,GC_ADJ,GC_MAT,gcl)}
imps={}
for model,(rd,inj,adj,mat,ld) in ALL_RMSE.items():
    imps[model]={}
    for var in CROSS_VARS:
        if var not in rd: continue
        cb=rd[var]["ctrl"][inj]
        imps[model][var]={"adj":imp(cb,np.nanmean(rd[var]["ins"][adj],axis=0)),"mat":imp(cb,np.nanmean(rd[var]["ins"][mat],axis=0))}

# ── Bar chart ──
log("\n--- Bar chart ---")
fig_b = build_bar_figure(
    "UV Insertion: RMSE Improvement (%) by Cloud Regime\n4 Models  |  Baseline = Day-30 Control",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, {v: v for v in CROSS_VARS}, imps,
    REGIME_NAMES, N_REG, "Cloud Regime", figsize=(20, 18))
for model,od in OUT.items(): fig_b.savefig(od/"fig_cloud2_uv_ins_rmse_improvement_bar.png",bbox_inches="tight",dpi=300); log(f"  saved -> {od.name}")
plt.close(fig_b)

# ── Time series ──
log("\n--- Time series ---")
fig_t = build_timeseries_figure(
    "UV Insertion: RMSE Time Series by Cloud Regime  (4 Models)\nDashed=Control  Solid=UV Ins  Blue=Adj  Pink=Mature",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_LABELS, ALL_RMSE,
    REGIME_COLORS, REGIME_SHORT, figsize=(26, 18), hspace=0.5, wspace=0.28)
for model,od in OUT.items(): fig_t.savefig(od/"fig_cloud2_uv_ins_rmse_timeseries.png",bbox_inches="tight",dpi=300); log(f"  saved -> {od.name}")
plt.close(fig_t)

# ── Summary ──
log("\n=== Summary ===")
rows=[]
for model,(rd,inj,adj,mat,ld) in ALL_RMSE.items():
    for var in CROSS_VARS:
        if var not in rd: continue
        ai=imps[model][var]["adj"]; mi=imps[model][var]["mat"]
        for r in range(N_REG):
            rows.append({"Model":MODEL_LABELS[model],"Variable":var,"Regime":REGIME_SHORT[r],
                         "Ctrl_RMSE_Day30":round(float(rd[var]["ctrl"][inj,r]),6),
                         "Adj_Imp_%":round(float(ai[r]),3),"Mat_Imp_%":round(float(mi[r]),3)})
df=pd.DataFrame(rows); df.to_csv(OUT_BASE/"table_uv_ins_cloud_regime_summary.csv",index=False)
for model,od in OUT.items(): df[df["Model"]==MODEL_LABELS[model]].to_csv(od/"uv_ins_cloud_regime_summary.csv",index=False)
log("\n=== cloud2 complete ===")
