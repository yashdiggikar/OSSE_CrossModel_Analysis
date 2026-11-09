#!/usr/bin/env python3
"""
terrain2 – UV Insertion by Terrain Class – 4 Models
Cross-vars: T500, Z500, Q850 (sp.hum Aurora/Prithvi/GC | RH% FCN)
Terrain: Land Low / Land Mod / Land High / Ocean / Polar(|lat|>60)
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
OUT_BASE  = BASE/"common/terrain2"
ERA5_DIR  = BASE/"AURORA_DATA/rmse_data"
ERA5_2022 = BASE/"era5_vs_cams/era5_2022"
MERRA_DIR = BASE/"prithvi/data/merra2"
MERRA_SFC = BASE/"prithvi/data/merra2/MERRA2_sfc_20220103.nc"
GC_DIR    = BASE/"gc_pred_files"

OUT={m:OUT_BASE/m for m in ["aurora","fcn","prithvi","graphcast"]}
for d in OUT.values(): (d/"intermediates").mkdir(parents=True,exist_ok=True)

AUR_CTRL=BASE/"auroraoutput_1year_1stfeb23.nc"
AUR_UV  =BASE/"Aurora_Pred_Files/auroraoutput_uv_inserted.nc"
FCN_CTRL=BASE/"FourCastNet/forecast_normal_60days_12hr.nc"
FCN_UV  =BASE/"FourCastNet/Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc"
PRI_CTRL=BASE/"prithviOSSE/run2/output/control_forecast_Jan2022.nc"
PRI_UV  =BASE/"prithviOSSE/uv_ins_run/output/uv_osse_forecast_Jan2022.nc"
GC_CTRL =GC_DIR/"gc_sm_240steps_STRICT.nc"
GC_UV   =GC_DIR/"gc_sm_240steps_uv_inj.nc"

N_TERR=5
TERR_NAMES=["Land Low\n(<300m)","Land Mod\n(300-1500m)","Land High\n(>1500m)","Ocean","Polar\n(|lat|>60)"]
TERR_SHORT=["Land Low","Land Mod","Land High","Ocean","Polar"]
TERR_COLORS=["#9CAF6E","#6B8E4E","#5C4A3D","#4A7A96","#A9BFC9"]

AURORA_IC=datetime(2023,2,1,12,0); FCN_IC=datetime(2023,2,1,0,0)
PRITHVI_IC=datetime(2022,1,1,12,0); GC_IC=datetime(2022,1,1,6,0)
N_AUR_FCN=120; N_PRI=240; N_GC=240
AUR_FCN_INJ=60; AUR_FCN_ADJ=slice(60,80); AUR_FCN_MAT=slice(80,120)
PRI_INJ=120; PRI_ADJ=slice(120,160); PRI_MAT=slice(160,240)
GC_INJ=120;  GC_ADJ=slice(120,160); GC_MAT=slice(160,240)

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
def cterr(elev,frl,fro,lat2d):
    pol=np.abs(lat2d)>60.0; m=np.full(elev.shape,3,dtype=np.int8)
    m[(~pol)&(frl>0.5)&(elev<300)]=0; m[(~pol)&(frl>0.5)&(elev>=300)&(elev<1500)]=1; m[(~pol)&(frl>0.5)&(elev>=1500)]=2; m[pol]=4
    return m
def rbc(fc,tr,mask,cw):
    e2=(fc-tr)**2; out=np.full(N_TERR,np.nan)
    for c in range(N_TERR):
        m=mask==c; w=cw[m]
        if w.sum()<1e-8: continue
        out[c]=np.sqrt(np.sum(w*e2[m])/w.sum())
    return out
def rseries(fa,ta,tm,cw,tidx=None):
    N=fa.shape[0]; out=np.full((N,N_TERR),np.nan)
    for s in range(N):
        tr=ta[tidx[s]] if tidx is not None else ta[s]; out[s]=rbc(fa[s],tr,tm,cw)
    return out
def imp(cb,im,eps=1e-12): return np.where(cb>eps,100*(cb-im)/cb,np.nan)
def era5_rh(q,t,p): e_s=6.112*np.exp(17.67*(t-273.15)/(t-29.65)); qs=0.622*e_s/(p-e_s); return np.clip(q/(qs+1e-12),0,1.5)*100

def rme(mf,nlat,la0,la1):
    ml=np.linspace(-89.5,90.0,360); el=np.linspace(la0,la1,nlat); en=np.linspace(0,359.75,1440)
    li=np.array([np.argmin(np.abs(ml-la)) for la in el]); ni=np.round(en/(360.0/576.0)).astype(int)%576
    lg,lg2=np.meshgrid(li,ni,indexing="ij"); return mf[lg,lg2]
def rmgc(mf):
    ml=np.linspace(-89.5,90.0,360); gl=np.linspace(-90.0,90.0,181); gn=np.linspace(0.0,359.0,360)
    li=np.array([np.argmin(np.abs(ml-la)) for la in gl]); ni=np.round(gn/(360.0/576.0)).astype(int)%576
    lg,lg2=np.meshgrid(li,ni,indexing="ij"); return mf[lg,lg2]

# ── Terrain masks ──
log("=== Terrain masks ===")
sds=xr.open_dataset(str(MERRA_SFC),engine="h5netcdf")
e360=sds["PHIS"].values[1:,:]/9.80665; frl=sds["FRLAND"].values[1:,:]; fro=sds["FROCEAN"].values[1:,:]
mla=sds["lat"].values[1:]; sds.close()
ml2d,_=np.meshgrid(mla,np.linspace(0,359.375,576),indexing="ij")
tm_m=cterr(e360,frl,fro,ml2d)
ea=rme(e360,720,90.0,-89.75); fra=rme(frl,720,90.0,-89.75); foa=rme(fro,720,90.0,-89.75)
ala=np.linspace(90.0,-89.75,720); ald=np.broadcast_to(ala[:,None],(720,1440)); tm_a=cterr(ea,fra,foa,ald)
ef=rme(e360,721,90.0,-90.0); frf=rme(frl,721,90.0,-90.0); fof=rme(fro,721,90.0,-90.0)
fla=np.linspace(90.0,-90.0,721); fld=np.broadcast_to(fla[:,None],(721,1440)); tm_f=cterr(ef,frf,fof,fld)
eg=rmgc(e360); frg=rmgc(frl); fog=rmgc(fro)
gl1=np.linspace(-90.0,90.0,181); gn1=np.linspace(0.0,359.0,360); gld=np.broadcast_to(gl1[:,None],(181,360)); tm_g=cterr(eg,frg,fog,gld)
log("GC terrain:"); [log(f"  {TERR_SHORT[c]:>10s}: {(tm_g==c).sum():6d} ({100*(tm_g==c).mean():.1f}%)") for c in range(N_TERR)]

acw=cos2d(720,90.0,-89.75,1440); fcw=cos2d(721,90.0,-90.0,1440); pcw=cos2d(360,-89.5,90.0,576); gcw=cos2d(181,-90.0,90.0,360)
av=[AURORA_IC+timedelta(hours=12*s) for s in range(N_AUR_FCN)]; fv=[FCN_IC+timedelta(hours=12*s) for s in range(N_AUR_FCN)]
pv=[PRITHVI_IC+timedelta(hours=6*s) for s in range(N_PRI)]; gv=[GC_IC+timedelta(hours=6*(s+1)) for s in range(N_GC)]
al=np.arange(N_AUR_FCN)*0.5; fl=al.copy(); pl=np.arange(N_PRI)*0.25; gcl=np.arange(N_GC)*0.25

# ── ERA5 2023 truth ──
log("\n=== ERA5 2023 ===")
E23=[ERA5_DIR/f"era5_pressure_2023_{m:02d}.grib" for m in [2,3,4]]
def le23(sn,lev):
    ds=[xr.open_dataset(str(f),engine="cfgrib",filter_by_keys={"shortName":sn,"typeOfLevel":"isobaricInhPa","level":int(lev)}) for f in E23]
    c=xr.concat(ds,dim="time").sortby("time"); ts=[pd.Timestamp(str(t)).to_pydatetime() for t in c.time.values]; d=c[sn].values.astype(np.float32)
    for x in ds: x.close(); return ts,d
log("  T500 …"); et,et5=le23("t",500); log("  Z500 …"); _,ez5=le23("z",500); log("  Q850 …"); _,eq8=le23("q",850); log("  T850 …"); _,et8=le23("t",850)
er8=era5_rh(eq8,et8,850.0).astype(np.float32); del et8; gc.collect()
ai=[nti(et,vt) for vt in av]; fi=[nti(et,vt) for vt in fv]
et5_720=et5[:,:-1,:]; ez5_720=ez5[:,:-1,:]; eq8_720=eq8[:,:-1,:]; er8_720=er8[:,:-1,:]

# ── ERA5 2022 → 1° (GC) ──
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
ard={}
for var,k,tr,ti in [("T500","t500",et5_720,ai),("Z500","z500",ez5_720,ai),("Q850","q850",eq8_720,ai)]:
    log(f"  {var} …"); c=laur(AUR_CTRL,k,N_AUR_FCN); i=laur(AUR_UV,k,N_AUR_FCN)
    ard[var]={"ctrl":rseries(c,tr,tm_a,acw,ti),"ins":rseries(i,tr,tm_a,acw,ti)}; del c,i; gc.collect()
del et5_720,ez5_720,eq8_720,er8_720; gc.collect()

# ── FCN ──
log("\n=== FCN ===")
def lfcn(fp,ch,n):
    with h5py.File(str(fp),"r") as f: return f["prediction"][:n,ch,:,:].astype(np.float32)
fcd={}
for var,ch,tr,ti in [("T500",FCN_CH["t500"],et5,fi),("Z500",FCN_CH["z500"],ez5,fi),("Q850",FCN_CH["r850"],er8,fi)]:
    log(f"  {var} ch={ch}…"); c=lfcn(FCN_CTRL,ch,N_AUR_FCN); i=lfcn(FCN_UV,ch,N_AUR_FCN)
    fcd[var]={"ctrl":rseries(c,tr,tm_f,fcw,ti),"ins":rseries(i,tr,tm_f,fcw,ti)}; del c,i; gc.collect()
del et5,ez5,eq8,er8; gc.collect()

# ── Prithvi ──
log("\n=== Prithvi ===")
def m2t(vn,li,ns,vts):
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
prd={}
for var,mn,ml_i,pl_i in [("T500","T",MERRA_500_IDX,PRI_500_IDX),("Z500","H",MERRA_500_IDX,PRI_500_IDX),("Q850","QV",MERRA_850_IDX,PRI_850_IDX)]:
    log(f"  {var} …"); tr=m2t(mn,ml_i,N_PRI,pv); c=lpri(PRI_CTRL,mn,pl_i); i=lpri(PRI_UV,mn,pl_i)
    prd[var]={"ctrl":rseries(c,tr,tm_m,pcw),"ins":rseries(i,tr,tm_m,pcw)}; del tr,c,i; gc.collect()

# ── GraphCast ──
log("\n=== GraphCast ===")
def lgc(fp,vn,li): ds=xr.open_dataset(str(fp),engine="h5netcdf",decode_timedelta=False); a=ds[vn].values[:N_GC,0,li,:,:]; ds.close(); return a.astype(np.float32)
gcd={}
for var,gcv,li,tr,ti in [("T500","temperature",GC_500_IDX,get5,gi),("Z500","geopotential",GC_500_IDX,gez5,gi),("Q850","specific_humidity",GC_850_IDX,geq8,gi)]:
    log(f"  {var} '{gcv}' …"); c=lgc(GC_CTRL,gcv,li); i=lgc(GC_UV,gcv,li)
    gcd[var]={"ctrl":rseries(c,tr,tm_g,gcw,ti),"ins":rseries(i,tr,tm_g,gcw,ti)}; del c,i; gc.collect()
del get5,gez5,geq8; gc.collect()

# ── CSVs ──
log("\n=== CSVs ===")
def sv(d,la,od,tag):
    rows=[]
    for var,dd in d.items():
        for s in range(len(la)):
            for c in range(N_TERR):
                rows.append({"step":s,"lead_day":round(float(la[s]),3),"variable":var,"terrain":TERR_SHORT[c],"ctrl_rmse":float(dd["ctrl"][s,c]),"ins_rmse":float(dd["ins"][s,c])})
    pd.DataFrame(rows).to_csv(od/"intermediates"/f"{tag}_uv_ins_terrain_rmse.csv",index=False); log(f"  saved -> {tag}")
sv(ard,al,OUT["aurora"],"aurora"); sv(fcd,fl,OUT["fcn"],"fcn"); sv(prd,pl,OUT["prithvi"],"prithvi"); sv(gcd,gcl,OUT["graphcast"],"graphcast")

ALL_RMSE={"aurora":(ard,AUR_FCN_INJ,AUR_FCN_ADJ,AUR_FCN_MAT,al),"fcn":(fcd,AUR_FCN_INJ,AUR_FCN_ADJ,AUR_FCN_MAT,fl),
           "prithvi":(prd,PRI_INJ,PRI_ADJ,PRI_MAT,pl),"graphcast":(gcd,GC_INJ,GC_ADJ,GC_MAT,gcl)}
imps={}
for model,(rd,inj,adj,mat,ld) in ALL_RMSE.items():
    imps[model]={}
    for var in CROSS_VARS:
        if var not in rd: continue
        cb=rd[var]["ctrl"][inj]
        imps[model][var]={"adj":imp(cb,np.nanmean(rd[var]["ins"][adj],axis=0)),"mat":imp(cb,np.nanmean(rd[var]["ins"][mat],axis=0))}

# ── Bar chart ──
log("\n--- Bar chart ---")
fb = build_bar_figure(
    "UV Insertion: RMSE Improvement (%) by Terrain Class\n4 Models  |  Baseline = Day-30 Control",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, {v: v for v in CROSS_VARS}, imps,
    TERR_NAMES, N_TERR, "Terrain Class", figsize=(22, 18))
for model,od in OUT.items(): fb.savefig(od/"fig_terrain2_uv_ins_rmse_improvement_bar.png",bbox_inches="tight",dpi=300); log(f"  saved -> {od.name}")
plt.close(fb)

# ── Time series ──
log("\n--- Time series ---")
ft = build_timeseries_figure(
    "UV Insertion: RMSE Time Series by Terrain Class  (4 Models)\nDashed=Control  Solid=UV Ins  Blue=Adj  Pink=Mature",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_LABELS, ALL_RMSE,
    TERR_COLORS, TERR_SHORT, figsize=(26, 18), hspace=0.5, wspace=0.28)
for model,od in OUT.items(): ft.savefig(od/"fig_terrain2_uv_ins_rmse_timeseries.png",bbox_inches="tight",dpi=300); log(f"  saved -> {od.name}")
plt.close(ft)

# ── Summary ──
log("\n=== Summary ===")
rows=[]
for model,(rd,inj,adj,mat,ld) in ALL_RMSE.items():
    for var in CROSS_VARS:
        if var not in rd: continue
        ai=imps[model][var]["adj"]; mi=imps[model][var]["mat"]
        for c in range(N_TERR):
            rows.append({"Model":MODEL_LABELS[model],"Variable":var,"Terrain":TERR_SHORT[c],
                         "Ctrl_RMSE_Day30":round(float(rd[var]["ctrl"][inj,c]),6),
                         "Adj_Imp_%":round(float(ai[c]),3),"Mat_Imp_%":round(float(mi[c]),3)})
df=pd.DataFrame(rows); df.to_csv(OUT_BASE/"table_uv_ins_terrain_summary.csv",index=False)
for model,od in OUT.items(): df[df["Model"]==MODEL_LABELS[model]].to_csv(od/"uv_ins_terrain_summary.csv",index=False)
log("\n=== terrain2 complete ===")
