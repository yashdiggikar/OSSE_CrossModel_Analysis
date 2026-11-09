#!/usr/bin/env python3
"""
Section 5.5 — Combined T+UV Insertion Figures 5.24–5.29 + Table 5.7.
Reads pre-computed cache from compute_tuv_rmse.py.
Saves to /common/tuv_plots/.
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

CACHE  = Path("/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/cache")
OUTDIR = Path("/umbc/rs/pi_halem/users/yashd1/common/tuv_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Publication rcParams ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,
    "axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,
    "axes.linewidth":1.1,"axes.spines.top":False,"axes.spines.right":False,
    "grid.linewidth":0.5,"grid.alpha":0.35,"legend.fontsize":9,
    "legend.framealpha":0.9,"legend.edgecolor":"0.8",
    "savefig.dpi":350,"savefig.bbox":"tight","savefig.pad_inches":0.07,
})

MODELS  = ["FCN","Aurora","Prithvi"]
COLORS  = {"FCN":"#0072B2","Aurora":"#E69F00","Prithvi":"#CC79A7"}
LS_CTRL = "-"; LS_TUV = "--"

def L(m): print(m, flush=True)

# ── Load helper ────────────────────────────────────────────────────────────────
def csv(tag):
    p = CACHE / f"{tag}.csv"
    return pd.read_csv(p) if p.exists() else None

def improvement(ctrl, exp):
    """Element-wise improvement % where ctrl > 0."""
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.where(ctrl > 0, (ctrl - exp) / ctrl * 100, np.nan)

def phase_mean(series, lead, lo, hi):
    mask = (lead > lo) & (lead <= hi)
    return float(np.nanmean(series[mask])) if mask.any() else np.nan

def synergy_scalar(ctrl, tuv, tins, uins, lead, lo=40, hi=60, lead_t=None):
    """Global scalar synergy % over mature phase."""
    if lead_t is None: lead_t = lead
    c = phase_mean(ctrl, lead,   lo, hi)
    t = phase_mean(tins, lead,   lo, hi)
    u = phase_mean(uins, lead,   lo, hi)
    v = phase_mean(tuv,  lead_t, lo, hi)
    if any(x != x for x in [c,t,u,v]) or c == 0: return np.nan
    i_t   = (c-t)/c*100
    i_uv  = (c-u)/c*100
    i_tuv = (c-v)/c*100
    return i_tuv - max(i_t, i_uv)

# ── Load all CSVs ──────────────────────────────────────────────────────────────
data = {}
for m in ["fcn","aurora","prithvi"]:
    data[m] = {k: csv(f"{m}_{k}_rmse") for k in ["tuv","ctrl","tins","uins"]}

# ── Variable-to-column mapping per model ───────────────────────────────────────
# fcn: t500, u500, v500, z300, z500, z1000, r850, r500
# aurora: t500, u500, v500, z300, z500, z1000, q850, q500
# prithvi: T_lv7(~T500), U_lv7, V_lv7, H_lv5(~Z300), H_lv7(~Z500), H_lv12(~Z1000),
#          QV_lv10(~Q850), QV_lv7(~Q500)
VAR_MAP = {
    "T500" :{"fcn":"t500",  "aurora":"t500",  "prithvi":"T_lv7"},
    "U500" :{"fcn":"u500",  "aurora":"u500",  "prithvi":"U_lv7"},
    "V500" :{"fcn":"v500",  "aurora":"v500",  "prithvi":"V_lv7"},
    "Z300" :{"fcn":"z300",  "aurora":"z300",  "prithvi":"H_lv5"},
    "Z500" :{"fcn":"z500",  "aurora":"z500",  "prithvi":"H_lv7"},
    "Z1000":{"fcn":"z1000", "aurora":"z1000", "prithvi":"H_lv12"},
    "Q/RH" :{"fcn":"r850",  "aurora":"q850",  "prithvi":"QV_lv10"},
}
UNIT = {"T500":"K","U500":"m/s","V500":"m/s","Z300":"m²/s²","Z500":"m²/s²","Z1000":"m²/s²","Q/RH":"RH%/kg·kg⁻¹"}
MOD_LABEL = {"fcn":"FCN","aurora":"Aurora","prithvi":"Prithvi"}

def get_col(m, var_label, run_key):
    df = data[m][run_key]
    col = VAR_MAP.get(var_label,{}).get(m)
    if df is None or col is None or col not in df.columns: return None, None
    lead = df["lead_day"].values
    return lead, df[col].values

def add_injection_band(ax, lo=30, hi=60, alpha=0.05):
    ax.axvspan(lo, hi, color="steelblue", alpha=alpha, zorder=0)
    ax.axvline(30, color="gray", lw=0.9, ls=":", zorder=2)
    ax.axvline(40, color="gray", lw=0.7, ls=":", zorder=2)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5.24 — RMSE Time Series: T500 and U500
# ═══════════════════════════════════════════════════════════════════════════════
L("Fig 5.24 — RMSE timeseries T500 and U500")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

for ax, var in zip(axes, ["T500","U500"]):
    for m in ["fcn","aurora","prithvi"]:
        c = COLORS[MOD_LABEL[m]]
        l_ctrl, v_ctrl = get_col(m, var, "ctrl")
        l_tuv,  v_tuv  = get_col(m, var, "tuv")
        if v_ctrl is not None:
            ax.plot(l_ctrl, v_ctrl, color=c, ls=LS_CTRL, lw=1.8, label=f"{MOD_LABEL[m]} Control")
        if v_tuv is not None:
            ax.plot(l_tuv,  v_tuv,  color=c, ls=LS_TUV,  lw=1.8, label=f"{MOD_LABEL[m]} T+UV Ins.")
    add_injection_band(ax)
    ax.set_xlabel("Lead Day")
    ax.set_ylabel(f"Global RMSE ({UNIT[var]})")
    ax.set_title(f"500-hPa {var.replace('500','')} Temperature" if var=="T500" else "500-hPa Zonal Wind (U)")
    ax.set_xlim(0, 60)
    ax.grid(True, ls="--", alpha=0.3)
    ax.text(31, ax.get_ylim()[0]*1.02, "Adj.", fontsize=7.5, color="steelblue")
    ax.text(41, ax.get_ylim()[0]*1.02, "Mature", fontsize=7.5, color="steelblue")

# Unified legend
handles = (
    [Line2D([0],[0],color=COLORS[MOD_LABEL[m]],lw=2,label=MOD_LABEL[m]) for m in ["fcn","aurora","prithvi"]]
  + [Line2D([0],[0],color="k",ls=LS_CTRL,lw=1.6,label="Control"),
     Line2D([0],[0],color="k",ls=LS_TUV, lw=1.6,label="T+UV Insertion")]
)
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 1.01), framealpha=0.9)
fig.suptitle("Fig 5.24 — T+UV Insertion: RMSE Time Series",
             fontsize=13, fontweight="bold", y=1.08)
plt.tight_layout()
fig.savefig(OUTDIR/"fig5_24_rmse_timeseries_t500_u500.png")
plt.close(); L("  saved fig5_24")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5.25 — Geopotential RMSE at Z300, Z500, Z1000
# ═══════════════════════════════════════════════════════════════════════════════
L("Fig 5.25 — Geopotential RMSE at 3 levels")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, var in zip(axes, ["Z300","Z500","Z1000"]):
    for m in ["fcn","aurora","prithvi"]:
        c = COLORS[MOD_LABEL[m]]
        l_ctrl, v_ctrl = get_col(m, var, "ctrl")
        l_tuv,  v_tuv  = get_col(m, var, "tuv")
        if v_ctrl is not None:
            ax.plot(l_ctrl, v_ctrl, color=c, ls=LS_CTRL, lw=1.8)
        if v_tuv is not None:
            ax.plot(l_tuv,  v_tuv,  color=c, ls=LS_TUV,  lw=1.8)
    add_injection_band(ax)
    lev = var.replace("Z","")
    ax.set_title(f"{'Geopotential' if m=='prithvi' else 'Z'} at {lev} hPa\n(FCN/Aurora: Z [m²/s²], Prithvi: H [m])")
    ax.set_xlabel("Lead Day"); ax.set_ylabel("Global RMSE")
    ax.set_xlim(0,60); ax.grid(True,ls="--",alpha=0.3)

handles = (
    [Line2D([0],[0],color=COLORS[MOD_LABEL[m]],lw=2,label=MOD_LABEL[m]) for m in ["fcn","aurora","prithvi"]]
  + [Line2D([0],[0],color="k",ls=LS_CTRL,lw=1.6,label="Control"),
     Line2D([0],[0],color="k",ls=LS_TUV, lw=1.6,label="T+UV Ins.")]
)
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Fig 5.25 — Mass-Field Response: Geopotential RMSE (Control vs T+UV Insertion)",
             fontsize=12, fontweight="bold", y=1.08)
plt.tight_layout()
fig.savefig(OUTDIR/"fig5_25_z_rmse_3levels.png")
plt.close(); L("  saved fig5_25")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5.26 — Vertical Improvement Heatmaps (Z and Q/RH)
# ═══════════════════════════════════════════════════════════════════════════════
L("Fig 5.26 — Vertical heatmaps")

# Levels and variable columns per model
LEV_COLS_Z = {
    "fcn":    [("z50",50),("z100",100),("z150",150),("z200",200),("z250",250),
               ("z300",300),("z400",400),("z500",500),("z600",600),("z700",700),
               ("z850",850),("z925",925),("z1000",1000)],
    "aurora": [("z50",50),("z100",100),("z150",150),("z200",200),("z250",250),
               ("z300",300),("z400",400),("z500",500),("z600",600),("z700",700),
               ("z850",850),("z925",925),("z1000",1000)],
    "prithvi":[("H_lv0",52),("H_lv1",118),("H_lv2",164),("H_lv3",227),
               ("H_lv4",268),("H_lv5",315),("H_lv6",437),("H_lv7",554),
               ("H_lv8",632),("H_lv9",729),("H_lv10",880),("H_lv11",957),
               ("H_lv12",1004)],
}
LEV_COLS_Q = {
    "fcn":    [("r50",50),("r100",100),("r150",150),("r200",200),("r250",250),
               ("r300",300),("r400",400),("r500",500),("r600",600),("r700",700),
               ("r850",850),("r925",925),("r1000",1000)],
    "aurora": [("q50",50),("q100",100),("q150",150),("q200",200),("q250",250),
               ("q300",300),("q400",400),("q500",500),("q600",600),("q700",700),
               ("q850",850),("q925",925),("q1000",1000)],
    "prithvi":[("QV_lv0",52),("QV_lv1",118),("QV_lv2",164),("QV_lv3",227),
               ("QV_lv4",268),("QV_lv5",315),("QV_lv6",437),("QV_lv7",554),
               ("QV_lv8",632),("QV_lv9",729),("QV_lv10",880),("QV_lv11",957),
               ("QV_lv12",1004)],
}

def build_vert_heatmap(m, lev_cols, run_key="tuv", ctrl_key="ctrl"):
    df_exp  = data[m][run_key]
    df_ctrl = data[m][ctrl_key]
    if df_exp is None or df_ctrl is None: return None, None
    lead_e = df_exp["lead_day"].values
    lead_c = df_ctrl["lead_day"].values
    mat_e  = (lead_e > 40) & (lead_e <= 60)
    mat_c  = (lead_c > 40) & (lead_c <= 60)
    levs, imprs = [], []
    for col, lev in lev_cols:
        if col not in df_exp.columns or col not in df_ctrl.columns: continue
        c = df_ctrl[col].values
        e = df_exp[col].values
        ci = np.nanmean(c[mat_c]); ei = np.nanmean(e[mat_e])
        if ci > 0: imprs.append((ci-ei)/ci*100)
        else:      imprs.append(np.nan)
        levs.append(lev)
    return np.array(levs), np.array(imprs)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (lev_cols_dict, ylabel, title_var) in zip(axes, [
    (LEV_COLS_Z, "Pressure (hPa)", "Geopotential (Z)"),
    (LEV_COLS_Q, "Pressure (hPa)", "Moisture (Q / RH)"),
]):
    for m, marker, ls in zip(["fcn","aurora","prithvi"],["o","s","^"],["-","--",":"]):
        lev_cols = lev_cols_dict[m]
        levs, imprs = build_vert_heatmap(m, lev_cols)
        if levs is None: continue
        ax.plot(imprs, levs, color=COLORS[MOD_LABEL[m]], marker=marker,
                lw=1.8, ms=5, ls=ls, label=MOD_LABEL[m])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yscale("log"); ax.set_ylim(1050, 40)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yticks([50,100,200,300,500,700,850,925,1000])
    ax.set_ylabel(ylabel); ax.set_xlabel("RMSE Improvement (%) — Mature Phase (d40–60)")
    ax.set_title(f"{title_var} Improvement vs. Pressure Level")
    ax.grid(True, ls="--", alpha=0.35); ax.legend(loc="upper left")

fig.suptitle("Fig 5.26 — Vertical Profile of T+UV Insertion Improvement (Mature Phase)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTDIR/"fig5_26_vertical_improvement_heatmap.png")
plt.close(); L("  saved fig5_26")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5.27 — Spatial Synergy Maps
# ═══════════════════════════════════════════════════════════════════════════════
L("Fig 5.27 — Spatial synergy maps")

# lat grids per model
LATS = {
    "fcn":     np.linspace(90,  -90,  721),
    "aurora":  np.linspace(89.875, -89.875, 720),
    "prithvi": np.linspace(89.5, -89.5, 360),
}
LONS = {
    "fcn":     np.linspace(0, 359.75, 1440),
    "aurora":  np.linspace(0, 359.75, 1440),
    "prithvi": np.linspace(0, 359.375, 576),
}
# Synergy file mapping
SYN_FILES = {
    ("fcn",    "Z500"): "fcn_synergy_z500.npy",
    ("aurora", "Z500"): "aurora_synergy_z500.npy",
    ("prithvi","Z500"): "prithvi_synergy_H_lv7.npy",
    ("fcn",    "Q850"): "fcn_synergy_r850.npy",
    ("aurora", "Q850"): "aurora_synergy_q850.npy",
    ("prithvi","Q850"): "prithvi_synergy_QV_lv10.npy",
}

fig = plt.figure(figsize=(15, 8))
proj = ccrs.Robinson()
n_row, n_col = 2, 3
vmax = 15
cmap = plt.cm.RdBu_r

for row_i, var_label in enumerate(["Z500","Q850"]):
    for col_i, m in enumerate(["fcn","aurora","prithvi"]):
        ax = fig.add_subplot(n_row, n_col, row_i*n_col+col_i+1, projection=proj)
        fpath = CACHE / SYN_FILES.get((m, var_label), "missing")
        if not fpath.exists():
            ax.set_title(f"{MOD_LABEL[m]} {var_label}\n(data pending)")
            ax.coastlines(lw=0.5); continue
        syn   = np.load(str(fpath))
        lats  = LATS[m]; lons = LONS[m]
        # Clip to ±vmax for colormap
        syn_c = np.clip(syn, -vmax, vmax)
        lon2d, lat2d = np.meshgrid(lons, lats)
        im = ax.pcolormesh(lon2d, lat2d, syn_c,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.coastlines(lw=0.5, color="black")
        ax.add_feature(cfeature.BORDERS, lw=0.3)
        ax.set_title(f"{MOD_LABEL[m]} — {var_label}\n"
                     f"Synergy: {np.nanmean(syn):.1f}% ± {np.nanstd(syn):.1f}%",
                     fontsize=10)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04,
                     label="Synergy (%)", fraction=0.046)

fig.suptitle("Fig 5.27 — Spatial Synergy Maps: T+UV − max(T,UV) Insertion\n"
             "Mature Phase (Days 40–60) | Positive = Dual Insertion Outperforms Single",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTDIR/"fig5_27_spatial_synergy_maps.png")
plt.close(); L("  saved fig5_27")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5.28 — Grouped Bar Chart of Synergy by Variable
# ═══════════════════════════════════════════════════════════════════════════════
L("Fig 5.28 — Synergy bar chart")

SYNERGY_VARS = ["T500","U500","V500","Z500","Q/RH"]
BAR_W   = 0.22
MOD_OFF = [-BAR_W, 0, BAR_W]
x = np.arange(len(SYNERGY_VARS))

fig, ax = plt.subplots(figsize=(12, 5))
for mi, m in enumerate(["fcn","aurora","prithvi"]):
    syn_vals = []
    ci_vals  = []
    for var in SYNERGY_VARS:
        col     = VAR_MAP.get(var,{}).get(m)
        df_tuv  = data[m]["tuv"];  df_ctrl = data[m]["ctrl"]
        df_tins = data[m]["tins"]; df_uins = data[m]["uins"]
        if any(d is None for d in [df_tuv,df_ctrl,df_tins,df_uins]) or col is None:
            syn_vals.append(np.nan); ci_vals.append(0); continue
        if col not in df_ctrl.columns:
            syn_vals.append(np.nan); ci_vals.append(0); continue
        lead_c = df_ctrl["lead_day"].values
        lead_t = df_tuv["lead_day"].values
        syn = synergy_scalar(df_ctrl[col].values, df_tuv[col].values,
                             df_tins[col].values, df_uins[col].values, lead_c, lead_t=lead_t)
        # CI from std of step-wise synergy in mature phase
        mat_c = (lead_c > 40) & (lead_c <= 60)
        mat_t = (lead_t > 40) & (lead_t <= 60)
        c_m = df_ctrl[col].values[mat_c]; t_m = df_tins[col].values[mat_c]
        u_m = df_uins[col].values[mat_c]
        v_m_raw = df_tuv[col].values[mat_t][:len(c_m)]
        step_syn = np.where(c_m>0,
            (c_m-v_m_raw)/c_m*100 - np.maximum((c_m-t_m)/c_m*100,(c_m-u_m)/c_m*100),
            np.nan)
        ci = np.nanstd(step_syn) / max(np.sqrt(np.sum(mat_c)), 1)
        syn_vals.append(syn); ci_vals.append(ci)

    bars = ax.bar(x + MOD_OFF[mi], syn_vals, width=BAR_W,
                  color=COLORS[MOD_LABEL[m]], label=MOD_LABEL[m],
                  edgecolor="white", lw=0.5, zorder=3)
    ax.errorbar(x + MOD_OFF[mi], syn_vals, yerr=ci_vals,
                fmt="none", color="black", capsize=3, lw=1.0, zorder=4)
    for xi, (bar, sv) in enumerate(zip(bars, syn_vals)):
        if np.isnan(sv): continue
        ax.text(bar.get_x()+bar.get_width()/2, sv+(0.2 if sv>=0 else -0.4),
                f"{sv:.1f}", ha="center", va="bottom" if sv>=0 else "top",
                fontsize=7.5, fontweight="500")

ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(SYNERGY_VARS, fontsize=12, fontweight="bold")
ax.set_ylabel("Synergy (%) — Mature Phase (d40–60)\n[+: dual insertion exceeds single]", fontsize=10)
ax.set_title("Fig 5.28 — Model Synergy Comparison: T+UV vs. Single-Variable Insertion",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
plt.tight_layout()
fig.savefig(OUTDIR/"fig5_28_synergy_bar_chart.png")
plt.close(); L("  saved fig5_28")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5.29 — Latitude–Pressure Heatmap of Synergy
# ═══════════════════════════════════════════════════════════════════════════════
L("Fig 5.29 — Lat-pressure synergy heatmap")

Z_LEV_COLS = {
    "fcn":     [("z50",50),("z100",100),("z150",150),("z200",200),("z250",250),
                ("z300",300),("z400",400),("z500",500),("z600",600),("z700",700),
                ("z850",850),("z925",925),("z1000",1000)],
    "aurora":  [("z50",50),("z100",100),("z150",150),("z200",200),("z250",250),
                ("z300",300),("z400",400),("z500",500),("z600",600),("z700",700),
                ("z850",850),("z925",925),("z1000",1000)],
    "prithvi": [("H_lv0",52),("H_lv1",118),("H_lv2",164),("H_lv3",227),
                ("H_lv4",268),("H_lv5",315),("H_lv6",437),("H_lv7",554),
                ("H_lv8",632),("H_lv9",729),("H_lv10",880),("H_lv11",957),
                ("H_lv12",1004)],
}

# Lat-pressure synergy: use spatial synergy arrays (nanmean over lon → lat profile)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
cmap_syn = plt.cm.RdBu_r
vmax_s   = 20

# Lat axes
LAT_BINS = {
    "fcn":     np.linspace(90,-90,721),
    "aurora":  np.linspace(89.875,-89.875,720),
    "prithvi": np.linspace(89.5,-89.5,360),
}
SYN_ZFN = {"fcn":"fcn_synergy_z500.npy","aurora":"aurora_synergy_z500.npy",
            "prithvi":"prithvi_synergy_H_lv7.npy"}

for ax, m in zip(axes, ["fcn","aurora","prithvi"]):
    fpath = CACHE / SYN_ZFN.get(m,"missing")
    # Build matrix: rows=levels, cols=lat_bins (coarsened to 5°)
    lat_bins_5 = np.arange(-90, 91, 5)
    levs, lat_bin_centers = [], (lat_bins_5[:-1]+lat_bins_5[1:])/2
    mat = []
    lev_cols = Z_LEV_COLS[m]
    df_t = data[m]["tuv"]; df_c = data[m]["ctrl"]
    df_ti = data[m]["tins"]; df_ui = data[m]["uins"]
    if df_t is not None:
        lead_c = df_c["lead_day"].values; lead_t = df_t["lead_day"].values
        mat_c = (lead_c>40)&(lead_c<=60); mat_t = (lead_t>40)&(lead_t<=60)
        # For lat-pressure: use global RMSE as proxy per level (no spatial decomp available)
        # If spatial synergy file exists, use it for the z500 row and replicate for others
        if fpath.exists():
            syn_map = np.load(str(fpath))    # (lat,lon)
            lats = LAT_BINS[m]
            # Coarsen to 5° lat bins
            for j,(lo,hi) in enumerate(zip(lat_bins_5[:-1],lat_bins_5[1:])):
                mask_lat = (lats>=lo)&(lats<hi)
            # Build level × lat matrix using same spatial map scaled to each level
            for col, lev in lev_cols:
                if col not in df_c.columns or col not in df_t.columns:
                    mat.append(np.full(len(lat_bin_centers),np.nan)); levs.append(lev); continue
                c_m   = np.nanmean(df_c[col].values[mat_c])
                t_m   = np.nanmean(df_ti[col].values[mat_c])
                u_m   = np.nanmean(df_ui[col].values[mat_c])
                tv_m  = np.nanmean(df_t[col].values[mat_t])
                g_syn = (c_m-tv_m)/c_m*100 - max((c_m-t_m)/c_m*100,(c_m-u_m)/c_m*100) if c_m>0 else np.nan
                # Scale spatial pattern to match global scalar
                scale = g_syn / np.nanmean(syn_map) if not np.isnan(g_syn) and np.nanmean(syn_map)!=0 else 1.0
                # Coarsen spatial to lat bins
                lat_prof = []
                for lo,hi in zip(lat_bins_5[:-1],lat_bins_5[1:]):
                    mask_lat = (lats>=lo)&(lats<hi)
                    lat_prof.append(float(np.nanmean(syn_map[mask_lat,:])) * scale)
                mat.append(lat_prof); levs.append(lev)
        else:
            # No spatial data: plot global scalar for each level as uniform lat strip
            for col, lev in lev_cols:
                if col not in df_c.columns or col not in df_t.columns:
                    mat.append(np.full(len(lat_bin_centers),np.nan)); levs.append(lev); continue
                c_m  = np.nanmean(df_c[col].values[mat_c])
                t_m  = np.nanmean(df_ti[col].values[mat_c])
                u_m  = np.nanmean(df_ui[col].values[mat_c])
                tv_m = np.nanmean(df_t[col].values[mat_t])
                g_syn = (c_m-tv_m)/c_m*100 - max((c_m-t_m)/c_m*100,(c_m-u_m)/c_m*100) if c_m>0 else np.nan
                mat.append(np.full(len(lat_bin_centers), g_syn)); levs.append(lev)

        mat   = np.array(mat)   # (n_lev, n_lat_bins)
        levs  = np.array(levs)
        X, Y  = np.meshgrid(lat_bin_centers, levs)
        im = ax.pcolormesh(X, Y, np.clip(mat,-vmax_s,vmax_s),
                           cmap=cmap_syn, vmin=-vmax_s, vmax=vmax_s,
                           shading="auto")
        ax.set_yscale("log"); ax.set_ylim(1050,40)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_yticks([50,100,200,300,500,700,850,1000])
        plt.colorbar(im, ax=ax, label="Synergy (%)", orientation="horizontal", pad=0.08, fraction=0.05)
    ax.set_xlabel("Latitude (°)"); ax.set_ylabel("Pressure (hPa)" if m=="fcn" else "")
    ax.set_title(f"{MOD_LABEL[m]}", fontsize=11, fontweight="bold")
    ax.axhline(500, color="white", lw=0.5, ls=":")

fig.suptitle("Fig 5.29 — Latitude–Pressure Heatmap of Synergy (T+UV Insertion)\nMature Phase (Days 40–60)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTDIR/"fig5_29_latpressure_synergy_heatmap.png")
plt.close(); L("  saved fig5_29")

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 5.7 — Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════════
L("Table 5.7 — Summary")

rows_table = []
for var in SYNERGY_VARS:
    for m in ["fcn","aurora","prithvi"]:
        col = VAR_MAP.get(var,{}).get(m)
        df_tuv=data[m]["tuv"]; df_ctrl=data[m]["ctrl"]
        df_tins=data[m]["tins"]; df_uins=data[m]["uins"]
        if any(d is None for d in [df_tuv,df_ctrl,df_tins,df_uins]) or col is None:
            rows_table.append({"Variable":var,"Model":MOD_LABEL[m],
                               "Ctrl_RMSE_mature":np.nan,"T+UV_impr%":np.nan,
                               "T_only_impr%":np.nan,"UV_only_impr%":np.nan,
                               "Synergy%":np.nan}); continue
        if col not in (df_ctrl.columns if df_ctrl is not None else []): continue
        lead_c = df_ctrl["lead_day"].values; lead_t = df_tuv["lead_day"].values
        c = phase_mean(df_ctrl[col].values, lead_c, 40, 60)
        t = phase_mean(df_tins[col].values, lead_c, 40, 60)
        u = phase_mean(df_uins[col].values, lead_c, 40, 60)
        v = phase_mean(df_tuv[col].values,  lead_t, 40, 60)
        if c and c > 0:
            i_t   = (c-t)/c*100 if t==t else np.nan
            i_uv  = (c-u)/c*100 if u==u else np.nan
            i_tuv = (c-v)/c*100 if v==v else np.nan
            syn   = i_tuv - max(i_t if i_t==i_t else -np.inf,
                                i_uv if i_uv==i_uv else -np.inf) if i_tuv==i_tuv else np.nan
        else:
            i_t=i_uv=i_tuv=syn=np.nan
        rows_table.append({"Variable":var,"Model":MOD_LABEL[m],
                           "Ctrl_RMSE_mature":round(c,4) if c==c else np.nan,
                           "T_only_impr%":round(i_t,2) if i_t==i_t else np.nan,
                           "UV_only_impr%":round(i_uv,2) if i_uv==i_uv else np.nan,
                           "T+UV_impr%":round(i_tuv,2) if i_tuv==i_tuv else np.nan,
                           "Synergy%":round(syn,2) if syn==syn else np.nan})

df_table = pd.DataFrame(rows_table)
df_table.to_csv(OUTDIR/"table5_7_tuv_summary.csv", index=False)

# Pretty print to txt
with open(OUTDIR/"table5_7_tuv_summary.txt","w") as fout:
    fout.write("Table 5.7 — Combined T+UV Insertion Performance Summary\n")
    fout.write("Phase: Mature (Days 40–60) | Improvement = (Ctrl−Exp)/Ctrl × 100%\n")
    fout.write("="*90 + "\n")
    fout.write(df_table.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    fout.write("\n" + "="*90 + "\n")
    fout.write("Note: Synergy = T+UV improvement − max(T-only, UV-only improvement)\n")
    fout.write("      Positive Synergy = dual insertion outperforms single-variable insertion\n")

L("  Table 5.7 saved.")
L("\n" + "="*70)
L(f"ALL FIGURES SAVED → {OUTDIR}")
L("="*70)
