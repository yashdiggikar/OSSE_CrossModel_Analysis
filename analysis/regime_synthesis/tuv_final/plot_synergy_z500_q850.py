#!/usr/bin/env python3
"""
Fig 5.27 — Spatial synergy maps for Z500 and Q850 (T+UV minus max(T,UV) insertion).
Fig 5.28 — Grouped bar chart (with CI) comparing synergy for Z500 and Q850 across
           the 3 models.

Synergy = (T+UV insertion improvement) - max(T-only improvement, UV-only improvement),
mature phase (lead day 40-60). Positive = dual insertion beats the best single
insertion alone.

Reads the (corrected) cache from compute_tuv_rmse.py / recompute_prithvi_tuv_rmse.py.
Outputs -> common/tuv_final/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

CACHE  = Path("/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/cache")
OUTDIR = Path("/umbc/rs/pi_halem/users/yashd1/common/tuv_final")
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":"serif",
    "font.serif":["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":11,"axes.titlesize":12,
    "axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,
    "axes.linewidth":1.1,"axes.spines.top":False,"axes.spines.right":False,
    "grid.linewidth":0.5,"grid.alpha":0.35,"legend.fontsize":9,
    "legend.framealpha":0.9,"legend.edgecolor":"0.8",
    "savefig.dpi":350,"savefig.bbox":"tight","savefig.pad_inches":0.07,
})

MOD_LABEL = {"fcn":"FCN","aurora":"Aurora","prithvi":"Prithvi","graphcast":"GraphCast"}
COLORS    = {"FCN":"#0072B2","Aurora":"#E69F00","Prithvi":"#CC79A7","GraphCast":"#009E73"}

VAR_MAP = {
    "Z500": {"fcn":"z500", "aurora":"z500", "prithvi":"H_lv7",  "graphcast":"z500"},
    "Q850": {"fcn":"r850", "aurora":"q850", "prithvi":"QV_lv10","graphcast":"q850"},
}

# GraphCast (GC-DETER-2023, Feb 2023 IC, full-res 0.25deg) is a later addition
# to this project than the fcn/aurora/prithvi tri-study this figure was
# originally built for -- see compute_graphcast_tuv_cache.py for how its
# cache/synergy files were derived from that study's own compute_rmse.py/
# compute_rmse_q.py output plus a new spatial mature-phase (lead day 40-60)
# MSE computation, matching this figure's own phase convention exactly
# (NOT the 30-60 convention GC-DETER-2023's compute_bias_maps.py uses
# elsewhere).

def csv(tag):
    p = CACHE / f"{tag}.csv"
    return pd.read_csv(p) if p.exists() else None

MODELS = ["fcn","aurora","prithvi","graphcast"]
data = {m: {k: csv(f"{m}_{k}_rmse") for k in ["tuv","ctrl","tins","uins"]}
        for m in MODELS}

def phase_mean(series, lead, lo=40, hi=60):
    mask = (lead > lo) & (lead <= hi)
    return float(np.nanmean(series[mask])) if mask.any() else np.nan

def synergy_scalar(ctrl, tuv, tins, uins, lead_c, lead_t):
    c = phase_mean(ctrl, lead_c); t = phase_mean(tins, lead_c); u = phase_mean(uins, lead_c)
    v = phase_mean(tuv, lead_t)
    if any(x != x for x in [c,t,u,v]) or c == 0: return np.nan
    i_t, i_uv, i_tuv = (c-t)/c*100, (c-u)/c*100, (c-v)/c*100
    return i_tuv - max(i_t, i_uv)

# ═══════════════════════════════════════════════════════════════════════════
# FIG 5.27 — Spatial Synergy Maps (Z500, Q850)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 5.27 -- spatial synergy maps (Z500, Q850)")

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
# GraphCast's own grid is loaded from its saved .npy files rather than
# hardcoded (compute_graphcast_tuv_cache.py saves the exact lat/lon arrays
# used, avoiding a sign/ordering mistake).
if (CACHE / "graphcast_lat.npy").exists():
    LATS["graphcast"] = np.load(CACHE / "graphcast_lat.npy")
    LONS["graphcast"] = np.load(CACHE / "graphcast_lon.npy")

SYN_FILES = {
    ("fcn",    "Z500"): "fcn_synergy_z500.npy",
    ("aurora", "Z500"): "aurora_synergy_z500.npy",
    ("prithvi","Z500"): "prithvi_synergy_H_lv7.npy",
    ("graphcast","Z500"): "graphcast_synergy_z500.npy",
    ("fcn",    "Q850"): "fcn_synergy_r850.npy",
    ("aurora", "Q850"): "aurora_synergy_q850.npy",
    ("prithvi","Q850"): "prithvi_synergy_QV_lv10.npy",
    ("graphcast","Q850"): "graphcast_synergy_q850.npy",
}

fig = plt.figure(figsize=(19.5, 9.3))
gs = fig.add_gridspec(2, 4, hspace=0.18, wspace=0.12,
                       top=0.88, bottom=0.06, left=0.02, right=0.99)
proj = ccrs.Robinson()
vmax = 15
LEVELS = np.linspace(-vmax, vmax, 11)
cmap = plt.cm.RdBu_r
norm = mcolors.BoundaryNorm(LEVELS, ncolors=cmap.N, extend="neither")

for row_i, var_label in enumerate(["Z500", "Q850"]):
    for col_i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[row_i, col_i], projection=proj)
        fpath = CACHE / SYN_FILES.get((m, var_label), "missing")
        if not fpath.exists():
            ax.set_title(f"{MOD_LABEL[m]} {var_label}\n(data pending)")
            ax.coastlines(lw=0.5)
            continue
        syn = np.load(str(fpath))
        lats, lons = LATS[m], LONS[m]
        syn_c = np.clip(syn, -vmax, vmax)
        lon2d, lat2d = np.meshgrid(lons, lats)
        im = ax.pcolormesh(lon2d, lat2d, syn_c, transform=ccrs.PlateCarree(),
                            cmap=cmap, norm=norm, shading="auto",
                            rasterized=True)
        ax.coastlines(lw=0.5, color="black")
        ax.add_feature(cfeature.BORDERS, lw=0.3)
        ax.set_title(f"{MOD_LABEL[m]} — {var_label}\n"
                      f"Synergy: {np.nanmean(syn):.1f}% ± {np.nanstd(syn):.1f}%",
                      fontsize=10, pad=6)
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05,
                             label="Synergy (%)", fraction=0.046, shrink=0.9,
                             ticks=LEVELS, spacing="uniform")
        cbar.ax.tick_params(labelsize=7.5)
        cbar.ax.set_xticklabels([f"{int(v)}" for v in LEVELS])

fig.suptitle("Fig 5.27 — Spatial Synergy Maps: T+UV − max(T,UV) Insertion\n"
             "Z500 and Q850 | Mature Phase (Days 40–60) | "
             "Positive = Dual Insertion Outperforms Single",
             fontsize=11, fontweight="bold", y=0.975)
fig.savefig(OUTDIR / "fig5_27_spatial_synergy_maps_z500_q850.png")
plt.close()
print("  -> Saved: fig5_27_spatial_synergy_maps_z500_q850.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 5.28 — Grouped Bar Chart of Synergy (Z500, Q850) Across Models
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 5.28 -- synergy bar chart (Z500, Q850)")

SYNERGY_VARS = ["Z500", "Q850"]
BAR_W   = 0.18
MOD_OFF = [-1.5 * BAR_W, -0.5 * BAR_W, 0.5 * BAR_W, 1.5 * BAR_W]
x = np.arange(len(SYNERGY_VARS))

fig, ax = plt.subplots(figsize=(9, 5.5))
for mi, m in enumerate(MODELS):
    syn_vals, ci_vals = [], []
    for var in SYNERGY_VARS:
        col = VAR_MAP.get(var, {}).get(m)
        df_tuv, df_ctrl = data[m]["tuv"], data[m]["ctrl"]
        df_tins, df_uins = data[m]["tins"], data[m]["uins"]
        if any(d is None for d in [df_tuv, df_ctrl, df_tins, df_uins]) or col is None \
           or col not in df_ctrl.columns:
            syn_vals.append(np.nan); ci_vals.append(0); continue

        lead_c = df_ctrl["lead_day"].values
        lead_t = df_tuv["lead_day"].values
        syn = synergy_scalar(df_ctrl[col].values, df_tuv[col].values,
                              df_tins[col].values, df_uins[col].values,
                              lead_c, lead_t)

        mat_c = (lead_c > 40) & (lead_c <= 60)
        mat_t = (lead_t > 40) & (lead_t <= 60)
        c_m = df_ctrl[col].values[mat_c]
        t_m = df_tins[col].values[mat_c]
        u_m = df_uins[col].values[mat_c]
        v_m_raw = df_tuv[col].values[mat_t][:len(c_m)]
        step_syn = np.where(c_m > 0,
            (c_m - v_m_raw) / c_m * 100 - np.maximum((c_m - t_m) / c_m * 100,
                                                       (c_m - u_m) / c_m * 100),
            np.nan)
        ci = np.nanstd(step_syn) / max(np.sqrt(np.sum(mat_c)), 1)
        syn_vals.append(syn); ci_vals.append(ci)

    bars = ax.bar(x + MOD_OFF[mi], syn_vals, width=BAR_W,
                   color=COLORS[MOD_LABEL[m]], label=MOD_LABEL[m],
                   edgecolor="white", lw=0.5, zorder=3)
    ax.errorbar(x + MOD_OFF[mi], syn_vals, yerr=ci_vals,
                fmt="none", color="black", capsize=3, lw=1.0, zorder=4)
    for bar, sv in zip(bars, syn_vals):
        if np.isnan(sv): continue
        ax.text(bar.get_x() + bar.get_width() / 2, sv + (0.2 if sv >= 0 else -0.4),
                 f"{sv:.1f}", ha="center", va="bottom" if sv >= 0 else "top",
                 fontsize=9, fontweight="500")

ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(SYNERGY_VARS, fontsize=13, fontweight="bold")
ax.set_ylabel("Synergy (%) — Mature Phase (d40–60)\n[+: dual insertion exceeds single]",
              fontsize=10)
ax.set_title("Fig 5.28 — Model Synergy Comparison: T+UV vs. Single-Variable Insertion\n"
             "Z500 and Q850", fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10, borderaxespad=0)
ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
plt.tight_layout()
fig.savefig(OUTDIR / "fig5_28_synergy_bar_chart_z500_q850.png")
plt.close()
print("  -> Saved: fig5_28_synergy_bar_chart_z500_q850.png")

print(f"\nDone -- 2 plots written to {OUTDIR}")
