#!/usr/bin/env python3
"""
Regenerate all latband + latslice plots for Aurora, FourCastNet, GraphCast-S, Prithvi.
Two changes vs originals:
  1. U500 and V500 shown independently  (wind500 replaced by v500)
  2. Same fixed y-scale across all bar charts  (YLIM = -20..105 %)
     + sharey='row' on time-series figures  (same RMSE scale per variable across bands/slices)

Reads only from existing CSVs — no data loading needed. Fast.
Saves directly into each model's plots directory (overwrites originals).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams, ticker
from pathlib import Path

rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    10.5,
    "axes.labelsize":    9.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "#444444",
    "xtick.bottom":      True,
    "ytick.left":        True,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.30,
    "grid.linestyle":    ":",
    "grid.linewidth":    0.6,
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
})

BASE   = Path("/umbc/rs/pi_halem/users/yashd1")
YLIM_BAR = (-20.0, 105.0)   # fixed for ALL bar charts

# ── Helpers ────────────────────────────────────────────────────────────────
def load_csv(valdir, fname):
    p = valdir / fname
    if not p.exists():
        print(f"    MISSING: {fname}"); return None
    return pd.read_csv(p)

def save_fig(fig, path):
    fig.savefig(str(path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  → {path.name}")

# ── Time-series figure  ────────────────────────────────────────────────────
def make_ts_figure(cfg, valdir, file_fn, region_cols, region_label,
                   focus_run, vars_list, title_suffix, out_path,
                   lead_col, mature_thresh):
    """
    rows = variables,  cols = bands/slices
    sharey='row' → same y-axis per variable across all regions.
    """
    all_runs = [cfg["ctrl"]] + cfg["ins_runs"]
    nrows    = len(vars_list)
    ncols    = len(region_cols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows + 1.8),
        sharey="row",
        gridspec_kw={"hspace": 0.55, "wspace": 0.12},
    )
    if nrows == 1: axes = [axes]
    if ncols == 1: axes = [[ax] for ax in axes]

    for row, var in enumerate(vars_list):
        lbl, unit = cfg["vars"][var]
        for col, reg in enumerate(region_cols):
            ax = axes[row][col]
            for run in all_runs:
                df = load_csv(valdir, file_fn(run, var))
                if df is None: continue
                y  = df[reg].values
                ld = df[lead_col].values
                col_r = cfg["run_colors"][run]
                lw    = 2.2 if run in (cfg["ctrl"], focus_run) else 1.0
                ls    = "-" if run in (cfg["ctrl"], focus_run) else ":"
                ax.plot(ld, y, color=col_r, linewidth=lw, linestyle=ls,
                        label=cfg["run_labels"][run], zorder=4)

            ax.axvspan(mature_thresh, ld[-1] if 'ld' in dir() else 60,
                       alpha=0.10, color="gray", zorder=1)
            ax.axvline(mature_thresh, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.7, zorder=2)
            ax.set_xlim(0, ld[-1] if 'ld' in dir() else 60)
            ax.set_xlabel(f"Lead {'day' if 'day' in lead_col else 'hour'}", fontsize=8.5)
            ax.grid(True, which="major", alpha=0.25, linewidth=0.5, axis="both")
            ax.tick_params(labelsize=8.5)
            ax.set_title(f"{lbl}  —  {reg}", fontsize=9.5,
                         fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(f"RMSE  [{unit}]", fontsize=8.5)
            else:
                ax.yaxis.set_tick_params(labelleft=False)

    # legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen: seen[l] = h
    seen["Mature phase"] = mpatches.Patch(color="gray", alpha=0.25,
                                          label="Mature phase")
    fig.legend(seen.values(), seen.keys(),
               loc="lower center", ncol=len(seen),
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.01))

    focus_lbl = cfg["run_labels"][focus_run]
    var_str   = " & ".join(cfg["vars"][v][0] for v in vars_list)
    fig.suptitle(
        f"{cfg['name']}  ·  {focus_lbl}  |  Latitude-{region_label} RMSE Time Series\n"
        f"Variables: {var_str}  ·  {title_suffix}",
        fontsize=11.5, fontweight="bold", y=1.02, color="#111111",
    )
    save_fig(fig, out_path)

# ── Bar-summary figure  ────────────────────────────────────────────────────
def make_bar_figure(cfg, valdir, file_fn, region_cols, region_tick_labels,
                    region_label, bar_vars, out_path, lead_col, mature_thresh):
    n_vars = len(bar_vars)
    n_regs = len(region_cols)
    n_runs = len(cfg["ins_runs"])
    bar_w  = 0.22
    x      = np.arange(n_regs)
    offs   = (np.arange(n_runs) - (n_runs - 1) / 2.0) * bar_w

    fig, axes = plt.subplots(
        2, n_vars,
        figsize=(max(3.5 * n_vars, 14), 9.0),
        gridspec_kw={"hspace": 0.60, "wspace": 0.36},
    )
    if n_vars == 1: axes = [[axes[0]], [axes[1]]]

    phases = [
        ("adj", "(a)  Adjustment Phase  (Lead Days 0–30)"),
        ("mat", "(b)  Mature Phase  (Lead Days 30–60)"),
    ]

    for row, (phase_key, phase_title) in enumerate(phases):
        for col, var in enumerate(bar_vars):
            ax   = axes[row][col]
            lbl, unit = cfg["vars"].get(var, (var, ""))

            for ri, ins_run in enumerate(cfg["ins_runs"]):
                df_c = load_csv(valdir, file_fn(cfg["ctrl"], var))
                df_i = load_csv(valdir, file_fn(ins_run, var))
                if df_c is None or df_i is None: continue

                mask = (df_c[lead_col] >= mature_thresh) if phase_key == "mat" \
                       else (df_c[lead_col] < mature_thresh)
                mask_i = (df_i[lead_col] >= mature_thresh) if phase_key == "mat" \
                         else (df_i[lead_col] < mature_thresh)

                vals = []
                for reg in region_cols:
                    if reg not in df_c.columns:
                        vals.append(np.nan); continue
                    cm = df_c.loc[mask,   reg].mean()
                    im = df_i.loc[mask_i, reg].mean()
                    pct = (100.0 * (cm - im) / abs(cm)
                           if (not np.isnan(cm) and cm != 0) else np.nan)
                    vals.append(pct)

                color = cfg["run_colors"][ins_run]
                rlbl  = cfg["run_labels"][ins_run]
                bars  = ax.bar(x + offs[ri], vals, bar_w * 0.88,
                               color=color, edgecolor="white",
                               linewidth=0.5, alpha=0.88,
                               label=rlbl, zorder=3)

                for bar_obj, v in zip(bars, vals):
                    if np.isnan(v): continue
                    disp    = float(np.clip(v, YLIM_BAR[0], YLIM_BAR[1]))
                    clipped = (v < YLIM_BAR[0]) or (v > YLIM_BAR[1])
                    txt     = f"{v:.0f}%*" if clipped else f"{v:.1f}%"
                    col_t   = "#CC0000" if clipped else "#333333"
                    if disp >= 0:
                        yp, va = min(disp + 1.5, YLIM_BAR[1] - 4), "bottom"
                    else:
                        yp, va = max(disp - 1.5, YLIM_BAR[0] + 2), "top"
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            yp, txt, ha="center", va=va, fontsize=6.5,
                            rotation=90, color=col_t,
                            fontweight="bold" if clipped else "normal")

            ax.axhline(0, color="#333333", linewidth=0.9,
                       linestyle="--", alpha=0.60, zorder=2)
            ax.set_ylim(YLIM_BAR)
            ax.set_xticks(x)
            ax.set_xticklabels(region_tick_labels, fontsize=8.5,
                               fontweight="bold", rotation=10)
            ax.set_title(f"{lbl}  [{unit}]", fontsize=10,
                         fontweight="bold", pad=5)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
            ax.tick_params(axis="y", which="minor", length=2, color="#bbbbbb")
            if col == 0:
                ax.set_ylabel(f"{phase_title}\n% RMSE Reduction vs Control",
                              fontsize=8.5)
            if row == 0 and col == n_vars - 1:
                handles = [plt.Rectangle((0, 0), 1, 1,
                                         facecolor=cfg["run_colors"][r],
                                         edgecolor="white", alpha=0.88)
                           for r in cfg["ins_runs"]]
                lbls = [cfg["run_labels"][r] for r in cfg["ins_runs"]]
                ax.legend(handles, lbls, fontsize=8.5, loc="upper right",
                          framealpha=0.92, edgecolor="#cccccc")

    fig.suptitle(
        f"{cfg['name']}  ·  Latitude-{region_label} % RMSE Improvement\n"
        "Positive = RMSE reduction vs Control  |  Cosine-latitude weighted  |  "
        "Fixed scale −20 to 105 %",
        fontsize=11.5, fontweight="bold", y=1.025, color="#111111",
    )
    fig.text(0.99, -0.015,
             "* value outside plot range; actual value annotated in red",
             ha="right", va="bottom", fontsize=7.5,
             color="#CC0000", style="italic")
    save_fig(fig, out_path)

# ══════════════════════════════════════════════════════════════════════════
# ── Model configs  ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
BASE_AUR = BASE / "Aurora_Pred_Files"
BASE_FCN = BASE / "FourCastNet"
BASE_GC  = BASE / "gc_pred_files"
BASE_PRI = BASE / "prithvi_new"

_RUN_COLORS_STD = {
    "control":  "#2c3e50", "Control":  "#2c3e50",
    "t_ins":    "#e74c3c", "temp_ins": "#e74c3c",
    "uv_ins":   "#3498db", "wind_ins": "#3498db",
    "z_ins":    "#f39c12", "geo_ins":  "#f39c12",
}
_VARS_STD = {
    "t500":   ("T500",    "K"),
    "u500":   ("U500",    "m s⁻¹"),
    "v500":   ("V500",    "m s⁻¹"),
    "z500":   ("Z500",    "m² s⁻²"),
    "wind10": ("Wind10m", "m s⁻¹"),
}
BAR_VARS_STD = ["t500", "u500", "v500", "z500", "wind10"]

MODELS = [
    # ── Aurora ───────────────────────────────────────────────────────────
    {
        "name":          "Aurora",
        "ctrl":          "control",
        "ins_runs":      ["t_ins", "uv_ins", "z_ins"],
        "run_labels":    {"control":"Control","t_ins":"T-Ins",
                          "uv_ins":"UV-Ins","z_ins":"Z-Ins"},
        "run_colors":    _RUN_COLORS_STD,
        "vars":          _VARS_STD,
        "lead_col":      "lead_day",
        "mature_thresh": 30.0,
        # latband
        "band_valdir":  BASE_AUR / "lats/values",
        "band_plotdir": BASE_AUR / "lats/plots",
        "band_cols":    ["Tropics","Midlatitudes","Polar"],
        "band_fn":      lambda run, var: f"Aurora_{run}_{var}_latband_rmse.csv",
        "band_ts_figs": [
            {"stem":"Figure_5_36_Aurora_t_insertion_latband_rmse",
             "run":"t_ins","vars":["u500","v500","z500"],
             "sfx":"Probes thermal-wind & hypsometric coupling by latitude"},
            {"stem":"Figure_5_37_Aurora_wind_insertion_latband_rmse",
             "run":"uv_ins","vars":["t500","z500"],
             "sfx":"Probes horizontal advection & geostrophic coupling by latitude"},
            {"stem":"Figure_5_38_Aurora_z_insertion_latband_rmse",
             "run":"z_ins","vars":["t500","u500","v500"],
             "sfx":"Probes geostrophic balance & hypsometric relationship by latitude"},
        ],
        "band_bar_stem": "Figure_5_39_Aurora_latband_bar_summary",
        # latslice
        "slice_valdir":  BASE_AUR / "lat_bands/values",
        "slice_plotdir": BASE_AUR / "lat_bands/plots",
        "slice_cols":    ["2°N","14°N","40°N","60°N"],
        "slice_fn":      lambda run, var: f"Aurora_{run}_{var}_latslice_rmse.csv",
        "slice_ts_figs": [
            {"stem":"Figure_LS2_Aurora_t_insertion_latslice_rmse",
             "run":"t_ins","vars":["u500","v500","z500"],
             "sfx":"Probes thermal-wind coupling by latitude slice"},
            {"stem":"Figure_LS3_Aurora_wind_insertion_latslice_rmse",
             "run":"uv_ins","vars":["t500","z500"],
             "sfx":"Probes geostrophic coupling by latitude slice"},
            {"stem":"Figure_LS4_Aurora_z_insertion_latslice_rmse",
             "run":"z_ins","vars":["t500","u500","v500"],
             "sfx":"Probes geostrophic balance by latitude slice"},
        ],
        "slice_bar_stem": "Figure_LS5_Aurora_latslice_bar_summary",
    },
    # ── FourCastNet ───────────────────────────────────────────────────────
    {
        "name":          "FourCastNet",
        "ctrl":          "control",
        "ins_runs":      ["t_ins","wind_ins","z_ins"],
        "run_labels":    {"control":"Control","t_ins":"T-Ins",
                          "wind_ins":"Wind-Ins","z_ins":"Z-Ins"},
        "run_colors":    _RUN_COLORS_STD,
        "vars":          _VARS_STD,
        "lead_col":      "lead_day",
        "mature_thresh": 30.0,
        "band_valdir":  BASE_FCN / "lats/values",
        "band_plotdir": BASE_FCN / "lats/plots",
        "band_cols":    ["Tropics","Midlatitudes","Polar"],
        "band_fn":      lambda run, var: f"FCN_{run}_{var}_latband_rmse.csv",
        "band_ts_figs": [
            {"stem":"Figure_5_36_FCN_t_insertion_latband_rmse",
             "run":"t_ins","vars":["u500","v500","z500"],
             "sfx":"Probes thermal-wind & hypsometric coupling by latitude"},
            {"stem":"Figure_5_37_FCN_wind_insertion_latband_rmse",
             "run":"wind_ins","vars":["t500","z500"],
             "sfx":"Probes horizontal advection & geostrophic coupling by latitude"},
            {"stem":"Figure_5_38_FCN_z_insertion_latband_rmse",
             "run":"z_ins","vars":["t500","u500","v500"],
             "sfx":"Probes geostrophic balance & hypsometric relationship by latitude"},
        ],
        "band_bar_stem": "Figure_5_39_FCN_latband_bar_summary",
        "slice_valdir":  BASE_FCN / "lat_bands/values",
        "slice_plotdir": BASE_FCN / "lat_bands/plots",
        "slice_cols":    ["2°N","14°N","40°N","60°N"],
        "slice_fn":      lambda run, var: f"FCN_{run}_{var}_latslice_rmse.csv",
        "slice_ts_figs": [
            {"stem":"Figure_LS2_FCN_t_insertion_latslice_rmse",
             "run":"t_ins","vars":["u500","v500","z500"],
             "sfx":"Probes thermal-wind coupling by latitude slice"},
            {"stem":"Figure_LS3_FCN_wind_insertion_latslice_rmse",
             "run":"wind_ins","vars":["t500","z500"],
             "sfx":"Probes geostrophic coupling by latitude slice"},
            {"stem":"Figure_LS4_FCN_z_insertion_latslice_rmse",
             "run":"z_ins","vars":["t500","u500","v500"],
             "sfx":"Probes geostrophic balance by latitude slice"},
        ],
        "slice_bar_stem": "Figure_LS5_FCN_latslice_bar_summary",
    },
    # ── GraphCast-S ───────────────────────────────────────────────────────
    {
        "name":          "GraphCast-S",
        "ctrl":          "control",
        "ins_runs":      ["t_ins","uv_ins","z_ins"],
        "run_labels":    {"control":"Control","t_ins":"T-Ins",
                          "uv_ins":"UV-Ins","z_ins":"Z-Ins"},
        "run_colors":    _RUN_COLORS_STD,
        "vars":          _VARS_STD,
        "lead_col":      "lead_day",
        "mature_thresh": 30.0,
        "band_valdir":  BASE_GC / "lats/values",
        "band_plotdir": BASE_GC / "lats/plots",
        "band_cols":    ["Tropics","Midlatitudes","Polar"],
        "band_fn":      lambda run, var: f"GC_{run}_{var}_latband_rmse.csv",
        "band_ts_figs": [
            {"stem":"Figure_GC_t_insertion_latband_rmse",
             "run":"t_ins","vars":["u500","v500","z500"],
             "sfx":"Probes thermal-wind & hypsometric coupling by latitude"},
            {"stem":"Figure_GC_wind_insertion_latband_rmse",
             "run":"uv_ins","vars":["t500","z500"],
             "sfx":"Probes horizontal advection & geostrophic coupling by latitude"},
            {"stem":"Figure_GC_z_insertion_latband_rmse",
             "run":"z_ins","vars":["t500","u500","v500"],
             "sfx":"Probes geostrophic balance & hypsometric relationship by latitude"},
        ],
        "band_bar_stem": "Figure_GC_latband_bar_summary",
        "slice_valdir":  BASE_GC / "lat_bands/values",
        "slice_plotdir": BASE_GC / "lat_bands/plots",
        "slice_cols":    ["2°N","14°N","40°N","60°N"],
        "slice_fn":      lambda run, var: f"GC_{run}_{var}_latslice_rmse.csv",
        "slice_ts_figs": [
            {"stem":"Figure_GC_t_insertion_latslice_rmse",
             "run":"t_ins","vars":["u500","v500","z500"],
             "sfx":"Probes thermal-wind coupling by latitude slice"},
            {"stem":"Figure_GC_wind_insertion_latslice_rmse",
             "run":"uv_ins","vars":["t500","z500"],
             "sfx":"Probes geostrophic coupling by latitude slice"},
            {"stem":"Figure_GC_z_insertion_latslice_rmse",
             "run":"z_ins","vars":["t500","u500","v500"],
             "sfx":"Probes geostrophic balance by latitude slice"},
        ],
        "slice_bar_stem": "Figure_GC_latslice_bar_summary",
    },
]

# ══════════════════════════════════════════════════════════════════════════
# ── Aurora / FCN / GC loop  ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
for cfg in MODELS:
    print(f"\n{'='*60}\n  {cfg['name']}")

    for kind, valdir, plotdir, cols, file_fn, ts_figs, bar_stem in [
        ("Band",  cfg["band_valdir"],  cfg["band_plotdir"],
         cfg["band_cols"],  cfg["band_fn"],
         cfg["band_ts_figs"],  cfg["band_bar_stem"]),
        ("Slice", cfg["slice_valdir"], cfg["slice_plotdir"],
         cfg["slice_cols"], cfg["slice_fn"],
         cfg["slice_ts_figs"], cfg["slice_bar_stem"]),
    ]:
        print(f"  Lat-{kind}")

        # time-series figures
        for fig_def in ts_figs:
            make_ts_figure(
                cfg, valdir, file_fn,
                region_cols    = cols,
                region_label   = kind,
                focus_run      = fig_def["run"],
                vars_list      = fig_def["vars"],
                title_suffix   = fig_def["sfx"],
                out_path       = plotdir / f"{fig_def['stem']}.png",
                lead_col       = cfg["lead_col"],
                mature_thresh  = cfg["mature_thresh"],
            )

        # bar summary
        make_bar_figure(
            cfg, valdir, file_fn,
            region_cols      = cols,
            region_tick_labels = cols,
            region_label     = kind,
            bar_vars         = BAR_VARS_STD,
            out_path         = plotdir / f"{bar_stem}.png",
            lead_col         = cfg["lead_col"],
            mature_thresh    = cfg["mature_thresh"],
        )

# ══════════════════════════════════════════════════════════════════════════
# ── Prithvi  ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n  Prithvi")

PRI_BAND_VALDIR  = BASE_PRI / "lats/values"
PRI_BAND_PLOTDIR = BASE_PRI / "lats/plots"
PRI_SLICE_VALDIR = BASE_PRI / "lat_bands/values"
PRI_SLICE_PLOTDIR = BASE_PRI / "lat_bands/plots"

PRI_CFG = {
    "name":          "Prithvi",
    "ctrl":          "Control",
    "ins_runs":      ["temp_ins","wind_ins","geo_ins"],
    "run_labels":    {"Control":"Control","temp_ins":"T-Ins",
                      "wind_ins":"Wind-Ins","geo_ins":"Geo-Ins"},
    "run_colors":    _RUN_COLORS_STD,
    "vars": {
        "T_lev51": ("T@532hPa",  "K"),
        "H_lev51": ("H@532hPa",  "m"),
        "U_lev51": ("U@532hPa",  "m s⁻¹"),
        "V_lev51": ("V@532hPa",  "m s⁻¹"),
        "T2M":     ("T2M",       "K"),
        "SLP":     ("SLP",       "hPa"),
    },
    "lead_col":      "lead_hour",
    "mature_thresh": 720.0,
}
PRI_BAR_VARS   = ["T_lev51","H_lev51","U_lev51","V_lev51","T2M","SLP"]
PRI_BAND_COLS  = ["Tropics","Midlatitudes","Polar"]
PRI_SLICE_COLS = ["2N","14N","40N","60N"]
PRI_BAND_FN    = lambda run, var: f"latband_rmse_{var}_{run}.csv"
PRI_SLICE_FN   = lambda run, var: f"latslice_rmse_{var}_{run}.csv"

PRI_TS_FIGS = [
    {"run":"temp_ins","vars":["H_lev51","U_lev51","V_lev51"],
     "sfx":"Probes thermal-wind coupling at ~532 hPa"},
    {"run":"wind_ins","vars":["T_lev51","H_lev51"],
     "sfx":"Probes advection & geostrophic coupling at ~532 hPa"},
    {"run":"geo_ins", "vars":["T_lev51","U_lev51","V_lev51"],
     "sfx":"Probes geostrophic balance at ~532 hPa"},
]

for kind, valdir, plotdir, cols, file_fn, ts_stem_pfx in [
    ("Band",  PRI_BAND_VALDIR,  PRI_BAND_PLOTDIR,  PRI_BAND_COLS,
     PRI_BAND_FN,  "latband_rmse"),
    ("Slice", PRI_SLICE_VALDIR, PRI_SLICE_PLOTDIR, PRI_SLICE_COLS,
     PRI_SLICE_FN, "latslice_rmse"),
]:
    print(f"  Prithvi Lat-{kind}")

    for fig_def in PRI_TS_FIGS:
        stem = f"{ts_stem_pfx}_{fig_def['run']}_ins"
        make_ts_figure(
            PRI_CFG, valdir, file_fn,
            region_cols   = cols,
            region_label  = kind,
            focus_run     = fig_def["run"],
            vars_list     = fig_def["vars"],
            title_suffix  = fig_def["sfx"],
            out_path      = plotdir / f"{stem}.png",
            lead_col      = PRI_CFG["lead_col"],
            mature_thresh = PRI_CFG["mature_thresh"],
        )

    make_bar_figure(
        PRI_CFG, valdir, file_fn,
        region_cols        = cols,
        region_tick_labels = cols,
        region_label       = kind,
        bar_vars           = PRI_BAR_VARS,
        out_path           = plotdir / f"{ts_stem_pfx}_bar_summary.png",
        lead_col           = PRI_CFG["lead_col"],
        mature_thresh      = PRI_CFG["mature_thresh"],
    )

print("\nDone — all latband + latslice plots regenerated.")
