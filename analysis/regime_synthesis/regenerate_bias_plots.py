#!/usr/bin/env python3
"""
regenerate_bias_plots.py
Regenerate all spatial bias plots for Aurora, FourCastNet, GraphCast-S, and Prithvi.
Reads pre-computed bias CSV (Aurora/FCN/GC) or npy (Prithvi) files.
Does NOT re-run any model forward pass.

Layout: one figure per variable.
  • Control-bias figures : 1 panel  (just the control bias map)
  • Insertion figures    : 3 panels (control bias | insertion bias | bias reduction)
  • Failure-mode figure  : 1 panel  (categorical map)

Improvements over original bias/plots/:
  • 14-level BoundaryNorm discrete colormap — no smooth gradient
  • Panel labels (a) (b) (c) on every subplot
  • 300 dpi, DejaVu Sans, research-paper rcParams
  • PiYG colormap for bias-reduction panel (green = improved)
  • Saves to bias/plots2/ per model; bias/plots/ is untouched

Output files (per model, example Aurora):
  Figure_Aurora_ctrl_bias_t500.png
  Figure_Aurora_t_ins_bias_t500.png   ← one per insertion × variable
  Figure_Aurora_failure_mode_map.png
"""
import warnings; warnings.filterwarnings("ignore")
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Publication rcParams ───────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    9.5,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.7,
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
})

BASE  = Path("/umbc/rs/pi_halem/users/yashd1")
PROJ  = ccrs.PlateCarree()
ALPHA = list(string.ascii_lowercase)
N_DISC = 14

# ── Colormap helpers ──────────────────────────────────────────────────────────
def disc_cmap_norm(vmax, n=N_DISC, name="RdBu_r"):
    vmax   = max(float(vmax), 1e-9)
    levels = np.linspace(-vmax, vmax, n + 1)
    cmap   = plt.get_cmap(name, n)
    norm   = mcolors.BoundaryNorm(levels, n)
    return cmap, norm, levels

def panel_label(ax, idx):
    ax.text(0.025, 0.965, f"({ALPHA[idx]})",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", color="#111111",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.5, alpha=0.88))

# ── Map panel helper ──────────────────────────────────────────────────────────
def draw_panel(ax, data, lons, lats, title, vmax, cmap_name="RdBu_r", pidx=None):
    cmap, norm, levels = disc_cmap_norm(vmax, name=cmap_name)
    im = ax.pcolormesh(lons, lats, data, transform=PROJ,
                       cmap=cmap, norm=norm, shading="auto", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.45, zorder=5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, linestyle=":", zorder=5)
    ax.set_global()
    ax.gridlines(draw_labels=False, linewidth=0.28, color="#888888",
                 alpha=0.45, linestyle="--", zorder=3)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    if pidx is not None:
        panel_label(ax, pidx)
    return im, levels

def add_cbar(fig, ax, im, label, levels):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      pad=0.05, shrink=0.88, aspect=28,
                      extend="both", extendfrac=0.03)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ticks = levels[::2]
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([f"{v:.2g}" for v in ticks], fontsize=7)
    return cb

# ═══════════════════════════════════════════════════════════════════════════════
# AURORA / FOURCASTNET / GRAPHCAST-S
# ═══════════════════════════════════════════════════════════════════════════════

STD_VARS = ["t500", "u500", "v500", "z500", "wind500"]
STD_VAR_LABELS = {
    "t500":    ("T500",    "K"),
    "u500":    ("U500",    "m s⁻¹"),
    "v500":    ("V500",    "m s⁻¹"),
    "z500":    ("Z500",    "m² s⁻²"),
    "wind500": ("Wind500", "m s⁻¹"),
}

MODELS_STD = {
    "Aurora": {
        "val_dir":  BASE / "Aurora_Pred_Files/bias/values",
        "plot_dir": BASE / "Aurora_Pred_Files/bias/plots2",
        "prefix":   "Aurora",
        "ins_runs": ["t_ins", "uv_ins", "z_ins"],
        "ins_labels": {
            "t_ins":  "T-Insertion",
            "uv_ins": "UV-Insertion",
            "z_ins":  "Z-Insertion",
        },
        "nlat": 720, "nlon": 1440,
        "lats": np.arange(90.0, -90.0, -0.25),
        "lons": np.linspace(0.0, 359.75, 1440),
        "mature_label": "Mature phase: Mar 3–Apr 1 2023  |  60 × 12-hr steps  |  IC: 2023-02-01 12Z",
    },
    "FourCastNet": {
        "val_dir":  BASE / "FourCastNet/bias/values",
        "plot_dir": BASE / "FourCastNet/bias/plots2",
        "prefix":   "FCN",
        "ins_runs": ["t_ins", "wind_ins", "z_ins"],
        "ins_labels": {
            "t_ins":    "T-Insertion",
            "wind_ins": "Wind-Insertion",
            "z_ins":    "Z-Insertion",
        },
        "nlat": 721, "nlon": 1440,
        "lats": np.linspace(90.0, -90.0, 721),
        "lons": np.linspace(0.0, 359.75, 1440),
        "mature_label": "Mature phase: Mar 3–Apr 1 2023  |  60 × 12-hr steps  |  IC: 2023-02-01 12Z",
    },
    "GraphCast-S": {
        "val_dir":  BASE / "gc_pred_files/bias/values",
        "plot_dir": BASE / "gc_pred_files/bias/plots2",
        "prefix":   "GC",
        "ins_runs": ["t_ins", "uv_ins", "z_ins"],
        "ins_labels": {
            "t_ins":  "T-Insertion",
            "uv_ins": "UV-Insertion",
            "z_ins":  "Z-Insertion",
        },
        "nlat": 181, "nlon": 360,
        "lats": np.arange(-90.0, 91.0, 1.0),
        "lons": np.arange(0.0, 360.0, 1.0),
        "mature_label": "Mature phase: Mar 2–Apr 1 2022  |  ≈60 × 6-hr steps  |  IC: 2022-01-01 06Z",
    },
}


def load_ctrl_bias(val_dir, prefix, var, nlat, nlon):
    path = val_dir / f"{prefix}_control_{var}_bias.csv"
    if not path.exists():
        return None
    arr = pd.read_csv(path, usecols=["bias"])["bias"].values
    if arr.size != nlat * nlon:
        return None
    b = arr.reshape(nlat, nlon).astype(np.float64)
    return b if np.abs(b).max() > 1e-6 else None


def load_combined(val_dir, prefix, ins_run, var, nlat, nlon):
    path = val_dir / f"{prefix}_{ins_run}_{var}_combined.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path, usecols=["control_bias", "insertion_bias", "bias_reduction"])
    if len(df) != nlat * nlon:
        return None, None, None
    cb = df["control_bias"].values.reshape(nlat, nlon).astype(np.float64)
    ib = df["insertion_bias"].values.reshape(nlat, nlon).astype(np.float64)
    br = df["bias_reduction"].values.reshape(nlat, nlon).astype(np.float64)
    return cb, ib, br


def make_ctrl_figure(lons, lats, bias_arr, lbl, unit, mname, ml):
    """Single-panel control bias figure for one variable."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5),
                           subplot_kw={"projection": PROJ})
    vmax = float(np.nanpercentile(np.abs(bias_arr), 97))
    im, lvl = draw_panel(ax, bias_arr, lons, lats,
                         f"Control — {lbl} Mature-Phase Mean Bias  [{unit}]",
                         vmax, pidx=0)
    add_cbar(fig, ax, im, unit, lvl)
    fig.suptitle(
        f"{mname}  ·  Control Run Mature-Phase Bias  —  {lbl}\n{ml}\n"
        "Bias = mean(forecast − ERA5)  |  Positive = model over-predicts  "
        "|  Colormap: RdBu_r  |  14 discrete levels",
        fontsize=10.5, fontweight="bold", y=1.02, color="#111111",
    )
    return fig


def make_ins_figure(lons, lats, cb, ib, br, lbl, unit,
                    mname, ins_lbl, ml):
    """Three-panel insertion comparison figure for one variable."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5),
                             subplot_kw={"projection": PROJ},
                             gridspec_kw={"wspace": 0.04})
    vb  = max(float(np.nanpercentile(np.abs(cb), 97)),
               float(np.nanpercentile(np.abs(ib), 97)))
    vr  = float(np.nanpercentile(np.abs(br), 97))

    im0, l0 = draw_panel(axes[0], cb, lons, lats,
                         f"(a)  Control — {lbl}  [{unit}]", vb, pidx=0)
    im1, l1 = draw_panel(axes[1], ib, lons, lats,
                         f"(b)  {ins_lbl} — {lbl}  [{unit}]", vb, pidx=1)
    im2, l2 = draw_panel(axes[2], br, lons, lats,
                         f"(c)  {lbl} Bias Reduction  [{unit}]  [+ = improved]",
                         vr, cmap_name="PiYG", pidx=2)

    add_cbar(fig, axes[0], im0, unit, l0)
    add_cbar(fig, axes[1], im1, unit, l1)
    add_cbar(fig, axes[2], im2, f"{unit}  [+ = improved]", l2)

    fig.suptitle(
        f"{mname}  ·  {ins_lbl} vs Control  —  {lbl}  Mature-Phase Bias\n{ml}\n"
        "Left/centre: RdBu_r  |  Right: PiYG (green = improved)  |  14 discrete levels  "
        "|  Bias Reduction = |Control| − |Insertion|",
        fontsize=10.5, fontweight="bold", y=1.02, color="#111111",
    )
    return fig


def failure_mode_figure(lons, lats, nlat, nlon,
                        ctrl_b, bias_reds, ins_labels_ordered,
                        T_thresh, diag_lbl, mature_label, mname, prefix, plot_dir):
    n_ins = len(bias_reds)
    if n_ins < 2:
        print(f"    {mname}: need ≥2 insertion runs for failure mode, skipping")
        return
    arrs = [bias_reds[k] for k in ins_labels_ordered]
    while len(arrs) < 3:
        arrs.append(np.zeros_like(ctrl_b))
    t_br, w_br, g_br = arrs[0], arrs[1], arrs[2]
    stk      = np.stack([t_br, w_br, g_br], axis=-1)
    best_idx = np.argmax(stk, axis=-1)
    wrst_idx = np.argmin(stk, axis=-1)
    max_val  = stk.max(axis=-1)
    min_val  = stk.min(axis=-1)
    class_arr = np.zeros(ctrl_b.shape, dtype=np.int8)
    all_impr  = (t_br > T_thresh) & (w_br > T_thresh) & (g_br > T_thresh)
    all_wors  = (t_br < -T_thresh) & (w_br < -T_thresh) & (g_br < -T_thresh)
    class_arr[(best_idx == 0) & (max_val >  T_thresh)] = 3
    class_arr[(best_idx == 1) & (max_val >  T_thresh)] = 4
    class_arr[(best_idx == 2) & (max_val >  T_thresh)] = 5
    class_arr[(wrst_idx == 0) & (min_val < -T_thresh)] = 6
    class_arr[(wrst_idx == 1) & (min_val < -T_thresh)] = 7
    class_arr[(wrst_idx == 2) & (min_val < -T_thresh)] = 8
    class_arr[all_impr] = 1
    class_arr[all_wors] = 2
    k0 = ins_labels_ordered[0].split("-")[0] if len(ins_labels_ordered) > 0 else "T"
    k1 = ins_labels_ordered[1].split("-")[0] if len(ins_labels_ordered) > 1 else "W"
    k2 = ins_labels_ordered[2].split("-")[0] if len(ins_labels_ordered) > 2 else "Z"
    CLASS_LABELS = {
        0: "Neutral / mixed",        1: "All insertions improve",
        2: "All insertions worsen",  3: f"{k0}-insertion best",
        4: f"{k1}-insertion best",   5: f"{k2}-insertion best",
        6: f"{k0}-insertion worst",  7: f"{k1}-insertion worst",
        8: f"{k2}-insertion worst",
    }
    COLORS = ["#b0b0b0", "#2ecc71", "#e74c3c",
              "#3498db", "#9b59b6", "#f39c12",
              "#1a5276", "#7d3c98", "#784212"]
    cmap_fm = mcolors.ListedColormap(COLORS)
    norm_fm = mcolors.BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap_fm.N)
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), subplot_kw={"projection": PROJ})
    im = ax.pcolormesh(lons, lats, class_arr,
                       cmap=cmap_fm, norm=norm_fm, transform=PROJ,
                       shading="auto", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.55, zorder=5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":", zorder=5)
    ax.set_global()
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#888",
                 alpha=0.45, linestyle="--")
    panel_label(ax, 0)
    cb = plt.colorbar(im, ax=ax, orientation="vertical",
                      pad=0.02, shrink=0.82, aspect=22)
    cb.set_ticks(range(9))
    cb.set_ticklabels([CLASS_LABELS[i] for i in range(9)], fontsize=7.5)
    cb.set_label("Insertion Response Class", fontsize=9)
    cos_lat = np.cos(np.deg2rad(lats))
    wt_2d   = np.tile(cos_lat[:, np.newaxis], (1, nlon))
    total_w = float(wt_2d.sum())
    pct_txt = "\n".join(
        f"{CLASS_LABELS[c]}: {100.*float(wt_2d[class_arr==c].sum())/total_w:.1f}%"
        for c in range(9))
    ax.text(0.01, 0.02, pct_txt, transform=ax.transAxes,
            fontsize=6.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.80))
    ax.set_title(
        f"Insertion Response Classification  —  diagnostic: {diag_lbl}\n"
        f"Threshold = {T_thresh:.3f}  |  Class = which insertion reduces {diag_lbl} bias most",
        fontsize=10.5, fontweight="bold", pad=8,
    )
    fig.suptitle(
        f"{mname}  ·  Failure-Mode / Insertion Response Summary\n{mature_label}",
        fontsize=12, fontweight="bold", y=1.01, color="#111111",
    )
    out = plot_dir / f"Figure_{prefix}_failure_mode_map.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out.name}")


def gen_std_figures(mname, cfg):
    val_dir    = cfg["val_dir"]
    plot_dir   = cfg["plot_dir"]
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix     = cfg["prefix"]
    nlat, nlon = cfg["nlat"], cfg["nlon"]
    lats, lons = cfg["lats"], cfg["lons"]
    ins_runs   = cfg["ins_runs"]
    ins_labels = cfg["ins_labels"]
    ml         = cfg["mature_label"]

    # Load control biases
    ctrl_biases = {}
    for var in STD_VARS:
        b = load_ctrl_bias(val_dir, prefix, var, nlat, nlon)
        if b is not None:
            ctrl_biases[var] = b
    if not ctrl_biases:
        print(f"  {mname}: no control bias CSVs found, skipping")
        return
    print(f"  {mname}: {list(ctrl_biases.keys())}")

    # ── Control bias: one figure per variable ──────────────────────────────────
    for var, b in ctrl_biases.items():
        lbl, unit = STD_VAR_LABELS[var]
        fig = make_ctrl_figure(lons, lats, b, lbl, unit, mname, ml)
        out = plot_dir / f"Figure_{prefix}_ctrl_bias_{var}.png"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    → {out.name}")

    # ── Insertion comparison: one figure per insertion × variable ──────────────
    diag_var = "z500" if "z500" in ctrl_biases else list(ctrl_biases.keys())[0]
    bias_reds_diag = {}

    for ins_run in ins_runs:
        ins_lbl = ins_labels[ins_run]
        for var in list(ctrl_biases.keys()):
            cb, ib, br = load_combined(val_dir, prefix, ins_run, var, nlat, nlon)
            if cb is None:
                continue
            if np.abs(ib).max() <= 1e-6 and np.abs(br).max() <= 1e-6:
                continue
            # store bias_red for failure mode
            if var == diag_var:
                bias_reds_diag[ins_lbl] = br
            lbl, unit = STD_VAR_LABELS[var]
            fig = make_ins_figure(lons, lats, cb, ib, br,
                                  lbl, unit, mname, ins_lbl, ml)
            out = plot_dir / f"Figure_{prefix}_{ins_run}_bias_{var}.png"
            fig.savefig(str(out), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"    → {out.name}")

    # ── Failure-mode map ───────────────────────────────────────────────────────
    ctrl_d   = ctrl_biases[diag_var]
    T_thresh = 0.10 * float(np.percentile(np.abs(ctrl_d), 90))
    diag_lbl = STD_VAR_LABELS[diag_var][0]
    ins_labels_ordered = [ins_labels[r] for r in ins_runs
                          if ins_labels[r] in bias_reds_diag]
    br_ordered = {k: bias_reds_diag[k] for k in ins_labels_ordered}
    failure_mode_figure(lons, lats, nlat, nlon,
                        ctrl_d, br_ordered, ins_labels_ordered,
                        T_thresh, diag_lbl, ml, mname, prefix, plot_dir)


for mname, cfg in MODELS_STD.items():
    print(f"\n{'='*60}\n  {mname}")
    gen_std_figures(mname, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# PRITHVI  (npy-based bias storage)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n  Prithvi")

PRI_VAL_DIR  = BASE / "prithvi_new/bias/values"
PRI_PLOT_DIR = BASE / "prithvi_new/bias/plots2"
PRI_PLOT_DIR.mkdir(parents=True, exist_ok=True)

PRI_RUNS     = ["Control", "temp_ins", "wind_ins", "geo_ins"]
PRI_INS_RUNS = ["temp_ins", "wind_ins", "geo_ins"]
PRI_INS_LBLS = {
    "temp_ins": "T-Insertion",
    "wind_ins": "Wind-Insertion",
    "geo_ins":  "Geo-Insertion",
}
PRI_VARS  = ["T_lev51", "H_lev51", "U_lev45", "SLP", "T2M"]
PRI_VLBLS = {
    "T_lev51": ("T@532hPa", "K"),
    "H_lev51": ("H@532hPa", "m"),
    "U_lev45": ("U@310hPa", "m s⁻¹"),
    "SLP":     ("SLP",      "hPa"),
    "T2M":     ("T2M",      "K"),
}
NLAT_PRI, NLON_PRI = 360, 576
lats_pri = np.arange(-89.5, 90.5, 0.5)
lons_pri = np.linspace(-180.0, 180.0 - 360.0 / 576, 576)
PRI_ML   = ("Mature phase: steps 120–239 (day 30–60)  "
            "|  MERRA-2 truth  |  IC: 2022-01-01 00Z")


def load_pri(run, var):
    path = PRI_VAL_DIR / f"bias_{run}_{var}.npy"
    if not path.exists():
        return None
    arr = np.load(str(path)).astype(np.float64)
    return arr if np.abs(arr).max() > 1e-6 else None


# Load all
pri_bias = {run: {var: load_pri(run, var) for var in PRI_VARS}
            for run in PRI_RUNS}

ctrl_pri = {v: pri_bias["Control"][v] for v in PRI_VARS
            if pri_bias["Control"][v] is not None}
print(f"  Prithvi control vars: {list(ctrl_pri.keys())}")

# ── Control bias: one figure per variable ─────────────────────────────────────
for var, b in ctrl_pri.items():
    lbl, unit = PRI_VLBLS[var]
    fig = make_ctrl_figure(lons_pri, lats_pri, b, lbl, unit, "Prithvi WxC", PRI_ML)
    # Override suptitle for MERRA-2
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5),
                           subplot_kw={"projection": PROJ})
    vmax = float(np.nanpercentile(np.abs(b), 97))
    im, lvl = draw_panel(ax, b, lons_pri, lats_pri,
                         f"Control — {lbl} Mature-Phase Mean Bias  [{unit}]",
                         vmax, pidx=0)
    add_cbar(fig, ax, im, unit, lvl)
    fig.suptitle(
        f"Prithvi WxC  ·  Control Run Mature-Phase Bias  —  {lbl}\n{PRI_ML}\n"
        "Bias = mean(forecast − MERRA-2)  |  Positive = model over-predicts  "
        "|  Colormap: RdBu_r  |  14 discrete levels",
        fontsize=10.5, fontweight="bold", y=1.02, color="#111111",
    )
    out = PRI_PLOT_DIR / f"Figure_Prithvi_ctrl_bias_{var}.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out.name}")

# ── Insertion comparison: one figure per insertion × variable ─────────────────
diag_var_pri = "H_lev51" if ctrl_pri.get("H_lev51") is not None else (
    list(ctrl_pri.keys())[0] if ctrl_pri else None)
br_diag_pri = {}

for ins_run in PRI_INS_RUNS:
    ins_lbl = PRI_INS_LBLS[ins_run]
    for var in list(ctrl_pri.keys()):
        cb_ = ctrl_pri[var]
        ib_ = pri_bias[ins_run][var]
        if ib_ is None:
            continue
        br_ = np.abs(cb_) - np.abs(ib_)
        if var == diag_var_pri:
            br_diag_pri[ins_lbl] = br_
        lbl, unit = PRI_VLBLS[var]
        fig = make_ins_figure(lons_pri, lats_pri, cb_, ib_, br_,
                              lbl, unit, "Prithvi WxC", ins_lbl, PRI_ML)
        # Override suptitle for MERRA-2
        fig.texts[0].set_text(
            f"Prithvi WxC  ·  {ins_lbl} vs Control  —  {lbl}  Mature-Phase Bias\n"
            f"{PRI_ML}\n"
            "Left/centre: RdBu_r  |  Right: PiYG (green = improved)  |  "
            "14 discrete levels  |  Bias Reduction = |Control| − |Insertion|"
        )
        out = PRI_PLOT_DIR / f"Figure_Prithvi_{ins_run}_bias_{var}.png"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    → {out.name}")

# ── Failure-mode map ──────────────────────────────────────────────────────────
if diag_var_pri and len(br_diag_pri) >= 2:
    ctrl_d   = ctrl_pri[diag_var_pri]
    T_thresh = 0.10 * float(np.percentile(np.abs(ctrl_d), 90))
    diag_lbl = PRI_VLBLS[diag_var_pri][0]
    ins_labels_ordered_pri = list(br_diag_pri.keys())
    failure_mode_figure(lons_pri, lats_pri, NLAT_PRI, NLON_PRI,
                        ctrl_d, br_diag_pri, ins_labels_ordered_pri,
                        T_thresh, diag_lbl, PRI_ML,
                        "Prithvi WxC", "Prithvi", PRI_PLOT_DIR)
else:
    print("  Prithvi: insufficient overlap for failure-mode figure, skipping")

print(f"\nDone — all figures written to bias/plots2/ in each model directory")
