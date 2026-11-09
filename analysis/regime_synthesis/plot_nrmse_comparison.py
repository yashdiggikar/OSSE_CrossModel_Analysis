"""
Cross-model NRMSE improvement grouped bar charts — Figure 5.11 style.
3 plots: T-insertion, Wind-insertion, Geo-insertion.

Layout  : X-axis = 4 models | colored bars = variables
Phase   : side-by-side subplots — Adjustment (days 0–30) vs Mature (days 30–60)
Variables shown (self-variable excluded per experiment):
  T-ins    → Z500, U500  (T500 excluded — trivial self-insertion)
  Wind-ins → T500, Z500  (U500 excluded — trivial self-insertion)
  Geo-ins  → T500, U500  (Z500 excluded — trivial self-insertion)
Scale   : fixed −15 to 105 % across ALL plots
Quality : 300 dpi, publication-ready
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from pathlib import Path

# ── Publication rcParams ───────────────────────────────────────────────────
rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   11,
    "ytick.labelsize":   10,
    "legend.fontsize":   11,
    "axes.linewidth":    0.9,
    "axes.edgecolor":    "#444444",
    "xtick.bottom":      False,
    "ytick.left":        True,
    "xtick.major.size":  0,
    "ytick.major.size":  4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.35,
    "grid.linestyle":    ":",
    "grid.linewidth":    0.7,
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
})

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path("/umbc/rs/pi_halem/users/yashd1")
OUT         = BASE / "common"
AURORA_DIR  = BASE / "Aurora_Pred_Files/nrmse/values"
FCN_DIR     = BASE / "FourCastNet/nrmse/values"
GC_DIR      = BASE / "gc_pred_files/nrmse/values"
PRITHVI_DIR = BASE / "prithvi_new/nrmse/values"

# ── Model configs ──────────────────────────────────────────────────────────
MODELS = {
    "Aurora": {
        "dir": AURORA_DIR, "lead_col": "lead_day", "time_unit": "day",
        "ctrl": "control",
        "ins":  {"t": "t_ins", "wind": "uv_ins", "geo": "z_ins"},
        "files": {
            "T500": "Aurora_t500_global_nrmse.csv",
            "Z500": "Aurora_z500_global_nrmse.csv",
            "U500": "Aurora_u500_global_nrmse.csv",
            "V500": "Aurora_v500_global_nrmse.csv",
        },
    },
    "FourCastNet": {
        "dir": FCN_DIR, "lead_col": "lead_day", "time_unit": "day",
        "ctrl": "control",
        "ins":  {"t": "t_ins", "wind": "wind_ins", "geo": "z_ins"},
        "files": {
            "T500": "FCN_t500_global_nrmse.csv",
            "Z500": "FCN_z500_global_nrmse.csv",
            "U500": "FCN_u500_global_nrmse.csv",
            "V500": "FCN_v500_global_nrmse.csv",
        },
    },
    "GraphCast-S": {
        "dir": GC_DIR, "lead_col": "lead_day", "time_unit": "day",
        "ctrl": "control",
        "ins":  {"t": "t_ins", "wind": "uv_ins", "geo": "z_ins"},
        "files": {
            "T500": "GC_t500_global_nrmse.csv",
            "Z500": "GC_z500_global_nrmse.csv",
            "U500": "GC_u500_global_nrmse.csv",
            "V500": "GC_v500_global_nrmse.csv",
        },
    },
    "Prithvi": {
        "dir": PRITHVI_DIR, "lead_col": "lead_hour", "time_unit": "hour",
        "ctrl": "Control",
        "ins":  {"t": "temp_ins", "wind": "wind_ins", "geo": "geo_ins"},
        "files": {
            "T500": "nrmse_T500.csv",
            "Z500": "nrmse_H500.csv",
            "U500": "nrmse_U500.csv",
            "V500": "nrmse_V500.csv",
        },
    },
}

MODEL_NAMES  = list(MODELS.keys())
MODEL_LABELS = ["Aurora", "FourCastNet", "GraphCast-S", "Prithvi"]

# ── Variable colours — colorblind-friendly (Wong 2011 palette) ─────────────
VAR_COLORS = {
    "T500": "#D55E00",   # vermilion
    "Z500": "#0072B2",   # blue
    "U500": "#009E73",   # green
    "V500": "#CC79A7",   # pink/purple
}
VAR_LABELS = {"T500": "T500", "Z500": "Z500", "U500": "U500", "V500": "V500"}

# ── Experiments (self-variable excluded) ───────────────────────────────────
EXPERIMENTS = [
    {
        "label": "Temperature Insertion",
        "key":   "t",
        "vars":  ["Z500", "U500", "V500"],   # T500 excluded (self)
        "fname": "Figure_5_11_T_insertion_NRMSE_improvement.png",
    },
    {
        "label": "Wind Insertion",
        "key":   "wind",
        "vars":  ["T500", "Z500"],            # U500 & V500 excluded (self)
        "fname": "Figure_5_11_Wind_insertion_NRMSE_improvement.png",
    },
    {
        "label": "Geopotential Insertion",
        "key":   "geo",
        "vars":  ["T500", "U500", "V500"],   # Z500 excluded (self)
        "fname": "Figure_5_11_Geo_insertion_NRMSE_improvement.png",
    },
]

YLIM = (-15.0, 105.0)   # fixed for ALL plots

# ── Helpers ────────────────────────────────────────────────────────────────
def mean_pct_improvement(df, ctrl_col, ins_col, lead_col, time_unit,
                          lo_day, hi_day):
    """Mean % NRMSE improvement = mean((ctrl−ins)/|ctrl|×100) over phase."""
    ld   = df[lead_col] / 24.0 if time_unit == "hour" else df[lead_col]
    mask = (ld >= lo_day) & (ld <= hi_day)
    sub  = df[mask].dropna(subset=[ctrl_col, ins_col])
    if sub.empty:
        return np.nan
    ctrl = sub[ctrl_col].values.astype(float)
    ins  = sub[ins_col].values.astype(float)
    nz   = ctrl != 0
    if nz.sum() == 0:
        return np.nan
    return float(np.nanmean((ctrl[nz] - ins[nz]) / np.abs(ctrl[nz]) * 100.0))


def compute_table(exp_key, plot_vars):
    """Returns table[model][var] = (adj_pct, mat_pct)."""
    table = {}
    for mname, cfg in MODELS.items():
        table[mname] = {}
        ins_col = cfg["ins"][exp_key]
        for var in plot_vars:
            if var not in cfg.get("files", {}):
                table[mname][var] = (np.nan, np.nan)
                continue
            df  = pd.read_csv(cfg["dir"] / cfg["files"][var]).dropna(how="all")
            adj = mean_pct_improvement(df, cfg["ctrl"], ins_col,
                                       cfg["lead_col"], cfg["time_unit"], 0, 30)
            mat = mean_pct_improvement(df, cfg["ctrl"], ins_col,
                                       cfg["lead_col"], cfg["time_unit"], 30, 60)
            table[mname][var] = (adj, mat)
    return table

# ── Plot function ──────────────────────────────────────────────────────────
def make_plot(exp):
    exp_label, exp_key, plot_vars, fname = (
        exp["label"], exp["key"], exp["vars"], exp["fname"]
    )
    table = compute_table(exp_key, plot_vars)

    print(f"\n{'='*60}")
    print(f"  {exp_label}")
    for var in plot_vars:
        print(f"  {var}:")
        for m in MODEL_NAMES:
            a, b = table[m][var]
            print(f"    {m:14s}  Adj={a:7.2f}%  Mat={b:7.2f}%")

    n_models  = len(MODEL_NAMES)
    n_vars    = len(plot_vars)
    bar_w     = 0.26
    grp_sep   = 0.55                         # gap between model groups
    grp_w     = n_vars * bar_w + grp_sep
    x_centers = np.arange(n_models) * grp_w

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.6), sharey=True,
        gridspec_kw={"wspace": 0.06}
    )

    phase_defs = [
        ("(a)   Adjustment Phase  (Lead Days 0–30)", 0),
        ("(b)   Mature Phase  (Lead Days 30–60)",    1),
    ]

    for ax, (phase_title, pidx) in zip(axes, phase_defs):

        # ── bars ──────────────────────────────────────────────────────────
        for mi, mname in enumerate(MODEL_NAMES):
            xc = x_centers[mi]
            var_offsets = (np.arange(n_vars) - (n_vars - 1) / 2.0) * bar_w

            for vi, var in enumerate(plot_vars):
                val   = table[mname][var][pidx]
                color = VAR_COLORS[var]
                xp    = xc + var_offsets[vi]

                if np.isnan(val):
                    ax.bar(xp, 1.2, width=bar_w * 0.85,
                           color="#dddddd", edgecolor="#aaaaaa",
                           linewidth=0.7, zorder=3)
                    ax.text(xp, 2.5, "N/A", ha="center", va="bottom",
                            fontsize=7.5, color="#888888", style="italic")
                    continue

                display = float(np.clip(val, YLIM[0], YLIM[1]))
                ax.bar(xp, display, width=bar_w * 0.85,
                       color=color, edgecolor="white",
                       linewidth=0.5, zorder=3, alpha=0.90)

                # value label
                clipped = (val < YLIM[0]) or (val > YLIM[1])
                lbl     = f"{val:.0f}%*" if clipped else f"{val:.1f}%"
                lbl_col = "#CC0000" if clipped else "#222222"
                if display >= 0:
                    ly, va = min(display + 1.8, YLIM[1] - 4), "bottom"
                else:
                    ly, va = max(display - 1.8, YLIM[0] + 2), "top"
                ax.text(xp, ly, lbl, ha="center", va=va,
                        fontsize=7.5, rotation=90, color=lbl_col,
                        fontweight="bold" if clipped else "normal")

        # ── zero line ─────────────────────────────────────────────────────
        ax.axhline(0, color="#333333", linewidth=1.1,
                   linestyle="--", alpha=0.65, zorder=2)

        # ── axes formatting ───────────────────────────────────────────────
        ax.set_xlim(x_centers[0] - grp_w * 0.55,
                    x_centers[-1] + grp_w * 0.55)
        ax.set_ylim(YLIM)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(MODEL_LABELS, fontsize=11, fontweight="bold")
        ax.set_title(phase_title, fontsize=11.5, fontweight="bold",
                     pad=9, loc="left", color="#222222")
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(
            matplotlib.ticker.MultipleLocator(10))
        ax.tick_params(axis="y", which="minor", length=2, color="#aaaaaa")

        # light vertical separators between model groups
        for mi in range(n_models - 1):
            sep_x = (x_centers[mi] + x_centers[mi + 1]) / 2
            ax.axvline(sep_x, color="#cccccc", linewidth=0.8,
                       linestyle="-", zorder=1)

    # ── shared y-label (left axis only) ───────────────────────────────────
    axes[0].set_ylabel(
        "Mean NRMSE Improvement  (%)\n"
        r"$\overline{(\mathrm{NRMSE}_{\mathrm{ctrl}} - "
        r"\mathrm{NRMSE}_{\mathrm{ins}}) / "
        r"|\mathrm{NRMSE}_{\mathrm{ctrl}}|} \times 100$",
        fontsize=10.5, labelpad=8
    )

    # ── legend ────────────────────────────────────────────────────────────
    var_handles = [
        mpatches.Patch(facecolor=VAR_COLORS[v], edgecolor="white",
                       alpha=0.90, label=VAR_LABELS[v])
        for v in plot_vars
    ]
    na_handle = mpatches.Patch(facecolor="#dddddd", edgecolor="#aaaaaa",
                               label="N/A (not computed)")
    fig.legend(
        handles=var_handles + [na_handle],
        loc="lower center", ncol=n_vars + 1,
        fontsize=10.5, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.07),
        handlelength=1.4, handleheight=1.0
    )

    # ── suptitle ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"Cross-Variable NRMSE Improvement — {exp_label}\n"
        "All Four Models  |  IC: 2022-01-01  |  60-day Forecast",
        fontsize=13, fontweight="bold", y=1.03, color="#111111"
    )

    # ── footnote for clipped bars ──────────────────────────────────────────
    fig.text(
        0.99, -0.04,
        "* value outside plot range; actual value annotated in red",
        ha="right", va="bottom", fontsize=8, color="#CC0000", style="italic"
    )

    out_path = OUT / fname
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  → Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────
for exp in EXPERIMENTS:
    make_plot(exp)

print("\nDone — all 3 plots written to", OUT)
