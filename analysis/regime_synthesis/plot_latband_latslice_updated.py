#!/usr/bin/env python3
"""
Regenerated latitude-band and latitude-slice % RMSE improvement bar charts
for all 4 models, with U500 and V500 shown independently.
Fixed scale −20 to 105 %, 300 dpi, research-paper quality.

Outputs (saved to /common/):
  Figure_latband_{model}_summary.png  (×4)
  Figure_latslice_{model}_summary.png (×4)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker
from pathlib import Path

# ── Publication rcParams ───────────────────────────────────────────────────
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

BASE = Path("/umbc/rs/pi_halem/users/yashd1")
OUT  = BASE / "common" / "lat_bands"
OUT.mkdir(parents=True, exist_ok=True)
YLIM = (-20.0, 105.0)

# ── Insertion run colours (consistent across all 4 models) ─────────────────
# T-insertion → red, Wind/UV-insertion → blue, Geo/Z-insertion → orange
TINS_RUNS  = {"t_ins", "temp_ins"}
WINS_RUNS  = {"uv_ins", "wind_ins"}
GINS_RUNS  = {"z_ins",  "geo_ins"}
def run_color(run):
    if run in TINS_RUNS: return "#e74c3c"
    if run in WINS_RUNS: return "#3498db"
    if run in GINS_RUNS: return "#f39c12"
    return "#888888"

# ── Model configurations ───────────────────────────────────────────────────
MODELS = {
    "Aurora": {
        "band_dir":      BASE / "Aurora_Pred_Files/lats/values",
        "slice_dir":     BASE / "Aurora_Pred_Files/lat_bands/values",
        "ctrl":          "control",
        "ins_runs":      ["t_ins", "uv_ins", "z_ins"],
        "run_labels":    {"t_ins": "T-Ins", "uv_ins": "UV-Ins", "z_ins": "Z-Ins"},
        "lead_col":      "lead_day",
        "mature_thresh": 30.0,
        "vars": ["t500", "u500", "v500", "z500", "wind10"],
        "var_labels": {"t500":"T500","u500":"U500","v500":"V500","z500":"Z500","wind10":"Wind10m"},
        "var_units":  {"t500":"K","u500":"m s⁻¹","v500":"m s⁻¹","z500":"m² s⁻²","wind10":"m s⁻¹"},
        "band_cols":  ["Tropics", "Midlatitudes", "Polar"],
        "slice_cols": ["2°N", "14°N", "40°N", "60°N"],
        "band_fn":  lambda run, var: f"Aurora_{run}_{var}_latband_rmse.csv",
        "slice_fn": lambda run, var: f"Aurora_{run}_{var}_latslice_rmse.csv",
    },
    "FourCastNet": {
        "band_dir":      BASE / "FourCastNet/lats/values",
        "slice_dir":     BASE / "FourCastNet/lat_bands/values",
        "ctrl":          "control",
        "ins_runs":      ["t_ins", "wind_ins", "z_ins"],
        "run_labels":    {"t_ins": "T-Ins", "wind_ins": "Wind-Ins", "z_ins": "Z-Ins"},
        "lead_col":      "lead_day",
        "mature_thresh": 30.0,
        "vars": ["t500", "u500", "v500", "z500", "wind10"],
        "var_labels": {"t500":"T500","u500":"U500","v500":"V500","z500":"Z500","wind10":"Wind10m"},
        "var_units":  {"t500":"K","u500":"m s⁻¹","v500":"m s⁻¹","z500":"m² s⁻²","wind10":"m s⁻¹"},
        "band_cols":  ["Tropics", "Midlatitudes", "Polar"],
        "slice_cols": ["2°N", "14°N", "40°N", "60°N"],
        "band_fn":  lambda run, var: f"FCN_{run}_{var}_latband_rmse.csv",
        "slice_fn": lambda run, var: f"FCN_{run}_{var}_latslice_rmse.csv",
    },
    "GraphCast-S": {
        "band_dir":      BASE / "gc_pred_files/lats/values",
        "slice_dir":     BASE / "gc_pred_files/lat_bands/values",
        "ctrl":          "control",
        "ins_runs":      ["t_ins", "uv_ins", "z_ins"],
        "run_labels":    {"t_ins": "T-Ins", "uv_ins": "UV-Ins", "z_ins": "Z-Ins"},
        "lead_col":      "lead_day",
        "mature_thresh": 30.0,
        "vars": ["t500", "u500", "v500", "z500", "wind10"],
        "var_labels": {"t500":"T500","u500":"U500","v500":"V500","z500":"Z500","wind10":"Wind10m"},
        "var_units":  {"t500":"K","u500":"m s⁻¹","v500":"m s⁻¹","z500":"m² s⁻²","wind10":"m s⁻¹"},
        "band_cols":  ["Tropics", "Midlatitudes", "Polar"],
        "slice_cols": ["2°N", "14°N", "40°N", "60°N"],
        "band_fn":  lambda run, var: f"GC_{run}_{var}_latband_rmse.csv",
        "slice_fn": lambda run, var: f"GC_{run}_{var}_latslice_rmse.csv",
    },
    "Prithvi": {
        "band_dir":      BASE / "prithvi_new/lats/values",
        "slice_dir":     BASE / "prithvi_new/lat_bands/values",
        "ctrl":          "Control",
        "ins_runs":      ["temp_ins", "wind_ins", "geo_ins"],
        "run_labels":    {"temp_ins": "T-Ins", "wind_ins": "Wind-Ins", "geo_ins": "Geo-Ins"},
        "lead_col":      "lead_hour",
        "mature_thresh": 720.0,   # 30 days in hours
        "vars": ["T_lev51", "H_lev51", "U_lev51", "V_lev51", "T2M", "SLP"],
        "var_labels": {"T_lev51":"T@532hPa","H_lev51":"H@532hPa",
                       "U_lev51":"U@532hPa","V_lev51":"V@532hPa",
                       "T2M":"T2M","SLP":"SLP"},
        "var_units":  {"T_lev51":"K","H_lev51":"m","U_lev51":"m s⁻¹","V_lev51":"m s⁻¹",
                       "T2M":"K","SLP":"hPa"},
        "band_cols":  ["Tropics", "Midlatitudes", "Polar"],
        "slice_cols": ["2N", "14N", "40N", "60N"],
        "band_fn":  lambda run, var: f"latband_rmse_{var}_{run}.csv",
        "slice_fn": lambda run, var: f"latslice_rmse_{var}_{run}.csv",
    },
}

# ── Data reader ────────────────────────────────────────────────────────────
def load_pct_improvement(cfg, vdir, file_fn, region_cols):
    """
    Returns data[var][ins_run][phase][region] = mean % NRMSE improvement.
    phase in ('adj', 'mat').
    """
    ctrl      = cfg["ctrl"]
    lead_col  = cfg["lead_col"]
    thresh    = cfg["mature_thresh"]
    data = {}

    for var in cfg["vars"]:
        ctrl_path = vdir / file_fn(ctrl, var)
        if not ctrl_path.exists():
            print(f"  MISSING ctrl: {ctrl_path.name}"); continue
        df_ctrl = pd.read_csv(ctrl_path)

        data[var] = {}
        for ins_run in cfg["ins_runs"]:
            ins_path = vdir / file_fn(ins_run, var)
            if not ins_path.exists():
                print(f"  MISSING ins:  {ins_path.name}"); continue
            df_ins = pd.read_csv(ins_path)

            data[var][ins_run] = {}
            for phase_key, is_mature in [("adj", False), ("mat", True)]:
                mask_c = (df_ctrl[lead_col] >= thresh) if is_mature else (df_ctrl[lead_col] < thresh)
                mask_i = (df_ins[lead_col]  >= thresh) if is_mature else (df_ins[lead_col]  < thresh)
                data[var][ins_run][phase_key] = {}
                for reg in region_cols:
                    if reg not in df_ctrl.columns or reg not in df_ins.columns:
                        data[var][ins_run][phase_key][reg] = np.nan; continue
                    cm = df_ctrl.loc[mask_c, reg].mean()
                    im = df_ins.loc[mask_i, reg].mean()
                    pct = (100.0 * (cm - im) / abs(cm)
                           if (not np.isnan(cm) and cm != 0) else np.nan)
                    data[var][ins_run][phase_key][reg] = pct
    return data

# ── Plot function ──────────────────────────────────────────────────────────
PHASE_INFO = [
    ("adj", "(a)  Adjustment Phase  (Lead Days 0–30)"),
    ("mat", "(b)  Mature Phase  (Lead Days 30–60)"),
]

def make_bar_figure(model_name, cfg, data, region_cols, region_tick_labels,
                    analysis_label, fname):
    vars_list = [v for v in cfg["vars"] if v in data and data[v]]
    n_vars    = len(vars_list)
    if n_vars == 0:
        print(f"  No data for {model_name} {analysis_label}"); return

    ins_runs  = cfg["ins_runs"]
    n_runs    = len(ins_runs)
    n_regs    = len(region_cols)

    bar_w   = 0.22 if n_runs == 3 else 0.20
    x       = np.arange(n_regs)
    offsets = (np.arange(n_runs) - (n_runs - 1) / 2.0) * bar_w

    col_w   = max(3.2, 10.5 / n_vars)
    fig, axes = plt.subplots(2, n_vars,
                             figsize=(col_w * n_vars, 9.0),
                             gridspec_kw={"hspace": 0.60, "wspace": 0.38})
    if n_vars == 1:
        axes = [[axes[0]], [axes[1]]]

    for row, (phase_key, phase_title) in enumerate(PHASE_INFO):
        for col, var in enumerate(vars_list):
            ax   = axes[row][col]
            lbl  = cfg["var_labels"][var]
            unit = cfg["var_units"][var]

            for ri, ins_run in enumerate(ins_runs):
                if ins_run not in data[var]: continue
                vals = [data[var][ins_run][phase_key].get(reg, np.nan)
                        for reg in region_cols]
                color = run_color(ins_run)
                rlabel = cfg["run_labels"][ins_run]

                bars = ax.bar(x + offsets[ri], vals, bar_w * 0.88,
                              color=color, edgecolor="white",
                              linewidth=0.5, alpha=0.88,
                              label=rlabel, zorder=3)

                for bar_obj, v in zip(bars, vals):
                    if np.isnan(v): continue
                    display  = float(np.clip(v, YLIM[0], YLIM[1]))
                    clipped  = (v < YLIM[0]) or (v > YLIM[1])
                    lbl_txt  = f"{v:.0f}%*" if clipped else f"{v:.1f}%"
                    lbl_col  = "#CC0000" if clipped else "#333333"
                    if display >= 0:
                        ypos, va = min(display + 1.5, YLIM[1] - 4), "bottom"
                    else:
                        ypos, va = max(display - 1.5, YLIM[0] + 2), "top"
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            ypos, lbl_txt,
                            ha="center", va=va, fontsize=6.5,
                            rotation=90, color=lbl_col,
                            fontweight="bold" if clipped else "normal")

            ax.axhline(0, color="#333333", linewidth=0.9,
                       linestyle="--", alpha=0.60, zorder=2)
            ax.set_ylim(YLIM)
            ax.set_xticks(x)
            ax.set_xticklabels(region_tick_labels, fontsize=8.5, fontweight="bold")
            ax.set_title(f"{lbl}  [{unit}]", fontsize=10, fontweight="bold", pad=5)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
            ax.tick_params(axis="y", which="minor", length=2, color="#bbbbbb")

            if col == 0:
                ax.set_ylabel(
                    f"{phase_title}\n% RMSE Reduction vs Control",
                    fontsize=8.5)

            # legend only once, top-right of last column
            if row == 0 and col == n_vars - 1:
                handles = [plt.Rectangle((0, 0), 1, 1,
                                         facecolor=run_color(r),
                                         edgecolor="white", alpha=0.88)
                           for r in ins_runs]
                labels  = [cfg["run_labels"][r] for r in ins_runs]
                ax.legend(handles, labels, fontsize=8.5,
                          loc="upper right", framealpha=0.92,
                          edgecolor="#cccccc")

    # vertical separators between variable groups
    for ax_row in axes:
        for ax in ax_row:
            for xi in range(n_regs - 1):
                sep = (xi + 0.5)
                ax.axvline(sep, color="#dddddd", linewidth=0.7,
                           linestyle="-", zorder=1)

    fig.suptitle(
        f"{model_name}  ·  Latitude-{analysis_label} % RMSE Improvement\n"
        "IC: Jan 2022 / Feb 2023  |  60-day Forecast  |  "
        "Cosine-latitude weighted  |  Positive = improvement vs Control",
        fontsize=11.5, fontweight="bold", y=1.025, color="#111111",
    )

    fig.text(0.99, -0.015,
             "* value outside plot range; actual value annotated in red",
             ha="right", va="bottom", fontsize=7.5,
             color="#CC0000", style="italic")

    out = OUT / fname
    fig.savefig(str(out), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  → Saved: {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────
for mname, cfg in MODELS.items():
    print(f"\n{'='*60}\n  {mname}")

    # latband
    print("  Loading latband data …")
    band_data = load_pct_improvement(
        cfg, cfg["band_dir"], cfg["band_fn"], cfg["band_cols"])
    make_bar_figure(
        mname, cfg, band_data,
        region_cols=cfg["band_cols"],
        region_tick_labels=cfg["band_cols"],
        analysis_label="Band",
        fname=f"Figure_latband_{mname.replace(' ','_').replace('-','_')}_summary.png",
    )

    # latslice
    print("  Loading latslice data …")
    slice_data = load_pct_improvement(
        cfg, cfg["slice_dir"], cfg["slice_fn"], cfg["slice_cols"])
    make_bar_figure(
        mname, cfg, slice_data,
        region_cols=cfg["slice_cols"],
        region_tick_labels=[s.replace("°","°") for s in cfg["slice_cols"]],
        analysis_label="Slice",
        fname=f"Figure_latslice_{mname.replace(' ','_').replace('-','_')}_summary.png",
    )

print(f"\nDone — 8 plots written to {OUT}")
