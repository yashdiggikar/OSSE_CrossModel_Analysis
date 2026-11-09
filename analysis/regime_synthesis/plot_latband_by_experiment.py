#!/usr/bin/env python3
"""
Experiment-centric latitude-band / latitude-slice % RMSE improvement charts.

Design (per user spec):
  - One figure PER INSERTION EXPERIMENT (T-Ins, Wind-Ins, Geo-Ins), not per model.
  - Each figure shows only the CROSS-VARIABLE targets, i.e. the variables that
    were NOT directly inserted (T-Ins excludes T500, Wind-Ins excludes U500/V500,
    Geo-Ins excludes Z500).
  - Each variable group has 4 bars, one per model (Aurora, FourCastNet,
    GraphCast-S, Prithvi), colored consistently across all figures.
  - Subplot grid: rows = phase (Adjustment / Mature), columns = region
    (Tropics / Midlatitudes / Polar for latband; 2N/14N/40N/60N for latslice).

Outputs (saved to /common/lats_FINAL/):
  Figure_latband_{Experiment}_crossvar.png   (x3)
  Figure_latslice_{Experiment}_crossvar.png  (x3)

Uses the corrected Prithvi lat-band/lat-slice CSVs (regenerated after fixing
the contaminated "Control" file in section_5_7_prithvi_latband_analysis.py /
section_5_7_prithvi_latslice_analysis.py).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker
from pathlib import Path

# ── Publication rcParams (research-grade) ──────────────────────────────────
rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    10.5,
    "axes.labelsize":    9.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.linewidth":    0.9,
    "axes.edgecolor":    "#333333",
    "xtick.bottom":      True,
    "ytick.left":        True,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.25,
    "grid.linestyle":    ":",
    "grid.linewidth":    0.6,
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
})

BASE = Path("/umbc/rs/pi_halem/users/yashd1")
OUT  = BASE / "common" / "lats_FINAL"
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_YLIM = (-20.0, 105.0)

# ── Model colors (consistent identity across every figure) ─────────────────
MODEL_ORDER  = ["Aurora", "FourCastNet", "GraphCast-S", "Prithvi"]
MODEL_COLORS = {
    "Aurora":      "#2E86AB",   # teal-blue
    "FourCastNet": "#E94B3C",   # coral-red
    "GraphCast-S": "#6A4C93",   # violet
    "Prithvi":     "#1B998B",   # emerald
}

# ── Experiment accent colors (kept consistent with original T/Wind/Geo scheme)
EXP_ACCENT = {"T-Ins": "#e74c3c", "Wind-Ins": "#3498db", "Geo-Ins": "#f39c12"}
EXP_TITLE  = {
    "T-Ins":    "Temperature Insertion",
    "Wind-Ins": "Wind (U+V) Insertion",
    "Geo-Ins":  "Geopotential (Z) Insertion",
}
# Cross-variable targets shown per experiment (inserted variable excluded)
EXP_SHOW_VARS = {
    "T-Ins":    ["U500", "V500", "Z500"],
    "Wind-Ins": ["T500", "Z500"],
    "Geo-Ins":  ["T500", "U500", "V500"],
}

# ── Per-model config: dirs, control key, run-name per experiment, var-key map,
#    lead column, mature-phase threshold, and the actual region column names
#    used inside that model's own CSVs (labels are unified at plot time). ────
MODELS = {
    "Aurora": {
        "band_dir":  BASE / "Aurora_Pred_Files/lats/values",
        "slice_dir": BASE / "Aurora_Pred_Files/lat_bands/values",
        "ctrl": "control",
        "exp_run":   {"T-Ins": "t_ins", "Wind-Ins": "uv_ins", "Geo-Ins": "z_ins"},
        "var_key":   {"T500": "t500", "U500": "u500", "V500": "v500", "Z500": "z500"},
        "lead_col": "lead_day", "mature_thresh": 30.0,
        "band_cols":  ["Tropics", "Midlatitudes", "Polar"],
        "slice_cols": ["2°N", "14°N", "40°N", "60°N"],
        "band_fn":  lambda run, var: f"Aurora_{run}_{var}_latband_rmse.csv",
        "slice_fn": lambda run, var: f"Aurora_{run}_{var}_latslice_rmse.csv",
    },
    "FourCastNet": {
        "band_dir":  BASE / "FourCastNet/lats/values",
        "slice_dir": BASE / "FourCastNet/lat_bands/values",
        "ctrl": "control",
        "exp_run":   {"T-Ins": "t_ins", "Wind-Ins": "wind_ins", "Geo-Ins": "z_ins"},
        "var_key":   {"T500": "t500", "U500": "u500", "V500": "v500", "Z500": "z500"},
        "lead_col": "lead_day", "mature_thresh": 30.0,
        "band_cols":  ["Tropics", "Midlatitudes", "Polar"],
        "slice_cols": ["2°N", "14°N", "40°N", "60°N"],
        "band_fn":  lambda run, var: f"FCN_{run}_{var}_latband_rmse.csv",
        "slice_fn": lambda run, var: f"FCN_{run}_{var}_latslice_rmse.csv",
    },
    "GraphCast-S": {
        "band_dir":  BASE / "gc_pred_files/lats/values",
        "slice_dir": BASE / "gc_pred_files/lat_bands/values",
        "ctrl": "control",
        "exp_run":   {"T-Ins": "t_ins", "Wind-Ins": "uv_ins", "Geo-Ins": "z_ins"},
        "var_key":   {"T500": "t500", "U500": "u500", "V500": "v500", "Z500": "z500"},
        "lead_col": "lead_day", "mature_thresh": 30.0,
        "band_cols":  ["Tropics", "Midlatitudes", "Polar"],
        "slice_cols": ["2°N", "14°N", "40°N", "60°N"],
        "band_fn":  lambda run, var: f"GC_{run}_{var}_latband_rmse.csv",
        "slice_fn": lambda run, var: f"GC_{run}_{var}_latslice_rmse.csv",
    },
    "Prithvi": {
        "band_dir":  BASE / "prithvi_new/lats/values",
        "slice_dir": BASE / "prithvi_new/lat_bands/values",
        "ctrl": "Control",
        "exp_run":   {"T-Ins": "temp_ins", "Wind-Ins": "wind_ins", "Geo-Ins": "geo_ins"},
        "var_key":   {"T500": "T_lev51", "U500": "U_lev51", "V500": "V_lev51", "Z500": "H_lev51"},
        "lead_col": "lead_hour", "mature_thresh": 720.0,  # 30 days in hours
        "band_cols":  ["Tropics", "Midlatitudes", "Polar"],
        "slice_cols": ["2N", "14N", "40N", "60N"],
        "band_fn":  lambda run, var: f"latband_rmse_{var}_{run}.csv",
        "slice_fn": lambda run, var: f"latslice_rmse_{var}_{run}.csv",
    },
}

REGION_TICKS = {
    "band":  ["0°–30°", "30°–60°", "60°–90°"],
    "slice": ["2°N", "14°N", "40°N", "60°N"],
}

PHASE_INFO = [
    ("adj", "Adjustment Phase (Days 0–30)"),
    ("mat", "Mature Phase (Days 30–60)"),
]


def pct_improvement(df_ctrl, df_ins, lead_col, thresh, region_col, is_mature):
    mask_c = (df_ctrl[lead_col] >= thresh) if is_mature else (df_ctrl[lead_col] < thresh)
    mask_i = (df_ins[lead_col]  >= thresh) if is_mature else (df_ins[lead_col]  < thresh)
    if region_col not in df_ctrl.columns or region_col not in df_ins.columns:
        return np.nan
    cm = df_ctrl.loc[mask_c, region_col].mean()
    im = df_ins.loc[mask_i, region_col].mean()
    if np.isnan(cm) or cm == 0:
        return np.nan
    return 100.0 * (cm - im) / abs(cm)


def load_experiment_data(exp_key, analysis):
    """data[var][model][phase][region_idx] = pct improvement (region_idx aligned
    positionally across models, since column header formatting differs)."""
    dir_key = "band_dir" if analysis == "band" else "slice_dir"
    fn_key  = "band_fn"  if analysis == "band" else "slice_fn"
    col_key = "band_cols" if analysis == "band" else "slice_cols"
    show_vars = EXP_SHOW_VARS[exp_key]
    n_regions = len(REGION_TICKS[analysis])

    data = {var: {} for var in show_vars}
    for model in MODEL_ORDER:
        cfg = MODELS[model]
        run = cfg["exp_run"][exp_key]
        vdir = cfg[dir_key]
        file_fn = cfg[fn_key]
        regions = cfg[col_key]

        for var in show_vars:
            vkey = cfg["var_key"][var]
            ctrl_path = vdir / file_fn(cfg["ctrl"], vkey)
            ins_path  = vdir / file_fn(run, vkey)
            if not ctrl_path.exists() or not ins_path.exists():
                print(f"  MISSING: {model} {exp_key} {var} "
                      f"({ctrl_path.name if not ctrl_path.exists() else ins_path.name})")
                continue
            df_ctrl = pd.read_csv(ctrl_path)
            df_ins  = pd.read_csv(ins_path)

            data[var][model] = {}
            for phase_key, is_mature in [("adj", False), ("mat", True)]:
                vals = []
                for ri in range(n_regions):
                    region_col = regions[ri]
                    vals.append(pct_improvement(
                        df_ctrl, df_ins, cfg["lead_col"], cfg["mature_thresh"],
                        region_col, is_mature))
                data[var][model][phase_key] = vals
    return data


def make_experiment_figure(exp_key, data, analysis, fname, phases=None):
    show_vars = EXP_SHOW_VARS[exp_key]
    region_ticks = REGION_TICKS[analysis]
    n_regions = len(region_ticks)
    n_models  = len(MODEL_ORDER)

    bar_w   = 0.19
    x       = np.arange(n_regions)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_w

    n_vars = len(show_vars)
    phases = PHASE_INFO if phases is None else phases
    n_rows = len(phases)

    # Widen the lower y-limit only for figures whose data actually exceeds
    # the default -20 floor, so the rest of the figures keep a consistent,
    # tighter range.
    all_vals = [v for var in show_vars for model in MODEL_ORDER
                if model in data[var]
                for phase_key, _ in phases
                for v in data[var][model][phase_key] if not np.isnan(v)]
    min_val = min(all_vals) if all_vals else DEFAULT_YLIM[0]
    if min_val < DEFAULT_YLIM[0]:
        ylim_low = 10.0 * np.floor((min_val - 5.0) / 10.0)
    else:
        ylim_low = DEFAULT_YLIM[0]
    YLIM = (ylim_low, DEFAULT_YLIM[1])

    fig, axes = plt.subplots(
        n_rows, n_vars,
        figsize=(max(5.2, 6.0 * n_vars / 1.4), 5.4 * n_rows + 0.6),
        gridspec_kw={"hspace": 0.65, "wspace": 0.4},
        squeeze=False,
    )

    accent = EXP_ACCENT[exp_key]

    for row, (phase_key, phase_title) in enumerate(phases):
        for col, var in enumerate(show_vars):
            ax = axes[row][col]

            for mi, model in enumerate(MODEL_ORDER):
                if model not in data[var]:
                    continue
                vals = data[var][model][phase_key]
                color = MODEL_COLORS[model]
                bars = ax.bar(x + offsets[mi], vals, bar_w,
                               color=color, edgecolor="white",
                               linewidth=0.6, alpha=0.92,
                               label=model, zorder=3)
                for bar_obj, v in zip(bars, vals):
                    if np.isnan(v):
                        continue
                    display = float(np.clip(v, YLIM[0], YLIM[1]))
                    clipped = (v < YLIM[0]) or (v > YLIM[1])
                    lbl_txt = f"{v:.0f}%*" if clipped else f"{v:.0f}%"
                    lbl_col = "#CC0000" if clipped else "#2b2b2b"
                    if display >= 0:
                        ypos, va = min(display + 2.5, YLIM[1] - 9), "bottom"
                    else:
                        # Anchor below the bar tip and grow DOWNWARD
                        # (va="top"), away from the bar, with a safe floor
                        # margin so the rotated label never spills past the
                        # axis frame into the x-tick-label area below.
                        ypos = max(display - 2.5, YLIM[0] + 11)
                        va = "top"
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, ypos,
                             lbl_txt, ha="center", va=va, fontsize=6.2,
                             rotation=90, color=lbl_col,
                             fontweight="bold" if clipped else "normal")

            ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--",
                       alpha=0.6, zorder=2)
            ax.set_ylim(YLIM)
            ax.set_xticks(x)
            ax.set_xticklabels(region_ticks, fontsize=8.5, fontweight="bold")
            ax.tick_params(axis="x", pad=10)
            ax.set_title(var, fontsize=11, fontweight="bold", pad=6,
                         color=accent)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
            ax.tick_params(axis="y", which="minor", length=2, color="#bbbbbb")
            for spine in ax.spines.values():
                spine.set_edgecolor("#888888")

            if col == 0:
                ax.set_ylabel(f"{phase_title}\n% RMSE Reduction vs Control",
                              fontsize=8.5)

            for xi in range(n_regions - 1):
                ax.axvline(xi + 0.5, color="#e3e3e3", linewidth=0.7, zorder=1)

    analysis_label = "Latitude-Band" if analysis == "band" else "Latitude-Slice"
    fig.suptitle(
        f"{EXP_TITLE[exp_key]} — Cross-Variable Impact ({analysis_label})\n"
        "IC: Jan 2022 / Feb 2023  |  60-day Forecast  |  4 Models  |  "
        "Cosine-latitude weighted  |  Positive = improvement vs Control",
        fontsize=12, fontweight="bold", y=1.04, color="#111111",
    )

    # single shared legend, centered horizontally below the whole figure
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m],
                                     edgecolor="white", alpha=0.92)
                       for m in MODEL_ORDER]
    fig.legend(legend_handles, MODEL_ORDER, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=n_models, fontsize=9.5,
               framealpha=0.94, edgecolor="#cccccc")

    fig.text(0.99, -0.09,
              "* value outside plot range; actual value annotated in red",
              ha="right", va="bottom", fontsize=7.5, color="#CC0000",
              style="italic")

    out = OUT / fname
    fig.savefig(str(out), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved: {out.name}")


MATURE_ONLY_PHASE = [("mat", "Mature Phase (Days 30–60)")]

value_rows = {"band": [], "slice": []}

for exp_key in ["T-Ins", "Wind-Ins", "Geo-Ins"]:
    print(f"\n{'='*60}\n  {EXP_TITLE[exp_key]}")
    for analysis in ["band", "slice"]:
        print(f"  Loading lat{analysis} data ...")
        data = load_experiment_data(exp_key, analysis)

        fname = f"Figure_lat{analysis}_{exp_key.replace('-', '_')}_crossvar.png"
        make_experiment_figure(exp_key, data, analysis, fname)

        fname_mat = f"Figure_lat{analysis}_{exp_key.replace('-', '_')}_crossvar_mature.png"
        make_experiment_figure(exp_key, data, analysis, fname_mat,
                                phases=MATURE_ONLY_PHASE)

        region_ticks = REGION_TICKS[analysis]
        for var, model_data in data.items():
            for model, phase_data in model_data.items():
                for phase_key, vals in phase_data.items():
                    for ri, v in enumerate(vals):
                        value_rows[analysis].append({
                            "experiment": exp_key, "variable": var, "model": model,
                            "phase": phase_key, "region": region_ticks[ri],
                            "pct_improvement": v,
                        })

for analysis in ["band", "slice"]:
    df_vals = pd.DataFrame(value_rows[analysis])
    out_csv = OUT / f"lat{analysis}_crossvar_pct_improvement.csv"
    df_vals.to_csv(out_csv, index=False)
    print(f"  -> Saved: {out_csv.name} ({len(df_vals)} rows)")

print(f"\nDone — 6 plots + 2 value CSVs written to {OUT}")
