#!/usr/bin/env python3
"""
Shared plotting style/helpers for the cloud-regime (tcc, cloud2) and
terrain-class (terrain, terrain2) cross-variable insertion figures.

Used both by the original (heavy, raw-data) generator scripts and by the
lightweight replot_*.py scripts that rebuild the figures from the cached
per-model intermediate RMSE CSVs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

ADJ_COLOR = "#2E86AB"   # dark teal-blue (matches lats_FINAL model palette)
MAT_COLOR = "#E94B3C"   # dark coral-red (matches lats_FINAL model palette)
CTRL_STYLE = dict(color="#666666", lw=1.6, ls="--", alpha=0.9, zorder=2)
INS_STYLE  = dict(lw=2.1, zorder=3, alpha=0.95)

POSITIVE_BAND = "#EEF3EA"
NEGATIVE_BAND = "#F7E9E8"
AXIS_DARK = "#111111"


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 12, "font.weight": "bold",
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 12, "axes.labelweight": "bold",
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10.5,
        "text.color": AXIS_DARK,
        "axes.edgecolor": AXIS_DARK, "axes.linewidth": 1.4,
        "axes.labelcolor": AXIS_DARK,
        "xtick.color": AXIS_DARK, "ytick.color": AXIS_DARK,
        "xtick.major.width": 1.3, "ytick.major.width": 1.3,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


# ──────────────────────────────────────────────────────────────────────────
# Cache I/O — rebuild the {var: {"ctrl": (N,n_cat), "ins": (N,n_cat)}} dict
# straight from the intermediates/*_rmse.csv files saved by the heavy scripts
# ──────────────────────────────────────────────────────────────────────────
def load_rmse_dict(csv_path, cross_vars, cat_col, cat_short):
    df = pd.read_csv(csv_path)
    out = {}
    lead = None
    for var in cross_vars:
        sub = df[df["variable"] == var]
        if sub.empty:
            continue
        piv_c = sub.pivot(index="lead_day", columns=cat_col, values="ctrl_rmse").reindex(columns=cat_short).sort_index()
        piv_i = sub.pivot(index="lead_day", columns=cat_col, values="ins_rmse").reindex(columns=cat_short).sort_index()
        if lead is None:
            lead = piv_c.index.values
        out[var] = {"ctrl": piv_c.values, "ins": piv_i.values}
    return lead, out


def improvement_pct(ctrl_base, ins_phase_mean, eps=1e-12):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ctrl_base > eps, 100.0 * (ctrl_base - ins_phase_mean) / ctrl_base, np.nan)


def phase_improvements(rmse_dict, cross_vars, inj_idx, adj_slice, mat_slice):
    out = {}
    for var in cross_vars:
        if var not in rmse_dict:
            continue
        d = rmse_dict[var]
        cb = d["ctrl"][inj_idx]
        out[var] = {
            "adj": improvement_pct(cb, np.nanmean(d["ins"][adj_slice], axis=0)),
            "mat": improvement_pct(cb, np.nanmean(d["ins"][mat_slice], axis=0)),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# Bar value labels — grow AWAY from the bar on both sides so a negative bar's
# label never overlaps the bar itself or the x-axis tick labels below it.
# ──────────────────────────────────────────────────────────────────────────
def add_bar_value_labels(ax, bars_vals_list, fontsize=7.5):
    ylo, yhi = ax.get_ylim()
    rng = max(yhi - ylo, 1e-6)
    pad = 0.14 * rng
    ax.set_ylim(ylo - pad, yhi + pad)
    for bars, vals in bars_vals_list:
        for rect, val in zip(bars, vals):
            if np.isnan(val):
                continue
            h = rect.get_height()
            if h >= 0:
                ypos, va = h + 0.02 * rng, "bottom"
            else:
                ypos, va = h - 0.02 * rng, "top"
            ax.text(rect.get_x() + rect.get_width() / 2, ypos, f"{val:+.1f}",
                     ha="center", va=va, fontsize=fontsize, rotation=90,
                     color=AXIS_DARK, fontweight="bold")


def style_bar_axis(ax, n_cat, cat_names, x_label=None):
    ax.axhline(0, color=AXIS_DARK, lw=1.1, ls="--", alpha=0.8)
    ax.set_xticks(np.arange(n_cat))
    ax.set_xticklabels(cat_names, fontsize=9, fontweight="bold", color=AXIS_DARK)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ylo, yhi = ax.get_ylim()
    ax.fill_between([-0.5, n_cat - 0.5], [0, 0], [max(yhi, 1)] * 2, color=POSITIVE_BAND, alpha=0.6, zorder=0)
    ax.fill_between([-0.5, n_cat - 0.5], [min(ylo, -1)] * 2, [0, 0], color=NEGATIVE_BAND, alpha=0.6, zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(AXIS_DARK)
        ax.spines[sp].set_linewidth(1.4)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10, fontweight="bold")


def compute_uniform_top(improvements, model_order, cross_vars, pad_frac=0.12, round_to=10):
    """Largest positive bar value across every panel of a figure, with
    headroom added and rounded up to a nice number, for use as a single
    shared positive y-limit across all panels in that figure."""
    vals = []
    for model in model_order:
        if model not in improvements:
            continue
        for var in cross_vars:
            if var not in improvements[model]:
                continue
            vals.append(np.nanmax(improvements[model][var]["adj"], initial=-np.inf))
            vals.append(np.nanmax(improvements[model][var]["mat"], initial=-np.inf))
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return 50.0
    raw_top = max(vals) * (1.0 + pad_frac)
    return float(np.ceil(raw_top / round_to) * round_to)


def build_bar_figure(suptitle, model_order, model_labels, cross_vars, var_titles,
                      improvements, cat_names, n_cat, x_label, figsize, top_ylim=None):
    x = np.arange(n_cat); bar_w = 0.35
    n_rows, n_cols = len(model_order), len(cross_vars)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=False, squeeze=False)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)

    for row, model in enumerate(model_order):
        for col, var in enumerate(cross_vars):
            ax = axes[row][col]
            if model not in improvements or var not in improvements[model]:
                ax.set_visible(False); continue
            ai = improvements[model][var]["adj"]; mi = improvements[model][var]["mat"]
            b1 = ax.bar(x - bar_w / 2, ai, bar_w, color=ADJ_COLOR, alpha=1.0, edgecolor="white", lw=0.6)
            b2 = ax.bar(x + bar_w / 2, mi, bar_w, color=MAT_COLOR, alpha=1.0, edgecolor="white", lw=0.6, hatch="///")
            if top_ylim is not None:
                ylo, _ = ax.get_ylim()
                ax.set_ylim(ylo, top_ylim)
            add_bar_value_labels(ax, [(b1, ai), (b2, mi)])
            is_last_row = (row == n_rows - 1)
            style_bar_axis(ax, n_cat, cat_names, x_label if is_last_row else None)
            if row == 0:
                ax.set_title(var_titles[var], fontsize=13, fontweight="bold", color="#1a237e", pad=6)
            if col == 0:
                ax.set_ylabel(f"{model_labels[model]}\nRMSE Imp. (%)", fontsize=11, fontweight="bold")

    fig.legend(handles=[
        mpatches.Patch(color=ADJ_COLOR, alpha=1.0, label="Adjustment Phase (Day 30–40)"),
        mpatches.Patch(color=MAT_COLOR, alpha=1.0, hatch="///", label="Mature Phase (Day 40–60)"),
    ], loc="lower center", ncol=2, fontsize=12, framealpha=0.95,
       bbox_to_anchor=(0.5, -0.02), edgecolor=AXIS_DARK)
    fig.tight_layout(rect=[0, 0.035, 1, 0.99])
    return fig


def build_timeseries_figure(suptitle, model_order, model_labels, cross_vars, var_labels,
                             all_rmse, cat_colors, cat_short, figsize, hspace=0.55, wspace=0.28):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.998)

    n_vars = len(cross_vars); n_models = len(model_order)
    outer = gridspec.GridSpec(n_vars, 1, figure=fig, hspace=hspace)

    for vi, var in enumerate(cross_vars):
        inner = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=outer[vi], wspace=wspace)
        for ci, model in enumerate(model_order):
            ax = fig.add_subplot(inner[ci])
            rdict, inj, adj, mat, lead = all_rmse[model]
            if var not in rdict:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            d = rdict[var]
            adj_x = [lead[adj.start], lead[min(adj.stop - 1, len(lead) - 1)]]
            mat_x = [lead[mat.start], lead[min(mat.stop - 1, len(lead) - 1)]]
            ax.axvspan(adj_x[0], adj_x[1], color="#D6E4F0", alpha=0.45, zorder=0)
            ax.axvspan(mat_x[0], mat_x[1], color="#F3D9D6", alpha=0.45, zorder=0)
            ax.axvline(lead[inj], color="#333333", lw=1.2, ls=":", zorder=4)
            n_cat = len(cat_short)
            for c in range(n_cat):
                ax.plot(lead, d["ctrl"][:, c], **CTRL_STYLE)
                ax.plot(lead, d["ins"][:, c], color=cat_colors[c], label=cat_short[c], **INS_STYLE)
            ax.set_xlim(lead[0], lead[-1])
            ax.margins(y=0.08)
            ax.set_xlabel("Lead Day", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.22, ls=":")
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            for sp in ["left", "bottom"]:
                ax.spines[sp].set_color(AXIS_DARK)
                ax.spines[sp].set_linewidth(1.4)
            ax.tick_params(axis="both", labelsize=9, width=1.3, length=4.5, colors=AXIS_DARK)
            ylo, yhi = ax.get_ylim()
            ax.text(lead[inj] + 0.4, yhi * 0.94, "Inject.", fontsize=7.5, color=AXIS_DARK,
                     va="top", fontweight="bold")
            if ci == 0:
                ax.set_ylabel(var_labels[var], fontsize=10, fontweight="bold")
            if vi == 0:
                ax.set_title(model_labels[model], fontsize=12, fontweight="bold", color="#1a237e")

    handles = [mlines.Line2D([], [], color=cat_colors[c], lw=2.4, label=cat_short[c])
               for c in range(len(cat_short))]
    handles += [mlines.Line2D([], [], **CTRL_STYLE, label="Control")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=11,
               framealpha=0.95, edgecolor=AXIS_DARK, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=[0, 0.035, 1, 0.985])
    return fig
