"""
Publication-quality grouped bar chart of NRMSE improvement (mature phase only).
One figure per experiment: T-insertion, UV-insertion, Z-insertion.
Mature phase = day 40–60.  Improvement = (control − experiment) / control × 100 %
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path

# ── rcParams — publication-grade Times New Roman style ────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          12,
    "axes.titlesize":     15,
    "axes.labelsize":     13,
    "xtick.labelsize":    12,
    "ytick.labelsize":    11,
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.size":   2.5,
    "xtick.major.width":  0.9,
    "ytick.major.width":  0.9,
    "grid.linewidth":     0.5,
    "grid.alpha":         0.45,
    "legend.frameon":     True,
    "legend.framealpha":  0.97,
    "legend.edgecolor":   "0.80",
    "legend.fontsize":    10.5,
    "figure.dpi":         150,
    "savefig.dpi":        400,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.12,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
FCN_DIR = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet/nrmse/values")
AUR_DIR = Path("/umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files/nrmse/values")
GC_DIR  = Path("/umbc/rs/pi_halem/users/yashd1/gc_pred_files/nrmse/values")
PRI_DIR = Path("/umbc/rs/pi_halem/users/yashd1/prithvi_new/nrmse/values")
OUT_DIR = Path("/umbc/rs/pi_halem/users/yashd1/common/nrmse_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Models & colours (ColorBrewer publication palette) ────────────────────────
MODELS = ["FCN", "Aurora", "GraphCast", "Prithvi"]
COLORS = ["#2166AC", "#D6604D", "#1A9850", "#762A83"]   # ColorBrewer RdBu/PRGn
BAR_W     = 0.14
MODEL_GAP = 0.05
GROUP_SEP = 0.9

# ── Experiment column names per model ─────────────────────────────────────────
EXP_COLS = {
    "FCN":       {"t": "t_ins",    "uv": "wind_ins", "z": "z_ins"},
    "Aurora":    {"t": "t_ins",    "uv": "uv_ins",   "z": "z_ins"},
    "GraphCast": {"t": "t_ins",    "uv": "uv_ins",   "z": "z_ins"},
    "Prithvi":   {"t": "temp_ins", "uv": "wind_ins", "z": "geo_ins"},
}

# ── Variable → file mapping ────────────────────────────────────────────────────
# (filename, dir, day_col, ctrl_col, hour_to_day)
VAR_FILES = {
    "FCN": {
        "T": ("FCN_t500_global_nrmse.csv",  FCN_DIR, "lead_day", "control", False),
        "U": ("FCN_u500_global_nrmse.csv",  FCN_DIR, "lead_day", "control", False),
        "V": ("FCN_v500_global_nrmse.csv",  FCN_DIR, "lead_day", "control", False),
        "Z": ("FCN_z500_global_nrmse.csv",  FCN_DIR, "lead_day", "control", False),
        "Q": ("FCN_q500_global_nrmse.csv",  FCN_DIR, "lead_day", "control", False),
    },
    "Aurora": {
        "T": ("Aurora_t500_global_nrmse.csv", AUR_DIR, "lead_day", "control", False),
        "U": ("Aurora_u500_global_nrmse.csv", AUR_DIR, "lead_day", "control", False),
        "V": ("Aurora_v500_global_nrmse.csv", AUR_DIR, "lead_day", "control", False),
        "Z": ("Aurora_z500_global_nrmse.csv", AUR_DIR, "lead_day", "control", False),
        "Q": ("Aurora_q500_global_nrmse.csv", AUR_DIR, "lead_day", "control", False),
    },
    "GraphCast": {
        "T": ("GC_t500_global_nrmse.csv",  GC_DIR, "lead_day", "control", False),
        "U": ("GC_u500_global_nrmse.csv",  GC_DIR, "lead_day", "control", False),
        "V": ("GC_v500_global_nrmse.csv",  GC_DIR, "lead_day", "control", False),
        "Z": ("GC_z500_global_nrmse.csv",  GC_DIR, "lead_day", "control", False),
        "Q": ("GC_q500_global_nrmse.csv",  GC_DIR, "lead_day", "control", False),
    },
    "Prithvi": {
        "T": ("nrmse_T500.csv", PRI_DIR, "lead_day", "Control", True),
        "U": ("nrmse_U500.csv", PRI_DIR, "lead_day", "Control", True),
        "V": ("nrmse_V500.csv", PRI_DIR, "lead_day", "Control", True),
        "Z": ("nrmse_H500.csv", PRI_DIR, "lead_day", "Control", True),
        "Q": ("nrmse_TQV.csv",  PRI_DIR, "lead_day", "Control", True),
    },
}

# ── Data helpers ───────────────────────────────────────────────────────────────
def mature_improvement(df, day_col, ctrl_col, exp_col):
    """NRMSE improvement % averaged over mature phase (day 40–60)."""
    sub = df[(df[day_col] > 40) & (df[day_col] <= 60)].dropna(subset=[ctrl_col, exp_col])
    sub = sub[sub[exp_col] != 0.0]
    if len(sub) == 0:
        return np.nan
    c, e = sub[ctrl_col].mean(), sub[exp_col].mean()
    if c == 0 or np.isnan(c) or np.isnan(e):
        return np.nan
    return (c - e) / c * 100.0

def get_mature(model, var_key, exp_key):
    info = VAR_FILES[model].get(var_key)
    if info is None:
        return np.nan
    fname, dirpath, day_col, ctrl_col, hour_to_day = info
    fpath = dirpath / fname
    if not fpath.exists():
        print(f"  MISSING: {fpath}")
        return np.nan
    df = pd.read_csv(fpath)
    if hour_to_day:
        df[day_col] = df["lead_hour"] / 24
    exp_col = EXP_COLS[model][exp_key]
    if exp_col not in df.columns:
        print(f"  MISSING col '{exp_col}' in {fname}")
        return np.nan
    return mature_improvement(df, day_col, ctrl_col, exp_col)

# ── Bar positions ──────────────────────────────────────────────────────────────
def bar_offsets(n=4):
    total  = n * BAR_W + (n - 1) * MODEL_GAP
    starts = [i * (BAR_W + MODEL_GAP) for i in range(n)]
    return [s - total / 2 + BAR_W / 2 for s in starts]

OFFSETS = bar_offsets(4)

# ── Core plotting function ─────────────────────────────────────────────────────
def make_plot(exp_key, exp_label, var_list, var_labels, filename):
    n_vars = len(var_list)
    fig_w  = 3.4 + n_vars * 2.1
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))

    group_x = np.arange(n_vars, dtype=float) * GROUP_SEP

    vals_all = {}
    for mi, (model, color) in enumerate(zip(MODELS, COLORS)):
        for vi, vk in enumerate(var_list):
            h = get_mature(model, vk, exp_key)
            x = group_x[vi] + OFFSETS[mi]
            if np.isnan(h):
                continue
            # Slightly lighter edge drawn from the bar colour itself
            ax.bar(x, h, width=BAR_W, color=color,
                   edgecolor="white", linewidth=0.7,
                   alpha=0.92, zorder=3)
            vals_all[(mi, vi)] = (x, h)

    # Value labels — compact, serif, above/below bar
    for (mi, vi), (x, h) in vals_all.items():
        if abs(h) < 0.4:
            continue
        pad    = 0.30 if h >= 0 else -0.40
        va_dir = "bottom" if h >= 0 else "top"
        ax.text(x, h + pad, f"{h:.1f}",
                ha="center", va=va_dir,
                fontsize=7.0, color="#111111", fontweight="bold",
                clip_on=True)

    ax.axhline(0, color="#222222", linewidth=0.9, zorder=4)

    # Alternating group shading — very subtle
    for vi in range(n_vars):
        if vi % 2 == 0:
            ax.axvspan(group_x[vi] - GROUP_SEP / 2 + 0.02,
                       group_x[vi] + GROUP_SEP / 2 - 0.02,
                       color="#F2F5F9", alpha=0.7, zorder=0)

    # Vertical separators
    for vi in range(n_vars - 1):
        xmid = (group_x[vi] + group_x[vi + 1]) / 2
        ax.axvline(xmid, color="#BBBBBB", linewidth=0.6, linestyle="--", zorder=1)

    ax.set_xticks(group_x)
    ax.set_xticklabels(var_labels, fontsize=13, fontweight="bold")
    ax.set_xlim(group_x[0] - GROUP_SEP * 0.55, group_x[-1] + GROUP_SEP * 0.55)

    ax.set_ylabel("NRMSE Improvement (%)\nMature Phase (Days 40–60)", fontsize=12)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", which="major", linestyle="--", color="#AAAAAA", zorder=0)
    ax.grid(axis="y", which="minor", linestyle=":",  color="#CCCCCC", zorder=0, alpha=0.3)

    ax.set_title(f"NRMSE Improvement — {exp_label}",
                 fontsize=15, fontweight="bold", pad=12)
    ax.text(0.5, 1.012,
            "Mature Phase (Days 40–60)  |  Positive = improvement over free-running control",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#555555", style="italic")

    # Legend — placed below axes, outside the plot
    legend_patches = [
        mpatches.Patch(facecolor=c, edgecolor="white", linewidth=0.7, label=m)
        for m, c in zip(MODELS, COLORS)
    ]
    leg = ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        handlelength=1.4,
        handleheight=0.9,
        borderpad=0.6,
        labelspacing=0.5,
        columnspacing=1.2,
        framealpha=0.97,
        edgecolor="0.80",
        fontsize=10.5,
    )
    # Q/RH footnote below legend
    ax.annotate(
        "‡ Q/RH: q500 (FCN, Aurora, GraphCast); TQV (Prithvi)",
        xy=(0.5, -0.21),
        xycoords="axes fraction",
        ha="center", va="top",
        fontsize=8.5, color="#444444", style="italic",
    )

    plt.tight_layout(rect=[0, 0.14, 1, 0.97])
    out = OUT_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Print summary table ────────────────────────────────────────────────────────
def print_summary():
    print("\n=== Mature-phase NRMSE improvement summary (%) ===")
    for exp_key, exp_label, var_list in [
        ("t",  "T-ins",  ["U","V","Z","Q"]),
        ("uv", "UV-ins", ["T","Z","Q"]),
        ("z",  "Z-ins",  ["T","U","V","Q"]),
    ]:
        print(f"\n{exp_label}:")
        for vk in var_list:
            row = f"  {vk:3s} |"
            for model in MODELS:
                v = get_mature(model, vk, exp_key)
                row += f"  {model:10s}: {v:+6.1f}%" if not np.isnan(v) else f"  {model:10s}:    N/A "
            print(row)


# ── Generate plots ─────────────────────────────────────────────────────────────
print("Generating publication-quality NRMSE improvement plots…\n")

make_plot(
    exp_key    = "t",
    exp_label  = "Temperature Insertion (T500)",
    var_list   = ["U", "V", "Z", "Q"],
    var_labels = ["U500", "V500", "Z500", "Q/RH"],
    filename   = "fig_nrmse_t_insertion.png",
)

make_plot(
    exp_key    = "uv",
    exp_label  = "Wind Insertion (U + V)",
    var_list   = ["T", "Z", "Q"],
    var_labels = ["T500", "Z500", "Q/RH"],
    filename   = "fig_nrmse_uv_insertion.png",
)

make_plot(
    exp_key    = "z",
    exp_label  = "Geopotential Insertion (Z500 / H500)",
    var_list   = ["T", "U", "V", "Q"],
    var_labels = ["T500", "U500", "V500", "Q/RH"],
    filename   = "fig_nrmse_z_insertion.png",
)

print_summary()
print(f"\nDone. All plots → {OUT_DIR}")
