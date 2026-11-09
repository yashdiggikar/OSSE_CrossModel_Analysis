#!/usr/bin/env python3
"""
Deduplicated regeneration of the 4 bar-chart "improvement by regime/terrain"
figures (one each from tcc, cloud2, terrain, terrain2) as single files
(not duplicated 4x into per-model subfolders). Within each of the 4 images,
every panel shares the SAME positive y-axis scale, set to the highest bar
value reached anywhere in that image (plus headroom, rounded up to a nice
number). Negative-side limits are left as auto-scaled per panel.

Reads the already-cached intermediates/*.csv from each of the 4 source
directories; does not touch or regenerate anything in those directories.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _regime_terrain_plot_common import (
    apply_style, load_rmse_dict, phase_improvements, build_bar_figure, compute_uniform_top,
)

COMMON = Path("/umbc/rs/pi_halem/users/yashd1/common")
OUT = Path(__file__).resolve().parent

MODEL_LABELS = {"aurora": "Aurora", "fcn": "FourCastNet", "prithvi": "Prithvi-WxC", "graphcast": "GraphCast"}
MODEL_ORDER  = ["aurora", "fcn", "prithvi", "graphcast"]
PHASE_INFO = {
    "aurora":    (60, slice(60, 80), slice(80, 120)),
    "fcn":       (60, slice(60, 80), slice(80, 120)),
    "prithvi":   (120, slice(120, 160), slice(160, 240)),
    "graphcast": (120, slice(120, 160), slice(160, 240)),
}

apply_style()


def load_improvements(base, csv_suffix, cat_col, cross_vars, cat_short):
    improvements = {}
    for model in MODEL_ORDER:
        csv_path = base / model / "intermediates" / f"{model}_{csv_suffix}"
        lead, rdict = load_rmse_dict(csv_path, cross_vars, cat_col, cat_short)
        inj, adj, mat = PHASE_INFO[model]
        improvements[model] = phase_improvements(rdict, cross_vars, inj, adj, mat)
    return improvements


# ── tcc : Temperature Insertion by Cloud Regime ─────────────────────────────
REGIME_NAMES  = ["Clear\n(TCC<10%)", "Cirrus\n(10–40%)", "Mixed\n(40–70%)", "Opaque\n(>70%)"]
REGIME_SHORT  = ["Clear", "Cirrus", "Mixed", "Opaque"]
TCC_VARS = ["U500", "V500", "Z500", "Q700"]
TCC_VAR_SHORT = {"U500": "U500", "V500": "V500", "Z500": "Z500", "Q700": "Q/R 700"}

imp = load_improvements(COMMON / "tcc", "t_ins_regime_rmse.csv", "regime", TCC_VARS, REGIME_SHORT)
top = compute_uniform_top(imp, MODEL_ORDER, TCC_VARS)
print(f"  tcc uniform top = {top}")
fig = build_bar_figure(
    "Figure 5.75 – Temperature Insertion: RMSE Improvement (%) by Cloud Regime\n"
    "Baseline = Day-30 Control RMSE  |  Positive = improvement  |  4 Models",
    MODEL_ORDER, MODEL_LABELS, TCC_VARS, TCC_VAR_SHORT, imp,
    REGIME_NAMES, 4, "Cloud Regime", figsize=(24, 18), top_ylim=top)
fig.savefig(OUT / "tcc_t_ins_rmse_improvement_by_regime.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved tcc_t_ins_rmse_improvement_by_regime.png")

# ── cloud2 : UV Insertion by Cloud Regime ───────────────────────────────────
CLOUD2_VARS = ["T500", "Z500", "Q850"]
imp = load_improvements(COMMON / "cloud2", "uv_ins_regime_rmse.csv", "regime", CLOUD2_VARS, REGIME_SHORT)
top = compute_uniform_top(imp, MODEL_ORDER, CLOUD2_VARS)
print(f"  cloud2 uniform top = {top}")
fig = build_bar_figure(
    "UV Insertion: RMSE Improvement (%) by Cloud Regime\n4 Models  |  Baseline = Day-30 Control",
    MODEL_ORDER, MODEL_LABELS, CLOUD2_VARS, {v: v for v in CLOUD2_VARS}, imp,
    ["Clear\n(<10%)", "Cirrus\n(10-40%)", "Mixed\n(40-70%)", "Opaque\n(>70%)"], 4,
    "Cloud Regime", figsize=(20, 18), top_ylim=top)
fig.savefig(OUT / "cloud2_uv_ins_rmse_improvement_bar.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved cloud2_uv_ins_rmse_improvement_bar.png")

# ── terrain : Temperature Insertion by Terrain Class ────────────────────────
TERR_NAMES  = ["Land Low\n(<300 m)", "Land Mod\n(300-1500 m)", "Land High\n(>1500 m)",
               "Ocean", "Polar\n(|phi|>60)"]
TERR_SHORT  = ["Land Low", "Land Mod", "Land High", "Ocean", "Polar"]
TERRAIN_VARS = ["U500", "V500", "Z500", "Q700"]
TERRAIN_VAR_SHORT = {"U500": "U500", "V500": "V500", "Z500": "Z500", "Q700": "Q/R 700"}

imp = load_improvements(COMMON / "terrain", "t_ins_terrain_rmse.csv", "terrain", TERRAIN_VARS, TERR_SHORT)
top = compute_uniform_top(imp, MODEL_ORDER, TERRAIN_VARS)
print(f"  terrain uniform top = {top}")
fig = build_bar_figure(
    "T Insertion: RMSE Improvement (%) by Terrain Class\n"
    "Baseline = Day-30 Control RMSE  |  Positive = improvement  |  4 Models",
    MODEL_ORDER, MODEL_LABELS, TERRAIN_VARS, TERRAIN_VAR_SHORT, imp,
    TERR_NAMES, 5, "Terrain Class", figsize=(26, 18), top_ylim=top)
fig.savefig(OUT / "terrain_t_ins_rmse_improvement_bar.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved terrain_t_ins_rmse_improvement_bar.png")

# ── terrain2 : UV Insertion by Terrain Class ────────────────────────────────
TERRAIN2_VARS = ["T500", "Z500", "Q850"]
imp = load_improvements(COMMON / "terrain2", "uv_ins_terrain_rmse.csv", "terrain", TERRAIN2_VARS, TERR_SHORT)
top = compute_uniform_top(imp, MODEL_ORDER, TERRAIN2_VARS)
print(f"  terrain2 uniform top = {top}")
fig = build_bar_figure(
    "UV Insertion: RMSE Improvement (%) by Terrain Class\n4 Models  |  Baseline = Day-30 Control",
    MODEL_ORDER, MODEL_LABELS, TERRAIN2_VARS, {v: v for v in TERRAIN2_VARS}, imp,
    ["Land Low\n(<300m)", "Land Mod\n(300-1500m)", "Land High\n(>1500m)", "Ocean", "Polar\n(|lat|>60)"], 5,
    "Terrain Class", figsize=(22, 18), top_ylim=top)
fig.savefig(OUT / "terrain2_uv_ins_rmse_improvement_bar.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved terrain2_uv_ins_rmse_improvement_bar.png")

print("Done — 4 deduplicated bar-summary images written to", OUT)
