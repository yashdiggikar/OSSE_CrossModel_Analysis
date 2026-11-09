#!/usr/bin/env python3
"""
Lightweight regeneration of the terrain (Temperature Insertion by Terrain
Class) figures from the cached per-model intermediate RMSE CSVs, applying
the shared cosmetic style without re-running fig_terrain_temp_ins.py's full
raw-data ingestion pipeline.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _regime_terrain_plot_common import (
    apply_style, load_rmse_dict, phase_improvements, build_bar_figure, build_timeseries_figure,
)

BASE = Path("/umbc/rs/pi_halem/users/yashd1/common/terrain")
OUT = {m: BASE / m for m in ["aurora", "fcn", "prithvi", "graphcast"]}

N_TERR = 5
TERR_NAMES  = ["Land Low\n(<300 m)", "Land Mod\n(300-1500 m)", "Land High\n(>1500 m)",
               "Ocean", "Polar\n(|phi|>60)"]
TERR_SHORT  = ["Land Low", "Land Mod", "Land High", "Ocean", "Polar"]
TERR_COLORS = ["#9CAF6E", "#6B8E4E", "#5C4A3D", "#4A7A96", "#A9BFC9"]

CROSS_VARS = ["U500", "V500", "Z500", "Q700"]
VAR_LABELS = {
    "U500": "U-wind 500 hPa (m/s)", "V500": "V-wind 500 hPa (m/s)",
    "Z500": "Geopotential 500 hPa (m2/s2)",
    "Q700": "Moisture 700 hPa [Sp. Humidity (Aurora/Prithvi/GC) | RH% (FCN)]",
}
VAR_SHORT = {"U500": "U500", "V500": "V500", "Z500": "Z500", "Q700": "Q/R 700"}
MODEL_LABELS = {"aurora": "Aurora", "fcn": "FourCastNet", "prithvi": "Prithvi-WxC", "graphcast": "GraphCast"}
MODEL_ORDER  = ["aurora", "fcn", "prithvi", "graphcast"]

PHASE_INFO = {
    "aurora":    (60, slice(60, 80), slice(80, 120)),
    "fcn":       (60, slice(60, 80), slice(80, 120)),
    "prithvi":   (120, slice(120, 160), slice(160, 240)),
    "graphcast": (120, slice(120, 160), slice(160, 240)),
}

apply_style()

ALL_RMSE = {}
improvements = {}
for model in MODEL_ORDER:
    csv_path = OUT[model] / "intermediates" / f"{model}_t_ins_terrain_rmse.csv"
    lead, rdict = load_rmse_dict(csv_path, CROSS_VARS, "terrain", TERR_SHORT)
    inj, adj, mat = PHASE_INFO[model]
    ALL_RMSE[model] = (rdict, inj, adj, mat, lead)
    improvements[model] = phase_improvements(rdict, CROSS_VARS, inj, adj, mat)
    print(f"  loaded {model}: {len(lead)} steps, vars={list(rdict.keys())}")

fig_bar = build_bar_figure(
    "T Insertion: RMSE Improvement (%) by Terrain Class\n"
    "Baseline = Day-30 Control RMSE  |  Positive = improvement  |  4 Models",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_SHORT, improvements,
    TERR_NAMES, N_TERR, "Terrain Class", figsize=(26, 18))
for model, out_dir in OUT.items():
    fig_bar.savefig(out_dir / "fig_terrain_t_ins_rmse_improvement_bar.png", bbox_inches="tight", dpi=300)
    print(f"  saved -> {out_dir.name}/fig_terrain_t_ins_rmse_improvement_bar.png")
plt.close(fig_bar)

fig_ts = build_timeseries_figure(
    "T Insertion: RMSE Time Series by Terrain Class\n"
    "Dashed = Control  |  Solid = T Insertion  |  Blue = Adj. phase  |  Pink = Mature phase",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_LABELS, ALL_RMSE,
    TERR_COLORS, TERR_SHORT, figsize=(28, 20), hspace=0.52, wspace=0.28)
for model, out_dir in OUT.items():
    fig_ts.savefig(out_dir / "fig_terrain_t_ins_rmse_timeseries.png", bbox_inches="tight", dpi=300)
    print(f"  saved -> {out_dir.name}/fig_terrain_t_ins_rmse_timeseries.png")
plt.close(fig_ts)

print("Done — terrain replot complete.")
