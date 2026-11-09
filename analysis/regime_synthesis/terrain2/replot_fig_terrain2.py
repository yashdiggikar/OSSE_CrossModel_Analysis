#!/usr/bin/env python3
"""
Lightweight regeneration of the terrain2 (UV Insertion by Terrain Class)
figures from the cached per-model intermediate RMSE CSVs, applying the
shared cosmetic style without re-running fig_terrain2_uv_ins.py's full
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

BASE = Path("/umbc/rs/pi_halem/users/yashd1/common/terrain2")
OUT = {m: BASE / m for m in ["aurora", "fcn", "prithvi", "graphcast"]}

N_TERR = 5
TERR_NAMES  = ["Land Low\n(<300m)", "Land Mod\n(300-1500m)", "Land High\n(>1500m)", "Ocean", "Polar\n(|lat|>60)"]
TERR_SHORT  = ["Land Low", "Land Mod", "Land High", "Ocean", "Polar"]
TERR_COLORS = ["#9CAF6E", "#6B8E4E", "#5C4A3D", "#4A7A96", "#A9BFC9"]

CROSS_VARS = ["T500", "Z500", "Q850"]
VAR_LABELS = {
    "T500": "Temperature 500 hPa (K)",
    "Z500": "Geopotential 500 hPa (m2/s2)",
    "Q850": "Moisture 850 hPa [Sp.Hum Aurora/Prithvi/GC | RH% FCN]",
}
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
    csv_path = OUT[model] / "intermediates" / f"{model}_uv_ins_terrain_rmse.csv"
    lead, rdict = load_rmse_dict(csv_path, CROSS_VARS, "terrain", TERR_SHORT)
    inj, adj, mat = PHASE_INFO[model]
    ALL_RMSE[model] = (rdict, inj, adj, mat, lead)
    improvements[model] = phase_improvements(rdict, CROSS_VARS, inj, adj, mat)
    print(f"  loaded {model}: {len(lead)} steps, vars={list(rdict.keys())}")

fb = build_bar_figure(
    "UV Insertion: RMSE Improvement (%) by Terrain Class\n4 Models  |  Baseline = Day-30 Control",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, {v: v for v in CROSS_VARS}, improvements,
    TERR_NAMES, N_TERR, "Terrain Class", figsize=(22, 18))
for model, od in OUT.items():
    fb.savefig(od / "fig_terrain2_uv_ins_rmse_improvement_bar.png", bbox_inches="tight", dpi=300)
    print(f"  saved -> {od.name}/fig_terrain2_uv_ins_rmse_improvement_bar.png")
plt.close(fb)

ft = build_timeseries_figure(
    "UV Insertion: RMSE Time Series by Terrain Class  (4 Models)\n"
    "Dashed=Control  Solid=UV Ins  Blue=Adj  Pink=Mature",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_LABELS, ALL_RMSE,
    TERR_COLORS, TERR_SHORT, figsize=(26, 18), hspace=0.5, wspace=0.28)
for model, od in OUT.items():
    ft.savefig(od / "fig_terrain2_uv_ins_rmse_timeseries.png", bbox_inches="tight", dpi=300)
    print(f"  saved -> {od.name}/fig_terrain2_uv_ins_rmse_timeseries.png")
plt.close(ft)

print("Done — terrain2 replot complete.")
