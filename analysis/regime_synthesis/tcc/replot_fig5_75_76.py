#!/usr/bin/env python3
"""
Lightweight regeneration of Figures 5.75/5.76 (cloud/tcc) from the cached
per-model intermediate RMSE CSVs, applying the shared cosmetic style
(legend below the figure, muted colors, Times New Roman, no overlapping
labels) without re-running the full raw-data ingestion pipeline in
fig5_75_76_temp_ins_cloud_regime.py.
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

BASE = Path("/umbc/rs/pi_halem/users/yashd1/common/tcc")
OUT = {m: BASE / m for m in ["aurora", "fcn", "prithvi", "graphcast"]}

REGIME_NAMES  = ["Clear\n(TCC<10%)", "Cirrus\n(10–40%)", "Mixed\n(40–70%)", "Opaque\n(>70%)"]
REGIME_SHORT  = ["Clear", "Cirrus", "Mixed", "Opaque"]
REGIME_COLORS = ["#D4A24E", "#6F9FB8", "#8C7AA9", "#3B4252"]
N_REG = 4

CROSS_VARS = ["U500", "V500", "Z500", "Q700"]
VAR_LABELS = {
    "U500": "U-wind 500 hPa (m/s)",
    "V500": "V-wind 500 hPa (m/s)",
    "Z500": "Geopotential 500 hPa (m2/s2)",
    "Q700": "Moisture 700 hPa  [Sp. Humidity (Aurora/Prithvi/GC) | RH% (FCN)]",
}
VAR_SHORT = {"U500": "U500", "V500": "V500", "Z500": "Z500", "Q700": "Q/R 700"}
MODEL_LABELS = {"aurora": "Aurora", "fcn": "FourCastNet", "prithvi": "Prithvi-WxC", "graphcast": "GraphCast"}
MODEL_ORDER  = ["aurora", "fcn", "prithvi", "graphcast"]

# (intermediate csv tag, day-30 injection step index, adj slice, mat slice)
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
    csv_path = OUT[model] / "intermediates" / f"{model}_t_ins_regime_rmse.csv"
    lead, rdict = load_rmse_dict(csv_path, CROSS_VARS, "regime", REGIME_SHORT)
    inj, adj, mat = PHASE_INFO[model]
    ALL_RMSE[model] = (rdict, inj, adj, mat, lead)
    improvements[model] = phase_improvements(rdict, CROSS_VARS, inj, adj, mat)
    print(f"  loaded {model}: {len(lead)} steps, vars={list(rdict.keys())}")

fig75 = build_bar_figure(
    "Figure 5.75 – Temperature Insertion: RMSE Improvement (%) by Cloud Regime\n"
    "Baseline = Day-30 Control RMSE  |  Positive = improvement  |  4 Models",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_SHORT, improvements,
    REGIME_NAMES, N_REG, "Cloud Regime", figsize=(24, 18))
for model, out_dir in OUT.items():
    fig75.savefig(out_dir / "fig5_75_t_ins_rmse_improvement_by_regime.png", bbox_inches="tight", dpi=300)
    print(f"  saved -> {out_dir.name}/fig5_75_t_ins_rmse_improvement_by_regime.png")
plt.close(fig75)

fig76 = build_timeseries_figure(
    "Figure 5.76 – Temperature Insertion: RMSE Time Series by Cloud Regime\n"
    "Dashed = Control  |  Solid = T Insertion  |  Blue = Adj. phase  |  Pink = Mature phase",
    MODEL_ORDER, MODEL_LABELS, CROSS_VARS, VAR_LABELS, ALL_RMSE,
    REGIME_COLORS, REGIME_SHORT, figsize=(28, 20), hspace=0.5, wspace=0.28)
for model, out_dir in OUT.items():
    fig76.savefig(out_dir / "fig5_76_t_ins_rmse_timeseries_by_regime.png", bbox_inches="tight", dpi=300)
    print(f"  saved -> {out_dir.name}/fig5_76_t_ins_rmse_timeseries_by_regime.png")
plt.close(fig76)

print("Done — tcc replot complete.")
