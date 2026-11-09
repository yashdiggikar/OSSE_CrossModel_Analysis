#!/usr/bin/env python3
"""
Z500 RMSE time series, 4 curves per model: Control, T-insertion, UV-insertion,
T+UV (TUV) insertion.

Control / T-ins / UV-ins  -> Multimodel-study/{Model}/{control,temp_ins,wind_ins}
TUV (combined T+U+V)      -> common/tuv_plots/cache/{model}_tuv_rmse.csv
                              (original-location pipeline: compute_tuv_rmse.py)

One figure per model: FourCastNet V2, Aurora, PrithviWxC.
Outputs -> common/tuv_final/{Model}_Z500_RMSE_timeseries.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker

MM   = Path("/umbc/rs/pi_halem/users/yashd1/Multimodel-study")
TUV_CACHE = Path("/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/cache")
OUT  = Path("/umbc/rs/pi_halem/users/yashd1/common/tuv_final")
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.weight":       "bold",
    "axes.titlesize":    21,
    "axes.titleweight":  "bold",
    "axes.labelsize":    19,
    "axes.labelweight":  "bold",
    "xtick.labelsize":   16,
    "ytick.labelsize":   16,
    "legend.fontsize":   15,
    "axes.linewidth":    2.1,
    "axes.edgecolor":    "#111111",
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#cccccc",
    "grid.linewidth":    0.75,
    "grid.linestyle":    "--",
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
})

CURVE_STYLE = {
    "Control": dict(color="#7F8C8D", lw=2.2, ls=(0, (5, 3)), zorder=2, label="Control"),
    "T-Ins":   dict(color="#C0392B", lw=2.4, ls="-",          zorder=3, label="Temperature Insertion"),
    "UV-Ins":  dict(color="#2E8B6B", lw=2.4, ls="-",          zorder=3, label="Wind (UV) Insertion"),
    "TUV-Ins": dict(color="#5B4E8C", lw=2.8, ls="-",          zorder=4, label="T+UV Insertion"),
}


def load_fcn_aurora(csv_path, lead_offset, var="z500"):
    df = pd.read_csv(csv_path, index_col=0)
    row = df.loc[var]
    times = pd.to_datetime(row.index)
    lead_days = (times - times[0]).total_seconds().values / 86400.0 + lead_offset
    return lead_days, row.values.astype(float)


def load_prithvi(csv_path, var="H_lev51"):
    df = pd.read_csv(csv_path)
    lead_days = df["lead_hour"].values / 24.0
    return lead_days, df[var].values.astype(float)


def load_tuv_cache(model_tag, var):
    df = pd.read_csv(TUV_CACHE / f"{model_tag}_tuv_rmse.csv")
    return df["lead_day"].values, df[var].values.astype(float)


def make_plot(model_title, series, fname, ylabel):
    fig, ax = plt.subplots(figsize=(11.5, 8.2))

    for key in ["Control", "T-Ins", "UV-Ins", "TUV-Ins"]:
        lead, vals = series[key]
        order = np.argsort(lead)
        ax.plot(lead[order], vals[order], **CURVE_STYLE[key])

    ax.set_xlim(-1.5, 61.5)
    ax.margins(y=0.10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel(ylabel)

    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_linewidth(2.1)
        ax.spines[spine_name].set_color("#111111")

    fig.suptitle(model_title, y=0.97, fontsize=21, fontweight="bold")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                     ncol=4, frameon=False, handlelength=2.2,
                     columnspacing=1.4, fontsize=12.5)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.savefig(str(OUT / fname), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved: {fname}")


# ── FourCastNet V2 ───────────────────────────────────────────────────────────
print("FourCastNet V2 ...")
fcn_series = {
    "Control": load_fcn_aurora(MM / "FCNv2/Control/rmse_all_vars.csv", lead_offset=0.0),
    "T-Ins":   load_fcn_aurora(MM / "FCNv2/temp_ins/rmse_all_vars.csv", lead_offset=0.0),
    "UV-Ins":  load_fcn_aurora(MM / "FCNv2/wind_ins/rmse_all_vars.csv", lead_offset=0.0),
    "TUV-Ins": load_tuv_cache("fcn", "z500"),
}
make_plot("FourCastNet V2 — Z500 RMSE", fcn_series,
          "FCNv2_Z500_RMSE_timeseries.png", "Z500 RMSE  [m² s⁻²]")

# ── Aurora ────────────────────────────────────────────────────────────────────
print("Aurora ...")
aur_series = {
    "Control": load_fcn_aurora(MM / "Aurora/control/aurora_rmse_all_vars.csv", lead_offset=0.5),
    "T-Ins":   load_fcn_aurora(MM / "Aurora/temp_ins/aurora_rmse_all_vars.csv", lead_offset=0.5),
    "UV-Ins":  load_fcn_aurora(MM / "Aurora/wind_ins/aurora_rmse_all_vars.csv", lead_offset=0.5),
    "TUV-Ins": load_tuv_cache("aurora", "z500"),
}
make_plot("Aurora — Z500 RMSE", aur_series,
          "Aurora_Z500_RMSE_timeseries.png", "Z500 RMSE  [m² s⁻²]")

# ── PrithviWxC ────────────────────────────────────────────────────────────────
print("PrithviWxC ...")
pri_series = {
    "Control": load_prithvi(MM / "PrithviWxC/control/rmse_upper.csv",   var="H_lev51"),
    "T-Ins":   load_prithvi(MM / "PrithviWxC/temp_ins/rmse_upper.csv",  var="H_lev51"),
    "UV-Ins":  load_prithvi(MM / "PrithviWxC/wind_ins/rmse_upper.csv",  var="H_lev51"),
    "TUV-Ins": load_tuv_cache("prithvi", "H_lv7"),
}
make_plot("PrithviWxC — Z500 (H@500hPa) RMSE", pri_series,
          "PrithviWxC_Z500_RMSE_timeseries.png", "Geopotential Height RMSE  [m]")

print(f"\nDone — 3 plots written to {OUT}")
