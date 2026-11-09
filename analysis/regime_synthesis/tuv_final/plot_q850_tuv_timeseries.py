#!/usr/bin/env python3
"""
Q850 / RH850 RMSE time series, 4 curves per model: Control, T-insertion,
UV-insertion, T+UV (TUV) insertion.

Control / T-ins / UV-ins  -> Multimodel-study/{Model}/{control,temp_ins,wind_ins}
TUV (combined T+U+V)      -> common/tuv_plots/cache/{model}_tuv_rmse.csv
                              (original-location pipeline: compute_tuv_rmse.py /
                              recompute_prithvi_tuv_rmse.py for the corrected
                              Prithvi MERRA level mapping)

Variable per model (850 hPa moisture, matching the existing table5_7 "Q/RH"
convention):
  FCNv2     -> r850     (relative humidity, %)
  Aurora    -> q850     (specific humidity, kg/kg)
  PrithviWxC-> QV_lev63 / QV_lv10  (specific humidity, kg/kg, same 850 hPa level)

One figure per model. Outputs -> common/tuv_final/{Model}_Q850_RMSE_timeseries.png
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


def load_fcn_aurora(csv_path, lead_offset, var):
    df = pd.read_csv(csv_path, index_col=0)
    row = df.loc[var]
    times = pd.to_datetime(row.index)
    lead_days = (times - times[0]).total_seconds().values / 86400.0 + lead_offset
    return lead_days, row.values.astype(float)


def load_prithvi(csv_path, var):
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


# ── FourCastNet V2 (relative humidity, %) ───────────────────────────────────
print("FourCastNet V2 ...")
fcn_series = {
    "Control": load_fcn_aurora(MM / "FCNv2/Control/rmse_all_vars.csv",  lead_offset=0.0, var="r850"),
    "T-Ins":   load_fcn_aurora(MM / "FCNv2/temp_ins/rmse_all_vars.csv", lead_offset=0.0, var="r850"),
    "UV-Ins":  load_fcn_aurora(MM / "FCNv2/wind_ins/rmse_all_vars.csv", lead_offset=0.0, var="r850"),
    "TUV-Ins": load_tuv_cache("fcn", "r850"),
}
make_plot("FourCastNet V2 — Q850/RH850 RMSE", fcn_series,
          "FCNv2_Q850_RMSE_timeseries.png", "RH850 RMSE  [%]")

# ── Aurora (specific humidity, kg/kg) ───────────────────────────────────────
print("Aurora ...")
aur_series = {
    "Control": load_fcn_aurora(MM / "Aurora/control/aurora_rmse_all_vars.csv",  lead_offset=0.5, var="q850"),
    "T-Ins":   load_fcn_aurora(MM / "Aurora/temp_ins/aurora_rmse_all_vars.csv", lead_offset=0.5, var="q850"),
    "UV-Ins":  load_fcn_aurora(MM / "Aurora/wind_ins/aurora_rmse_all_vars.csv", lead_offset=0.5, var="q850"),
    "TUV-Ins": load_tuv_cache("aurora", "q850"),
}
make_plot("Aurora — Q850 RMSE", aur_series,
          "Aurora_Q850_RMSE_timeseries.png", "Q850 RMSE  [kg kg⁻¹]")

# ── PrithviWxC (specific humidity, kg/kg, 850 hPa) ──────────────────────────
print("PrithviWxC ...")
pri_series = {
    "Control": load_prithvi(MM / "PrithviWxC/control/rmse_upper.csv",   var="QV_lev63"),
    "T-Ins":   load_prithvi(MM / "PrithviWxC/temp_ins/rmse_upper.csv",  var="QV_lev63"),
    "UV-Ins":  load_prithvi(MM / "PrithviWxC/wind_ins/rmse_upper.csv",  var="QV_lev63"),
    "TUV-Ins": load_tuv_cache("prithvi", "QV_lv10"),
}
make_plot("PrithviWxC — Q850 (QV@850hPa) RMSE", pri_series,
          "PrithviWxC_Q850_RMSE_timeseries.png", "Specific Humidity RMSE  [kg kg⁻¹]")

print(f"\nDone — 3 plots written to {OUT}")
