#!/usr/bin/env python3
"""
Figure 5.86 — Spatial sensitivity map (Mature Phase): where the perturbed
(observation-error) global T-insertion forecast differs from the perfect
(noise-free) global full T-insertion forecast, T500, for all 3 models.

Sensitivity(lat,lon) = RMS over the mature-phase window of
    [ T_perturbed(t) - T_full_insertion(t) ]

Mature phase = days ~30-60 (post-injection-onset), matched in absolute valid
time between each model's "global full insertion" (perfect) file and its
"T_PERTURBED" (Gaussian obs-error) file.

Data sources (verified empirically this session, see inline notes):
  Aurora : full = Aurora_Pred_Files/auroraoutput_temp_inserted.nc   (12h cadence, 120 steps,
                  lead_time accurate, true onset idx=61 i.e. lead=732h)
           pert = CC_aurora/T_PERTURBED/output/aurora_t_perturbed.nc (6h cadence, 240 steps,
                  lead_time mislabeled by 1 step -> content = idx*6h, onset idx=121)
  FCNv2  : full = FourCastNet/Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc
                  (12h cadence, 120 steps, time accurate, onset idx=60 i.e. lead=720h)
           pert = fcn_inference/t_perturbed/forecast_t_perturbed_global_day30_60.nc
                  (6h cadence, 240 steps, content = idx*6h, onset idx=120)
  Prithvi: full = prithviOSSE/run2/output/osse_forecast_Jan2022.nc (6h cadence, 240 steps,
                  vt = IC_T1 + (idx+1)*6h, onset idx=120 -- IDENTICAL formula/IC to the
                  established PRITHVI_IC convention used throughout this project)
           pert = prithviOSSE/perturbed_temp_ins/output/perturbed_temp_osse_forecast_Jan2022.nc
                  (same formula/IC, onset idx=120) -> direct 1:1 index match, no conversion.

Both files per model share the IDENTICAL native grid (same model/resolution), so no
land/ocean masking or regridding is needed for this figure.
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

BASE = Path("/umbc/rs/pi_halem/users/yashd1")
OUT_DIR = BASE / "final_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUR_FULL = BASE / "Aurora_Pred_Files" / "auroraoutput_temp_inserted.nc"
AUR_PERT = BASE / "CC_aurora" / "T_PERTURBED" / "output" / "aurora_t_perturbed.nc"
FCN_FULL = BASE / "FourCastNet" / "Prediction_Files_Fcn" / "t_ins" / "t_ins" / "forecast_truth_injected_60days_12hr_FINAL.nc"
FCN_PERT = BASE / "fcn_inference" / "t_perturbed" / "forecast_t_perturbed_global_day30_60.nc"
PRI_FULL = BASE / "prithviOSSE" / "run2" / "output" / "osse_forecast_Jan2022.nc"
PRI_PERT = BASE / "prithviOSSE" / "perturbed_temp_ins" / "output" / "perturbed_temp_osse_forecast_Jan2022.nc"

FCN_T500_CH = 54
PRI_LI_500 = 7

# ─── PER-MODEL MATURE-PHASE INDEX PAIRS (full_idx, pert_idx), verified onsets ─

AURORA_FULL_IDX = list(range(61, 120))                  # 12h cadence, lead 732h..1428h
AURORA_PERT_IDX = [2 * i for i in AURORA_FULL_IDX]       # 6h cadence, matched lead

FCN_FULL_IDX = list(range(60, 120))                      # 12h cadence, lead 720h..1428h
FCN_PERT_IDX = [2 * i for i in FCN_FULL_IDX]

PRI_FULL_IDX = list(range(120, 240))                     # 6h cadence, direct 1:1 match
PRI_PERT_IDX = list(range(120, 240))

# ─── SENSITIVITY COMPUTATION ──────────────────────────────────────────────────

def sensitivity_map(full_path, pert_path, full_idx, pert_idx, get_full, get_pert):
    f_full = h5py.File(str(full_path), "r")
    f_pert = h5py.File(str(pert_path), "r")
    sumsq = None
    n = len(full_idx)
    for fi, pi in zip(full_idx, pert_idx):
        diff = get_pert(f_pert, pi) - get_full(f_full, fi)
        sumsq = diff ** 2 if sumsq is None else sumsq + diff ** 2
    f_full.close(); f_pert.close()
    return np.sqrt(sumsq / n)

def aurora_sensitivity():
    rms = sensitivity_map(AUR_FULL, AUR_PERT, AURORA_FULL_IDX, AURORA_PERT_IDX,
                           lambda h, i: np.array(h["t500"][i]),
                           lambda h, i: np.array(h["t500"][i]))
    with h5py.File(str(AUR_PERT), "r") as h:
        lat, lon = np.array(h["lat"]), np.array(h["lon"])
    return lat, lon, rms

def fcn_sensitivity():
    rms = sensitivity_map(FCN_FULL, FCN_PERT, FCN_FULL_IDX, FCN_PERT_IDX,
                           lambda h, i: np.array(h["prediction"][i, FCN_T500_CH, :, :]),
                           lambda h, i: np.array(h["prediction"][i, FCN_T500_CH, :, :]))
    with h5py.File(str(FCN_PERT), "r") as h:
        lat, lon = np.array(h["lat"]), np.array(h["lon"])
    return lat, lon, rms

def prithvi_sensitivity():
    rms = sensitivity_map(PRI_FULL, PRI_PERT, PRI_FULL_IDX, PRI_PERT_IDX,
                           lambda h, i: np.array(h["T"][i, PRI_LI_500, :, :]),
                           lambda h, i: np.array(h["T"][i, PRI_LI_500, :, :]))
    with h5py.File(str(PRI_PERT), "r") as h:
        lat, lon = np.array(h["lat"]), np.array(h["lon"])
    return lat, lon, rms

# ─── PLOTTING ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 13,
    "axes.titlesize": 15.5,
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

def plot_figure(model_data, out_path):
    n = len(model_data)
    vmax = max(np.nanpercentile(rms, 99) for _, _, _, rms in model_data)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 4.6), constrained_layout=True)
    im = None
    for ax, (name, lat, lon, rms) in zip(axes, model_data):
        im = ax.imshow(rms, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        origin="upper" if lat[0] > lat[-1] else "lower",
                        cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(name, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        gmean = float(np.nanmean(rms))
        ax.text(0.02, 0.04, f"Global mean: {gmean:.2f} K", transform=ax.transAxes,
                 fontsize=11, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55, ec="none"))

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.012, aspect=30)
    cbar.set_label("T500 RMS Difference, Perturbed vs. Full Insertion (K)",
                    fontsize=13, fontweight="bold")

    fig.suptitle("Figure 5.86 — Spatial Sensitivity to Observation-Error Perturbation (Mature Phase)\n"
                 "T500 RMS Difference Between Perturbed-Insertion and Perfect Global Full-Insertion Forecasts",
                 fontsize=15.5, fontweight="bold")
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Computing Aurora sensitivity ...")
    lat_a, lon_a, rms_a = aurora_sensitivity()
    print("Computing FCNv2 sensitivity ...")
    lat_f, lon_f, rms_f = fcn_sensitivity()
    print("Computing Prithvi WxC sensitivity ...")
    lat_p, lon_p, rms_p = prithvi_sensitivity()

    model_data = [("Aurora", lat_a, lon_a, rms_a),
                  ("FCNv2", lat_f, lon_f, rms_f),
                  ("Prithvi WxC", lat_p, lon_p, rms_p)]

    plot_figure(model_data, OUT_DIR / "fig5_86_spatial_sensitivity_perturbed_vs_full.png")

    rows = [{"model": name, "global_mean_rms_K": float(np.nanmean(rms)),
             "global_max_rms_K": float(np.nanmax(rms))}
            for name, _, _, rms in model_data]
    pd.DataFrame(rows).to_csv(OUT_DIR / "fig5_86_spatial_sensitivity_summary.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
