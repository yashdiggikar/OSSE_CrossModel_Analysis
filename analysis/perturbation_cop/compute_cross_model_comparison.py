#!/usr/bin/env python3
"""
Cross-model sensitivity comparison across FCNv2 / Aurora / Prithvi WxC's
Correlated OSSE Perturbation experiments.

See CROSS_MODEL_COMPARISON.md for the full method description and
rationale. Reads (does not recompute) each model's already-existing
global_rmse_<experiment>.csv (from that model's own
compute_global_rmse_corr.py) and spectral_blur_summary_<experiment>.csv
(from that model's compute_spectral_blur_corr.py, new this session).
A model missing either input for a given experiment type is skipped for
that comparison, with a warning printed, not a hard failure -- this script
is meant to run on whichever subset of models currently has complete data.

Output (into PERTURBATION_EXPERIMENTS/CrossModel/):
    cross_model_effectiveness_<temp|uv>.csv
    cross_model_damping_<temp|uv>.csv
    cross_model_blurring_<temp|uv>_<band>.csv
    cross_model_effectiveness_<temp|uv>.png
    cross_model_blurring_<temp|uv>_<band>.png

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_cross_model_comparison.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path("/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS")
OUT_DIR = BASE / "CrossModel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INJECT_LEAD_HOUR = 720  # target lead hour (30 days) for the injection step -- looked up by
                         # nearest lead_hour match per model, NOT a fixed row index: models'
                         # global_rmse CSVs are not row-aligned to the same lead_hour (e.g. FCN's
                         # row 0 is lead_hour=0, Aurora's row 0 is lead_hour=6 -- its saved output
                         # has no t=0 row), so a hardcoded row index silently picks different real
                         # times per model. See SESSION_LOG_2026-07-22.md continuation entry for
                         # 2026-07-24 for how this was found.
BANDS = ["equator_5N", "midlat_45N", "highlat_70N"]

# Normalized variable-category -> per-model channel label, per CROSS_MODEL_COMPARISON.md Section 5.
# All 3 models' "vs truth" scripts (global_rmse/latband_rmse/spatial_bias) and
# the correlation-length/spectral-blur scripts were expanded (2026-07-25) to a
# full T/U/V/Z/humidity x 300/500/1000hPa grid, computed for every experiment
# regardless of which family was directly inserted -- so every model's CSVs
# now use the SAME "<LABEL><hpa>" naming convention at the representative
# levels below (e.g. "T500", "Z500"), including Prithvi, which previously
# used its own "T_LEV51"/"U_LEV51"/"V_LEV51" labels (bare, no hpa suffix) for
# its single ~532hPa representative level -- those labels no longer exist in
# Prithvi's CSVs and are replaced here with "T500"/"U500"/"V500" (Prithvi's
# nearest-native-level value at that label, same physical level as before,
# just renamed for consistency with FCN/Aurora). Two new categories
# (geopotential_upper_air, humidity_upper_air) added below now that Z and
# humidity are computed for every model/experiment -- FCN's humidity is
# relative humidity ("R", known absent from FCN's ERA5 archive at every
# level per compute_global_rmse_corr.py's own discovery, so FCN's
# humidity_upper_air rows will just be all-NaN/skipped, not a bug).
MODELS = {
    "FCN": {
        "dir": "FCN",
        "experiments": {"temp": "t_perturbed_corr", "uv": "uv_perturbed_corr"},
        "rmse_categories": {
            "temp_upper_air": "T500", "temp_surface": "T2M",
            "wind_u_upper_air": "U500", "wind_v_upper_air": "V500",
            "wind_u_surface": "U10", "wind_v_surface": "V10",
            "geopotential_upper_air": "Z500", "humidity_upper_air": "R500",
        },
        "spectral_categories": {
            "temp_upper_air": "T500", "wind_u_upper_air": "U500", "wind_v_upper_air": "V500",
            "geopotential_upper_air": "Z500", "humidity_upper_air": "R500",
        },
    },
    "Aurora": {
        "dir": "Aurora",
        "experiments": {"temp": "T_PERTURBED_CORR", "uv": "UV_PERTURBED_CORR"},
        "rmse_categories": {
            "temp_upper_air": "T500", "temp_surface": "T2M",
            "wind_u_upper_air": "U500", "wind_v_upper_air": "V500",
            "wind_u_surface": "U10", "wind_v_surface": "V10",
            "geopotential_upper_air": "Z500", "humidity_upper_air": "Q500",
        },
        "spectral_categories": {
            "temp_upper_air": "T500", "wind_u_upper_air": "U500", "wind_v_upper_air": "V500",
            "geopotential_upper_air": "Z500", "humidity_upper_air": "Q500",
        },
    },
    "Prithvi": {
        "dir": "Prithvi",
        "experiments": {"temp": "perturbed_corr_temp_ins", "uv": "perturbed_corr_uv_ins"},
        "rmse_categories": {
            "temp_upper_air": "T500", "temp_surface": "T2M",
            "wind_u_upper_air": "U500", "wind_v_upper_air": "V500",
            "wind_u_surface": "U10M", "wind_v_surface": "V10M",
            "geopotential_upper_air": "Z500", "humidity_upper_air": "Q500",
        },
        "spectral_categories": {
            "temp_upper_air": "T500", "wind_u_upper_air": "U500", "wind_v_upper_air": "V500",
            "geopotential_upper_air": "Z500", "humidity_upper_air": "Q500",
        },
    },
}

TEMP_CATEGORIES = ["temp_upper_air", "temp_surface", "geopotential_upper_air", "humidity_upper_air"]
WIND_CATEGORIES = ["wind_u_upper_air", "wind_v_upper_air", "wind_u_surface", "wind_v_surface",
                    "geopotential_upper_air", "humidity_upper_air"]


def effectiveness_and_damping(experiment_type):
    categories = TEMP_CATEGORIES if experiment_type == "temp" else WIND_CATEGORIES
    eff_rows, damp_rows = [], []

    for model, cfg in MODELS.items():
        exp_name = cfg["experiments"][experiment_type]
        csv_path = BASE / cfg["dir"] / exp_name / f"global_rmse_{exp_name}.csv"
        if not csv_path.exists():
            print(f"  [{experiment_type}] {model}: SKIP — {csv_path.name} not found (not yet computed).")
            continue
        df = pd.read_csv(csv_path)

        for cat in categories:
            label = cfg["rmse_categories"][cat]
            ctrl_col, exp_col = f"control_rmse_{label}", f"experiment_rmse_{label}"
            if ctrl_col not in df.columns or exp_col not in df.columns:
                print(f"  [{experiment_type}] {model}/{cat}: SKIP — columns for '{label}' not in CSV.")
                continue

            inject_idx = (df["lead_hour"] - INJECT_LEAD_HOUR).abs().idxmin()
            last_idx = len(df) - 1

            post = df.iloc[inject_idx:last_idx + 1]
            eff = (post[ctrl_col] - post[exp_col]) / post[ctrl_col]
            for row_idx, (lead_hour, e) in enumerate(zip(post["lead_hour"], eff)):
                eff_rows.append({"model": model, "category": cat, "lead_hour": lead_hour,
                                  "step": inject_idx + row_idx,
                                  "control_rmse": post[ctrl_col].iloc[row_idx],
                                  "experiment_rmse": post[exp_col].iloc[row_idx],
                                  "effectiveness": e})

            eff_first = (df[ctrl_col].iloc[inject_idx] - df[exp_col].iloc[inject_idx]) / df[ctrl_col].iloc[inject_idx]
            eff_last = (df[ctrl_col].iloc[last_idx] - df[exp_col].iloc[last_idx]) / df[ctrl_col].iloc[last_idx]
            damp_rows.append({"model": model, "category": cat,
                               "lead_hour_first": df["lead_hour"].iloc[inject_idx],
                               "lead_hour_last": df["lead_hour"].iloc[last_idx],
                               "effectiveness_first": eff_first, "effectiveness_last": eff_last,
                               "damping_ratio": eff_last / eff_first if eff_first != 0 else np.nan})

    return pd.DataFrame(eff_rows), pd.DataFrame(damp_rows)


def blurring_comparison(experiment_type, band):
    # geopotential/humidity included for both experiment types -- cross-variable
    # coupling response fields, not directly-inserted, same as the RMSE side.
    base = ["temp_upper_air"] if experiment_type == "temp" else ["wind_u_upper_air", "wind_v_upper_air"]
    categories = base + ["geopotential_upper_air", "humidity_upper_air"]
    rows = []
    for model, cfg in MODELS.items():
        exp_name = cfg["experiments"][experiment_type]
        csv_path = BASE / cfg["dir"] / exp_name / f"spectral_blur_summary_{exp_name}.csv"
        if not csv_path.exists():
            print(f"  [{experiment_type}/{band}] {model}: SKIP — {csv_path.name} not found (not yet computed).")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["band"] == band]
        for cat in categories:
            label = cfg["spectral_categories"][cat]
            sub = df[df["variable"] == label]
            for _, r in sub.iterrows():
                rows.append({"model": model, "category": cat, "step": r["step"],
                             "effective_resolution_wavenumber": r["effective_resolution_wavenumber"]})
    return pd.DataFrame(rows)


for experiment_type in ["temp", "uv"]:
    print(f"\n=== {experiment_type} ===")

    eff_df, damp_df = effectiveness_and_damping(experiment_type)
    if not eff_df.empty:
        eff_df.to_csv(OUT_DIR / f"cross_model_effectiveness_{experiment_type}.csv", index=False)
        damp_df.to_csv(OUT_DIR / f"cross_model_damping_{experiment_type}.csv", index=False)
        print(f"  Saved: {OUT_DIR / f'cross_model_effectiveness_{experiment_type}.csv'}")
        print(f"  Saved: {OUT_DIR / f'cross_model_damping_{experiment_type}.csv'}")

        categories = eff_df["category"].unique()
        fig, axes = plt.subplots(1, len(categories), figsize=(5 * len(categories), 4), squeeze=False)
        for ax, cat in zip(axes[0], categories):
            for model in eff_df["model"].unique():
                # dropna is required, not cosmetic: control files have a coarser cadence than
                # the experiment (e.g. FCN/Aurora control is 12-hourly vs 6-hourly experiment),
                # so "effectiveness" is NaN at every other row. A line with alternating NaN gaps
                # is many isolated 2-point segments; matplotlib's default path simplification
                # drops them and the line silently renders blank (legend entry still appears).
                sub = eff_df[(eff_df["category"] == cat) & (eff_df["model"] == model)].dropna(subset=["effectiveness"])
                ax.plot(sub["lead_hour"], sub["effectiveness"], label=model, lw=1.5)
            ax.set_title(cat, fontsize=9); ax.set_xlabel("lead hour"); ax.set_ylabel("effectiveness")
            ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle(f"Cross-model insertion effectiveness — {experiment_type}")
        fig.savefig(OUT_DIR / f"cross_model_effectiveness_{experiment_type}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    else:
        print(f"  [{experiment_type}] no models had usable global_rmse CSVs yet — nothing to compare.")

    for band in BANDS:
        blur_df = blurring_comparison(experiment_type, band)
        if blur_df.empty:
            print(f"  [{experiment_type}/{band}] no models had usable spectral_blur_summary CSVs yet.")
            continue
        blur_df.to_csv(OUT_DIR / f"cross_model_blurring_{experiment_type}_{band}.csv", index=False)
        print(f"  Saved: {OUT_DIR / f'cross_model_blurring_{experiment_type}_{band}.csv'}")

        fig, ax = plt.subplots(figsize=(7, 5))
        for model in blur_df["model"].unique():
            for cat in blur_df["category"].unique():
                sub = blur_df[(blur_df["model"] == model) & (blur_df["category"] == cat)]
                if sub.empty:
                    continue
                ax.plot(sub["step"], sub["effective_resolution_wavenumber"],
                        marker="o", ms=3, label=f"{model} {cat}")
        ax.set_xlabel("step"); ax.set_ylabel("effective-resolution wavenumber")
        ax.set_title(f"Cross-model blurring rate — {experiment_type}, band={band}")
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
        fig.savefig(OUT_DIR / f"cross_model_blurring_{experiment_type}_{band}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

print("\nDone.")
