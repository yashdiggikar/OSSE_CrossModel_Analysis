#!/usr/bin/env python3
"""
Spectral blurring diagnostic for the Prithvi WxC Correlated OSSE
Perturbation experiments (perturbed_corr_temp_ins, perturbed_corr_uv_ins).

See ../SPECTRAL_BLURRING_ANALYSIS.md for the full method description and
rationale — same method as FCN/Aurora's versions, adapted to Prithvi's
non-square grid (0.5 deg lat x 0.625 deg lon) and its 14-native-level `T`/
`U`/`V`/`H`/`QV` cubes.

Full variable/level grid (matching this project's other "vs truth" scripts,
per user request): T, U, V, Z (as H), Q (as QV) at the native levels nearest
300/500/1000 hPa, plus surface T2M/U10M/V10M -- computed for BOTH experiments
regardless of which family was directly inserted (cross-variable coupling
included). No MERRA-2 truth needed here (experiment-minus-control, both
Prithvi's own output).

Output per experiment (into PERTURBATION_EXPERIMENTS/Prithvi/<experiment>/):
    spectral_blur_<experiment>.csv
    spectral_blur_summary_<experiment>.csv
    spectral_blur_map_<experiment>_<band>.png

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_spectral_blur_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
import time
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
PRITHVI   = BASE / "prithviOSSE"
CTRL_FILE = PRITHVI / "run2" / "output" / "control_forecast_Jan2022.nc"
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "Prithvi"

EXPERIMENTS = [
    ("perturbed_corr_temp_ins", PRITHVI / "perturbed_corr_temp_ins" / "output" / "perturbed_corr_temp_osse_forecast_Jan2022.nc", "temp"),
    ("perturbed_corr_uv_ins",   PRITHVI / "perturbed_corr_uv_ins"   / "output" / "perturbed_corr_uv_osse_forecast_Jan2022.nc",   "uv"),
]

SNAPSHOT_STEPS = [120, 140, 160, 180, 200, 220, 239]
LEVEL_APPROX_HPA = [52, 118, 164, 226, 265, 310, 424, 532, 604, 693, 832, 903, 946, 961]
LEV_VALS = [34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72]
TARGET_HPA = [300, 500, 1000]
REPR_IDX = {}
for target in TARGET_HPA:
    nearest_i = min(range(len(LEVEL_APPROX_HPA)), key=lambda i: abs(LEVEL_APPROX_HPA[i] - target))
    REPR_IDX[target] = nearest_i

PRITHVI_LATS = np.linspace(-89.5, 90.0, 360)
PRITHVI_LONS = np.linspace(-180.0, 179.375, 576)
GRID_DEG_LAT = 0.5

BANDS = {"equator_5N": 5.0, "midlat_45N": 45.0, "highlat_70N": 70.0}
BAND_ROWS = {name: int(np.argmin(np.abs(PRITHVI_LATS - lat))) for name, lat in BANDS.items()}
WINDOW_DEG = 5.0

UPPER_VARNAMES = {"T": "T", "U": "U", "V": "V", "Z": "H", "Q": "QV"}
SFC_VARNAMES = {"T2M": "T2M", "U10M": "U10M", "V10M": "V10M"}


def open_with_retry(path, attempts=5, delay_s=15):
    for _attempt in range(attempts):
        try:
            return nc4.Dataset(str(path))
        except OSError as e:
            print(f"  transient open error ({e!r}) on {path.name}, retrying in {delay_s}s ...")
            time.sleep(delay_s)
    return None


def zonal_power_spectrum(field2d, row_center, window_rows):
    r0 = max(0, row_center - window_rows)
    r1 = min(field2d.shape[0], row_center + window_rows + 1)
    powers = []
    for r in range(r0, r1):
        line = field2d[r, :] - field2d[r, :].mean()
        fft = np.fft.rfft(line)
        powers.append(np.abs(fft) ** 2)
    return np.mean(powers, axis=0)


def effective_resolution_wavenumber(power, frac=0.01):
    ref = power[1]
    if ref <= 0:
        return np.nan
    below = np.where(power[1:] <= frac * ref)[0]
    return float(below[0] + 1) if len(below) else np.nan


for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    exp_ds = open_with_retry(nc_file)
    if exp_ds is None:
        print(f"[{exp_name}] SKIP — could not open experiment file after retries.")
        continue
    ctrl_ds = open_with_retry(CTRL_FILE)
    if ctrl_ds is None:
        print(f"[{exp_name}] SKIP — could not open {CTRL_FILE} after retries.")
        exp_ds.close()
        continue

    spectrum_rows, summary_rows = [], []
    band_figs = {b: plt.subplots(figsize=(7, 5)) for b in BANDS}
    cmap = plt.get_cmap("viridis")
    first_label_plotted = None

    # Upper-air variables, evaluated at 3 representative native levels each --
    # every variable for both experiments (cross-variable coupling included).
    for label, varname in UPPER_VARNAMES.items():
        if varname not in exp_ds.variables or varname not in ctrl_ds.variables:
            print(f"  [{exp_name}] {label}: variable unavailable, skipping.")
            continue

        for hpa in TARGET_HPA:
            lvl_idx = REPR_IDX[hpa]
            var_label = f"{label}{hpa}"
            if first_label_plotted is None:
                first_label_plotted = var_label

            for si, step in enumerate(SNAPSHOT_STEPS):
                exp_field = np.asarray(exp_ds.variables[varname][step, lvl_idx, :, :])
                ctrl_field = np.asarray(ctrl_ds.variables[varname][step, lvl_idx, :, :])
                diff = exp_field - ctrl_field

                for band_name, row in BAND_ROWS.items():
                    window_rows = int(WINDOW_DEG / GRID_DEG_LAT)
                    power = zonal_power_spectrum(diff, row, window_rows)
                    wavenumbers = np.arange(len(power))
                    for wn, p in zip(wavenumbers, power):
                        spectrum_rows.append({"variable": var_label, "step": step, "band": band_name,
                                               "wavenumber": int(wn), "power": float(p)})
                    eff_res = effective_resolution_wavenumber(power)
                    summary_rows.append({"variable": var_label, "step": step, "band": band_name,
                                          "effective_resolution_wavenumber": eff_res})

                    fig, ax = band_figs[band_name]
                    color = cmap(si / max(1, len(SNAPSHOT_STEPS) - 1))
                    ax.loglog(wavenumbers[1:], power[1:], color=color, lw=1.2,
                              label=f"{var_label} step {step}" if var_label == first_label_plotted else None)

            print(f"  [{exp_name}] {var_label} (~{LEVEL_APPROX_HPA[lvl_idx]}hPa): spectral blur computed "
                  f"for {len(SNAPSHOT_STEPS)} lead times x {len(BANDS)} bands")

        pd.DataFrame(spectrum_rows).to_csv(out_dir / f"spectral_blur_{exp_name}.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(out_dir / f"spectral_blur_summary_{exp_name}.csv", index=False)

    # Surface variables (no level index)
    for sfc_label, sfc_varname in SFC_VARNAMES.items():
        if sfc_varname not in exp_ds.variables or sfc_varname not in ctrl_ds.variables:
            print(f"  [{exp_name}] {sfc_label} (surface): variable unavailable, skipping.")
            continue

        for si, step in enumerate(SNAPSHOT_STEPS):
            exp_field = np.asarray(exp_ds.variables[sfc_varname][step, :, :])
            ctrl_field = np.asarray(ctrl_ds.variables[sfc_varname][step, :, :])
            diff = exp_field - ctrl_field

            for band_name, row in BAND_ROWS.items():
                window_rows = int(WINDOW_DEG / GRID_DEG_LAT)
                power = zonal_power_spectrum(diff, row, window_rows)
                wavenumbers = np.arange(len(power))
                for wn, p in zip(wavenumbers, power):
                    spectrum_rows.append({"variable": sfc_label, "step": step, "band": band_name,
                                           "wavenumber": int(wn), "power": float(p)})
                eff_res = effective_resolution_wavenumber(power)
                summary_rows.append({"variable": sfc_label, "step": step, "band": band_name,
                                      "effective_resolution_wavenumber": eff_res})

        print(f"  [{exp_name}] {sfc_label} (surface): spectral blur computed "
              f"for {len(SNAPSHOT_STEPS)} lead times x {len(BANDS)} bands")

        pd.DataFrame(spectrum_rows).to_csv(out_dir / f"spectral_blur_{exp_name}.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(out_dir / f"spectral_blur_summary_{exp_name}.csv", index=False)

    for band_name, (fig, ax) in band_figs.items():
        ax.set_xlabel("zonal wavenumber (cycles / global ring)")
        ax.set_ylabel("power")
        ax.set_title(f"{exp_name}: zonal power spectrum, band={band_name}\n(experiment - control, color = lead time)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3, which="both")
        fig.savefig(out_dir / f"spectral_blur_map_{exp_name}_{band_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {out_dir / f'spectral_blur_{exp_name}.csv'}")
    print(f"  Saved: {out_dir / f'spectral_blur_summary_{exp_name}.csv'}")
    exp_ds.close()
    ctrl_ds.close()

print("\nDone.")
