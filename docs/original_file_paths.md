# Original File Paths

Every raw prediction file and truth/reference dataset that feeds into the
values in this directory. None of these are copied into `Organized_results/`
or `Organized_plots/` (they range 5–92 GB each) — only the computed
`values/`/`plots/` derived from them live here. This file is the complete
dependency chain if you need to trace any number all the way back to its
source, or need the raw files to recompute something rather than just
replot it (`Organized_results/` alone cannot recompute — see `map.md` and
`replotting_instructions.md` for what replotting from data already here
does and doesn't need).

All paths are absolute, rooted at `/umbc/rs/pi_halem/users/yashd1/`.

---

## 1. Model prediction files

### Aurora (Microsoft, 0.25°)

| Experiment | Path | Size |
|---|---|---|
| Control | `auroraoutput_1year_1stfeb23.nc` | 31.1 GB |
| T-Ins | `Aurora_Pred_Files/auroraoutput_temp_inserted.nc` | 34.3 GB |
| Wind(UV)-Ins | `Aurora_Pred_Files/auroraoutput_uv_inserted.nc` | 34.3 GB |
| Geo(Z)-Ins | `Aurora_Pred_Files/auroraoutput_geopotential_inserted.nc` | 34.3 GB |
| T+UV (TUV synergy study only) | `CC_aurora/TUV_GLOBAL/output/aurora_tuv_global.nc` | 14.4 GB |

IC: 2023-02-01 12:00 UTC. 120 steps × 12h = 60 days. Grid 720×1440 (no
South Pole row).

⚠️ `auroraoutput_60day_12hr.nc` also exists at top level but is **truncated**
(18 GB vs. the expected ~34 GB) and is not referenced by any script — the
`Control` row above (`auroraoutput_1year_1stfeb23.nc`, using its first 120
steps) is what's actually used everywhere.

### FourCastNet v2 (NVIDIA)

| Experiment | Path | Size |
|---|---|---|
| Control | `FourCastNet/forecast_normal_60days_12hr.nc` | 21.9 GB |
| T-Ins | `FourCastNet/Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc` | 36.4 GB |
| Wind-Ins | `FourCastNet/Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc` | 36.4 GB |
| Z-Ins | `FourCastNet/Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc` | 36.4 GB |
| T+UV (TUV synergy study only) | `fcn_inference/tuv_global/forecast_tuv_global_day30_60.nc` | 40.0 GB |

IC: 2023-02-01 00:00 UTC. 120 steps × 12h = 60 days. Grid 721×1440 (both poles).

⚠️ `coupling/` does **not** read any of the raw files above — it reads
pre-computed per-level RMSE instead (see §3).

⚠️ `FourCastNet/Prediction_Files_Fcn/t_sp_ins/` (surface-pressure insertion)
also exists but is **not used by any script** in this study — a run that
exists on disk but was never analyzed.

### GraphCast-Small (Google DeepMind)

| Experiment | Path | Size |
|---|---|---|
| Control (STRICT) | `gc_pred_files/gc_sm_240steps_STRICT.nc` | 5.19 GB |
| T-Ins | `gc_pred_files/gc_sm_240steps_temp_inj.nc` | 5.19 GB |
| UV-Ins | `gc_pred_files/gc_sm_240steps_uv_inj.nc` | 5.19 GB |
| Z-Ins | `gc_pred_files/gc_sm_240steps_z_inj.nc` | 5.19 GB |
| T+UV | *(not run — GraphCast is not part of the TUV synergy study)* | — |

IC: 2022-01-01 06:00 UTC. 240 steps × 6h = 60 days. Grid 181×360 (1°,
ascending latitude — opposite orientation from the other 3 models).

Note the `_inj` (injection) suffix in these filenames vs. `_ins`
(insertion) used by the other 3 models — same meaning, different naming
convention from whoever generated each model's files. Internally, every
script's run-keys still use `_ins` (`t_ins`, `uv_ins`, `z_ins`) for
consistency with the rest of the study.

### Prithvi WxC (NASA/IBM)

| Experiment | Path | Size |
|---|---|---|
| Control | `prithviOSSE/run2/output/control_forecast_Jan2022.nc` | 22.9 GB |
| T-Ins (temp_ins) | `prithviOSSE/t_sp_ins/output/t_sp_osse_forecast_Jan2022.nc` | 22.5 GB |
| Wind-Ins | `prithviOSSE/uv_ins_run/output/uv_osse_forecast_Jan2022.nc` | 22.2 GB |
| Geo-Ins | `prithviOSSE/geo_ins_run/output/geo_osse_forecast_Jan2022.nc` | 22.6 GB |
| T+UV (TUV synergy study) | `prithviOSSE/t_and_uv/output/t_and_uv_osse_forecast_Jan2022.nc` | 21.8 GB |

IC: 2022-01-01 00:00 UTC. 240 steps × 6h = 60 days. Grid 360×576 (0.5°,
ascending latitude), 14 native model levels (not standard isobaric levels
— see `replotting_instructions.md` §1.4 for the level-to-pressure mapping).

⚠️ Two other files exist and are **not** the control, despite easy
confusion:
- `prithviOSSE/run2/output/osse_forecast_Jan2022.nc` — this is actually
  a **T-insertion** run (`--inject-temp` output), historically mislabeled
  as "Control" in several scripts — see `map.md` §5 Bug 1 for the full
  story. Fully fixed as of 2026-07-14; nothing in `Organized_results/`
  references it as Control anymore.
- `prithvi/output/forecast_60day_Jan2022.nc` — an older, differently-shaped
  (361 vs. 360 latitude rows) control run from a pre-OSSE pipeline, used
  as a stand-in before the real control file existed. Also no longer
  referenced by anything in `Organized_results/`.

---

## 2. Truth / reference data (what predictions are scored against)

| Model(s) | Source | Path |
|---|---|---|
| Aurora & FCN | ERA5 pressure levels, Feb–Apr 2023 | `AURORA_DATA/rmse_data/era5_pressure_2023_{02,03,04}.grib` |
| Aurora & FCN | ERA5 surface, Feb–Apr 2023 | `AURORA_DATA/rmse_data/era5_surface_2023_{02,03,04}.grib` |
| GraphCast | ERA5 pressure levels, Jan–Mar 2022 | `era5_vs_cams/era5_2022/era5_pressure_2022_{01,02,03}.grib` |
| GraphCast | ERA5 10m wind (surface), 2016–2025 archive | `ALL_ERA5/2016_2025_surface/era5_2016_2025.grib` (92 GB — the file behind the wind10 fix, `map.md` §5 Bug 4) |
| Prithvi | MERRA-2 pressure levels, daily | `prithvi/data/merra2/MERRA_pres_YYYYMMDD.nc` (140 daily files, 2022-01-01 to ~2022-03-11 — MERRA-2 truth coverage cuts off 2022-03-11 23:00 UTC) |
| Prithvi | MERRA-2 surface, daily | `prithvi/data/merra2/MERRA2_sfc_YYYYMMDD.nc` |

All ERA5 truth is loaded via `cfgrib`; all GraphCast/Prithvi truth
comparisons involve a regrid step (GraphCast: ERA5 0.25°→1° bilinear;
Prithvi: works natively on MERRA-2's own 0.5° grid, no regrid needed).

---

## 3. Exception — FCN `coupling/` uses a different, pre-computed source

Not the raw `.nc` files in §1 — instead, per-level RMSE already computed
elsewhere:

```
fcn_rmse/FCNv2/Control/rmse_all_vars.csv
fcn_rmse/FCNv2/temp_ins/rmse_all_vars.csv
fcn_rmse/FCNv2/wind_ins/rmse_all_vars.csv
fcn_rmse/FCNv2/geo_ins/rmse_all_vars.csv
```

---

## 4. Where each of these is actually referenced

For the exact script(s) that reference each path (useful if you need to
verify or recompute something), see `map.md` §3's per-metric `scripts/`
listings — every script's `FC_FILES`/`GC_FILES`/`CTRL_FILE`/etc. dict
points at the paths above. All Prithvi scripts were audited 2026-07-15 to
confirm zero remaining references to the two mislabeled files noted in §1.
