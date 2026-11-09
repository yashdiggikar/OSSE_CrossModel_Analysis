# PERTURBATION_EXPERIMENTS — Full File Map

**Status as of 2026-07-25: YES, everything is complete.** All 3 models
(FourCastNetv2, Aurora, Prithvi WxC) × both variables (temperature, wind) =
**6 experiments**, each with all **7 CSV metric types** computed, all bias
`.npy` fields saved, all diagnostic plots rendered, the 3-model cross-model
comparison computed, and paper-ready figures generated. Verified by direct
file count on disk on the date above (not assumed from a script's "Done"
message — every count below was independently re-derived with `find`/`ls`
immediately before this file was written):

| Check | Expected | Found |
|---|---|---|
| CSVs (6 experiments × 7 types + 10 CrossModel) | 52 | **52** |
| PNGs (per-experiment bias/corr/spectral maps + 30 paper_figures + 8 CrossModel) | — | **275** |
| NPYs (bias-map arrays, control/experiment × variable × {mean,day60}) | — | **414** |

This file is the complete directory map — every path below is absolute and
was confirmed to exist at the time of writing. Companion docs: `METHOD_OVERVIEW.md`
(the scientific method), `SESSION_LOG_2026-07-22.md` (the full narrative history
of how everything here was built, bugs found/fixed, and why), and the 3
per-model `IMPLEMENTATION.md` files (perturbation-generation code detail).

All paths below are rooted at `/umbc/rs/pi_halem/users/yashd1/` unless
otherwise absolute.

---

## 1. Top-level documentation

| File | What it is |
|---|---|
| `PERTURBATION_EXPERIMENTS/METHOD_OVERVIEW.md` | The scientific method: how the correlated (spatially + vertically correlated, non-divergent-wind) perturbation is designed, with citations. |
| `PERTURBATION_EXPERIMENTS/CORRELATION_LENGTH_VERIFICATION.md` | Method + per-model implementation for the empirical correlation-length/vertical-correlation verification (updated 2026-07-25 for the full T/U/V/Z/humidity × 300/500/1000 hPa grid). |
| `PERTURBATION_EXPERIMENTS/SPECTRAL_BLURRING_ANALYSIS.md` | Method + per-model implementation for the spectral-blurring diagnostic (same 2026-07-25 update). |
| `PERTURBATION_EXPERIMENTS/CROSS_MODEL_COMPARISON.md` | Method for the 3-model insertion-effectiveness/damping/blurring-rate comparison. |
| `PERTURBATION_EXPERIMENTS/RESULTS_ANALYSIS_ALL_MODELS.md` | **The physical-analysis writeup** — all 3 models, all 6 experiments, every metric type, current version (2026-07-25 rewrite). This is the one to read for findings/conclusions. |
| `PERTURBATION_EXPERIMENTS/RESULTS_ANALYSIS_FCN_AURORA.md` | Superseded (2-model, pre-Prithvi version) — kept for provenance only, not for further use. |
| `PERTURBATION_EXPERIMENTS/SESSION_LOG_2026-07-22.md` | Full chronological session log — every bug found/fixed, every job ID, every decision, across all continuation sessions. Read this first if context is ever lost. |
| `PERTURBATION_EXPERIMENTS/FCN/IMPLEMENTATION.md` | FCN-specific: channel indices, level tables, perturbation-injection code detail. |
| `PERTURBATION_EXPERIMENTS/Aurora/IMPLEMENTATION.md` | Aurora-specific, same scope. |
| `PERTURBATION_EXPERIMENTS/Prithvi/IMPLEMENTATION.md` | Prithvi-specific, same scope. |
| `PERTURBATION_EXPERIMENTS/map.md` | This file. |

---

## 2. Top-level scripts

| File | What it does |
|---|---|
| `PERTURBATION_EXPERIMENTS/run_analysis.slurm` | Generic SLURM batch wrapper: `sbatch run_analysis.slurm <ModelDir> <script.py>`. 64G mem, 4 CPU, 6h walltime, `chip-cpu`/`general`/`medium` QOS. Used for every analysis job in this project. |
| `PERTURBATION_EXPERIMENTS/make_paper_figures.py` | Reads all 6 experiments' CSVs/`.npy` files, produces the 30 files in `paper_figures/` (§6). |
| `PERTURBATION_EXPERIMENTS/compute_cross_model_comparison.py` | Reads all 6 experiments' `global_rmse_*.csv` + `spectral_blur_summary_*.csv`, produces the 18 files in `CrossModel/` (§7). Run this last (depends on per-model outputs already existing). |

---

## 3. Raw model output (the actual 6 forecasts + 3 controls — multi-GB `.nc` files, not duplicated into this directory)

| Model | Experiment | Raw file (absolute path) | Size |
|---|---|---|---|
| FCN | Control | `FourCastNet/forecast_normal_60days_12hr.nc` | 21.9 GB |
| FCN | T-insertion (`t_perturbed_corr`) | `fcn_inference/t_perturbed_corr/forecast_t_perturbed_corr_global_day30_60.nc` | 44.2 GB |
| FCN | UV-insertion (`uv_perturbed_corr`) | `fcn_inference/uv_perturbed_corr/forecast_uv_perturbed_corr_global_day30_60.nc` | 44.7 GB |
| Aurora | Control | `auroraoutput_1year_1stfeb23.nc` | 31.1 GB |
| Aurora | T-insertion (`T_PERTURBED_CORR`) | `CC_aurora/T_PERTURBED_CORR/output/aurora_t_perturbed_corr.nc` | 13.0 GB |
| Aurora | UV-insertion (`UV_PERTURBED_CORR`) | `CC_aurora/UV_PERTURBED_CORR/output/aurora_uv_perturbed_corr.nc` | 16.3 GB |
| Prithvi | Control | `prithviOSSE/run2/output/control_forecast_Jan2022.nc` | 22.9 GB |
| Prithvi | T-insertion (`perturbed_corr_temp_ins`) | `prithviOSSE/perturbed_corr_temp_ins/output/perturbed_corr_temp_osse_forecast_Jan2022.nc` | 22.9 GB |
| Prithvi | UV-insertion (`perturbed_corr_uv_ins`) | `prithviOSSE/perturbed_corr_uv_ins/output/perturbed_corr_uv_osse_forecast_Jan2022.nc` | 22.9 GB |

Each is also reachable from inside `PERTURBATION_EXPERIMENTS/` via a symlink
(browsable without duplicating the multi-GB files):
`{FCN,Aurora,Prithvi}/<experiment>_FULL_DIR` → the real directory above
(e.g. `FCN/t_perturbed_corr_FULL_DIR` → `fcn_inference/t_perturbed_corr`).

*(Note: this table is the COP/noisy-truth experiment set specifically. The
separate main-AMS-paper perfect-truth insertion experiments — 4 models
including GraphCast — live at different paths entirely; see
`Organized_results/file_paths.md` for those, not this file.)*

---

## 4. Per-model analysis scripts (identical 5-script pattern × 3 models)

Each of `PERTURBATION_EXPERIMENTS/{FCN,Aurora,Prithvi}/` contains the same 5
scripts (source of truth — edit here, not the CSVs):

- `compute_global_rmse_corr.py`
- `compute_latband_rmse_corr.py`
- `compute_spatial_bias_corr.py`
- `compute_correlation_length_corr.py`
- `compute_spectral_blur_corr.py`

All 5 compute the full T/U/V/Z/humidity × 300/500/1000 hPa grid (+ surface
T2M/U10(M)/V10(M)) for **both** experiments per model, regardless of which
variable family was directly inserted (cross-variable coupling included).

---

## 5. Per-experiment outputs — all 6 experiments, full paths

### 5.1 FCN — `PERTURBATION_EXPERIMENTS/FCN/t_perturbed_corr/`

| File | Rows/content |
|---|---|
| `global_rmse_t_perturbed_corr.csv` | Cosine-lat-weighted RMSE vs. ERA5, control + experiment, full 240-step time series, all 18 variables. |
| `latband_rmse_t_perturbed_corr.csv` | Same RMSE, at 4 individual latitude slices (2N/14N/40N/60N, not averaged bands). |
| `spatial_bias_t_perturbed_corr.csv` | Mean signed bias summary (global + 3 lat bands), control vs. experiment. |
| `bias_{control,experiment}_<VAR>.npy` (30 files) | Full 2D bias field, day 30–60 mean, per variable. |
| `bias_{control,experiment}_<VAR>_day60.npy` (30 files) | Same, day-60-only snapshot (added 2026-07-25). |
| `spatial_bias_map_t_perturbed_corr_<VAR>.png` (15 files) | 3-panel (control/experiment/diff) bias map, day 30–60 mean, per variable. |
| `spatial_bias_map_t_perturbed_corr_<VAR>_day60.png` (15 files) | Same, day-60 snapshot. |
| `corr_length_horizontal_t_perturbed_corr.csv` | Empirical zonal/meridional e-folding length vs. 250 km target, 3 bands × 3 levels (300/500/1000 hPa) × 5 variables + surface. |
| `corr_length_vertical_t_perturbed_corr.csv` | Level×level empirical vs. theoretical correlation matrix, all 13 native levels, 5 variables. |
| `corr_length_horizontal_map_t_perturbed_corr.png` | Lag-correlation curves at 500 hPa vs. theoretical Gaussian. |
| `corr_length_vertical_map_t_perturbed_corr.png` | Empirical/theoretical/difference heatmaps. |
| `spectral_blur_t_perturbed_corr.csv` | Full 1D zonal power spectrum, 7 lead times × 3 bands × 18 variables. |
| `spectral_blur_summary_t_perturbed_corr.csv` | Effective-resolution wavenumber summary of the above. |
| `spectral_blur_map_t_perturbed_corr_<band>.png` (3 files) | Log-log power spectrum, color-graded by lead time, one per band. |

**`PERTURBATION_EXPERIMENTS/FCN/uv_perturbed_corr/`** — identical file set,
`uv_perturbed_corr` in place of `t_perturbed_corr` in every filename.

### 5.2 Aurora — `PERTURBATION_EXPERIMENTS/Aurora/T_PERTURBED_CORR/`

Same 7-CSV / bias-map / plot pattern as §5.1, filenames use
`T_PERTURBED_CORR` (all-caps — an existing naming inconsistency from how
Aurora's experiments were originally set up, not introduced by any script
here). Variable count is slightly higher than FCN's (72 vs. 60 `.npy`, 41 vs.
35 `.png`) because Aurora's humidity variable (`Q`, specific humidity) has
real ERA5 truth data at every level, unlike FCN's `R` (relative humidity),
which is absent from this project's ERA5 archive at every level — a known,
permanent data-archive gap, not a bug (see `RESULTS_ANALYSIS_ALL_MODELS.md`
§2.1 and §7).

**`PERTURBATION_EXPERIMENTS/Aurora/UV_PERTURBED_CORR/`** — identical file
set, `UV_PERTURBED_CORR` in place of `T_PERTURBED_CORR`.

### 5.3 Prithvi — `PERTURBATION_EXPERIMENTS/Prithvi/perturbed_corr_temp_ins/`

Same pattern again, filenames use `perturbed_corr_temp_ins`. Prithvi's
variable/level counts are the largest of the three (74–76 `.npy`, 42–43
`.png`) because it has both Z (as `H`, geopotential height) and Q (as `QV`,
specific humidity) with real truth data (MERRA-2), and its vertical check
spans 14 native levels rather than FCN/Aurora's 13.

**`PERTURBATION_EXPERIMENTS/Prithvi/perturbed_corr_uv_ins/`** — identical
file set, `perturbed_corr_uv_ins` in place of `perturbed_corr_temp_ins`.

---

## 6. Paper-ready figures — `PERTURBATION_EXPERIMENTS/paper_figures/`

30 files total (5 figure types × 2 experiments × 3 models), produced by
`make_paper_figures.py`:

| Model dir | Files (× `{t_perturbed_corr\|uv_perturbed_corr}` or model-equivalent experiment name) |
|---|---|
| `paper_figures/FCN/` | `global_rmse_timeseries_<exp>.png`, `latband_rmse_timeseries_<exp>.png`, `spatial_bias_map_<exp>.png`, `spatial_bias_map_<exp>_day60.png`, `corr_length_summary_<exp>.png` — 10 files (5 × 2 experiments) |
| `paper_figures/Aurora/` | Same 5 types, `<exp>` = `T_PERTURBED_CORR`/`UV_PERTURBED_CORR` — 10 files |
| `paper_figures/Prithvi/` | Same 5 types, `<exp>` = `perturbed_corr_temp_ins`/`perturbed_corr_uv_ins` — 10 files |

`global_rmse_timeseries`/`latband_rmse_timeseries` show only the
directly-inserted variable family at all 3 levels (T300/500/1000+T2M for
temp; U/V×300/500/1000+surface for wind) — deliberately not the full
18-variable cross-variable grid, kept readable for a paper figure.
`corr_length_summary` shows the 500 hPa representative level only, filtered
from the full 3-level CSV.

---

## 7. Cross-model comparison — `PERTURBATION_EXPERIMENTS/CrossModel/`

18 files, produced by `compute_cross_model_comparison.py`, spanning all 3
models in single tables/plots:

| File | Content |
|---|---|
| `cross_model_effectiveness_temp.csv` / `.png` | Insertion effectiveness `(ctrl_rmse−exp_rmse)/ctrl_rmse`, all 3 models, per lead time, for `temp_upper_air`, `temp_surface`, `geopotential_upper_air`, `humidity_upper_air` categories. |
| `cross_model_effectiveness_uv.csv` / `.png` | Same, for `wind_u_upper_air`, `wind_v_upper_air`, `wind_u_surface`, `wind_v_surface`, `geopotential_upper_air`, `humidity_upper_air`. |
| `cross_model_damping_temp.csv` | First-vs-last effectiveness + damping ratio per model/category, temp experiment. |
| `cross_model_damping_uv.csv` | Same, wind experiment. |
| `cross_model_blurring_{temp,uv}_{equator_5N,midlat_45N,highlat_70N}.csv` / `.png` (6 CSV + 6 PNG) | Effective-resolution-wavenumber trajectory, all 3 models, per band. |

---

## 8. Logs — `PERTURBATION_EXPERIMENTS/logs/`

184 files (`.out`/`.err` pairs from every `sbatch run_analysis.slurm` job run
across this project's entire history), named
`perturb_analysis_<jobid>.{out,err}`. ~206 KB total. Kept for provenance/
debugging — `SESSION_LOG_2026-07-22.md` cross-references specific job IDs to
specific entries here throughout its narrative.

---

## 9. What is intentionally NOT in this map

- The main AMS manuscript's own (perfect-truth, 4-model-including-GraphCast)
  insertion experiments — a completely separate experiment set at different
  paths. See `Organized_results/file_paths.md` and `Organized_results/map.md`
  for that inventory; this file covers only the COP (noisy/correlated-
  perturbation) side-project under `PERTURBATION_EXPERIMENTS/`.
- `__pycache__/` directories under each model dir — Python bytecode cache,
  not a data product, regenerates automatically, safe to ignore or delete.
