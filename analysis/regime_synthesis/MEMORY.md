# Session Memory — Multi-Model Insertion Analysis (Aurora / FCNv2 / GraphCast-S / PrithviWxC)

This file documents everything investigated, fixed, and generated during this working
session. It is meant to be read cold by a future session (human or AI) picking this
project back up — every file path, bug, fix, and script is recorded in full so nothing
has to be re-derived from scratch.

Session dates: 2026-06-17 → 2026-06-18.

---

## 1. Background — directory layout discovered

`/umbc/rs/pi_halem/users/yashd1/common/` hosts cross-model comparison analysis for an
Aurora weather-model data-assimilation study. Four AI weather models are compared:
**Aurora, FourCastNet V2 (FCN), GraphCast-Small (GC), PrithviWxC**. The study inserts
ERA5/MERRA-2 truth for individual variables (Temperature, Wind/UV, Geopotential/Z) into
otherwise free-running 60-day forecasts and measures the RMSE impact, including on
variables that were *not* inserted ("cross-variable" effects), and the effect of
inserting **T+UV together** ("synergy").

### 1.1 `common/tcc/`, `common/cloud2/`, `common/terrain/`, `common/terrain2/`
Four parallel experiment sets: temp-insertion vs wind-insertion, crossed with
cloud-regime vs terrain-regime stratification.

| Dir | Experiment | Script | Regime truth |
|---|---|---|---|
| `tcc/` | T-insertion | `fig5_75_76_temp_ins_cloud_regime.py` | `ALL_ERA5/tcc/era5_total_cloud_cover_{2023_Feb_Apr,2022_jan_feb}_6hourly.nc` |
| `cloud2/` | UV-insertion | `fig_cloud2_uv_ins.py` | same TCC files |
| `terrain/` | T-insertion | `fig_terrain_temp_ins.py` | `prithvi/data/merra2/MERRA2_sfc_20220103.nc` (land/sea mask) |
| `terrain2/` | UV-insertion | `fig_terrain2_uv_ins.py` | same MERRA_SFC |

File paths used by all four (`BASE = /umbc/rs/pi_halem/users/yashd1`):

```
AUR_CTRL   = BASE/auroraoutput_1year_1stfeb23.nc
AUR_T_INS  = BASE/Aurora_Pred_Files/auroraoutput_temp_inserted.nc
AUR_UV_INS = BASE/Aurora_Pred_Files/auroraoutput_uv_inserted.nc
FCN_CTRL   = BASE/FourCastNet/forecast_normal_60days_12hr.nc
FCN_T_INS  = BASE/FourCastNet/Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc
FCN_UV_INS = BASE/FourCastNet/Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc
GC_CTRL    = BASE/gc_pred_files/gc_sm_240steps_STRICT.nc
GC_T_INS   = BASE/gc_pred_files/gc_sm_240steps_temp_inj.nc
GC_UV_INS  = BASE/gc_pred_files/gc_sm_240steps_uv_inj.nc
PRI_CTRL   = BASE/prithvi/output/forecast_60day_Jan2022.nc        (tcc/terrain — correct)
PRI_T_INS  = BASE/prithviOSSE/run2/output/osse_forecast_Jan2022.nc
PRI_CTRL*  = BASE/prithviOSSE/run2/output/osse_forecast_Jan2022.nc (cloud2/terrain2 — ⚠ MISMATCHED, see §3)
PRI_UV_INS = BASE/prithviOSSE/uv_ins_run/output/uv_osse_forecast_Jan2022.nc
MERRA_DIR  = BASE/prithvi/data/merra2   (MERRA_pres_{date}.nc, used for Prithvi truth)
```

**RMSE methodology in these 4 scripts:** cosine-latitude weighted, **regime-masked**
(only grid points belonging to a given cloud-cover bin or terrain class are averaged
into that regime's RMSE) — *not* a global RMSE. See `rmse_by_regime()` /
`compute_rmse_series()` in `tcc/fig5_75_76_temp_ins_cloud_regime.py` lines ~122-151.

### 1.2 `common/lat_bands/`, `common/nrmse_plots/`, `common/tuv_plots/` (pre-existing, top-level)
- `common/plot_latband_latslice_updated.py` — **original** lat-band/lat-slice script.
  Produces **one figure per MODEL** (not per experiment), with **all variables shown
  including the inserted one itself**, and bars = the 3 insertion runs (not models).
  This was judged not to match the desired analysis design (see §2) but was **left
  untouched** — its 8 PNGs (`Figure_lat{band,slice}_{Aurora,FourCastNet,GraphCast_S,Prithvi}_summary.png`)
  still exist in `common/lat_bands/`, dated 2026-06-08.
- `common/nrmse_plots/plot_nrmse_improvements.py` — one figure **per experiment**
  (T/UV/Z insertion), global NRMSE % improvement, reads from
  `{FourCastNet,Aurora_Pred_Files,gc_pred_files,prithvi_new}/nrmse/values/*.csv`.
- `common/tuv_plots/compute_tuv_rmse.py` + `plot_tuv_analysis.py` — the **T+UV combined
  ("synergy") study**: does inserting T and UV together beat doing either alone? Global
  (not regime-masked) cosine-lat-weighted RMSE. Produces Fig 5.24–5.29 + Table 5.7.
  **GraphCast is not part of this study** (no combined T+UV run exists for it).

  File paths (`compute_tuv_rmse.py`):
  ```
  FCN_TUV  = /umbc/rs/pi_halem/users/yashd1/fcn_inference/tuv_global/forecast_tuv_global_day30_60.nc
             (despite the filename, this file actually covers the FULL 60-day IC
              2023-02-01 00:00 → 2023-04-01 18:00, 240 steps @ 6h — verified via h5py)
  FCN_CTRL = FourCastNet/forecast_normal_60days_12hr.nc
  FCN_TINS = FourCastNet/Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc
  FCN_UINS = FourCastNet/Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc
  AUR_TUV  = CC_aurora/TUV_GLOBAL/output/aurora_tuv_global.nc
  AUR_CTRL = auroraoutput_1year_1stfeb23.nc
  AUR_TINS = Aurora_Pred_Files/auroraoutput_temp_inserted.nc
  AUR_UINS = Aurora_Pred_Files/auroraoutput_uv_inserted.nc
  PRI_TUV  = prithviOSSE/t_and_uv/output/t_and_uv_osse_forecast_Jan2022.nc
  PRI_CTRL = prithviOSSE/run2/output/control_forecast_Jan2022.nc   (FIXED this session, see §3)
  PRI_TINS = prithviOSSE/run2/output/osse_forecast_Jan2022.nc
  PRI_UINS = prithviOSSE/uv_ins_run/output/uv_osse_forecast_Jan2022.nc
  ```
  Cache dir: `common/tuv_plots/cache/{model}_{run}_rmse.csv` (run ∈ ctrl/tins/uins/tuv),
  plus `{model}_synergy_{var}.npy`, `{model}_impr_tuv_{var}.npy` (spatial fields,
  mature-phase day 40-60).

  Variables per model: Aurora/FCN → `t500,u500,v500,z300,z500,z1000,{q850,r850},{q500,r500}`
  (Aurora uses q = specific humidity; FCN uses r = relative humidity, derived from
  ERA5 q+T via Clausius-Clapeyron in `era5_rh()`); Prithvi →
  `T_lv7,U_lv7,V_lv7,H_lv5,H_lv7,H_lv12,QV_lv7,QV_lv10` (lv-index = **Prithvi's own
  ascending position**, not a literal pressure level — see §3 Bug B).

### 1.3 `prithvi_new/` — Prithvi's own lat-band/lat-slice pipeline
- `prithvi_new/lats/section_5_7_prithvi_latband_analysis.py` → writes
  `prithvi_new/lats/values/latband_rmse_{var}_{run}.csv`
- `prithvi_new/lat_bands/section_5_7_prithvi_latslice_analysis.py` → writes
  `prithvi_new/lat_bands/values/latslice_rmse_{var}_{run}.csv`
- Both use `FC_FILES = {"Control":…, "temp_ins":…, "wind_ins":…, "geo_ins":…}`,
  variables `T2M, SLP, T_lev63, T_lev51, H_lev51, U_lev45`, MERRA-2 truth via
  `load_merra_pres_all()` which **correctly** remaps Prithvi level → MERRA index via
  `MERRA_LEV_DESC.index(lev_int)` (this is the reference-correct implementation that
  `compute_tuv_rmse.py` was missing — see Bug B).
- Cosine-lat weighted, **band/region-masked** (Tropics/Midlatitudes/Polar or lat-slices
  2°N/14°N/40°N/60°N), mature phase = steps ≥120 (day 30+), 240 steps × 6h, IC
  2022-01-01 00:00.

### 1.4 `/umbc/rs/pi_halem/users/yashd1/Multimodel-study/` — separate data source
User-maintained directory (synced from a Mac, has `.DS_Store` files), documented by its
own `map.txt`. Used as the source for Control/T-ins/UV-ins curves in the final
`tuv_final/` Z500/Q850 plots (§5).

```
FCNv2/{Control,temp_ins,wind_ins,geo_ins}/rmse_all_vars.csv
  - row-indexed: pd.read_csv(path, index_col=0)
  - rows = var names (t500,u300,z1000,r850,...); cols = datetime strings, 12h step,
    120 cols, IC = 2023-02-01 00:00 (lead day 0 at first column)

Aurora/{control,temp_ins,wind_ins,geo_ins}/aurora_rmse_all_vars.csv
  - same row-indexed format; IC = 2023-02-01 12:00 (lead day starts at 0.5,
    i.e. init = first column − 12h)
  - NOTE: Aurora/control/ was MISSING when first checked this session — user added
    aurora_rmse_all_vars.csv (150513 bytes) to Multimodel-study/Aurora/control/ mid-session.

PrithviWxC/{control,temp_ins,wind_ins,geo_ins}/rmse_upper.csv   (rmse_surface.csv also
  present per dir but NOT used for pressure-level variables)
  - column-indexed: lead_hour column (0,6,...,1440, 241 rows) + VAR_levNN columns
    e.g. T_lev51, H_lev51, U_lev45, QV_lev63
  - Level→hPa map: lev72=1000, lev71=975, lev68=925, lev63=850, lev56=700, lev53=600,
    lev51=500, lev48=400, lev45=300, lev44=250, lev43=200, lev41=150, lev39=100, lev34=50
  - Var prefixes: T=temperature, U=zonal wind, V=meridional wind,
    H=geopotential height [m] (×9.80665 for m²/s² if needed), QV=specific humidity

GraphCast/        — NOT used (only 3 models compared in tuv_final/ outputs:
                    FCN, Aurora, Prithvi)
CAMS_aurora/, CAMS_Prithvi/  — exist but explicitly excluded from the 4-model study
```

---

## 2. Redesign: experiment-centric cross-variable lat-band/lat-slice charts

**User's design spec:** instead of one chart per model showing all variables (including
the inserted one) with insertion-type as the bar color — which is what
`plot_latband_latslice_updated.py` does — build **one chart per insertion experiment**
(T-Ins / Wind-Ins / Geo-Ins), showing only the **cross-variable targets** (variables that
were *not* inserted), with **4 bars per variable group, one per model**.

Cross-variable exclusion rule:
- T-Ins experiment → show **U500, V500, Z500** (exclude T500)
- Wind-Ins experiment → show **T500, Z500** (exclude U500, V500)
- Geo-Ins experiment → show **T500, U500, V500** (exclude Z500)

**New script:** `/umbc/rs/pi_halem/users/yashd1/common/plot_latband_by_experiment.py`

- Reuses the same underlying per-model/per-run CSVs as the original script
  (`{Model}/lats/values/*.csv` for latband, `{Model}/lat_bands/values/*.csv` for
  latslice — note the directory-naming is swapped/confusing: "lats" = latband data,
  "lat_bands" = latslice data, inherited from the original pipeline).
- `MODEL_ORDER = ["Aurora", "FourCastNet", "GraphCast-S", "Prithvi"]`
- `MODEL_COLORS` (consistent across every figure):
  Aurora `#2E86AB` (teal-blue), FourCastNet `#E94B3C` (coral-red),
  GraphCast-S `#6A4C93` (violet), Prithvi `#1B998B` (emerald)
- `EXP_ACCENT` (title/strip color per experiment, matches original semantic convention):
  T-Ins `#e74c3c` (red), Wind-Ins `#3498db` (blue), Geo-Ins `#f39c12` (orange)
- Per-model config maps canonical var names → each model's own column key, and
  canonical experiment names → each model's own run-name (since naming differs:
  Aurora/GC use `t_ins/uv_ins/z_ins`; FCN uses `t_ins/wind_ins/z_ins`; Prithvi uses
  `temp_ins/wind_ins/geo_ins`; var keys: `t500/u500/v500/z500` for Aurora/FCN/GC vs
  `T_lev51/U_lev51/V_lev51/H_lev51` for Prithvi).
- Subplot grid: rows = phase (Adjustment Days 0-30 / Mature Days 30-60), columns =
  region. Region tick labels — **changed mid-session** per user request from category
  names to literal degree ranges for the "band" analysis:
  `REGION_TICKS["band"] = ["0°–30°", "30°–60°", "60°–90°"]` (was
  `["Tropics","Midlatitudes","Polar"]` — only the **display** label changed; the
  underlying `band_cols` used to look up actual CSV columns are still the literal
  strings `"Tropics"/"Midlatitudes"/"Polar"` since those are real column headers in the
  data). `REGION_TICKS["slice"] = ["2°N","14°N","40°N","60°N"]` (unchanged, already
  numeric).
- Output directory: **`/umbc/rs/pi_halem/users/yashd1/common/lats_FINAL/`** (originally
  written to `common/lat_bands/`, moved to a dedicated `lats_FINAL/` dir later in the
  session so as not to clobber/confuse with the original 8 model-centric PNGs).

**Bug found & fixed — label/tick overlap:** clipped (out-of-range, |value|>20%) negative
bar value labels were rotated 90° and anchored with `va="top"`, which makes matplotlib
grow the text **downward** from the anchor point — for clipped bars anchored near the
y-axis floor (`YLIM[0]+2 = -18`), the text spilled past the axis frame and visually
merged with the x-tick region labels below (e.g. `"-50%*"` and `"-22%*"` overlapping
`"2°N"`). **Fix:** all negative-value labels now anchor with `va="bottom"` (grow
**upward**, away from the tick labels) at `ypos = max(display - 0.5, YLIM[0] + 6)`, plus
added `ax.tick_params(axis="x", pad=7)` for extra buffer. Verified via pixel-cropped
zoom on the exact panel that had the collision (Wind-Ins T500, 2°N, mature phase) —
confirmed fixed.

**Outputs — 12 files in `common/lats_FINAL/`:**
```
Figure_latband_T_Ins_crossvar.png            (full: adjustment + mature rows)
Figure_latband_Wind_Ins_crossvar.png
Figure_latband_Geo_Ins_crossvar.png
Figure_latslice_T_Ins_crossvar.png
Figure_latslice_Wind_Ins_crossvar.png
Figure_latslice_Geo_Ins_crossvar.png
Figure_latband_T_Ins_crossvar_mature.png     (mature-phase-only: single row, added later)
Figure_latband_Wind_Ins_crossvar_mature.png
Figure_latband_Geo_Ins_crossvar_mature.png
Figure_latslice_T_Ins_crossvar_mature.png
Figure_latslice_Wind_Ins_crossvar_mature.png
Figure_latslice_Geo_Ins_crossvar_mature.png
```
The mature-only variants were produced by parameterizing `make_experiment_figure()`
with a `phases=` argument (`MATURE_ONLY_PHASE = [("mat", "Mature Phase (Days 30–60)")]`)
instead of the default 2-row `PHASE_INFO`; `plt.subplots(..., squeeze=False)` makes the
function correctly handle both 1-row and 2-row layouts without special-casing.

---

## 3. Bugs found in the Prithvi data pipeline (the big one)

### Bug A — Contaminated "Control" file in `prithvi_new` lat-band/lat-slice analysis
**Files:** `prithvi_new/lats/section_5_7_prithvi_latband_analysis.py` (line 51),
`prithvi_new/lat_bands/section_5_7_prithvi_latslice_analysis.py` (line 48).

```python
FC_FILES = {
    "Control":  Path(".../prithviOSSE/run2/output/osse_forecast_Jan2022.nc"),  # ← WRONG
    "temp_ins": Path(".../prithviOSSE/t_sp_ins/output/t_sp_osse_forecast_Jan2022.nc"),
    "wind_ins": Path(".../prithviOSSE/uv_ins_run/output/uv_osse_forecast_Jan2022.nc"),
    "geo_ins":  Path(".../prithviOSSE/geo_ins_run/output/geo_osse_forecast_Jan2022.nc"),
}
```

`osse_forecast_Jan2022.nc` is documented in `prithviOSSE/run2/run_forecast_osse.py`'s own
docstring as the **`--inject-temp` output** — i.e. it has MERRA-2 truth temperature
**injected starting at step 121 (~day 30)**. Using it as "Control" means the "control"
RMSE for T_lev51/H_lev51 etc. collapses to near-zero in the mature phase (because
temperature was literally overwritten with the truth being scored against), making any
comparison run (wind_ins, geo_ins) look catastrophically worse than it really is:
e.g. observed `T_lev51` mature RMSE: Control=0.08K vs wind_ins=5.6K →
`(0.08-5.6)/0.08*100 ≈ -6875%` "improvement" (nonsensical).

**Fix:**
1. The genuine un-injected baseline (`control_forecast_Jan2022.nc`, produced by the same
   script **without** `--inject-temp`) **did not exist on disk** — `run2/output/`
   contained only `osse_forecast_Jan2022.nc` (22.5GB).
2. Created `/umbc/rs/pi_halem/users/yashd1/prithviOSSE/run2/forecast_control.slurm`
   (copy of the existing `forecast_osse.slurm`, GPU partition, L40S, 24h walltime, same
   venv/env-var setup, but calls `python run2/run_forecast_osse.py --truth-source merra2`
   **without** `--inject-temp`).
3. User submitted via `sbatch`; ran as **job 189094**, started 13:38, finished 20:45:43
   (~7 hours), producing `prithviOSSE/run2/output/control_forecast_Jan2022.nc` (22.9GB,
   240 steps, confirmed log line `"Temperature injection DISABLED — pure control run"`).
4. Patched both `section_5_7_*` scripts' `"Control"` path to point at the new file.
5. Reran both scripts (`python prithvi_new/lats/section_5_7_prithvi_latband_analysis.py`,
   `python prithvi_new/lat_bands/section_5_7_prithvi_latslice_analysis.py`) — regenerated
   all of `prithvi_new/lats/values/*.csv` and `prithvi_new/lat_bands/values/*.csv`
   (confirmed via log: `T_lev51` Control mature RMSE now `4.88/11.33/14.83` K
   [Tropics/Midlat/Polar] instead of the broken `0.08`).

### Bug B — Prithvi/MERRA level-index mismatch in `compute_tuv_rmse.py`
**File:** `common/tuv_plots/compute_tuv_rmse.py`, function `load_merra2()`
(originally lines 79-101) and the `PRI_VARS` loop (originally lines 280-313).

Prithvi's own NetCDF files store 14 pressure levels **ascending** by level-ID:
`LEVELS_ASC = [34, 39, 41, 43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72]`
(e.g. position 7 = level "51" = 532 hPa — this is what `"H_lv7"` etc. actually mean:
**Prithvi's own ascending array position**, not a literal pressure value).

MERRA-2 pressure files store the *same* 14 levels **descending**:
`MERRA_LEV_DESC = [72, 71, 68, 63, 56, 53, 51, 48, 45, 44, 43, 41, 39, 34]`.

The buggy code reused the Prithvi position (e.g. `7`) directly as the MERRA array index:
```python
def load_merra2(mvar, lev_idx, target_times, ...):
    ...
    result[k] = f[mvar][idx, lev_idx, -nlat:, :nlon]   # lev_idx=7 used as-is on MERRA
```
But MERRA position 7 is level **48** (424 hPa), not level **51** (532 hPa) — so truth
was being read one level too high for every `T_lv7/U_lv7/V_lv7/H_lv7/QV_lv7` comparison
(and similarly wrong for `H_lv5`→pos8 vs correct, `H_lv12`→pos1 vs correct, `QV_lv10`→pos3
vs correct). The reference-correct implementation already existed in
`prithvi_new/lats/section_5_7_prithvi_latband_analysis.py`
(`lev_idx = MERRA_LEV_DESC.index(lev_int)`), which this script never used.

Climatological height differs by hundreds to ~2000m between adjacent levels, so this
produced a large, roughly **constant** (not lead-time-growing) RMSE bias — diagnosed by
noticing Prithvi's `H_lv7` "Control" RMSE was flat at ~727-2364m for the *entire*
60-day series instead of starting near 0 and growing with lead time like every other
variable/model.

**Fix:** added
```python
def pri_idx_to_merra_idx(pri_lev_idx):
    return MERRA_LEV_DESC.index(LEVELS_ASC[pri_lev_idx])
```
and changed the call site to `load_merra2(mvar, pri_idx_to_merra_idx(lidx), pri_times)`.
Applied to `compute_tuv_rmse.py` (for future full reruns) and to a new standalone script
(see Bug C below).

### Bug C — Corrupted `PRI_CTRL` file in `compute_tuv_rmse.py`
**File:** `common/tuv_plots/compute_tuv_rmse.py`, line ~268 (before fix):
```python
PRI_CTRL = Path("/umbc/rs/pi_halem/users/yashd1/prithvi/output/forecast_60day_Jan2022.nc")
```
This file is explicitly documented as corrupted in
`prithviOSSE/run2/run_forecast.py`'s docstring: *"Fix vs scripts/run_forecast.py (job
149657): ... so the corrected NC does not overwrite the archived (corrupted)
output/forecast_60day_Jan2022.nc."* Verified directly via h5py:

| File | `H` variable shape |
|---|---|
| `prithvi/output/forecast_60day_Jan2022.nc` (old, buggy) | `(240, 14, **361**, 576)` |
| `prithviOSSE/run2/output/control_forecast_Jan2022.nc` (new, correct) | `(240, 14, **360**, 576)` |
| `prithviOSSE/run2/output/osse_forecast_Jan2022.nc` (T-ins) | `(240, 14, 360, 576)` |
| `prithviOSSE/uv_ins_run/output/uv_osse_forecast_Jan2022.nc` (UV-ins) | `(240, 14, 360, 576)` |
| `prithviOSSE/t_and_uv/output/t_and_uv_osse_forecast_Jan2022.nc` (TUV-ins) | `(240, 14, 360, 576)` |

The old control file has **361** latitude rows vs the correct **360** used by every
other run; `load_pri()`'s slicing `[:, lev_idx, -nlat:, :]` with `nlat=360` silently
drops a *different* row than intended, causing a 1-row latitude misalignment relative
to every comparison run.

**Fix:** changed `PRI_CTRL` to
`prithviOSSE/run2/output/control_forecast_Jan2022.nc` (the same file generated to fix
Bug A) in both `compute_tuv_rmse.py` and the new standalone recompute script. This also
has the benefit of being a properly apples-to-apples baseline since it's generated by
the exact same `run2`-family code path as T-ins/UV-ins/TUV-ins.

**Combined verification (Bug B + Bug C fixed):** Prithvi `H_lv7` "Control" RMSE:
- Before any fix: flat ~727-2364m the whole 60 days (broken on two counts)
- After Bug B fix only (still old corrupted control file): flat ~727-850m (level fixed,
  grid-corruption bug still present)
- After Bug B + Bug C both fixed: `31.8m` at day 0, growing smoothly to `~110-117m` by
  day 35-55 — physically realistic, matches the shape of every other model/variable.

**New standalone script:**
`/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/recompute_prithvi_tuv_rmse.py` —
self-contained re-implementation of just the Prithvi section of `compute_tuv_rmse.py`
with both fixes applied, so FCN/Aurora caches don't need to be touched/re-run. Regenerates:
```
cache/prithvi_ctrl_rmse.csv
cache/prithvi_tins_rmse.csv
cache/prithvi_uins_rmse.csv
cache/prithvi_tuv_rmse.csv
cache/prithvi_synergy_H_lv7.npy
cache/prithvi_synergy_QV_lv10.npy
cache/prithvi_impr_tuv_H_lv7.npy
cache/prithvi_impr_tuv_QV_lv10.npy
```
Run twice during the session: once with Bug B fix only (to confirm the level-mapping fix
in isolation), then again after also fixing Bug C (final, correct state now on disk).

### Bug D — Aurora/FCN TUV curve diverges from Control *before* day 30 (investigated, **not fixed**, root cause is upstream data generation)

**Symptom reported by user:** in the Z500/Q850 timeseries plots (§5), the TUV-Ins curve
should be *bit-identical* to Control/T-Ins/UV-Ins for lead days 0-30, since no insertion
happens until day 30 — but visually they don't overlap exactly.

**Quantitative check** (matching on common `lead_day` values, comparing raw cached RMSE,
not just visually):

| Model | Day 1 rel. diff (Control vs TUV) | Day 29.5 rel. diff |
|---|---|---|
| Prithvi | -0.01% | +0.03% |
| FCN | -0.04% | -8.11% |
| Aurora | **-74.0%** | +7.95% |

Prithvi's TUV run is a genuine continuation of the same model execution (branches at day
30), hence near-perfect match. FCN and Aurora's TUV runs are **separate, independently
re-run inference jobs**.

**Investigation steps (Aurora, the worst case):**
1. Confirmed IC alignment: both `AUR_CTRL` (`auroraoutput_1year_1stfeb23.nc`) and
   `AUR_TUV` (`CC_aurora/TUV_GLOBAL/output/aurora_tuv_global.nc`) report identical
   `base_time` = `2023-02-01 12:00:00` (read directly via h5py).
2. Diffed the two scripts that generated these files:
   `CC_aurora/T_12H/aurora/1yr_inf.py` (control) vs `CC_aurora/TUV_GLOBAL/inf.py` (tuv)
   — **identical** IC source files (`AURORA_DATA/2023-02-01-{surface-level,atmospheric}.nc`),
   identical batch construction code, identical checkpoint
   (`microsoft/aurora` / `aurora-0.25-pretrained.ckpt`), identical `use_lora=False`,
   identical `torch.cuda.amp.autocast(dtype=torch.bfloat16)` precision setting.
3. Checked `CC_aurora/TUV_GLOBAL/aurora/rollout.py` injection gating logic directly —
   `if truth_atmos is not None and step >= start_inject_step` (`start_inject_step=121`,
   0-indexed `step` from `range(steps)`) — correctly gated, no off-by-one bug causing
   premature injection.
4. **Smoking gun:** directly diffed the **raw predicted z500 field** (not RMSE-vs-truth)
   between the two `.nc` files at lead day 1 (`auroraoutput_1year_1stfeb23.nc` index 2
   [12h step] vs `aurora_tuv_global.nc` index 3 [6h step], both = lead 24h):
   ```
   Control z500 mean ≈ 53900.39 m²/s²
   TUV     z500 mean ≈ 53867.59 m²/s²
   mean abs diff = 143.58,  max abs diff = 1536.0   (typical error signal is only ~150-270)
   ```
   The model OUTPUT itself already differs substantially by day 1 — this rules out any
   analysis-side bug (truth-matching, indexing, plotting) and confirms the divergence is
   baked into the raw forecast files.

**Conclusion:** Aurora (and to a lesser extent FCN) ran control / T-ins / UV-ins / TUV-ins
as 4 **separate** inference jobs from the same nominal IC, each under `bfloat16`
mixed-precision autocast. `bfloat16` GPU inference is not guaranteed bit-reproducible
across separate executions (kernel/reduction-order non-determinism), and these models'
autoregressive chaotic dynamics amplify that tiny per-step noise rapidly — within a few
6-hour steps it's already comparable in magnitude to the actual forecast-error signal.
Prithvi's pipeline avoids this because its 4 variants are literal branches of one
continuous run (saved/forked state), not independent reruns.

**Not fixed.** The only real remedy is regenerating Aurora's (and FCN's) experiment
variants as branches of a single shared run (checkpoint state at day 30, fork from
there) instead of 4 independent from-IC GPU jobs — a nontrivial recompute that was
*not* undertaken this session; left as a documented, known caveat. **If revisited:** look
at adding a save/restore of the rollout `batch` state at step 120 in `1yr_inf.py`, and
have `T_12H`/`UV_12H`/`TUV_GLOBAL`'s `inf.py` scripts load that saved state instead of
re-building `batch` from the raw IC files.

---

## 4. SLURM jobs run this session

| Job | Script | Purpose | Result |
|---|---|---|---|
| 189094 (`prithvi_control`) | `prithviOSSE/run2/forecast_control.slurm` | Generate missing true Prithvi control baseline | Success, 22.9GB output, ~7h runtime (13:38→20:45) |

---

## 5. `common/tuv_final/` — final deliverable plots (8 files, all built on corrected data)

### 5.1 Z500 RMSE timeseries — `plot_z500_tuv_timeseries.py`
4 curves (Control / Temperature-Ins / Wind(UV)-Ins / T+UV-Ins) per model, one figure per
model. Control/T-Ins/UV-Ins sourced from `Multimodel-study/{Model}/{control,temp_ins,wind_ins}/`;
TUV-Ins sourced from `common/tuv_plots/cache/{model}_tuv_rmse.csv` (the corrected cache,
post Bug B+C fixes).

Variable per model: FCN/Aurora → `z500`; Prithvi → `H_lev51` (Multimodel-study side) /
`H_lv7` (tuv cache side) — both = 532 hPa, matching positions confirmed.

Curve styling: Control = dashed blue `#1a56db`, T-Ins = red `#d62728`, UV-Ins = green
`#2ca02c`, TUV-Ins = purple `#6a4c93` (thicker line). Font: DejaVu Serif bold
(Times New Roman not installed on this system — confirmed via `matplotlib.font_manager`).

Outputs:
```
common/tuv_final/FCNv2_Z500_RMSE_timeseries.png
common/tuv_final/Aurora_Z500_RMSE_timeseries.png
common/tuv_final/PrithviWxC_Z500_RMSE_timeseries.png
```

### 5.2 Q850 RMSE timeseries — `plot_q850_tuv_timeseries.py`
Same structure/styling, for 850 hPa moisture (matching the existing `table5_7` "Q/RH"
convention). FCN → `r850` (relative humidity, %); Aurora → `q850` (specific humidity,
kg/kg); Prithvi → `QV_lev63` (Multimodel-study) / `QV_lv10` (tuv cache) — both = 850 hPa.

Outputs:
```
common/tuv_final/FCNv2_Q850_RMSE_timeseries.png
common/tuv_final/Aurora_Q850_RMSE_timeseries.png
common/tuv_final/PrithviWxC_Q850_RMSE_timeseries.png
```

### 5.3 Synergy maps + bar chart — `plot_synergy_z500_q850.py`
Restricted version of the original `tuv_plots/plot_tuv_analysis.py` Fig 5.27/5.28, scoped
to just Z500 + Q850 (per user request), using the corrected Prithvi cache.

- **Fig 5.27** spatial synergy maps: 2×3 grid (rows = Z500/Q850, cols = FCN/Aurora/
  Prithvi), Robinson projection (`cartopy`), synergy = `i_tuv - max(i_t, i_uv)` per grid
  point, mature phase (day 40-60), clipped to ±15%, RdBu_r colormap (red = positive =
  dual insertion beats best single insertion).
  **Layout bug fixed:** originally used `fig.add_subplot()` + `plt.tight_layout()`,
  which doesn't reserve enough vertical space for Cartopy GeoAxes + per-axis horizontal
  colorbars — the Q850 row's titles collided with the Z500 row's colorbars. Fixed by
  switching to an explicit `fig.add_gridspec(2, 3, hspace=0.65, wspace=0.22, top=0.90,
  bottom=0.05, left=0.03, right=0.99)` with `shrink=0.9` colorbars and no `tight_layout()`
  call. Verified visually after the fix — no more overlap.
- **Fig 5.28** grouped bar chart with error bars: 2 variable groups (Z500, Q850) × 3
  model bars, CI = std-of-step-wise-synergy / √n over the mature phase.

  Results (for the record): Z500 synergy — FCN +11.6%, Aurora **-7.0%** (interference),
  Prithvi +2.1%. Q850 synergy — FCN +2.5%, Aurora +11.4%, Prithvi +6.6%.

Outputs:
```
common/tuv_final/fig5_27_spatial_synergy_maps_z500_q850.png
common/tuv_final/fig5_28_synergy_bar_chart_z500_q850.png
```

### 5.4 Source scripts (all in `common/tuv_final/`)
```
plot_z500_tuv_timeseries.py
plot_q850_tuv_timeseries.py
plot_synergy_z500_q850.py
```

---

## 6. Known stale / unaddressed items (read before trusting old PNGs)

- **`common/tuv_plots/fig5_24_*.png` through `fig5_29_*.png` and
  `table5_7_tuv_summary.csv`/`.txt` are STALE.** They were generated by
  `plot_tuv_analysis.py` *before* Bug B/C were fixed. The underlying cache CSVs/NPYs
  **have** been corrected in place, but `plot_tuv_analysis.py` itself was **not
  rerun** this session (only the new restricted `tuv_final/plot_synergy_z500_q850.py`
  was built/run, covering just Z500/Q850 for Fig 5.27/5.28). If the full Fig 5.24-5.29
  set is needed with corrected Prithvi numbers, rerun
  `common/tuv_plots/plot_tuv_analysis.py` (no code changes needed there — it just reads
  the cache, which is already fixed).
- **`common/lat_bands/*.png` (the original 8 model-centric plots from
  `plot_latband_latslice_updated.py`, dated 2026-06-08) were left untouched** — they
  predate the Prithvi fixes and were never the design the user wanted anyway (see §2).
  Not deleted, just superseded by `common/lats_FINAL/`.
- **Bug D (Aurora/FCN pre-day-30 TUV drift) is unresolved** — see §3 Bug D. Purely a
  data-generation issue; would need a real GPU rerun to fix properly.
- The very first cross-pipeline inconsistency noticed (§1.1): `cloud2/`'s and
  `terrain2/`'s `PRI_CTRL` points at `prithviOSSE/run2/output/osse_forecast_Jan2022.nc`
  (the T-insertion file) instead of a genuine baseline — same family of bug as Bug A,
  but in a *different* script (`fig_cloud2_uv_ins.py` / `fig_terrain2_uv_ins.py`), and
  **was not fixed this session** (only flagged). If those 4 directories' plots are
  revisited, this should be patched the same way Bug A was (point at
  `prithviOSSE/run2/output/control_forecast_Jan2022.nc`).
- **UPDATE 2026-09-15 (see §8): `tuv_final/plot_synergy_z500_q850.py` (Fig
  5.27/5.28) is no longer 3-model-only** — GraphCast was added. The STALE
  `tuv_plots/fig5_24_*.png` through `fig5_29_*.png` full-set plots
  (`plot_tuv_analysis.py`, 3-model, never rerun) are UNCHANGED by this and
  remain stale/3-model as described above.

---

## 7. Quick file-path index (everything created/modified this session)

**Created:**
```
common/plot_latband_by_experiment.py
common/lats_FINAL/Figure_latband_{T_Ins,Wind_Ins,Geo_Ins}_crossvar.png
common/lats_FINAL/Figure_latslice_{T_Ins,Wind_Ins,Geo_Ins}_crossvar.png
common/lats_FINAL/Figure_latband_{T_Ins,Wind_Ins,Geo_Ins}_crossvar_mature.png
common/lats_FINAL/Figure_latslice_{T_Ins,Wind_Ins,Geo_Ins}_crossvar_mature.png
common/tuv_plots/recompute_prithvi_tuv_rmse.py
prithviOSSE/run2/forecast_control.slurm
prithviOSSE/run2/output/control_forecast_Jan2022.nc   (22.9GB, via SLURM job 189094)
common/tuv_final/plot_z500_tuv_timeseries.py
common/tuv_final/plot_q850_tuv_timeseries.py
common/tuv_final/plot_synergy_z500_q850.py
common/tuv_final/FCNv2_Z500_RMSE_timeseries.png
common/tuv_final/Aurora_Z500_RMSE_timeseries.png
common/tuv_final/PrithviWxC_Z500_RMSE_timeseries.png
common/tuv_final/FCNv2_Q850_RMSE_timeseries.png
common/tuv_final/Aurora_Q850_RMSE_timeseries.png
common/tuv_final/PrithviWxC_Q850_RMSE_timeseries.png
common/tuv_final/fig5_27_spatial_synergy_maps_z500_q850.png
common/tuv_final/fig5_28_synergy_bar_chart_z500_q850.png
common/MEMORY.md   (this file)
```

**Modified (bug fixes):**
```
prithvi_new/lats/section_5_7_prithvi_latband_analysis.py        (Control path, Bug A)
prithvi_new/lat_bands/section_5_7_prithvi_latslice_analysis.py  (Control path, Bug A)
prithvi_new/lats/values/*.csv          (regenerated after Bug A fix)
prithvi_new/lat_bands/values/*.csv     (regenerated after Bug A fix)
common/tuv_plots/compute_tuv_rmse.py   (Bug B: pri_idx_to_merra_idx(); Bug C: PRI_CTRL path)
common/tuv_plots/cache/prithvi_{ctrl,tins,uins,tuv}_rmse.csv        (regenerated, Bug B+C)
common/tuv_plots/cache/prithvi_synergy_{H_lv7,QV_lv10}.npy         (regenerated, Bug B+C)
common/tuv_plots/cache/prithvi_impr_tuv_{H_lv7,QV_lv10}.npy        (regenerated, Bug B+C)
```

**Modified (user-side, not by me):**
```
Multimodel-study/Aurora/control/aurora_rmse_all_vars.csv   (user added this file mid-session)
```

**Left untouched (for reference / comparison):**
```
common/plot_latband_latslice_updated.py        (original per-model design, superseded)
common/lat_bands/*.png                         (original 8 model-centric PNGs, dated 2026-06-08)
common/tuv_plots/plot_tuv_analysis.py          (original full Fig5.24-29 script — STALE outputs, see §6)
common/tuv_plots/fig5_2{4,5,6,9}_*.png         (STALE, see §6)
common/tuv_plots/table5_7_tuv_summary.{csv,txt} (STALE, see §6)
cloud2/fig_cloud2_uv_ins.py, terrain2/fig_terrain2_uv_ins.py  (have the same PRI_CTRL
                                                  mislabeling as Bug A but NOT fixed)
```

---

## 8. ADDENDUM 2026-09-15 — GraphCast added to the TUV synergy figure

A later session (picking up `Graphcast-full/GC-DETER-2023` and
`prithvi-feb23` — the two later, full-resolution single-model OSSE
studies that superseded the small-GraphCast/Jan-2022-Prithvi runs this
directory's §1 file-path table lists) added **GraphCast** as a 4th
model to Fig 5.27/5.28 (`tuv_final/plot_synergy_z500_q850.py`), which
this file's §5 originally described as 3-model-only (FCN/Aurora/
Prithvi). Full writeup, including a diagnosed-and-explained (not a
bug) extreme-outlier finding in GraphCast's Z500 spatial-synergy title
statistic driven by low geopotential variance in the tropics, is in
`Graphcast-full/session_log.md` §38 — not duplicated here in full,
only indexed.

**Quick summary for a reader of this file specifically:**
- New script: `tuv_plots/compute_graphcast_tuv_cache.py` — builds
  `cache/graphcast_{ctrl,tins,uins,tuv}_rmse.csv` (reusing
  GC-DETER-2023's own already-computed RMSE, no new per-model truth
  loader needed unlike the FCN/Aurora/Prithvi sections of
  `compute_tuv_rmse.py`) and `cache/graphcast_{synergy,impr_tuv}_
  {z500,q850}.npy` (new spatial computation, GraphCast's native
  0.25deg 721x1440 grid, saved alongside `cache/graphcast_{lat,lon}
  .npy`).
- `plot_synergy_z500_q850.py` edited in place: `MOD_LABEL`/`COLORS`/
  `VAR_MAP` gained a `graphcast` entry, the spatial-map grid widened
  2x3 -> 2x4, the bar chart widened to 4 bars, and the bar-chart
  legend moved outside the axes (was overlapping 2 bar labels even at
  3 models).
- Result (mature-phase, days 40-60): GraphCast Z500 synergy = **-28.4%**
  (large negative — consistent with this later study's own established
  "T-insertion degrades Z" finding: UV-insertion alone already captures
  most of the achievable Z500 benefit, so adding T on top actively
  hurts relative to UV-only); GraphCast Q850 synergy = **+14.3%**
  (second-highest of the 4 models: Aurora 19.0 > GraphCast 14.3 >
  Prithvi 6.6 > FCN 2.5).
- `common/tuv_final/PHYSICAL_ANALYSIS.txt` (the full scientific
  interpretation narrative, §5 below in that report's own numbering)
  was deliberately NOT rewritten to discuss GraphCast — the request
  this addendum responds to was specifically about missing data/
  figures, now fixed; a GraphCast paragraph at the same interpretive
  depth as the existing FCN/Aurora/Prithvi sections is separate,
  larger work, left as an open item.
