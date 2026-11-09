# Prithvi WxC — Correlated OSSE Perturbation Implementation

See `../METHOD_OVERVIEW.md` for the cross-model method description and
citations. This file documents the Prithvi-specific code, paths, and
hyperparameters.

---

## 1. Location

| | Path |
|---|---|
| Temperature experiment | `/umbc/rs/pi_halem/users/yashd1/prithviOSSE/perturbed_corr_temp_ins/run_forecast_perturbed_corr_temp.py` |
| Wind experiment | `/umbc/rs/pi_halem/users/yashd1/prithviOSSE/perturbed_corr_uv_ins/run_forecast_perturbed_corr_uv.py` |
| SLURM job scripts | `forecast_perturbed_corr_temp.slurm`, `forecast_perturbed_corr_uv.slurm` (same dirs) |
| Base template (uncorrelated baseline, unmodified) | `prithviOSSE/perturbed_temp_ins/run_forecast_perturbed_temp.py` |

Unlike FCN and Aurora, Prithvi WxC has **no per-experiment package fork** —
there is a single shared `PrithviWxC` install (symlinked `data/`/`model/`
into `/umbc/rs/pi_halem/users/yashd1/prithvi/`), and the entire rollout +
insertion logic lives directly in each experiment's own
`run_forecast_<name>.py` as a self-contained script (hand-rolled
autoregressive loop calling `model(batch)` directly, not reusing any
package-level rollout helper). The two new scripts here follow that exact
same self-contained convention.

## 2. Where the perturbation logic actually lives

Both new scripts are structurally near-identical to
`perturbed_temp_ins/run_forecast_perturbed_temp.py`: same CLI flags
(`--inject-temp`/`--inject-uv`, `--inject-start-step`, `--truth-source`),
same `build_static()`/`build_climate()`/`unpad()` helpers (byte-identical),
same main autoregressive loop structure. The only changes are:

- New module-level constants: `LEVEL_APPROX_HPA`, `HORIZ_CORR_LENGTH_KM`,
  `SIGMA_PX_LAT`/`SIGMA_PX_LON` (two values, not one — see Section 3),
  `VERT_L_LOGP`, `VERT_CHOL_T`/`VERT_CHOL_UV`.
- New helper function `_correlated_level_fields(...)` (temperature script)
  and additionally `_streamfunction_to_uv(...)` +
  `_generate_correlated_wind_perturbation(...)` (wind script).
- The injection function itself
  (`inject_temperature_perturbed_corr()` / `inject_uv_perturbed_corr()`,
  renamed from `inject_temperature_perturbed()` /
  the land/ocean/coastal-style `inject_uv_land()` pattern) calls these new
  helpers instead of per-point `rng.normal(...)`.

## 3. Grid and channel conventions (Prithvi WxC "large_rollout", verified)

- **Native grid is non-square in degrees**: 0.5 deg latitude x 0.625 deg
  longitude (MERRA-2 native), unlike FCN/Aurora's uniform 0.25 deg x
  0.25 deg. This means a single isotropic pixel-sigma would NOT correspond
  to the same physical correlation length in both directions — so, unlike
  the FCN/Aurora implementations, **two separate pixel-sigma values** are
  computed:
  ```
  SIGMA_PX_LAT = 250.0 / (111.0 * 0.5)   ≈ 4.5 grid cells
  SIGMA_PX_LON = 250.0 / (111.0 * 0.625) ≈ 3.6 grid cells
  ```
  passed to `scipy.ndimage.gaussian_filter` as a tuple
  `sigma=(SIGMA_PX_LAT, SIGMA_PX_LON)` (scipy supports per-axis sigma).
- **14 native MERRA-2 model levels** (not 13 standard pressure levels like
  FCN/Aurora) — ascending index 0 (TOA, model level 34) to index 13
  (near-surface, model level 72). Approximate pressure equivalents (the
  same verified table used elsewhere in this project, `hydro/master.md`
  Section 3.1):
  ```
  LEVEL_APPROX_HPA = [52, 118, 164, 226, 265, 310, 424, 532, 604, 693, 832, 903, 946, 961]
  ```
  used only to build the log-pressure vertical correlation matrix — the
  actual model levels (used for reading/writing MERRA-2 files) remain
  `cfg.levels` (native indices), unchanged.
- Channel indices (computed programmatically via `surface_vars.index(...)`/
  `vertical_vars.index(...)`, unchanged from `perturbed_temp_ins`/
  `land_uv_ins`):
  - T2M = 12, TS = 16 (surface); T upper-air = channels `118:132` (14 levels)
  - U10M = 17, V10M = 18 (surface); U upper-air = `132:146`, V upper-air = `146:160`

## 4. Two separate correlated fields per injection (surface vs. upper-air)

Unlike FCN/Aurora (which only insert wind/temperature at pressure levels,
no surface-vs-upper-air split in the correlation structure), Prithvi's
existing baseline (`perturbed_temp_ins`, `land_uv_ins`) separately injects
**both** a surface channel pair (T2M+TS, or U10M+V10M) **and** the 14-level
upper-air field. This implementation preserves that structure and adds
correlation at both levels of the hierarchy:

- **Surface (T2M, TS)**: treated as two "adjacent levels" with their own
  small 2x2 correlation matrix (`corr(T2M,TS) = 0.7`, a documented,
  reasonable-but-not-empirically-derived choice — T2M and skin temperature
  are physically closely linked, so assuming they share a substantial
  fraction of their observation-error structure is defensible but should be
  flagged as an assumption, not a measured value, if reviewed).
- **Surface (U10M, V10M)**: single-level (`n_lev=1`) streamfunction
  perturbation — a non-divergent 2D wind perturbation at one level, no
  vertical correlation needed since there is only one surface level.
- **Upper-air (14 levels)**: full `_correlated_level_fields`/
  `_generate_correlated_wind_perturbation` treatment as described in
  `METHOD_OVERVIEW.md`.
- Surface and upper-air draws use **distinct RNG seeds**
  (`step` vs. `step + 1_000_000`) so they are reproducible but not
  accidentally identical/correlated with each other.

## 5. Hyperparameters used

Temperature sigma-by-level (K) — **unchanged from `perturbed_temp_ins`**:
```
[2.0,1.8,1.5,1.2,1.0,0.8,0.7,0.7,0.8,0.8,1.0,1.0,1.2,1.2]  (index 0=TOA..13=surface)
SIGMA_T2M = SIGMA_TS = 1.2 K
```

Wind sigma-by-level (m/s) — newly constructed, same jet-stream-level
reasoning as FCN/Aurora, reindexed to Prithvi's 14-native-level ascending
(TOA->surface) order:
```
[1.5,1.6,1.8,2.2,2.0,1.8,1.4,1.2,1.1,1.0,1.0,1.1,1.3,1.3]
SIGMA_U10M = SIGMA_V10M = 1.5 m/s
```

## 6. IC and truth source

- IC: same as `perturbed_temp_ins`/all other Prithvi experiments —
  `Merra2Dataset`, MERRA-2, fixed 2022-01-01T00:00/T06:00.
- Truth: MERRA-2 daily files,
  `prithviOSSE/data/merra2/MERRA2_sfc_YYYYMMDD.nc` /
  `MERRA_pres_YYYYMMDD.nc` (symlinked to `prithvi/data/merra2/`, per
  `Organized_results/file_paths.md` Section 2 — coverage confirmed through
  ~2022-03-11, comfortably covering the 60-day/2022-01-01 to
  ~2022-03-02 experiment window).
- Model config: `"large_rollout"` (fixed 6h forecast variant), matching
  every other prithviOSSE experiment — **not** the `"large"`
  flexible-lead-time config used by the unrelated `hydro/` embedding-
  extraction project.

## 7. Environment

`venvs/aurora310` (same shared environment used for Aurora AND Prithvi
inference in this project, per the existing `forecast_perturbed_temp.slurm`
template) — scipy 1.15.3 already present, no environment changes needed.

## 8. SLURM submission (2026-07-22)

```
sbatch prithviOSSE/perturbed_corr_temp_ins/forecast_perturbed_corr_temp.slurm  # job 203462
sbatch prithviOSSE/perturbed_corr_uv_ins/forecast_perturbed_corr_uv.slurm      # job 203463
```

Both requested 80G mem, 1 GPU, 24h walltime, `chip-gpu` cluster, `gpu`
partition, `pi_halem` account, pinned to nodes `g20-07,g20-08,g20-10` — same
resource request as `perturbed_temp_ins/`'s original job.
