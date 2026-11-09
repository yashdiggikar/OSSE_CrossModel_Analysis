# Empirical Correlation-Length Verification — Cross-Model Method

Companion to `METHOD_OVERVIEW.md` (which describes how the correlated
perturbation is *generated*) and the three `IMPLEMENTATION.md` files (which
describe each model's code/grid/channel specifics). This file describes a
**validation analysis**, not a change to the perturbation-generation code:
it checks, empirically, on the real saved output of the six SLURM
experiments, whether the injected perturbation actually has the horizontal
and vertical correlation structure `METHOD_OVERVIEW.md` §2.1/2.2 claims it
does.

**Why this is needed for the paper, specifically**: `METHOD_OVERVIEW.md` §7
already reports validation numbers, but those came from a **standalone unit
test on a bare 721x1440 synthetic grid**, run *before* any of the six SLURM
jobs were submitted. It confirms the generator function is implemented
correctly in isolation — it does not confirm that the field actually
written into the real, saved multi-GB experiment output has the claimed
structure (e.g., a bug in how the perturbation is sliced, padded, or
inserted into the model's state tensor could silently break the correlation
structure between generation and insertion, without changing forecast skill
enough to be obvious from RMSE alone). This analysis closes that gap using
the real data.

---

## 1. How the perturbation field is recovered (it is never saved directly)

The perturbation is generated fresh, in memory, once per autoregressive
step, and immediately **hard-replaces** the model's state at the
truth-insertion channels (`pred.atmos_vars["t"] = truth_t`-style — see
`FCN/IMPLEMENTATION.md` §2, `Aurora/IMPLEMENTATION.md` §2,
`Prithvi/IMPLEMENTATION.md` §2). It is discarded immediately after use and
never written to disk on its own.

Because the replacement is *hard* (not blended, not Kalman-gain-weighted),
at the first post-injection step:

```
experiment_output[step] = truth[same_valid_time] + perturbation[step]
=>  perturbation[step] = experiment_output[step] - truth[same_valid_time]
```

This is an **exact** recovery, not an approximation — subtracting truth from
the saved experiment output at the injection step gives back precisely the
random field that was added, using only data that is already saved. This is
the method used here, in preference to re-deriving the field by
re-importing and re-running each model's (three different, hand-forked)
perturbation-generation code with the same RNG seed — the reconstruction
approach validates the actual real data product end-to-end, with no
dependency on any package internals.

**Injection step used**: step index 120 (0-indexed), i.e. the 121st
autoregressive step / the *first* post-injection step, matching the
`INJECT_STEP = 120` convention already used by every `compute_spatial_bias_*`
script in this project. Using the very first post-injection step matters:
by the *second* post-injection step, the field has already been passed once
through the model's own forward dynamics, which will have started to
distort/advect/smooth it (this distortion is in fact the subject of the
separate `SPECTRAL_BLURRING_ANALYSIS.md`, not this document — here, the
goal is to check the raw, as-injected field before the model has touched
it at all).

---

## 2. Horizontal correlation-length check

**Updated 2026-07-25**: this check now runs at **three** representative
pressure levels (300/500/1000 hPa) instead of the single 500 hPa level used
in the original version of this analysis, and is computed for **every**
variable (T, U, V, Z, and humidity — relative humidity `R` for FCN, specific
humidity `Q` for Aurora/Prithvi) **for both experiments**, regardless of
which family was directly inserted — matching the cross-variable-coupling
convention already established in this project's `compute_global_rmse_corr.py`
et al. So the temp-insertion experiment's correlation-length CSV now also
reports the horizontal structure of its U/V/Z/humidity *response* fields, not
just the directly-injected T field, and vice versa for the wind experiment.
This costs almost no extra ERA5/MERRA-2 I/O for the *vertical* check (§3
already loads all native levels in one pass per variable), but does require
one additional full-level truth load per extra variable (Z, humidity) beyond
the 1–2 variables the pre-expansion version loaded.

### 2.1 Method

For three representative latitude bands (near-equator, mid-latitude,
high-latitude — exact rows depend on each model's grid, see §4) **and** each
of the three representative pressure levels above, take the recovered
perturbation field and compute two **1D lag-correlation curves**, matching
the style already used for development-time validation in
`METHOD_OVERVIEW.md` §7:

- **Zonal** (longitude direction): shift the field along that latitude row
  by `k` grid cells (periodic wraparound, matching the perturbation
  generator's own `mode="wrap"` longitude treatment) and compute the
  Pearson correlation between the row and its shifted copy, for
  `k = 0, 1, 2, ..., 40`.
- **Meridional** (latitude direction): same, but shifting a representative
  longitude column by `k` grid cells (no wraparound — the generator itself
  uses `mode="nearest"` at the poles, so neither should this check).

Each lag `k` (grid cells) is converted to a physical distance in km using
that band's **true** km-per-degree spacing:
- Meridional: `km_per_deg_lat * grid_spacing_deg_lat` — constant everywhere
  (`km_per_deg_lat = 111.0`, matching the generator's own conversion
  constant, see `FCN/IMPLEMENTATION.md` §4).
- Zonal: `111.0 * cos(latitude) * grid_spacing_deg_lon` — **shrinks toward
  the poles**, which is exactly the source of the pole-distortion limitation
  `METHOD_OVERVIEW.md` §2.1 already flags. Running this check at three
  different latitude bands is specifically designed to make that distortion
  visible and quantified with real numbers, not just asserted.

The **empirical correlation length** at each band/direction is reported as
the e-folding distance — the lag (in km) at which the correlation curve
first drops to `1/e ≈ 0.368` — found by linear interpolation between the
two bracketing lag points.

### 2.2 An important subtlety this check is designed to resolve

The perturbation generator (`METHOD_OVERVIEW.md` §2.1) computes its
Gaussian-filter pixel width directly from the *nominal* 250 km target:
`SIGMA_PX = 250 / (111 * grid_spacing_deg)`. But the **autocorrelation
function of a field produced by Gaussian-smoothing white noise with
kernel-std `sigma` is itself a Gaussian with std `sigma * sqrt(2)`**
(smoothing correlates the field with a width equal to the *convolution* of
the kernel with its own mirror image, which doubles the variance of the
resulting Gaussian). This means there are two plausible answers for what
"250 km" ends up meaning physically in the realized field:

- If the empirical e-folding length comes out close to **250 km**, the
  kernel-std was, in effect, being used directly as the target
  autocorrelation length (a common simplifying convention in practice,
  even though not strictly the same quantity).
- If it comes out close to **250 * sqrt(2) ≈ 354 km**, the theoretical
  smoothing-kernel relationship is dominating and the *realized* field is
  meaningfully broader than the nominally "targeted" length.

This is exactly the kind of precise methodological detail a paper's methods
section (and a careful reviewer) will want stated correctly rather than
assumed — this check answers it empirically instead of guessing, and the
result should be quoted directly in the paper's methods text once known.

---

## 3. Vertical correlation check

### 3.1 Method

Build the **full per-level cube** of the recovered perturbation field (13
levels for FCN/Aurora, 14 for Prithvi — see §4 for exact variable/level
handling per model), flatten each level's 2D field spatially, and compute
the full **level x level Pearson correlation matrix** (a 13x13 or 14x14
symmetric matrix, unit diagonal by construction).

Compare directly against the **theoretical** matrix from
`METHOD_OVERVIEW.md` §2.2:

```
corr(l_i, l_j) = exp( -| ln(p_i) - ln(p_j) | / 0.5 )
```

using each model's own pressure/level table (`TARGET_LEVELS` for FCN/Aurora,
`LEVEL_APPROX_HPA` for Prithvi — both already defined in the respective
`IMPLEMENTATION.md` files, reused here unchanged, not re-derived).

Headline number reported: the RMS difference between the empirical and
theoretical matrices (off-diagonal elements only, since the diagonal is
trivially 1.0 in both by construction and would artificially deflate the
error).

### 3.2 What this validates that the horizontal check does not

The vertical structure is built by a **Cholesky-mixing** step
(`METHOD_OVERVIEW.md` §2.2: `mixed[l] = sum_j A[l,j] * smoothed[j]`, where
`A` comes from Cholesky-factoring the target correlation matrix) applied
*after* independently spatially-smoothing each level. This is a completely
separate code path from the horizontal smoothing, with its own
opportunities for indexing bugs (e.g. a transposed `A` matrix, or levels
mixed in the wrong order relative to the pressure table) that the
horizontal check alone would never catch, since a transposed-but-still-
correlated mixing matrix would still produce spatially smooth, horizontally
well-correlated fields per level — just with the wrong *inter-level*
relationships. Checking both independently is deliberate.

---

## 4. Per-model implementation specifics

| | Grid used | Latitude bands (nearest actual row) | Level cube source | Level/pressure table used |
|---|---|---|---|---|
| **FCN** | 721 lat (90->-90) x 1440 lon (0->360), 0.25 deg | ~5N, ~45N, ~70N | Loop over 13 `channel`-indexed variables per prefix (`t50..t1000`, `u50..u1000`, `v50..v1000`, `z50..z1000`, `r50..r1000`), stacked into `(13, 721, 1440)` cubes, one per variable | `TARGET_LEVELS = [50,100,150,200,250,300,400,500,600,700,850,925,1000]` (`FCN/IMPLEMENTATION.md` §3); horizontal check evaluated at the 3 entries `[300,500,1000]` |
| **Aurora** | 720 lat (no south-pole row) x 1440 lon, 0.25 deg | ~5N, ~45N, ~70N | Loop over 13 top-level variables per prefix (`t50..t1000`, `u50..u1000`, `v50..v1000`, `z50..z1000`, `q50..q1000`), stacked into `(13, 720, 1440)` cubes | Same `TARGET_LEVELS` list; horizontal check evaluated at `[300,500,1000]` |
| **Prithvi** | 360 lat (-89.5->90) x 576 lon (-180->179.375), 0.5 x 0.625 deg | ~5N, ~45N, ~70N | Single slice `ds.variables[varname][step, :, :, :]` per variable (`"T"`,`"U"`,`"V"`,`"H"` for geopotential height,`"QV"` for specific humidity) — already a `(14, 360, 576)` cube, no loop needed | `LEVEL_APPROX_HPA = [52,118,164,226,265,310,424,532,604,693,832,903,946,961]` (`Prithvi/IMPLEMENTATION.md` §3); horizontal check evaluated at the nearest native levels to 300/500/1000 hPa (Prithvi has no exact match at any of the 3 — nearest are used, same convention as this project's RMSE/bias scripts) |

**Variable coverage, all 3 models (updated 2026-07-25)**: T, U, V, Z
(geopotential/geopotential height), and humidity (R for FCN — relative
humidity; Q for Aurora/Prithvi — specific humidity), each evaluated for
**both** experiments (`t_perturbed_corr`/`T_PERTURBED_CORR`/
`perturbed_corr_temp_ins` and `uv_perturbed_corr`/`UV_PERTURBED_CORR`/
`perturbed_corr_uv_ins`) — not just the directly-inserted family. **FCN's
relative-humidity (`r`) data is absent from this project's ERA5 archive at
every level** (confirmed independently in `compute_global_rmse_corr.py`'s
loading loop, which fails identically) — FCN's `R` rows are therefore
structurally empty (the script gracefully skips this variable, printing
`no ERA5 data available at any level`), not a bug in this check.

Truth source per model (for the `experiment_output - truth` subtraction),
same files already used by each model's existing RMSE/bias scripts:
- FCN/Aurora: ERA5 pressure-level GRIB (`AURORA_DATA/rmse_data/
  era5_pressure_2023_{02,03,04}.grib`), opened via `cfgrib`, nearest-in-time
  matched to the injection step's valid time (tolerance 3h, same
  `nearest_idx`/`era5_idx` helpers already used elsewhere in this project).
  Surface truth (T2M/U10/V10) uses the companion surface GRIB files
  (`era5_surface_2023_{02,03,04}.grib`).
- Prithvi: MERRA-2 daily files (`prithviOSSE/data/merra2/
  MERRA_pres_YYYYMMDD.nc`), exact-hour matched (MERRA-2 files are already
  3-hourly, matching the 6h-step cadence exactly at `t_idx = hour // 3`).

Surface-variable checks (T2M for all variables' surface counterpart; U10/V10
or U10M/V10M for wind) are **not** included in the vertical-correlation check
(there is only one surface level, nothing to correlate against) but **are**
included in the horizontal check, using the same 3-latitude-band method as
the upper-air levels, with `level_hpa` left blank/`NaN` for those rows in the
output CSV (see §5).

---

## 5. Output files

Per model, per experiment (`t_perturbed_corr`/`uv_perturbed_corr`,
`T_PERTURBED_CORR`/`UV_PERTURBED_CORR`,
`perturbed_corr_temp_ins`/`perturbed_corr_uv_ins`), written to
`PERTURBATION_EXPERIMENTS/<Model>/<experiment>/`:

- `corr_length_horizontal_<experiment>.csv` — one row per (variable,
  `level_hpa`, band, direction): empirical e-folding length (km), target
  length (km), ratio, `derived_field` flag. **Updated 2026-07-25**: added the
  `level_hpa` column (300/500/1000 for upper-air rows, blank/`NaN` for
  surface rows) since the check now runs at 3 levels instead of 1; existing
  consumers (`make_paper_figures.py`'s `plot_corr_length_summary`) filter to
  `level_hpa == 500` (or blank) to keep the quick-look summary plot a single
  point per band/direction rather than 3 overlapping ones.
- `corr_length_vertical_<experiment>.csv` — long-format level-pair table:
  level_i, level_j, pressure_i, pressure_j, empirical correlation,
  theoretical correlation, difference. Unchanged in structure, now populated
  for all 5 variables (T/U/V/Z/humidity) instead of 1–2.
- `corr_length_horizontal_map_<experiment>.png` — lag-correlation curves at
  the 500 hPa level specifically (empirical, 3 bands x 2 directions)
  overlaid with the theoretical Gaussian curve(s) from §2.2 — the plot
  itself was not widened to show all 3 levels, to keep it readable.
- `corr_length_vertical_map_<experiment>.png` — empirical vs. theoretical
  correlation-matrix heatmaps for the first-processed variable, side by
  side, plus a difference panel.

---

## 6. Known limitations (stated explicitly, for the paper's limitations section)

- **Single-realization estimate.** Each check uses the perturbation field
  from exactly one step (step 120). `METHOD_OVERVIEW.md` §7's own synthetic
  validation already noted empirical correlation estimates from a single
  realization deviate somewhat from the theoretical target even when the
  generator is implemented correctly (e.g. its reported vertical
  level-0/1 empirical 0.25 vs theoretical 0.250 was a lucky near-exact
  match; level-0/12 was -0.03 vs a theoretical 0.0025, a larger absolute
  deviation from a single draw). A future extension could average this
  check over several steps for a tighter estimate; not done here to keep
  the check fast and because it is meant to catch code bugs (large,
  qualitative discrepancies), not to precisely re-estimate the generator's
  statistics.
- **Pole distortion is expected, not a bug.** As documented in
  `METHOD_OVERVIEW.md` §2.1, the zonal correlation length is expected to
  shrink at high latitudes relative to the meridional length and relative
  to the nominal 250 km target — this check is designed to *quantify* that
  known, already-documented simplification, not to "pass/fail" against a
  uniform 250 km everywhere.
- **Wind perturbations are a derivative field.** `u'`/`v'` are differentiated
  from a streamfunction, not smoothed directly (`METHOD_OVERVIEW.md` §2.3)
  — differentiating a Gaussian-correlated field changes its correlation
  function shape (it is no longer exactly Gaussian-shaped, and has a
  different, shorter effective length than the parent streamfunction). The
  horizontal check on `u'`/`v'` fields should therefore **not** be expected
  to reproduce the same e-folding length as the temperature check even if
  everything is implemented correctly — this is flagged in the CSV output
  (a `derived_field` column, `True` for wind, `False` for temperature) so
  results are not misread as a discrepancy.
- **Added 2026-07-25 — the cross-variable rows validate a different, weaker
  claim than the directly-injected variable's own rows.** Only the
  directly-inserted variable (T in the temp experiment; U/V in the wind
  experiment) is hard-replaced with the exact correlated-noise design target
  every 6 hours — its recovered-perturbation statistics are therefore a true
  test of the *generator's* correctness (the original purpose of this
  document). The newly-added Z/humidity (and the wind experiment's own T)
  rows are the model's own learned *response* to that injection, not a
  second copy of the injected field — their vertical/horizontal correlation
  structure reflects whatever the model's own dynamics produced, which was
  never designed to match the 0.5-log-pressure/250 km targets in the first
  place. A tight match on a response variable's rows would be a genuinely
  interesting finding about the model's own learned dynamics; a poor match
  is expected and is **not** evidence of a bug, unlike a poor match on the
  directly-injected variable's own rows, which would be.
