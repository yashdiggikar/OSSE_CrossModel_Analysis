# Replotting Instructions

A checklist of everything to check **before** building a new figure from
`Organized_results/` (or `Organized_plots/common/`), independent of what
style you're going for. These are correctness/interpretation issues, not
aesthetic ones — get any of the "hard requirements" wrong and the resulting
plot is scientifically incorrect, not just differently styled.

See `map.md` in this same directory for what every file/folder actually
contains, and `file_paths.md` for every original prediction/truth file this
data traces back to (only needed if you're recomputing, not replotting).
This file is specifically about pitfalls when *using* the data already here.

---

## 0. Quick-reference checklist

Before plotting anything new, confirm:

- [ ] Am I aggregating over lat/lon myself? → apply cosine-latitude weighting (§1.1).
- [ ] Am I combining Z (Aurora/FCN/GC) with H (Prithvi)? → convert units first (§1.2).
- [ ] Am I combining a moisture variable across models? → check which quantity each model actually stores (§1.3).
- [ ] Am I treating any Prithvi "500 hPa"/"850 hPa"/"300 hPa" label as an exact pressure? → it's an approximation (§1.4).
- [ ] Am I building a multi-model spatial map (not just a bar/line chart)? → resample onto a common grid first (§1.5).
- [ ] Am I plotting a "trivial" self-insertion cell as if it were a real finding? → exclude it or label it explicitly (§2.4).
- [ ] Am I citing GraphCast's non-100% "trivial" diagonal or a stale FCN coupling file as if something's wrong? → both are known, already-explained (§2.1, §2.2).
- [ ] Am I filtering the mature/adjustment phase by column value, not a hardcoded step index? (§3.2)
- [ ] Does the variable I want actually exist for all 4 models? → check the per-model variable lists before assuming uniformity (§4).

---

## 1. Hard requirements — get these wrong and the plot is wrong

### 1.1 Cosine-latitude weighting

Every RMSE, bias, and coupling number in this study was computed with
`weight ∝ cos(lat)` — grid cells near the poles cover much less physical
area than cells near the equator and must be down-weighted accordingly.
This is already baked into every pre-aggregated CSV (`nrmse/`, `lats/`,
`lat_bands/`, `coupling/`, `bias_summary.csv`'s MAB columns). It is **not**
baked into the raw spatial `lat, lon, bias` CSVs — if you're computing your
own new aggregate (e.g. a regional mean) directly from those, weight each
row by `cos(radians(lat))` before averaging. A flat/unweighted mean will
systematically overweight the poles.

### 1.2 Geopotential (Z) vs. geopotential height (H) — not the same unit

Aurora, FourCastNet, and GraphCast store **Z**, raw geopotential, in
**m²/s²**. Prithvi stores **H**, geopotential *height* (already divided by
g = 9.80665), in **meters**. Converting: `Z [m²/s²] = H [m] × 9.80665`.
Never plot Z500 (Aurora/FCN/GC) directly against H_lev51 (Prithvi) on a
shared axis without converting one to match the other — the raw magnitudes
differ by a factor of ~9.8 for no physical reason.

### 1.3 Moisture variable — different physical quantities, not just different names

| Model | Quantity stored | Units |
|---|---|---|
| Aurora | specific humidity (q) | kg/kg |
| Prithvi | specific humidity (QV) | kg/kg |
| GraphCast | specific humidity (q) | kg/kg |
| FourCastNet | **relative humidity (r)**, derived from ERA5 q+T via a Magnus/Clausius-Clapeyron approximation, since FCN itself only outputs RH | % |

FCN's moisture number is not directly comparable in magnitude to the other
3 models' — only *relative* comparisons (% RMSE improvement, sign of bias)
are meaningful across all 4. Don't plot raw kg/kg next to raw % on one axis.

### 1.4 Prithvi's pressure levels are approximations, not exact matches

Prithvi has only 14 native model levels; there is no exact 500/850/300 hPa
level. The labels used throughout this study are the *nearest available*
level:

| Label used | Actual level | Actual pressure |
|---|---|---|
| "500 hPa" (`T_lev51`, `H_lev51`, `U_lev51`) | level 51 | **~532 hPa** |
| "850 hPa" (`T_lev63`) | level 63 | **~832 hPa** |
| "300 hPa" (`T_lev45`, `U_lev45`) | level 45 | **~310 hPa** |

A ~30 hPa offset is small but real. If you're building a precise
multi-model vertical comparison, either note this explicitly in the figure
or accept it as a known approximation — don't treat Prithvi's number as
exactly co-located with the other 3 models' literal pressure surface.

### 1.5 Grid orientation and longitude range differ by model

| Model | Lat order | Native lon range |
|---|---|---|
| Aurora | descending, 90→-89.75 | 0–360° |
| FourCastNet | descending, 90→-90 | 0–360° |
| GraphCast | **ascending**, -90→90 | 0–360° |
| Prithvi | ascending, -89.5→90 | 0–360° (converted 2026-07-15; native was ±180°, see `map.md` §5) |

All 4 models' spatial `bias/values/*.csv` files now use the same `lat, lon`
column convention (0–360° longitude), so any single-model spatial plot can
read the CSV directly with no reprojection. But the underlying **grid
resolution and point locations still differ per model** (720×1440 down to
181×360) — if you're building a *combined* multi-model spatial map (not
just separate panels), you must resample every model onto one common grid
first. Don't index-align raw arrays from different models expecting them
to line up.

---

## 2. Known data-quality caveats — real, already-investigated, don't re-flag as new bugs

### 2.1 GraphCast's "trivial" self-insertion never reaches ~0/100% the way the other 3 models do

E.g. Wind-Ins→U500 mature RMSE is ~0.14 m/s for Aurora/FCN/Prithvi, but
GraphCast's own trivial diagonal sits at 86–93%, not ~100%, consistently
across two independently-written scripts (`nrmse` and `coupling`).
Investigated and confirmed genuine (RMSE drops cleanly at day 30 and
stabilizes at a real, non-noisy lower plateau) — likely a regrid residual
or "soft nudge" characteristic of how the GraphCast insertion runs were
generated upstream, not an analysis bug. See `map.md` §6.

### 2.2 One stale leftover file in FCN's coupling/

`FCN_coupling_summary_table5_8.csv` and `Figure_FCN_global_nrmse_timeseries.png`
are leftovers from an earlier/different version of the coupling script,
with different (outdated) numbers than the current `compute_coupling.py`
output. Present in the directory but don't source data from them — use
`FCN_coupling_matrix.csv`/`FCN_coupling_summary.csv` instead. See `map.md` §3.5.

### 2.3 Aurora's `nrmse/plots/` is empty by design

That pipeline is CSV-only — no figures are generated at that stage. Not a
missing-file bug. (Any global-NRMSE figure comes from
`Cross_Model_Synthesis/nrmse_comparison/` instead.)

### 2.4 "Trivial" cells are not findings — exclude or label them

Any insertion experiment's own inserted variable trivially improves to
near-0 RMSE / near-100% by construction (that's the point of forcing it to
truth). Plotting these as if they were a real result is misleading. The
convention used throughout this study (see `Cross_Model_Synthesis/latband_latslice_crossvar/`)
is to only show the **non-inserted, cross-variable** response:

| Insertion experiment | Show these variables | Exclude (trivial) |
|---|---|---|
| T-Ins | U, V, Z/H | T |
| Wind-Ins | T, Z/H | U, V |
| Geo-Ins | T, U, V | Z/H |

If you're building a new figure directly from the per-model CSVs (not
reusing an existing combined one), apply this same exclusion rule, or
clearly label trivial cells as such (e.g. grey them out) rather than
showing a ~100% bar next to genuinely-interesting ~20-80% bars.

### 2.5 Bug 3's Aurora/FCN pre-day-30 TUV divergence (unresolved, documented)

Per `map.md` §5 Bug 3 area / `common/MEMORY.md`: Aurora's and FCN's
combined-insertion (TUV) forecast files diverge slightly from Control
*before* day 30, when they should be bit-identical (insertion hasn't
started yet). Root-caused to `bfloat16` GPU inference non-determinism
across separately-run experiment variants — not fixable without a GPU
rerun. If plotting the early (day 0–30) portion of any TUV timeseries for
Aurora or FCN, expect a small, non-physical discrepancy there; it is not a
sign of a new bug in this data.

---

## 3. Structural/schema conventions

### 3.1 Run-key naming differs per model

| Experiment | Aurora | FCN | GraphCast | Prithvi |
|---|---|---|---|---|
| Control | `control` | `control` | `control` | `Control` |
| T-Ins | `t_ins` | `t_ins` | `t_ins` | `temp_ins` |
| Wind-Ins | `uv_ins` | `wind_ins` | `uv_ins` | `wind_ins` |
| Geo-Ins | `z_ins` | `z_ins` | `z_ins` | `geo_ins` |

Column/filename lookups must use the right key per model — don't assume
one naming scheme works for all 4.

### 3.2 Mature phase = day 30–60, but the step index differs by model

Aurora/FCN: 12-hourly cadence, steps 60–119. GraphCast/Prithvi: 6-hourly
cadence, steps 120–239. **Always filter by the `lead_day`/`lead_hour`
column value** (e.g. `lead_day >= 30`), never by a hardcoded step index —
the same index number means a different lead time in different models.

### 3.3 Adjustment-phase (day 0–30) responses are only real for Aurora and GraphCast

FCN and Prithvi implement insertion as a day-30 restart-type perturbation,
so their pre-day-30 values are trivially ≈0 improvement by construction
(nothing has been inserted yet). Aurora and GraphCast insert continuously
from the initial condition, so their adjustment-phase numbers are genuine.
Don't compare "adjustment phase" bars across all 4 models as if they mean
the same thing — for 2 of the 4 models it's not a measurement, it's a
tautology.

### 3.4 `lats/` vs `lat_bands/` naming is swapped from what it suggests

`lats/` holds the **broad** 3-band (Tropics/Midlatitudes/Polar) data.
`lat_bands/` holds the **fine** 4-slice (2°N/14°N/40°N/60°N) data. Inherited
from the original pipeline's naming, kept as-is for citation consistency.
Double-check which one you're reading from before labeling a new figure.

---

## 4. Variable coverage — don't assume uniformity, check first

Not every variable exists for every model in every metric. Concretely
confirmed gaps and their status:

- **Prithvi's `bias/` pipeline** originally covered only `T2M, SLP, T_lev51,
  H_lev51, U_lev45` — no true "500 hPa" wind (U_lev51) until it was added
  2026-07-15 (see `map.md` §5). It's there now, but this is a live example
  of the general risk: **before assuming a variable is available for all 4
  models, check that model's actual variable list** (documented per-metric
  in `map.md` §3) rather than assuming symmetry.
- **GraphCast has no TUV (combined T+UV) run** — `Cross_Model_Synthesis/tuv_synergy/`
  only covers Aurora/FCN/Prithvi.
- **GraphCast's `wind10`** was NaN until fixed 2026-07-15 (`map.md` §5 Bug 4)
  — now populated, but if you're working from an older un-synced copy of
  this directory, check it isn't still NaN there.
- If a variable you want isn't in the per-model CSVs, it likely genuinely
  wasn't computed for that model — check `map.md` §3's per-metric variable
  lists before assuming it's a missing-file bug, and check the model's
  `scripts/` folder to see whether computing it fresh (from the original
  prediction files, listed in this project's file-path inventory) is
  feasible before concluding it can't be done.

---

## 5. When in doubt

`map.md`'s "Data-quality history" (§5) and "Known, deliberate
non-uniformities" (§6) sections are the canonical record of every
correctness issue found and fixed (or found and explained) in this data.
If a number looks surprising, check there first before assuming either
"this is a new bug" or "this must be fine."
