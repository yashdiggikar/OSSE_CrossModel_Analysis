# Organized_results — Directory Map

This directory is a **curated, read-only copy** of the thesis Chapter 5 OSSE
(truth-insertion) analysis outputs, reorganized for browsing. Every file here
is a `cp` of an original — nothing was moved, nothing was deleted from the
source directories. Source directories: `Aurora_Pred_Files/`, `FourCastNet/`,
`gc_pred_files/`, `prithvi_new/`, and `common/` (all under
`/umbc/rs/pi_halem/users/yashd1/`).

If a plot or CSV here looks wrong, the fix belongs in the **source**
directory (then re-copy here) — don't edit files in `Organized_results/`
directly, they'll just get overwritten next sync.

Last verified consistent with source: 2026-07-15 (all values/plots/scripts
counts match source exactly in both `Organized_results/` and the companion
`Organized_plots/common/`; every model's "trivial" self-insertion values
spot-checked as physically sane; zero stale plots found anywhere; see
"Data-quality history" section for the 4 real bugs and 2 portability/gap
improvements made along the way).

**2026-07-17 addition:** five new directories — `cloud/`, `terrain/`,
`lat_band/`, `lat_slice/`, `tuv/` — were added, curating `common/`'s
cloud-regime/terrain/latitude-stratified and TUV-synergy analyses the same
way §3 already curates the per-model metrics. They live under
**`Organized_results/common/`** (added as their own subdirectory shortly
after the initial build, to keep them visually grouped and out of the
top-level listing alongside the per-model dirs). See §4a below. Two real,
previously-undocumented bugs were found and fixed as part of this pass —
see §5, Bug 5 and Bug 6.

⚠️ **Naming disambiguation** — `Organized_results/common/` (this
subdirectory) is NOT the same thing as `/umbc/rs/pi_halem/users/yashd1/common/`
(the *source* directory referenced throughout this file and in §4a, one
level up from `Organized_results/` itself). Every reference to `common/tcc/`,
`common/terrain2/`, `common/lats_FINAL/`, etc. below means the source
directory; every reference to `Organized_results/common/cloud/` etc. means
this curated copy. The names collide because the curated copies happen to
cover exactly the source directory's content — it is not the same folder.

**2026-07-17 second addition:** `Organized_results/extended_exp/` — a
*different* study (land-only/ocean-only insertion, not cloud/terrain/
latitude), sourced from `CC_aurora/`, `fcn_inference/`, `prithviOSSE/`
(NOT from `common/`). Has its own `extended_exp/map.md` — see that file for
full detail, this file's §3/§4a don't cover it. Fully self-contained,
including newly-extracted propagation-map arrays that didn't previously
exist as reusable files (see `extended_exp/map.md` §5).

See also, in this same directory: `replotting_instructions.md` — a checklist
of unit, convention, and data-quality caveats to check *before* building
any new figure from this data, independent of style; and `file_paths.md` —
every original prediction file and truth-data path this study's numbers
trace back to (not needed for replotting, only for recomputing).

---

## 1. What this study is

Four AI weather models (**Aurora** — Microsoft 0.25°, **FourCastNet v2**
(FCN) — NVIDIA, **GraphCast-Small** (GC) — Google DeepMind, **Prithvi WxC**
— NASA/IBM) are each run as a 60-day free-running forecast (**Control**),
then re-run with a single truth field forced ("inserted") into the forecast
starting at day 30: temperature (**T-Ins**), wind U+V (**Wind-Ins**), or
geopotential/height (**Geo-Ins**). The analysis asks: how much does forcing
one field correct the *other*, non-inserted fields (cross-variable
coupling), and does that correction hold up spatially, by latitude, by
terrain, and by cloud regime?

| Model | Grid | Lat convention | IC (init condition) | Cadence / steps | Truth source |
|---|---|---|---|---|---|
| Aurora | 720×1440 (no S. pole row) | descending, 90→-89.75 | 2023-02-01 12Z | 12h × 120 = 60d | ERA5 |
| FourCastNet | 721×1440 (both poles) | descending, 90→-90 | 2023-02-01 00Z | 12h × 120 = 60d | ERA5 |
| GraphCast-S | 181×360 (1°) | **ascending**, -90→90 | 2022-01-01 06Z | 6h × 240 = 60d | ERA5 (0.25°→1° regrid) |
| Prithvi WxC | 360×576 (0.5°) | ascending, -89.5→90 | 2022-01-01 00Z | 6h × 240 = 60d | MERRA-2 |

**Run-key naming differs per model** (same experiments, different strings
in the CSVs/scripts):

| Experiment | Aurora | FCN | GraphCast | Prithvi |
|---|---|---|---|---|
| Control | `control` | `control` | `control` | `Control` |
| Temperature insertion | `t_ins` | `t_ins` | `t_ins` | `temp_ins` |
| Wind insertion | `uv_ins` | `wind_ins` | `uv_ins` | `wind_ins` |
| Geopotential insertion | `z_ins` | `z_ins` | `z_ins` | `geo_ins` |

**Mature phase** (the window almost every "% improvement" number is
computed over) = day 30–60, i.e. steps 60–119 for Aurora/FCN, steps
120–239 for GraphCast/Prithvi.

---

## 2. Top-level layout

```
Organized_results/
├── map.md                        ← this file
├── Aurora/            {nrmse, lats, lat_bands, bias, coupling}/{values, scripts, plots}
├── FourCastNet/       (same 5 metrics)
├── GraphCast/         (same 5 metrics)
├── Prithvi/            (same 5 metrics)
├── Cross_Model_Synthesis/
│   ├── latband_latslice_crossvar/  {scripts, plots}   (no values — see §4)
│   ├── nrmse_comparison/            {scripts, plots}   (no values — see §4)
│   └── tuv_synergy/                 {values, scripts, plots}
└── common/                        ← NOT the same as ../common/, see warning above
    ├── cloud/
    │   ├── T_Ins/    {values, scripts, plots}   ← tcc/ (T-Ins × cloud regime)
    │   ├── Wind_Ins/ {values, scripts, plots}   ← cloud2/ (UV-Ins × cloud regime)
    │   └── PHYSICAL_ANALYSIS_{T,UV}_Ins.txt
    ├── terrain/
    │   ├── T_Ins/    {values, scripts, plots}   ← terrain/ (T-Ins × terrain class)
    │   ├── Wind_Ins/ {values, scripts, plots}   ← terrain2/ (UV-Ins × terrain class)
    │   └── PHYSICAL_ANALYSIS_{T,UV}_Ins.txt
    ├── lat_band/    {values, scripts, plots}  ← broad Tropics/Midlat/Polar, see §4a
    ├── lat_slice/   {values, scripts, plots}  ← fine 2N/14N/40N/60N, see §4a
    └── tuv/         {values, scripts, plots}  ← T+UV synergy study, see §4a
```

All five §4a directories are **fully self-contained as of 2026-07-17**: every
script that lives in a `scripts/` folder here reads only from the `values/`
folder one level up in the SAME directory and writes only to the sibling
`plots/` folder — verified by grepping for stray absolute paths (zero hits)
and by actually running every such script and diffing its output against
the pre-existing values (byte-identical in every case; see §4a.6). The
`fig5_75_76_temp_ins_cloud_regime.py`-style "main" pipeline scripts (one per
`T_Ins`/`Wind_Ins` folder) are the one exception — they read raw multi-GB
prediction/truth files from outside `Organized_results/` entirely and are
kept only for provenance/full-recompute reference, not for self-contained
replotting; use the `replot_*.py` script in the same folder for that.

Note: `common/cloud/`, `common/terrain/`, `common/lat_band/`, `common/lat_slice/`,
`common/tuv/` (§4a) partially
overlap in subject matter with `Cross_Model_Synthesis/latband_latslice_crossvar/`
and `Cross_Model_Synthesis/tuv_synergy/` (§4) — both trace back to the same
`common/lats_FINAL/` and `common/tuv_plots/`+`tuv_final/` sources. The §4a
versions are newer (2026-07-17), self-contained (see above), and add real
`values/` CSVs where §4's versions had none (`latband_latslice_crossvar/values/`
is still empty — see §4.1). Prefer §4a for anything new; §4's folders are
kept for continuity
with anything already citing them.

Every metric folder always has exactly 3 subfolders: `values/` (CSVs / .npy
— the numbers actually plotted), `scripts/` (every `.py` responsible for
producing those values and/or plots), `plots/` (every `.png`/`.pdf`
currently on disk for that metric).

---

## 3. The 5 per-model metrics

### 3.1 `nrmse/` — Global RMSE / NRMSE (no thesis section number)

Cosine-latitude-weighted RMSE over the **whole globe**, for a handful of
headline variables (T500, U500, V500, Z500, Wind500, Wind10 + model-specific
extras), Control vs each insertion run, normalized to NRMSE. This is the
simplest, coarsest metric in the set — a first-look "did insertion help at
all" sanity check before the more detailed breakdowns below.

- **values/**: `{Model}_{var}_global_rmse.csv` / `_nrmse.csv` (one row per
  timestep, one column per run) + a `..._std_normalization.csv` /
  `..._era5_std_normalization.csv` table of σ used to normalize.
  Prithvi additionally has `rmse_{VAR}.csv`/`nrmse_{VAR}.csv` for 11
  variables (T2M, SLP, U10M, V10M, TQV, T850, T500, T300, H500, U300,
  Wind500) since its main script (`compute_prithvi_nrmse.py`) tracks more
  fields than the other three models' equivalents.
- **plots/**: **empty for Aurora, FourCastNet, and GraphCast** — these
  pipelines are CSV-only, no figures are generated at this stage (any
  NRMSE figure you see elsewhere, e.g. `Figure_5_11_*` or `fig_nrmse_*`,
  comes from `Cross_Model_Synthesis/nrmse_comparison/`, not from here).
  **Prithvi has 12 PNGs** (`rmse_{VAR}.png` per variable + `rmse_summary_bar.png`)
  because `compute_prithvi_nrmse.py` plots inline as it goes.
- **scripts/**:
  - Aurora: `compute_aurora_nrmse.py` (main, 5 core vars) +
    `compute_q_nrmse_aurora.py` (adds q500) + `patch_v500_aurora.py` (adds
    v500) + `patch_v500_aurora_lats.py` (cross-cutting — *also* writes
    v500 rows into `lats/values/` and `lat_bands/values/`, hence it also
    appears in those two scripts/ folders).
  - FCN: `compute_global_nrmse.py` (main) + `compute_q_nrmse_fcn.py` (adds
    derived RH500) + `patch_v500_fcn.py` + `patch_v500_fcn_lats.py`
    (cross-cutting, same v500-into-lats pattern as Aurora).
  - GraphCast: `compute_gc_nrmse.py` (main) + `compute_q_nrmse_gc.py` +
    `patch_v500_gc.py` + `patch_v500_gc_lats.py` (cross-cutting) +
    `patch_wind10_gc.py` (2026-07-15, fixes a wrong GRIB filter key that
    left `wind10` entirely NaN — see §5 Bug 4).
  - Prithvi: `compute_prithvi_nrmse.py` (main, 11 vars) +
    `fix_prithvi_t2m_slp_t850_t300_nrmse.py` (2026-07-14, fixes the 4 vars
    corrupted by the Control-mislabeling bug) + `fix_prithvi_t500_u500_v500_h500_tqv_nrmse.py`
    (2026-07-14, fixes a *second*, independent bug — see §5) +
    `patch_uv_lev51_prithvi_lats.py` (cross-cutting into lats/lat_bands) +
    `regenerate_stale_t_plots.py` (2026-07-14, surgical replot of the 7
    PNGs that predated one or both fixes — see §5). Two earlier scripts,
    `fix_prithvi_t500_nrmse.py`, `fix_prithvi_all_nrmse.py`, and
    `patch_uv500_prithvi.py`, are **no longer present** — their output was
    confirmed wrong (see §5) and is now fully superseded by
    `fix_prithvi_t500_u500_v500_h500_tqv_nrmse.py`.

### 3.2 `lats/` — Latitude-BAND analysis (broad regimes) — thesis §5.7

Same Control-vs-insertion RMSE comparison as above, but split into 3
zonal, hemisphere-pooled bands: **Tropics** (|lat|<30°), **Midlatitudes**
(30–60°), **Polar** (60–90°). Reveals whether a model's cross-variable
coupling is dynamically organized (e.g. geostrophic balance should be
stronger at higher latitude, where the Coriolis parameter is larger).

- **values/**: `{Model}_{run}_{var}_latband_rmse.csv` per run×variable, plus
  a `..._pct_improvement_summary.csv`. Prithvi's naming is inverted:
  `latband_rmse_{var}_{run}.csv`.
- **plots/**: 5 figures — a "setup" map showing the 3 bands
  (`Figure_5_35...` / `Figure_GC_latband_setup` / not present for Prithvi
  under this name), 3 time-series figures (one per insertion experiment,
  showing only the *cross*-variables — e.g. T-Ins shows U500/V500/Z500,
  not T500 itself), and 1 bar-chart summary.
- **scripts/**: the model's own `section_5_7_*_latband_analysis.py` (computes
  values + first-pass plots) + `regenerate_lat_plots.py` (a `common/`
  script that **overwrites these same plot files in place** with an
  improved style — fixed y-scale across panels, U500/V500 shown
  independently instead of combined "Wind500" — confirmed via file mtimes
  that its output is what's actually on disk today for Aurora/FCN/GraphCast/Prithvi)
  + the cross-cutting v500/U-V-lev51 patch script (adds the extra variable
  the main script's original variable list didn't cover). FCN originally
  also had its own `replot_5_7.py` (an intermediate refinement, analogous
  to `bias/`'s `replot_5_8.py`) — **removed**, since mtimes showed its
  output was itself fully overwritten by the later `regenerate_lat_plots.py`
  run, so nothing on disk is actually attributable to it (unlike
  `replot_5_8.py`, whose output was confirmed still current — see `bias/`
  below).

### 3.3 `lat_bands/` — Latitude-SLICE analysis (fine regimes) — thesis §5.7 companion

**Naming gotcha inherited from the original pipeline:** despite the
directory name, this is the *finer-grained* companion to `lats/`, not a
"bands" analysis — it uses 4 narrow (±5°) reference latitudes instead of 3
broad bands: **2°N** (ITCZ), **14°N** (subtropics), **40°N** (storm
track/jet core), **60°N** (subpolar). This decomposition catches sign
reversals the broad-band average in `lats/` smooths away (e.g. GraphCast's
temperature-insertion benefit to Z500 is positive in the tropics but
*negative* at 40–60°N — invisible in the 3-band view).

- **values/**: `{Model}_{run}_{var}_latslice_rmse.csv` (Prithvi:
  `latslice_rmse_{var}_{run}.csv`) + a pct-improvement summary.
- **plots/**: same 5-figure structure as `lats/` (setup map, 3 experiment
  time-series, 1 bar chart), labeled `Figure_LS1`–`Figure_LS5` for
  Aurora/FCN (not thesis-numbered 5.x, since these are the "slice"
  companion figures) or `Figure_GC_latslice_*` / `latslice_rmse_*` for
  GraphCast/Prithvi.
- **scripts/**: `section_5_7_*_latslice_analysis.py` (or
  `section_gc_latslice_analysis.py`) + `regenerate_lat_plots.py`
  (overwrites in place, same as `lats/`) + the cross-cutting patch script.

### 3.4 `bias/` — Spatial bias analysis — thesis §5.8

2D maps of mean bias (forecast − truth) over the mature phase, for each
run; bias-*reduction* maps (`|Control bias| − |insertion bias|`, positive =
improved) for each insertion; and a 9-class categorical "failure-mode" map
classifying, per grid cell, which insertion (if any) helps most/least.

- **values/**: Aurora/FCN/GraphCast store this as flattened-grid CSVs with
  explicit `lat, lon, bias` columns — `{Model}_control_{var}_bias.csv`,
  `{Model}_{run}_{var}_combined.csv` (control/insertion/reduction columns),
  `{Model}_failure_mode_map.csv`, `{Model}_failure_mode_area_summary.csv`
  (37 files each). **Prithvi originally stored the same information as raw
  2D `.npy` arrays** (no embedded coordinates — required knowing the grid
  convention from §1 to reconstruct lat/lon axes). Converted 2026-07-15 to
  the same self-describing `lat, lon, bias` CSV format the other 3 models
  use (`bias_{run}_{var}.csv`, 24 files, one per run×variable; longitude
  remapped from Prithvi's native ±180° range to the 0–360° range the other
  3 models already use, sorted by lat/lon), via `convert_bias_npy_to_csv.py`
  — see §5. Plus one `bias_summary.csv` (unchanged, was already a plain
  summary table). All 4 models' spatial bias data is now uniformly
  self-describing — no grid-convention lookup needed for any of them.
  **U_lev51 (~532 hPa, the "500 hPa" wind proxy) was also added** — the
  original Prithvi `bias/` pipeline tracked wind only at U_lev45 (~310 hPa),
  unlike Aurora/FCN/GraphCast which all have true U500, so a genuine 4-way
  U500-bias comparison wasn't previously possible. Computed 2026-07-15 via
  `compute_prithvi_bias_u_lev51.py` (identical mature-phase/cosine-weighted
  methodology as the other 5 variables) — see §5. `U_lev45` (~310 hPa) is
  still present too; it's a different level, not a duplicate.
- **plots/**: two generations of figures merged into one folder here:
  - The **original** run's output (`Figure_5_43`–`Figure_5_47` for
    Aurora/FCN, equivalent numbering for GraphCast/Prithvi) — 10 files for
    Aurora/FCN/GraphCast (PNG+PDF pairs), 6 for Prithvi (PNG only, includes
    `bias_map_U_lev51.png` added 2026-07-15 — see above) — from the model's
    own `section_5_8_*_bias_analysis.py` (Prithvi additionally has the
    standalone `compute_prithvi_bias_u_lev51.py`).
  - A **second, more polished set** (`Figure_{Prefix}_ctrl_bias_{var}.png`,
    `Figure_{Prefix}_{run}_bias_{var}.png`, `Figure_{Prefix}_failure_mode_map.png`)
    from `common/regenerate_bias_plots.py`, which reads the same values
    and writes into a *separate* `plots2/` directory (does **not**
    overwrite the original plots) — 25 files for Aurora/FCN/GraphCast, 20
    for Prithvi. Both sets are merged flat into this folder's `plots/`.
- **scripts/**: `section_5_8_*_bias_analysis.py` (or `section_gc_bias_analysis.py`)
  + `regenerate_bias_plots.py` (the source of the plots2-style figures —
  this was, until this cleanup, an "orphaned" figure set with no known
  generating script; mystery solved). FCN additionally has `replot_5_8.py`,
  a legitimate third-generation refinement of the *original* plot set
  (higher DPI, clearer legends) — confirmed via mtime that its output is
  what's still on disk (i.e. it was **not** itself superseded by anything
  later).

### 3.5 `coupling/` — Cross-variable coupling matrix (own thesis section, Fig 5.30 / Table 5.8)

The most quantitative summary metric: for every insertion experiment (A)
and every response-variable group (B ∈ {T, U, V, Z/H, Q}), a single
coupling score `C(A→B) = mean over all pressure levels of
100×(RMSE_ctrl(B) − RMSE_insA(B)) / RMSE_ctrl(B)`, computed at every
available pressure level (13 for Aurora/FCN/GraphCast, 14 for Prithvi's
native MERRA-2 levels) and averaged. This is the only metric that goes
beyond 500 hPa.

- **values/**: `{Model}_coupling_matrix.csv` (the headline A×B table),
  `{Model}_coupling_per_level.csv` (full vertical breakdown), and
  `{Model}_coupling_summary.csv` (top/second/weakest response per
  insertion). 3 files for Aurora/GraphCast/Prithvi; **FCN has 4** — the
  4th (`FCN_coupling_summary_table5_8.csv`) is a **stale leftover** from an
  earlier/different version of the coupling script, timestamped ~20
  minutes before the current `compute_coupling.py`'s own output and
  containing different numbers (e.g. T-Ins→Z500 30.7% there vs 29.9% in
  the current matrix) — kept here because it's genuinely on disk in the
  source, but don't cite it.
- **plots/**: heatmap (Fig 5.30 style), bar chart, vertical coupling
  profile, and a rendered summary table image — 4 files for
  Aurora/GraphCast/Prithvi, 5 for FCN (includes the same stale
  `Figure_FCN_global_nrmse_timeseries.png` leftover).
- **scripts/**: one script each — `compute_aurora_coupling.py`,
  `compute_coupling.py` (FCN), `compute_gc_coupling.py`,
  `compute_prithvi_coupling.py`. No patch/cross-cutting scripts here.

---

## 4. `Cross_Model_Synthesis/` — combined 4-model figures

Unlike §3, these three folders hold figures where **all 4 models appear as
bars/lines on the same plot** — they don't split naturally into
per-model folders, hence their own top-level location. Two of the three
are plot-only (no dedicated values file of their own — their numbers live
in the per-model `nrmse/values/`, `lats/values/`, `lat_bands/values/`
folders under §3, which these scripts read directly at plot time).

### 4.1 `latband_latslice_crossvar/`
The redesigned cross-variable latitude figures: **one chart per insertion
experiment** (not per model), 4 models as grouped bars, showing only the
non-inserted ("cross") variables. Source name in `common/`: `lats_FINAL/`.
- **plots/** (12): `Figure_latband_{T,Wind,Geo}_Ins_crossvar.png` +
  `_mature.png` variants (full 2-row adjustment+mature layout vs
  mature-phase-only), and the same 6 for `latslice`.
- **scripts/**: `plot_latband_by_experiment.py` (reads from every model's
  own `lats/values/`+`lat_bands/values/` under §3, no local CSV cache).
- **values/**: empty — this is the one genuine "you'd have to reconstruct
  it from the per-model CSVs" gap identified in this project; nothing was
  lost, it's just that this particular script never saves its assembled
  intermediate table.

### 4.2 `nrmse_comparison/`
Global NRMSE % improvement, one figure per experiment (T/Wind/Geo
insertion), all 4 models compared. Two independent scripts producing two
visually different but conceptually overlapping figure sets:
- **plots/** (6): `Figure_5_11_{T,Wind,Geo}_insertion_NRMSE_improvement.png`
  (from `plot_nrmse_comparison.py`, saved at `common/` top level) and
  `fig_nrmse_{t,uv,z}_insertion.png` (from `plot_nrmse_improvements.py`,
  saved under `common/nrmse_plots/`).
- **scripts/**: both of the above. Same "no local values" situation as §4.1
  — both read directly from each model's own `nrmse/values/*.csv`.

### 4.3 `tuv_synergy/`
The combined **Temperature+Wind ("TUV") insertion** study — does inserting
both at once beat the better of the two single-variable insertions?
`Synergy = improvement(T+UV) − max(improvement(T), improvement(UV))`.
GraphCast is **not** part of this study (no TUV run exists for it — only
Aurora/FCN/Prithvi). This is the one Cross_Model_Synthesis folder with real
`values/`.
- **values/** (25): `{model}_{ctrl,tins,uins,tuv}_rmse.csv` (cached global
  RMSE time series per run), `{model}_synergy_{var}.npy` /
  `{model}_impr_tuv_{var}.npy` (spatial synergy fields for Z500-equivalent
  and Q850-equivalent), and `table5_7_tuv_summary.csv` (the Table 5.7
  headline numbers).
- **plots/** (14): the original **Fig 5.24–5.29** set (T500/U500 timeseries,
  Z-field-at-3-levels, vertical improvement heatmap, spatial synergy maps,
  synergy bar chart, lat-pressure synergy heatmap — covers 7 variables) +
  the **scoped-down final versions** (`{Model}_{Z500,Q850}_RMSE_timeseries.png`,
  `fig5_27/28_..._z500_q850.png` — Z500/Q850 only, cleaner layout, used as
  the actual thesis deliverable per `common/tuv_final/PHYSICAL_ANALYSIS.txt`).
  Both sets are current and correct as of 2026-07-14 (see §5).
- **scripts/**: `compute_tuv_rmse.py` (main cache builder, all 3 models),
  `recompute_prithvi_tuv_rmse.py` (Prithvi-only re-implementation with two
  bug fixes applied — see §5), `plot_tuv_analysis.py` (produces the full
  Fig 5.24–5.29 set from cache), `plot_z500_tuv_timeseries.py` /
  `plot_q850_tuv_timeseries.py` / `plot_synergy_z500_q850.py` (produce the
  scoped-down final set).

---

## 4a. `Organized_results/common/{cloud,terrain,lat_band,lat_slice,tuv}` — stratified analyses (added 2026-07-17)

These curate `common/`'s cloud-regime, terrain-class, latitude, and TUV
synergy analyses — the same content §3's per-model dirs cover for the 5
core metrics, but for the "stratify by environment" family of experiments,
which previously only lived in `common/` with no `Organized_results/` copy.
`cloud/` and `terrain/` are split internally into `T_Ins/` and `Wind_Ins/`
subfolders (one per source sub-experiment); `lat_band`/`lat_slice` are split
from one source (`lats_FINAL/`) by figure type instead; `tuv/` merges the
full and scoped-down TUV figure sets into one flat folder. **All five are
fully self-contained** (2026-07-17 update, see §4a.6) — every `values/`
folder holds every CSV/NPY its sibling `scripts/` need, and every
`replot_*.py`/`plot_*.py` script reads and writes only within its own
directory, verified by actually running each one.

### 4a.1 `cloud/` — cloud-regime stratified (merges `common/tcc/` + `common/cloud2/`)
T-insertion → U500/V500/Z500/Q700 response (`tcc/`) and UV-insertion →
T500/Z500/Q850 response (`cloud2/`), both stratified by ERA5 cloud regime
(Clear/Cirrus/Mixed/Opaque). Split into `T_Ins/` and `Wind_Ins/` subfolders,
each with its own `values/scripts/plots`.
- **`T_Ins/values/`** (5): `table_t_ins_cloud_regime_summary.csv` (combined
  4-model table) + `{model}_t_ins_regime_rmse.csv` ×4 models (per-model
  summary CSVs, which are just a row-filter of the combined table, were not
  duplicated in). **`Wind_Ins/values/`**: same pattern for `uv_ins`.
- **`T_Ins/scripts/`**: `fig5_75_76_temp_ins_cloud_regime.py` (the "main"
  pipeline — reads raw multi-GB prediction/ERA5/TCC files from OUTSIDE
  `Organized_results/`, kept for provenance only, NOT self-contained),
  `replot_fig5_75_76.py` (self-contained — patched 2026-07-17 to read only
  `../values/*.csv` and write only `../plots/`; verified by running it and
  diffing output), `_regime_terrain_plot_common.py` (shared plot-style
  module, duplicated into every `scripts/` folder that needs it).
  **`Wind_Ins/scripts/`**: `fig_cloud2_uv_ins.py` (main, not self-contained)
  + `replot_fig_cloud2.py` (self-contained) + the same shared module.
- **plots/**: one combined-4-model bar chart + one timeseries per
  sub-experiment (the identical file was previously saved once per model
  subfolder in `common/` — only one copy is kept here) + the matching
  uniform-scale dedup bar chart from `common/tcc_terrain_4/make_4_bar_summaries.py`
  (that script itself is provenance-only, needs the full `common/` tree).
- **`PHYSICAL_ANALYSIS_T_Ins.txt`** / **`PHYSICAL_ANALYSIS_UV_Ins.txt`**
  (top-level, sibling to `T_Ins/`/`Wind_Ins/`): the written interpretation
  for each sub-experiment (source: `tcc/`'s and `cloud2/`'s own
  `PHYSICAL_ANALYSIS.txt`).

### 4a.2 `terrain/` — terrain-class stratified (merges `common/terrain/` + `common/terrain2/`)
Same structure as `cloud/` (`T_Ins/`+`Wind_Ins/` subfolders, self-contained
`replot_*.py` scripts, provenance-only main scripts), stratified instead by
terrain class (Land Low/Mod/High, Ocean, Polar).

### 4a.3 `lat_band/` and `lat_slice/` (source: `common/lats_FINAL/` + each model's own `lats/values`, `lat_bands/values`)
Split from the single `lats_FINAL/` source by figure type — `lat_band` =
the broad Tropics/Midlatitudes/Polar figures (`Figure_latband_*`),
`lat_slice` = the fine 2°N/14°N/40°N/60°N figures (`Figure_latslice_*`).
Unlike `Cross_Model_Synthesis/latband_latslice_crossvar/` (§4.1), these
have real **values/**, organized as `values/{Aurora,FourCastNet,GraphCast,Prithvi}/`
— the per-model raw ingredient CSVs each model's own `lats/values`
(band) or `lat_bands/values` (slice) folder normally holds (132 files
each, copied in full 2026-07-17). **scripts/** holds one standalone,
self-contained script each — `plot_latband.py` / `plot_latslice.py` (split
out of `common/plot_latband_by_experiment.py`, restricted to one analysis
type, reading only the local per-model `values/` subfolders) — which
regenerates all 6 plots AND the summary CSV
(`lat{band,slice}_crossvar_pct_improvement.csv`, saved into `values/`)
purely from local files. Verified byte-identical to the original combined
script's output. Both share the same `PHYSICAL_ANALYSIS.txt` (source:
`lats_FINAL/`'s, which covers band and slice together).

### 4a.4 `tuv/` (merges `common/tuv_plots/` + `common/tuv_final/`)
- **values/** (26 top-level + `Multimodel-study/` subtree): the full
  `tuv_plots/cache/*.csv`+`*.npy` (per-model global RMSE series + spatial
  synergy/improvement fields for Z500-equivalent and Q850-equivalent),
  `table5_7_tuv_summary.{csv,txt}`, plus (added 2026-07-17) 9 files copied
  from the external `Multimodel-study/` directory
  (`{FCNv2,Aurora,PrithviWxC}/{control,temp_ins,wind_ins}/*.csv`) — these
  were a previously-undocumented second external dependency (`tuv_final`'s
  Z500/Q850 timeseries scripts read Control/T-ins/UV-ins curves from
  `Multimodel-study/`, a source outside both `common/` and
  `Organized_results/`, not just from the `tuv_plots/` cache).
- **scripts/**: `compute_tuv_rmse.py`, `recompute_prithvi_tuv_rmse.py`
  (both read raw multi-GB prediction files, provenance-only, not
  self-contained) + `plot_tuv_analysis.py` (full Fig 5.24–5.29 set) +
  `tuv_final/`'s 3 scoped-plotting scripts (Z500/Q850 only) — these 4
  plotting scripts were all patched 2026-07-17 to read only from the local
  `values/` (incl. `values/Multimodel-study/`) and write only to `plots/`;
  self-contained, verified by running each and diffing `table5_7_tuv_summary.csv`
  byte-identical to source.
- **plots/** (16): both the full Fig 5.24–5.29 set and the scoped-down
  final Z500/Q850 versions — see §4.3 for why these are two valid,
  non-redundant sets, not a stale/current pair.
- **PHYSICAL_ANALYSIS.txt**: source `tuv_final/`'s.

### 4a.5 Two bugs fixed while building this section (see §5, Bug 5 and Bug 6, for the full story)
`tcc/` and `terrain/` (the T-insertion experiments) were, until 2026-07-17,
still reading Prithvi's control from an older, pre-OSSE stand-in file
instead of the same corrected `control_forecast_Jan2022.nc` used everywhere
else — fixed and fully recomputed; several of the written findings in
`PHYSICAL_ANALYSIS_T_Ins.txt` changed materially as a result (Prithvi's
Z500 is no longer the study's single strongest result, and no longer peaks
at Land High — see Bug 5). Separately, `cloud2/`'s and `terrain2/`'s
`PHYSICAL_ANALYSIS_UV_Ins.txt` had a stale "Prithvi T500 is invalid,
excluded" caveat left over from a data bug that was actually fixed back on
2026-07-14 — the caveat text was never updated to match; now fixed (Bug 6).

### 4a.6 Self-containment pass (2026-07-17) — what "self-contained" means here, precisely

Originally (2026-07-17) the 5 §4a directories held only "final" values/
plots and copies of the original scripts, unmodified — meaning every script
still had `common/...` (or, for `lat_band`/`lat_slice`, the original
per-model source directories; for `tuv`, also `Multimodel-study/`)
hardcoded as its input path. The data was complete; the scripts, if
actually run, were not — they'd reach outside `Organized_results/`.

Fixed by, for each of the 9 relevant scripts (`replot_fig5_75_76.py`,
`replot_fig_cloud2.py`, `replot_fig_terrain.py`, `replot_fig_terrain2.py`,
`plot_latband.py`, `plot_latslice.py`, `plot_tuv_analysis.py`,
`plot_synergy_z500_q850.py`, `plot_z500_tuv_timeseries.py`,
`plot_q850_tuv_timeseries.py` — 10 total, `plot_latband.py`/`plot_latslice.py`
being new splits of one original script):
1. Copying every CSV/NPY the script actually reads into the local
   `values/` folder (including, for `lat_band`/`lat_slice`, the 132
   per-model per-file source CSVs each, and for `tuv`, 9 files from the
   external `Multimodel-study/` directory that had not been identified as
   a dependency before this pass).
2. Rewriting the script's path constants (`BASE`, `CACHE`, `OUTDIR`, `MM`,
   `TUV_CACHE`, `OUT`, `band_dir`/`slice_dir`) to resolve relative to
   `Path(__file__).resolve().parent` instead of an absolute `/umbc/.../common/...`
   string.
3. Actually running every patched script from within its own directory and
   confirming (a) zero errors, (b) zero references to any path outside the
   script's own directory tree (`grep -n "/umbc" <script>` returns nothing
   for any of the 10), and (c) output numerically/byte-identical to the
   pre-patch original — `pandas.DataFrame.equals()` for every regenerated
   CSV (`lat{band,slice}_crossvar_pct_improvement.csv`, `table5_7_tuv_summary.csv`)
   after normalizing one cosmetic en-dash-vs-hyphen difference caught in
   this process.

What "self-contained" does NOT mean here: the "main" full-pipeline scripts
(`fig5_75_76_temp_ins_cloud_regime.py`, `fig_cloud2_uv_ins.py`,
`fig_terrain_temp_ins.py`, `fig_terrain2_uv_ins.py`,
`compute_tuv_rmse.py`, `recompute_prithvi_tuv_rmse.py`,
`make_4_bar_summaries.py`) are still copied in for provenance but still
require the original raw prediction/truth files (tens of GB, listed in
`file_paths.md`) — making those self-contained would mean copying the raw
study data itself, which defeats the purpose of a curated results
directory. Self-containment here specifically means: **every figure and
every values/ CSV in `cloud/`, `terrain/`, `lat_band/`, `lat_slice/`,
`tuv/` can be regenerated in a different style, or re-verified numerically,
using only files inside that same directory** — which is the practical
thing "replotting from Organized_results alone" (per `replotting_instructions.md`'s
original framing for §2/§3) has always meant.

---

## 5. Data-quality history (why some Prithvi files have odd mixed dates)

### Bug 1 — Control mislabeled as the T-insertion run

Prithvi's pipeline had a real, now-fixed bug worth understanding if a
number looks surprising: for most of its life, several scripts used
`prithviOSSE/run2/output/osse_forecast_Jan2022.nc` — **which is actually
the temperature-insertion run**, not a control — as if it were the
free-running "Control" baseline. From the mature phase onward, that file's
temperature fields are MERRA-2 truth by construction, so any Control-vs-truth
RMSE computed against it for a temperature field was ≈0, making every
comparison run look nonsensically bad or good.

Fix timeline:
- **2026-06-17**: the genuine control run (`control_forecast_Jan2022.nc`)
  was generated for the first time.
- **2026-06-17/18**: `lats/` and `lat_bands/` scripts repointed and re-run.
- **2026-07-14**: `bias/`, `coupling/`, and the remaining `nrmse/`
  variables (T2M/SLP/T850/T300) fixed and re-run; the 3 scripts that still
  referenced the wrong file (`compute_prithvi_nrmse.py`,
  `patch_uv500_prithvi.py`, `patch_uv_lev51_prithvi_lats.py`) had their
  Control path corrected even though they weren't re-run wholesale;
  several **stale plots** that had been generated by "regenerate" scripts
  (`common/regenerate_bias_plots.py`, `common/regenerate_lat_plots.py`)
  *before* the fix — using the still-broken values at the time — were
  identified by mtime and re-run against the corrected data; the TUV
  study's Prithvi cache had a separate, earlier fix (below) but its full
  Fig 5.24–5.29 plot script hadn't been re-run since — done same day.

### Bug 2 — one-step time misalignment in the T500/U500/V500/H500/TQV nrmse fix (found + fixed 2026-07-14)

While verifying that every calculation traces back to the correct
prediction file (not just the correct *path*, but correctly time-matched
data), a second, independent bug was found in `fix_prithvi_t500_nrmse.py`
and `fix_prithvi_all_nrmse.py` (both now removed — see below). Those two
scripts read a **hardcoded** `step_times = IC + 6*(k+1)h` formula and used
it both to fetch MERRA-2 truth *and* to index every `FC_FILES` entry by raw
array position. That formula only matches the old control file's own real
time axis (`prithvi/output/forecast_60day_Jan2022.nc`, confirmed via direct
`h5py` inspection: `time[:5] = [6,12,18,24,30]`) — but `temp_ins`/`wind_ins`/
`geo_ins` (and the current `control_forecast_Jan2022.nc`) all share a
*different* real time axis, one step later (`time[:5] = [12,18,24,30,36]`).
Reading an insertion run's raw index `k` and treating it as lead-hour
`6*(k+1)` therefore silently compared each insertion run's forecast against
MERRA-2 truth from one 6-hour step in the past, across the *entire* series
— not just near the day-30 transition.

**Empirically confirmed** before fixing: `temp_ins` T500 mature-phase RMSE
was **1.48 K** in the delivered `rmse_T500.csv` — it should be ≈0, since
temperature is directly forced to MERRA-2 truth under `temp_ins`. That
1.48 K almost exactly matches the independently-measured ~1.5 K RMS
one-step temporal derivative of T500, confirming the mechanism. Every other
script in `prithvi_new/` was checked and confirmed to read each file's
*actual* `time` array dynamically (not a hardcoded formula), so this bug
was isolated to just these two scripts, not systemic.

**Fix:** `fix_prithvi_t500_u500_v500_h500_tqv_nrmse.py` (2026-07-14)
recomputes T500/U500/V500/H500/TQV using the same dynamic per-file time
matching every other script already used, and the current
`control_forecast_Jan2022.nc`. Post-fix, every "trivial" self-insertion
case is correctly ≈0 (T500 under temp_ins: 0.14 K; U500/V500 under
wind_ins: ~0.14 K; H500 under geo_ins: 1.86 K). `fix_prithvi_t500_nrmse.py`,
`fix_prithvi_all_nrmse.py`, and `patch_uv500_prithvi.py` (whose only job,
U500/V500, is now fully superseded) were removed from
`Organized_results/Prithvi/nrmse/scripts/` since their output was
confirmed wrong, not just stylistically superseded. The corrected values
also propagated into `Cross_Model_Synthesis/nrmse_comparison/` (both
`plot_nrmse_comparison.py` and `plot_nrmse_improvements.py` read these same
5 CSVs directly) — both re-run, all 6 figures there are current.

### Bug 3 — TUV study level-index and control-file bugs

A third, independent Prithvi bug, isolated to the **TUV study specifically**
(`common/tuv_plots/compute_tuv_rmse.py`): (a) a Prithvi-native-level-index
was used directly as a MERRA-2 array index without remapping (off by one
position — climatological height differs by hundreds of meters between
adjacent levels), and (b) `PRI_CTRL` pointed at an older, differently-shaped
(361 vs 360 latitude rows) control file, causing a silent 1-row grid
misalignment. Both fixed 2026-06-17/18 in `recompute_prithvi_tuv_rmse.py`
and propagated back into `compute_tuv_rmse.py`.

### Bug 4 — GraphCast wind10 (10m wind speed) all-NaN, wrong GRIB filter key

Found during a final end-to-end verification pass (2026-07-15) that checked
every model's "trivial" self-insertion values (e.g. does Wind-Ins actually
drive its own U/V500 RMSE to ~0?) and separately audited every plot for
staleness relative to its values. Aurora and FCN passed cleanly. GraphCast's
`wind10` (10m wind speed, derived from ERA5 `10u`/`10v`) was **completely
NaN** in `nrmse/values/GC_wind10_global_{rmse,nrmse}.csv` and the
corresponding `lats/`/`lat_bands/` files, in all 3 scripts that compute it
(`nrmse/compute_gc_nrmse.py`, `lats/section_gc_latband_analysis.py`,
`lat_bands/section_gc_latslice_analysis.py`).

**Root cause:** all 3 scripts' `load_era5_surface()` function filtered the
ERA5 surface GRIB file (`ALL_ERA5/2016_2025_surface/era5_2016_2025.grib`,
92 GB, 14,612 timesteps) with `filter_by_keys={"shortName": shortname,
"typeOfLevel": "heightAboveGround"}` — but direct inspection via
`cfgrib.open_datasets()` showed the file's `10u`/`10v` variables are
actually stored with `typeOfLevel="surface"`. The wrong key matched zero
GRIB messages, so `list(da.data_vars)[0]` threw `IndexError: list index out
of range`, caught by a bare `except Exception` and logged as `SKIP surface
10u: list index out of range` — the script continued without crashing,
silently leaving `wind10` as its NaN-initialized default for the rest of
the run.

**Fix:** changed `"heightAboveGround"` → `"surface"` in all 3 scripts.
Re-running the full scripts to pick up the fix repeatedly hung/was killed —
they open this file via `xarray`+`cfgrib` with `indexpath=""` (disabling
cfgrib's index cache to avoid concurrent-run `.idx` collisions), which
forces a full re-scan of the 92 GB file on every open; each of the 3
scripts got through every other variable correctly but never finished
loading `wind10` in reasonable time. Wrote `nrmse/patch_wind10_gc.py`
instead — streams the file message-by-message via the low-level `eccodes`
API, skipping full array decode for the >99% of messages outside
GraphCast's Jan-Mar 2022 forecast window, and writes directly into the 10
existing `wind10` CSV files (global + 4 runs × latband + 4 runs ×
latslice). Ran in 331s (scanned all 44,418 messages, matched all
240×2 target timesteps). Post-fix: `control` mature RMSE = 3.89 m/s,
`uv_ins` = 2.54 m/s — a genuine, sane improvement, consistent with
GraphCast's established imperfect-but-real insertion pattern (see §6).

### Bug 5 — `common/tcc/` and `common/terrain/` reading Prithvi's control from a stale, pre-OSSE stand-in file (found + fixed 2026-07-17)

Found while auditing `common/` for the §4a curation above. `common/cloud2/`
and `common/terrain2/` (the UV-insertion cloud/terrain experiments) already
had their Prithvi `PRI_CTRL` fixed to point at the real
`control_forecast_Jan2022.nc` on 2026-07-14 (Bug 1's fix propagating here
too). `common/tcc/` and `common/terrain/` (the T-insertion counterparts) had
not — both still read `prithvi/output/forecast_60day_Jan2022.nc`, an older,
361-latitude, pre-OSSE run used as a stand-in before the real control
existed (2026-06-17). This wasn't the same mislabeling as Bug 1 (the
stand-in file genuinely is a free-running control, not a disguised
insertion run), so it didn't produce nonsensical values — just
inconsistent-with-everything-else ones, computed against a different,
older control run than every other Prithvi analysis in this study now uses.

**Fix:** repointed `PRI_CTRL` in both scripts to
`prithviOSSE/run2/output/control_forecast_Jan2022.nc` and reran both
(`common/tcc/fig5_75_76_temp_ins_cloud_regime.py`,
`common/terrain/fig_terrain_temp_ins.py`) via SLURM on the `chip-cpu`
cluster (~8 min each; confirmed via `sacct`/log inspection that neither
script actually needs a GPU despite the original `.slurm` files requesting
one — `run_fig5_75_76_cpu.slurm`/`run_terrain_fig_cpu.slurm` were added as
faster-queuing CPU-only alternates).

**Numbers changed materially, not just cosmetically** — the old stand-in
control apparently had different absolute error characteristics than the
real control:
- tcc (cloud regime): Prithvi Z500 mature-phase improvement from T-insertion
  dropped from 91.8–94.6% (old, all 4 cloud regimes) to 48.5–51.9% (new) —
  still the flattest (most regime-independent) Z500 profile of the 4
  models, but no longer the *largest* (Aurora's 55.9–65.5% is now larger in
  every regime). Q700 flipped sign in 2 of 4 regimes (Clear/Cirrus went from
  weakly positive to negative).
- terrain: Prithvi Z500 was previously reported as monotonically increasing
  with elevation, peaking 97.8% at Land High — the single largest
  cross-variable number anywhere in the 4-report series. The corrected data
  shows the opposite shape: Land Mod is Prithvi's actual peak (61.0%), and
  Land High is its *weakest* land class (47.2%) — coinciding with, not
  contrasting against, Prithvi's already-known V500 collapse at Land High
  (still present, -6.9% mature, essentially unchanged by this fix). Land
  High is now understood as a genuine double failure (Z500 AND V500) rather
  than "strong Z500, isolated V500 failure."

`PHYSICAL_ANALYSIS_T_Ins.txt` in the new `Organized_results/common/cloud/` and
`Organized_results/common/terrain/` (§4a.1–4a.2) reflect the corrected findings
throughout (Sections 1, 4, and 6 of each were rewritten with the new
numbers); the `common/tcc/PHYSICAL_ANALYSIS.txt` and
`common/terrain/PHYSICAL_ANALYSIS.txt` source files were fixed in place
first, then re-copied here, per this file's own "fix in source" policy.

### Bug 6 — stale "Prithvi T500 invalid" caveat left in cloud2/terrain2 prose after the underlying data was already fixed (found + fixed 2026-07-17)

`common/cloud2/PHYSICAL_ANALYSIS.txt` and `common/terrain2/PHYSICAL_ANALYSIS.txt`
both explicitly flagged Prithvi's T500 row as invalid and excluded it from
interpretation — correct advice when written (2026-06-18), describing Bug 1
before it was fixed for these two scripts. Bug 1 *was* fixed for
`cloud2/fig_cloud2_uv_ins.py` and `terrain2/fig_terrain2_uv_ins.py` on
2026-07-14 (confirmed via script edit timestamp 18:52 and full CSV/plot
regeneration at 19:04 that day), but the written prose was never
regenerated to match — so as of this audit, both `PHYSICAL_ANALYSIS.txt`
files were describing a data problem that no longer existed, three days
after the fix.

**Fix:** re-derived the actual (valid) Prithvi T500 finding from the
corrected CSVs — wind insertion turns out to *harm* Prithvi's T500 in every
cloud regime and every terrain class (mature-phase -14% to -29% by regime,
-6.6% to -34.6% by terrain), worsening from Adjustment to Mature phase in
every case — and added it as new Section 1.4 (cloud2) / 1.5 (terrain2),
updating the Section 0 caveat, Section 4 model comparison, and Section 6
conclusions to incorporate it, rather than just deleting the old caveat.
This extends the already-known "wind insertion harms Prithvi's Q850"
finding to a second field (T500), strengthening rather than changing the
report's headline conclusion about Prithvi's wind pathway.

### Portability improvement (not a bug) — Prithvi bias maps converted to lat/lon CSV

Raised while checking whether `Organized_results/` alone (no access to the
original source directories) has everything needed to replot every figure
in a different style. It mostly does — every other model's spatial `bias/`
data already ships with explicit `lat, lon` columns, so it can be pivoted
and replotted without knowing anything about the model's native grid.
Prithvi's `bias_{run}_{var}.npy` files were the one exception: raw arrays
with no embedded coordinates, requiring the reader to already know Prithvi's
grid convention (from §1) to reconstruct axes correctly. Not incorrect
data, just less portable than the other 3 models.

**Fix:** wrote `prithvi_new/bias/convert_bias_npy_to_csv.py`, which builds
the same `lats_1d = np.arange(-89.5, 90.5, 0.5)` / `lons_1d =
np.arange(-180.0, 180.0, 360.0/576)` grid the original script used, remaps
longitude to the 0–360° range Aurora/FCN/GraphCast's bias CSVs already use,
and writes `bias_{run}_{var}.csv` (columns: `lat, lon, bias`) for all 20
files. Ran once against `prithvi_new/bias/values/` (the `.npy` files are
left in place there, untouched); `Organized_results/Prithvi/bias/values/`
was updated to carry only the new CSVs (dropping the now-redundant `.npy`
copies, to avoid the two formats coexisting). All 4 models' spatial bias
maps are now uniformly self-describing.

### Data gap closed (not a bug) — Prithvi's missing U500-equivalent bias data

Raised by a direct question about whether U500 spatial bias + bias-reduction
data exists for all 4 models. It didn't: Aurora, FCN, and GraphCast's `bias/`
pipelines all track true U500, but Prithvi's original `bias/` pipeline only
ever computed wind bias at `U_lev45` (~310 hPa) — a different, lower level —
never at `U_lev51` (~532 hPa, the level its own `T_lev51`/`H_lev51` already
use as the "500 hPa" proxy). Not a bug (that variable was simply never in
the original script's scope), but it meant a genuine 4-model U500 comparison
wasn't previously possible.

**Fix:** wrote `prithvi_new/bias/compute_prithvi_bias_u_lev51.py`, using the
identical mature-phase (day 30–60), cosine-latitude-weighted methodology as
Prithvi's other 5 bias variables. Computed for all 4 runs and saved directly
in the new uniform `lat, lon, bias` CSV format (see above) — `wind_ins`
correctly comes out trivially ≈0 (0.000 m/s MAB, since wind is directly
inserted), `temp_ins` shows a small genuine improvement (+0.15 m/s MAB
reduction), `geo_ins` a small genuine degradation (−0.08 m/s). Also added a
matching `bias_map_U_lev51.png` in the same 2×2 style as the other 5
variables' plots. `U_lev45` (~310 hPa) is left in place too — it's a
different level, not superseded.

As of 2026-07-15, every script and figure in the original §3/§4 sections of
this directory reflected the corrected data for Bugs 1-4 above — confirmed
by grepping for the old mislabeled file path across all of
`Organized_results/` (zero hits), re-verifying every values/plots file count
against source, and spot-checking every model's trivial self-insertion
values. As of 2026-07-17, the same is true for the §4a sections
(`Organized_results/common/{cloud,terrain,lat_band,lat_slice,tuv}/`) and
Bugs 5-6 above.

---

## 6. Known, deliberate non-uniformities (not bugs)

- **`lats/` = broad bands, `lat_bands/` = fine slices** — backwards from
  what the names suggest, inherited from the original pipeline, kept as-is
  throughout for consistency with existing scripts/citations.
- **GraphCast's self-insertion never reaches the near-zero RMSE Aurora/FCN/
  Prithvi show** (e.g. Wind-Ins→U500 mature RMSE ~0.14 m/s for the other 3
  models, but GraphCast's own coupling/nrmse trivial diagonal sits at
  86-93%, not ~100%). Verified this is a genuine property of the raw
  GraphCast insertion data, not an analysis bug: the RMSE clearly drops at
  day 30 and stabilizes at a real, lower-but-nonzero plateau (not noisy or
  inconsistent, which a matching/alignment bug would produce), and the
  same ~86-93% figure is reproduced independently by two separately-written
  scripts (`nrmse` and `coupling`). Likely a regrid-residual or "soft nudge"
  characteristic of how the GraphCast insertion runs were originally
  generated (outside this analysis layer) — not something fixable here.
- **FCN `coupling/values/` and `plots/` have one extra stale file each**
  (see §3.5) — harmless leftovers, not cleaned up since they don't
  interfere with the current, correct outputs.
- **Aurora `nrmse/plots/` is genuinely empty** — that pipeline never
  produces figures, not a copy error.
