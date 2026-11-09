# Cross-Model Sensitivity Comparison — Method

Companion to `METHOD_OVERVIEW.md`, `CORRELATION_LENGTH_VERIFICATION.md`, and
`SPECTRAL_BLURRING_ANALYSIS.md`. This document describes the third new
analysis added to strengthen the paper: a **direct, systematic comparison**
of how FourCastNetv2, Aurora, and Prithvi WxC each respond to the *identical*
correlated-perturbation design (same target correlation lengths, same
Ingleby-style magnitude tables, same 240-step/60-day OSSE structure). No
single-model OSSE study can make this comparison — it is specific to this
project's three-model design and is treated here as the primary novel
contribution among the three new analyses.

---

## 1. Motivation

`METHOD_OVERVIEW.md` §6 item 4 already calls for comparing the correlated
vs. uncorrelated perturbation baselines *within* each model. This analysis
asks the orthogonal, model-facing question instead: **given the same
correlated perturbation, do the three architecturally different models
differ in (a) how much forecast skill the truth-insertion recovers, (b) how
quickly that recovered skill decays back toward the free-running control as
lead time grows within the post-injection window, and (c) how fast each
model smooths away the fine-scale structure of the inserted signal?**
(a) and (b) use the RMSE/bias results already computed by each model's
existing `compute_global_rmse_corr.py`; (c) uses
`SPECTRAL_BLURRING_ANALYSIS.md`'s output directly.

---

## 2. Metric 1: Insertion effectiveness (normalizes across models' different
   native RMSE units/scales)

```
effectiveness(t) = (control_rmse(t) - experiment_rmse(t)) / control_rmse(t)
```

computed per lead time `t` within the post-injection window (steps
121-240), per model, per **normalized variable category** (see §4's mapping
table — each model names its channels differently, e.g. FCN/Aurora's
`"T500"` vs. Prithvi's `"T_LEV51"`, so a shared category label is used for
the comparison plots/tables).

`effectiveness = 1` means the experiment forecast exactly matches truth at
that lead time (perfect recovery); `effectiveness = 0` means the correlated
perturbation insertion made no difference vs. the free-running control;
negative values mean insertion made the forecast *worse* than not inserting
anything, which is possible and worth flagging directly if it occurs for
any model.

This is unitless by construction, so a FCN Kelvin-RMSE and an Aurora
Kelvin-RMSE (which need not be numerically comparable due to different
native grids/interpolation/verification-truth matching) become directly,
fairly comparable as "how much of the truth-insertion's benefit survived."

## 3. Metric 2: Damping ratio (how fast the insertion's benefit decays
   within the post-injection window itself)

```
damping_ratio = effectiveness(step 240) / effectiveness(step 121)
```

per model, per variable category. A damping ratio near 1 means the model
"remembers" the inserted truth almost as well 119 steps (~30 days) later as
immediately after insertion; a damping ratio near 0 means the model quickly
reverts toward its own free-running trajectory regardless of the insertion.
This directly probes each architecture's effective memory/persistence of
externally-inserted information — itself a meaningful, citable
model-comparison result independent of raw forecast skill.

## 4. Metric 3: Blurring rate comparison

Pulled directly from each model's `spectral_blur_summary_<experiment>.csv`
(`SPECTRAL_BLURRING_ANALYSIS.md` §2.4's effective-resolution wavenumber),
plotted as effective-resolution-wavenumber vs. lead time, one line per
model, on the same axes, per latitude band. A model whose curve declines
faster is blurring the inserted signal's fine-scale structure faster than
the others.

## 5. Normalized variable-category mapping (resolves each model's different
   channel-naming conventions)

| Category | FCN label | Aurora label | Prithvi label |
|---|---|---|---|
| `temp_upper_air` | `T500` | `T500` | `T500` (~532 hPa, nearest native level to 500 hPa) |
| `temp_surface` | `T2M` | `T2M` | `T2M` |
| `wind_u_upper_air` | `U500` | `U500` | `U500` |
| `wind_v_upper_air` | `V500` | `V500` | `V500` |
| `wind_u_surface` | `U10` | `U10` | `U10M` |
| `wind_v_surface` | `V10` | `V10` | `V10M` |
| `geopotential_upper_air` | `Z500` | `Z500` | `Z500` |
| `humidity_upper_air` | `R500` (relative humidity; ERA5 lacks this variable at every level in this project's archive, so this row is all-NaN/skipped for FCN — not a bug) | `Q500` (specific humidity) | `Q500` (specific humidity) |

(Updated 2026-07-25: all 3 models' upper-air labels now use the same
`"<LABEL><hpa>"` naming at every representative level — Prithvi previously
used its own bare `T_LEV51`/`U_LEV51`/`V_LEV51` labels for this same
~532hPa level; those were renamed to `T500`/`U500`/`V500` when the 3 "vs
truth" scripts were expanded to a full T/U/V/Z/humidity x 300/500/1000hPa
grid, computed for every experiment regardless of which family was directly
inserted. `geopotential_upper_air`/`humidity_upper_air` are new categories
added at the same time, made possible by that same expansion — both are
cross-variable response fields (never directly perturbed by either
experiment), included for both `temp` and `uv` experiment types.)

This mapping (representative-level choice, surface-variable naming) matches
the convention independently established in each model's own
`compute_global_rmse_corr.py` (see each script's `var_defs`/`var_pairs`/
`UPPER_VARS` list) — reused here, not re-derived, so the cross-model
comparison is directly consistent with each model's individual RMSE results
already on disk.

---

## 6. Output files

Written to `PERTURBATION_EXPERIMENTS/CrossModel/`:

- `cross_model_effectiveness_<experiment_type>.csv` (`experiment_type` =
  `temp` or `uv`) — one row per (model, variable_category, step):
  `control_rmse`, `experiment_rmse`, `effectiveness`.
- `cross_model_damping_<experiment_type>.csv` — one row per (model,
  variable_category): `effectiveness_step121`, `effectiveness_step240`,
  `damping_ratio`.
- `cross_model_blurring_<experiment_type>_<band>.csv` — one row per (model,
  step): `effective_resolution_wavenumber`.
- `cross_model_effectiveness_<experiment_type>.png` — effectiveness vs. lead
  time, one line per model, faceted by variable category.
- `cross_model_blurring_<experiment_type>_<band>.png` — effective-resolution
  wavenumber vs. lead time, one line per model.

---

## 7. Dependencies (this analysis runs last, deliberately)

This script reads, rather than recomputes, three things that must already
exist on disk:
1. Each model's `global_rmse_<experiment>.csv` (from each model's own
   `compute_global_rmse_corr.py`, already part of this project's core
   analysis pipeline, unrelated to this session's 3 new additions).
2. Each model's `spectral_blur_summary_<experiment>.csv`
   (`SPECTRAL_BLURRING_ANALYSIS.md`, new this session).

If any model's inputs are missing for a given experiment type, that model
is simply omitted from the corresponding comparison (with a printed
warning, not a hard failure) — this analysis is written to run on
whichever subset of the 3 models currently has complete data, so it does
not need to wait for the slowest model.

---

## 8. Known limitations

- **Not a true multivariate/energy-norm comparison.** Each metric here is
  computed per single variable/level, not combined into one aggregate
  cross-variable skill number — a `total energy norm`-style combination
  (as used in some predictability/DA literature) would be a natural,
  not-yet-implemented extension if reviewers want one composite number per
  model.
- **Different native grids and truth-matching pipelines per model** (FCN
  721x1440 vs ERA5's own grid interpolation; Aurora's cropped 720x1440;
  Prithvi's 360x576 vs MERRA-2) mean the underlying RMSE values being
  normalized into "effectiveness" are not perfectly like-for-like in how
  they were computed — the unitless ratio in §2 substantially mitigates
  this but does not eliminate every difference in verification methodology
  between the three models' pre-existing RMSE pipelines (which this
  analysis reuses, not rebuilds).
- **IC/truth-window differences** — FCN/Aurora use 2023 ERA5-based
  experiments; Prithvi uses 2022 MERRA-2-based experiments (different
  years, different truth products, per each model's own `IMPLEMENTATION.md`
  §5/§6). The cross-model comparison is therefore about relative *model
  behavior* under matched perturbation design, not an apples-to-apples
  comparison of the same real weather event across models — this should be
  stated plainly wherever this comparison is used in the paper.
