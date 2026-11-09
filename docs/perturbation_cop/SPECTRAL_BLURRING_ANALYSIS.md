# Spectral Blurring Diagnostic — Cross-Model Method

Companion to `METHOD_OVERVIEW.md`, `CORRELATION_LENGTH_VERIFICATION.md`, and
the three `IMPLEMENTATION.md` files. This document describes a **novel**
analysis (not called for anywhere in `METHOD_OVERVIEW.md` §6's original
validation list) added specifically to strengthen the paper: it quantifies,
per model, how fast each architecture smooths away the fine-scale structure
of the inserted correlated perturbation as the forecast rolls forward.

---

## 1. Motivation

ML-based weather forecast models trained with pixelwise loss functions
(MSE/L1) are well documented in the literature to progressively blur /
over-smooth fine-scale structure with increasing lead time — this is a
known, actively-discussed weakness distinct from a model's aggregate RMSE
skill (a blurred forecast can still have a *low* RMSE against a smooth
verification field while having thrown away real physical variance). Since
this study already injects the *same*, carefully-designed correlated
perturbation into three architecturally different models (FourCastNetv2,
Aurora, Prithvi WxC), it is a natural, low-additional-cost extension to ask:
**do the three architectures blur the identical injected signal at
different rates?** This is a comparison no single-model OSSE study can make,
and is one of this project's most distinctive potential contributions (see
also `CROSS_MODEL_COMPARISON.md`, which uses this diagnostic's output
alongside the RMSE/bias results).

**Updated 2026-07-25**: this diagnostic was extended from a single
representative variable/level (T500 for the temp experiment; U500/V500 for
the wind experiment) to the full T/U/V/Z/humidity x 300/500/1000 hPa grid,
computed for both experiments regardless of which family was directly
inserted. This adds a genuinely new question the single-variable version
could not ask: does the *cross-variable response* field (e.g. the
temp-insertion experiment's own Z/humidity response) show the same blurring
trajectory as the directly-injected field, or a different one? See §5's
last limitation for why these two are not the same claim.

---

## 2. Method

### 2.1 What signal is analyzed, and why it's *not* the same reconstruction
     used in `CORRELATION_LENGTH_VERIFICATION.md`

`CORRELATION_LENGTH_VERIFICATION.md` recovers the **pure injected noise**
(`experiment_output − truth`, at the first post-injection step only) to
check the generator's own statistical properties before the model has had
any chance to touch it.

This diagnostic instead uses `experiment_output − control_output`
(experiment minus the *unperturbed control forecast*, the same reference
already used by every `compute_global_rmse_corr.py`/`compute_spatial_bias_
corr.py` in this project), computed at **several lead times spanning the
post-injection window**. This is a deliberately different signal for a
deliberately different purpose:

- At the *first* post-injection step, `experiment − control` contains both
  the injected noise *and* the "jump" from hard-replacing the control's own
  (by-then 120-steps-drifted) free-running state with the true state —
  i.e. it is not pure noise, and that is fine here, since the question this
  diagnostic asks is "what does the model do, going forward, to whatever
  the perturbation experiment did differently from control" — both
  components are legitimately part of that.
- At *later* post-injection steps, this difference field has been passed
  through the model's own autoregressive dynamics one or more times.
  Tracking how its **power spectrum** changes between the first
  post-injection step and later ones is exactly the blurring signature: if
  high-wavenumber (fine spatial scale) power decays faster than
  low-wavenumber power as lead time increases, the model is smoothing the
  signal; if the spectral shape is preserved, it isn't.

### 2.2 Lead times sampled

Step indices `120, 140, 160, 180, 200, 220, 239` (0-indexed) — i.e. the
first post-injection step, then every ~5 days (20 steps x 6h) through the
end of the 240-step/60-day forecast. Seven snapshots is enough to see a
trend without seven times the I/O cost of a full per-step sweep.

### 2.3 Power spectrum computation

For each snapshot and each of the same three latitude bands used in
`CORRELATION_LENGTH_VERIFICATION.md` (near-equator, mid-latitude,
high-latitude — chosen for consistency across the two diagnostics, and
because zonal wavenumber content is itself latitude-dependent on a lat/lon
grid):

1. Take a small window of rows centered on the band (±5 degrees latitude)
   to average over, reducing single-row sampling noise.
2. For each row in the window, compute the 1D real FFT along the longitude
   axis (periodic — the grid already wraps in longitude) and take the
   squared magnitude (power) at each zonal wavenumber.
3. Average the power spectrum across all rows in the latitude window.

This yields one power-spectral-density curve per (snapshot, band). Plotted
on log-log axes (standard for atmospheric kinetic-energy-spectrum-style
figures), a straight-line region indicates a power-law spectrum; a
downward-curving high-wavenumber tail relative to earlier lead times is the
blurring signature.

### 2.4 Summary metric: effective-resolution wavenumber

For direct, single-number cross-model/cross-lead-time comparison (rather
than only a qualitative log-log plot), an **effective-resolution
wavenumber** is computed per snapshot: the zonal wavenumber at which power
first drops to 1% of the power at the lowest resolved (non-zero) wavenumber.
A shrinking effective-resolution wavenumber with increasing lead time means
the model is losing fine-scale content over time — i.e., blurring. This
number is what gets plotted (per model, per band) in
`CROSS_MODEL_COMPARISON.md`'s cross-model blurring-rate comparison.

**Note on this metric vs. the literature**: this project's threshold (power
drops to 1% of the wavenumber-1 value) is a locally-chosen convention, not
one taken from a published source. A separate, independently-validated 2026
evaluation framework (*PhysMetrics.Weather*, arXiv:2606.10642) defines
"effective resolution" differently — the wavelength where the kinetic-energy
retention ratio `R(k) = Ē_pred(k)/Ē_ref(k)` (predicted vs. reference spectrum,
not this diagnostic's experiment-minus-control difference spectrum) drops
below 50% **for 5 consecutive wavenumbers**. The two are not directly
interchangeable — this project's metric measures how fast the
*insertion-attributable signal itself* loses fine structure, while
PhysMetrics.Weather's measures how far a full forecast's own spectrum has
drifted from truth — but re-implementing the latter, stricter, published
threshold on this project's existing power spectra (§2.3 already computes
everything needed) would make the numbers directly citable against published
GraphCast/Pangu-Weather/FuXi values. Not yet done — a candidate follow-up.

---

## 3. Per-model implementation specifics

**Updated 2026-07-25**: this diagnostic now covers the full T/U/V/Z/humidity
× 300/500/1000 hPa grid plus surface (T2M, U10/U10M, V10/V10M) — **18
variables total per model**, computed for **both** experiments regardless of
which family was directly inserted (cross-variable coupling included), not
just one representative level of the directly-inserted family as in the
original version of this diagnostic. Because this check reads
`experiment − control` directly from each model's own saved output (no ERA5/
MERRA-2 truth needed at all), extending the variable/level coverage costs
almost nothing — it is 18 array reads per snapshot instead of 1–2, not 18
additional file opens or truth loads.

| | Difference field source | Grid / latitude-window rows | Wavenumber units |
|---|---|---|---|
| **FCN** | `prediction[step, ch_idx[<var><level>], :, :]` (experiment) minus the same read from the control file `FourCastNet/forecast_normal_60days_12hr.nc`, both opened once and reused across all 7 snapshots x 18 variables (same open-once-reuse-handle discipline established this session after the FCN segfault investigation). Variables: `t/u/v/z/r` × `{300,500,1000}` + `t2m/u10m/v10m` | 721 lat x 1440 lon, ±5 deg window ≈ ±20 rows at 0.25 deg spacing | cycles per 1440-point ring -> physical km via each band's true zonal km/degree (same conversion as `CORRELATION_LENGTH_VERIFICATION.md` §2.1) |
| **Aurora** | Same pattern, `exp_ds.variables[<var><level>]` minus `CTRL_FILE`'s (`auroraoutput_1year_1stfeb23.nc`) same variable, both opened once. Variables: `t/u/v/z/q` × `{300,500,1000}` + `2t/10u/10v` | 720 lat x 1440 lon, ±5 deg window ≈ ±20 rows | Same as FCN (identical 0.25 deg grid) |
| **Prithvi** | `ds.variables[<VAR>][step, lvl_idx, :, :]` at the 3 nearest-native levels to 300/500/1000 hPa (Prithvi has no exact match at any of the 3) minus the same slice from `prithviOSSE/run2/output/control_forecast_Jan2022.nc`. Variables: `T/U/V/H(as Z)/QV(as Q)` × 3 nearest levels + `T2M/U10M/V10M` | 360 lat x 576 lon, ±5 deg window ≈ ±10 rows at 0.5 deg spacing | cycles per 576-point ring -> km via 0.625 deg zonal spacing at each band |

**FCN's relative-humidity (`r`) rows are structurally empty** — same known
ERA5-archive gap documented in `CORRELATION_LENGTH_VERIFICATION.md` §4, but
note it does **not** actually apply here: this diagnostic reads FCN's own
saved model-output channels (`r300`/`r500`/`r1000`), not ERA5 truth, and FCN's
model output *does* have those channels (73 total channels including `r50`
through `r1000`). Whether they are populated with usable values depends on
whether the original inference run ever produced meaningful humidity output
— not yet separately verified here; treat FCN humidity numbers from this
specific diagnostic as unconfirmed rather than assumed-good or
assumed-missing.

---

## 4. Output files

Per model, per experiment, written to
`PERTURBATION_EXPERIMENTS/<Model>/<experiment>/`:

- `spectral_blur_<experiment>.csv` — long-format: snapshot step, band,
  variable, wavenumber, power (raw FFT values, for anyone wanting to
  re-plot or re-derive the summary metric differently).
- `spectral_blur_summary_<experiment>.csv` — one row per (snapshot, band,
  variable): the effective-resolution wavenumber from §2.4.
- `spectral_blur_map_<experiment>_<band>.png` — log-log power spectrum,
  one curve per snapshot lead time (color-graded by lead time), one PNG per
  latitude band.

---

## 5. Known limitations

- **1D zonal spectrum only, not a full 2D spherical-harmonic decomposition.**
  A true spectral atmospheric-science analysis would typically use spherical
  harmonics (accounting for the sphere's actual geometry) rather than a
  per-latitude-band 1D FFT along longitude rings. The 1D-per-band approach
  is a standard, much cheaper approximation that is adequate for a
  *comparative* (model-vs-model, lead-time-vs-lead-time) diagnostic — the
  absolute wavenumber-to-physical-scale mapping is less rigorous near the
  poles (same distortion already documented in
  `CORRELATION_LENGTH_VERIFICATION.md` §2.1) than a spherical-harmonic
  treatment would be, which is acceptable for relative comparisons but
  should be stated as a simplification if absolute length-scale claims are
  made from these plots in the paper.
- **`experiment − control`, not `experiment − truth`.** As explained in
  §2.1, this is intentional, but it does mean the signal analyzed here
  conflates the injected noise with the truth-insertion "jump" at the first
  snapshot — the diagnostic answers "how does the model treat *whatever
  differs from its own unperturbed trajectory* over time," which is the
  right question for a blurring diagnostic, but is not the same signal
  `CORRELATION_LENGTH_VERIFICATION.md` validates.
- **3 representative levels (300/500/1000 hPa), not a full 3D vertical
  treatment.** Updated 2026-07-25 from the original single-level (500 hPa)
  version to match this project's other diagnostics' level convention, but
  this is still 3 discrete levels, not a continuous vertical spectrum — a
  natural future extension.
- **Added 2026-07-25 — cross-variable rows measure a different thing than
  the directly-inserted variable's own rows.** For the directly-inserted
  family (T in the temp experiment; U/V in the wind experiment), the
  `experiment − control` difference at the first snapshot is dominated by
  the actual injected signal, and its decay over time is a genuine blurring
  measurement. For the newly-added cross-variable rows (Z/humidity in both
  experiments; U/V/Z/humidity in the temp experiment; T/Z/humidity in the
  wind experiment), the same difference field instead reflects the model's
  own *learned response* to the injection — its "effective-resolution
  wavenumber" trajectory says something about how finely-resolved that
  learned response is, which is a legitimate and interesting question, but
  is answering a different question than "how fast does this model blur an
  injected signal," and the two should not be conflated when reading the
  per-variable numbers in `spectral_blur_summary_*.csv`.
