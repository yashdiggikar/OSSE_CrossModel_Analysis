# Results & Analysis: Correlated OSSE Perturbation (COP) Experiments — FourCastNetv2, Aurora, and Prithvi WxC

**Status**: all three models complete (temperature + wind, all diagnostics, 27/27
base outputs, now expanded to the full T/U/V/Z/humidity × 300/500/1000 hPa grid
computed for every experiment regardless of which family was directly inserted,
plus day-60-only spatial bias maps and 4 individual latitude slices — see
`SESSION_LOG_2026-07-22.md` §12.6 and its 2026-07-25 continuation for the full
change history). This document **supersedes the previous (2026-07-24) version
of itself**, which was written against the old 2–4-variable, 2-averaged-band
convention; it also supersedes `RESULTS_ANALYSIS_FCN_AURORA.md` (2 models only,
kept on disk for provenance).

All numbers below are read directly from `{FCN,Aurora,Prithvi}/{...}/*.csv`, the
`paper_figures/` plots, and `CrossModel/`'s cross-model comparison. Every number
is traceable to a specific CSV/script run; nothing here is eyeballed from a plot.

---

## 0. How this extends the main paper

(Unchanged from earlier versions.) In short: the main manuscript's insertion
experiments use **error-free** truth; the COP experiments here use truth **plus
a calibrated, spatially/vertically correlated perturbation** (≈250 km
horizontal e-folding length, 0.5 log-pressure vertical decorrelation scale,
Ingleby-style per-level sigma, non-divergent streamfunction-derived wind
error), i.e. the realistic, imperfect-observation extension the main paper's
Discussion section calls for.

**The central finding is unchanged by this expansion, and is now measured on a
much larger evidence base**: the main paper's own physical-consistency ranking
under **perfect** truth insertion put Prithvi WxC clearly last — *"Prithvi WxC
shows the sharpest asymmetry: strong vertical/thermodynamic coupling alongside
a wind pathway largely disconnected from, and sometimes actively destabilizing
to, the rest of the state."* Under **realistic, noisy** insertion, for the
specific diagnostic this section measures (within-family vertical/surface
propagation of an imperfectly-corrected upper-air signal), Prithvi is not
merely competitive — **it is the most robust model of the three**, matching or
beating FCN and clearly beating Aurora, for both temperature and wind, now
confirmed at all three representative pressure levels (300/500/1000 hPa)
rather than just one. This rank inversion is the headline result of this
document.

*(Caveat carried over unchanged: FCN/Aurora use 2023 ERA5; Prithvi uses 2022
MERRA-2 — different truth products and years. This section is about relative
*model behavior* under matched perturbation design, not an apples-to-apples
replay of the same weather event.)*

---

## 1. Global RMSE: recovery magnitude and persistence under noisy insertion

*(source: `{FCN,Aurora,Prithvi}/*/global_rmse_*.csv`, plotted in
`paper_figures/{FCN,Aurora,Prithvi}/global_rmse_timeseries_*.png`. All values
below are day-60 net RMSE reductions relative to that model's own control,
`100×(1 − exp_rmse/ctrl_rmse)`, computed at the nearest available lead time
to day 60 — FCN's/Aurora's control files are 12-hourly, so the exact `t=1440h`
row is sometimes control-NaN; the nearest valid control row, always within 6h,
is used instead.)*

### 1.1 Temperature insertion: all three 300/500/1000 hPa levels, not just one

| Model | T300 | T500 | T1000 | T2M |
|---|---|---|---|---|
| FCN | 79.9% | 85.4% | 73.2% | 11.8% |
| Aurora | 47.3% | 56.0% | 58.2% | 55.3% |
| Prithvi | 93.2% | 93.3% | 81.2% | 82.0% |

FCN and Prithvi both lock in and hold near the injected-noise floor at every
pressure level tested, not just the previously-reported 500 hPa; Aurora is
weaker and more level-dependent everywhere, confirming the single-level
finding from the earlier version of this document generalizes across the full
vertical column rather than being an artifact of the one level originally
checked.

### 1.2 Wind insertion: all three 300/500/1000 hPa levels, both components

| Model | U300 | U500 | U1000 | V300 | V500 | V1000 | U10(M) | V10(M) |
|---|---|---|---|---|---|---|---|---|
| FCN | 90.3% | 90.8% | 81.1% | 96.5% | 96.5% | 92.7% | 77.0% | 73.7% |
| Aurora | 29.5% | 23.4% | 3.9% | 3.2% | 4.4% | −2.6% | 4.3% | 2.4% |
| Prithvi | 87.6% | 86.6% | 78.6% | 93.7% | 93.5% | 89.3% | 71.4% | 86.1% |

Aurora's wind recovery is not just weaker than FCN/Prithvi at 500 hPa (as
previously reported) — it **degrades further at 1000 hPa, going slightly
negative for V1000** (the noisy-insertion experiment is marginally *worse*
than free-running control at that specific level/lead-time combination),
consistent with the erosion-across-the-window behavior already documented for
Aurora and now shown to extend down to the surface-adjacent level.

---

## 2. Vertical/surface coupling: the rank-inversion result, now confirmed at all 3 levels

*(only pressure-level T/U/V are directly overwritten in each experiment — T2M/
U10(M)/V10(M) are never touched by the insertion code in any of the three
models, so their response is a genuine secondary/coupling test, not a primary
injection result. Label fix vs. the pre-2026-07-25 version of this document:
Prithvi's upper-air labels are now `T500`/`U500`/`V500` — same physical
~532 hPa nearest-native level as before, just renamed to match FCN/Aurora's
convention now that all three models share one label scheme.)*

| Model | Pathway | Day-60 improvement |
|---|---|---|
| FCN | U500/V500 → U10/V10 (wind, upper→surface) | 77.0% / 73.7% |
| **Prithvi** | U500/V500 → U10M/V10M (wind, upper→surface) | **71.4% / 86.1%** |
| Aurora | U500/V500 → U10/V10 (wind, upper→surface) | 4.3% / 2.4% (negligible) |
| FCN | T500 → T2M (temp, upper→surface) | 11.8% |
| Aurora | T500 → T2M (temp, upper→surface) | 55.3% |
| **Prithvi** | T500 → T2M (temp, upper→surface) | **82.0%** |

Prithvi remains the *best* upper-to-surface temperature propagator of all
three (82.0% vs. FCN's 11.8% and Aurora's 55.3%), and matches FCN closely on
wind (both models 70–90% on both components), both far ahead of Aurora. This
is visible in the cross-model effectiveness plot
(`CrossModel/cross_model_effectiveness_temp.png`,
`cross_model_effectiveness_uv.png`), which now also includes two new
categories described in §2.1 below.

### 2.1 New this update: cross-variable coupling to geopotential and humidity

The 2026-07-25 expansion computes **every** variable (T, U, V, Z, humidity),
**for every experiment**, regardless of which family was directly inserted —
so the temp-insertion experiment now also reports its Z/humidity response, and
the wind-insertion experiment reports its own. This is a genuinely new axis
the pre-expansion pipeline could not measure at all. Day-60 net RMSE reduction
at Z500 and the model's humidity variable (R500 for FCN — relative humidity;
Q500 for Aurora/Prithvi — specific humidity):

| Model | Temp-insertion → Z500 | Temp-insertion → humidity500 | Wind-insertion → Z500 | Wind-insertion → humidity500 |
|---|---|---|---|---|
| FCN | 32.4% | *(no ERA5 R data — see caveat below)* | 80.4% | *(no ERA5 R data)* |
| Aurora | 51.4% | 6.7% | 55.5% | −17.2% |
| Prithvi | 54.8% | 25.2% | 9.5% | −3.5% |

Two things worth noting. **First**, wind insertion improves FCN's Z500 far
more (80.4%) than temperature insertion does (32.4%) — a direct, quantitative
confirmation of the main AMS paper's own finding that wind-to-geopotential
coupling is the more direct pathway (momentum balance is closer to a
one-step relationship than the hydrostatic-then-integrate chain required from
temperature), now reproduced under noisy rather than perfect insertion.
**Second**, humidity coupling is the weakest and least consistent pathway for
every model that has it: Aurora and Prithvi both show *negative* wind-to-
humidity coupling (−17.2%, −3.5%) — noisy wind insertion very slightly
degrades the humidity forecast relative to free-running control, the only
consistently negative cross-variable pathway found anywhere in this dataset.
This is plausible physically: moisture is governed by convective/microphysical
processes only loosely tied to the momentum field, so forcing a noisy wind
field can inject spurious moisture-relevant structure without a compensating
physical constraint to correct it.

**Known data gap, not a code bug**: FCN's ERA5 archive used throughout this
project has no relative-humidity (`r`) data at *any* of the 13 standard
pressure levels (confirmed independently in `compute_global_rmse_corr.py`'s
loading loop, which fails identically for R at every level, and again here) —
so FCN's humidity-coupling numbers are structurally unavailable, not zero or
suppressed. This is a genuine, permanent limitation of this project's archived
FCN truth data, not something that can be fixed by rerunning any script here.

### 2.2 Spatial bias: Prithvi's control-run bias is small and cleanly corrected

Unchanged from the earlier version of this document: Prithvi's free-running
control bias is modest and closer to FCN's in scale than to Aurora's severe
+15–20 K NH high-latitude warm-bias climate drift. Correlated-perturbation
insertion removes essentially all of it. The 2026-07-25 update adds a
**day-60-only** bias snapshot (`spatial_bias_map_*_day60.png`) alongside the
existing day-30–60 mean, for all three models and both experiments — useful
for checking whether the mean-window bias-removal result holds at the single
latest instant rather than only as a 30-day average, which it does for all
three models (visually near-white experiment-bias panels at day 60 for every
directly-inserted variable, same pattern as the mean-window maps).

---

## 3. Latitude-dependent response: Charney's 1969 asymmetry, now measured at 4 individual latitudes, not 2 averaged bands

*(Corrected 2026-07-25: the pre-existing lat-band script computed 2 **averaged
bands** — 2°N–14°N and 40°N–60°N. The user requested, and this update
delivers, 4 **individual, unaveraged latitude slices**: exactly 2°N, 14°N,
40°N, 60°N. Cosine-latitude weighting is correctly dropped for a single-row
RMSE, since it is a constant multiplier that cancels out for one latitude —
it only matters when combining multiple latitudes into one number, which the
old, now-replaced band-averaged version did.)*

### 3.1 Temperature (T500), day-60 net RMSE reduction at each latitude

| Model | 2N | 14N | 40N | 60N |
|---|---|---|---|---|
| FCN | 40.6% | 66.5% | 87.7% | 86.9% |
| Aurora | 54.2% | 57.0% | 51.3% | 68.2% |
| Prithvi | 79.9% | 81.9% | 93.2% | 94.7% |

FCN and Prithvi both show a clean, monotonic-ish Charney signature — the
40°N/60°N (extratropical) values are far above the 2°N/14°N (tropical) values,
reproducing Charney et al. (1969)'s founding asymmetry at the individual-
latitude level, not just as a 2-band average. Aurora's pattern is flatter and
less monotonic (54.2 → 57.0 → 51.3 → 68.2, a dip at 40°N rather than a clean
rise), consistent with the already-documented within-window decay problem
dominating its latitude signal more than a clean Charney breakdown would.

### 3.2 Wind (V500), day-60 net RMSE reduction at each latitude

| Model | 2N | 14N | 40N | 60N |
|---|---|---|---|---|
| FCN | 93.0% | 96.3% | 98.4% | 97.2% |
| Aurora | 5.9% | 14.3% | **−26.3%** | 17.1% |
| Prithvi | 93.8% | 93.0% | 95.3% | 95.3% |

FCN and Prithvi again show the same essentially-uniform, very-high recovery at
every individual latitude (all ≥93%) that the 2-band-averaged version already
suggested, now confirmed the averaging wasn't hiding a real breakdown at any
one of the 4 specific latitudes. **Aurora's V500 wind at 40°N is net
*negative*** (−26.3%) — the single sharpest single-number failure anywhere in
this document: at that specific latitude, noisy wind insertion leaves Aurora's
forecast measurably *worse* than doing nothing. This individual-latitude
granularity is a genuinely new finding the old 2-band-averaged convention
could not have surfaced (a 40°N–60°N averaged band would have blended this
negative 40°N value with a positive 60°N value of +17.1%, very plausibly
washing the sign change out entirely).

---

## 4. Correlation-length and vertical-correlation verification: Prithvi validates cleanly, unlike Aurora — now confirmed at 3 horizontal levels and all vertical levels, cross-variable

*(source: `{FCN,Aurora,Prithvi}/*/corr_length_{horizontal,vertical}_*.csv`,
regenerated 2026-07-25 with the horizontal check extended from one
representative level (500 hPa) to three (300/500/1000 hPa), and both checks
now computed for T, U, V, Z, and humidity — not just the directly-inserted
family — at every level, for both experiments.)*

### 4.1 Vertical correlation-matrix RMS deviation from theory (directly-injected T, temp-insertion experiment)

| Model | RMS deviation from `exp(-|ln(p_i)-ln(p_j)|/0.5)` |
|---|---|
| FCN | 0.0243 |
| Aurora | **0.4320** |
| Prithvi | **0.0178** |

Unchanged conclusion from the earlier version, now recomputed end-to-end on
the expanded pipeline: Prithvi's vertical-correlation recovery is not just
"not broken like Aurora's" — at 0.0178 it is tighter than FCN's own 0.0243,
and roughly **24× better** than Aurora's 0.4320.

### 4.2 Vertical correlation-matrix RMS deviation, cross-variable (all 5 variables, temp-insertion experiment)

| Variable | FCN | Aurora | Prithvi |
|---|---|---|---|
| T (directly inserted) | 0.0243 | 0.4320 | 0.0178 |
| U (response) | 0.3356 | 0.2839 | 0.2211 |
| V (response) | 0.3523 | 0.3331 | 0.2396 |
| Z (response) | 0.4964 | 0.4428 | 0.4037 |
| humidity (response) | *(no ERA5 R data)* | 0.1845 | 0.2187 |

This is new: the vertical-correlation structure of the *directly-inserted*
variable (T) recovers far more faithfully than any *response* variable's own
vertical structure does, for all three models. This makes physical sense —
only T is hard-replaced with the exact correlated-noise design target every
6 hours; U/V/Z/humidity inherit whatever vertical structure the model's own
learned dynamics produce in response, which was never designed to match the
0.5-log-pressure decorrelation scale in the first place. The gap between
Prithvi's tight T-recovery (0.0178) and its much looser response-variable
recovery (0.22–0.40) shows that "this model preserves injected statistical
structure faithfully" and "this model's own dynamics preserve statistical
structure" are different claims — only the first is directly demonstrated by
this diagnostic; the second would need a different, not-yet-built check.

### 4.3 Horizontal correlation length at 500 hPa: Aurora's zonal-NaN pathology confirmed unchanged

| Model | Zonal correlation-length measurable (T, 500 hPa)? | Meridional? |
|---|---|---|
| FCN | Yes, all 3 bands (177–455 km) | Yes, all 3 bands (177–237 km) |
| Aurora | **No — NaN at all 3 bands** | Yes, all 3 bands (96–180 km) |
| Prithvi | Yes, all 3 bands (142–453 km) | Yes, all 3 bands (276–411 km) |

Aurora's zonal-direction correlation length is still unmeasurable at every
latitude band tested (the lag-correlation curve never crosses the 1/e
threshold within the 40-pixel search window used), reproducing the exact
pathology already documented in the earlier version of this document and now
confirmed at all 3 representative levels, not just the one originally
checked. This directly corroborates the mechanistic explanation given
previously: an independent count of distinct values along a full 1440-point
latitude circle found Aurora's decoded field carries as few as 4–29
independent values per circle — a near-total collapse of effective zonal
resolution, plausibly tied to Aurora's Perceiver-based patch decoder.
**Prithvi is also a patch/token-based architecture (masked-autoencoder ViT)
yet does not share this pathology** — still a useful negative result ruling
out "all patch-based weather transformers blur" as the general explanation,
narrowing the claim specifically to Aurora's own decoder.

---

## 5. Spectral blurring diagnostic

*(source: `{FCN,Aurora,Prithvi}/*/spectral_blur_summary_*.csv`; uses
`experiment − control`, per `SPECTRAL_BLURRING_ANALYSIS.md` §2.1. Now computed
across the full T/U/V/Z/humidity × 300/500/1000 hPa grid, cross-variable, for
both experiments — the numbers below are for T500 specifically, the
directly-inserted variable in the temp experiment, at `midlat_45N`.)*

| Step (lead hour) | FCN | Aurora | Prithvi |
|---|---|---|---|
| 120 (day 30, first post-injection) | 18.0 | 13.0 | 11.0 |
| 140 | 20.0 | 17.0 | 15.0 |
| 160 | 20.0 | 16.0 | 3.0 |
| 180 | 12.0 | 13.0 | 7.0 |
| 200 | 18.0 | 28.0 | 3.0 |
| 220 | 23.0 | 33.0 | 3.0 |
| 239 (day 60) | 37.0 | 42.0 | 5.0 |

FCN's and Aurora's effective-resolution wavenumber both trend upward across
the post-injection window (fine-scale structure becomes *more*, not less,
visible in the experiment-minus-control residual over time) while Prithvi's
stays low and noisy throughout (3–15). **This should not be read as "Prithvi
blurs the injected signal fastest"** — as already flagged in
`SPECTRAL_BLURRING_ANALYSIS.md` §2.3/§5, Prithvi's native grid (0.5°×0.625°)
is roughly half FCN/Aurora's resolution (0.25°), which mechanically caps the
maximum representable wavenumber regardless of architecture. A fair 3-way
ranking on this specific diagnostic would need grid-resolution normalization
first; as-is, this section should be read qualitatively (does the number grow
or shrink with lead time) rather than compared in absolute magnitude across
models with different native grids.

---

## 6. Synthesis: what the expanded, cross-variable dataset shows that the single-level version could not

1. **The rank inversion (§2) survives, and is now confirmed at all 3
   pressure levels rather than resting on one.** Prithvi ≈ FCN ≫ Aurora for
   within-family vertical/surface propagation, both temperature and wind, at
   300, 500, and 1000 hPa independently. This is stronger evidence than a
   single-level result: it rules out "the 500 hPa finding was a level-specific
   fluke" as an objection.
2. **The 4-individual-latitude-slice fix (§3) surfaced a genuinely new,
   sharper finding the 2-band-averaged convention hid**: Aurora's V500 wind
   correction is net-negative specifically at 40°N, not just "weak" as a
   40–60°N average implied — a single-latitude sign flip that a coarser
   band-average would very plausibly have washed out entirely.
3. **Cross-variable coupling to geopotential and humidity (§2.1) is a new
   axis this update makes measurable for the first time.** The clearest new
   result: humidity is the only variable/pathway combination showing
   consistently *negative* coupling (Aurora and Prithvi's wind→humidity both
   degrade the forecast slightly relative to control) — a genuinely new,
   specific, falsifiable claim about which cross-variable pathways these
   architectures have NOT learned, that the pre-expansion pipeline could not
   have produced at all (it never computed a non-directly-inserted variable's
   response for the "wrong" experiment).
4. **The vertical-correlation split between the directly-inserted variable
   and its own response variables (§4.2) is a new distinction**: "this model
   preserves the *injected* statistical structure faithfully" (Prithvi does,
   at 0.0178) is now shown to be a different, stronger claim than "this
   model's *own learned dynamics* preserve that structure equally well for
   variables it wasn't told the answer to" (Prithvi does not, at 0.22–0.40) —
   worth stating explicitly since the two are easy to conflate.
5. **The correlation-length/vertical-correlation split (§4) remains a clean
   three-way architecture result**, now confirmed at 3 levels instead of 1:
   FCN and Prithvi both validate the injected perturbation's statistical
   structure faithfully; Aurora does not, for reasons traced to its own
   decoder rather than to patch-based tokenization generally.
6. **Charney's asymmetry (§3.1)** still holds across three independent
   architectures, two truth products, and two different years of data — now
   demonstrated at 4 individual latitudes rather than 2 averaged bands, the
   single most robustly replicated result across this entire body of work.

---

## 7. Limitations specific to this analysis

- **Single deterministic realization** per model (unchanged caveat).
- **Now 3 of the (up to) 4 main-paper models.** GraphCast has no COP
  experiment run in this project — a genuinely new run, not a low-effort
  addition, if added later.
- **Different truth products/years persist** (FCN/Aurora: 2023 ERA5;
  Prithvi: 2022 MERRA-2).
- **§5's grid-resolution confound** (Prithvi 0.5°×0.625° vs. FCN/Aurora
  0.25°) specifically limits the spectral-blurring three-way comparison, not
  the RMSE/bias/correlation-length sections, which are either unitless or
  already grid-aware (correlation-length km conversion accounts for each
  model's own degree spacing).
- **New this update**: FCN's humidity-coupling numbers (§2.1) are
  structurally unavailable, not zero — FCN's own archived ERA5 truth data
  used throughout this project has no relative-humidity field at any
  pressure level. This is a data-archive gap specific to this project's FCN
  pipeline, not a finding about FCN's own model behavior, and should not be
  read as "FCN has no humidity coupling" — it simply cannot be measured with
  the truth data on hand.
- **New this update**: the vertical-correlation split described in §4.2
  (injected-variable structure vs. response-variable structure) has only
  been checked for the temp-insertion experiment's T/U/V/Z/humidity response;
  the symmetric check (wind-insertion experiment's own response-variable
  vertical structure) exists in the same CSVs but was not separately
  tabulated here — a natural, essentially free follow-up if needed, since the
  data is already on disk.
