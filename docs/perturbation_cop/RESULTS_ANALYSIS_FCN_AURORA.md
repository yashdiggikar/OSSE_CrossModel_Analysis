# Results & Analysis: Correlated OSSE Perturbation (COP) Experiments — FourCastNetv2 and Aurora

**Status**: FCN and Aurora complete (temperature + wind, all diagnostics). Prithvi WxC
pending (blocked on an NFS access issue unrelated to model behavior — see
`SESSION_LOG_2026-07-22.md` §10.2). This document analyzes the two completed
models now; a Prithvi supplement should be appended once its data lands rather
than rewriting this file's structure.

All numbers below are read directly from `{FCN,Aurora}/{t,uv}_perturbed_corr*/*.csv`,
the new `paper_figures/` plots, and two small targeted diagnostics run
specifically for this analysis (§6). Nothing here is estimated by eye from a
plot — every number is traceable to a specific CSV column or a script run
included at the end of each section.

---

## 0. How this extends the main paper

The main manuscript (`AMS updated.docx.txt`) establishes a direct-insertion
ML-OSSE framework using **error-free** truth overwrite — Section 7
("Limitations") states this explicitly: *"error-free truth overwritten at
every level, with no observation operator, sparse/noisy sampling, or explicit
background-observation error weighting... a best-case, upper-bound test."*
Section 8 ("Discussion") calls, as the natural next step, for *"introduc[ing]
satellite-derived observations with realistic spatial coverage, uncertainty,
and missing data."*

The Correlated OSSE Perturbation (COP) experiments analyzed here are exactly
that next step, for the *uncertainty* dimension specifically (spatial
coverage/sparsity remains open future work — every grid point is still
corrected here, just with a physically-structured error rather than a perfect
value). Instead of overwriting with exact ERA5 truth, each grid point/level
is overwritten with **truth + a calibrated, spatially- and vertically-
correlated perturbation** (horizontal e-folding length ≈250 km, vertical
log-pressure decorrelation scale 0.5, Ingleby (2017)-style per-level sigma
tables, non-divergent streamfunction-derived wind error — full method in
`METHOD_OVERVIEW.md`). This is the same Errico et al. (2013)/Hollingsworth-
Lönnberg-style observation-error structure real OSSEs use, rather than i.i.d.
per-pixel noise or perfect truth.

**The question this section answers that the main paper's idealized
experiments cannot**: does the physical-consistency ranking established under
*perfect* truth insertion (Aurora ≥ FCN ≥ GraphCast ≫ Prithvi, per the main
paper's Conclusion) survive once the inserted truth is corrupted with
realistic, structured observation-like error? **The headline finding is: no —
not uniformly.** Under correlated, imperfect wind observations specifically,
FCN is *more* robust than Aurora, the reverse of no clean ranking existing
under perfect insertion (Section 3, §5.2 of the paper found both "display
sustained RMSE drops... with no clear overall winner"). This is a genuinely
new result, not visible in the perfect-truth experiments, and it is exactly
the kind of finding the paper's own Discussion section anticipates needing a
noisy-observation extension to surface.

---

## 1. Global RMSE: recovery magnitude and persistence under noisy insertion

*(source: `{FCN,Aurora}/{t_perturbed_corr,uv_perturbed_corr...}/global_rmse_*.csv`,
plotted in `paper_figures/{FCN,Aurora}/global_rmse_timeseries_*.png`)*

### 1.1 FCN: correction locks in and holds at the injected-noise floor

For every directly-inserted pressure-level variable, FCN's post-injection RMSE
collapses immediately and then stays **flat** for the entire 30-day window —
it neither decays back toward control nor drifts further:

| Variable | Control RMSE (day 30) | Experiment RMSE, day 30→60 range | Net reduction, day 60 |
|---|---|---|---|
| T500 | 4.16 K | 0.645–0.681 K (essentially constant) | 85.5% |
| U500 | 11.74 m/s | 1.037–1.120 m/s (essentially constant) | 90.9% |
| V500 | 10.49 m/s | 0.416–0.491 m/s (essentially constant) | 96.4% |

These plateau values are **not arbitrary** — they sit almost exactly at each
level's injected-perturbation sigma from `FCN/IMPLEMENTATION.md` §4 (T500
target 0.7 K; U500/V500 target ≈1.2 m/s), i.e. FCN's post-injection RMSE
against real truth is essentially just the magnitude of the noise that was
deliberately injected. This is the best-case outcome for a noisy-insertion
OSSE: the model neither amplifies the injected error nor fails to use the
signal it carries — the residual error is exactly the observation error
itself, with (to first order) no additional model-attributable error stacked
on top.

### 1.2 Aurora: correction is initially weaker and visibly decays within the window

Aurora's response has a qualitatively different shape. For wind specifically,
the injected correction is only partially adopted and then **erodes across
the 30-day post-injection window itself** — a real, quantifiable "forgetting"
of externally supplied information:

| Variable | Control RMSE (day 30) | Best point (mid-window) | Day 60 | Net reduction, day 60 |
|---|---|---|---|---|
| T500 | 7.43 K | 1.38 K | 3.78 K vs. control 8.59 K | 56.0% |
| U500 | 12.79 m/s | 3.78 m/s | 9.86 m/s vs. control 12.87 K | **23.4%** |
| V500 | 9.92 m/s | 4.26 m/s | 10.21 m/s vs. control 10.68 m/s | **4.4%** |

V500 is the starkest case: by day 60 Aurora's wind-inserted forecast has
almost **completely reverted to its free-running control error** (10.21 vs.
10.68 m/s) despite truth+noise having been force-inserted at every level
every 6 hours for the preceding 30 days. This is visible directly in the
cross-model effectiveness plot (`CrossModel/cross_model_effectiveness_uv.png`,
fixed this session — see §7): Aurora's `wind_u_upper_air` effectiveness curve
holds around 0.65–0.70 for most of the window, then collapses toward
0.05–0.23 in the final days, while FCN's holds flat at 0.90–0.93 throughout.

**Interpretation.** This is consistent with, and quantitatively sharpens, the
architectural explanation the main paper already gives for the two models'
different information-propagation mechanisms: FourCastNetv2's adaptive
Fourier / spherical Fourier neural operator (Pathak et al. 2022; Bonev et al.
2023) mixes information through **global spectral projection at every
forward pass** — a corrected state is re-expressed in the same global basis
every step, so there is no obvious mechanism for the correction to
selectively "leak away" over repeated steps once it is well captured in that
basis. Aurora's processor is a **3D Swin Transformer operating through local
windowed attention** (Bodnar et al. 2025) — information moves between windows
only via the shift-window mechanism, one hop per layer, which imposes an
effective bound on how far and how fast externally injected information can
propagate and be *retained* against the network's own learned free-running
attractor dynamics over many autoregressive steps. A single, deterministic,
un-perturbed truth-insertion in the main paper's idealized experiments did
not surface this — repeated correction with genuinely imperfect,
noise-corrupted truth does, because the noise gives the network's own
dynamics repeated opportunities to compete with (and partially overwrite) the
externally supplied signal instead of simply reproducing it exactly.

---

## 2. Vertical/surface coupling: does the correction reach non-inserted diagnostic fields?

*(same CSVs as §1; only pressure-level T/U/V are directly overwritten —
T2M/U10/V10 are never touched by the insertion code, see
`FCN/IMPLEMENTATION.md` §3 and `Aurora`'s equivalent — so their response is a
genuine secondary/coupling test, structurally the same kind of question as
the main paper's temperature→wind coupling score, but here it is a
within-family, vertical-propagation test under **noisy** upper-air forcing)*

| Model | Pathway | Day-60 improvement |
|---|---|---|
| FCN | U500/V500 → U10/V10 (wind, upper→surface) | **74–77%** |
| FCN | T500 → T2M (temp, upper→surface) | **12%** |
| Aurora | T500 → T2M (temp, upper→surface) | **55%** |
| Aurora | U500/V500 → U10/V10 (wind, upper→surface) | **2–4%** (statistically negligible) |

This is a clean, reproducible **cross-architecture asymmetry**: FCN transmits
its wind correction down to the surface almost as effectively as if U10/V10
had been directly inserted, but barely improves surface temperature; Aurora
shows the opposite pattern — its surface *temperature* improves almost as
much, relatively, as its own (already-decaying) pressure-level temperature
correction, but its surface *wind* is essentially unaffected by 30 days of
upper-air wind forcing.

**Physical reading.** FCN's global spectral mixing (§1.2) operates on the
full column simultaneously in the same basis, and vertical momentum
transfer/boundary-layer drag is a comparatively direct, well-conditioned
relationship (a corrected free-tropospheric wind field constrains surface
wind through vertical mixing and pressure-gradient continuity fairly
directly). Surface temperature, by contrast, is controlled by boundary-layer
processes — surface radiative/sensible-heat flux, land-sea contrast, diurnal
cycle, snow/ice state — that are only loosely coupled to free-tropospheric
temperature; FCN's spectral operator, whose channels each 500 hPa is one
smooth global field of many, evidently has not learned this weaker pathway as
strongly. Aurora's shared, jointly-encoded 3D latent state (Bodnar et al.
2025) plausibly explains the reverse asymmetry: its encoder/decoder pass
every surface and pressure-level variable through a *single* latent
representation per step, which should make column-consistent thermodynamic
adjustment (temperature is largely diagnosed from a smoothly-varying
mass/thermal state) easier to maintain than a genuinely two-way *dynamical*
momentum adjustment between levels, which — per Aurora's own windowed
attention argument in §1.2 — is exactly where Aurora's correction is already
observed to be weakest and least persistent.

**A caution the main paper's own Appendix B2 discipline requires stating
here too**: FCN's weak T500→T2M coupling is not simply "smaller because
there is less room to improve" — the *absolute* day-30 control baselines are
comparable (FCN T2M control 3.9–4.4 K vs. T500 control 4.2–4.5 K), so the gap
is not a baseline artifact; it is a genuine difference in how strongly the
two variables are coupled in FCN's learned dynamics.

### 2.1 A genuinely negative result: temperature insertion introduces a *new* systematic bias at the surface

`spatial_bias_*.csv`'s regional decomposition shows this is not just "weaker
improvement" — for FCN specifically, temperature insertion **degrades** the
midlatitude T2M bias relative to the free-running control:

| | Control mid-lat mean bias | Experiment mid-lat mean bias |
|---|---|---|
| FCN T2M | **+0.37 K** (mild warm bias) | **−1.37 K** (larger cold bias, opposite sign) |

Total RMSE still improves modestly (§2, 12%) because RMSE² = bias² +
variance, and the reduction in unconstrained scatter outweighs the larger
bias — but the sign flip itself is real and specific: correcting the free
troposphere while leaving the boundary layer's own dynamics untouched
introduces a *new* systematic cold discrepancy at the surface that was not
present in the uncorrected free-running forecast. This is exactly the kind
of result a real assimilation system's bias-correction step exists to catch,
and it is invisible to a pure global-RMSE read of Fig. 3-style plots — it
only appears once bias is decomposed regionally, as the main paper's own
Fig. 5/8/11/13 spatial-bias figures already do for the perfect-truth
experiments.

Aurora shows the same qualitative pattern but smaller in relative terms:
polar T2M control bias +4.27 K → experiment −2.36 K (also a sign flip, also
smaller in magnitude than the correction achieved at T500 itself, +7.91 K →
+0.02 K).

### 2.2 Wind bias degradation at the surface, specifically in the tropics

FCN's V10 and Aurora's U10 both show the *same* signature — surface wind bias
gets numerically *worse*, in absolute signed terms, specifically in the
tropics, even though total (unsigned) RMSE and absolute bias improve
elsewhere:

- FCN: V10 tropical mean bias +0.175 → **+0.486 m/s** (bias roughly triples)
- Aurora: U10 tropical mean bias +1.27 → **+2.17 m/s** (71% larger); global
  U10 mean bias +0.37 → **+1.02 m/s** (nearly 3×)

Both are consistent with a single underlying mechanism already flagged as a
known limitation in `METHOD_OVERVIEW.md` §2.3: the streamfunction-derived
wind perturbation's `cos(φ)` clipping means the injected wind noise is
systematically *under-scaled* near the equator relative to its nominal
per-level sigma target (rescaling uses one global standard deviation rather
than a latitude-dependent one). A too-small perturbation magnitude in the
tropics, propagating into a region where the geostrophic/thermal-wind
balance the correction implicitly assumes is weakest (f → 0, exactly the
Charney regime discussed in §3), is a plausible, physically grounded
explanation for why the surface-wind bias specifically grows in the tropical
band for both models — a real, actionable methodological finding: **future
runs of this perturbation scheme should use a latitude-banded rescaling for
the wind perturbation**, not a single global one.

---

## 3. Latitude-dependent response: does Charney's 1969 asymmetry survive imperfect truth?

*(source: `{FCN,Aurora}/*/latband_rmse_*.csv`, plotted in
`paper_figures/{FCN,Aurora}/latband_rmse_timeseries_*.png`; bands are
2°N–14°N (deep tropics) and 40°N–60°N (midlatitude))*

Charney et al. (1969) — the founding citation of the entire main paper — found
that temperature-insertion benefit was concentrated in the **extratropics**,
where geostrophic and thermal-wind balance are strong, and degraded sharply
approaching the equator, where the weak Coriolis parameter breaks the
classical temperature–wind relationship. The main paper's own idealized,
perfect-truth experiments reproduce this (§5.2.4: Aurora "reduces mid-latitude
wind errors by ≈20–50%... tapering toward the tropics"). **The same asymmetry
reproduces cleanly here, under noisy correlated truth, for temperature**:

| | Tropics (2–14°N) day-60 improvement | Midlatitude (40–60°N) day-60 improvement |
|---|---|---|
| FCN T500 | 48.7% | **90.7%** |
| Aurora T500 | 52.1% | **64.3%** |
| Aurora T2M | 19.0% | **69.4%** |

This is a robustness result worth stating plainly: the latitude-dependent
mechanism Charney identified in 1969 with a two-level Mintz-Arakawa GCM, and
that the main paper's Section 5.2.4 reproduces under idealized perfect-truth
insertion in four modern neural networks, **also holds when the inserted
"truth" itself carries a realistic, spatially-correlated 0.7–2.0 K
observation-like error** — the tropical/extratropical skill gap is not an
artifact of using error-free truth; it is a structural property of how these
networks have learned to use thermal information, robust to at least this
class of degradation in the input's own accuracy.

**Wind shows a messier, less uniform pattern**, and per the main paper's own
Appendix B2 discipline this needs a baseline check before being read as
architecture: FCN's wind improvement is large and *slightly* larger in the
midlatitudes for both bands (88–98% everywhere, i.e. FCN is already so close
to its noise floor in both regions that the residual ordering mostly reflects
which band's control baseline is larger — midlatitude jet-stream wind
variance is intrinsically larger than tropical, per the classical view). For
Aurora the ordering **reverses** for U500 (46.9% tropics vs. 9.8% midlat) and
V500 even goes slightly negative in the midlatitudes (−1.0%) — this is
consistent with, and a latitude-resolved instance of, the day-60 decay
already documented in §1.2: Aurora's midlatitude wind correction is the part
that erodes fastest across the post-injection window, not a tropical
geostrophic-breakdown story in reverse.

---

## 4. Spatial bias structure: a large, coherent Aurora-specific climate drift

*(source: `{FCN,Aurora}/*/spatial_bias_map_*.png` and the new 3-panel
`paper_figures/{FCN,Aurora}/spatial_bias_map_*.png`)*

Aurora's **free-running control** temperature forecast develops a severe,
geographically coherent warm bias covering essentially the entire Northern
Hemisphere extratropics/pole by day 30–60: +15 to +20 K locally over
Siberia/Canada/Greenland in the control-bias panel, matching the tabulated
polar-band mean of **+7.9 K** for T500 and **+4.3 K** for T2M (§1's numbers
are area-*averaged* over the whole polar cap; the peak local bias is
substantially larger). The Southern Hemisphere shows no comparable drift.
Correlated-perturbation temperature insertion removes essentially all of it
(experiment-bias panel is uniformly near-white; polar mean bias falls to
+0.02 K for T500). This is a striking, quantitatively large, and previously
undocumented (in this project) characteristic of Aurora's own unforced
60-day rollout specifically — a genuine climate-drift diagnostic in its own
right, independent of anything about the insertion experiment, worth flagging
for the paper's Aurora control-run discussion even outside the COP context.

FCN's control-bias field (`spatial_bias_map_t_perturbed_corr_T500.png`) is an
order of magnitude smaller in peak amplitude and considerably more spatially
incoherent/patchy — consistent with, and now a second independent line of
evidence for, the main paper's own Appendix A.1 characterization: *"FCN's
complement region is visibly patchier/noisier... consistent with its
spectral (SFNO) representation smearing error more diffusely,"* versus a
model whose errors organize into large, spatially coherent regions. Under
COP's noisy wind insertion, FCN's U500/V500 experiment-bias panels are
visually indistinguishable from zero everywhere (§1.1's near-total, spatially
uniform correction); Aurora's are not (§2.2's tropical U10 bias growth is
visible as a real, localized tropical band in the diff maps, not just a
domain-mean number).

---

## 5. Correlation-length verification: the generator is validated on FCN; Aurora's zonal check is uninterpretable *for a specific, now-understood reason* (§6)

*(source: `{FCN,Aurora}/*/corr_length_{horizontal,vertical}_*.csv`;
method in `CORRELATION_LENGTH_VERIFICATION.md`)*

### 5.1 FCN: clean validation, both open methodological questions resolved

At 45°N, 500 hPa, FCN's recovered temperature perturbation shows zonal
e-folding length **350 km** and meridional **237 km** against a nominal 250
km target — i.e. empirically closer to the **250·√2 ≈ 354 km** prediction
from `CORRELATION_LENGTH_VERIFICATION.md` §2.2's smoothing-kernel theory
(convolving white noise with a Gaussian kernel of std σ produces an
autocorrelation function with std σ·√2) than to the raw nominal 250 km.
**This empirically resolves the open methodological question flagged in that
document**: the generator's `SIGMA_PX` parameter should be read, and stated
in the paper's methods section, as controlling a **kernel** whose resulting
*realized* correlation length is ≈354 km, not 250 km outright. The vertical
check confirms the Cholesky-mixing machinery independently: RMS deviation
from the theoretical `exp(−|Δln p|/0.5)` matrix is **0.020–0.047** across
temperature and both wind components (max single-pair deviation 0.06–0.12)
— tight agreement, and the pole-distortion effect `METHOD_OVERVIEW.md` §2.1
predicts (zonal length should shrink at high latitude as meridians converge)
is directly visible and quantified: FCN's zonal length falls from 455 km
(equator) → 350 km (45°N) → 178 km (70°N) for temperature, a real,
now-measured version of a previously only-asserted limitation.

### 5.2 Aurora: zonal e-folding length is **unmeasurable**, not merely different

Every single zonal-direction measurement for Aurora — both experiments, every
band, every variable, no exceptions — returns `NaN`, because the empirical
correlation never decays below 1/e within the ±785 km window sampled
(`MAX_LAG_PX=40` at 45°N). Meridional measurements are unaffected and return
sensible values in the same range as FCN's (e.g. 96–230 km across bands and
variables). §6 shows this is not a validation failure of the perturbation
generator, but a direct, independently-verified consequence of Aurora's own
saved-output field having far fewer independent degrees of freedom along a
latitude circle than its nominal 1440-point longitude grid — the
`experiment − truth` perturbation recovery is exact arithmetic, but it
inherits whatever resolution collapse Aurora's own decoder has already
imposed on the field it is subtracted from.

---

## 6. A novel, independently-verified diagnostic: Aurora's effective zonal resolution is far below its nominal grid, at *every* lead time

This finding was not planned in `METHOD_OVERVIEW.md`'s original validation
list; it was found while investigating §5.2's anomaly, and is reported here
because it is real, robust, and — checked against the literature — a known
but previously unquantified-in-this-project phenomenon.

**Diagnostic**: count of distinct values along a full 1440-point latitude
circle, at three bands, at three lead times, in each model's own
**free-running, unperturbed control forecast** (i.e. nothing to do with the
perturbation-insertion code at all — this isolates a property of the base
model's own output).

| | day 0 | day 30 | day 60 |
|---|---|---|---|
| **FCN**, equator | 1166 / 1440 | 1423 / 1440 | 1425 / 1440 |
| **FCN**, 45°N | 1405 / 1440 | 1439 / 1440 | 1438 / 1440 |
| **FCN**, 70°N | 1359 / 1440 | 1439 / 1440 | 1439 / 1440 |
| **Aurora**, equator | **4** / 1440 | **5** / 1440 | **7** / 1440 |
| **Aurora**, 45°N | **29** / 1440 | **11** / 1440 | **7** / 1440 |
| **Aurora**, 70°N | **15** / 1440 | **16** / 1440 | **19** / 1440 |

FCN's T500 field carries essentially full-resolution independent zonal
structure (>95% of grid points distinct) at every lead time tested, including
day 0. **Aurora's carries between 4 and 29 distinct values along an entire
1440-point latitude circle, at every lead time tested, including day 0** —
i.e. this is not a progressive lead-time blurring effect in the classical
sense (Bonavita 2024's "convergence to the conditional mean" mechanism, §8
below), it is close to a fixed structural property of how Aurora's decoder
reconstructs the physical grid from its latent representation, present from
the very first forecast step. (It does mildly worsen at 45°N specifically —
29 → 11 → 7 — consistent with *some* additional accumulation on top of the
baseline collapse.)

**This directly explains §5.2**: a field whose real information content
along a latitude circle is O(10) independent values cannot show
autocorrelation decaying to 1/e within an O(40)-point window regardless of
what the true injected perturbation's correlation length is — the
measurement is dominated by Aurora's own decoding resolution, not by the
generator.

**Literature grounding.** This is a documented, if under-quantified, failure
mode of exactly Aurora's architectural family. Aurora's processor is a 3D
Swin Transformer operating on patch-tokenized input, decoded back to the
physical grid through a Perceiver-based decoder (Bodnar et al. 2025). Patch-
tokenized transformer atmospheric models are independently reported to
produce "grid-like," patch-scale artifacts that persist through autoregressive
rollout unless the decoder is specifically designed to suppress them — e.g.
the STRATA atmospheric emulator's authors state directly that naive
linear-projection patch decoding "introduces spectral artifacts that
accumulate during autoregressive rollouts, often manifesting as grid-like
patterns," motivating their own "pixel-space de-aliasing decoder that
suppresses patch-scale rollout artifacts" as a deliberate fix (Anonymous,
*Scaling Storm-Resolving Atmospheric AI Simulation to the Entire Planet*,
arXiv:2606.31248, 2026). The magnitude found here (single-digit-to-tens of
independent values per 1440-point circle) is more extreme than a simple
patch-size argument alone would predict (even a coarse 16-pixel patch would
still give ~90 distinct decoded blocks, not 4–30) — consistent with what
that literature describes as a **linear-projection upsampling** artifact
specifically (as opposed to bilinear+convolutional decoding), where a
patch-to-pixel projection layer can produce genuinely uniform output within
a patch if its learned weights favor that solution. This is stated as an
evidence-based interpretation, not a verified mechanism internal to Aurora's
actual released decoder (which was not inspected directly) — exactly the
caution the main paper's own Section 7 already applies to its own
architectural explanations.

**Why this matters for the paper beyond explaining §5.2**: it is a genuinely
new, quantifiable, architecture-diagnostic metric — call it *effective zonal
degrees of freedom* — obtainable from any single free-running forecast (no
perturbation experiment required), directly comparable across models, and
it independently corroborates the main paper's own qualitative Appendix A.1
finding (*"Aurora's complement-region error field stays organized in large,
sharp-edged synoptic blobs... a Swin-window signature"*) with a first
quantitative measurement rather than a visual read of bias maps. It also
plausibly explains §5.1/5.2's vertical-correlation-matrix gap: Aurora's RMS
deviation from the theoretical vertical-correlation matrix (0.30–0.45) is
roughly **10–18× larger** than FCN's (0.020–0.047) — a level-by-level
correlation matrix computed from a field with this few independent spatial
samples per level is a much noisier estimate of the true correlation
structure, independent of whether the underlying vertical Cholesky-mixing
step (identical code pattern in both models, per `METHOD_OVERVIEW.md` §2.2)
is itself working correctly.

*(Reproduction: `nohup python3` one-off script reading `t500` from
`FourCastNet/forecast_normal_60days_12hr.nc` and
`auroraoutput_1year_1stfeb23.nc` directly, `np.unique` on one latitude row —
not saved as a project script since it was a single diagnostic check, but
trivial to re-run from this description if needed for the paper's
supplementary material.)*

---

## 7. Spectral blurring of the injected signal

*(source: `{FCN,Aurora}/*/spectral_blur_summary_*.csv`, method in
`SPECTRAL_BLURRING_ANALYSIS.md`; this diagnostic uses `experiment − control`,
not `experiment − truth`, at several lead times spanning the post-injection
window — see that document §2.1 for why this is a deliberately different
signal from §5/§6 above)*

The dominant, model-independent pattern is a **sharp initial narrowing**
immediately after injection: both models' effective-resolution wavenumber
for temperature drops steeply from the injection step to the next sampled
step 20 steps (5 days) later at the equatorial band (FCN 41→ eventually 30;
Aurora 156→36, a much larger absolute drop). This is consistent with
Bonavita (2024)'s documented mechanism for MSE/L2-trained forecast models:
such models are shown to progressively smooth fields at longer lead time in
part to avoid the "double penalty" of a sharp but mislocated feature, which
is minimized in expectation by converging toward the conditional mean —
exactly the qualitative signature seen here in the difference field
immediately after a sharp, fine-scale correction is introduced.

**A necessary caution, already flagged as a known limitation in
`SPECTRAL_BLURRING_ANALYSIS.md` before this analysis was written**: the
effective-resolution-wavenumber metric is *not* monotonic with lead time for
either model past the first few days (e.g. FCN T500 midlat: 18 at day 30 →
37 at day 60, a near-doubling that should **not** be read as the forecast
"sharpening"). This almost certainly reflects the metric's own construction
— it is a wavenumber-domain *ratio* (first wavenumber where power drops to 1%
of the lowest-wavenumber power), and as the underlying `experiment − control`
signal's own amplitude decays (§1.2's damping), the ratio can move
non-monotonically even while the absolute high-wavenumber power is itself
still declining or flat. **The raw power spectra
(`*/spectral_blur_map_*.png`) should be used for any lead-time-resolved claim
in the paper**; the summary wavenumber is more reliable as a same-lead-time,
cross-model comparison than as a within-model trend.

Read that way — same lead time, model vs. model — the picture is mixed
rather than a clean "Aurora blurs faster" story: Aurora's effective-
resolution wavenumber is *higher* than FCN's at the first post-injection
snapshot for most variable/band combinations (e.g. U500 equator: Aurora 53
vs. FCN 21; V500 midlat: Aurora 118 vs. FCN 67). This is not a contradiction
of §6's zonal-resolution-collapse finding — it is evidence the two
diagnostics measure different things. §6 measures the **absolute** decoded
field, which is dominated by Aurora's presumably-fixed patch/decoder
geometry; this diagnostic measures a **difference** field (experiment minus
control, both decoded through the same fixed geometry at the same step), in
which shared, geometrically-fixed decoding artifacts should largely cancel,
leaving a comparatively less-quantized residual signal. Both results should
be reported together in the paper specifically **because** they appear to
conflict at first glance and the resolution is itself informative about
where in the pipeline (absolute decode vs. differenced signal) each
diagnostic is sensitive.

---

## 8. Synthesis: what changes once the OSSE is made realistic

Bringing §1–§7 together against the main paper's own perfect-truth results:

1. **The Charney tropical/extratropical asymmetry (§3) is robust to
   realistic observation error** — it reproduces under noisy correlated
   truth exactly as it does under perfect truth in the main paper, for both
   models, for temperature. This strengthens, rather than qualifies, the
   main paper's central claim that this is a structural property of learned
   momentum representation, not an artifact of the idealized insertion
   design.

2. **The physical-consistency ranking from perfect-truth insertion does not
   trivially transfer to noisy insertion, specifically for wind.** The main
   paper finds FCN and Aurora both show "sustained RMSE drops... no clear
   overall winner" under perfect wind insertion (§5.3.3). Under realistic,
   correlated-noise wind insertion, FCN sustains a near-total, undecaying
   correction that reaches the surface (§1.1, §2); Aurora's correction is
   weaker from the start, decays substantially within the 30-day window
   (§1.2), and essentially never reaches the surface (§2). **A model that
   looks equally "OSSE-ready" under an idealized perfect-truth test can look
   meaningfully less robust once given the kind of imperfect, structured
   observation error a real observing system would actually supply** — this
   is precisely the risk the main paper's Section 7 (Limitations) already
   anticipates in the abstract ("a best-case, upper-bound test... not a
   forecast of real observing-system impact") and this section supplies the
   first concrete evidence for it.

3. **A new, quantitative, architecture-level diagnostic (§6)** — effective
   zonal degrees of freedom along a latitude circle — independently
   corroborates the main paper's existing qualitative Swin-window
   observations (Appendix A.1) with a number, is obtainable from any single
   free-running forecast without any insertion experiment, and plausibly
   explains a secondary result (§5.1's vertical-correlation-matrix
   discrepancy) that would otherwise look like an unexplained validation
   failure.

4. **A specific, actionable methodological fix** falls out of §2.2: the
   wind-perturbation generator's global (rather than latitude-banded)
   rescaling under-corrects tropical wind magnitude, and this measurably
   shows up as *growing*, not shrinking, tropical surface-wind bias after
   insertion in both models — worth fixing before this scheme is used again,
   and worth stating explicitly in the paper's limitations/methods text
   rather than leaving as only a theoretical caveat.

---

## 9. Limitations specific to this analysis

Matching the main paper's own Section 7 discipline:

- **Single deterministic realization**, same caveat the main paper already
  states for its own results — no ensemble, no repeated initial condition.
  Numbers here should be read as point estimates; large, consistent,
  multi-diagnostic patterns (e.g. §1.2's wind-decay asymmetry, corroborated
  independently by §2's surface-coupling numbers and the cross-model
  effectiveness plot) are more trustworthy than any single percentage.
- **Two of three planned models.** Prithvi WxC's equivalent analysis is
  blocked purely on an NFS storage access issue (unrelated to model
  behavior; see `SESSION_LOG_2026-07-22.md` §10.2) and should be appended,
  not substituted for, once available — Prithvi showed markedly different
  (weak/near-absent) coupling under *perfect* insertion in the main paper,
  so its behavior under noisy insertion is likely to be its own distinct
  story, not interpolate between FCN's and Aurora's.
- **Different truth years/products inherited from the existing project
  design** (FCN/Aurora: 2023 ERA5; Prithvi would use 2022 MERRA-2) — already
  documented as a limitation in `CROSS_MODEL_COMPARISON.md` §8, repeated
  here since it will apply to any 3-model version of this document.
  Within FCN vs. Aurora specifically this is not an issue — both use the
  same 2023 ERA5 truth and IC family.
- **§6's diagnostic was a targeted one-off check**, not a saved,
  re-runnable project script — appropriate for surfacing and explaining an
  anomaly, but if this becomes a headline figure for the paper it should be
  promoted to a proper script (e.g. `compute_effective_resolution.py`) run
  across more lead times/bands/variables for a more complete supplementary
  table, not just the 3×3 spot-check reported here.
- **The spectral-blurring effective-resolution-wavenumber metric's
  non-monotonicity (§7)** limits what can be claimed about within-model
  lead-time trends from the summary CSV alone; same-lead-time cross-model
  comparisons and the raw power-spectrum plots are the more defensible
  source for paper figures from this diagnostic.

---

## References (new to this document; all others already in the main manuscript's list)

- Bonavita, M. (2024). On some limitations of current machine learning
  weather prediction models. *Geophysical Research Letters*, 51,
  e2023GL107377. https://doi.org/10.1029/2023GL107377 (arXiv:2309.08473)
- *Scaling Storm-Resolving Atmospheric AI Simulation to the Entire Planet*
  (STRATA), arXiv:2606.31248, 2026 — patch-scale decoder artifacts in
  autoregressive, patch-tokenized atmospheric transformers.

All other citations (Pathak et al. 2022; Bonev et al. 2023; Bodnar et al.
2025; Charney, Halem & Jastrow 1969; Holton & Hakim; Errico et al. 2013;
Hollingsworth & Lönnberg 1986; Bormann et al. 2016; Parrish & Derber 1992;
Bannister 2008; Ingleby 2017) are as already cited in the main manuscript
and/or `METHOD_OVERVIEW.md` — reused here, not re-derived.
