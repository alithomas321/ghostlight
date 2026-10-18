# Results

> **SYNTHETIC KNOWN-TRUTH STUDY · NOT REAL-SURVEY PERFORMANCE**

## Evidence chronology

- **Before the saved run:** under Alison Thomas's portfolio brief, Codex selected the
  question and froze the grid and run contract locally. One primary cell, the paired
  coverage estimand, seed, and five-point interpretation rule were fixed before the workflow
  inspected the computed oracle. The exact oracle was then evaluated before Monte Carlo;
  its 14.680-point analytic gap remained available to the same AI-assisted
  workflow that chose the design.
- **Original stochastic evidence:** one seeded full-grid command generated all
  81 rows and 162,000 synthetic datasets. The primary
  coverage counts, paired outcomes, and coefficient estimates below come from that run.
- **Later deterministic audit:** on 2026-08-08, red-team review added fixed-design residual
  adequacy calculations from the declared means and designs. No new random draws were
  created, and protected primary and scenario results remained unchanged.
- **Current status:** the later audit changed the interpretation of the original result;
  Alison's human science, copy, contribution, and release review remains pending.

## The question and primary design

When a static unresolved blend is offset from a microlensed source, how much can a
co-located-blend model reduce 95% interval coverage for `theta_E` while leaving its formal
standard error nearly unchanged? Ghostlight compares that model with an offset-aware fit on
identical noise realizations.

The primary cell uses 32 epochs, 0.25 mas Gaussian noise per coordinate, blend ratio
`g = 0.3`, a 20 mas source-to-blend offset at 67°, and 2,000 paired realizations.
The injected Einstein radius is 4.7923 mas. This is a
**locally prespecified primary validation**, not external registration or independent
confirmation. The other
80 rows were generated in the same stochastic run but
are interpreted exploratorily.

## Original seeded coverage evidence

| Fit | Bias (mas) | RMSE (mas) | 95% coverage | Strict narrow-miss rate (10% half-width) |
|---|---:|---:|---:|---:|
| Offset-aware | 0.0003 | 0.3539 | 95.80% | N/A — interval too wide for threshold |
| Co-located-blend | -0.3981 | 0.5327 | 80.70% | N/A — interval too wide for threshold |

The formal standard error changed only from 0.3618 mas to 0.3611 mas, a 0.19% reduction. The co-located analytic
omitted-variable bias was -0.3983 mas;
the Monte Carlo bias was -0.3981 ± 0.0079 mas Monte Carlo SE. Its
exact normal-theory coverage oracle was 80.32%.

Offset-aware minus co-located-blend coverage is 15.10 percentage points (paired Monte
Carlo SE 0.93 points; Monte Carlo 95% interval [13.28, 16.92] points). The
paired table contains 346 offset-aware-only and
44 co-located-blend-only covering pairs. No population-level p-value is attached to this
fixed synthetic contrast.

The locally prespecified five-point interpretation rule selected
**the locally prespecified undercoverage result**. The primary strict 10%-half-width diagnostic was ineligible for both
models; `N/A` records a fixed interval that was too wide for that threshold, not a measured
zero.

## Later deterministic model-adequacy audit

The later audit asked whether the co-located mean model would pass an ordinary residual
check. Deterministic projection of the already-declared primary design gave residual
`df = 58`, noncentrality `lambda = 988.036`,
expected reduced `Q = 18.035`, omitted-mean residual RMS
0.982 mas, and 100.00% analytic power for the ordinary 5%
chi-square lack-of-fit check. The offset-aware model had expected reduced
`Q = 1.000`.

No new random draws were used to obtain these quantities. They are analytic fixed-design
properties, not observed residual statistics. The frozen artifacts retained coefficient
estimates but not per-trial residual vectors, so observed residual `Q`, observed-data
p-values, and empirical rejection rates remain unavailable.

This later result changed the interpretation: formal coefficient covariance is conditional
on the assumed mean and **covariance is not a model check**. The seeded coverage contrast is
real for the declared design, but it occurs inside a grossly rejected mean model rather than
a model that would survive a complete residual analysis.

## Mechanism and verification

Direct flux weighting adds a time-dependent blend-offset term to the baseline-relative
apparent centroid. The co-located model omits that term, allowing part of it to project into
the `theta_E` coefficient. Both fits receive identical noise, so the primary contrast retains
the pairing.

The Monte Carlo bias and coverage are checked against exact linear-Gaussian omitted-variable
oracles. A later scalar-equation/QR path repeats the flux weighting, geometry, stacking,
design, covariance, bias, and residual calculation without calling the production physics,
stacking, design, or SVD-solver functions. It reduces shared-code risk but remains an
AI-authored internal self-check in the same specification and environment, not external
validation.

## Exploratory sensitivity from the same run

The grid contains 21 distinct true-mean vectors and
63 distinct mean-plus-noise data-generating distributions.
Its 27 `g = 0` rows include 18 deliberate negative-control repeats.

After repeated controls are first collapsed within their shared DGPs, unweighted
offset-aware macro coverage across 63 DGPs is 95.07%
(oracle-MCSE normal 95% interval [94.95%, 95.18%]). This is a
descriptive fixed-grid average, not a population estimand or heterogeneous binomial sample.

The largest eligible co-located-blend strict narrow-miss rate is
100.00% in `n32_s0.10_g1.0_b20.0`.

It is an exploratory conditional stress cell, not the primary headline or an estimate of
real-world failure frequency. Later grid-wide adequacy quantities are reported beside these
coverage summaries; they are not retroactively part of the original primary outcome.

## Claim boundary and explanatory mass ridge

No real image, light curve, or astrometric catalog is analyzed. The synthetic truth is on
the historical scale of OB110462, but Ghostlight is not an OB110462 reproduction and its
round 6-solar-mass truth is not the current best mass. The bright-neighbor PSF-extraction
bias discussed in the observational literature is distinct from the unresolved
flux-weighted-centroid mechanism isolated here.

The conditional astrometric fit estimates `theta_E` while holding the photometric trajectory
and blend ratio fixed. Without an independently measured relative parallax,
`theta_E^2 = kappa M pi_rel` defines a mass–distance ridge rather than one fitted mass. The
ridge figure is an explanatory nonclaim: its gold point is known only to the simulator, and
its shading is a truth-centered design illustration rather than a realized confidence
interval or posterior.
