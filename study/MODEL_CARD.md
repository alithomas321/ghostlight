# Model card

## Model name

Ghostlight conditional astrometric point-lens model, version 1.0.

## Intended use

Use this package to reproduce a controlled teaching and methods experiment about omitted
blend-offset nuisance parameters in astrometric microlensing. It is designed for portfolio
review, code inspection, statistical calibration checks, and extension into richer
synthetic studies.

## Not intended for

Do not use Ghostlight to:

- infer the mass or identity of a real lens;
- reanalyze OB110462 or any other event;
- estimate HST, Roman, Gaia, OGLE, or MOA performance;
- predict compact-object yields or population rates;
- validate a point-spread-function extraction pipeline;
- claim that a narrow interval implies a detection.

## Inputs and outputs

The model receives synthetic epochs, known independent Gaussian errors, fixed photometric
geometry, and a known blend flux ratio. It returns a conditional WLS estimate and
known-sigma covariance for `theta_E` plus baseline and, in the offset-aware model,
blend-offset nuisance coefficients.

The calibration-sensitivity atlas compares this model with a deliberately misspecified
ablation that fixes the blend offset to zero.

## Assumptions

- single point source and dark point lens;
- unresolved lensed images;
- rectilinear lens-source trajectory;
- static unresolved blend relative to the source;
- exact, known blend flux ratio and photometric geometry;
- linear baseline proper motion and simplified annual source-parallax factors;
- independent homoscedastic Gaussian astrometric errors with known sigma;
- no selection or censoring.

## Validation

Tests cover analytic lensing limits, angular-unit relationships, the direct flux-weighted
blend identity, WLS recovery, SVD covariance calibration, Wilson intervals, exact coverage
oracles, paired draws, deterministic generation, hash verification, path safety, and claim
boundaries. Every fixed design also receives a residual noncentrality/expected-`Q` analysis
and exact lack-of-fit rejection power from SciPy. A separate scalar/QR reference path repeats
the physics, design, bias, standard-error, and residual-lambda calculations without calling
the production equation, stacking, design, or SVD-solver functions. Metamorphic tests cover
zero blend/offset, `A = 1`, angle periodicity/antisymmetry, stacking, sign, and units.

## Known limitations and risks

Real astrometry can contain temporal and cross-coordinate correlation, outliers,
heteroscedasticity, detector systematics, chromatic effects, crowding-dependent PSF bias,
and uncertain reference-frame transformations. A real blend can move. Photometric and
astrometric parameters may be jointly uncertain and multimodal. Binary lenses or sources
can imitate structure absent here. The simplified parallax factors are not tied to a sky
position or observatory ephemeris.

Misspecification beyond the one controlled omission is not surveyed, so the atlas cannot be
read as a comprehensive robustness guarantee.

The primary co-located-blend fit is not a plausible accepted model after residual checking:
its expected reduced `Q` is 18.035 and a standard 5% lack-of-fit test has effectively 100%
power. The coverage result therefore demonstrates that coefficient covariance is not a
model check; it does not demonstrate an undetectable failure. Observed residual p-values and
empirical rejection rates are unavailable because the frozen artifacts did not retain
per-trial residual vectors.

## Ethical and communication boundary

Every figure is labeled “SYNTHETIC KNOWN-TRUTH STUDY · NOT REAL-SURVEY PERFORMANCE.” The
package distinguishes the 2023 OB110462 PSF-extraction positional bias from the unresolved
flux-weighting mechanism studied here and notes the 2025 11-year analysis that conclusively
confirms the black hole. Removing those labels would materially misrepresent the work.
