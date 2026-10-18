# Methodology

## 1. Question and estimand

Ghostlight asks a single conditional question:

> When the true source-to-blend offset is nonzero, how does fixing that offset to zero alter
> estimation and 95% interval coverage for the angular Einstein radius, `theta_E`?

The estimand is `theta_E` in milliarcseconds. The photometric trajectory (`t0`, `tE`, `u0`,
and its tangent-plane angle) and blend flux ratio `g` are supplied exactly. This is a
known-truth calibration experiment, not an end-to-end event analysis.

## 2. Point-lens physics

Define the tangent-plane vector

```text
u(t) = [theta_source(t) - theta_lens(t)] / theta_E,
```

so it points from lens to source. In the rectilinear point-lens model, an unrotated
trajectory is `[tau, u0]`, where `tau = (t - t0) / tE`; Ghostlight rotates it by a fixed
angle in the tangent plane. Its norm is `u = |u|`.

The point-source, point-lens magnification is

```text
A(u) = (u^2 + 2) / [u sqrt(u^2 + 4)].
```

The unresolved centroid of the two dark-lens images is shifted from the unlensed source by

```text
delta(t) = theta_E u(t) / [u(t)^2 + 2].
```

Under this sign convention, `delta` points away from the lens. Its magnitude is largest at
`u = sqrt(2)` and equals `theta_E / (2 sqrt(2))`; that analytic limit is a unit test.

The Einstein-radius relation is

```text
theta_E^2 = kappa M pi_rel,
kappa = 8.144 mas / solar mass,
pi_rel = (1 / D_L - 1 / D_S) mas                 [distances in kpc].
```

## 3. Exact unresolved-blend centroid

Let `F_S` be source flux, `F_B` constant blend flux, `g = F_B / F_S`, and
`b = theta_B - theta_S` the source-to-blend offset. The raw lensed centroid is

```text
theta_obs = [F_S A(theta_S + delta) + F_B theta_B] / (F_S A + F_B)
          = theta_S + [A / (A + g)] delta + [g / (A + g)] b.
```

At the same time, the unlensed blended baseline is

```text
theta_base = theta_S + [g / (1 + g)] b.
```

Subtracting the baseline gives the exact transient used in the simulation:

```text
theta_obs - theta_base
  = [A / (A + g)] delta
    + [g(1 - A) / ((A + g)(1 + g))] b.
```

No small-blend or small-offset expansion is used. Direct raw flux weighting and this
reduced expression agree to machine precision in the tests. Limits are also checked:

- `g = 0` returns the dark-lens shift `delta`;
- `b = 0` leaves only the diluted gravitational term;
- at `A = 1` with a supplied nonzero `delta`, the offset term vanishes while the gravitational
  term remains `delta / (1 + g)`;
- in the coupled far-separation limit `u -> infinity`, both `A -> 1` and `delta -> 0`, so
  the entire baseline-relative transient vanishes. `A -> 1` alone is not sufficient.

The study assumes a static blend offset relative to the source. It does not model a
neighbor's own proper motion, color-dependent PSF, detector effects, or image-extraction
bias.

## 4. Astrometric baseline and linear design

The same-time unlensed blended baseline is modeled in two coordinates with:

- one position at `t0` and one proper-motion coefficient per coordinate;
- one source-parallax coefficient multiplying fixed annual two-coordinate factors;
- `theta_E` multiplying the known gravitational template;
- for the offset-aware model and `g > 0`, one blend-offset coefficient per coordinate.

The annual factors are a deterministic circular-orbit projection with a fixed phase and
0.72 axial ratio. They exercise the source-parallax nuisance but are not computed for a
specific sky coordinate or real observatory ephemeris.

Every cadence spans exactly six years, from `t0 - 3 * 365.25` to
`t0 + 3 * 365.25` days. The mapping `sign(q)|q|^1.45` for evenly spaced `q` in `[-1, 1]`
concentrates epochs toward the event while retaining those exact endpoints.

After stacking all x observations and then all y observations,

```text
y = X beta + epsilon,
epsilon ~ Normal(0, Sigma),
Sigma = sigma_ast^2 I.
```

The offset-aware and co-located-blend fits receive exactly the same simulated `y` in each
paired replicate. The co-located-blend design removes the two blend-offset columns but
still uses the correct `g`.

## 5. SVD weighted least squares

Ghostlight whitens the design and data, then solves with a singular-value decomposition:

```text
X_w = Sigma^(-1/2) X = U S V^T,
beta_hat = V S^(-1) U^T Sigma^(-1/2) y,
Cov(beta_hat) = V S^(-2) V^T.
```

No direct matrix inverse is used. Every cell must have full column rank. The output records:

- numerical rank;
- the whitened design's raw condition number;
- a scale-neutral condition number after normalizing each whitened column to unit norm;
- effective marginal information for `theta_E`, `1 / Var(theta_E_hat)`.

Because `sigma_ast` is known by construction, covariance is not rescaled by residuals.

## 6. Analytic omitted-variable oracle

Let `X_N` be the co-located-blend design and `r_miss` the true baseline-relative
blend-offset term.
The exact WLS expectation shift from omitting that term is

```text
b = (X_N^T W X_N)^(-1) X_N^T W r_miss,
```

implemented with the same SVD solution operator rather than a direct inverse. Its
`theta_E` component is the analytic bias oracle.

If `SE` is the known standard error and `Delta = b_theta / SE`, the exact coverage of a
symmetric `± z SE` interval is

```text
P(cover) = Phi(z - Delta) - Phi(-z - Delta),
z = 1.959963984540054.
```

For the correctly specified offset-aware model, `Delta` is zero up to floating-point error
and the oracle is 95%.

### 6.1 Residual model-adequacy oracle

Coefficient covariance answers a conditional precision question inside a supplied model; it
is not a model check. For design `X`, known covariance `Sigma`, and true mean `mu`, Ghostlight
also projects the true mean into each fitted model space. Let

```text
r_mu = mu - X E[beta_hat]
lambda = r_mu^T Sigma^(-1) r_mu
df = number of scalar observations - rank(X).
```

For Gaussian errors and a fixed linear design, the minimized residual statistic has the exact
sampling distribution

```text
Q ~ noncentral-chi-square(df, lambda),
E[Q] = df + lambda,
E[Q / df] = 1 + lambda / df.
```

SciPy's validated `chi2` and `ncx2` distributions provide the 5% critical value, analytic
tail areas, and exact fixed-design rejection power. The per-cell table distinguishes the
deterministic mean-discrepancy `Q = lambda` and `E[Q]` tail areas from observed-data p-values.
Observed `Q`, observed reduced `Q`, observed p-values, and empirical rejection rates are null:
the frozen artifacts save coefficient estimates but not each trial's residual vector. Filling
those fields would require rerunning the frozen stochastic draws.

At the primary cell, the co-located-blend model has `df = 58`, `lambda = 988.036250`,
`E[Q/df] = 18.035108`, deterministic omitted-mean residual RMS `0.982283 mas`, and effectively
100% power at alpha `0.05`. The offset-aware model has `df = 56`, numerical `lambda` below
`3e-26`, `E[Q/df] = 1`, and the nominal 5% rejection probability. Thus the interval-coverage
contrast occurs inside a mean model that an ordinary residual check would overwhelmingly
reject; it is not a hidden failure of a complete workflow.

The primary residual-support table reports the deterministic x/y residual at every epoch in
mas and sigma units, and the companion figure plots those time series.

### 6.2 Implementation-independent reference oracle

The production analytic path shares the production SVD solver, so Ghostlight now includes a
second reference implementation with different failure modes. It repeats the trajectory,
magnification, direct flux weighting, coordinate stacking, design construction, covariance,
bias, and residual-lambda calculations using scalar equations and a QR solve. It does not
import or call the production physics, stacking, design-matrix, or SVD-solver functions.
Metamorphic tests cover zero blend, zero offset, `A = 1`, 180° antisymmetry, 360° periodicity,
x-then-y stacking, sign, and mas scaling. Every model cell records both answers and their
numerical differences.

## 7. Locally prespecified design

The 81-cell study grid was fixed locally before the full run. One cell and its paired
coverage estimand define a **locally prespecified primary validation**; the rest of the
calibration-sensitivity atlas is exploratory. This same-session freeze is not external
registration or independent replication:

| Axis | Values |
|---|---|
| Epochs | 12, 32, 72 |
| Gaussian astrometric sigma per coordinate | 0.10, 0.25, 0.50 mas |
| Blend flux ratio `g` | 0, 0.3, 1.0 |
| Blend offset magnitude | 0, 5, 20 mas |

This is `3^4 = 81` grid cells. Ignoring noise scale, each cadence has seven distinct true
means: one at `g = 0`, three at `g = 0.3`, and three at `g = 1`. The grid therefore contains
`3 × 7 = 21` distinct true-mean vectors. Crossing those with three noise scales gives
`21 × 3 = 63` distinct mean-plus-sigma data-generating distributions. Within each of the
nine cadence/noise strata, the three `g = 0` offset rows share one distribution; the two
extra rows per stratum produce `9 × 2 = 18` deliberate negative-control repeats. Each grid
cell still receives its own 2,000 paired noise draws from seed `20260805`, for 162,000
unique synthetic datasets and 324,000 fitted model/dataset combinations.

In short: **21 distinct true-mean vectors**, **63 distinct mean-plus-sigma distributions**,
and **18 repeated zero-flux negative controls**.

The numerical choices were frozen before the oracle or seeded outputs were inspected:

| Choice | Rationale and interpretation boundary |
|---|---|
| Parallax axial ratio `0.72` | Neutral synthetic asymmetry used to exercise the parallax nuisance; it is not calculated from a sky position or ephemeris. |
| Cadence exponent `1.45` | Pre-run design choice that concentrates epochs near the event while preserving exact six-year endpoints; it was neither optimized nor taken from the literature. |
| Blend orientation `67°` | Non-cardinal synthetic orientation fixed before oracle inspection; it is neither representative nor optimized. |
| Primary cell: 32 epochs, 0.25 mas noise, `g = 0.3`, 20 mas offset | Middle cadence, middle noise, moderate blend ratio, and maximum offset form a deliberate stress setting—not a selected best-looking cell. |
| Injected truth values | Round historical-scale context for an OB110462-scale synthetic event; they are not current parameter estimates for that event. |

The synthetic truth is deliberately described as **OB110462-scale**, not as a reproduction.
Its round 6-solar-mass value is a historically published-scale benchmark, not the current
best mass; Sahu et al. (2025) report `7.15 ± 0.83` solar masses from an 11-year baseline.

| Quantity | Injected value |
|---|---:|
| Lens mass (known only to the simulator) | 6.0 solar masses |
| Source parallax | 0.11 mas |
| Relative parallax | 0.47 mas |
| Angular Einstein radius | `sqrt(8.144 * 6 * 0.47)` mas |
| Einstein time | 276 days |
| Impact parameter `u0` | 0.05 |

## 8. Error and uncertainty analysis

For estimates `theta_hat_i`, truth `theta`, and `N = 2000`:

```text
bias = mean(theta_hat_i - theta)
RMSE = sqrt(mean((theta_hat_i - theta)^2))
MCSE(bias) = sd(theta_hat_i - theta) / sqrt(N).
```

The bias table gives a normal Monte Carlo 95% interval around the simulated mean. RMSE gets
a fixed-seed 400-resample nonparametric percentile bootstrap interval. Those percentile
endpoints are supplemental, deliberately coarse bootstrap Monte Carlo summaries; they are
not headline evidence and do not enter any validation check. They quantify simulation
error, not observational uncertainty.

Bias and RMSE use finite estimates only and report that finite sample size. A nonfinite fit
remains in the full coverage denominator as uncovered, but can never enter a false-precision
numerator. Counts and rates are explicit. The locally prespecified validation checks require zero
nonfinite estimates; this accounting rule prevents silent denominator changes if an
exploratory configuration produces one.

Coverage is the fraction of known-sigma normal 95% intervals containing the injected
`theta_E`. Coverage and strict narrow-miss proportions receive Wilson score intervals.
Those intervals remain functions of the observed counts. For Monte Carlo standard errors,
the declared linear-Gaussian model supplies an exact event probability `p_oracle`, so each
cell saves

```text
MCSE(coverage) = sqrt[p_oracle (1 - p_oracle) / N].
```

This oracle-probability evaluation avoids the mathematically misleading zero plug-in MCSE
when a finite simulation happens to observe 0 or `N` events. For an eligible strict
narrow-miss diagnostic, the oracle event probability is `1 - p_oracle` and the same
binomial variance results. If a fixed interval is too wide for a threshold, its strict
count, rate, Wilson endpoints, oracle rate, and MCSE are structurally undefined and saved
as null/blank, not as zero. At true 95% coverage and 2,000 trials, the coverage MCSE is
about 0.49 percentage points.

For each finite fit, the standardized error is
`z_i = (theta_E_hat_i - theta_E_true) / SE`. Every model cell saves its empirical mean and
sample variance beside the analytic mean (the standardized omitted-variable bias) and unit
variance oracle. Under the declared normal model, the variance diagnostic uses approximate
Monte Carlo standard error `sqrt(2 / (finite_n - 1))`; every cell must agree with one within
four such MCSE. The standardized mean discrepancy is also exposed, while the equivalent
bias-oracle screen remains the binding mean-calibration check.

Before the run, the primary false-precision diagnostic was defined as:

```text
the 95% interval misses truth
AND
its half-width <= 10% of true theta_E
(equivalently, total width <= 20%).
```

The width threshold separates a genuinely uninformative miss from a seemingly decisive
wrong answer. Parallel exploratory sensitivities use 25% and 50% half-widths. These are
diagnostic definitions, not universal scientific conventions, and they are distinct from
the locally prespecified paired coverage-loss estimand.

Offset-orientation variants were prespecified as exploratory. Ghostlight therefore sweeps
0–360° in one-degree steps for the primary design using only the exact omitted-variable and
coverage oracles. The fixed 67° orientation is highlighted. No additional noise draw,
post-hoc Monte Carlo, or primary headline is created from this sensitivity curve.

The locally prespecified primary estimand is offset-aware minus co-located-blend coverage on paired
noise draws.
Its Monte Carlo standard error is the sample standard deviation of paired indicator
differences divided by `sqrt(N)`. A normal Monte Carlo 95% interval and both discordant-pair
counts are reported; no population-level p-value is assigned. A coverage loss of at least
five percentage points activates the locally prespecified undercoverage interpretation; a smaller point
estimate activates the negative-result pivot.

The local analysis specification was frozen in the study record before the seeded random
draw. It has no external immutable registration, so every public and machine-facing field
says **locally prespecified** rather than implying third-party review or timestamping.

Before any random generator is constructed, all 81 × 2 fixed model-cell designs must pass
rank and finite-value preflight. Any failure stops the locally prespecified validation run and requires a
transparent amendment to the locally frozen plan; rank is not misrepresented as a
replicate-varying failure. Validation checks also require Monte Carlo bias and coverage to agree
with their analytic oracles within four MCSE, offset-aware primary coverage to be
compatible with 95% within four MCSE, every standardized-error variance to agree with one
within four normal-model MCSE, and no nonfinite estimate.

These many correlated 4-MCSE comparisons are engineering/calibration screens, not a family
of hypothesis tests: no family-wise-error or false-discovery control is claimed. Their count,
tolerance, and scope are machine-readable. The separate residual adequacy result carries the
model-checking conclusion.

`prespecified_validation=True` accepts only the exact frozen `StudyConfig`, checked by complete
dataclass equality and a saved SHA-256 specification fingerprint. Any custom configuration
is explicitly exploratory. The verification protocol additionally recomputes the complete
machine-readable locally prespecified metrics and compares them exactly with the saved JSON.

### 8.1 Duplicate-control weighting

The former all-row pooled-binomial coverage summary over-weighted the 18 repeated zero-flux
controls and treated heterogeneous cells as one binomial sample. It has been removed. For the
descriptive offset-aware overview, repeated `g = 0` rows are first aggregated within each
shared cadence/noise DGP; coverage is then macro-averaged with one weight per distinct DGP.
The accompanying Monte Carlo standard error propagates the exact per-DGP binomial oracle.
No Wilson interval is attached to this heterogeneous macro-average, and it is not a
population estimand, multiplicity-adjusted test, or validation check.

## 9. Mass–distance non-identifiability

The simulator knows `pi_rel`, but the conditional astrometric fit does not estimate it.
Therefore the analysis never turns `theta_E` into one claimed mass. Instead, for each
assumed lens distance in front of the source, it plots

```text
M(D_L) = theta_E^2 / [kappa (1 / D_L - 1 / D_S)].
```

The displayed band maps `theta_E_true ± z SE` for the primary offset-aware design along that
ridge. The table samples from 0.25 kpc to `0.98 D_S`; the figure deliberately clips that to
a readable segment (`D_L <= 7.5` kpc and `M <= 45` solar masses). Conditional mass diverges
as `D_L -> D_S`, so the panel is not the whole ridge. Its band is a **truth-centered
primary-design 95% uncertainty band** for explanation—not a realized fitted confidence
interval, mass posterior, or uncertainty interval in distance.

## 10. Interpretation boundary

This experiment can show that one specified model omission causes bias and undercoverage
under its synthetic assumptions. It cannot estimate the frequency of that omission in a
survey, validate a PSF-reduction pipeline, identify a real compact object, or forecast a
discovery yield. Correlated astrometric errors, uncertain photometric geometry, uncertain
blend ratio, microlens parallax, binary lenses/sources, luminous lenses, and selection
effects are outside the model.
