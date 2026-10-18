---
study: ghostlight-identifiability-2026-08
created_at: 2026-08-05
status: historical-local-specification-frozen-before-primary-run
current_interpretation: locally-prespecified-primary-validation-not-confirmatory
frozen_at: 2026-08-05T23:40:00Z
---

# Locally frozen analysis specification

This file retains its historical filename for study continuity. It was completed and
timestamped locally before the saved primary run, but it was **not** deposited in an
external immutable registry. “Prespecified” or “locally frozen” is therefore the
accurate public description; it is not represented as a formal preregistration.
Later deviations belong in an amendment section and must not silently rewrite the
original plan.

The original 2026-08-05 record used “confirmatory” below. The 2026-08-08 red-team
adjudication supersedes that design-status label: because the cell was chosen and
frozen locally within the same AI-assisted build session, the defensible phrase is
**locally prespecified primary validation**. The hypothesis, cell, estimand, seed,
decision rule, and saved stochastic outcomes are unchanged.

## Hypotheses

- H1: in the primary nonzero-offset blend cell, the offset-aware model has
  higher 95% interval coverage than the co-located model that omits blend offset.
- Null hypothesis: the coverage difference is zero or negative.
- Directional or non-directional: directional, specified before the run.

## Primary outcome and estimand

The primary estimand is `coverage_offset_aware - coverage_co_located` for the untruncated
conditional 95% normal interval for `theta_E` (mas), evaluated across 2,000 paired
realizations in the cell with 32 astrometric epochs, 0.25 mas known Gaussian noise
per coordinate, blend/source flux ratio 0.3, and a 20 mas static blend offset.
Intervals are conditional on the supplied point-lens geometry, blend ratio, and
known noise. They are not full mass posteriors.

## Sample-size or precision logic

At 2,000 realizations, the Monte Carlo standard error of 95% coverage is about
`sqrt(.95*.05/2000) = 0.0049` (0.49 percentage points); at 50% it is 1.12 points.
Wilson intervals accompany all reported proportions. The count was fixed from this
precision target and computational feasibility, not selected after viewing results.

## Exclusions and missing data

There are no missing data or imputation. Every planned realization is counted.
Rank-deficient or non-finite fits are failures and remain in the denominator. No
outliers are removed. Input validation occurs before random draws are generated.

## Models and tests

- Primary model/test: paired comparison of two stacked, whitened linear GLS fits;
  descriptive Wilson intervals are reported for each coverage rate and the signed
  coverage difference is the primary effect size.
- Assumption checks: rank, condition number, zero-noise recovery, standardized-error
  mean/variance, and analytic omitted-variable bias/coverage agreement within 4 MCSE.
- Confidence/credible interval: untruncated known-sigma normal 95% interval using
  `z = 1.959963984540054`; Wilson 95% intervals for coverage and false precision.
- Multiple-comparison correction: none for the single primary cell. All other grid
  cells and false-precision thresholds are explicitly exploratory.
- Effect-size reporting: coverage difference, bias in mas and percent of truth,
  RMSE in mas, and interval half-width relative to the true `theta_E`.

## Baselines, controls, and seeds

- Baselines: co-located/no-offset blend model versus the offset-aware model.
- Controls: `g=0` and `offset=0`; analytic point-lens and omitted-variable oracles.
- Random seeds: root `20260805`, with stable independent child streams per cell.
- Number of repeated runs/folds: exactly 2,000 paired realizations in each of 81 cells.

## Stopping and decision rules

No early stopping is permitted. PROCEED when unit tests pass, the offset-aware model's
primary-cell coverage is compatible with nominal calibration, and Monte Carlo
results agree with analytic oracles within 4 MCSE. REFINE implementation—not the
hypothesis—if an oracle or deterministic-reproduction check fails. PIVOT the public
thesis if the co-located primary-cell coverage loss is smaller than 5 percentage points;
report that negative result rather than searching for a more dramatic cell.

## Locally prespecified primary versus exploratory outputs

The primary-scenario row in `scenario_metrics.csv`, its analytic-oracle comparison,
and the offset-aware-versus-co-located coverage figure are locally prespecified primary
validation evidence. They are not externally registered confirmatory evidence. The full 81-cell grid,
mass-distance ridge, event illustration, offset orientation variants, and false-
precision thresholds at 10%, 25%, and 50% are exploratory or explanatory.

## Amendments

Append dated changes here with rationale and the stage at which each change was made.

### 2026-08-05 — fixed-design failure accounting and threshold notation

**Stage:** before any saved primary random draw or grid result was generated.

The rank of each Ghostlight design matrix depends only on its prespecified scenario,
not on a noise realization. All 81 offset-aware and co-located designs will therefore be preflighted
before the root random seed is consumed. If any fixed design is rank-deficient or
non-finite, the primary run will not begin; this is a failed implementation/configuration
gate rather than 2,000 independently rank-deficient fits. Successful scenario outputs will
still report rank-failure and non-finite-fit counts and rates explicitly. Any realization-level
non-finite result, should one occur after a successful preflight, remains in the coverage and
false-precision denominator as originally specified.

The exploratory false-precision thresholds `10%`, `25%`, and `50%` refer to **interval
half-width divided by true theta_E**. The prespecified primary diagnostic is the `10%`
half-width threshold, equivalently a total interval width no greater than `20%` of true
`theta_E`. This records the notation already used in the standalone decision log; it does
not change the threshold after observing a result.

### 2026-08-05 — post-run terminology clarification

**Stage:** after the single frozen run; wording only, with no change to inputs,
estimands, decision rules, random draws, or numerical results.

Public terminology was changed from “preregistered” to “locally prespecified” and
from the value-laden model labels “honest/naive” to the structural labels
“offset-aware/co-located.” The specification was frozen locally before execution,
but no third-party registry supplied an immutable timestamp. This amendment narrows
the provenance claim and makes the comparator definition explicit.

### 2026-08-05 — post-run reporting-contract hardening

**Stage:** after the single frozen run; reporting and validation only, with no new
random draw and no change to an empirical count, estimate, oracle, estimand, or decision
threshold.

An adversarial implementation audit found that an empirical plug-in binomial MCSE is
spuriously zero when a finite simulation happens to produce an event count of 0 or
2,000. Generated per-cell coverage and eligible strict-diagnostic MCSEs now use the
exact normal-theory event probability; Wilson endpoints remain count-based. Strict
diagnostics that fail their fixed half-width eligibility rule now expose count, rate,
interval, oracle, and MCSE as structurally undefined (`N/A`/null), never as zero.

The same audit added pre-RNG collision and physical-input validation, enforced the
fixed 95% interval contract, and corrected design accounting from an ambiguous count to
81 executed rows, 21 distinct true-mean vectors, 63 distinct mean-plus-sigma data-
generating distributions, and 18 repeated zero-flux controls. The 400-resample RMSE
bootstrap is labeled supplemental and coarse and remains outside all headlines and
gates. The migration was performed from saved artifacts with RNG construction disabled,
and protected numerical fields were asserted byte-for-byte/value-for-value unchanged.

### 2026-08-08 — post-red-team design-status and model-adequacy amendment

**Stage:** after the single frozen stochastic run; deterministic reporting and
documentation only. No new random draws were created, and a projection of all protected
scenario summaries plus paired primary outcomes matched before and after with SHA-256
`185f7c2512eeb13f2456bc8275afec19dbbc1ef8ff67a6dc2d4ba016364f5b9d`.

An adversarial scientific review determined that “confirmatory” overstates a primary
cell frozen locally in the same AI-assisted session. Current-facing materials therefore
use “locally prespecified primary validation” and explicitly state the lack of external
registration and independent Stage 1 review. Historical occurrences are retained only
as provenance for the original record and are superseded by this amendment.

The same review identified a missing whole-data model-adequacy check. Deterministic
fixed-design residual calculations were added from the declared means and designs. In
the primary co-located model, residual noncentrality is `988.036250` on 58 degrees of
freedom, expected reduced chi-square is `18.035108`, and the analytic 5% rejection power
is effectively 1. The offset-aware model has negligible noncentrality and nominal 5%
rejection probability under its declared Gaussian DGP. These are analytic properties of
the fixed synthetic designs, not observed-data p-values: residual vectors were not saved,
so empirical rejection counts remain unavailable rather than being reconstructed through
a stochastic rerun.

An independent scalar/QR reference path, duplicate-DGP macro summaries, explicit
multiplicity boundaries, a primary mean-residual table/figure, and a renamed
calibration-sensitivity grid were added. These improve internal falsifiability but do
not create external validation or change any observational boundary.
