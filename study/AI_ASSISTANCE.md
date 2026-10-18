# AI assistance disclosure

## Direction and implementation provenance

Alison set the portfolio goal, same-session constraint, scientific constraints, quality
bar, and intended voice. Under that brief, Codex selected Ghostlight and generated the
scientific implementation, tests, analysis, documentation, and draft public copy. Codex
also designed and ran the automated and analytic self-checks described below.

Under Alison's brief, Codex chose and froze the proposed locally prespecified grid, primary cell,
seed, replicate count, synthetic truth, interval definition, and claim boundaries before
the seeded run. That was a local analysis-specification freeze within the same
AI-assisted workflow, not an independent technical choice by Alison and not an external
registration.

## Chronology boundary

The relevant sequence is:

1. **2026-08-05 selection:** under Alison's brief, Codex screened candidate directions and
   selected Ghostlight's conditional calibration question.
2. **2026-08-05 pre-freeze specification:** Codex drafted and locally froze the grid,
   primary cell, estimand, seed, interval definition, five-point pivot, and
   primary/exploratory boundary before inspecting the exact primary gap oracle or seeded
   result.
3. **2026-08-05 post-freeze, pre-run checks:** fixed-design preflight passed and Codex
   inspected the exact analytic oracle, which predicted a 14.680-point primary gap. The
   oracle was known before Monte Carlo; the saved paired counts were not.
4. **2026-08-05 post-run reporting:** Codex ran the frozen stochastic experiment once and
   made documented terminology and reporting-contract corrections without changing the
   primary cell, estimand, or saved stochastic outcomes.
5. **2026-08-08 later red-team remediation:** Codex added deterministic residual adequacy,
   the scalar/QR reference, distinct-DGP summaries, and multiplicity boundaries from saved
   evidence and fixed designs, without creating new random draws.
6. **2026-08-08 AI-authenticity implementation:** Codex corrected provenance and story order,
   refreshed deterministic reporting, and synchronized the public mirror without new draws.

Residual adequacy is a later post-run diagnostic that changed the interpretation; it did not
motivate selection or become primary. The historical local freeze is preserved at
[`preregistration.md`](preregistration.md); its authoritative research copy remains at
`research/studies/ghostlight-identifiability-2026-08/preregistration.md`.

## Human review status

If this candidate is released under Alison's name, Alison will be the accountable human for
the final interpretation and presentation. Her explicit human science, copy, contribution,
and release sign-off is still pending and is required before public release. No independent
peer review has been performed; this disclosure does not manufacture a human or external
review that has not occurred.

## Verification performed

Automated and analytic checks passed for the stated synthetic model. The AI-generated
scientific code was treated as untrusted until checked against:

- a direct flux-weighted centroid calculation;
- analytic point-lens limits;
- an SVD covariance Monte Carlo calibration;
- a closed-form omitted-variable bias;
- an exact normal-coverage oracle;
- fixed-design residual noncentrality, expected reduced `Q`, chi-square tail, and power
  calculations backed by SciPy;
- a second scalar/QR reference implementation that does not import or call the production
  physics, stacking, design, or SVD-solver functions, plus metamorphic boundary/sign/unit tests;
- deterministic metric recomputation and SHA-256 artifact verification;
- explicit text tests preventing observational or survey-performance claims.

The literature manifest links each scientific source to its limited role. No citation is
invented at runtime and no observational data is represented as having been analyzed.
These checks were designed and executed within the same AI-assisted workflow and therefore
are self-checks, not independent peer review.

The 2026-08-08 red-team pass materially changed the AI-drafted interpretation: a routine
residual check would decisively reject the primary co-located-blend model. The public lesson
now states that covariance is not a model check and does not describe this as a hidden failure
of a complete analysis. No frozen stochastic draws were rerun to fill unavailable residual
statistics.

The subsequent 2026-08-08 AI-authenticity provenance/chronology remediation was also
Codex-executed. It separated the historical stages, changed future-accountability wording,
and synchronized the public mirror; it did not perform or document Alison's human science,
copy, contribution, or release review. The internal session log records that work without
turning the AI remediation into human review.

## Residual responsibility

Passing tests proves consistency with the stated synthetic model, not correctness for a
real instrument or event. Any future observational use requires independent domain review,
real-data provenance, a sky- and observatory-specific astrometric model, correlated-error
checks, and a new locally prespecified analysis plan with an auditable freeze point.
