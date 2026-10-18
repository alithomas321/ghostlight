"""Analytic lack-of-fit diagnostics for the fixed known-noise linear models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2, ncx2

from ghostlight.inference import DesignMatrix, LinearSolver

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ResidualModelAdequacy:
    """Fixed-design residual diagnostics under a Gaussian known-sigma model.

    ``mean_residual`` is the component of the true mean that remains after its
    best fit in the supplied model space. Its squared whitened norm is the
    noncentrality parameter of the minimized residual statistic.
    """

    mean_residual: FloatArray
    degrees_of_freedom: int
    noncentrality_lambda: float
    mean_lack_of_fit_reduced_q: float
    mean_lack_of_fit_null_tail_probability: float
    expected_q: float
    expected_reduced_q: float
    expected_q_null_tail_probability: float
    chi_square_critical_95: float
    rejection_power_at_5pct: float
    omitted_mean_residual_rms_mas: float
    omitted_mean_residual_max_abs_mas: float

    def metric_fields(self) -> dict[str, float | int | bool | None | str]:
        """Return stable, JSON-safe names for saved per-model-cell metrics."""

        return {
            "residual_degrees_of_freedom": self.degrees_of_freedom,
            "residual_noncentrality_lambda": self.noncentrality_lambda,
            "residual_mean_lack_of_fit_q": self.noncentrality_lambda,
            "residual_mean_lack_of_fit_reduced_q": self.mean_lack_of_fit_reduced_q,
            "residual_mean_lack_of_fit_null_tail_probability": (
                self.mean_lack_of_fit_null_tail_probability
            ),
            "residual_expected_q": self.expected_q,
            "residual_expected_reduced_q": self.expected_reduced_q,
            "residual_expected_q_null_tail_probability": (self.expected_q_null_tail_probability),
            "residual_chi_square_critical_95": self.chi_square_critical_95,
            "residual_rejection_power_at_5pct": self.rejection_power_at_5pct,
            "omitted_mean_residual_rms_mas": self.omitted_mean_residual_rms_mas,
            "omitted_mean_residual_max_abs_mas": self.omitted_mean_residual_max_abs_mas,
            "residual_observed_q": None,
            "residual_observed_reduced_q": None,
            "residual_observed_null_p_value": None,
            "residual_empirical_rejection_count": None,
            "residual_empirical_rejection_rate": None,
            "residual_empirical_rejection_wilson95_low": None,
            "residual_empirical_rejection_wilson95_high": None,
            "residual_empirical_rejection_available": False,
            "residual_empirical_unavailability_reason": (
                "The frozen artifacts preserve coefficient estimates, not per-trial residual "
                "vectors or residual Q; an empirical rejection rate would require rerunning "
                "the frozen stochastic draws."
            ),
            "residual_tail_probability_scope": (
                "Analytic chi-square tail areas at the deterministic mean-discrepancy Q and "
                "at E[Q], not observed-data p-values."
            ),
        }


def _sigma_vector(sigma_mas: float | ArrayLike, count: int) -> FloatArray:
    sigma = np.asarray(sigma_mas, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(count, float(sigma))
    if sigma.shape != (count,) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be finite, positive, and match the observation vector")
    return sigma


def residual_model_adequacy(
    design: DesignMatrix,
    solver: LinearSolver,
    true_mean: ArrayLike,
    sigma_mas: float | ArrayLike,
) -> ResidualModelAdequacy:
    """Evaluate analytic residual behavior without drawing or refitting noise.

    For minimized known-sigma residual statistic ``Q``, the sampling law is
    ``noncentral-chi-square(df, lambda)`` under the supplied true mean. Therefore
    ``E[Q] = df + lambda`` and the power calculation is exact for this fixed
    Gaussian linear design.
    """

    mean = np.asarray(true_mean, dtype=float)
    matrix = np.asarray(design.values, dtype=float)
    if mean.shape != (matrix.shape[0],) or np.any(~np.isfinite(mean)):
        raise ValueError("true_mean must be finite and match the design rows")
    if solver.rank != matrix.shape[1]:
        raise ValueError("residual diagnostics require a full-column-rank fitted design")
    sigma = _sigma_vector(sigma_mas, matrix.shape[0])

    expected_coefficients = solver.estimate(mean)
    mean_residual = mean - matrix @ expected_coefficients
    whitened_residual = mean_residual / sigma
    noncentrality = float(whitened_residual @ whitened_residual)
    # Roundoff may produce a tiny negative value only through a future refactor;
    # the norm itself is nonnegative by construction.
    noncentrality = max(0.0, noncentrality)
    degrees_of_freedom = matrix.shape[0] - solver.rank
    if degrees_of_freedom <= 0:
        raise ValueError("residual diagnostics require positive residual degrees of freedom")

    expected_q = degrees_of_freedom + noncentrality
    critical = float(chi2.ppf(0.95, degrees_of_freedom))
    return ResidualModelAdequacy(
        mean_residual=mean_residual,
        degrees_of_freedom=degrees_of_freedom,
        noncentrality_lambda=noncentrality,
        mean_lack_of_fit_reduced_q=noncentrality / degrees_of_freedom,
        mean_lack_of_fit_null_tail_probability=float(chi2.sf(noncentrality, degrees_of_freedom)),
        expected_q=expected_q,
        expected_reduced_q=expected_q / degrees_of_freedom,
        expected_q_null_tail_probability=float(chi2.sf(expected_q, degrees_of_freedom)),
        chi_square_critical_95=critical,
        rejection_power_at_5pct=float(ncx2.sf(critical, degrees_of_freedom, noncentrality)),
        omitted_mean_residual_rms_mas=float(np.sqrt(np.mean(mean_residual**2))),
        omitted_mean_residual_max_abs_mas=float(np.max(np.abs(mean_residual))),
    )
