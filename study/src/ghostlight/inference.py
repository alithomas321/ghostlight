"""Weighted linear inference for the conditional astrometric model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ghostlight.config import TruthConfig
from ghostlight.physics import blend_coefficients, lens_trajectory, parallax_factors

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DesignMatrix:
    values: FloatArray
    parameter_names: tuple[str, ...]
    times_days: FloatArray
    offset_aware_blend_model: bool


@dataclass(frozen=True)
class LinearSolver:
    """SVD solution operator for a fixed known-noise design."""

    projection: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray
    rank: int
    condition_number: float
    normalized_condition_number: float
    effective_theta_information_per_mas2: float
    parameter_names: tuple[str, ...]

    def estimate(self, observations: ArrayLike) -> FloatArray:
        data = np.asarray(observations, dtype=float)
        if data.shape[-1] != self.projection.shape[1]:
            raise ValueError("observation length does not match the design")
        return data @ self.projection.T


def stack_coordinates(vectors: ArrayLike) -> FloatArray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape (epochs, 2)")
    return np.concatenate((values[:, 0], values[:, 1]))


def unstack_coordinates(values: ArrayLike) -> FloatArray:
    stacked = np.asarray(values, dtype=float)
    if stacked.ndim != 1 or stacked.size % 2:
        raise ValueError("stacked coordinate vector must have even length")
    count = stacked.size // 2
    return np.column_stack((stacked[:count], stacked[count:]))


def make_observation_schedule(
    epoch_count: int,
    truth: TruthConfig,
    baseline_years: float = 6.0,
) -> FloatArray:
    """Fixed event-centered cadence with exact +/- half-baseline endpoints."""

    if epoch_count < 5:
        raise ValueError("at least five epochs are required")
    if baseline_years <= 0:
        raise ValueError("baseline_years must be positive")
    phase = np.linspace(-1.0, 1.0, epoch_count)
    concentrated = np.sign(phase) * np.abs(phase) ** 1.45
    half_baseline_days = 0.5 * baseline_years * 365.25
    return concentrated * half_baseline_days + truth.t_0_days


def build_design_matrix(
    times_days: ArrayLike,
    truth: TruthConfig,
    blend_flux_ratio: float,
    *,
    offset_aware_blend_model: bool,
) -> DesignMatrix:
    """Build the conditional astrometric design with known photometric geometry."""

    times = np.asarray(times_days, dtype=float)
    if times.ndim != 1:
        raise ValueError("times_days must be one-dimensional")
    count = times.size
    u_vector = lens_trajectory(times, truth)
    u_squared = np.sum(u_vector**2, axis=1)
    gravitational_weight, blend_weight = blend_coefficients(np.sqrt(u_squared), blend_flux_ratio)
    theta_template = gravitational_weight[:, None] * u_vector / (u_squared[:, None] + 2.0)
    years = (times - truth.t_0_days) / 365.25
    factors = parallax_factors(times, truth.parallax_phase_deg)
    zeros = np.zeros(count)
    ones = np.ones(count)

    columns = [
        stack_coordinates(theta_template),
        np.concatenate((ones, zeros)),
        np.concatenate((zeros, ones)),
        np.concatenate((years, zeros)),
        np.concatenate((zeros, years)),
        stack_coordinates(factors),
    ]
    names = [
        "theta_e_mas",
        "x0_mas",
        "y0_mas",
        "mu_x_mas_per_year",
        "mu_y_mas_per_year",
        "source_parallax_mas",
    ]

    # With g=0 the blend columns are identically zero and scientifically irrelevant.
    if offset_aware_blend_model and blend_flux_ratio > 0:
        columns.extend(
            [
                np.concatenate((blend_weight, zeros)),
                np.concatenate((zeros, blend_weight)),
            ]
        )
        names.extend(("blend_dx_mas", "blend_dy_mas"))

    return DesignMatrix(
        values=np.column_stack(columns),
        parameter_names=tuple(names),
        times_days=times,
        offset_aware_blend_model=offset_aware_blend_model,
    )


def prepare_solver(
    design: DesignMatrix,
    sigma_mas: float | ArrayLike,
) -> LinearSolver:
    """Prepare a known-sigma WLS solver using an SVD of the whitened design."""

    matrix = np.asarray(design.values, dtype=float)
    sigma = np.asarray(sigma_mas, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(matrix.shape[0], float(sigma))
    if sigma.shape != (matrix.shape[0],) or np.any(sigma <= 0):
        raise ValueError("sigma must be positive and match the observation vector")

    whitened = matrix / sigma[:, None]
    u, singular_values, vt = np.linalg.svd(whitened, full_matrices=False)
    tolerance = np.finfo(float).eps * max(whitened.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank != matrix.shape[1]:
        raise np.linalg.LinAlgError(
            f"rank-deficient design: rank {rank} for {matrix.shape[1]} parameters"
        )

    inverse_singular = 1.0 / singular_values
    whitened_projection = (vt.T * inverse_singular[None, :]) @ u.T
    projection = whitened_projection / sigma[None, :]
    covariance = (vt.T * inverse_singular[None, :] ** 2) @ vt
    standard_errors = np.sqrt(np.diag(covariance))

    column_norms = np.linalg.norm(whitened, axis=0)
    normalized = whitened / column_norms[None, :]
    normalized_singular = np.linalg.svd(normalized, compute_uv=False)

    return LinearSolver(
        projection=projection,
        covariance=covariance,
        standard_errors=standard_errors,
        rank=rank,
        condition_number=float(singular_values[0] / singular_values[-1]),
        normalized_condition_number=float(normalized_singular[0] / normalized_singular[-1]),
        effective_theta_information_per_mas2=float(1.0 / covariance[0, 0]),
        parameter_names=design.parameter_names,
    )


def solve_wls(
    design: DesignMatrix,
    observations: ArrayLike,
    sigma_mas: float | ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    solver = prepare_solver(design, sigma_mas)
    return solver.estimate(observations), solver.covariance
