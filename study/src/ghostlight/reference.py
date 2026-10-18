"""Implementation-independent scalar/QR reference oracle for Ghostlight.

This module intentionally does not import or call the production physics,
design-matrix, stacking, or SVD solver functions. It repeats the defining
equations in a compact scalar path so tests can detect shared implementation
errors rather than merely exercising the same primitives twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ghostlight.config import TruthConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ReferenceCellOracle:
    theta_bias_mas: float
    theta_standard_error_mas: float
    residual_noncentrality_lambda: float
    residual_degrees_of_freedom: int


def reference_stack_coordinates(vectors: ArrayLike) -> FloatArray:
    """Reference x-then-y coordinate serialization."""

    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape (epochs, 2)")
    return np.asarray([*values[:, 0], *values[:, 1]], dtype=float)


def reference_blend_vector(offset_mas: float, angle_deg: float) -> FloatArray:
    """Reference source-to-blend vector in milliarcseconds."""

    angle = math.radians(angle_deg)
    return np.array([offset_mas * math.cos(angle), offset_mas * math.sin(angle)])


def reference_magnification(u: float) -> float:
    if not math.isfinite(u) or u <= 0:
        raise ValueError("u must be finite and strictly positive")
    return (u * u + 2.0) / (u * math.sqrt(u * u + 4.0))


def reference_centroid_from_amplification(
    amplification: float,
    dark_shift_mas: ArrayLike,
    blend_flux_ratio: float,
    blend_offset_mas: ArrayLike,
) -> FloatArray:
    """Direct observed-minus-baseline flux weighting for one epoch."""

    delta = np.asarray(dark_shift_mas, dtype=float)
    offset = np.asarray(blend_offset_mas, dtype=float)
    if delta.shape != (2,) or offset.shape != (2,):
        raise ValueError("dark shift and blend offset must be two-vectors")
    if amplification <= 0 or blend_flux_ratio < 0:
        raise ValueError("amplification must be positive and blend ratio nonnegative")
    observed_relative_source = (amplification * delta + blend_flux_ratio * offset) / (
        amplification + blend_flux_ratio
    )
    unlensed_baseline_relative_source = blend_flux_ratio * offset / (1.0 + blend_flux_ratio)
    return observed_relative_source - unlensed_baseline_relative_source


def _reference_geometry(times: FloatArray, truth: TruthConfig) -> tuple[FloatArray, FloatArray]:
    trajectory = []
    parallax = []
    trajectory_angle = math.radians(truth.trajectory_angle_deg)
    parallax_phase = math.radians(truth.parallax_phase_deg)
    cosine = math.cos(trajectory_angle)
    sine = math.sin(trajectory_angle)
    for time in times:
        tau = (float(time) - truth.t_0_days) / truth.t_e_days
        trajectory.append([tau * cosine - truth.u_0 * sine, tau * sine + truth.u_0 * cosine])
        phase = 2.0 * math.pi * float(time) / 365.25 + parallax_phase
        parallax.append([math.cos(phase), 0.72 * math.sin(phase)])
    return np.asarray(trajectory), np.asarray(parallax)


def _reference_problem(
    times_days: ArrayLike,
    truth: TruthConfig,
    blend_flux_ratio: float,
    blend_offset_mas: float,
    *,
    offset_aware_blend_model: bool,
) -> tuple[FloatArray, FloatArray]:
    times = np.asarray(times_days, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times_days must be a nonempty one-dimensional array")
    trajectory, parallax = _reference_geometry(times, truth)
    offset = reference_blend_vector(blend_offset_mas, truth.blend_angle_deg)
    years = (times - truth.t_0_days) / 365.25

    theta_template = []
    blend_weights = []
    true_mean = []
    for index, vector in enumerate(trajectory):
        u_squared = float(vector @ vector)
        u = math.sqrt(u_squared)
        amplification = reference_magnification(u)
        dark_unit_shift = vector / (u_squared + 2.0)
        dark_shift = truth.theta_e_mas * dark_unit_shift
        transient = reference_centroid_from_amplification(
            amplification,
            dark_shift,
            blend_flux_ratio,
            offset,
        )
        baseline = (
            np.asarray(truth.baseline_position_mas)
            + years[index] * np.asarray(truth.proper_motion_mas_per_year)
            + truth.source_parallax_mas * parallax[index]
        )
        true_mean.append(baseline + transient)
        theta_template.append(amplification / (amplification + blend_flux_ratio) * dark_unit_shift)
        blend_weights.append(
            blend_flux_ratio
            * (1.0 - amplification)
            / ((amplification + blend_flux_ratio) * (1.0 + blend_flux_ratio))
        )

    count = times.size
    zeros = np.zeros(count)
    ones = np.ones(count)
    theta_column = reference_stack_coordinates(np.asarray(theta_template))
    columns = [
        theta_column,
        np.concatenate((ones, zeros)),
        np.concatenate((zeros, ones)),
        np.concatenate((years, zeros)),
        np.concatenate((zeros, years)),
        reference_stack_coordinates(parallax),
    ]
    if offset_aware_blend_model and blend_flux_ratio > 0:
        weights = np.asarray(blend_weights)
        columns.extend(
            [
                np.concatenate((weights, zeros)),
                np.concatenate((zeros, weights)),
            ]
        )
    return np.column_stack(columns), reference_stack_coordinates(np.asarray(true_mean))


def reference_cell_oracle(
    times_days: ArrayLike,
    truth: TruthConfig,
    blend_flux_ratio: float,
    blend_offset_mas: float,
    sigma_mas: float,
    *,
    offset_aware_blend_model: bool,
) -> ReferenceCellOracle:
    """Fit the independent reference design with QR, not the production SVD solver."""

    if not math.isfinite(sigma_mas) or sigma_mas <= 0:
        raise ValueError("sigma_mas must be finite and positive")
    matrix, true_mean = _reference_problem(
        times_days,
        truth,
        blend_flux_ratio,
        blend_offset_mas,
        offset_aware_blend_model=offset_aware_blend_model,
    )
    whitened_matrix = matrix / sigma_mas
    whitened_mean = true_mean / sigma_mas
    q_matrix, r_matrix = np.linalg.qr(whitened_matrix, mode="reduced")
    if np.linalg.matrix_rank(r_matrix) != matrix.shape[1]:
        raise np.linalg.LinAlgError("reference design is rank deficient")
    coefficients = np.linalg.solve(r_matrix, q_matrix.T @ whitened_mean)
    r_inverse = np.linalg.solve(r_matrix, np.eye(r_matrix.shape[0]))
    covariance = r_inverse @ r_inverse.T
    residual = true_mean - matrix @ coefficients
    noncentrality = float((residual / sigma_mas) @ (residual / sigma_mas))
    return ReferenceCellOracle(
        theta_bias_mas=float(coefficients[0] - truth.theta_e_mas),
        theta_standard_error_mas=float(math.sqrt(covariance[0, 0])),
        residual_noncentrality_lambda=max(0.0, noncentrality),
        residual_degrees_of_freedom=matrix.shape[0] - matrix.shape[1],
    )
