"""Exact point-lens and flux-weighted unresolved-centroid equations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ghostlight.config import KAPPA_MAS_PER_SOLAR_MASS, TruthConfig

FloatArray = NDArray[np.float64]


def relative_parallax_mas(lens_distance_kpc: ArrayLike, source_distance_kpc: float) -> FloatArray:
    """Return pi_rel in mas, using 1 kpc <-> 1 mas parallax."""

    lens = np.asarray(lens_distance_kpc, dtype=float)
    if source_distance_kpc <= 0 or np.any(lens <= 0):
        raise ValueError("distances must be positive")
    return 1.0 / lens - 1.0 / source_distance_kpc


def theta_e_mas(mass_solar: ArrayLike, pi_rel_mas: ArrayLike) -> FloatArray:
    """Angular Einstein radius: theta_E = sqrt(kappa M pi_rel)."""

    mass = np.asarray(mass_solar, dtype=float)
    pi_rel = np.asarray(pi_rel_mas, dtype=float)
    if np.any(mass < 0) or np.any(pi_rel < 0):
        raise ValueError("mass and relative parallax cannot be negative")
    return np.sqrt(KAPPA_MAS_PER_SOLAR_MASS * mass * pi_rel)


def mass_solar(theta_e: ArrayLike, pi_rel_mas: ArrayLike) -> FloatArray:
    """Conditional lens mass for a supplied relative parallax."""

    theta = np.asarray(theta_e, dtype=float)
    pi_rel = np.asarray(pi_rel_mas, dtype=float)
    if np.any(pi_rel <= 0):
        raise ValueError("relative parallax must be positive")
    return theta**2 / (KAPPA_MAS_PER_SOLAR_MASS * pi_rel)


def magnification(u: ArrayLike) -> FloatArray:
    """Total point-source, point-lens magnification."""

    value = np.asarray(u, dtype=float)
    if np.any(value <= 0):
        raise ValueError("u must be strictly positive")
    return (value**2 + 2.0) / (value * np.sqrt(value**2 + 4.0))


def lens_trajectory(times_days: ArrayLike, truth: TruthConfig) -> FloatArray:
    """Dimensionless lens-to-source vector u in a tangent plane."""

    times = np.asarray(times_days, dtype=float)
    tau = (times - truth.t_0_days) / truth.t_e_days
    local = np.column_stack((tau, np.full_like(tau, truth.u_0)))
    angle = np.deg2rad(truth.trajectory_angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return local @ rotation.T


def dark_centroid_shift_mas(u_vector: ArrayLike, theta_e: float) -> FloatArray:
    """Unresolved dark-lens image centroid relative to the unlensed source."""

    vector = np.asarray(u_vector, dtype=float)
    if vector.shape[-1] != 2:
        raise ValueError("u_vector must end in two tangent-plane coordinates")
    u_squared = np.sum(vector**2, axis=-1, keepdims=True)
    return theta_e * vector / (u_squared + 2.0)


def blend_coefficients(u: ArrayLike, blend_flux_ratio: float) -> tuple[FloatArray, FloatArray]:
    """Return gravitational and baseline-relative blend coefficients.

    The first coefficient multiplies the dark-lens centroid shift. The second
    multiplies the source-to-blend offset in the observed-minus-unlensed-baseline
    centroid.
    """

    if blend_flux_ratio < 0:
        raise ValueError("blend flux ratio cannot be negative")
    amplification = magnification(u)
    gravitational = amplification / (amplification + blend_flux_ratio)
    blend = (
        blend_flux_ratio
        * (1.0 - amplification)
        / ((amplification + blend_flux_ratio) * (1.0 + blend_flux_ratio))
    )
    return gravitational, blend


def baseline_relative_centroid_mas(
    u_vector: ArrayLike,
    theta_e: float,
    blend_flux_ratio: float,
    blend_offset_mas: ArrayLike,
) -> FloatArray:
    """Exact lensed centroid minus the same-time unlensed blended baseline."""

    vector = np.asarray(u_vector, dtype=float)
    offset = np.asarray(blend_offset_mas, dtype=float)
    if offset.shape != (2,):
        raise ValueError("blend_offset_mas must be a two-vector")
    u = np.linalg.norm(vector, axis=-1)
    gravitational_weight, blend_weight = blend_coefficients(u, blend_flux_ratio)
    delta = dark_centroid_shift_mas(vector, theta_e)
    return gravitational_weight[:, None] * delta + blend_weight[:, None] * offset


def direct_flux_weighted_difference_mas(
    source_position_mas: ArrayLike,
    u_vector: ArrayLike,
    theta_e: float,
    blend_flux_ratio: float,
    blend_offset_mas: ArrayLike,
) -> FloatArray:
    """Compute the same centroid difference directly from flux weighting."""

    source = np.asarray(source_position_mas, dtype=float)
    vector = np.asarray(u_vector, dtype=float)
    offset = np.asarray(blend_offset_mas, dtype=float)
    u = np.linalg.norm(vector, axis=-1)
    amplification = magnification(u)
    delta = dark_centroid_shift_mas(vector, theta_e)
    blend_position = source + offset
    observed = (amplification[:, None] * (source + delta) + blend_flux_ratio * blend_position) / (
        amplification + blend_flux_ratio
    )[:, None]
    baseline = (source + blend_flux_ratio * blend_position) / (1.0 + blend_flux_ratio)
    return observed - baseline


def parallax_factors(times_days: ArrayLike, phase_deg: float) -> FloatArray:
    """Deterministic two-coordinate annual parallax factors for the synthetic field."""

    times = np.asarray(times_days, dtype=float)
    phase = np.deg2rad(phase_deg)
    orbital_phase = 2.0 * np.pi * times / 365.25 + phase
    # The 0.72 axial ratio represents a fixed projected ecliptic geometry.
    return np.column_stack((np.cos(orbital_phase), 0.72 * np.sin(orbital_phase)))


def astrometric_baseline_mas(times_days: ArrayLike, truth: TruthConfig) -> FloatArray:
    """Linear position/proper motion plus annual source-parallax nuisance."""

    times = np.asarray(times_days, dtype=float)
    years = (times - truth.t_0_days) / 365.25
    position = np.asarray(truth.baseline_position_mas)
    proper_motion = np.asarray(truth.proper_motion_mas_per_year)
    return (
        position[None, :]
        + years[:, None] * proper_motion[None, :]
        + truth.source_parallax_mas * parallax_factors(times, truth.parallax_phase_deg)
    )
