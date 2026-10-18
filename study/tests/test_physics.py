import numpy as np
import pytest

from ghostlight.config import KAPPA_MAS_PER_SOLAR_MASS, TruthConfig
from ghostlight.physics import (
    baseline_relative_centroid_mas,
    blend_coefficients,
    dark_centroid_shift_mas,
    direct_flux_weighted_difference_mas,
    magnification,
    mass_solar,
    relative_parallax_mas,
    theta_e_mas,
)


def test_einstein_radius_and_mass_are_inverse_with_mas_kpc_units():
    pi_rel = relative_parallax_mas(1.0 / 0.58, 1.0 / 0.11)
    assert float(pi_rel) == pytest.approx(0.47)
    theta = theta_e_mas(6.0, pi_rel)
    assert float(theta) == pytest.approx(np.sqrt(KAPPA_MAS_PER_SOLAR_MASS * 6.0 * 0.47))
    assert float(mass_solar(theta, pi_rel)) == pytest.approx(6.0)


def test_point_lens_analytic_limits():
    assert magnification(np.array([1000.0]))[0] == pytest.approx(1.0, rel=1e-11)
    assert magnification(np.array([1e-5]))[0] == pytest.approx(1e5, rel=1e-5)
    theta = 3.0
    u = np.array([[np.sqrt(2.0), 0.0]])
    shift = dark_centroid_shift_mas(u, theta)
    assert np.linalg.norm(shift[0]) == pytest.approx(theta / (2.0 * np.sqrt(2.0)))


def test_exact_blend_formula_matches_direct_flux_weighting():
    u_vector = np.array([[0.2, 0.1], [0.8, -0.4], [3.0, 1.0]])
    source = np.array([[10.0, -3.0], [10.2, -3.1], [11.0, -3.4]])
    offset = np.array([7.0, -2.5])
    reduced = baseline_relative_centroid_mas(u_vector, 4.8, 0.3, offset)
    direct = direct_flux_weighted_difference_mas(source, u_vector, 4.8, 0.3, offset)
    np.testing.assert_allclose(reduced, direct, rtol=1e-13, atol=1e-13)


def test_blend_limits_and_signs():
    u_vector = np.array([[0.5, 0.2], [2.0, -0.1]])
    dark = dark_centroid_shift_mas(u_vector, 2.0)
    no_blend = baseline_relative_centroid_mas(u_vector, 2.0, 0.0, np.array([50.0, 10.0]))
    np.testing.assert_allclose(no_blend, dark)

    u = np.linalg.norm(u_vector, axis=1)
    gravity_weight, blend_weight = blend_coefficients(u, 0.4)
    assert np.all((gravity_weight > 0) & (gravity_weight < 1))
    assert np.all(blend_weight < 0)
    colocated = baseline_relative_centroid_mas(u_vector, 2.0, 0.4, np.zeros(2))
    np.testing.assert_allclose(colocated, gravity_weight[:, None] * dark)


def test_invalid_physical_inputs_are_rejected():
    with pytest.raises(ValueError):
        magnification([0.0])
    with pytest.raises(ValueError):
        theta_e_mas(-1.0, 0.1)
    with pytest.raises(ValueError):
        relative_parallax_mas(-1.0, 8.0)


def test_ob110462_scale_truth_is_internally_consistent():
    truth = TruthConfig()
    assert truth.theta_e_mas == pytest.approx(4.7922, rel=2e-4)
    assert truth.source_distance_kpc == pytest.approx(1 / 0.11)
    assert truth.lens_distance_kpc == pytest.approx(1 / 0.58)
    assert truth.relative_proper_motion_mas_per_year == pytest.approx(
        truth.theta_e_mas / (276 / 365.25)
    )
