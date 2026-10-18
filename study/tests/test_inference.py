import numpy as np
import pytest

from ghostlight.config import TruthConfig
from ghostlight.inference import (
    build_design_matrix,
    make_observation_schedule,
    prepare_solver,
    solve_wls,
    stack_coordinates,
)
from ghostlight.physics import (
    astrometric_baseline_mas,
    baseline_relative_centroid_mas,
    lens_trajectory,
    parallax_factors,
)


def test_wls_recovers_known_parameters_without_noise():
    truth = TruthConfig()
    times = make_observation_schedule(32, truth)
    design = build_design_matrix(times, truth, 0.3, offset_aware_blend_model=True)
    beta = np.array([truth.theta_e_mas, 0.35, -0.20, 2.4, -1.3, 0.11, 3.0, -4.0])
    observations = design.values @ beta
    estimate, covariance = solve_wls(design, observations, 0.25)
    np.testing.assert_allclose(estimate, beta, rtol=1e-11, atol=1e-11)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_svd_covariance_matches_monte_carlo_sampling_covariance():
    truth = TruthConfig()
    times = make_observation_schedule(32, truth)
    design = build_design_matrix(times, truth, 0.3, offset_aware_blend_model=True)
    solver = prepare_solver(design, 0.25)
    rng = np.random.default_rng(8128)
    draws = rng.normal(0.0, 0.25, size=(20_000, design.values.shape[0]))
    estimates = solver.estimate(draws)
    empirical = np.cov(estimates, rowvar=False)
    np.testing.assert_allclose(
        np.diag(empirical), np.diag(solver.covariance), rtol=0.035, atol=1e-8
    )


def test_solver_records_rank_condition_and_effective_information():
    truth = TruthConfig()
    design = build_design_matrix(
        make_observation_schedule(12, truth), truth, 1.0, offset_aware_blend_model=True
    )
    solver = prepare_solver(design, 0.5)
    assert solver.rank == design.values.shape[1]
    assert solver.condition_number >= 1
    assert solver.normalized_condition_number >= 1
    assert solver.effective_theta_information_per_mas2 == pytest.approx(1 / solver.covariance[0, 0])


def test_g_zero_omits_unidentifiable_blend_columns():
    truth = TruthConfig()
    design = build_design_matrix(
        make_observation_schedule(12, truth), truth, 0.0, offset_aware_blend_model=True
    )
    assert "blend_dx_mas" not in design.parameter_names
    assert design.values.shape[1] == 6


def test_schedule_is_deterministic_symmetric_and_has_requested_size():
    truth = TruthConfig()
    first = make_observation_schedule(32, truth)
    second = make_observation_schedule(32, truth)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, -first[::-1])
    assert first.size == 32
    assert first[0] == pytest.approx(truth.t_0_days - 3 * 365.25)
    assert first[-1] == pytest.approx(truth.t_0_days + 3 * 365.25)


def test_nonzero_event_time_defines_baseline_position_and_matches_design():
    truth = TruthConfig(t_0_days=713.5)
    at_event = astrometric_baseline_mas(np.array([truth.t_0_days]), truth)[0]
    expected_at_event = np.asarray(truth.baseline_position_mas) + (
        truth.source_parallax_mas
        * parallax_factors(np.array([truth.t_0_days]), truth.parallax_phase_deg)[0]
    )
    np.testing.assert_allclose(at_event, expected_at_event, rtol=0, atol=1e-14)

    times = make_observation_schedule(32, truth)
    design = build_design_matrix(times, truth, 0.0, offset_aware_blend_model=True)
    mean = astrometric_baseline_mas(times, truth) + baseline_relative_centroid_mas(
        lens_trajectory(times, truth), truth.theta_e_mas, 0.0, np.zeros(2)
    )
    estimate, _ = solve_wls(design, stack_coordinates(mean), 0.25)
    expected_parameters = np.array(
        [
            truth.theta_e_mas,
            *truth.baseline_position_mas,
            *truth.proper_motion_mas_per_year,
            truth.source_parallax_mas,
        ]
    )
    np.testing.assert_allclose(estimate, expected_parameters, rtol=1e-11, atol=1e-11)
