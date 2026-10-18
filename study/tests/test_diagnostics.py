from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ghostlight.config import StudyConfig
from ghostlight.inference import make_observation_schedule
from ghostlight.reference import (
    reference_blend_vector,
    reference_cell_oracle,
    reference_centroid_from_amplification,
    reference_stack_coordinates,
)
from ghostlight.simulation import deterministic_model_fields


def test_primary_residual_diagnostics_reproduce_independent_lack_of_fit_result():
    config = StudyConfig()
    colocated = deterministic_model_fields(config, 32, 0.25, 0.3, 20.0, "colocated_blend")
    aware = deterministic_model_fields(config, 32, 0.25, 0.3, 20.0, "offset_aware_blend")

    assert colocated["residual_degrees_of_freedom"] == 58
    assert colocated["residual_noncentrality_lambda"] == pytest.approx(988.036250199551)
    assert colocated["residual_expected_reduced_q"] == pytest.approx(18.0351077620612)
    assert colocated["omitted_mean_residual_rms_mas"] == pytest.approx(0.982282622561093)
    assert colocated["residual_rejection_power_at_5pct"] == pytest.approx(1.0)
    assert colocated["residual_mean_lack_of_fit_null_tail_probability"] < 1e-160
    assert aware["residual_degrees_of_freedom"] == 56
    assert aware["residual_noncentrality_lambda"] < 1e-20
    assert aware["residual_expected_reduced_q"] == pytest.approx(1.0)
    assert aware["residual_rejection_power_at_5pct"] == pytest.approx(0.05)
    for row in (aware, colocated):
        assert row["residual_observed_q"] is None
        assert row["residual_observed_null_p_value"] is None
        assert row["residual_empirical_rejection_rate"] is None
        assert row["residual_empirical_rejection_available"] is False
        assert "would require rerunning" in row["residual_empirical_unavailability_reason"]


def test_reference_flux_weighting_metamorphic_limits_and_sign():
    delta = np.array([2.0, -1.0])
    offset = np.array([7.0, 3.0])
    amplification = 4.0
    blend_ratio = 0.3

    no_blend = reference_centroid_from_amplification(amplification, delta, 0.0, offset)
    np.testing.assert_allclose(no_blend, delta)
    zero_offset = reference_centroid_from_amplification(
        amplification, delta, blend_ratio, np.zeros(2)
    )
    np.testing.assert_allclose(
        zero_offset,
        amplification / (amplification + blend_ratio) * delta,
    )
    unit_amplification = reference_centroid_from_amplification(1.0, delta, blend_ratio, offset)
    np.testing.assert_allclose(unit_amplification, delta / (1.0 + blend_ratio))
    # For A > 1, a positive source-to-blend offset contributes with a negative sign.
    with_offset = reference_centroid_from_amplification(
        amplification, np.zeros(2), blend_ratio, offset
    )
    assert np.all(with_offset < 0)


def test_reference_orientation_is_360_periodic_and_180_antisymmetric():
    config = StudyConfig()
    times = make_observation_schedule(32, config.truth, config.schedule_baseline_years)

    def oracle(angle):
        truth = replace(config.truth, blend_angle_deg=angle)
        return reference_cell_oracle(
            times,
            truth,
            0.3,
            20.0,
            0.25,
            offset_aware_blend_model=False,
        )

    base = oracle(67.0)
    opposite = oracle(247.0)
    wrapped = oracle(427.0)
    assert opposite.theta_bias_mas == pytest.approx(-base.theta_bias_mas, abs=1e-12)
    assert opposite.residual_noncentrality_lambda == pytest.approx(
        base.residual_noncentrality_lambda, rel=1e-12
    )
    assert wrapped.theta_bias_mas == pytest.approx(base.theta_bias_mas, abs=1e-12)
    assert wrapped.residual_noncentrality_lambda == pytest.approx(
        base.residual_noncentrality_lambda, rel=1e-12
    )
    np.testing.assert_allclose(
        reference_blend_vector(20.0, 427.0), reference_blend_vector(20.0, 67.0)
    )


def test_reference_stacking_and_mas_scaling_contract():
    vectors = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    np.testing.assert_array_equal(reference_stack_coordinates(vectors), [1, 2, 3, 10, 20, 30])

    config = StudyConfig()
    times = make_observation_schedule(32, config.truth, config.schedule_baseline_years)
    small = reference_cell_oracle(
        times, config.truth, 0.3, 5.0, 0.25, offset_aware_blend_model=False
    )
    double = reference_cell_oracle(
        times, config.truth, 0.3, 10.0, 0.25, offset_aware_blend_model=False
    )
    assert double.theta_bias_mas == pytest.approx(2 * small.theta_bias_mas, abs=1e-12)
    assert double.residual_noncentrality_lambda == pytest.approx(
        4 * small.residual_noncentrality_lambda, rel=1e-12
    )


def test_reference_path_does_not_import_production_equation_or_solver_modules():
    source = (Path(__file__).parents[1] / "src/ghostlight/reference.py").read_text()
    assert "ghostlight.physics" not in source
    assert "ghostlight.inference" not in source
    assert "prepare_solver" not in source
