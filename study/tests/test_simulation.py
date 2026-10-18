from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest

from ghostlight.config import StudyConfig, TruthConfig
from ghostlight.simulation import (
    _metric_row,
    exact_normal_coverage,
    preflight_designs,
    run_study,
    wilson_interval,
)


def test_wilson_interval_known_values_and_boundaries():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038315, rel=1e-6)
    assert high == pytest.approx(0.5961685, rel=1e-6)
    zero_low, zero_high = wilson_interval(0, 100)
    assert zero_low == pytest.approx(0.0, abs=1e-15)
    assert zero_high == pytest.approx(0.0369935, rel=1e-5)
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_exact_normal_coverage_is_nominal_when_unbiased_and_declines_with_bias():
    z = 1.959963984540054
    assert exact_normal_coverage(0.0, z) == pytest.approx(0.95)
    assert exact_normal_coverage(2.0, z) < 0.5
    assert exact_normal_coverage(-2.0, z) == pytest.approx(exact_normal_coverage(2.0, z))
    positive_tail = exact_normal_coverage(10.873, z)
    negative_tail = exact_normal_coverage(-10.873, z)
    assert np.isfinite(positive_tail)
    assert positive_tail > 0
    assert positive_tail < 1e-17
    assert negative_tail == pytest.approx(positive_tail, rel=1e-14)


def test_offset_aware_calibration_and_co_located_coverage_match_exact_oracles():
    config = replace(
        StudyConfig(),
        trials_per_scenario=6_000,
        bootstrap_resamples=8,
        cadences=(32,),
        astrometric_sigmas_mas=(0.25,),
        blend_flux_ratios=(0.3,),
        blend_offsets_mas=(5.0,),
    )
    results = run_study(config)
    assert len(results.metrics_rows) == 2
    for row in results.metrics_rows:
        oracle = row["oracle_coverage_95"]
        mcse = np.sqrt(oracle * (1 - oracle) / row["trials"])
        assert abs(row["coverage_95"] - oracle) <= 4 * mcse + 1 / row["trials"]
        assert abs(row["bias_mas"] - row["analytic_bias_mas"]) <= 4 * row["bias_mcse_mas"]
        for threshold in (10, 25, 50):
            assert f"false_precision_{threshold}pct_halfwidth_rate" in row
            assert f"false_precision_{threshold}pct_halfwidth_wilson95_low" in row
            assert f"false_precision_{threshold}pct_halfwidth_wilson95_high" in row
        assert abs(row["standardized_error_sample_variance"] - 1.0) <= (
            4 * row["standardized_error_variance_mcse_normal_oracle"]
        )
        assert row["standardized_error_analytic_mean"] == pytest.approx(
            row["analytic_standardized_bias"]
        )


def test_locally_prespecified_grid_has_expected_scale_and_paired_count():
    config = StudyConfig()
    assert config.scenario_count == 81
    assert config.synthetic_dataset_count == 162_000
    assert config.trials_per_scenario == 2_000
    assert config.cadences == (12, 32, 72)
    assert config.blend_offsets_mas == (0.0, 5.0, 20.0)
    assert config.schedule_baseline_years == 6.0
    assert config.primary_false_precision_half_width_fraction == 0.10
    assert config.exploratory_false_precision_half_width_fractions == (0.25, 0.50)
    config.validate(prespecified_validation=True)
    with pytest.raises(ValueError, match="exact frozen StudyConfig"):
        replace(config, seed=1).validate(prespecified_validation=True)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        replace(config, bootstrap_resamples=0).validate(prespecified_validation=False)
    with pytest.raises(ValueError, match="trials_per_scenario must be an integer"):
        replace(config, trials_per_scenario=True).validate(prespecified_validation=False)
    with pytest.raises(ValueError, match="at least two"):
        replace(config, trials_per_scenario=1).validate(prespecified_validation=False)
    with pytest.raises(ValueError, match="bootstrap_resamples must be an integer"):
        replace(config, bootstrap_resamples=4.5).validate(prespecified_validation=False)
    with pytest.raises(ValueError, match="cadences must contain integer"):
        replace(config, cadences=(12, 32.0, 72)).validate(prespecified_validation=False)


def test_interval_contract_requires_95pct_and_matching_normal_quantile():
    config = StudyConfig()
    ninety_z = NormalDist().inv_cdf(0.95)
    with pytest.raises(ValueError, match="require interval_level = 0.95"):
        replace(config, interval_level=0.90, interval_z=ninety_z).validate(
            prespecified_validation=False
        )
    with pytest.raises(ValueError, match="standard-normal quantile"):
        replace(config, interval_z=1.8).validate(prespecified_validation=False)
    with pytest.raises(ValueError, match="interval_level must be a finite"):
        replace(config, interval_level=np.nan).validate(prespecified_validation=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mass_solar", np.inf, "mass_solar must be a finite"),
        ("source_parallax_mas", 0.0, "source_parallax_mas must be positive"),
        ("relative_parallax_mas", -0.1, "relative_parallax_mas must be positive"),
        ("t_e_days", 0.0, "t_e_days must be positive"),
        ("u_0", 0.0, "u_0 must be positive"),
        ("t_0_days", np.nan, "t_0_days must be a finite"),
        ("trajectory_angle_deg", np.inf, "trajectory_angle_deg must be a finite"),
        ("blend_angle_deg", np.nan, "blend_angle_deg must be a finite"),
        ("parallax_phase_deg", np.inf, "parallax_phase_deg must be a finite"),
    ],
)
def test_truth_config_rejects_invalid_or_nonfinite_scalars(field, value, message):
    truth = replace(TruthConfig(), **{field: value})
    with pytest.raises(ValueError, match=message):
        replace(StudyConfig(), truth=truth).validate(prespecified_validation=False)


def test_truth_config_rejects_nonfinite_vectors_and_study_axes():
    with pytest.raises(ValueError, match="baseline_position_mas"):
        replace(
            StudyConfig(),
            truth=replace(TruthConfig(), baseline_position_mas=(0.0, np.nan)),
        ).validate(prespecified_validation=False)
    with pytest.raises(ValueError, match="proper_motion_mas_per_year"):
        replace(
            StudyConfig(),
            truth=replace(TruthConfig(), proper_motion_mas_per_year=(0.0, np.inf)),
        ).validate(prespecified_validation=False)
    with pytest.raises(ValueError, match="astrometric_sigmas_mas must be a finite"):
        replace(StudyConfig(), astrometric_sigmas_mas=(0.25, np.nan)).validate(
            prespecified_validation=False
        )
    with pytest.raises(ValueError, match="schedule_baseline_years must be a finite"):
        replace(StudyConfig(), schedule_baseline_years=np.inf).validate(
            prespecified_validation=False
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(StudyConfig(), seed=-1).validate(prespecified_validation=False)
    with pytest.raises(ValueError, match="unique raw values"):
        replace(StudyConfig(), blend_flux_ratios=(0.3, 0.3)).validate(prespecified_validation=False)


def test_near_colliding_custom_axes_are_rejected_before_rng(monkeypatch):
    config = replace(
        StudyConfig(),
        trials_per_scenario=8,
        bootstrap_resamples=2,
        cadences=(12,),
        astrometric_sigmas_mas=(0.101, 0.102),
        blend_flux_ratios=(0.3,),
        blend_offsets_mas=(5.0,),
    )
    rng_constructed = False

    def forbidden_rng(*args, **kwargs):
        nonlocal rng_constructed
        rng_constructed = True
        raise AssertionError("RNG construction must follow scenario-identity validation")

    monkeypatch.setattr(np.random, "default_rng", forbidden_rng)
    with pytest.raises(ValueError, match="scenario IDs collide"):
        run_study(config)
    assert rng_constructed is False


def test_focus_trials_are_paired_draws():
    config = replace(
        StudyConfig(),
        trials_per_scenario=25,
        bootstrap_resamples=2,
        cadences=(32,),
        astrometric_sigmas_mas=(0.25,),
        blend_flux_ratios=(0.3,),
        blend_offsets_mas=(20.0,),
    )
    results = run_study(config)
    assert len(results.focus_trials) == 25
    first = results.focus_trials[0]
    assert first["paired_co_located_minus_offset_aware_mas"] == pytest.approx(
        first["co_located_theta_e_mas"] - first["offset_aware_theta_e_mas"]
    )
    paired = results.primary_paired_summary
    assert paired["paired_trials"] == 25
    assert paired["analytic_coverage_gap_oracle"] == pytest.approx(
        paired["analytic_offset_aware_coverage_oracle"]
        - paired["analytic_co_located_coverage_oracle"]
    )
    assert paired["population_p_value"] is None
    assert "No population hypothesis test" in paired["uncertainty_scope"]


def test_cell_draws_are_stable_when_grid_tuples_are_reordered():
    base = StudyConfig()
    first_config = replace(
        base,
        trials_per_scenario=20,
        bootstrap_resamples=2,
        cadences=(12, 32),
        astrometric_sigmas_mas=(0.25,),
        blend_flux_ratios=(0.3,),
        blend_offsets_mas=(20.0,),
    )
    second_config = replace(first_config, cadences=(32, 12))
    first = run_study(first_config)
    second = run_study(second_config)
    assert first.focus_trials == second.focus_trials
    first_scenario = next(row for row in first.scenario_rows if row["epochs"] == 32)
    second_scenario = next(row for row in second.scenario_rows if row["epochs"] == 32)
    assert first_scenario["science_spawn_key"] == second_scenario["science_spawn_key"]


def test_preflight_covers_every_fixed_locally_prespecified_design():
    summary = preflight_designs(StudyConfig())
    assert summary["status"] == "passed"
    assert summary["model_cell_count"] == 162
    assert summary["fixed_design_rank_failure_count"] == 0
    assert summary["fixed_design_rank_failure_rate"] == 0
    assert summary["fixed_design_nonfinite_failure_count"] == 0
    assert summary["fixed_design_nonfinite_failure_rate"] == 0
    assert "locally prespecified analysis specification" in summary["policy"]
    assert "preregister" not in summary["policy"]


def test_boundary_event_mcse_uses_oracle_and_ineligible_diagnostics_are_null():
    common = {
        "scenario_id": "fixture",
        "model": "fixture",
        "theta_truth": 4.8,
        "z": 1.959963984540054,
        "false_precision_half_width_fractions": (0.10, 0.25, 0.50),
        "analytic_bias": 0.4,
        "solver_rank": 2,
        "condition_number": 2.0,
        "normalized_condition_number": 1.5,
        "effective_information": 400.0,
        "bootstrap_rng": np.random.default_rng(17),
        "bootstrap_resamples": 10,
    }
    boundary = _metric_row(
        **common,
        estimates=np.full(20, 5.4),
        theta_se=0.05,
    )
    oracle = boundary["oracle_coverage_95"]
    expected_mcse = np.sqrt(oracle * (1 - oracle) / boundary["trials"])
    assert boundary["coverage_95"] == 0.0
    assert boundary["coverage_mcse"] == pytest.approx(expected_mcse)
    assert boundary["coverage_mcse"] > 0
    assert boundary["false_precision_10pct_halfwidth_rate"] == 1.0
    assert boundary["false_precision_10pct_halfwidth_mcse"] == pytest.approx(expected_mcse)
    assert boundary["false_precision_10pct_halfwidth_mcse"] > 0
    assert "exact normal-theory coverage oracle" in boundary["coverage_mcse_basis"]

    ineligible = _metric_row(
        **common,
        estimates=np.full(20, 5.4),
        theta_se=1.5,
    )
    for threshold in (10, 25, 50):
        prefix = f"false_precision_{threshold}pct_halfwidth"
        assert ineligible[f"{prefix}_eligible"] is False
        for suffix in (
            "count",
            "rate",
            "wilson95_low",
            "wilson95_high",
            "mcse",
            "mcse_basis",
        ):
            assert ineligible[f"{prefix}_{suffix}"] is None
        assert ineligible[f"oracle_{prefix}_rate"] is None
    assert ineligible["false_confidence_rate"] is None
    assert ineligible["false_confidence_mcse"] is None


def test_nonfinite_estimates_stay_in_coverage_denominator_but_not_false_precision():
    row = _metric_row(
        scenario_id="fixture",
        model="fixture",
        estimates=np.array([4.8, np.nan, 4.89]),
        theta_truth=4.8,
        theta_se=0.05,
        z=1.959963984540054,
        false_precision_half_width_fractions=(0.10, 0.25, 0.50),
        analytic_bias=0.05,
        solver_rank=2,
        condition_number=2.0,
        normalized_condition_number=1.5,
        effective_information=400.0,
        bootstrap_rng=np.random.default_rng(7),
        bootstrap_resamples=10,
    )
    assert row["finite_n"] == 2
    assert row["nonfinite_estimate_failure_count"] == 1
    assert row["nonfinite_estimate_failure_rate"] == pytest.approx(1 / 3)
    assert row["coverage_95"] == pytest.approx(2 / 3)
    assert row["false_precision_10pct_halfwidth_count"] == 0
    assert row["false_precision_25pct_halfwidth_count"] == 0
    assert row["false_precision_50pct_halfwidth_count"] == 0
