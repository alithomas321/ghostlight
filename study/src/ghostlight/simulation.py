"""Paired Monte Carlo calibration-sensitivity atlas and uncertainty summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ghostlight.config import StudyConfig, scenario_id, scenario_spawn_identity
from ghostlight.diagnostics import residual_model_adequacy
from ghostlight.inference import (
    build_design_matrix,
    make_observation_schedule,
    prepare_solver,
    stack_coordinates,
)
from ghostlight.physics import (
    astrometric_baseline_mas,
    baseline_relative_centroid_mas,
    blend_coefficients,
    lens_trajectory,
)
from ghostlight.reference import reference_cell_oracle

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StudyResults:
    metrics_rows: list[dict[str, Any]]
    scenario_rows: list[dict[str, Any]]
    focus_trials: list[dict[str, Any]]
    primary_paired_summary: dict[str, Any]
    preflight_summary: dict[str, Any]


class PreflightError(RuntimeError):
    """Raised before random draws when a fixed design is unusable."""


def normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def exact_normal_coverage(standardized_bias: float, z: float) -> float:
    """Coverage oracle for N(bias, se^2) with a symmetric z*se interval."""

    absolute_bias = abs(standardized_bias)
    return normal_cdf(z - absolute_bias) - normal_cdf(-z - absolute_bias)


def _bootstrap_rmse_interval(
    estimates: FloatArray,
    truth: float,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[float, float]:
    if resamples <= 0:
        return math.nan, math.nan
    count = estimates.size
    indices = rng.integers(0, count, size=(resamples, count))
    errors = estimates[indices] - truth
    values = np.sqrt(np.mean(errors**2, axis=1))
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def _metric_row(
    *,
    scenario_id: str,
    model: str,
    estimates: FloatArray,
    theta_truth: float,
    theta_se: float,
    z: float,
    false_precision_half_width_fractions: tuple[float, ...],
    analytic_bias: float,
    solver_rank: int,
    condition_number: float,
    normalized_condition_number: float,
    effective_information: float,
    bootstrap_rng: np.random.Generator,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    count = estimates.size
    finite_mask = np.isfinite(estimates)
    finite_estimates = estimates[finite_mask]
    finite_count = finite_estimates.size
    nonfinite_count = count - finite_count
    finite_errors = finite_estimates - theta_truth
    if finite_count:
        bias: float | None = float(np.mean(finite_errors))
        rmse: float | None = float(np.sqrt(np.mean(finite_errors**2)))
        rmse_low, rmse_high = _bootstrap_rmse_interval(
            finite_estimates, theta_truth, bootstrap_rng, bootstrap_resamples
        )
        theta_mean: float | None = float(np.mean(finite_estimates))
    else:
        bias = None
        rmse = None
        rmse_low, rmse_high = None, None
        theta_mean = None
    if finite_count > 1:
        error_sd = float(np.std(finite_errors, ddof=1))
        bias_mcse: float | None = error_sd / math.sqrt(finite_count)
        standardized_variance: float | None = float(np.var(finite_errors / theta_se, ddof=1))
        standardized_variance_mcse: float | None = math.sqrt(2.0 / (finite_count - 1))
    else:
        bias_mcse = None
        standardized_variance = None
        standardized_variance_mcse = None
    half_width = z * theta_se
    covered = np.zeros(count, dtype=bool)
    covered[finite_mask] = np.abs(finite_errors) <= half_width
    coverage_count = int(np.sum(covered))
    coverage = coverage_count / count
    coverage_low, coverage_high = wilson_interval(coverage_count, count, z)
    interval_width = 2.0 * half_width
    standardized_bias = analytic_bias / theta_se
    oracle_coverage = exact_normal_coverage(standardized_bias, z)
    oracle_binomial_mcse = math.sqrt(oracle_coverage * (1.0 - oracle_coverage) / count)
    row = {
        "scenario_id": scenario_id,
        "model": model,
        "trials": count,
        "finite_n": finite_count,
        "theta_e_truth_mas": theta_truth,
        "theta_e_mean_mas": theta_mean,
        "bias_mas": bias,
        "bias_fraction": None if bias is None else bias / theta_truth,
        "bias_mcse_mas": bias_mcse,
        "bias_mc95_low_mas": (None if bias is None or bias_mcse is None else bias - z * bias_mcse),
        "bias_mc95_high_mas": (None if bias is None or bias_mcse is None else bias + z * bias_mcse),
        "analytic_bias_mas": analytic_bias,
        "analytic_standardized_bias": standardized_bias,
        "standardized_error_mean": None if bias is None else bias / theta_se,
        "standardized_error_mean_mcse": (None if bias_mcse is None else bias_mcse / theta_se),
        "standardized_error_analytic_mean": standardized_bias,
        "standardized_error_sample_variance": standardized_variance,
        "standardized_error_variance_oracle": 1.0,
        "standardized_error_variance_mcse_normal_oracle": standardized_variance_mcse,
        "rmse_mas": rmse,
        "rmse_bootstrap95_low_mas": rmse_low,
        "rmse_bootstrap95_high_mas": rmse_high,
        "oracle_rmse_mas": math.sqrt(theta_se**2 + analytic_bias**2),
        "theta_e_standard_error_mas": theta_se,
        "interval_width_mas": interval_width,
        "interval_fractional_width": interval_width / theta_truth,
        "interval_fractional_half_width": half_width / theta_truth,
        "coverage_count": coverage_count,
        "coverage_95": coverage,
        "coverage_wilson95_low": coverage_low,
        "coverage_wilson95_high": coverage_high,
        "coverage_mcse": oracle_binomial_mcse,
        "coverage_mcse_basis": (
            "Binomial Monte Carlo SE evaluated at the exact normal-theory coverage oracle; "
            "Wilson endpoints remain count-based."
        ),
        "oracle_coverage_95": oracle_coverage,
        "design_rank": solver_rank,
        "design_rank_failure": False,
        "design_rank_failure_rate": 0.0,
        "nonfinite_estimate_failure_count": nonfinite_count,
        "nonfinite_estimate_failure_rate": nonfinite_count / count,
        "design_condition_number": condition_number,
        "normalized_design_condition_number": normalized_condition_number,
        "effective_theta_information_per_mas2": effective_information,
    }
    for index, threshold in enumerate(false_precision_half_width_fractions):
        threshold_percent = int(round(100 * threshold))
        eligible = half_width <= threshold * theta_truth
        prefix = f"false_precision_{threshold_percent}pct_halfwidth"
        if eligible:
            # Nonfinite estimates count as uncovered in coverage's full denominator,
            # but never enter a strict narrow-miss numerator.
            events = finite_mask & (~covered)
            event_count: int | None = int(np.sum(events))
            rate: float | None = event_count / count
            low, high = wilson_interval(event_count, count, z)
            mcse: float | None = oracle_binomial_mcse
            oracle_rate: float | None = 1.0 - oracle_coverage
            mcse_basis: str | None = (
                "Binomial Monte Carlo SE evaluated from the complementary exact "
                "normal-coverage oracle; Wilson endpoints remain count-based."
            )
        else:
            event_count = None
            rate = None
            low, high = None, None
            mcse = None
            oracle_rate = None
            mcse_basis = None
        row[f"{prefix}_count"] = event_count
        row[f"{prefix}_rate"] = rate
        row[f"{prefix}_wilson95_low"] = low
        row[f"{prefix}_wilson95_high"] = high
        row[f"{prefix}_mcse"] = mcse
        row[f"{prefix}_mcse_basis"] = mcse_basis
        row[f"oracle_{prefix}_rate"] = oracle_rate
        row[f"{prefix}_eligible"] = eligible
        if index == 0:
            # Stable primary aliases consumed by the generated atlas and portfolio UI.
            row["false_confidence_count"] = event_count
            row["false_confidence_rate"] = rate
            row["false_confidence_wilson95_low"] = low
            row["false_confidence_wilson95_high"] = high
            row["false_confidence_mcse"] = mcse
            row["false_confidence_mcse_basis"] = mcse_basis
            row["oracle_false_confidence_rate"] = oracle_rate
            row["narrow_interval"] = eligible
    return row


def _blend_vector(offset_mas: float, angle_deg: float) -> FloatArray:
    angle = np.deg2rad(angle_deg)
    return offset_mas * np.array([np.cos(angle), np.sin(angle)])


def _fixed_model_fields(
    config: StudyConfig,
    *,
    times: FloatArray,
    sigma: float,
    flux_ratio: float,
    offset: float,
    true_mean: FloatArray,
    design,
    solver,
    offset_aware_blend_model: bool,
) -> dict[str, Any]:
    """Calculate deterministic production and independent-reference oracles."""

    theta_truth = config.truth.theta_e_mas
    if offset_aware_blend_model:
        analytic_bias = float(solver.estimate(true_mean)[0] - theta_truth)
    else:
        u_vector = lens_trajectory(times, config.truth)
        _, blend_weight = blend_coefficients(np.linalg.norm(u_vector, axis=1), flux_ratio)
        missed_centroid = (
            blend_weight[:, None] * _blend_vector(offset, config.truth.blend_angle_deg)[None, :]
        )
        analytic_bias = float(solver.estimate(stack_coordinates(missed_centroid))[0])

    adequacy = residual_model_adequacy(design, solver, true_mean, sigma)
    reference = reference_cell_oracle(
        times,
        config.truth,
        flux_ratio,
        offset,
        sigma,
        offset_aware_blend_model=offset_aware_blend_model,
    )
    fields = adequacy.metric_fields()
    fields.update(
        {
            "analytic_bias_mas": analytic_bias,
            "reference_analytic_bias_mas": reference.theta_bias_mas,
            "reference_theta_e_standard_error_mas": reference.theta_standard_error_mas,
            "reference_residual_noncentrality_lambda": (reference.residual_noncentrality_lambda),
            "reference_residual_degrees_of_freedom": (reference.residual_degrees_of_freedom),
            "analytic_bias_minus_reference_mas": (analytic_bias - reference.theta_bias_mas),
            "theta_standard_error_minus_reference_mas": (
                float(solver.standard_errors[0]) - reference.theta_standard_error_mas
            ),
            "residual_noncentrality_lambda_minus_reference": (
                adequacy.noncentrality_lambda - reference.residual_noncentrality_lambda
            ),
            "reference_oracle_method": (
                "Independent scalar flux-weighting/design equations with a QR linear solve; "
                "does not call production physics, stacking, design, or SVD solver functions."
            ),
        }
    )
    return fields


def deterministic_model_fields(
    config: StudyConfig,
    epochs: int,
    sigma: float,
    flux_ratio: float,
    offset: float,
    model: str,
) -> dict[str, Any]:
    """Reconstruct per-cell analytic fields without any stochastic draw."""

    if model not in {"offset_aware_blend", "colocated_blend"}:
        raise ValueError(f"unknown model: {model}")
    truth = config.truth
    times = make_observation_schedule(epochs, truth, config.schedule_baseline_years)
    blend_vector = _blend_vector(offset, truth.blend_angle_deg)
    u_vector = lens_trajectory(times, truth)
    true_mean = stack_coordinates(
        astrometric_baseline_mas(times, truth)
        + baseline_relative_centroid_mas(u_vector, truth.theta_e_mas, flux_ratio, blend_vector)
    )
    offset_aware = model == "offset_aware_blend"
    design = build_design_matrix(
        times,
        truth,
        flux_ratio,
        offset_aware_blend_model=offset_aware,
    )
    solver = prepare_solver(design, sigma)
    return _fixed_model_fields(
        config,
        times=times,
        sigma=sigma,
        flux_ratio=flux_ratio,
        offset=offset,
        true_mean=true_mean,
        design=design,
        solver=solver,
        offset_aware_blend_model=offset_aware,
    )


def _indexed_scenarios(config: StudyConfig):
    return (
        (cadence_index, sigma_index, flux_index, offset_index, epochs, sigma, flux_ratio, offset)
        for cadence_index, epochs in enumerate(config.cadences)
        for sigma_index, sigma in enumerate(config.astrometric_sigmas_mas)
        for flux_index, flux_ratio in enumerate(config.blend_flux_ratios)
        for offset_index, offset in enumerate(config.blend_offsets_mas)
    )


def preflight_designs(config: StudyConfig) -> dict[str, Any]:
    """Check every fixed design before creating any Monte Carlo random draw."""

    failures: list[dict[str, Any]] = []
    checked = 0
    for _, _, _, _, epochs, sigma, flux_ratio, offset in _indexed_scenarios(config):
        selected_scenario_id = scenario_id(epochs, sigma, flux_ratio, offset)
        times = make_observation_schedule(epochs, config.truth, config.schedule_baseline_years)
        for model, offset_aware in (
            ("offset_aware_blend", True),
            ("colocated_blend", False),
        ):
            checked += 1
            try:
                design = build_design_matrix(
                    times, config.truth, flux_ratio, offset_aware_blend_model=offset_aware
                )
                if not np.all(np.isfinite(design.values)):
                    raise FloatingPointError("nonfinite design value")
                solver = prepare_solver(design, sigma)
                if not all(
                    np.all(np.isfinite(value))
                    for value in (solver.projection, solver.covariance, solver.standard_errors)
                ):
                    raise FloatingPointError("nonfinite solver value")
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                failures.append(
                    {
                        "scenario_id": selected_scenario_id,
                        "model": model,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
    rank_failures = sum(item["error_type"] == "LinAlgError" for item in failures)
    nonfinite_failures = sum(item["error_type"] == "FloatingPointError" for item in failures)
    summary = {
        "status": "passed" if not failures else "failed",
        "scenario_count": config.scenario_count,
        "model_cell_count": checked,
        "fixed_design_rank_failure_count": rank_failures,
        "fixed_design_rank_failure_rate": rank_failures / checked,
        "fixed_design_nonfinite_failure_count": nonfinite_failures,
        "fixed_design_nonfinite_failure_rate": nonfinite_failures / checked,
        "other_fixed_design_failure_count": len(failures) - rank_failures - nonfinite_failures,
        "failures": failures,
        "policy": (
            "Any failure stops the locally prespecified validation run before random draws "
            "and requires a "
            "transparent amendment to the locally prespecified analysis specification."
        ),
    }
    if failures:
        raise PreflightError(f"fixed-design preflight failed: {failures}")
    return summary


def run_study(config: StudyConfig) -> StudyResults:
    """Run the fixed grid with paired noise draws for both structural fits."""

    config.validate(prespecified_validation=False)
    preflight_summary = preflight_designs(config)
    truth = config.truth
    theta_truth = truth.theta_e_mas
    metric_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    focus_trials: list[dict[str, Any]] = []
    primary_paired_summary: dict[str, Any] = {}

    for (
        _,
        _,
        _,
        _,
        epochs,
        sigma,
        flux_ratio,
        offset,
    ) in _indexed_scenarios(config):
        selected_scenario_id = scenario_id(epochs, sigma, flux_ratio, offset)
        # Encode the cell values, not their tuple positions, so reordering the
        # configuration cannot silently change a scenario's draws.
        cell_identity = scenario_spawn_identity(epochs, sigma, flux_ratio, offset)
        science_spawn_key = (*cell_identity, 0)
        bootstrap_spawn_key = (*cell_identity, 1)
        rng = np.random.default_rng(
            np.random.SeedSequence(config.seed, spawn_key=science_spawn_key)
        )
        bootstrap_rng = np.random.default_rng(
            np.random.SeedSequence(config.seed, spawn_key=bootstrap_spawn_key)
        )
        times = make_observation_schedule(epochs, truth, config.schedule_baseline_years)
        blend_vector = _blend_vector(offset, truth.blend_angle_deg)
        u_vector = lens_trajectory(times, truth)
        baseline = astrometric_baseline_mas(times, truth)
        transient = baseline_relative_centroid_mas(u_vector, theta_truth, flux_ratio, blend_vector)
        true_mean = stack_coordinates(baseline + transient)

        offset_aware_design = build_design_matrix(
            times, truth, flux_ratio, offset_aware_blend_model=True
        )
        co_located_design = build_design_matrix(
            times, truth, flux_ratio, offset_aware_blend_model=False
        )
        offset_aware_solver = prepare_solver(offset_aware_design, sigma)
        co_located_solver = prepare_solver(co_located_design, sigma)

        noise = rng.normal(0.0, sigma, size=(config.trials_per_scenario, 2 * epochs))
        observations = true_mean[None, :] + noise
        offset_aware_theta = offset_aware_solver.estimate(observations)[:, 0]
        co_located_theta = co_located_solver.estimate(observations)[:, 0]

        offset_aware_fixed = _fixed_model_fields(
            config,
            times=times,
            sigma=sigma,
            flux_ratio=flux_ratio,
            offset=offset,
            true_mean=true_mean,
            design=offset_aware_design,
            solver=offset_aware_solver,
            offset_aware_blend_model=True,
        )
        co_located_fixed = _fixed_model_fields(
            config,
            times=times,
            sigma=sigma,
            flux_ratio=flux_ratio,
            offset=offset,
            true_mean=true_mean,
            design=co_located_design,
            solver=co_located_solver,
            offset_aware_blend_model=False,
        )
        offset_aware_expectation_bias = float(offset_aware_fixed["analytic_bias_mas"])
        co_located_omitted_bias = float(co_located_fixed["analytic_bias_mas"])

        scenario_rows.append(
            {
                "scenario_id": selected_scenario_id,
                "epochs": epochs,
                "sigma_mas_per_coordinate": sigma,
                "blend_flux_ratio": flux_ratio,
                "blend_offset_mas": offset,
                "blend_offset_x_mas": float(blend_vector[0]),
                "blend_offset_y_mas": float(blend_vector[1]),
                "schedule_first_day": float(times[0]),
                "schedule_last_day": float(times[-1]),
                "paired_trials": config.trials_per_scenario,
                "science_spawn_key": ":".join(str(value) for value in science_spawn_key),
                "bootstrap_spawn_key": ":".join(str(value) for value in bootstrap_spawn_key),
            }
        )

        false_precision_thresholds = (
            config.primary_false_precision_half_width_fraction,
            *config.exploratory_false_precision_half_width_fractions,
        )

        offset_aware_row = _metric_row(
            scenario_id=selected_scenario_id,
            model="offset_aware_blend",
            estimates=offset_aware_theta,
            theta_truth=theta_truth,
            theta_se=float(offset_aware_solver.standard_errors[0]),
            z=config.interval_z,
            false_precision_half_width_fractions=false_precision_thresholds,
            analytic_bias=offset_aware_expectation_bias,
            solver_rank=offset_aware_solver.rank,
            condition_number=offset_aware_solver.condition_number,
            normalized_condition_number=offset_aware_solver.normalized_condition_number,
            effective_information=offset_aware_solver.effective_theta_information_per_mas2,
            bootstrap_rng=bootstrap_rng,
            bootstrap_resamples=config.bootstrap_resamples,
        )
        offset_aware_row.update(
            {key: value for key, value in offset_aware_fixed.items() if key != "analytic_bias_mas"}
        )
        metric_rows.append(offset_aware_row)
        co_located_row = _metric_row(
            scenario_id=selected_scenario_id,
            model="colocated_blend",
            estimates=co_located_theta,
            theta_truth=theta_truth,
            theta_se=float(co_located_solver.standard_errors[0]),
            z=config.interval_z,
            false_precision_half_width_fractions=false_precision_thresholds,
            analytic_bias=co_located_omitted_bias,
            solver_rank=co_located_solver.rank,
            condition_number=co_located_solver.condition_number,
            normalized_condition_number=co_located_solver.normalized_condition_number,
            effective_information=co_located_solver.effective_theta_information_per_mas2,
            bootstrap_rng=bootstrap_rng,
            bootstrap_resamples=config.bootstrap_resamples,
        )
        co_located_row.update(
            {key: value for key, value in co_located_fixed.items() if key != "analytic_bias_mas"}
        )
        metric_rows.append(co_located_row)

        is_focus = epochs == 32 and sigma == 0.25 and flux_ratio == 0.3 and offset == 20.0
        if is_focus:
            offset_aware_half = config.interval_z * offset_aware_solver.standard_errors[0]
            co_located_half = config.interval_z * co_located_solver.standard_errors[0]
            offset_aware_covers = np.abs(offset_aware_theta - theta_truth) <= offset_aware_half
            co_located_covers = np.abs(co_located_theta - theta_truth) <= co_located_half
            offset_aware_finite = np.isfinite(offset_aware_theta)
            co_located_finite = np.isfinite(co_located_theta)
            offset_aware_covers &= offset_aware_finite
            co_located_covers &= co_located_finite
            paired_differences = offset_aware_covers.astype(float) - co_located_covers.astype(float)
            coverage_gap = float(np.mean(paired_differences))
            coverage_gap_mcse = float(np.std(paired_differences, ddof=1)) / math.sqrt(
                config.trials_per_scenario
            )
            primary_paired_summary = {
                "definition": (
                    "offset-aware coverage minus co-located-blend coverage on paired noise draws"
                ),
                "paired_trials": config.trials_per_scenario,
                "coverage_gap": coverage_gap,
                "coverage_gap_mcse": coverage_gap_mcse,
                "coverage_gap_mc95_low": coverage_gap - config.interval_z * coverage_gap_mcse,
                "coverage_gap_mc95_high": coverage_gap + config.interval_z * coverage_gap_mcse,
                "discordant_offset_aware_only_count": int(
                    np.sum(offset_aware_covers & ~co_located_covers)
                ),
                "discordant_co_located_only_count": int(
                    np.sum(~offset_aware_covers & co_located_covers)
                ),
                "both_cover_count": int(np.sum(offset_aware_covers & co_located_covers)),
                "neither_covers_count": int(np.sum(~offset_aware_covers & ~co_located_covers)),
                "offset_aware_nonfinite_count": int(np.sum(~offset_aware_finite)),
                "co_located_nonfinite_count": int(np.sum(~co_located_finite)),
                "analytic_offset_aware_coverage_oracle": exact_normal_coverage(
                    offset_aware_expectation_bias / offset_aware_solver.standard_errors[0],
                    config.interval_z,
                ),
                "analytic_co_located_coverage_oracle": exact_normal_coverage(
                    co_located_omitted_bias / co_located_solver.standard_errors[0],
                    config.interval_z,
                ),
                "analytic_coverage_gap_oracle": (
                    exact_normal_coverage(
                        offset_aware_expectation_bias / offset_aware_solver.standard_errors[0],
                        config.interval_z,
                    )
                    - exact_normal_coverage(
                        co_located_omitted_bias / co_located_solver.standard_errors[0],
                        config.interval_z,
                    )
                ),
                "population_p_value": None,
                "uncertainty_scope": (
                    "No population hypothesis test or p-value; uncertainty is Monte Carlo "
                    "execution error for the fixed synthetic design."
                ),
            }
            for trial, (offset_aware_value, co_located_value) in enumerate(
                zip(offset_aware_theta, co_located_theta, strict=True)
            ):
                focus_trials.append(
                    {
                        "trial": trial,
                        "offset_aware_theta_e_mas": float(offset_aware_value),
                        "offset_aware_ci95_low_mas": float(offset_aware_value - offset_aware_half),
                        "offset_aware_ci95_high_mas": float(offset_aware_value + offset_aware_half),
                        "co_located_theta_e_mas": float(co_located_value),
                        "co_located_ci95_low_mas": float(co_located_value - co_located_half),
                        "co_located_ci95_high_mas": float(co_located_value + co_located_half),
                        "paired_co_located_minus_offset_aware_mas": float(
                            co_located_value - offset_aware_value
                        ),
                        "offset_aware_covers_truth": bool(offset_aware_covers[trial]),
                        "co_located_covers_truth": bool(co_located_covers[trial]),
                    }
                )

    return StudyResults(
        metric_rows,
        scenario_rows,
        focus_trials,
        primary_paired_summary,
        preflight_summary,
    )
