"""Deterministic tables, figures, narrative, and artifact verification."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    _mpl_config = Path(tempfile.gettempdir()) / "ghostlight-matplotlib"
    _mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_config)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ghostlight.config import StudyConfig
from ghostlight.diagnostics import residual_model_adequacy
from ghostlight.inference import (
    build_design_matrix,
    make_observation_schedule,
    prepare_solver,
    stack_coordinates,
    unstack_coordinates,
)
from ghostlight.physics import (
    astrometric_baseline_mas,
    baseline_relative_centroid_mas,
    blend_coefficients,
    lens_trajectory,
    magnification,
    mass_solar,
    relative_parallax_mas,
)
from ghostlight.simulation import StudyResults, exact_normal_coverage

SYNTHETIC_LABEL = "SYNTHETIC KNOWN-TRUTH STUDY · NOT REAL-SURVEY PERFORMANCE"
OFFSET_AWARE_LABEL = "Offset-aware"
COLOCATED_BLEND_LABEL = "Co-located-blend"
STRICT_NARROW_MISS_INELIGIBLE = "N/A — interval too wide for threshold"


def _is_true(value: Any) -> bool:
    """Interpret saved CSV booleans without treating the string ``False`` as true."""

    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _strict_narrow_miss_cell(row: dict[str, Any]) -> tuple[float, bool, str]:
    """Return the strict diagnostic value, eligibility, and public label."""

    eligible = _is_true(row["false_precision_10pct_halfwidth_eligible"])
    if not eligible:
        return math.nan, False, STRICT_NARROW_MISS_INELIGIBLE
    rate = float(row["false_precision_10pct_halfwidth_rate"])
    return rate, True, f"{100 * rate:.2f}%"


def _primary_gap_annotation(paired: dict[str, Any]) -> str:
    """Format the paired primary estimand without hiding Monte Carlo uncertainty."""

    return (
        f"{100 * float(paired['coverage_gap']):.2f} pp\n"
        f"MC95 [{100 * float(paired['coverage_gap_mc95_low']):.2f}, "
        f"{100 * float(paired['coverage_gap_mc95_high']):.2f}] pp\n"
        f"Analytic oracle {100 * float(paired['analytic_coverage_gap_oracle']):.2f} pp"
    )


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _metric_with_scenario(results: StudyResults) -> list[dict[str, Any]]:
    scenarios = {row["scenario_id"]: row for row in results.scenario_rows}
    merged = []
    for row in results.metrics_rows:
        combined = {**scenarios[row["scenario_id"]], **row}
        is_primary = (
            combined["epochs"] == 32
            and combined["sigma_mas_per_coordinate"] == 0.25
            and combined["blend_flux_ratio"] == 0.3
            and combined["blend_offset_mas"] == 20.0
        )
        paired = results.primary_paired_summary if is_primary else {}
        combined.update(
            {
                "is_prespecified_primary_cell": is_primary,
                "paired_coverage_gap": paired.get("coverage_gap"),
                "paired_coverage_gap_mcse": paired.get("coverage_gap_mcse"),
                "paired_coverage_gap_mc95_low": paired.get("coverage_gap_mc95_low"),
                "paired_coverage_gap_mc95_high": paired.get("coverage_gap_mc95_high"),
                "paired_analytic_coverage_gap_oracle": paired.get("analytic_coverage_gap_oracle"),
                "paired_uncertainty_scope": paired.get("uncertainty_scope"),
            }
        )
        merged.append(combined)
    return merged


def _distinct_dgp_macro_coverage(
    rows: list[dict[str, Any]],
    model: str,
    z: float,
) -> dict[str, Any]:
    """Weight each distinct mean-plus-sigma DGP once in a descriptive summary."""

    selected = [row for row in rows if row["model"] == model]
    grouped: dict[tuple[int, float, float, float], list[dict[str, Any]]] = {}
    for row in selected:
        flux_ratio = float(row["blend_flux_ratio"])
        effective_offset = 0.0 if flux_ratio == 0.0 else float(row["blend_offset_mas"])
        key = (
            int(row["epochs"]),
            float(row["sigma_mas_per_coordinate"]),
            flux_ratio,
            effective_offset,
        )
        grouped.setdefault(key, []).append(row)

    rates = []
    variances = []
    for group in grouped.values():
        total = sum(int(row["trials"]) for row in group)
        covered = sum(int(row["coverage_count"]) for row in group)
        oracle_values = {float(row["oracle_coverage_95"]) for row in group}
        if len(oracle_values) != 1:
            raise ValueError("duplicate-control rows disagree on their coverage oracle")
        oracle = oracle_values.pop()
        rates.append(covered / total)
        variances.append(oracle * (1.0 - oracle) / total)

    count = len(rates)
    macro_average = float(np.mean(rates))
    macro_mcse = math.sqrt(sum(variances)) / count
    return {
        "definition": (
            "Unweighted macro-average of empirical coverage across distinct true-mean-plus-"
            "sigma DGPs. Repeated g=0 control rows are first aggregated within their common "
            "DGP; no heterogeneous pooled-binomial Wilson interval is reported."
        ),
        "model": model,
        "grid_row_count": len(selected),
        "distinct_dgp_count": count,
        "repeated_zero_flux_control_rows_collapsed": len(selected) - count,
        "macro_coverage_95": macro_average,
        "macro_coverage_mcse_oracle": macro_mcse,
        "macro_coverage_mc95_low": macro_average - z * macro_mcse,
        "macro_coverage_mc95_high": macro_average + z * macro_mcse,
        "minimum_distinct_dgp_coverage_95": min(rates),
        "maximum_distinct_dgp_coverage_95": max(rates),
        "scope": (
            "Descriptive calibration summary, not a population estimand, binomial sample, "
            "multiplicity-adjusted test, or validation check."
        ),
    }


def _standardized_discrepancy(difference: float, mcse: float) -> float:
    if mcse > 0:
        return abs(difference) / mcse
    return 0.0 if difference == 0 else math.inf


def _metric_discrepancy(observed: Any, expected: Any, mcse: Any) -> float:
    if observed is None or expected is None or mcse is None:
        return math.inf
    values = (float(observed), float(expected), float(mcse))
    if not all(math.isfinite(value) for value in values):
        return math.inf
    return _standardized_discrepancy(values[0] - values[1], values[2])


def evaluate_validation_checks(
    config: StudyConfig,
    results: StudyResults,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate locally prespecified numerical validation checks."""

    bias_ratios = [
        _metric_discrepancy(
            row["bias_mas"],
            row["analytic_bias_mas"],
            row["bias_mcse_mas"],
        )
        for row in rows
    ]
    standardized_mean_ratios = [
        _metric_discrepancy(
            row["standardized_error_mean"],
            row["standardized_error_analytic_mean"],
            row["standardized_error_mean_mcse"],
        )
        for row in rows
    ]
    standardized_variance_ratios = [
        _metric_discrepancy(
            row["standardized_error_sample_variance"],
            row["standardized_error_variance_oracle"],
            row["standardized_error_variance_mcse_normal_oracle"],
        )
        for row in rows
    ]
    coverage_ratios = []
    for row in rows:
        oracle = float(row["oracle_coverage_95"])
        oracle_mcse = math.sqrt(oracle * (1.0 - oracle) / int(row["trials"]))
        coverage_ratios.append(_metric_discrepancy(row["coverage_95"], oracle, oracle_mcse))

    offset_aware_primary = next(
        row
        for row in rows
        if row["model"] == "offset_aware_blend"
        and row["epochs"] == 32
        and row["sigma_mas_per_coordinate"] == 0.25
        and row["blend_flux_ratio"] == 0.3
        and row["blend_offset_mas"] == 20.0
    )
    nominal_mcse = math.sqrt(0.95 * 0.05 / int(offset_aware_primary["trials"]))
    nominal_ratio = _standardized_discrepancy(
        float(offset_aware_primary["coverage_95"]) - 0.95, nominal_mcse
    )
    nonfinite_count = sum(int(row["nonfinite_estimate_failure_count"]) for row in rows)
    preflight_passed = results.preflight_summary["status"] == "passed"
    independent_bias_agreement = all(
        math.isclose(
            float(row["analytic_bias_mas"]),
            float(row["reference_analytic_bias_mas"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        for row in rows
    )
    independent_se_agreement = all(
        math.isclose(
            float(row["theta_e_standard_error_mas"]),
            float(row["reference_theta_e_standard_error_mas"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        for row in rows
    )
    independent_lambda_agreement = all(
        math.isclose(
            float(row["residual_noncentrality_lambda"]),
            float(row["reference_residual_noncentrality_lambda"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for row in rows
    )
    offset_aware_lambda_max = max(
        float(row["residual_noncentrality_lambda"])
        for row in rows
        if row["model"] == "offset_aware_blend"
    )
    validation_check_flags = {
        "analytic_bias_agrees_with_monte_carlo_within_4_mcse": max(bias_ratios) <= 4.0,
        "coverage_oracle_agrees_with_monte_carlo_within_4_mcse": max(coverage_ratios) <= 4.0,
        "standardized_error_variance_agrees_with_unit_oracle_within_4_mcse": (
            max(standardized_variance_ratios) <= 4.0
        ),
        "offset_aware_primary_coverage_compatible_with_95pct_within_4_mcse": (nominal_ratio <= 4.0),
        "all_fixed_designs_pass_preflight": preflight_passed,
        "no_nonfinite_estimates": nonfinite_count == 0,
        "independent_reference_bias_agrees": independent_bias_agreement,
        "independent_reference_standard_error_agrees": independent_se_agreement,
        "independent_reference_residual_lambda_agrees": independent_lambda_agreement,
        "offset_aware_true_mean_is_in_model_space": offset_aware_lambda_max <= 1e-18,
    }
    coverage_gap = float(results.primary_paired_summary["coverage_gap"])
    pivot_threshold = 0.05
    classification = (
        "prespecified_undercoverage_result"
        if coverage_gap >= pivot_threshold
        else "negative_result_pivot"
    )
    return {
        "all_validation_checks_pass": all(validation_check_flags.values()),
        "validation_check_flags": validation_check_flags,
        "validation_check_scope": (
            "Engineering/calibration screening tolerances, not null-hypothesis tests and not "
            "evidence that either fitted model is adequate for an observed dataset."
        ),
        "multiple_check_policy": (
            "The 4-MCSE screens are simultaneous descriptive checks over many correlated "
            "cells. They do not control family-wise error or false discovery rate and may not "
            "be interpreted as a family of prespecified hypothesis tests."
        ),
        "monte_carlo_screen_comparison_count": (
            len(bias_ratios) + len(standardized_variance_ratios) + len(coverage_ratios) + 1
        ),
        "analytic_bias_maximum_discrepancy_mcse": max(bias_ratios),
        "standardized_error_mean_maximum_discrepancy_mcse": max(standardized_mean_ratios),
        "standardized_error_variance_maximum_discrepancy_mcse": max(standardized_variance_ratios),
        "coverage_oracle_maximum_discrepancy_mcse": max(coverage_ratios),
        "offset_aware_primary_nominal_coverage_discrepancy_mcse": nominal_ratio,
        "screening_tolerance_mcse": 4.0,
        "nonfinite_estimate_failure_count": nonfinite_count,
        "offset_aware_maximum_residual_noncentrality_lambda": offset_aware_lambda_max,
        "prespecified_interpretation_threshold": {
            "locally_prespecified_primary_estimand": (
                "paired primary coverage difference: offset-aware minus co-located-blend"
            ),
            "coverage_gap": coverage_gap,
            "threshold_absolute_coverage_points": pivot_threshold,
            "classification": classification,
            "rule": (
                "The token prespecified_undercoverage_result denotes the locally "
                "prespecified primary interpretation when the point estimate is at least "
                "0.05; otherwise use negative_result_pivot. Neither token denotes external "
                "registration or independent confirmation. Strict narrow-miss diagnostics "
                "remain exploratory in both cases."
            ),
        },
    }


def summarize_metrics(config: StudyConfig, results: StudyResults) -> dict[str, Any]:
    rows = _metric_with_scenario(results)
    focus = [
        row
        for row in rows
        if row["epochs"] == 32
        and row["sigma_mas_per_coordinate"] == 0.25
        and row["blend_flux_ratio"] == 0.3
        and row["blend_offset_mas"] == 20.0
    ]
    if len(focus) != 2:
        raise RuntimeError("the locally prespecified primary cell is missing")

    co_located_rows = [row for row in rows if row["model"] == "colocated_blend"]
    offset_aware_macro_coverage = _distinct_dgp_macro_coverage(
        rows, "offset_aware_blend", config.interval_z
    )
    eligible_false_rows = [
        row for row in co_located_rows if row["false_precision_10pct_halfwidth_eligible"]
    ]
    worst_false = (
        max(eligible_false_rows, key=lambda row: float(row["false_confidence_rate"]))
        if eligible_false_rows
        else None
    )
    worst_bias = max(co_located_rows, key=lambda row: abs(float(row["analytic_bias_mas"])))
    oracle_discrepancies = []
    for row in rows:
        oracle = float(row["oracle_coverage_95"])
        oracle_mcse = math.sqrt(oracle * (1.0 - oracle) / int(row["trials"]))
        if oracle_mcse > 0:
            oracle_discrepancies.append(abs(float(row["coverage_95"]) - oracle) / oracle_mcse)

    validation_checks = evaluate_validation_checks(config, results, rows)
    mean_structures_per_cadence = sum(
        1 if flux_ratio == 0 else len(config.blend_offsets_mas)
        for flux_ratio in config.blend_flux_ratios
    )
    distinct_true_means = len(config.cadences) * mean_structures_per_cadence
    distinct_data_generating_distributions = distinct_true_means * len(
        config.astrometric_sigmas_mas
    )
    adequacy_fields = (
        "residual_degrees_of_freedom",
        "residual_noncentrality_lambda",
        "residual_mean_lack_of_fit_q",
        "residual_mean_lack_of_fit_reduced_q",
        "residual_mean_lack_of_fit_null_tail_probability",
        "residual_expected_q",
        "residual_expected_reduced_q",
        "residual_expected_q_null_tail_probability",
        "residual_chi_square_critical_95",
        "residual_rejection_power_at_5pct",
        "omitted_mean_residual_rms_mas",
        "omitted_mean_residual_max_abs_mas",
        "residual_observed_q",
        "residual_observed_reduced_q",
        "residual_observed_null_p_value",
        "residual_empirical_rejection_count",
        "residual_empirical_rejection_rate",
        "residual_empirical_rejection_available",
        "residual_empirical_unavailability_reason",
        "residual_tail_probability_scope",
    )
    return {
        "schema_version": "3.0",
        "study_label": SYNTHETIC_LABEL,
        "study_kind": "conditional known-truth Monte Carlo calibration study",
        "configuration": config.to_dict(),
        "model_definitions": {
            "offset_aware_blend": (
                "Fits the two static source-to-blend offset components whenever g > 0."
            ),
            "colocated_blend": (
                "Fixes the source-to-blend offset to zero while retaining the known blend "
                "flux ratio; evaluated against synthetic data with controlled offsets."
            ),
        },
        "interval_definition": (
            "Known-sigma normal 95% WLS interval conditional on supplied photometric "
            "trajectory, blend ratio, Gaussian independent per-coordinate errors, and model."
        ),
        "false_confidence_definition": (
            "Exploratory diagnostic: a 95% interval that misses true theta_E and has "
            "half-width no more than "
            f"{100 * config.primary_false_precision_half_width_fraction:.0f}% of true "
            "theta_E (equivalently, total width no more than "
            f"{200 * config.primary_false_precision_half_width_fraction:.0f}%)."
        ),
        "false_precision_sensitivity_half_width_fractions": [
            config.primary_false_precision_half_width_fraction,
            *config.exploratory_false_precision_half_width_fractions,
        ],
        "event_rate_mcse_definition": (
            "For per-cell coverage and eligible strict narrow-miss rates, the binomial "
            "Monte Carlo SE is evaluated at the exact normal-theory event probability, "
            "not the empirical plug-in rate. Paired primary-gap MCSE remains the sample "
            "standard deviation of paired indicator differences divided by sqrt(N)."
        ),
        "ineligible_strict_diagnostic_policy": (
            "When fixed interval half-width exceeds a diagnostic threshold, its count, "
            "rate, Wilson endpoints, oracle rate, and MCSE are structurally null."
        ),
        "rmse_bootstrap_scope": (
            f"Fixed-seed {config.bootstrap_resamples}-resample percentile endpoints are "
            "supplemental coarse bootstrap Monte Carlo summaries and are not headline or "
            "validation-check evidence."
        ),
        "model_adequacy_definition": (
            "For each fixed design, minimized known-sigma residual Q follows a noncentral "
            "chi-square law with df = observations - fitted rank and noncentrality equal to "
            "the squared whitened norm of the true-mean component outside the model space. "
            "Observed Q and empirical rejection fields remain null because the frozen "
            "artifacts did not retain per-trial residual vectors."
        ),
        "primary_model_adequacy": {
            row["model"]: {field: row[field] for field in adequacy_fields} for row in focus
        },
        "independent_reference_oracle": {
            "method": (
                "Independent scalar flux weighting and design construction with a QR solve; "
                "the path does not call production physics, stacking, design, or SVD solver "
                "functions."
            ),
            "maximum_absolute_bias_difference_mas": max(
                abs(float(row["analytic_bias_minus_reference_mas"])) for row in rows
            ),
            "maximum_absolute_standard_error_difference_mas": max(
                abs(float(row["theta_standard_error_minus_reference_mas"])) for row in rows
            ),
            "maximum_absolute_residual_lambda_difference": max(
                abs(float(row["residual_noncentrality_lambda_minus_reference"])) for row in rows
            ),
        },
        "design_accounting": {
            "grid_cells": config.scenario_count,
            "distinct_true_mean_vectors": distinct_true_means,
            "distinct_mean_plus_sigma_data_generating_distributions": (
                distinct_data_generating_distributions
            ),
            "repeated_zero_flux_negative_control_cells": (
                config.scenario_count - distinct_data_generating_distributions
            ),
            "zero_flux_grid_rows": (
                len(config.cadences)
                * len(config.astrometric_sigmas_mas)
                * len(config.blend_offsets_mas)
                if 0.0 in config.blend_flux_ratios
                else 0
            ),
        },
        "primary_cell": {row["model"]: row for row in focus},
        "primary_paired_coverage_gap": results.primary_paired_summary,
        "fixed_design_preflight": results.preflight_summary,
        "validation_checks": validation_checks,
        "highlights": {
            "offset_aware_distinct_dgp_macro_coverage": offset_aware_macro_coverage,
            "largest_co_located_false_confidence_cell": worst_false,
            "largest_absolute_co_located_analytic_bias_cell": worst_bias,
            "maximum_coverage_oracle_discrepancy_mcse": max(oracle_discrepancies),
            "design_rank_failure_count": sum(bool(row["design_rank_failure"]) for row in rows),
            "design_rank_failure_rate": sum(bool(row["design_rank_failure"]) for row in rows)
            / len(rows),
            "design_rank_failure_denominator_model_cells": len(rows),
        },
        "claim_boundary": (
            "No real event is fit and no instrument, survey, discovery yield, or operational "
            "performance is estimated. OB110462 supplies scale context only."
        ),
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "#07111f",
            "axes.facecolor": "#0d1a2b",
            "axes.edgecolor": "#7893ad",
            "axes.labelcolor": "#eaf3f7",
            "xtick.color": "#c7d8e3",
            "ytick.color": "#c7d8e3",
            "text.color": "#eaf3f7",
            "grid.color": "#35506b",
            "legend.facecolor": "#0d1a2b",
            "legend.edgecolor": "#7893ad",
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=170,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        metadata={"Software": "Ghostlight 1.0", "Title": SYNTHETIC_LABEL},
    )
    plt.close(fig)


def _synthetic_banner(fig: plt.Figure, *, y: float = 0.995) -> None:
    fig.text(0.5, y, SYNTHETIC_LABEL, ha="center", va="top", color="#f3b562", fontsize=8)


def figure_event_geometry(config: StudyConfig, path: Path) -> None:
    truth = config.truth
    half_baseline_days = 0.5 * config.schedule_baseline_years * 365.25
    times = np.linspace(
        truth.t_0_days - half_baseline_days,
        truth.t_0_days + half_baseline_days,
        600,
    )
    u_vector = lens_trajectory(times, truth)
    u = np.linalg.norm(u_vector, axis=1)
    amp = magnification(u)
    blend = 20.0 * np.array(
        [np.cos(np.deg2rad(truth.blend_angle_deg)), np.sin(np.deg2rad(truth.blend_angle_deg))]
    )
    transient = baseline_relative_centroid_mas(u_vector, truth.theta_e_mas, 0.3, blend)
    unblended = baseline_relative_centroid_mas(u_vector, truth.theta_e_mas, 0.0, np.zeros(2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(times, amp, color="#f3b562", lw=2)
    axes[0].set(
        xlabel="Days from peak", ylabel="Magnification A(u)", title="Known photometric geometry"
    )
    axes[0].grid(alpha=0.35)
    axes[1].plot(unblended[:, 0], unblended[:, 1], color="#5bd6d0", lw=2, label="g = 0")
    axes[1].plot(
        transient[:, 0], transient[:, 1], color="#f06c9b", lw=2, label="g = 0.3, |b| = 20 mas"
    )
    axes[1].scatter([0], [0], s=22, color="#eaf3f7", zorder=3)
    axes[1].set(
        xlabel="Transient x (mas)", ylabel="Transient y (mas)", title="Baseline-relative centroid"
    )
    axes[1].axis("equal")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.35)
    axes[2].plot(times, transient[:, 0], color="#5bd6d0", label="x")
    axes[2].plot(times, transient[:, 1], color="#f06c9b", label="y")
    axes[2].axhline(0, color="#7893ad", lw=0.8)
    axes[2].set(
        xlabel="Days from peak", ylabel="Centroid shift (mas)", title="Signal the fit must explain"
    )
    axes[2].legend()
    axes[2].grid(alpha=0.35)
    fig.suptitle("Ghostlight’s controlled astrometric event", y=0.88, fontsize=14)
    fig.subplots_adjust(top=0.76, wspace=0.22)
    _synthetic_banner(fig)
    _save_figure(fig, path)


def figure_calibration_sensitivity_atlas(results: StudyResults, path: Path) -> None:
    rows = _metric_with_scenario(results)
    colocated = [row for row in rows if row["model"] == "colocated_blend"]
    cadences = sorted({int(row["epochs"]) for row in colocated})
    sigmas = sorted({float(row["sigma_mas_per_coordinate"]) for row in colocated})
    fluxes = sorted({float(row["blend_flux_ratio"]) for row in colocated})
    offsets = sorted({float(row["blend_offset_mas"]) for row in colocated})
    fig, axes = plt.subplots(
        len(fluxes), len(offsets), figsize=(10.5, 8.2), sharex=True, sharey=True
    )
    axes = np.asarray(axes, dtype=object).reshape(len(fluxes), len(offsets))
    cmap = plt.get_cmap("magma").with_extremes(bad="#17283a")
    image = None
    for i, flux in enumerate(fluxes):
        for j, offset in enumerate(offsets):
            grid = np.full((len(sigmas), len(cadences)), np.nan)
            eligible_grid = np.zeros((len(sigmas), len(cadences)), dtype=bool)
            for yi, sigma in enumerate(sigmas):
                for xi, cadence in enumerate(cadences):
                    row = next(
                        item
                        for item in colocated
                        if item["epochs"] == cadence
                        and item["sigma_mas_per_coordinate"] == sigma
                        and item["blend_flux_ratio"] == flux
                        and item["blend_offset_mas"] == offset
                    )
                    value, eligible, _ = _strict_narrow_miss_cell(row)
                    grid[yi, xi] = value
                    eligible_grid[yi, xi] = eligible
            image = axes[i, j].imshow(
                np.ma.masked_invalid(grid),
                vmin=0,
                vmax=1,
                cmap=cmap,
                aspect="auto",
                origin="lower",
            )
            axes[i, j].set_title(f"g={flux:g}, |b|={offset:g} mas", fontsize=9)
            axes[i, j].set_xticks(range(len(cadences)), cadences)
            axes[i, j].set_yticks(range(len(sigmas)), [f"{value:.2f}" for value in sigmas])
            for yi in range(len(sigmas)):
                for xi in range(len(cadences)):
                    if not eligible_grid[yi, xi]:
                        axes[i, j].add_patch(
                            Rectangle(
                                (xi - 0.5, yi - 0.5),
                                1,
                                1,
                                fill=False,
                                hatch="////",
                                edgecolor="#7893ad",
                                linewidth=0.45,
                            )
                        )
                    axes[i, j].text(
                        xi,
                        yi,
                        "N/A" if not eligible_grid[yi, xi] else f"{100 * grid[yi, xi]:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=(
                            "#07111f"
                            if eligible_grid[yi, xi] and grid[yi, xi] >= 0.70
                            else "#eaf3f7"
                        ),
                    )
    for axis in axes[-1, :]:
        axis.set_xlabel("Epochs")
    for axis in axes[:, 0]:
        axis.set_ylabel("σ per coordinate (mas)")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.02)
    colorbar.set_label("Exploratory strict narrow-miss rate\n(eligible cells only)")
    fig.suptitle(
        "Exploratory strict narrow-miss rate for the co-located-blend fit\n"
        "10% half-width criterion · hatched N/A cells are ineligible",
        y=0.965,
        fontsize=13,
    )
    _synthetic_banner(fig)
    _save_figure(fig, path)


def primary_mean_residual_rows(config: StudyConfig) -> list[dict[str, Any]]:
    """Return deterministic residual-by-time support for the primary fixed design."""

    epochs, sigma, flux_ratio, offset = 32, 0.25, 0.3, 20.0
    truth = config.truth
    times = make_observation_schedule(epochs, truth, config.schedule_baseline_years)
    angle = np.deg2rad(truth.blend_angle_deg)
    blend_vector = offset * np.array([np.cos(angle), np.sin(angle)])
    u_vector = lens_trajectory(times, truth)
    true_mean = stack_coordinates(
        astrometric_baseline_mas(times, truth)
        + baseline_relative_centroid_mas(
            u_vector,
            truth.theta_e_mas,
            flux_ratio,
            blend_vector,
        )
    )
    rows: list[dict[str, Any]] = []
    for model, offset_aware in (
        ("offset_aware_blend", True),
        ("colocated_blend", False),
    ):
        design = build_design_matrix(
            times,
            truth,
            flux_ratio,
            offset_aware_blend_model=offset_aware,
        )
        solver = prepare_solver(design, sigma)
        adequacy = residual_model_adequacy(design, solver, true_mean, sigma)
        residual = unstack_coordinates(adequacy.mean_residual)
        for index, time in enumerate(times):
            rows.append(
                {
                    "model": model,
                    "epoch_index": index,
                    "time_days_from_peak": float(time - truth.t_0_days),
                    "mean_residual_x_mas": float(residual[index, 0]),
                    "mean_residual_y_mas": float(residual[index, 1]),
                    "mean_residual_x_sigma": float(residual[index, 0] / sigma),
                    "mean_residual_y_sigma": float(residual[index, 1] / sigma),
                    "residual_degrees_of_freedom": adequacy.degrees_of_freedom,
                    "residual_noncentrality_lambda": adequacy.noncentrality_lambda,
                    "residual_expected_reduced_q": adequacy.expected_reduced_q,
                }
            )
    return rows


def figure_primary_mean_residuals(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot the true-mean component that each primary model leaves unexplained."""

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    for axis, (model, label) in zip(
        axes,
        (
            ("offset_aware_blend", OFFSET_AWARE_LABEL),
            ("colocated_blend", COLOCATED_BLEND_LABEL),
        ),
        strict=True,
    ):
        selected = [row for row in rows if row["model"] == model]
        times = [float(row["time_days_from_peak"]) for row in selected]
        axis.plot(
            times,
            [float(row["mean_residual_x_sigma"]) for row in selected],
            color="#5bd6d0",
            marker="o",
            markersize=2.8,
            label="x",
        )
        axis.plot(
            times,
            [float(row["mean_residual_y_sigma"]) for row in selected],
            color="#f06c9b",
            marker="o",
            markersize=2.8,
            label="y",
        )
        axis.axhline(0, color="#7893ad", lw=0.8)
        axis.set_title(
            f"{label}\n"
            f"λ={float(selected[0]['residual_noncentrality_lambda']):.2f}, "
            f"E[Q/df]={float(selected[0]['residual_expected_reduced_q']):.2f}"
        )
        axis.set_xlabel("Days from peak")
        axis.grid(alpha=0.35)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Deterministic omitted-mean residual / σ")
    fig.suptitle(
        "Primary fixed-design residual check: the misspecified mean is conspicuous",
        y=0.89,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.68, wspace=0.12)
    _synthetic_banner(fig, y=0.975)
    _save_figure(fig, path)


def figure_coverage_oracle(results: StudyResults, path: Path) -> None:
    rows = _metric_with_scenario(results)
    fig, ax = plt.subplots(figsize=(6.3, 5.1))
    for model, color, marker, label in (
        ("offset_aware_blend", "#5bd6d0", "o", OFFSET_AWARE_LABEL),
        ("colocated_blend", "#f06c9b", "^", COLOCATED_BLEND_LABEL),
    ):
        selected = [row for row in rows if row["model"] == model]
        ax.scatter(
            [row["oracle_coverage_95"] for row in selected],
            [row["coverage_95"] for row in selected],
            s=24,
            alpha=0.72,
            color=color,
            marker=marker,
            label=label,
        )
    ax.plot([0, 1], [0, 1], color="#f3b562", lw=1.5, linestyle="--", label="Monte Carlo = oracle")
    ax.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel="Exact normal-theory coverage oracle",
        ylabel="Observed Monte Carlo coverage",
        title="Simulation agrees with the analytic coverage calculation",
    )
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8)
    _synthetic_banner(fig)
    _save_figure(fig, path)


def figure_bias_validation(results: StudyResults, path: Path) -> None:
    rows = [row for row in _metric_with_scenario(results) if row["model"] == "colocated_blend"]
    fig, ax = plt.subplots(figsize=(6.3, 5.1))
    colors = [row["blend_offset_mas"] for row in rows]
    scatter = ax.scatter(
        [row["analytic_bias_mas"] for row in rows],
        [row["bias_mas"] for row in rows],
        c=colors,
        cmap="viridis",
        s=34,
        alpha=0.8,
    )
    extent = max(abs(float(row["analytic_bias_mas"])) for row in rows) * 1.08 + 1e-6
    ax.plot([-extent, extent], [-extent, extent], color="#f3b562", linestyle="--")
    ax.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        xlabel="Analytic omitted-variable bias (mas)",
        ylabel="Monte Carlo mean bias (mas)",
        title="Co-located-blend misspecification has a calculable bias",
    )
    ax.grid(alpha=0.35)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Blend offset |b| (mas)")
    _synthetic_banner(fig)
    _save_figure(fig, path)


def figure_primary_coverage(results: StudyResults, path: Path) -> None:
    """Show the locally prespecified primary coverage estimand and paired gap."""

    rows = _metric_with_scenario(results)
    selected = [
        row
        for row in rows
        if row["epochs"] == 32
        and row["sigma_mas_per_coordinate"] == 0.25
        and row["blend_flux_ratio"] == 0.3
        and row["blend_offset_mas"] == 20.0
    ]
    by_model = {row["model"]: row for row in selected}
    offset_aware = by_model["offset_aware_blend"]
    co_located = by_model["colocated_blend"]
    paired = results.primary_paired_summary

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    values = np.array([offset_aware["coverage_95"], co_located["coverage_95"]])
    lowers = values - np.array(
        [offset_aware["coverage_wilson95_low"], co_located["coverage_wilson95_low"]]
    )
    uppers = (
        np.array([offset_aware["coverage_wilson95_high"], co_located["coverage_wilson95_high"]])
        - values
    )
    axes[0].bar(
        [0, 1],
        values,
        yerr=np.vstack((lowers, uppers)),
        color=["#5bd6d0", "#f06c9b"],
        capsize=5,
        width=0.62,
    )
    axes[0].axhline(0.95, color="#f3b562", linestyle="--", label="Nominal 95%")
    axes[0].set_xticks([0, 1], [OFFSET_AWARE_LABEL, "Co-located-\nblend"])
    axes[0].set(
        ylim=(0, 1.03),
        ylabel="Empirical 95% interval coverage",
        title="Primary-cell coverage\n(Wilson 95% intervals)",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.35)

    gap = paired["coverage_gap"]
    gap_low = paired["coverage_gap_mc95_low"]
    gap_high = paired["coverage_gap_mc95_high"]
    axes[1].errorbar(
        [0],
        [gap],
        yerr=[[gap - gap_low], [gap_high - gap]],
        fmt="o",
        markersize=9,
        color="#5bd6d0",
        ecolor="#eaf3f7",
        capsize=6,
    )
    analytic_gap = float(paired["analytic_coverage_gap_oracle"])
    axes[1].axhline(0.05, color="#f3b562", linestyle="--", label="Prespecified 5 pp pivot")
    axes[1].axhline(
        analytic_gap,
        color="#f06c9b",
        linestyle=":",
        label=f"Analytic oracle {100 * analytic_gap:.2f} pp",
    )
    axes[1].axhline(0.0, color="#7893ad", lw=0.8)
    axes[1].text(
        0.73,
        0.214,
        _primary_gap_annotation(paired),
        ha="right",
        va="top",
        fontsize=8.3,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#0d1a2b", "edgecolor": "#7893ad"},
    )
    axes[1].set_xticks([0], ["Offset-aware −\nco-located-blend"])
    axes[1].set(
        xlim=(-0.8, 0.8),
        ylim=(0.0, 0.22),
        ylabel="Paired coverage gap",
        title="Locally prespecified paired gap\n(Monte Carlo 95% interval)",
    )
    axes[1].legend(fontsize=7.5, loc="lower right")
    axes[1].grid(axis="y", alpha=0.35)
    fig.suptitle("Locally prespecified primary coverage comparison", y=0.88, fontsize=14)
    fig.subplots_adjust(top=0.70, wspace=0.20)
    _synthetic_banner(fig)
    _save_figure(fig, path)


def orientation_sensitivity_rows(config: StudyConfig) -> list[dict[str, Any]]:
    """Analytic exploratory sweep of the primary blend-offset orientation."""

    truth = config.truth
    times = make_observation_schedule(32, truth, config.schedule_baseline_years)
    flux_ratio = 0.3
    offset_mas = 20.0
    sigma_mas = 0.25
    co_located_design = build_design_matrix(
        times, truth, flux_ratio, offset_aware_blend_model=False
    )
    solver = prepare_solver(co_located_design, sigma_mas)
    u_vector = lens_trajectory(times, truth)
    _, blend_weight = blend_coefficients(np.linalg.norm(u_vector, axis=1), flux_ratio)
    theta_se = float(solver.standard_errors[0])
    rows = []
    for angle_deg in range(361):
        angle = np.deg2rad(angle_deg)
        blend_vector = offset_mas * np.array([np.cos(angle), np.sin(angle)])
        missed = blend_weight[:, None] * blend_vector[None, :]
        bias = float(solver.estimate(stack_coordinates(missed))[0])
        standardized_bias = bias / theta_se
        rows.append(
            {
                "orientation_angle_deg": angle_deg,
                "is_prespecified_orientation": angle_deg == int(round(truth.blend_angle_deg)),
                "analytic_omitted_bias_mas": bias,
                "analytic_standardized_bias": standardized_bias,
                "analytic_coverage_95": exact_normal_coverage(standardized_bias, config.interval_z),
                "primary_design_theta_e_standard_error_mas": theta_se,
                "analysis_role": "exploratory analytic sensitivity; no additional Monte Carlo",
            }
        )
    return rows


def figure_orientation_sensitivity(rows: list[dict[str, Any]], path: Path) -> None:
    angles = np.array([row["orientation_angle_deg"] for row in rows])
    bias = np.array([row["analytic_omitted_bias_mas"] for row in rows])
    coverage = np.array([row["analytic_coverage_95"] for row in rows])
    selected = next(row for row in rows if row["is_prespecified_orientation"])
    selected_angle = selected["orientation_angle_deg"]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.1), sharex=True)
    axes[0].plot(angles, bias, color="#5bd6d0", lw=2)
    axes[0].axhline(0, color="#7893ad", lw=0.8)
    axes[0].axvline(selected_angle, color="#f3b562", linestyle="--", label="Fixed 67° design")
    axes[0].set(
        ylabel="Analytic co-located-blend bias (mas)",
        title="Offset direction changes the omitted-variable projection",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.35)
    axes[1].plot(angles, coverage, color="#f06c9b", lw=2)
    axes[1].axhline(0.95, color="#7893ad", linestyle=":", label="Nominal 95%")
    axes[1].axvline(selected_angle, color="#f3b562", linestyle="--")
    axes[1].set(
        xlim=(0, 360),
        ylim=(-0.02, 1.02),
        xlabel="Source-to-blend orientation (degrees)",
        ylabel="Exact co-located-blend 95% coverage oracle",
        title="Coverage sensitivity is analytic, not an extra simulation headline",
    )
    axes[1].grid(alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "EXPLORATORY ORIENTATION SENSITIVITY · ANALYTIC · NO EXTRA MONTE CARLO",
        y=0.95,
        fontsize=11,
        color="#f3b562",
    )
    _synthetic_banner(fig)
    _save_figure(fig, path)


def mass_distance_rows(config: StudyConfig, results: StudyResults) -> list[dict[str, Any]]:
    source_distance = config.truth.source_distance_kpc
    distances = np.linspace(0.25, 0.98 * source_distance, 320)
    pi_rel = relative_parallax_mas(distances, source_distance)
    primary = next(
        row
        for row in _metric_with_scenario(results)
        if row["model"] == "offset_aware_blend"
        and row["epochs"] == 32
        and row["sigma_mas_per_coordinate"] == 0.25
        and row["blend_flux_ratio"] == 0.3
        and row["blend_offset_mas"] == 20.0
    )
    half_width = 0.5 * float(primary["interval_width_mas"])
    theta = config.truth.theta_e_mas
    low_theta = theta - half_width
    high_theta = theta + half_width
    return [
        {
            "lens_distance_kpc": float(distance),
            "relative_parallax_mas": float(parallax),
            "mass_at_true_theta_e_solar": float(mass_solar(theta, parallax)),
            "mass_at_truth_minus_primary_design_halfwidth_solar": float(
                mass_solar(low_theta, parallax)
            ),
            "mass_at_truth_plus_primary_design_halfwidth_solar": float(
                mass_solar(high_theta, parallax)
            ),
        }
        for distance, parallax in zip(distances, pi_rel, strict=True)
    ]


def figure_mass_distance(config: StudyConfig, rows: list[dict[str, Any]], path: Path) -> None:
    distance = np.array([row["lens_distance_kpc"] for row in rows])
    mass = np.array([row["mass_at_true_theta_e_solar"] for row in rows])
    low = np.array([row["mass_at_truth_minus_primary_design_halfwidth_solar"] for row in rows])
    high = np.array([row["mass_at_truth_plus_primary_design_halfwidth_solar"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    ax.fill_between(
        distance,
        low,
        high,
        color="#5bd6d0",
        alpha=0.22,
        label="Truth-centered primary-design 95% uncertainty band",
    )
    ax.plot(distance, mass, color="#5bd6d0", lw=2, label="Mass–distance ridge at true θE")
    ax.scatter(
        [config.truth.lens_distance_kpc],
        [config.truth.mass_solar],
        color="#f3b562",
        s=55,
        zorder=3,
        label="Simulation truth (known only to us)",
    )
    ax.set(
        xlim=(0.25, min(7.5, config.truth.source_distance_kpc * 0.98)),
        ylim=(0, 45),
        xlabel="Assumed lens distance (kpc)",
        ylabel="Conditional lens mass (solar masses)",
        title="Astrometric scale alone does not identify one mass",
    )
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8, loc="upper left")
    _synthetic_banner(fig)
    _save_figure(fig, path)


def write_results_markdown(path: Path, metrics: dict[str, Any]) -> None:
    offset_aware = metrics["primary_cell"]["offset_aware_blend"]
    colocated = metrics["primary_cell"]["colocated_blend"]
    worst = metrics["highlights"]["largest_co_located_false_confidence_cell"]
    paired = metrics["primary_paired_coverage_gap"]
    pivot = metrics["validation_checks"]["prespecified_interpretation_threshold"]
    offset_aware_strict = _strict_narrow_miss_cell(offset_aware)[2]
    colocated_strict = _strict_narrow_miss_cell(colocated)[2]
    worst_strict = None if worst is None else _strict_narrow_miss_cell(worst)[2]
    pivot_label = (
        "the locally prespecified undercoverage result"
        if pivot["classification"] == "prespecified_undercoverage_result"
        else "the locally prespecified negative-result pivot"
    )
    trials_per_cell = f"{metrics['configuration']['trials_per_scenario']:,}"
    execution_summary = (
        f"{metrics['design_accounting']['grid_cells']} rows and "
        f"{metrics['configuration']['synthetic_dataset_count']:,} synthetic datasets"
    )
    primary_truth = f"{offset_aware['theta_e_truth_mas']:.4f}"
    colocated_analytic_bias = f"{colocated['analytic_bias_mas']:.4f}"
    colocated_bias = f"{colocated['bias_mas']:.4f}"
    colocated_bias_mcse = f"{colocated['bias_mcse_mas']:.4f}"
    colocated_oracle_coverage = f"{100 * colocated['oracle_coverage_95']:.2f}%"
    offset_aware_se = f"{offset_aware['theta_e_standard_error_mas']:.4f}"
    colocated_se = f"{colocated['theta_e_standard_error_mas']:.4f}"
    colocated_se_reduction = (
        100
        * (
            float(offset_aware["theta_e_standard_error_mas"])
            - float(colocated["theta_e_standard_error_mas"])
        )
        / float(offset_aware["theta_e_standard_error_mas"])
    )
    formal_se_change = (
        f"from {offset_aware_se} mas to {colocated_se} mas, a "
        f"{colocated_se_reduction:.2f}% reduction"
    )
    paired_gap = f"{100 * paired['coverage_gap']:.2f}"
    paired_oracle = f"{100 * paired['analytic_coverage_gap_oracle']:.3f}"
    paired_gap_mcse = f"{100 * paired['coverage_gap_mcse']:.2f}"
    paired_gap_mc95 = (
        f"[{100 * paired['coverage_gap_mc95_low']:.2f}, "
        f"{100 * paired['coverage_gap_mc95_high']:.2f}]"
    )
    discordant_pairs = (
        f"{paired['discordant_offset_aware_only_count']} offset-aware-only and\n"
        f"{paired['discordant_co_located_only_count']} co-located-blend-only covering pairs"
    )
    macro = metrics["highlights"]["offset_aware_distinct_dgp_macro_coverage"]
    macro_coverage = f"{100 * macro['macro_coverage_95']:.2f}%"
    macro_mc95 = (
        f"[{100 * macro['macro_coverage_mc95_low']:.2f}%, "
        f"{100 * macro['macro_coverage_mc95_high']:.2f}%]"
    )
    co_located_expected_reduced_q = f"{colocated['residual_expected_reduced_q']:.3f}"
    co_located_lambda = f"{colocated['residual_noncentrality_lambda']:.3f}"
    co_located_residual_rms = f"{colocated['omitted_mean_residual_rms_mas']:.3f}"
    co_located_power = f"{100 * colocated['residual_rejection_power_at_5pct']:.2f}%"
    offset_aware_expected_reduced_q = f"{offset_aware['residual_expected_reduced_q']:.3f}"
    design_accounting = metrics["design_accounting"]
    design_accounting_sentence = (
        f"The grid contains {design_accounting['distinct_true_mean_vectors']} distinct "
        "true-mean vectors and\n"
        f"{design_accounting['distinct_mean_plus_sigma_data_generating_distributions']} "
        "distinct mean-plus-noise data-generating distributions."
    )
    if design_accounting["zero_flux_grid_rows"]:
        design_accounting_sentence += (
            f"\nIts {design_accounting['zero_flux_grid_rows']} `g = 0` rows include "
            f"{design_accounting['repeated_zero_flux_negative_control_cells']} deliberate "
            "negative-control repeats."
        )
    if worst is None:
        atlas_strict_summary = (
            "No co-located-blend atlas cell met the strict 10% half-width eligibility criterion."
        )
    else:
        atlas_strict_summary = (
            "The largest eligible co-located-blend strict narrow-miss rate is\n"
            f"{worst_strict} in `{worst['scenario_id']}`."
        )
    offset_aware_table_row = (
        f"| {OFFSET_AWARE_LABEL} | {offset_aware['bias_mas']:.4f} | "
        f"{offset_aware['rmse_mas']:.4f} | {100 * offset_aware['coverage_95']:.2f}% | "
        f"{offset_aware_strict} |"
    )
    colocated_table_row = (
        f"| {COLOCATED_BLEND_LABEL} | {colocated['bias_mas']:.4f} | "
        f"{colocated['rmse_mas']:.4f} | {100 * colocated['coverage_95']:.2f}% | "
        f"{colocated_strict} |"
    )
    text = f"""# Results

> **{SYNTHETIC_LABEL}**

## Evidence chronology

- **Before the saved run:** under Alison Thomas's portfolio brief, Codex selected the
  question and froze the grid and run contract locally. One primary cell, the paired
  coverage estimand, seed, and five-point interpretation rule were fixed before the workflow
  inspected the computed oracle. The exact oracle was then evaluated before Monte Carlo;
  its {paired_oracle}-point analytic gap remained available to the same AI-assisted
  workflow that chose the design.
- **Original stochastic evidence:** one seeded full-grid command generated all
  {execution_summary}. The primary
  coverage counts, paired outcomes, and coefficient estimates below come from that run.
- **Later deterministic audit:** on 2026-08-08, red-team review added fixed-design residual
  adequacy calculations from the declared means and designs. No new random draws were
  created, and protected primary and scenario results remained unchanged.
- **Current status:** the later audit changed the interpretation of the original result;
  Alison's human science, copy, contribution, and release review remains pending.

## The question and primary design

When a static unresolved blend is offset from a microlensed source, how much can a
co-located-blend model reduce 95% interval coverage for `theta_E` while leaving its formal
standard error nearly unchanged? Ghostlight compares that model with an offset-aware fit on
identical noise realizations.

The primary cell uses 32 epochs, 0.25 mas Gaussian noise per coordinate, blend ratio
`g = 0.3`, a 20 mas source-to-blend offset at 67°, and {trials_per_cell} paired realizations.
The injected Einstein radius is {primary_truth} mas. This is a
**locally prespecified primary validation**, not external registration or independent
confirmation. The other
{design_accounting["grid_cells"] - 1} rows were generated in the same stochastic run but
are interpreted exploratorily.

## Original seeded coverage evidence

| Fit | Bias (mas) | RMSE (mas) | 95% coverage | Strict narrow-miss rate (10% half-width) |
|---|---:|---:|---:|---:|
{offset_aware_table_row}
{colocated_table_row}

The formal standard error changed only {formal_se_change}. The co-located analytic
omitted-variable bias was {colocated_analytic_bias} mas;
the Monte Carlo bias was {colocated_bias} ± {colocated_bias_mcse} mas Monte Carlo SE. Its
exact normal-theory coverage oracle was {colocated_oracle_coverage}.

Offset-aware minus co-located-blend coverage is {paired_gap} percentage points (paired Monte
Carlo SE {paired_gap_mcse} points; Monte Carlo 95% interval {paired_gap_mc95} points). The
paired table contains {discordant_pairs}. No population-level p-value is attached to this
fixed synthetic contrast.

The locally prespecified five-point interpretation rule selected
**{pivot_label}**. The primary strict 10%-half-width diagnostic was ineligible for both
models; `N/A` records a fixed interval that was too wide for that threshold, not a measured
zero.

## Later deterministic model-adequacy audit

The later audit asked whether the co-located mean model would pass an ordinary residual
check. Deterministic projection of the already-declared primary design gave residual
`df = {colocated["residual_degrees_of_freedom"]}`, noncentrality `lambda = {co_located_lambda}`,
expected reduced `Q = {co_located_expected_reduced_q}`, omitted-mean residual RMS
{co_located_residual_rms} mas, and {co_located_power} analytic power for the ordinary 5%
chi-square lack-of-fit check. The offset-aware model had expected reduced
`Q = {offset_aware_expected_reduced_q}`.

No new random draws were used to obtain these quantities. They are analytic fixed-design
properties, not observed residual statistics. The frozen artifacts retained coefficient
estimates but not per-trial residual vectors, so observed residual `Q`, observed-data
p-values, and empirical rejection rates remain unavailable.

This later result changed the interpretation: formal coefficient covariance is conditional
on the assumed mean and **covariance is not a model check**. The seeded coverage contrast is
real for the declared design, but it occurs inside a grossly rejected mean model rather than
a model that would survive a complete residual analysis.

## Mechanism and verification

Direct flux weighting adds a time-dependent blend-offset term to the baseline-relative
apparent centroid. The co-located model omits that term, allowing part of it to project into
the `theta_E` coefficient. Both fits receive identical noise, so the primary contrast retains
the pairing.

The Monte Carlo bias and coverage are checked against exact linear-Gaussian omitted-variable
oracles. A later scalar-equation/QR path repeats the flux weighting, geometry, stacking,
design, covariance, bias, and residual calculation without calling the production physics,
stacking, design, or SVD-solver functions. It reduces shared-code risk but remains an
AI-authored internal self-check in the same specification and environment, not external
validation.

## Exploratory sensitivity from the same run

{design_accounting_sentence}

After repeated controls are first collapsed within their shared DGPs, unweighted
offset-aware macro coverage across {macro["distinct_dgp_count"]} DGPs is {macro_coverage}
(oracle-MCSE normal 95% interval {macro_mc95}). This is a
descriptive fixed-grid average, not a population estimand or heterogeneous binomial sample.

{atlas_strict_summary}

It is an exploratory conditional stress cell, not the primary headline or an estimate of
real-world failure frequency. Later grid-wide adequacy quantities are reported beside these
coverage summaries; they are not retroactively part of the original primary outcome.

## Claim boundary and explanatory mass ridge

No real image, light curve, or astrometric catalog is analyzed. The synthetic truth is on
the historical scale of OB110462, but Ghostlight is not an OB110462 reproduction and its
round 6-solar-mass truth is not the current best mass. The bright-neighbor PSF-extraction
bias discussed in the observational literature is distinct from the unresolved
flux-weighted-centroid mechanism isolated here.

The conditional astrometric fit estimates `theta_E` while holding the photometric trajectory
and blend ratio fixed. Without an independently measured relative parallax,
`theta_E^2 = kappa M pi_rel` defines a mass–distance ridge rather than one fitted mass. The
ridge figure is an explanatory nonclaim: its gold point is known only to the simulator, and
its shading is a truth-centered design illustration rather than a realized confidence
interval or posterior.
"""
    path.write_text(text)


def write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifacts = []
    for file in files:
        relative = file.relative_to(output_dir)
        logical_path = (Path("outputs") / relative).as_posix()
        artifacts.append(
            {
                "path": logical_path,
                "bytes": file.stat().st_size,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "hash_algorithm": "sha256",
        "artifacts": artifacts,
    }
    _json_dump(output_dir / "artifact_manifest.json", manifest)
    return manifest


def verify_artifacts(project_root: Path) -> list[str]:
    manifest_path = project_root / "outputs" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verified = []
    root = project_root.resolve()
    listed_paths = [item["path"] for item in manifest["artifacts"]]
    if len(listed_paths) != len(set(listed_paths)):
        raise ValueError("artifact manifest contains duplicate paths")
    for item in manifest["artifacts"]:
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "outputs"
        ):
            raise ValueError(f"unsafe artifact path: {item['path']}")
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ValueError(f"artifact escapes project root: {item['path']}")

    output_dir = root / "outputs"
    actual_paths = {
        (Path("outputs") / path.relative_to(output_dir)).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    listed_set = set(listed_paths)
    if listed_set != actual_paths:
        missing = sorted(actual_paths - listed_set)
        nonexistent = sorted(listed_set - actual_paths)
        raise ValueError(
            "artifact manifest is incomplete or lists nonexistent files: "
            f"unlisted={missing}, nonexistent={nonexistent}"
        )

    for item in manifest["artifacts"]:
        target = (root / Path(item["path"])).resolve()
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise ValueError(f"artifact verification failed: {item['path']}")
        verified.append(item["path"])
    return verified


def write_all_outputs(
    output_dir: Path,
    config: StudyConfig,
    results: StudyResults,
    *,
    enforce_prespecified_validation_checks: bool = False,
) -> dict[str, Any]:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics_rows = _metric_with_scenario(results)
    metrics = summarize_metrics(config, results)
    if (
        enforce_prespecified_validation_checks
        and not metrics["validation_checks"]["all_validation_checks_pass"]
    ):
        raise RuntimeError(
            f"locally prespecified validation checks failed: {metrics['validation_checks']}"
        )
    mass_rows = mass_distance_rows(config, results)
    orientation_rows = orientation_sensitivity_rows(config)
    residual_rows = primary_mean_residual_rows(config)

    _json_dump(output_dir / "metrics.json", metrics)
    _write_csv(tables / "scenario_metrics.csv", metrics_rows)
    _write_csv(tables / "scenario_inputs.csv", results.scenario_rows)
    _write_csv(tables / "primary_cell_trials.csv", results.focus_trials)
    _write_csv(tables / "mass_distance_ridge.csv", mass_rows)
    _write_csv(tables / "orientation_sensitivity.csv", orientation_rows)
    _write_csv(tables / "primary_mean_residuals.csv", residual_rows)
    write_results_markdown(output_dir / "RESULTS.md", metrics)

    _style()
    figure_event_geometry(config, figures / "event_geometry.png")
    figure_calibration_sensitivity_atlas(results, figures / "calibration_sensitivity_atlas.png")
    figure_coverage_oracle(results, figures / "coverage_oracle.png")
    figure_bias_validation(results, figures / "omitted_bias_validation.png")
    figure_primary_coverage(results, figures / "primary_coverage_gap.png")
    figure_orientation_sensitivity(orientation_rows, figures / "orientation_sensitivity.png")
    figure_mass_distance(config, mass_rows, figures / "mass_distance_degeneracy.png")
    figure_primary_mean_residuals(residual_rows, figures / "primary_mean_residuals.png")
    write_artifact_manifest(output_dir)
    return metrics
