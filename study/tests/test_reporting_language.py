import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

import ghostlight.reporting as reporting
from ghostlight.config import StudyConfig
from ghostlight.simulation import run_study


@pytest.fixture
def primary_case():
    config = replace(
        StudyConfig(),
        trials_per_scenario=32,
        bootstrap_resamples=4,
        cadences=(32,),
        astrometric_sigmas_mas=(0.25,),
        blend_flux_ratios=(0.3,),
        blend_offsets_mas=(20.0,),
    )
    return config, run_study(config)


def test_strict_narrow_miss_marks_ineligible_cells_as_na():
    value, eligible, label = reporting._strict_narrow_miss_cell(
        {
            "false_precision_10pct_halfwidth_eligible": False,
            "false_precision_10pct_halfwidth_rate": 0.0,
        }
    )
    assert math.isnan(value)
    assert eligible is False
    assert label == "N/A — interval too wide for threshold"

    value, eligible, label = reporting._strict_narrow_miss_cell(
        {
            "false_precision_10pct_halfwidth_eligible": "True",
            "false_precision_10pct_halfwidth_rate": 0.05,
        }
    )
    assert value == pytest.approx(0.05)
    assert eligible is True
    assert label == "5.00%"


def test_atlas_masks_and_hatches_ineligible_cells(monkeypatch, primary_case, tmp_path):
    _, results = primary_case
    captured = []
    monkeypatch.setattr(reporting, "_save_figure", lambda figure, path: captured.append(figure))

    reporting.figure_calibration_sensitivity_atlas(results, tmp_path / "atlas.png")

    figure = captured.pop()
    try:
        atlas_axis = figure.axes[0]
        assert np.ma.getmaskarray(atlas_axis.images[0].get_array()).all()
        assert any(patch.get_hatch() == "////" for patch in atlas_axis.patches)
        visible_text = "\n".join(text.get_text() for text in figure.findobj(match=Text))
        assert "N/A" in visible_text
        assert "Exploratory strict narrow-miss rate" in visible_text
        assert "eligible cells only" in visible_text
    finally:
        plt.close(figure)


def test_primary_plot_uses_prespecified_wording_and_oracle(monkeypatch, primary_case, tmp_path):
    _, results = primary_case
    captured = []
    monkeypatch.setattr(reporting, "_save_figure", lambda figure, path: captured.append(figure))

    reporting.figure_primary_coverage(results, tmp_path / "primary.png")

    figure = captured.pop()
    try:
        visible_text = "\n".join(text.get_text() for text in figure.findobj(match=Text))
        assert "Locally prespecified primary coverage comparison" in visible_text
        assert "Prespecified 5 pp pivot" in visible_text
        assert "Analytic oracle" in visible_text
        assert "Preregistered" not in visible_text
        assert figure.axes[1].get_ylim() == pytest.approx((0.0, 0.22))
    finally:
        plt.close(figure)


def test_primary_annotation_exposes_point_interval_and_analytic_oracle():
    assert reporting._primary_gap_annotation(
        {
            "coverage_gap": 0.151,
            "coverage_gap_mc95_low": 0.1328,
            "coverage_gap_mc95_high": 0.1692,
            "analytic_coverage_gap_oracle": 0.1468,
        }
    ) == ("15.10 pp\nMC95 [13.28, 16.92] pp\nAnalytic oracle 14.68 pp")


def test_duplicate_controls_are_aggregated_before_dgp_macro_average():
    rows = [
        {
            "model": "offset_aware_blend",
            "epochs": 32,
            "sigma_mas_per_coordinate": 0.25,
            "blend_flux_ratio": 0.0,
            "blend_offset_mas": offset,
            "trials": 100,
            "coverage_count": covered,
            "oracle_coverage_95": 0.95,
        }
        for offset, covered in ((0.0, 90), (5.0, 100), (20.0, 80))
    ]
    rows.append(
        {
            "model": "offset_aware_blend",
            "epochs": 32,
            "sigma_mas_per_coordinate": 0.25,
            "blend_flux_ratio": 0.3,
            "blend_offset_mas": 20.0,
            "trials": 100,
            "coverage_count": 50,
            "oracle_coverage_95": 0.5,
        }
    )
    summary = reporting._distinct_dgp_macro_coverage(rows, "offset_aware_blend", 1.96)
    assert summary["grid_row_count"] == 4
    assert summary["distinct_dgp_count"] == 2
    assert summary["repeated_zero_flux_control_rows_collapsed"] == 2
    assert summary["macro_coverage_95"] == pytest.approx(0.7)
    assert "no heterogeneous pooled-binomial Wilson" in summary["definition"]


def test_primary_mean_residual_support_has_both_coordinates_and_models():
    rows = reporting.primary_mean_residual_rows(StudyConfig())
    assert len(rows) == 64
    assert {row["model"] for row in rows} == {"offset_aware_blend", "colocated_blend"}
    colocated = [row for row in rows if row["model"] == "colocated_blend"]
    aware = [row for row in rows if row["model"] == "offset_aware_blend"]
    assert max(abs(row["mean_residual_x_sigma"]) for row in colocated) > 4
    assert max(abs(row["mean_residual_y_sigma"]) for row in colocated) > 10
    squared_whitened_norm = sum(
        row["mean_residual_x_sigma"] ** 2 + row["mean_residual_y_sigma"] ** 2 for row in colocated
    )
    assert squared_whitened_norm == pytest.approx(988.036250199551)
    assert max(abs(row["mean_residual_x_sigma"]) for row in aware) < 1e-10
    assert max(abs(row["mean_residual_y_sigma"]) for row in aware) < 1e-10


def test_generated_results_uses_public_labels_and_no_false_zero(primary_case, tmp_path):
    config, results = primary_case
    metrics = reporting.summarize_metrics(config, results)
    target = tmp_path / "RESULTS.md"
    reporting.write_results_markdown(target, metrics)
    text = target.read_text()

    assert "Strict narrow-miss rate (10% half-width)" in text
    assert "N/A — interval too wide for threshold" in text
    assert "Offset-aware" in text
    assert "Co-located-blend" in text
    assert "standard error nearly unchanged" in text
    assert "precise but wrong" not in text
    assert "locally prespecified primary validation" in text
    assert "covariance is not a model check" in text
    assert "grossly rejected mean model" in text
    assert "locally prespecified five-point interpretation rule" in text
    assert "## Evidence chronology" in text
    assert "## Original seeded coverage evidence" in text
    assert "## Later deterministic model-adequacy audit" in text
    assert "No new random draws were used" in text
    assert "Alison's human science, copy, contribution, and release review remains pending" in text
    assert "14.680-point analytic gap" in text
    assert text.index("## Original seeded coverage evidence") < text.index(
        "## Later deterministic model-adequacy audit"
    )
    assert text.index("## Later deterministic model-adequacy audit") < text.index(
        "## Exploratory sensitivity from the same run"
    )
    assert "public-thesis" not in text
    assert "No co-located-blend atlas cell met the strict 10%" in text
    assert "prespecified_undercoverage_result" not in text
    assert "preregistered_undercoverage_thesis" not in text
    assert "## Preregistered" not in text


def test_methodology_design_accounting_is_exact():
    methodology = (Path(__file__).parents[1] / "METHODOLOGY.md").read_text()
    assert "81` grid cells" in methodology
    assert "21 distinct true-mean vectors" in methodology
    assert "63 distinct mean-plus-sigma" in methodology
    assert "3 × 7 = 21" in methodology
    assert "21 × 3 = 63" in methodology
    assert "9 × 2 = 18" in methodology
    assert "deliberate negative-control repeats" in methodology
    assert "`A -> 1` alone is not sufficient" in methodology


def test_supporting_copy_does_not_overclaim_independence_or_registration():
    root = Path(__file__).parents[1]
    readme = (root / "PROJECT_NOTES.md").read_text()
    linkedin_post = (root / "LINKEDIN_POST.md").read_text()
    assistance = (root / "AI_ASSISTANCE.md").read_text()
    review_status = (root / "REVIEW_STATUS.md").read_text()
    normalized_assistance = " ".join(assistance.split())
    assert "independent checks" not in readme
    assert "separate checks with different failure modes" in readme
    assert "81 executed parameter-grid rows" in readme
    assert "Those 81 rows contain 21 distinct true-mean vectors" in readme
    assert "81 observing conditions" not in readme
    assert "27 `g = 0` conditions" in readme
    assert "18 are deliberate negative-control repeats" in readme
    assert "public-thesis" not in readme
    assert "new preregistered analysis plan" not in assistance
    assert "auditable freeze point" in assistance
    assert "helped scaffold" not in normalized_assistance
    assert (
        "Alison set the portfolio goal, same-session constraint, scientific constraints"
        in normalized_assistance
    )
    assert "Codex selected Ghostlight and generated the" in normalized_assistance
    assert (
        "scientific implementation, tests, analysis, documentation, and draft public copy"
        in normalized_assistance
    )
    assert "Codex chose and froze the proposed locally prespecified grid" in normalized_assistance
    assert "not an independent technical choice by Alison" in normalized_assistance
    assert "human science, copy, contribution, and release sign-off is still pending" in (
        normalized_assistance
    )
    assert "If this candidate is released under Alison's name" in normalized_assistance
    assert "6. **2026-08-08 AI-authenticity implementation:**" in assistance
    assert "synchronized the public mirror without new draws" in normalized_assistance
    assert "was also Codex-executed" in normalized_assistance
    assert "it did not perform or document Alison's human science" in normalized_assistance
    assert "No independent peer review has been performed" in normalized_assistance
    assert "self-checks, not independent peer review" in normalized_assistance
    for public_copy in (readme, linkedin_post):
        normalized_copy = " ".join(public_copy.split())
        assert "audited with Codex" not in normalized_copy
        assert "Codex" in normalized_copy
        assert "Alison Thomas's portfolio brief" in normalized_copy
        assert "Automated and analytic checks passed" in normalized_copy
        assert "human science, copy, contribution, and release review is pending" in (
            normalized_copy
        )
        assert "no independent peer review has been performed" in normalized_copy
        assert "I directed" not in normalized_copy
        assert "I placed" not in normalized_copy
        assert "I required" not in normalized_copy
    assert "Current state: release candidate" in review_status
    assert "AI-generated research-software and red-team case study" in review_status
    assert "automated and analytic checks passed" in review_status
    assert "human science, copy, contribution, and release adjudication is" in review_status
    assert "present AI remediation is also not Alison's review" in review_status
    assert "no independent peer review" in review_status
    assert "public indexing and active promotion are withheld" in review_status


def test_unadjudicated_authorship_metadata_is_not_public_release_metadata():
    root = Path(__file__).parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    citation = (root / "CITATION.cff").read_text()
    assert "authors =" not in pyproject
    assert "AI-generated known-truth" in pyproject
    assert "Internal pre-release metadata only; do not cite or redistribute" in citation
    assert "not as an adjudicated sole author" in citation
