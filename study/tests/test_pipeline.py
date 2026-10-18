import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ghostlight.config import StudyConfig, validate_project_root
from ghostlight.pipeline import (
    _canonical_digest,
    _frozen_evidence_projection_hash,
    _reconcile_staged_outputs,
    run_pipeline,
)
from ghostlight.reporting import verify_artifacts


def _small_config() -> StudyConfig:
    return replace(
        StudyConfig(),
        trials_per_scenario=32,
        bootstrap_resamples=4,
        cadences=(12, 32),
        astrometric_sigmas_mas=(0.25,),
        blend_flux_ratios=(0.3,),
        blend_offsets_mas=(20.0,),
    )


def _project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "ghostlight-microlensing-lab"\nversion = "0.0.0"\n'
    )
    return tmp_path


def test_pipeline_is_deterministic_and_emits_required_artifacts(tmp_path):
    root = _project(tmp_path)
    run_pipeline(root, _small_config(), prespecified_validation=False)
    first_manifest = (root / "outputs" / "artifact_manifest.json").read_bytes()
    first_metrics = (root / "outputs" / "metrics.json").read_bytes()
    run_pipeline(root, _small_config(), prespecified_validation=False)
    assert (root / "outputs" / "artifact_manifest.json").read_bytes() == first_manifest
    assert (root / "outputs" / "metrics.json").read_bytes() == first_metrics
    assert len(list((root / "outputs" / "figures").glob("*.png"))) == 8
    assert len(list((root / "outputs" / "tables").glob("*.csv"))) == 6
    assert (root / "outputs" / "tables" / "scenario_metrics.csv").is_file()
    with (root / "outputs" / "tables" / "scenario_metrics.csv").open(newline="") as handle:
        scenario_rows = list(csv.DictReader(handle))
    assert {row["model"] for row in scenario_rows} == {
        "offset_aware_blend",
        "colocated_blend",
    }
    primary_rows = [row for row in scenario_rows if row["is_prespecified_primary_cell"] == "True"]
    nonprimary_rows = [
        row for row in scenario_rows if row["is_prespecified_primary_cell"] == "False"
    ]
    assert len(primary_rows) == 2
    assert len(nonprimary_rows) == 2
    for field in (
        "paired_coverage_gap",
        "paired_coverage_gap_mcse",
        "paired_coverage_gap_mc95_low",
        "paired_coverage_gap_mc95_high",
        "paired_analytic_coverage_gap_oracle",
        "paired_uncertainty_scope",
    ):
        assert primary_rows[0][field]
        assert primary_rows[0][field] == primary_rows[1][field]
        assert all(not row[field] for row in nonprimary_rows)
    with (root / "outputs" / "tables" / "orientation_sensitivity.csv").open(newline="") as handle:
        orientation_rows = list(csv.DictReader(handle))
    assert len(orientation_rows) == 361
    highlighted = [row for row in orientation_rows if row["is_prespecified_orientation"] == "True"]
    assert len(highlighted) == 1
    assert highlighted[0]["orientation_angle_deg"] == "67"
    assert "no additional Monte Carlo" in highlighted[0]["analysis_role"]
    expected_artifacts = {
        "outputs/RESULTS.md",
        "outputs/metrics.json",
        "outputs/figures/event_geometry.png",
        "outputs/figures/calibration_sensitivity_atlas.png",
        "outputs/figures/coverage_oracle.png",
        "outputs/figures/omitted_bias_validation.png",
        "outputs/figures/primary_coverage_gap.png",
        "outputs/figures/orientation_sensitivity.png",
        "outputs/figures/mass_distance_degeneracy.png",
        "outputs/figures/primary_mean_residuals.png",
        "outputs/tables/scenario_metrics.csv",
        "outputs/tables/scenario_inputs.csv",
        "outputs/tables/primary_cell_trials.csv",
        "outputs/tables/mass_distance_ridge.csv",
        "outputs/tables/orientation_sensitivity.csv",
        "outputs/tables/primary_mean_residuals.csv",
    }
    assert set(verify_artifacts(root)) == expected_artifacts


def test_output_reconciliation_is_noop_for_equal_files_and_removes_stale_files(tmp_path):
    root = _project(tmp_path)
    output = root / "outputs"
    staging = root / ".ghostlight-reporting-fixture"
    (output / "tables").mkdir(parents=True)
    (staging / "tables").mkdir(parents=True)
    unchanged = output / "tables" / "unchanged.csv"
    unchanged.write_text("a,b\n1,2\n")
    (staging / "tables" / "unchanged.csv").write_bytes(unchanged.read_bytes())
    changed = output / "changed.txt"
    changed.write_text("old\n")
    (staging / "changed.txt").write_text("new\n")
    (output / "stale 2.txt").write_text("stale conflict copy\n")
    before = (unchanged.stat().st_ino, unchanged.stat().st_mtime_ns)

    _reconcile_staged_outputs(staging, output, root)

    assert (unchanged.stat().st_ino, unchanged.stat().st_mtime_ns) == before
    assert changed.read_text() == "new\n"
    assert not (output / "stale 2.txt").exists()
    assert not staging.exists()


def test_manifest_rejects_path_traversal(tmp_path):
    root = _project(tmp_path)
    run_pipeline(root, _small_config(), prespecified_validation=False)
    manifest_path = root / "outputs" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original_manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["path"] = "outputs/../escape.txt"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe artifact path"):
        verify_artifacts(root)

    duplicate_manifest = json.loads(json.dumps(original_manifest))
    duplicate_manifest["artifacts"].append(duplicate_manifest["artifacts"][0])
    manifest_path.write_text(json.dumps(duplicate_manifest))
    with pytest.raises(ValueError, match="duplicate paths"):
        verify_artifacts(root)

    manifest_path.write_text(json.dumps(original_manifest))
    (root / "outputs" / "unlisted.txt").write_text("not in manifest")
    with pytest.raises(ValueError, match="incomplete"):
        verify_artifacts(root)


def test_project_root_validation_rejects_unrelated_directory(tmp_path):
    with pytest.raises(ValueError):
        validate_project_root(tmp_path)


def test_exploratory_pipeline_requires_report_anchor_before_rng(tmp_path, monkeypatch):
    root = _project(tmp_path)
    config = replace(_small_config(), cadences=(12,))

    def forbidden_run(*args, **kwargs):
        raise AssertionError("run_study must not be reached without the report anchor")

    monkeypatch.setattr("ghostlight.pipeline.run_study", forbidden_run)
    with pytest.raises(ValueError, match="reporting requires the primary anchor.*random draw"):
        run_pipeline(root, config, prespecified_validation=False)
    assert not (root / "outputs").exists()


def test_canonical_metric_digest_treats_json_tuple_roundtrip_as_equal():
    payload = {"grid": (12, 32, 72), "nested": {"thresholds": (0.1, 0.25, 0.5)}}
    roundtripped = json.loads(json.dumps(payload))
    assert _canonical_digest(payload) == _canonical_digest(roundtripped)


def test_frozen_evidence_projection_hash_covers_every_old_metric_and_paired_field():
    rows = [{"scenario_id": "a", "bias_mas": 0.1, "coverage_count": 19}]
    columns = ("scenario_id", "bias_mas", "coverage_count")
    paired = {"coverage_gap": 0.2, "discordant_offset_aware_only_count": 4}
    baseline = _frozen_evidence_projection_hash(rows, columns, paired)
    assert baseline != _frozen_evidence_projection_hash(
        [{**rows[0], "coverage_count": 18}], columns, paired
    )
    assert baseline != _frozen_evidence_projection_hash(
        rows, columns, {**paired, "coverage_gap": 0.1}
    )


def test_generated_claim_boundaries_are_explicit(tmp_path):
    root = _project(tmp_path)
    metrics = run_pipeline(root, _small_config(), prespecified_validation=False)
    results_text = (root / "outputs" / "RESULTS.md").read_text()
    assert "SYNTHETIC KNOWN-TRUTH STUDY" in results_text
    assert "not an OB110462 reproduction" in results_text
    assert "No real image, light curve, or astrometric catalog is analyzed" in results_text
    assert "No real event is fit" in metrics["claim_boundary"]
    assert "validation_checks" in metrics
    assert "prespecified_interpretation_threshold" in metrics["validation_checks"]
    assert metrics["schema_version"] == "3.0"
    assert "primary_model_adequacy" in metrics
    serialized_metrics = json.dumps(metrics).lower()
    assert "preregister" not in serialized_metrics
    assert "honest_" not in serialized_metrics
    assert "naive_" not in serialized_metrics
    assert metrics["fixed_design_preflight"]["status"] == "passed"
    assert metrics["highlights"]["design_rank_failure_rate"] == 0
