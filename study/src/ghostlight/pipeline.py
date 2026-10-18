"""End-to-end deterministic Ghostlight pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from ghostlight.config import StudyConfig, validate_project_root
from ghostlight.reporting import summarize_metrics, verify_artifacts, write_all_outputs
from ghostlight.simulation import (
    StudyResults,
    deterministic_model_fields,
    preflight_designs,
    run_study,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_regular_files(directory: Path) -> set[Path]:
    files: set[Path] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"refusing generated-output symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(directory))
    return files


def _reconcile_staged_outputs(staging: Path, output: Path, root: Path) -> None:
    """Install a staged output tree without tombstoning an existing cloud-backed root.

    Replacing the complete ``outputs`` directory caused macOS File Provider to rehydrate
    stale numbered conflict copies. Unchanged files now retain their inode and modification
    time; changed regular files are overwritten in place from the fully rendered staging
    tree, and unlisted files are removed before artifact verification.
    """

    if output.parent != root or output.name != "outputs":
        raise RuntimeError("refusing to replace an unsafe output path")
    if staging.parent != root or not staging.name.startswith(
        (".ghostlight-build-", ".ghostlight-reporting-")
    ):
        raise RuntimeError("refusing to install from an unsafe staging path")
    if not output.exists():
        staging.rename(output)
        return
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError("refusing to reconcile a non-directory output path")

    staged_files = _relative_regular_files(staging)
    output_files = _relative_regular_files(output)
    for relative in sorted(staged_files):
        source = staging / relative
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(f"refusing to overwrite non-file output: {target}")
            if source.stat().st_size == target.stat().st_size and _file_sha256(
                source
            ) == _file_sha256(target):
                continue
            with source.open("rb") as source_handle, target.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            shutil.copystat(source, target)
        else:
            shutil.copy2(source, target)

    for relative in sorted(output_files - staged_files, reverse=True):
        stale = output / relative
        if stale.parent == output or output in stale.parents:
            stale.unlink()
        else:  # pragma: no cover - defensive path containment guard
            raise RuntimeError(f"refusing to remove output outside root: {stale}")
    for directory in sorted(
        (path for path in output.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        with suppress(OSError):
            directory.rmdir()
    shutil.rmtree(staging)


def run_pipeline(
    project_root: Path,
    config: StudyConfig | None = None,
    *,
    prespecified_validation: bool = True,
) -> dict[str, Any]:
    root = validate_project_root(project_root)
    selected = config or StudyConfig()
    selected.validate(prespecified_validation=prespecified_validation)
    selected.validate_reporting_contract()

    staging = Path(tempfile.mkdtemp(prefix=".ghostlight-build-", dir=root))
    try:
        results = run_study(selected)
        metrics = write_all_outputs(
            staging,
            selected,
            results,
            enforce_prespecified_validation_checks=prespecified_validation,
        )
        output = root / "outputs"
        _reconcile_staged_outputs(staging, output, root)
        verify_artifacts(root)
        return metrics
    except Exception:
        if (
            staging.exists()
            and staging.parent == root
            and staging.name.startswith(".ghostlight-build-")
        ):
            shutil.rmtree(staging)
        raise


def _canonical_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_pipeline(project_root: Path, *, recompute: bool = False) -> dict[str, Any]:
    root = validate_project_root(project_root)
    verified = verify_artifacts(root)
    metrics = json.loads((root / "outputs" / "metrics.json").read_text())
    if metrics.get("schema_version") != "3.0":
        raise ValueError("saved metrics do not use the release schema")
    serialized_metrics = json.dumps(metrics).lower()
    if "preregister" in serialized_metrics:
        raise ValueError("saved metrics contain obsolete preregistration terminology")
    obsolete_model_terms = (
        "honest_blend_offset",
        "naive_colocated_blend",
        '"honest_',
        '"naive_',
        "honest minus naive",
        "honest coverage",
    )
    if any(term in serialized_metrics for term in obsolete_model_terms):
        raise ValueError("saved metrics contain obsolete value-laden model identifiers")
    frozen = StudyConfig()
    frozen.validate(prespecified_validation=True)
    expected = metrics["configuration"]["synthetic_dataset_count"]
    if expected != 162_000:
        raise ValueError(f"unexpected locally prespecified dataset count: {expected}")
    if (
        metrics["configuration"]["specification_fingerprint_sha256"]
        != frozen.specification_fingerprint
    ):
        raise ValueError("saved metrics do not match the frozen locally prespecified specification")
    if metrics["study_label"].find("SYNTHETIC") < 0:
        raise ValueError("synthetic claim boundary is missing")
    if not metrics["validation_checks"]["all_validation_checks_pass"]:
        raise ValueError("saved metrics did not pass the locally prespecified validation checks")

    recomputed_digest = None
    if recompute:
        recomputed = summarize_metrics(frozen, run_study(frozen))
        saved_digest = _canonical_digest(metrics)
        recomputed_digest = _canonical_digest(recomputed)
        if recomputed_digest != saved_digest:
            raise ValueError(
                "deterministic metric recomputation disagrees with metrics.json: "
                f"saved={saved_digest}, recomputed={recomputed_digest}"
            )
    return {
        "verified_artifact_count": len(verified),
        "synthetic_dataset_count": expected,
        "metrics_recomputed": recompute,
        "metrics_sha256": recomputed_digest or _canonical_digest(metrics),
    }


def _parse_csv_value(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [
            {key: _parse_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _frozen_evidence_projection_hash(
    metric_rows: list[dict[str, Any]],
    metric_columns: tuple[str, ...],
    primary_paired_summary: dict[str, Any],
) -> str:
    """Hash every pre-refresh metric value plus the paired primary summary."""

    projection = {
        "scenario_metrics_columns": metric_columns,
        "scenario_metrics_rows": [
            {column: row[column] for column in metric_columns} for row in metric_rows
        ],
        "primary_paired_coverage_gap": primary_paired_summary,
    }
    return _canonical_digest(projection)


def refresh_reporting_artifacts(project_root: Path) -> dict[str, Any]:
    """Enrich verified frozen tables with analytic diagnostics, without random draws.

    The existing artifact manifest is verified first. Saved coefficient summaries and
    primary trial rows are read as inputs; only fixed-design equations and deterministic
    report rendering are recomputed. This function never calls :func:`run_study`.
    """

    root = validate_project_root(project_root)
    verified_before = verify_artifacts(root)
    output = root / "outputs"
    tables = output / "tables"
    saved_metrics = json.loads((output / "metrics.json").read_text())
    if saved_metrics.get("schema_version") not in {"2.0", "3.0"}:
        raise ValueError("refresh requires a verified Ghostlight schema 2.0 or 3.0 bundle")

    config = StudyConfig()
    config.validate(prespecified_validation=True)
    if (
        saved_metrics["configuration"]["specification_fingerprint_sha256"]
        != config.specification_fingerprint
    ):
        raise ValueError("saved bundle does not match the frozen locally prespecified design")

    scenario_rows = _read_csv_rows(tables / "scenario_inputs.csv")
    combined_metric_rows = _read_csv_rows(tables / "scenario_metrics.csv")
    focus_trials = _read_csv_rows(tables / "primary_cell_trials.csv")
    source_metric_columns = tuple(combined_metric_rows[0])
    frozen_evidence_hash_before = _frozen_evidence_projection_hash(
        combined_metric_rows,
        source_metric_columns,
        saved_metrics["primary_paired_coverage_gap"],
    )
    scenario_columns = set(scenario_rows[0]) - {"scenario_id"}
    merged_only_columns = {
        "is_prespecified_primary_cell",
        "paired_coverage_gap",
        "paired_coverage_gap_mcse",
        "paired_coverage_gap_mc95_low",
        "paired_coverage_gap_mc95_high",
        "paired_analytic_coverage_gap_oracle",
        "paired_uncertainty_scope",
    }
    metric_rows = []
    for combined in combined_metric_rows:
        metric = {
            key: value
            for key, value in combined.items()
            if key not in scenario_columns and key not in merged_only_columns
        }
        fixed = deterministic_model_fields(
            config,
            int(combined["epochs"]),
            float(combined["sigma_mas_per_coordinate"]),
            float(combined["blend_flux_ratio"]),
            float(combined["blend_offset_mas"]),
            str(combined["model"]),
        )
        if not abs(float(metric["analytic_bias_mas"]) - float(fixed["analytic_bias_mas"])) <= 1e-10:
            raise ValueError(
                f"saved analytic bias changed for {combined['scenario_id']} {combined['model']}"
            )
        metric.update({key: value for key, value in fixed.items() if key != "analytic_bias_mas"})
        metric_rows.append(metric)

    results = StudyResults(
        metrics_rows=metric_rows,
        scenario_rows=scenario_rows,
        focus_trials=focus_trials,
        primary_paired_summary=saved_metrics["primary_paired_coverage_gap"],
        preflight_summary=preflight_designs(config),
    )
    staging = Path(tempfile.mkdtemp(prefix=".ghostlight-reporting-", dir=root))
    try:
        metrics = write_all_outputs(
            staging,
            config,
            results,
            enforce_prespecified_validation_checks=True,
        )
        refreshed_combined_rows = _read_csv_rows(staging / "tables/scenario_metrics.csv")
        frozen_evidence_hash_after = _frozen_evidence_projection_hash(
            refreshed_combined_rows,
            source_metric_columns,
            metrics["primary_paired_coverage_gap"],
        )
        if frozen_evidence_hash_after != frozen_evidence_hash_before:
            raise ValueError(
                "refresh would alter a pre-existing scenario metric or primary paired summary: "
                f"before={frozen_evidence_hash_before}, after={frozen_evidence_hash_after}"
            )
        for relative in (
            Path("tables/scenario_inputs.csv"),
            Path("tables/primary_cell_trials.csv"),
        ):
            if (staging / relative).read_bytes() != (output / relative).read_bytes():
                raise ValueError(f"refresh would alter frozen stochastic evidence: {relative}")
        _reconcile_staged_outputs(staging, output, root)
    except Exception:
        if (
            staging.exists()
            and staging.parent == root
            and staging.name.startswith(".ghostlight-reporting-")
        ):
            shutil.rmtree(staging)
        raise
    verified_after = verify_artifacts(root)
    return {
        "source_artifact_count_verified": len(verified_before),
        "refreshed_artifact_count_verified": len(verified_after),
        "synthetic_dataset_count": metrics["configuration"]["synthetic_dataset_count"],
        "random_draws_created": False,
        "frozen_evidence_projection_sha256_before": frozen_evidence_hash_before,
        "frozen_evidence_projection_sha256_after": frozen_evidence_hash_after,
        "metrics_sha256": _canonical_digest(metrics),
    }
