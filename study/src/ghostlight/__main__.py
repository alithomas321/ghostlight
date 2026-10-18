"""Command-line interface for Ghostlight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ghostlight microlensing calibration lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    verify_parser.add_argument(
        "--recompute",
        action="store_true",
        help=(
            "recompute locally prespecified metrics with the deterministic verification RNG and "
            "compare them without rewriting artifacts"
        ),
    )
    refresh_parser = subparsers.add_parser("refresh-reporting")
    refresh_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from ghostlight.pipeline import refresh_reporting_artifacts, run_pipeline, verify_pipeline

    if args.command == "run":
        result = run_pipeline(args.project_root)
        summary = {
            "synthetic_dataset_count": result["configuration"]["synthetic_dataset_count"],
            "primary_cell": result["primary_cell"],
        }
    elif args.command == "verify":
        summary = verify_pipeline(args.project_root, recompute=args.recompute)
    else:
        summary = refresh_reporting_artifacts(args.project_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
