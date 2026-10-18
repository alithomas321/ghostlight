"""Locally prespecified configuration for the Ghostlight synthetic study."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from itertools import product
from numbers import Real
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

KAPPA_MAS_PER_SOLAR_MASS = 8.144
REPORT_INTERVAL_LEVEL = 0.95
REPORT_PRIMARY_ANCHOR = (32, 0.25, 0.3, 20.0)
REPORT_FALSE_PRECISION_THRESHOLDS = (0.10, 0.25, 0.50)


def scenario_id(epochs: int, sigma: float, flux_ratio: float, offset: float) -> str:
    """Stable scenario identifier used by saved artifacts."""

    return f"n{epochs}_s{sigma:.2f}_g{flux_ratio:.1f}_b{offset:.1f}"


def scenario_spawn_identity(
    epochs: int, sigma: float, flux_ratio: float, offset: float
) -> tuple[int, int, int, int]:
    """Quantized identity used by the frozen SeedSequence contract."""

    return (
        int(epochs),
        int(round(100 * sigma)),
        int(round(10 * flux_ratio)),
        int(round(10 * offset)),
    )


def _require_finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _require_two_finite_reals(value: Any, name: str) -> None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two finite real numbers")
    for item in value:
        _require_finite_real(item, name)


@dataclass(frozen=True)
class TruthConfig:
    """OB110462-scale synthetic truth; not a reproduction of that event."""

    mass_solar: float = 6.0
    source_parallax_mas: float = 0.11
    relative_parallax_mas: float = 0.47
    t_e_days: float = 276.0
    t_0_days: float = 0.0
    u_0: float = 0.05
    trajectory_angle_deg: float = 32.0
    blend_angle_deg: float = 67.0
    baseline_position_mas: tuple[float, float] = (0.35, -0.20)
    proper_motion_mas_per_year: tuple[float, float] = (2.4, -1.3)
    parallax_phase_deg: float = 23.0

    def validate(self) -> None:
        """Validate the complete synthetic truth before geometry is constructed."""

        positive = {
            "mass_solar": self.mass_solar,
            "source_parallax_mas": self.source_parallax_mas,
            "relative_parallax_mas": self.relative_parallax_mas,
            "t_e_days": self.t_e_days,
            "u_0": self.u_0,
        }
        for name, value in positive.items():
            if _require_finite_real(value, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "t_0_days": self.t_0_days,
            "trajectory_angle_deg": self.trajectory_angle_deg,
            "blend_angle_deg": self.blend_angle_deg,
            "parallax_phase_deg": self.parallax_phase_deg,
        }.items():
            _require_finite_real(value, name)
        _require_two_finite_reals(self.baseline_position_mas, "baseline_position_mas")
        _require_two_finite_reals(
            self.proper_motion_mas_per_year,
            "proper_motion_mas_per_year",
        )
        if not math.isfinite(self.source_distance_kpc) or self.source_distance_kpc <= 0:
            raise ValueError("source parallax must imply a finite positive source distance")
        if not math.isfinite(self.lens_distance_kpc) or self.lens_distance_kpc <= 0:
            raise ValueError("parallaxes must imply a finite positive lens distance")
        theta_squared = (
            KAPPA_MAS_PER_SOLAR_MASS * float(self.mass_solar) * float(self.relative_parallax_mas)
        )
        if not math.isfinite(theta_squared) or theta_squared <= 0:
            raise ValueError("mass and relative parallax must imply a finite Einstein radius")
        relative_motion = math.sqrt(theta_squared) / (float(self.t_e_days) / 365.25)
        if not math.isfinite(relative_motion) or relative_motion <= 0:
            raise ValueError("truth must imply a finite positive relative proper motion")

    @property
    def theta_e_mas(self) -> float:
        return float(
            np.sqrt(KAPPA_MAS_PER_SOLAR_MASS * self.mass_solar * self.relative_parallax_mas)
        )

    @property
    def source_distance_kpc(self) -> float:
        return 1.0 / self.source_parallax_mas

    @property
    def lens_distance_kpc(self) -> float:
        return 1.0 / (self.source_parallax_mas + self.relative_parallax_mas)

    @property
    def relative_proper_motion_mas_per_year(self) -> float:
        return self.theta_e_mas / (self.t_e_days / 365.25)


@dataclass(frozen=True)
class StudyConfig:
    """Frozen locally prespecified primary-validation grid and Monte Carlo settings."""

    seed: int = 20_260_805
    trials_per_scenario: int = 2_000
    bootstrap_resamples: int = 400
    cadences: tuple[int, ...] = (12, 32, 72)
    astrometric_sigmas_mas: tuple[float, ...] = (0.10, 0.25, 0.50)
    blend_flux_ratios: tuple[float, ...] = (0.0, 0.3, 1.0)
    blend_offsets_mas: tuple[float, ...] = (0.0, 5.0, 20.0)
    interval_level: float = 0.95
    interval_z: float = 1.959963984540054
    primary_false_precision_half_width_fraction: float = 0.10
    exploratory_false_precision_half_width_fractions: tuple[float, ...] = (0.25, 0.50)
    schedule_baseline_years: float = 6.0
    truth: TruthConfig = field(default_factory=TruthConfig)

    @property
    def scenario_count(self) -> int:
        return (
            len(self.cadences)
            * len(self.astrometric_sigmas_mas)
            * len(self.blend_flux_ratios)
            * len(self.blend_offsets_mas)
        )

    @property
    def synthetic_dataset_count(self) -> int:
        return self.scenario_count * self.trials_per_scenario

    def validate(self, *, prespecified_validation: bool = True) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.trials_per_scenario, int) or isinstance(
            self.trials_per_scenario, bool
        ):
            raise ValueError("trials_per_scenario must be an integer count")
        if not isinstance(self.bootstrap_resamples, int) or isinstance(
            self.bootstrap_resamples, bool
        ):
            raise ValueError("bootstrap_resamples must be an integer count")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in self.cadences):
            raise ValueError("cadences must contain integer epoch counts")
        if self.trials_per_scenario < 2:
            raise ValueError("trials_per_scenario must be at least two")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if prespecified_validation and self.trials_per_scenario < 400:
            raise ValueError(
                "locally prespecified validation runs require at least 400 trials per scenario"
            )
        if not self.cadences or min(self.cadences) < 5:
            raise ValueError("each cadence must contain at least five epochs")
        axes = {
            "cadences": self.cadences,
            "astrometric_sigmas_mas": self.astrometric_sigmas_mas,
            "blend_flux_ratios": self.blend_flux_ratios,
            "blend_offsets_mas": self.blend_offsets_mas,
        }
        for name, values in axes.items():
            if not values:
                raise ValueError(f"{name} cannot be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique raw values")
        sigmas = [
            _require_finite_real(value, "astrometric_sigmas_mas")
            for value in self.astrometric_sigmas_mas
        ]
        flux_ratios = [
            _require_finite_real(value, "blend_flux_ratios") for value in self.blend_flux_ratios
        ]
        offsets = [
            _require_finite_real(value, "blend_offsets_mas") for value in self.blend_offsets_mas
        ]
        if any(value <= 0 for value in sigmas):
            raise ValueError("astrometric precision values must be positive")
        if any(value < 0 for value in flux_ratios):
            raise ValueError("blend flux ratios cannot be negative")
        if any(value < 0 for value in offsets):
            raise ValueError("blend offsets cannot be negative")
        self.truth.validate()
        interval_level = _require_finite_real(self.interval_level, "interval_level")
        interval_z = _require_finite_real(self.interval_z, "interval_z")
        if interval_level != REPORT_INTERVAL_LEVEL:
            raise ValueError(
                "reportable configurations require interval_level = 0.95 because saved "
                "field names and prose are explicitly 95%"
            )
        expected_z = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
        if interval_z <= 0 or not math.isclose(
            interval_z,
            expected_z,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "interval_z must equal the two-sided standard-normal quantile implied by "
                "interval_level"
            )
        thresholds = (
            self.primary_false_precision_half_width_fraction,
            *self.exploratory_false_precision_half_width_fractions,
        )
        threshold_values = [
            _require_finite_real(value, "false-precision half-width thresholds")
            for value in thresholds
        ]
        if any(value <= 0 for value in threshold_values) or tuple(
            sorted(threshold_values)
        ) != tuple(threshold_values):
            raise ValueError("false-precision half-width thresholds must be positive and ordered")
        if len(set(int(round(100 * value)) for value in threshold_values)) != len(threshold_values):
            raise ValueError("false-precision thresholds collide in saved field names")
        baseline_years = _require_finite_real(
            self.schedule_baseline_years,
            "schedule_baseline_years",
        )
        if baseline_years <= 0:
            raise ValueError("schedule baseline must be positive")
        scenarios = list(
            product(
                self.cadences,
                self.astrometric_sigmas_mas,
                self.blend_flux_ratios,
                self.blend_offsets_mas,
            )
        )
        identifiers = [scenario_id(*scenario) for scenario in scenarios]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                "scenario IDs collide under fixed-precision serialization; choose axis values "
                "farther apart"
            )
        spawn_identities = [scenario_spawn_identity(*scenario) for scenario in scenarios]
        if len(set(spawn_identities)) != len(spawn_identities):
            raise ValueError(
                "scenario spawn identities collide under the frozen SeedSequence encoding; "
                "choose axis values farther apart"
            )
        if prespecified_validation:
            frozen = StudyConfig()
            if asdict(self) != asdict(frozen):
                raise ValueError(
                    "prespecified_validation=True requires the exact frozen StudyConfig; "
                    "custom configurations are exploratory only"
                )

    def validate_reporting_contract(self) -> None:
        """Require anchors and thresholds used by the generated public report."""

        epochs, sigma, flux_ratio, offset = REPORT_PRIMARY_ANCHOR
        if not (
            epochs in self.cadences
            and sigma in self.astrometric_sigmas_mas
            and flux_ratio in self.blend_flux_ratios
            and offset in self.blend_offsets_mas
        ):
            raise ValueError(
                "reporting requires the primary anchor (32 epochs, 0.25 mas, g=0.3, "
                "20 mas offset) before any random draw"
            )
        thresholds = (
            self.primary_false_precision_half_width_fraction,
            *self.exploratory_false_precision_half_width_fractions,
        )
        if thresholds != REPORT_FALSE_PRECISION_THRESHOLDS:
            raise ValueError(
                "reporting requires the fixed 10%, 25%, and 50% half-width diagnostics"
            )

    @property
    def specification_fingerprint(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["truth"]["theta_e_mas"] = self.truth.theta_e_mas
        payload["truth"]["source_distance_kpc"] = self.truth.source_distance_kpc
        payload["truth"]["lens_distance_kpc"] = self.truth.lens_distance_kpc
        payload["truth"]["relative_proper_motion_mas_per_year"] = (
            self.truth.relative_proper_motion_mas_per_year
        )
        payload["scenario_count"] = self.scenario_count
        payload["synthetic_dataset_count"] = self.synthetic_dataset_count
        payload["specification_fingerprint_sha256"] = self.specification_fingerprint
        return payload


def validate_project_root(project_root: Path) -> Path:
    """Resolve a project root and reject unsafe or unrelated paths."""

    root = project_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project root does not exist: {root}")
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or "ghostlight-microlensing-lab" not in pyproject.read_text():
        raise ValueError("project root is not a Ghostlight checkout")
    return root
