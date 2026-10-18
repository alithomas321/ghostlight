import json
from pathlib import Path

import numpy as np

from ghostlight.parity import canonical_parity_fixture


def test_saved_parity_fixture_matches_canonical_python_exactly():
    fixture_path = Path(__file__).parents[1] / "data" / "canonical_parity_fixture.json"
    saved = json.loads(fixture_path.read_text())
    assert saved == canonical_parity_fixture()
    assert [case["name"] for case in saved["cases"]] == [
        "primary",
        "zero_blend_flux",
        "zero_blend_offset",
        "maximum_blend_primary_orientation",
        "maximum_blend_rotated_orientation",
    ]
    assert len(saved["times_days"]) == 5
    assert len({case["blend_orientation_deg"] for case in saved["cases"]}) >= 3
    for case in saved["cases"]:
        np.testing.assert_allclose(
            case["baseline_relative_centroid_xy_mas"],
            case["direct_flux_weighted_difference_xy_mas"],
            rtol=1e-13,
            atol=1e-13,
        )


def test_parity_limit_cases_are_structurally_distinct():
    fixture = canonical_parity_fixture()
    cases = {case["name"]: case for case in fixture["cases"]}
    np.testing.assert_allclose(
        cases["zero_blend_flux"]["baseline_relative_centroid_xy_mas"],
        fixture["dark_centroid_shift_xy_mas"],
        rtol=1e-13,
        atol=1e-13,
    )
    assert cases["zero_blend_offset"]["blend_offset_xy_mas"] == [0.0, 0.0]
    assert cases["maximum_blend_primary_orientation"]["blend_flux_ratio"] == 1.0
    assert cases["maximum_blend_primary_orientation"]["blend_offset_magnitude_mas"] == 20.0
