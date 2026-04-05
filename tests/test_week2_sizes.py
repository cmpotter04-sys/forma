"""
tests/test_week2_sizes.py

AC-1 Extended Size Range tests.
Runs simulations across 5 garment sizes and 3 body sizes to verify
clearance monotonicity, fit/no-fit thresholds, and cross-body behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.geometer.xpbd_simulate import run_simulation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BODY_M_PLY = ROOT / "data" / "bodies" / "makehuman_male_M.ply"
BODY_S_PLY = ROOT / "data" / "bodies" / "makehuman_male_S.ply"
BODY_XL_PLY = ROOT / "data" / "bodies" / "makehuman_male_XL.ply"
FABRIC_LIB = ROOT / "data" / "fabrics" / "fabric_library.json"

SIZES = ["XS", "S", "M", "L", "XL"]


def _pattern_path(size: str) -> Path:
    return ROOT / "data" / "patterns" / f"tshirt_size_{size}.json"


def _manifest_path(size: str) -> Path:
    return ROOT / "seam_manifests" / f"tshirt_size_{size}_manifest.json"


# ---------------------------------------------------------------------------
# Module-scoped fixtures (expensive — run each simulation once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fabric_params():
    with open(FABRIC_LIB) as f:
        return json.load(f)["fabrics"]["cotton_jersey_default"]


@pytest.fixture(scope="module")
def five_size_results(fabric_params):
    """Run all 5 sizes on M body. Returns dict[size] -> sim_result."""
    results = {}
    for size in SIZES:
        results[size] = run_simulation(
            body_mesh_path=BODY_M_PLY,
            pattern_path=_pattern_path(size),
            seam_manifest_path=_manifest_path(size),
            fabric_params=fabric_params,
        )
    return results


@pytest.fixture(scope="module")
def crossbody_m_on_s(fabric_params):
    """M garment on S body."""
    return run_simulation(
        body_mesh_path=BODY_S_PLY,
        pattern_path=_pattern_path("M"),
        seam_manifest_path=_manifest_path("M"),
        fabric_params=fabric_params,
    )


@pytest.fixture(scope="module")
def crossbody_m_on_xl(fabric_params):
    """M garment on XL body."""
    return run_simulation(
        body_mesh_path=BODY_XL_PLY,
        pattern_path=_pattern_path("M"),
        seam_manifest_path=_manifest_path("M"),
        fabric_params=fabric_params,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWeek2Sizes:

    def test_chest_front_monotonic(self, five_size_results):
        """chest_front delta_mm must be strictly monotonic: XS < S < M < L < XL."""
        values = [five_size_results[s]["clearance_map"]["chest_front"] for s in SIZES]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], (
                f"Monotonicity broken: {SIZES[i]}={values[i]:.1f}mm "
                f">= {SIZES[i+1]}={values[i+1]:.1f}mm"
            )

    def test_xs_is_no_fit(self, five_size_results):
        """XS on M body must have fit=False (at least one red region, delta < -25mm)."""
        from src.geometer.clearance import classify_severity
        cm = five_size_results["XS"]["clearance_map"]
        has_red = any(classify_severity(v) == "red" for v in cm.values())
        assert has_red, (
            f"XS should have at least one red region. "
            f"Values: {cm}"
        )

    def test_s_is_yellow_fit(self, five_size_results):
        """S on M body must have fit=True (no red regions; yellow is acceptable)."""
        from src.geometer.clearance import classify_severity
        cm = five_size_results["S"]["clearance_map"]
        has_red = any(classify_severity(v) == "red" for v in cm.values())
        assert not has_red, (
            f"S should have no red regions. Values: {cm}"
        )

    def test_m_l_xl_are_green_fit(self, five_size_results):
        """M, L, XL on M body: all regions should be green (positive clearance or > -10mm)."""
        from src.geometer.clearance import classify_severity
        for size in ["M", "L", "XL"]:
            cm = five_size_results[size]["clearance_map"]
            for region, val in cm.items():
                severity = classify_severity(val)
                assert severity == "green", (
                    f"Size {size} region {region} has severity={severity} "
                    f"(delta={val:.1f}mm), expected green"
                )

    def test_crossbody_m_on_s_positive(self, crossbody_m_on_s):
        """M garment on S body should have positive chest_front clearance (ease)."""
        cf = crossbody_m_on_s["clearance_map"]["chest_front"]
        assert cf > 0.0, (
            f"M on S body chest_front should be positive, got {cf:.1f}mm"
        )

    def test_crossbody_m_on_xl_negative(self, crossbody_m_on_xl):
        """M garment on XL body should have negative chest_front clearance (tight)."""
        cf = crossbody_m_on_xl["clearance_map"]["chest_front"]
        assert cf < 0.0, (
            f"M on XL body chest_front should be negative, got {cf:.1f}mm"
        )
