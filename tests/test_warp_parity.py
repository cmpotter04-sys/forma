"""
tests/test_warp_parity.py — AC-3: Warp vs CPU regression test

Runs both backends on identical inputs (M-on-M, cotton_jersey_default)
and asserts that results match within tolerance:
  - Per-region clearance delta ≤ 0.5mm
  - Per-region strain ratio delta ≤ 0.02
  - Verdict agreement (fit: true/false matches)

Automatically skips if warp-lang is not installed (local dev environment).
On Colab with GPU, this test must pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Check Warp availability (need warp.sim for the GPU backend)
try:
    import warp as wp
    import warp.sim
    HAS_WARP = True
except (ImportError, ModuleNotFoundError):
    HAS_WARP = False

# Test data paths
DATA_DIR = _ROOT / "data"
BODY_M = str(DATA_DIR / "bodies" / "makehuman_male_M.ply")
PATTERN_M = str(DATA_DIR / "patterns" / "tshirt_size_M.json")
MANIFEST_M = str(_ROOT / "seam_manifests" / "tshirt_size_M_manifest.json")

REQUIRED_REGIONS = [
    "chest_front", "chest_side", "shoulder_left",
    "shoulder_right", "upper_back", "waist",
]


def _check_test_data_exists() -> bool:
    """Return True if all test data files exist."""
    return (
        Path(BODY_M).exists()
        and Path(PATTERN_M).exists()
        and Path(MANIFEST_M).exists()
    )


def _strain_map_by_region(strain_map):
    """Convert strain_map list to dict keyed by region name."""
    return {entry["region"]: entry for entry in strain_map}


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
@pytest.mark.skipif(not _check_test_data_exists(), reason="Test data not found")
class TestWarpParity:
    """AC-3: Side-by-side CPU vs Warp regression test."""

    @pytest.fixture(scope="class")
    def cpu_verdict(self):
        """Run CPU backend once for the test class."""
        from pipeline import run_fit_check
        return run_fit_check(
            BODY_M, PATTERN_M, MANIFEST_M,
            fabric_id="cotton_jersey_default",
            backend="cpu",
        )

    @pytest.fixture(scope="class")
    def warp_verdict(self):
        """Run Warp backend once for the test class."""
        from pipeline import run_fit_check
        return run_fit_check(
            BODY_M, PATTERN_M, MANIFEST_M,
            fabric_id="cotton_jersey_default",
            backend="warp",
        )

    def test_clearance_parity(self, cpu_verdict, warp_verdict):
        """Per-region clearance delta ≤ 0.5mm."""
        cpu_strain = _strain_map_by_region(cpu_verdict["strain_map"])
        warp_strain = _strain_map_by_region(warp_verdict["strain_map"])

        for region in REQUIRED_REGIONS:
            cpu_delta = cpu_strain[region]["delta_mm"]
            warp_delta = warp_strain[region]["delta_mm"]
            diff = abs(cpu_delta - warp_delta)
            assert diff <= 0.5, (
                f"Region {region}: clearance mismatch "
                f"(cpu={cpu_delta:.3f} vs warp={warp_delta:.3f}, "
                f"delta={diff:.3f}mm > 0.5mm)"
            )

    def test_strain_ratio_parity(self, cpu_verdict, warp_verdict):
        """Per-region strain ratio delta ≤ 0.02."""
        cpu_strain = _strain_map_by_region(cpu_verdict["strain_map"])
        warp_strain = _strain_map_by_region(warp_verdict["strain_map"])

        for region in REQUIRED_REGIONS:
            cpu_sr = cpu_strain[region]["median_strain_ratio"]
            warp_sr = warp_strain[region]["median_strain_ratio"]
            diff = abs(cpu_sr - warp_sr)
            assert diff <= 0.02, (
                f"Region {region}: strain ratio mismatch "
                f"(cpu={cpu_sr:.4f} vs warp={warp_sr:.4f}, "
                f"delta={diff:.4f} > 0.02)"
            )

    def test_verdict_agreement(self, cpu_verdict, warp_verdict):
        """fit: true/false must agree between backends."""
        assert cpu_verdict["fit"] == warp_verdict["fit"], (
            f"Verdict mismatch: cpu fit={cpu_verdict['fit']}, "
            f"warp fit={warp_verdict['fit']}"
        )

    def test_severity_agreement(self, cpu_verdict, warp_verdict):
        """Per-region severity (green/yellow/red) must agree."""
        cpu_sm = _strain_map_by_region(cpu_verdict["strain_map"])
        warp_sm = _strain_map_by_region(warp_verdict["strain_map"])
        for region in REQUIRED_REGIONS:
            cpu_sev = cpu_sm[region]["severity"]
            warp_sev = warp_sm[region]["severity"]
            assert cpu_sev == warp_sev, (
                f"Region {region}: severity mismatch "
                f"(cpu={cpu_sev}, warp={warp_sev})"
            )


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
@pytest.mark.skipif(not _check_test_data_exists(), reason="Test data not found")
def test_warp_backend_produces_valid_verdict():
    """Smoke test: Warp backend returns a structurally valid verdict."""
    from pipeline import run_fit_check

    verdict = run_fit_check(
        BODY_M, PATTERN_M, MANIFEST_M,
        fabric_id="cotton_jersey_default",
        backend="warp",
    )

    # Basic structure checks
    assert isinstance(verdict, dict)
    assert "fit" in verdict
    assert "strain_map" in verdict
    assert "ease_map" in verdict
    assert "verdict_id" in verdict
    assert verdict["verdict_id"].startswith("vrd_")

    strain_by_region = _strain_map_by_region(verdict["strain_map"])
    for region in REQUIRED_REGIONS:
        assert region in strain_by_region, f"Missing region: {region}"
        rm = strain_by_region[region]
        assert "delta_mm" in rm
        assert "severity" in rm
        assert "median_strain_ratio" in rm


# --- CPU-only sanity tests (always run) ------------------------------------

def test_cpu_pipeline_still_works():
    """Regression: CPU backend import and basic validation."""
    from pipeline import run_fit_check
    # Check that the function signature accepts backend parameter
    import inspect
    sig = inspect.signature(run_fit_check)
    assert "backend" in sig.parameters, "pipeline.run_fit_check missing 'backend' param"
    assert sig.parameters["backend"].default == "cpu", "Default backend should be 'cpu'"


def test_invalid_backend_raises():
    """ValueError for unknown backend."""
    from pipeline import run_fit_check
    if not _check_test_data_exists():
        pytest.skip("Test data not found")
    with pytest.raises(ValueError, match="Unknown backend"):
        run_fit_check(
            BODY_M, PATTERN_M, MANIFEST_M,
            backend="invalid_backend",
        )
