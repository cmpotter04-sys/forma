"""
tests/test_warp_parity.py — AC-3: Warp vs CPU parity regression test

Runs both backends on identical inputs (M-on-M, cotton_jersey_default) and
asserts that results agree within tolerance:
  - Per-region clearance delta ≤ 0.5mm
  - Per-region strain ratio delta ≤ 0.02
  - Verdict match (fit: true/false agrees)

The entire module is skipped when warp-lang is not importable (local dev,
CPU-only CI).  On Colab with GPU runtime all 5 tests must pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Module-level skip: if warp is not importable, skip every test in this file.
# pytest.importorskip() must be called at module scope (not inside a function).
warp = pytest.importorskip("warp", reason="warp-lang not installed — skipping parity tests")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Test asset paths
# ---------------------------------------------------------------------------

BODY_M    = str(ROOT / "data" / "bodies"          / "makehuman_male_M.ply")
PATTERN_M = str(ROOT / "data" / "patterns"        / "tshirt_size_M.json")
MANIFEST_M = str(ROOT / "data" / "seam_manifests" / "tshirt_size_M_seam_manifest.json")
FABRIC_ID  = "cotton_jersey_default"

REQUIRED_REGIONS = [
    "chest_front",
    "chest_side",
    "shoulder_left",
    "shoulder_right",
    "upper_back",
    "waist",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_exists() -> bool:
    return (
        Path(BODY_M).exists()
        and Path(PATTERN_M).exists()
        and Path(MANIFEST_M).exists()
    )


def _strain_by_region(verdict: dict) -> dict:
    """Convert verdict['strain_map'] list → dict keyed by region name."""
    return {entry["region"]: entry for entry in verdict.get("strain_map", [])}


# ---------------------------------------------------------------------------
# Combined fixture — runs BOTH backends once, cached for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def both_verdicts() -> dict:
    """
    Run both CPU and Warp backends on the same M-on-M inputs.

    Returns
    -------
    dict with keys "cpu" and "warp", each holding a full fit_verdict dict.
    """
    if not _data_exists():
        pytest.skip(
            "Test assets missing — expected:\n"
            f"  {BODY_M}\n  {PATTERN_M}\n  {MANIFEST_M}"
        )

    from pipeline import run_fit_check

    cpu_verdict = run_fit_check(
        body_mesh_path=BODY_M,
        pattern_path=PATTERN_M,
        seam_manifest_path=MANIFEST_M,
        fabric_id=FABRIC_ID,
        backend="cpu",
    )

    warp_verdict = run_fit_check(
        body_mesh_path=BODY_M,
        pattern_path=PATTERN_M,
        seam_manifest_path=MANIFEST_M,
        fabric_id=FABRIC_ID,
        backend="warp",
    )

    return {"cpu": cpu_verdict, "warp": warp_verdict}


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------

class TestWarpCpuParity:
    """Five parity tests comparing Warp and CPU backends on M-on-M inputs."""

    def test_region_count_matches(self, both_verdicts):
        """
        Both backends must return strain_map entries for exactly the same
        regions.  A mismatch here indicates a backend-specific region-
        segmentation bug.
        """
        cpu_regions  = {e["region"] for e in both_verdicts["cpu"]["strain_map"]}
        warp_regions = {e["region"] for e in both_verdicts["warp"]["strain_map"]}

        assert cpu_regions == warp_regions, (
            f"Region sets differ.\n"
            f"  CPU only : {cpu_regions - warp_regions}\n"
            f"  Warp only: {warp_regions - cpu_regions}"
        )
        for region in REQUIRED_REGIONS:
            assert region in cpu_regions, (
                f"Required region {region!r} missing from both backends"
            )

    def test_chest_front_clearance_delta(self, both_verdicts):
        """
        chest_front: |cpu_delta_mm - warp_delta_mm| ≤ 0.5mm.
        Tolerance spec: FORMA_PHASE2_EXECUTOR_SPEC.md AC-3.
        """
        cpu_sm  = _strain_by_region(both_verdicts["cpu"])
        warp_sm = _strain_by_region(both_verdicts["warp"])

        cpu_delta  = cpu_sm["chest_front"]["delta_mm"]
        warp_delta = warp_sm["chest_front"]["delta_mm"]
        diff = abs(cpu_delta - warp_delta)

        assert diff <= 0.5, (
            f"chest_front clearance parity failure: "
            f"cpu={cpu_delta:.3f}mm  warp={warp_delta:.3f}mm  "
            f"delta={diff:.3f}mm  (limit 0.5mm)"
        )

    def test_waist_clearance_delta(self, both_verdicts):
        """
        waist: |cpu_delta_mm - warp_delta_mm| ≤ 0.5mm.
        Tolerance spec: FORMA_PHASE2_EXECUTOR_SPEC.md AC-3.
        """
        cpu_sm  = _strain_by_region(both_verdicts["cpu"])
        warp_sm = _strain_by_region(both_verdicts["warp"])

        cpu_delta  = cpu_sm["waist"]["delta_mm"]
        warp_delta = warp_sm["waist"]["delta_mm"]
        diff = abs(cpu_delta - warp_delta)

        assert diff <= 0.5, (
            f"waist clearance parity failure: "
            f"cpu={cpu_delta:.3f}mm  warp={warp_delta:.3f}mm  "
            f"delta={diff:.3f}mm  (limit 0.5mm)"
        )

    def test_fit_verdict_matches(self, both_verdicts):
        """
        fit: true/false must agree between CPU and Warp backends.
        Tolerance spec: FORMA_PHASE2_EXECUTOR_SPEC.md AC-3.
        """
        cpu_fit  = both_verdicts["cpu"]["fit"]
        warp_fit = both_verdicts["warp"]["fit"]

        assert cpu_fit == warp_fit, (
            f"Fit verdict mismatch: cpu={cpu_fit}  warp={warp_fit}.\n"
            "Both backends must agree on whether the garment fits."
        )

    def test_strain_ratio_deltas(self, both_verdicts):
        """
        Per-region |cpu_median_strain_ratio - warp_median_strain_ratio| ≤ 0.02
        for all required regions.
        Tolerance spec: FORMA_PHASE2_EXECUTOR_SPEC.md AC-3.
        """
        cpu_sm  = _strain_by_region(both_verdicts["cpu"])
        warp_sm = _strain_by_region(both_verdicts["warp"])

        failures = []
        for region in REQUIRED_REGIONS:
            cpu_sr  = cpu_sm.get(region, {}).get("median_strain_ratio")
            warp_sr = warp_sm.get(region, {}).get("median_strain_ratio")

            if cpu_sr is None or warp_sr is None:
                failures.append(
                    f"  {region}: median_strain_ratio missing "
                    f"(cpu={cpu_sr!r}, warp={warp_sr!r})"
                )
                continue

            diff = abs(cpu_sr - warp_sr)
            if diff > 0.02:
                failures.append(
                    f"  {region}: cpu={cpu_sr:.4f}  warp={warp_sr:.4f}  "
                    f"delta={diff:.4f}  (limit 0.02)"
                )

        assert not failures, (
            "Strain ratio parity failures (|cpu - warp| > 0.02):\n"
            + "\n".join(failures)
        )
