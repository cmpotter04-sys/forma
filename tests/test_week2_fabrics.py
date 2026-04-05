"""
tests/test_week2_fabrics.py

AC-3 acceptance tests: Fabric parameter sensitivity.
Verifies that different fabric materials produce measurably different
simulation results on the same garment+body pair (M T-shirt on M body).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BODY_M = ROOT / "data" / "bodies" / "makehuman_male_M.ply"
PATTERN_M = ROOT / "data" / "patterns" / "tshirt_size_M.json"
MANIFEST_M = ROOT / "seam_manifests" / "tshirt_size_M_manifest.json"
FABRIC_LIB = ROOT / "data" / "fabrics" / "fabric_library.json"

FABRICS = ["cotton_jersey_default", "silk_charmeuse", "denim_12oz"]


def _load_fabric(fabric_id: str) -> dict:
    with open(FABRIC_LIB) as f:
        lib = json.load(f)
    return lib["fabrics"][fabric_id]


@pytest.fixture(scope="module")
def fabric_results() -> dict[str, dict]:
    """Run M garment on M body for all 3 fabrics, cached for the module."""
    from src.geometer.xpbd_simulate import run_simulation

    results = {}
    for fid in FABRICS:
        results[fid] = run_simulation(
            str(BODY_M), str(PATTERN_M), str(MANIFEST_M), _load_fabric(fid),
        )
    return results


class TestFabricSensitivity:
    def test_all_fabrics_fit(self, fabric_results):
        """All 3 fabrics should produce fit=True on M garment + M body."""
        from src.geometer.clearance import classify_severity

        for fid in FABRICS:
            r = fabric_results[fid]
            for region, delta in r["clearance_map"].items():
                sr = r["strain_ratio_map"].get(region, 1.0)
                sev = classify_severity(delta, sr)
                assert sev != "red", (
                    f"{fid} region {region}: severity={sev} "
                    f"(delta={delta:.1f}mm, strain={sr:.4f})"
                )

    def test_fabrics_produce_different_clearance(self, fabric_results):
        """At least 2 fabrics differ by > 0.5mm on chest_front delta_mm."""
        deltas = {
            fid: fabric_results[fid]["clearance_map"]["chest_front"]
            for fid in FABRICS
        }
        diffs = []
        for i, f1 in enumerate(FABRICS):
            for f2 in FABRICS[i + 1:]:
                diffs.append(abs(deltas[f1] - deltas[f2]))
        assert max(diffs) > 0.5, (
            f"All chest_front deltas too similar: {deltas}; "
            f"max diff = {max(diffs):.3f}mm (need > 0.5mm)"
        )

    def test_fabric_does_not_flip_verdict(self, fabric_results):
        """No fabric should cause fit=False on a correctly-fitting pair."""
        from src.geometer.clearance import classify_severity

        for fid in FABRICS:
            r = fabric_results[fid]
            has_red = any(
                classify_severity(
                    r["clearance_map"][reg],
                    r["strain_ratio_map"].get(reg, 1.0),
                ) == "red"
                for reg in r["clearance_map"]
            )
            assert not has_red, (
                f"Fabric {fid} flipped verdict to fit=False"
            )

    def test_denim_stiffer_than_silk(self, fabric_results):
        """Denim chest_front delta_mm >= silk (stiffer fabric stands off body more)."""
        denim_delta = fabric_results["denim_12oz"]["clearance_map"]["chest_front"]
        silk_delta = fabric_results["silk_charmeuse"]["clearance_map"]["chest_front"]
        assert denim_delta >= silk_delta, (
            f"Denim ({denim_delta:.1f}mm) should have >= clearance "
            f"than silk ({silk_delta:.1f}mm)"
        )
