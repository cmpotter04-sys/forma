"""
tests/test_week2_pipeline.py

AC-5 acceptance tests: Pipeline integration function.
Verifies run_fit_check and run_batch_fit_check produce correct, deterministic,
schema-conformant verdicts with typed exceptions for invalid inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BODY_M = str(ROOT / "data" / "bodies" / "makehuman_male_M.ply")
PATTERN_M = str(ROOT / "data" / "patterns" / "tshirt_size_M.json")
MANIFEST_M = str(ROOT / "seam_manifests" / "tshirt_size_M_manifest.json")

PATTERN_S = str(ROOT / "data" / "patterns" / "tshirt_size_S.json")
MANIFEST_S = str(ROOT / "seam_manifests" / "tshirt_size_S_manifest.json")

PATTERN_XL = str(ROOT / "data" / "patterns" / "tshirt_size_XL.json")
MANIFEST_XL = str(ROOT / "seam_manifests" / "tshirt_size_XL_manifest.json")


@pytest.fixture(scope="module")
def verdict_m() -> dict:
    """Run M garment on M body once, cached for the module."""
    from src.pipeline import run_fit_check
    return run_fit_check(BODY_M, PATTERN_M, MANIFEST_M)


class TestReturnsValidVerdict:
    def test_returns_dict(self, verdict_m):
        assert isinstance(verdict_m, dict)

    def test_has_fit_key(self, verdict_m):
        assert "fit" in verdict_m
        assert isinstance(verdict_m["fit"], bool)

    def test_has_strain_map(self, verdict_m):
        assert "strain_map" in verdict_m
        assert isinstance(verdict_m["strain_map"], list)
        assert len(verdict_m["strain_map"]) >= 6

    def test_has_confidence(self, verdict_m):
        assert "confidence" in verdict_m
        assert verdict_m["confidence"] == 1.0


class TestVerdictMatchesSchema:
    REQUIRED_TOP = [
        "verdict_id", "fit", "confidence", "body_source",
        "scan_method", "scan_accuracy_mm", "garment_id",
        "body_profile_id", "strain_map", "ease_map",
        "simulation_ms", "convergence_step", "final_kinetic_energy_j",
        "tunnel_through_pct", "fabric_params_used",
    ]

    REQUIRED_REGIONS = [
        "chest_front", "chest_side", "shoulder_left",
        "shoulder_right", "upper_back", "waist",
    ]

    REQUIRED_FABRIC_FIELDS = [
        "fabric_id", "type", "density_kg_m2",
        "stretch_stiffness", "bend_stiffness", "shear_stiffness", "damping",
    ]

    def test_all_top_level_fields(self, verdict_m):
        for field in self.REQUIRED_TOP:
            assert field in verdict_m, f"Missing top-level field: {field}"

    def test_verdict_id_format(self, verdict_m):
        vid = verdict_m["verdict_id"]
        assert vid.startswith("vrd_"), f"verdict_id must start with 'vrd_': {vid}"
        hex_part = vid[4:]
        assert len(hex_part) == 12, f"verdict_id hex part must be 12 chars: {vid}"
        int(hex_part, 16)  # raises ValueError if not hex

    def test_body_source_enum(self, verdict_m):
        assert verdict_m["body_source"] in [
            "synthetic_mannequin", "standard_photo", "precision_suit"
        ]

    def test_strain_map_all_regions(self, verdict_m):
        regions = {e["region"] for e in verdict_m["strain_map"]}
        for r in self.REQUIRED_REGIONS:
            assert r in regions, f"strain_map missing region: {r}"

    def test_strain_map_has_strain_ratio(self, verdict_m):
        for entry in verdict_m["strain_map"]:
            assert "median_strain_ratio" in entry, (
                f"Region {entry['region']} missing median_strain_ratio"
            )

    def test_strain_map_severities_valid(self, verdict_m):
        for entry in verdict_m["strain_map"]:
            assert entry["severity"] in {"green", "yellow", "red"}

    def test_ease_map_all_regions(self, verdict_m):
        regions = {e["region"] for e in verdict_m["ease_map"]}
        for r in self.REQUIRED_REGIONS:
            assert r in regions, f"ease_map missing region: {r}"

    def test_fabric_params_used(self, verdict_m):
        fp = verdict_m["fabric_params_used"]
        for field in self.REQUIRED_FABRIC_FIELDS:
            assert field in fp, f"fabric_params_used missing: {field}"

    def test_fit_consistency(self, verdict_m):
        has_red = any(e["severity"] == "red" for e in verdict_m["strain_map"])
        if verdict_m["fit"]:
            assert not has_red, "fit=True but red regions exist"
        else:
            assert has_red, "fit=False but no red regions"


class TestTypedExceptions:
    def test_file_not_found_body(self):
        from src.pipeline import run_fit_check
        with pytest.raises(FileNotFoundError):
            run_fit_check("nonexistent.ply", PATTERN_M, MANIFEST_M)

    def test_file_not_found_pattern(self):
        from src.pipeline import run_fit_check
        with pytest.raises(FileNotFoundError):
            run_fit_check(BODY_M, "nonexistent.json", MANIFEST_M)

    def test_file_not_found_manifest(self):
        from src.pipeline import run_fit_check
        with pytest.raises(FileNotFoundError):
            run_fit_check(BODY_M, PATTERN_M, "nonexistent.json")

    def test_bad_fabric_id(self):
        from src.pipeline import run_fit_check
        with pytest.raises(ValueError, match="Unknown fabric_id"):
            run_fit_check(BODY_M, PATTERN_M, MANIFEST_M, fabric_id="unicorn_velvet")


class TestDeterministic:
    def test_two_runs_identical(self, verdict_m):
        from src.pipeline import run_fit_check
        v2 = run_fit_check(BODY_M, PATTERN_M, MANIFEST_M)
        for e1, e2 in zip(verdict_m["strain_map"], v2["strain_map"]):
            assert e1["delta_mm"] == e2["delta_mm"], (
                f"{e1['region']}: {e1['delta_mm']} != {e2['delta_mm']}"
            )
            assert e1.get("median_strain_ratio") == e2.get("median_strain_ratio")
        assert verdict_m["fit"] == v2["fit"]


class TestBatch:
    def test_batch_returns_correct_count(self):
        from src.pipeline import run_batch_fit_check
        verdicts = run_batch_fit_check(
            BODY_M,
            [PATTERN_S, PATTERN_M, PATTERN_XL],
            [MANIFEST_S, MANIFEST_M, MANIFEST_XL],
        )
        assert len(verdicts) == 3

    def test_batch_length_mismatch(self):
        from src.pipeline import run_batch_fit_check
        with pytest.raises(ValueError, match="same length"):
            run_batch_fit_check(
                BODY_M,
                [PATTERN_S, PATTERN_M],
                [MANIFEST_S],
            )
