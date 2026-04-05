"""
tests/test_female_bodies.py

Tests for female body mesh generation and measurement validation.

Covers:
- File existence for all three female sizes (S, M, XL)
- Vertex topology parity with the male mesh (21,833 verts)
- body_profile JSON v1.2 schema compliance for female M
- MeasurementValidator behaviour: happy path, strict mode, edge cases
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
BODIES_DIR = ROOT / "data" / "bodies"

FEMALE_CONFIGS = {
    "S":  {
        "ply":     BODIES_DIR / "makehuman_female_S.ply",
        "profile": BODIES_DIR / "makehuman_female_S.json",
        "height_cm": 163.0, "chest_cm": 84.0, "waist_cm": 66.0, "hip_cm": 90.0,
    },
    "M":  {
        "ply":     BODIES_DIR / "makehuman_female_M.ply",
        "profile": BODIES_DIR / "makehuman_female_M.json",
        "height_cm": 167.0, "chest_cm": 92.0, "waist_cm": 74.0, "hip_cm": 98.0,
    },
    "XL": {
        "ply":     BODIES_DIR / "makehuman_female_XL.ply",
        "profile": BODIES_DIR / "makehuman_female_XL.json",
        "height_cm": 172.0, "chest_cm": 104.0, "waist_cm": 86.0, "hip_cm": 110.0,
    },
}

# Add src to path so the validator module is importable when running from the
# repo root without a package install.
sys.path.insert(0, str(ROOT / "src"))

from sculptor.measurement_validator import validate_measurements, ValidationResult  # noqa: E402


# ---------------------------------------------------------------------------
# Female body generation — file existence and topology
# ---------------------------------------------------------------------------

class TestFemalePlyExists:
    """PLY files must exist for all three ISO female sizes."""

    def test_female_s_ply_exists(self):
        assert FEMALE_CONFIGS["S"]["ply"].exists(), (
            f"Missing: {FEMALE_CONFIGS['S']['ply']}"
        )

    def test_female_m_ply_exists(self):
        assert FEMALE_CONFIGS["M"]["ply"].exists(), (
            f"Missing: {FEMALE_CONFIGS['M']['ply']}"
        )

    def test_female_xl_ply_exists(self):
        assert FEMALE_CONFIGS["XL"]["ply"].exists(), (
            f"Missing: {FEMALE_CONFIGS['XL']['ply']}"
        )

    def test_female_m_vertex_count(self):
        """Female M must share the same 21,833-vertex topology as all MakeHuman meshes."""
        import trimesh  # only needed for this one topology check

        mesh = trimesh.load(str(FEMALE_CONFIGS["M"]["ply"]), process=False)
        assert len(mesh.vertices) == 21_833, (
            f"Expected 21,833 vertices, got {len(mesh.vertices)}"
        )

    def test_female_m_profile_valid(self):
        """body_profile JSON for female M must contain body_source and achieved_measurements."""
        with open(FEMALE_CONFIGS["M"]["profile"]) as f:
            profile = json.load(f)

        assert "body_source" in profile, "body_source key missing from female M profile"
        assert "achieved_measurements" in profile, (
            "achieved_measurements key missing from female M profile"
        )
        assert profile["body_source"] == "synthetic_mannequin"


# ---------------------------------------------------------------------------
# Measurement validator
# ---------------------------------------------------------------------------

class TestMeasurementValidator:
    """Validate measurement_validator.py behaviour against real and synthetic profiles."""

    def test_validate_male_m_all_ok(self):
        """Validating male M against its own stored targets should return valid=True."""
        profile_path = BODIES_DIR / "makehuman_male_M.json"
        with open(profile_path) as f:
            profile = json.load(f)
        targets = profile["measurements"]  # exactly the targets the sculptor was given

        result = validate_measurements(profile_path, targets)

        assert isinstance(result, ValidationResult)
        assert result.valid is True

    def test_validate_female_m_returns_result(self):
        """validate_measurements must return a ValidationResult for female M."""
        profile_path = FEMALE_CONFIGS["M"]["profile"]
        targets = {"chest_cm": 92.0, "waist_cm": 74.0, "hip_cm": 98.0, "height_cm": 167.0}

        result = validate_measurements(profile_path, targets)

        assert isinstance(result, ValidationResult)

    def test_validate_deltas_have_correct_keys(self):
        """Every MeasurementDelta must expose the five required attributes."""
        profile_path = FEMALE_CONFIGS["M"]["profile"]
        targets = {"chest_cm": 92.0, "waist_cm": 74.0, "hip_cm": 98.0}

        result = validate_measurements(profile_path, targets)

        assert len(result.deltas) > 0, "Expected at least one delta"
        for delta in result.deltas:
            assert hasattr(delta, "key")
            assert hasattr(delta, "target_cm")
            assert hasattr(delta, "achieved_cm")
            assert hasattr(delta, "delta_mm")
            assert hasattr(delta, "within_tolerance")

    def test_validate_strict_mode_female_chest(self):
        """
        Pass a target 25cm off from achieved (e.g. 67.0 vs ~92cm achieved).
        Delta ~250mm >> 15mm tolerance, so strict=True must mark it out of tolerance.
        """
        profile_path = FEMALE_CONFIGS["M"]["profile"]
        targets = {"chest_cm": 67.0}  # deliberately wrong — 25cm off

        result = validate_measurements(profile_path, targets, strict=True)

        chest_delta = next(d for d in result.deltas if d.key == "chest_cm")
        assert chest_delta.within_tolerance is False, (
            f"Expected chest WARN in strict mode, got delta={chest_delta.delta_mm:.1f}mm"
        )
        assert result.valid is False, (
            "strict=True must set valid=False when any measurement exceeds tolerance"
        )

    def test_validate_missing_key_skipped(self):
        """A key absent from achieved_measurements must be silently skipped, not raise."""
        profile_path = FEMALE_CONFIGS["M"]["profile"]
        # 'arm_cm' does not exist in any body profile
        targets = {"chest_cm": 92.0, "arm_cm": 30.0}

        result = validate_measurements(profile_path, targets)

        # Only chest_cm should appear; arm_cm silently dropped
        assert all(d.key != "arm_cm" for d in result.deltas)
        assert any(d.key == "chest_cm" for d in result.deltas)

    def test_validate_exact_match_valid(self):
        """Passing achieved values as targets must yield all deltas ≈ 0 and valid=True."""
        profile_path = FEMALE_CONFIGS["M"]["profile"]
        with open(profile_path) as f:
            profile = json.load(f)
        # Use the achieved values verbatim as the targets
        targets = {k: v for k, v in profile["achieved_measurements"].items()
                   if isinstance(v, (int, float))}

        result = validate_measurements(profile_path, targets)

        assert result.valid is True
        for delta in result.deltas:
            assert abs(delta.delta_mm) < 0.1, (
                f"{delta.key}: expected ~0mm delta, got {delta.delta_mm:.3f}mm"
            )

    def test_validate_summary_string(self):
        """result.summary() must return a non-empty string containing 'valid='."""
        profile_path = FEMALE_CONFIGS["M"]["profile"]
        targets = {"chest_cm": 92.0, "waist_cm": 74.0}

        result = validate_measurements(profile_path, targets)
        summary = result.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "valid=" in summary

    def test_validate_nonexistent_profile_raises(self):
        """validate_measurements must raise FileNotFoundError for a missing profile path."""
        bad_path = BODIES_DIR / "does_not_exist.json"
        targets = {"chest_cm": 92.0}

        with pytest.raises(FileNotFoundError):
            validate_measurements(bad_path, targets)
