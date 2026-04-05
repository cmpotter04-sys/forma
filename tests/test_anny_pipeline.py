"""
tests/test_anny_pipeline.py

Integration tests for the Anny parametric body path through the full pipeline.

Tests the flow: anny_measurements dict → generate_anny_body() → simulation →
fit_verdict dict, using the CPU backend only (no GPU required).

Sizes verified against ANNY_SIZES presets defined in src/sculptor/anny_body.py.
All pattern and manifest files are sourced from data/patterns/ and
seam_manifests/ respectively.  A test is skipped if the required files are
absent rather than failing hard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

PATTERN_S = ROOT / "data" / "patterns" / "tshirt_size_S.json"
MANIFEST_S = ROOT / "seam_manifests" / "tshirt_size_S_manifest.json"

PATTERN_XL = ROOT / "data" / "patterns" / "tshirt_size_XL.json"
MANIFEST_XL = ROOT / "seam_manifests" / "tshirt_size_XL_manifest.json"

PATTERN_M = ROOT / "data" / "patterns" / "tshirt_size_M.json"
MANIFEST_M = ROOT / "seam_manifests" / "tshirt_size_M_manifest.json"

# ---------------------------------------------------------------------------
# Anny measurement presets — mirrors ANNY_SIZES in src/sculptor/anny_body.py
# ---------------------------------------------------------------------------

ANNY_S = {
    "height_cm": 166, "chest_cm": 86, "waist_cm": 67,
    "hips_cm": 92, "inseam_cm": 74, "shoulder_width_cm": 37,
}

ANNY_M = {
    "height_cm": 168, "chest_cm": 90, "waist_cm": 72,
    "hips_cm": 96, "inseam_cm": 76, "shoulder_width_cm": 38,
}

ANNY_XL = {
    "height_cm": 172, "chest_cm": 104, "waist_cm": 87,
    "hips_cm": 110, "inseam_cm": 78, "shoulder_width_cm": 41,
}

# ---------------------------------------------------------------------------
# Required top-level verdict fields (v1.2 schema)
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL = [
    "verdict_id", "fit", "confidence", "body_source",
    "scan_method", "scan_accuracy_mm", "garment_id",
    "body_profile_id", "strain_map", "ease_map",
    "simulation_ms", "convergence_step", "final_kinetic_energy_j",
    "tunnel_through_pct", "fabric_params_used",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _files_exist(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _run_anny(measurements: dict, pattern: Path, manifest: Path) -> dict:
    """Convenience wrapper around run_fit_check for Anny inputs."""
    from src.pipeline import run_fit_check
    return run_fit_check(
        anny_measurements=measurements,
        pattern_path=str(pattern),
        seam_manifest_path=str(manifest),
        backend="cpu",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnnyMeasurementsAccepted:
    """run_fit_check accepts anny_measurements and returns a verdict dict."""

    @pytest.mark.skipif(
        not _files_exist(PATTERN_M, MANIFEST_M),
        reason="tshirt_size_M pattern or manifest not found",
    )
    def test_anny_measurements_accepted(self):
        verdict = _run_anny(ANNY_M, PATTERN_M, MANIFEST_M)
        assert isinstance(verdict, dict), "run_fit_check must return a dict"
        assert "fit" in verdict, "verdict must contain 'fit' key"
        assert isinstance(verdict["fit"], bool), "'fit' must be a bool"


class TestAnnyBodySource:
    """The body_source field must be 'anny_parametric' for Anny-generated bodies."""

    @pytest.mark.skipif(
        not _files_exist(PATTERN_M, MANIFEST_M),
        reason="tshirt_size_M pattern or manifest not found",
    )
    def test_anny_verdict_has_correct_body_source(self):
        verdict = _run_anny(ANNY_M, PATTERN_M, MANIFEST_M)
        assert verdict.get("body_source") == "anny_parametric", (
            f"Expected body_source='anny_parametric', got {verdict.get('body_source')!r}"
        )


class TestAnnyVerdictSchema:
    """Verdicts produced via the Anny path must be fully v1.2-schema-conformant."""

    @pytest.mark.skipif(
        not _files_exist(PATTERN_M, MANIFEST_M),
        reason="tshirt_size_M pattern or manifest not found",
    )
    def test_all_top_level_fields_present(self):
        verdict = _run_anny(ANNY_M, PATTERN_M, MANIFEST_M)
        for field in REQUIRED_TOP_LEVEL:
            assert field in verdict, f"verdict missing required field: {field!r}"

    @pytest.mark.skipif(
        not _files_exist(PATTERN_M, MANIFEST_M),
        reason="tshirt_size_M pattern or manifest not found",
    )
    def test_verdict_id_format(self):
        verdict = _run_anny(ANNY_M, PATTERN_M, MANIFEST_M)
        vid = verdict["verdict_id"]
        assert vid.startswith("vrd_"), f"verdict_id must start with 'vrd_': {vid!r}"
        hex_part = vid[4:]
        assert len(hex_part) == 12, (
            f"verdict_id hex part must be exactly 12 chars: {vid!r}"
        )
        int(hex_part, 16)  # raises ValueError if not valid hex

    @pytest.mark.skipif(
        not _files_exist(PATTERN_M, MANIFEST_M),
        reason="tshirt_size_M pattern or manifest not found",
    )
    def test_fit_consistency_with_strain_map(self):
        verdict = _run_anny(ANNY_M, PATTERN_M, MANIFEST_M)
        has_red = any(e["severity"] == "red" for e in verdict["strain_map"])
        if verdict["fit"]:
            assert not has_red, "fit=True but red regions present in strain_map"
        else:
            assert has_red, "fit=False but no red regions in strain_map"


class TestAnnySizeSVsSizeS:
    """Anny size-S body against tshirt_size_S pattern — no exceptions, valid schema."""

    @pytest.mark.skipif(
        not _files_exist(PATTERN_S, MANIFEST_S),
        reason="tshirt_size_S pattern or manifest not found",
    )
    def test_anny_size_s_fits_size_s_pattern(self):
        verdict = _run_anny(ANNY_S, PATTERN_S, MANIFEST_S)
        assert isinstance(verdict, dict)
        for field in REQUIRED_TOP_LEVEL:
            assert field in verdict, f"verdict missing field: {field!r}"
        assert verdict.get("body_source") == "anny_parametric"
        severities = {e["severity"] for e in verdict["strain_map"]}
        assert severities <= {"green", "yellow", "red"}, (
            f"Unexpected severity values: {severities - {'green', 'yellow', 'red'}}"
        )


class TestAnnySizeXLVsSizeXL:
    """Anny size-XL body against tshirt_size_XL pattern — no exceptions, valid schema."""

    @pytest.mark.skipif(
        not _files_exist(PATTERN_XL, MANIFEST_XL),
        reason="tshirt_size_XL pattern or manifest not found",
    )
    def test_anny_size_xl_fits_size_xl_pattern(self):
        verdict = _run_anny(ANNY_XL, PATTERN_XL, MANIFEST_XL)
        assert isinstance(verdict, dict)
        for field in REQUIRED_TOP_LEVEL:
            assert field in verdict, f"verdict missing field: {field!r}"
        assert verdict.get("body_source") == "anny_parametric"
        severities = {e["severity"] for e in verdict["strain_map"]}
        assert severities <= {"green", "yellow", "red"}, (
            f"Unexpected severity values: {severities - {'green', 'yellow', 'red'}}"
        )


class TestAnnyInputValidation:
    """Error conditions for the anny_measurements parameter."""

    @pytest.mark.skipif(
        not _files_exist(PATTERN_M, MANIFEST_M),
        reason="tshirt_size_M pattern or manifest not found",
    )
    def test_anny_missing_measurements_raises(self):
        """
        Passing an empty dict for anny_measurements must raise ValueError or
        KeyError.  generate_anny_body() expects all six measurement keys; an
        empty dict means they are missing, so a TypeError (unexpected keyword
        args via **{}) would actually not reach that function — the pipeline
        calls generate_anny_body(**anny_measurements), so an empty dict
        results in missing required positional arguments → TypeError.
        We accept ValueError, KeyError, or TypeError.
        """
        from src.pipeline import run_fit_check
        with pytest.raises((ValueError, KeyError, TypeError)):
            run_fit_check(
                anny_measurements={},
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                backend="cpu",
            )

    def test_anny_and_body_mesh_path_mutual_exclusion(self):
        """
        Providing both anny_measurements and body_mesh_path must raise
        ValueError immediately (before any file I/O).
        """
        from src.pipeline import run_fit_check
        with pytest.raises(ValueError, match="not both"):
            run_fit_check(
                body_mesh_path="some_body.ply",
                anny_measurements=ANNY_M,
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                backend="cpu",
            )

    def test_neither_body_nor_anny_raises(self):
        """
        Omitting both body_mesh_path and anny_measurements must raise
        ValueError.
        """
        from src.pipeline import run_fit_check
        with pytest.raises(ValueError):
            run_fit_check(
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                backend="cpu",
            )
