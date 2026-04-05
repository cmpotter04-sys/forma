"""
tests/test_pipeline_integration.py

Real (non-mocked) integration tests for run_fit_check() and run_batch_fit_check().
Uses CPU backend only — no GPU required.

These tests actually execute the cloth simulation against real test assets.
Run time: ~2–5 seconds per test (XPBD, ~50 substeps).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

BODY_M = ROOT / "data" / "bodies" / "makehuman_male_M.ply"
PATTERN_M = ROOT / "data" / "patterns" / "tshirt_size_M.json"
MANIFEST_M = ROOT / "data" / "seam_manifests" / "tshirt_size_M_seam_manifest.json"
FABRIC_ID = "cotton_jersey_default"


@pytest.fixture(scope="module")
def verdict_m():
    """Run M tshirt on M body once; cache for the module."""
    from pipeline import run_fit_check
    return run_fit_check(
        body_mesh_path=str(BODY_M),
        pattern_path=str(PATTERN_M),
        seam_manifest_path=str(MANIFEST_M),
        fabric_id=FABRIC_ID,
        backend="cpu",
    )


class TestRunFitCheckBasic:
    def test_returns_dict(self, verdict_m):
        assert isinstance(verdict_m, dict)

    def test_has_fit_key(self, verdict_m):
        assert "fit" in verdict_m

    def test_m_on_m_fits(self, verdict_m):
        """M-size tshirt on M-size body must pass."""
        assert verdict_m["fit"] is True

    def test_has_strain_map(self, verdict_m):
        assert "strain_map" in verdict_m
        assert len(verdict_m["strain_map"]) > 0

    def test_has_verdict_id(self, verdict_m):
        vid = verdict_m.get("verdict_id", "")
        assert vid.startswith("vrd_"), f"verdict_id={vid!r}"
        assert len(vid) == len("vrd_") + 12

    def test_body_source_synthetic(self, verdict_m):
        assert verdict_m.get("body_source") == "synthetic_mannequin"

    def test_chest_front_positive(self, verdict_m):
        """Garment should be larger than body at chest_front → positive delta."""
        regions = {r["region"]: r["delta_mm"] for r in verdict_m["strain_map"]}
        assert regions["chest_front"] > 0.0

    def test_all_regions_non_red(self, verdict_m):
        """No region should be red for a well-fitting size pair."""
        red = [r for r in verdict_m["strain_map"] if r["severity"] == "red"]
        assert not red, f"Red regions: {[r['region'] for r in red]}"


class TestRunFitCheckErrors:
    def test_missing_body_file_raises(self):
        from pipeline import run_fit_check
        with pytest.raises(FileNotFoundError, match="Body mesh"):
            run_fit_check(
                body_mesh_path="/nonexistent/body.ply",
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
            )

    def test_missing_pattern_raises(self):
        from pipeline import run_fit_check
        with pytest.raises(FileNotFoundError, match="Pattern"):
            run_fit_check(
                body_mesh_path=str(BODY_M),
                pattern_path="/nonexistent/pattern.json",
                seam_manifest_path=str(MANIFEST_M),
            )

    def test_unknown_fabric_raises(self):
        from pipeline import run_fit_check
        with pytest.raises(ValueError, match="Unknown fabric_id"):
            run_fit_check(
                body_mesh_path=str(BODY_M),
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                fabric_id="nonexistent_fabric",
            )

    def test_unknown_backend_raises(self):
        from pipeline import run_fit_check
        with pytest.raises(ValueError, match="Unknown backend"):
            run_fit_check(
                body_mesh_path=str(BODY_M),
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                backend="turbo",
            )

    def test_both_body_sources_raises(self):
        from pipeline import run_fit_check
        with pytest.raises(ValueError, match="not both"):
            run_fit_check(
                body_mesh_path=str(BODY_M),
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                anny_measurements={"height_cm": 180, "chest_cm": 96, "waist_cm": 80,
                                    "hips_cm": 95, "inseam_cm": 82, "shoulder_width_cm": 44},
            )

    def test_neither_body_source_raises(self):
        from pipeline import run_fit_check
        with pytest.raises(ValueError, match="must be provided"):
            run_fit_check(
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
            )

    def test_warp_unavailable_raises_import(self):
        """warp-lang is not installed locally — must raise ImportError, not crash."""
        from pipeline import run_fit_check
        with pytest.raises(ImportError):
            run_fit_check(
                body_mesh_path=str(BODY_M),
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                backend="warp",
            )


class TestRunBatchFitCheck:
    def test_batch_returns_list(self):
        from pipeline import run_batch_fit_check
        results = run_batch_fit_check(
            body_mesh_path=str(BODY_M),
            pattern_paths=[str(PATTERN_M)],
            seam_manifest_paths=[str(MANIFEST_M)],
            fabric_id=FABRIC_ID,
        )
        assert isinstance(results, list)
        assert len(results) == 1

    def test_batch_single_result_matches_single(self, verdict_m):
        from pipeline import run_batch_fit_check
        results = run_batch_fit_check(
            body_mesh_path=str(BODY_M),
            pattern_paths=[str(PATTERN_M)],
            seam_manifest_paths=[str(MANIFEST_M)],
            fabric_id=FABRIC_ID,
        )
        assert results[0]["fit"] == verdict_m["fit"]

    def test_batch_mismatched_lengths_raises(self):
        from pipeline import run_batch_fit_check
        with pytest.raises(ValueError, match="same length"):
            run_batch_fit_check(
                body_mesh_path=str(BODY_M),
                pattern_paths=[str(PATTERN_M), str(PATTERN_M)],
                seam_manifest_paths=[str(MANIFEST_M)],
            )
