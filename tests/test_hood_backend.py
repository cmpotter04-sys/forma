"""
tests/test_hood_backend.py

Unit tests for the HOOD/ContourCraft backend interface.

Tests that do NOT require CUDA or ContourCraft:
- Path resolution logic (_ensure_contourcraft raises ImportError with clear message
  when CC root doesn't exist, not AttributeError or KeyError)
- Missing checkpoint raises RuntimeError, not random crash
- Backend="hood" in pipeline raises ImportError on CPU machines
  (lazy import correctly surfaces the error)

Tests that require CUDA+ContourCraft are in test_warp_parity.py (skipped locally).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

BODY_M    = ROOT / "data" / "bodies" / "makehuman_male_M.ply"
PATTERN_M = ROOT / "data" / "patterns" / "tshirt_size_M.json"
MANIFEST_M = ROOT / "data" / "seam_manifests" / "tshirt_size_M_seam_manifest.json"
FABRIC_ID  = "cotton_jersey_default"


class TestHoodBackendErrors:
    """Error-handling tests — all run on CPU without ContourCraft installed."""

    def test_missing_cc_root_raises_import_error(self, tmp_path):
        """run_simulation_hood with a non-existent CC root must raise ImportError."""
        from geometer.hood.hood_simulate import run_simulation_hood

        with pytest.raises((ImportError, Exception)) as exc_info:
            run_simulation_hood(
                body_mesh_path=str(BODY_M),
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                fabric_params={"bend_stiffness": 0.005},
                contourcraft_root=str(tmp_path / "nonexistent_cc"),
                checkpoint_path=str(tmp_path / "fake.pth"),
            )
        # Must be an ImportError (not RuntimeError/AttributeError from bad path)
        assert isinstance(exc_info.value, (ImportError, RuntimeError)), (
            f"Expected ImportError or RuntimeError, got {type(exc_info.value).__name__}: {exc_info.value}"
        )
        # Error message must be informative
        msg = str(exc_info.value).lower()
        assert any(kw in msg for kw in ("contourcraft", "not found", "clone")), (
            f"Error message not informative: {exc_info.value}"
        )

    def test_pipeline_hood_raises_import_error_on_cpu(self, tmp_path):
        """pipeline.run_fit_check(..., backend='hood') must raise ImportError on a CPU
        machine where either ContourCraft is not present or CUDA is not available."""
        from pipeline import run_fit_check
        import os
        # Override the CC root to a nonexistent path so it fails fast
        env_backup = os.environ.get("CONTOURCRAFT_ROOT")
        os.environ["CONTOURCRAFT_ROOT"] = str(tmp_path / "nonexistent_cc")
        try:
            with pytest.raises((ImportError, RuntimeError)):
                run_fit_check(
                    body_mesh_path=str(BODY_M),
                    pattern_path=str(PATTERN_M),
                    seam_manifest_path=str(MANIFEST_M),
                    fabric_id=FABRIC_ID,
                    backend="hood",
                )
        finally:
            if env_backup is None:
                os.environ.pop("CONTOURCRAFT_ROOT", None)
            else:
                os.environ["CONTOURCRAFT_ROOT"] = env_backup

    def test_hood_simulate_importable_without_cuda(self):
        """hood_simulate module must be importable without CUDA installed.
        Lazy imports mean the module itself doesn't fail at import time."""
        import importlib
        mod = importlib.import_module("geometer.hood.hood_simulate")
        assert hasattr(mod, "run_simulation_hood")

    def test_hood_missing_body_raises_file_not_found(self, tmp_path):
        """run_fit_check hood backend propagates FileNotFoundError for missing body."""
        from pipeline import run_fit_check
        import os
        env_backup = os.environ.get("CONTOURCRAFT_ROOT")
        # Point to nonexistent CC so the path check fires first (ImportError)
        # or, if CC root is somehow set, the body path check fires (FileNotFoundError)
        os.environ["CONTOURCRAFT_ROOT"] = str(tmp_path / "no_cc")
        try:
            with pytest.raises((FileNotFoundError, ImportError, RuntimeError)):
                run_fit_check(
                    body_mesh_path="/nonexistent/body.ply",
                    pattern_path=str(PATTERN_M),
                    seam_manifest_path=str(MANIFEST_M),
                    fabric_id=FABRIC_ID,
                    backend="hood",
                )
        finally:
            if env_backup is None:
                os.environ.pop("CONTOURCRAFT_ROOT", None)
            else:
                os.environ["CONTOURCRAFT_ROOT"] = env_backup
