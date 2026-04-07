from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pattern_maker.load_patterns import load_pattern
from tests.dxf_fixture_factory import write_repo_aligned_tshirt_dxf
from tailor.dxf_bundle import write_dxf_pipeline_bundle
from tailor.seam_converter import load_and_validate_manifest


def test_write_dxf_pipeline_bundle_produces_loader_compatible_files(tmp_path: Path):
    dxf_path = write_repo_aligned_tshirt_dxf(tmp_path / "fixture.dxf")

    pattern_path, manifest_path = write_dxf_pipeline_bundle(dxf_path, tmp_path / "bundle")

    pattern = load_pattern(pattern_path)
    manifest = load_and_validate_manifest(manifest_path)

    assert pattern_path.exists()
    assert manifest_path.exists()
    assert set(pattern["panels"]) == {"front_ftorso", "back_btorso", "left_sleeve", "right_sleeve"}
    assert manifest["panel_count"] == 4
    assert manifest["validation"]["total_seam_pairs"] >= 4
    assert "inference_complete" in manifest["validation"]
    assert "dropped_seam_candidates" in manifest["validation"]
    assert all(sp["valid"] for sp in manifest["seam_pairs"])
