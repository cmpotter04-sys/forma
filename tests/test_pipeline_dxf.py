from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_fit_check_dxf
from tests.dxf_fixture_factory import write_repo_aligned_tshirt_dxf


BODY_M = ROOT / "data" / "bodies" / "makehuman_male_M.ply"


def test_run_fit_check_dxf_returns_verdict_schema(tmp_path: Path):
    dxf_path = write_repo_aligned_tshirt_dxf(tmp_path / "fixture.dxf")

    verdict = run_fit_check_dxf(
        dxf_path=str(dxf_path),
        body_mesh_path=str(BODY_M),
        backend="cpu",
    )

    assert isinstance(verdict, dict)
    assert "fit" in verdict
    assert "strain_map" in verdict
    assert "verdict_id" in verdict
    assert verdict["body_source"] == "synthetic_mannequin"


def test_run_fit_check_dxf_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="DXF pattern"):
        run_fit_check_dxf(
            dxf_path="/nonexistent/pattern.dxf",
            body_mesh_path=str(BODY_M),
        )
