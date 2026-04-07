from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cli


_VERDICT = {
    "fit": True,
    "strain_map": [{"region": "waist", "delta_mm": 3.5, "severity": "green"}],
}


def test_fit_check_dxf_cli_calls_pipeline_and_prints_json(capsys: pytest.CaptureFixture[str]):
    argv = [
        "forma",
        "fit-check-dxf",
        "--body",
        "/fake/body.ply",
        "--dxf",
        "/fake/pattern.dxf",
    ]
    with patch.object(sys, "argv", argv):
        with patch("cli.run_fit_check_dxf", return_value=_VERDICT) as mock_run:
            cli.main()

    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload["fit"] is True
    assert "fit=True" in out.err
    _, kwargs = mock_run.call_args
    assert kwargs["body_mesh_path"] == "/fake/body.ply"
    assert kwargs["dxf_path"] == "/fake/pattern.dxf"


def test_fit_check_dxf_cli_file_error_exits_2(capsys: pytest.CaptureFixture[str]):
    argv = [
        "forma",
        "fit-check-dxf",
        "--body",
        "/fake/body.ply",
        "--dxf",
        "/missing/pattern.dxf",
    ]
    with patch.object(sys, "argv", argv):
        with patch("cli.run_fit_check_dxf", side_effect=FileNotFoundError("DXF pattern not found")):
            with pytest.raises(SystemExit) as exc:
                cli.main()

    out = capsys.readouterr()
    assert exc.value.code == 2
    assert "DXF pattern not found" in out.err
