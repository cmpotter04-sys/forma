from __future__ import annotations

import sys
from pathlib import Path

import ezdxf


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from pattern_maker.load_dxf import load_dxf_pattern
from tailor.dxf_seam_inference import (
    DXFSeamInferenceError,
    build_tshirt_dxf_seam_manifest,
    infer_tshirt_edge_roles,
)


def _add_panel(msp, layer: str, points: list[tuple[float, float]]) -> None:
    msp.add_lwpolyline(points, format="xy", close=True, dxfattribs={"layer": layer})


def _write_torso_only_tshirt_dxf(path: Path) -> Path:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    front = [
        (0, 0), (180, 0), (180, 180), (150, 220), (110, 245),
        (90, 230), (70, 245), (30, 220), (0, 180),
    ]
    back = [
        (220, 0), (400, 0), (400, 180), (370, 220), (320, 240),
        (300, 240), (250, 240), (220, 180),
    ]
    left_sleeve = [
        (-120, 120), (-40, 120), (-20, 160), (-40, 210),
        (-80, 230), (-120, 210), (-140, 160),
    ]
    right_sleeve = [
        (440, 120), (520, 120), (540, 160), (520, 210),
        (480, 230), (440, 210), (420, 160),
    ]

    _add_panel(msp, "front_body", front)
    _add_panel(msp, "back_body", back)
    _add_panel(msp, "left_sleeve", left_sleeve)
    _add_panel(msp, "right_sleeve", right_sleeve)
    doc.saveas(path)
    return path


def test_infer_tshirt_edge_roles_finds_torso_side_seams(tmp_path: Path):
    dxf_path = _write_torso_only_tshirt_dxf(tmp_path / "torso_tshirt.dxf")
    pattern = load_dxf_pattern(dxf_path)

    roles = infer_tshirt_edge_roles(pattern)
    role_keys = {(item.panel_id, item.role) for item in roles}

    assert ("front_body", "left_side") in role_keys
    assert ("front_body", "right_side") in role_keys
    assert ("back_body", "left_side") in role_keys
    assert ("back_body", "right_side") in role_keys
    assert ("front_body", "hem") in role_keys
    assert ("back_body", "hem") in role_keys


def test_infer_tshirt_edge_roles_exposes_upper_body_and_sleeve_semantics(tmp_path: Path):
    dxf_path = _write_torso_only_tshirt_dxf(tmp_path / "torso_tshirt.dxf")
    pattern = load_dxf_pattern(dxf_path)

    roles = infer_tshirt_edge_roles(pattern)
    role_keys = {(item.panel_id, item.role) for item in roles}

    assert ("front_body", "neckline") in role_keys
    assert ("front_body", "left_shoulder") in role_keys
    assert ("front_body", "right_shoulder") in role_keys
    assert ("back_body", "neckline") in role_keys
    assert ("left_sleeve", "cuff") in role_keys
    assert ("left_sleeve", "sleeve_cap") in role_keys
    assert ("right_sleeve", "cuff") in role_keys
    assert ("right_sleeve", "sleeve_cap") in role_keys


def test_build_tshirt_dxf_seam_manifest_emits_high_confidence_side_seams(tmp_path: Path):
    dxf_path = _write_torso_only_tshirt_dxf(tmp_path / "torso_tshirt.dxf")
    pattern = load_dxf_pattern(dxf_path)

    manifest = build_tshirt_dxf_seam_manifest(pattern)

    assert manifest["panel_count"] == 4
    assert manifest["validation"]["all_seams_valid"] is True
    assert manifest["validation"]["total_seam_pairs"] >= 2

    seam_pairs = manifest["seam_pairs"]
    assert len(seam_pairs) >= 2
    seam_edges = {(sp["edge_a"], sp["edge_b"]) for sp in seam_pairs}
    assert any("front_body" in a and "back_body" in b for a, b in seam_edges)

    unmatched = set(manifest["validation"]["unmatched_edges"])
    assert any(edge.startswith("left_sleeve_") for edge in unmatched)
    assert any(edge.startswith("right_sleeve_") for edge in unmatched)


def test_build_tshirt_dxf_seam_manifest_can_emit_shoulder_seams(tmp_path: Path):
    dxf_path = _write_torso_only_tshirt_dxf(tmp_path / "torso_tshirt.dxf")
    pattern = load_dxf_pattern(dxf_path)

    manifest = build_tshirt_dxf_seam_manifest(pattern)
    seam_edges = {(sp["edge_a"], sp["edge_b"]) for sp in manifest["seam_pairs"]}

    assert any("left_shoulder" in edge["label"]
               for panel in manifest["panels"]
               for edge in panel["edges"])
    assert any("right_shoulder" in edge["label"]
               for panel in manifest["panels"]
               for edge in panel["edges"])
    assert any(
        "front_body" in a and "back_body" in b
        for a, b in seam_edges
    )


def test_build_tshirt_dxf_seam_manifest_raises_when_all_seams_dropped(tmp_path: Path):
    # Build a DXF where front and back side edges have wildly mismatched lengths
    # (> 2mm tolerance) so all seam candidates get dropped.
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Both panels are simple rectangles (no shoulder geometry → no shoulder
    # role assignments). The only seam candidates are left/right side seams.
    # front is 60mm tall, back is 200mm tall → side-edge diff = 140mm >> 2mm,
    # so all candidates get dropped.
    _add_panel(msp, "front_body", [(0, 0), (180, 0), (180, 60), (0, 60)])
    _add_panel(msp, "back_body",  [(220, 0), (400, 0), (400, 200), (220, 200)])
    _add_panel(msp, "left_sleeve", [
        (-120, 120), (-40, 120), (-20, 160), (-40, 210), (-120, 210),
    ])
    _add_panel(msp, "right_sleeve", [
        (750, 120), (830, 120), (850, 160), (830, 210), (750, 210),
    ])
    dxf_path = tmp_path / "mismatched.dxf"
    doc.saveas(dxf_path)

    pattern = load_dxf_pattern(dxf_path)
    with pytest.raises(DXFSeamInferenceError, match="No valid seam pairs"):
        build_tshirt_dxf_seam_manifest(pattern)
