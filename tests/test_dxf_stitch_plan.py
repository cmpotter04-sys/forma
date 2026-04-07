from __future__ import annotations

import sys
from pathlib import Path

import ezdxf


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pattern_maker.load_dxf import load_dxf_pattern
from tailor.dxf_stitch_plan import build_tshirt_stitch_plan


def _add_panel(msp, layer: str, points: list[tuple[float, float]]) -> None:
    msp.add_lwpolyline(points, format="xy", close=True, dxfattribs={"layer": layer})


def _write_fixture(path: Path) -> Path:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    _add_panel(
        msp,
        "front_body",
        [
            (0, 0),
            (180, 0),
            (180, 185),
            (150, 205),
            (120, 232),
            (90, 220),
            (60, 232),
            (30, 205),
            (0, 185),
        ],
    )
    _add_panel(
        msp,
        "back_body",
        [
            (220, 0),
            (400, 0),
            (400, 185),
            (370, 208),
            (340, 228),
            (310, 230),
            (280, 228),
            (250, 208),
            (220, 185),
        ],
    )
    _add_panel(msp, "left_sleeve", [(-120, 120), (-40, 120), (-20, 160), (-40, 210), (-120, 210)])
    _add_panel(msp, "right_sleeve", [(440, 120), (520, 120), (540, 160), (520, 210), (440, 210)])
    doc.saveas(path)
    return path


def test_build_tshirt_stitch_plan_exposes_sleeve_cap_targets(tmp_path: Path):
    dxf_path = _write_fixture(tmp_path / "fixture.dxf")
    pattern = load_dxf_pattern(dxf_path)

    plan = build_tshirt_stitch_plan(pattern)

    assert len(plan["torso_pairs"]) == 4
    assert len(plan["sleeve_caps"]) == 2
    assert len(plan["underarm_pairs"]) == 2

    left_cap = next(item for item in plan["sleeve_caps"] if item["side"] == "left")
    right_cap = next(item for item in plan["sleeve_caps"] if item["side"] == "right")

    assert left_cap["sleeve_cap_edge"] is not None
    assert right_cap["sleeve_cap_edge"] is not None
    assert len(left_cap["attach_to"]) == 2
    assert len(right_cap["attach_to"]) == 2
    assert all(edge is not None for edge in left_cap["attach_to"])
    assert all(edge is not None for edge in right_cap["attach_to"])
