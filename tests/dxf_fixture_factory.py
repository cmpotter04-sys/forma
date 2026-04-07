from __future__ import annotations

from pathlib import Path

import ezdxf


def add_panel(msp, layer: str, points: list[tuple[float, float]]) -> None:
    msp.add_lwpolyline(points, format="xy", close=True, dxfattribs={"layer": layer})


def write_repo_aligned_tshirt_dxf(path: Path) -> Path:
    """
    Write a DXF fixture whose panel topology matches the current T-shirt intake
    heuristics and yields a full shirt stitch plan.

    The geometry is intentionally balanced rather than brand-specific: the DXF
    intake path is still heuristic, so the fixture should reflect the topology
    we expect the current rules to support reliably.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    front_body = [
        (0.0, 0.0),
        (180.0, 0.0),
        (180.0, 185.0),
        (150.0, 205.0),
        (120.0, 232.0),
        (90.0, 220.0),
        (60.0, 232.0),
        (30.0, 205.0),
        (0.0, 185.0),
    ]
    back_body = [
        (220.0, 0.0),
        (400.0, 0.0),
        (400.0, 185.0),
        (370.0, 208.0),
        (340.0, 228.0),
        (310.0, 230.0),
        (280.0, 228.0),
        (250.0, 208.0),
        (220.0, 185.0),
    ]
    left_sleeve = [
        (-120.0, 120.0),
        (-40.0, 120.0),
        (-20.0, 160.0),
        (-40.0, 210.0),
        (-120.0, 210.0),
    ]
    right_sleeve = [
        (440.0, 120.0),
        (520.0, 120.0),
        (540.0, 160.0),
        (520.0, 210.0),
        (440.0, 210.0),
    ]

    add_panel(msp, "front_body", front_body)
    add_panel(msp, "back_body", back_body)
    add_panel(msp, "left_sleeve", left_sleeve)
    add_panel(msp, "right_sleeve", right_sleeve)
    doc.saveas(path)
    return path


def write_messy_tshirt_dxf(path: Path) -> Path:
    """
    Write a DXF fixture with 6 closed polylines: the standard 4 shirt panels
    plus 2 tiny annotation rectangles (simulating grainline arrows or notch
    marks often present in real brand DXFs).

    The 4 shirt panels are geometrically identical to write_repo_aligned_tshirt_dxf
    so the same seam-inference rules apply. The extras sit on a layer named
    "annotation" and are too small to be confused with real panels.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    front_body = [
        (0.0, 0.0),
        (180.0, 0.0),
        (180.0, 185.0),
        (150.0, 205.0),
        (120.0, 232.0),
        (90.0, 220.0),
        (60.0, 232.0),
        (30.0, 205.0),
        (0.0, 185.0),
    ]
    back_body = [
        (220.0, 0.0),
        (400.0, 0.0),
        (400.0, 185.0),
        (370.0, 208.0),
        (340.0, 228.0),
        (310.0, 230.0),
        (280.0, 228.0),
        (250.0, 208.0),
        (220.0, 185.0),
    ]
    left_sleeve = [
        (-120.0, 120.0),
        (-40.0, 120.0),
        (-20.0, 160.0),
        (-40.0, 210.0),
        (-120.0, 210.0),
    ]
    right_sleeve = [
        (440.0, 120.0),
        (520.0, 120.0),
        (540.0, 160.0),
        (520.0, 210.0),
        (440.0, 210.0),
    ]

    add_panel(msp, "front_body", front_body)
    add_panel(msp, "back_body", back_body)
    add_panel(msp, "left_sleeve", left_sleeve)
    add_panel(msp, "right_sleeve", right_sleeve)

    # Tiny grainline/notch rectangles — should be silently dropped by labeler.
    add_panel(msp, "annotation", [(600.0, 10.0), (610.0, 10.0), (610.0, 15.0), (600.0, 15.0)])
    add_panel(msp, "annotation", [(620.0, 10.0), (625.0, 10.0), (625.0, 12.0), (620.0, 12.0)])

    doc.saveas(path)
    return path
