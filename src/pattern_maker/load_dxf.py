"""
src/pattern_maker/load_dxf.py

Deterministic DXF panel ingestion for Forma.

Current scope:
  - extracts closed panel outlines from DXF modelspace
  - supports lightweight DXF panel entities (LWPOLYLINE / POLYLINE)
  - converts outlines into the same canonical pattern dict shape returned by
    load_patterns.load_pattern(), so Tailor and Geometer can eventually share
    the same downstream contract

This is the first step for brand DXF ingestion; seam inference is intentionally
out of scope for this module.
"""

from __future__ import annotations
from pathlib import Path

import ezdxf
import numpy as np

from tailor.panel_labeler import label_tshirt_panels


class DXFLoadError(Exception):
    """Raised when a DXF file cannot be converted into canonical panel records."""


DxfLoadError = DXFLoadError


def _as_closed_vertices_mm(entity) -> list[list[float]]:
    if entity.dxftype() == "LWPOLYLINE":
        if not entity.closed:
            raise DXFLoadError("Encountered open LWPOLYLINE; panel outlines must be closed")
        points = [[float(x), float(y)] for x, y, *_ in entity.get_points("xy")]
    elif entity.dxftype() == "POLYLINE":
        if not entity.is_closed:
            raise DXFLoadError("Encountered open POLYLINE; panel outlines must be closed")
        points = [[float(v.dxf.location.x), float(v.dxf.location.y)] for v in entity.vertices]
    else:
        raise DXFLoadError(f"Unsupported DXF panel entity: {entity.dxftype()}")

    if len(points) < 3:
        raise DXFLoadError("Panel outline must contain at least 3 points")

    if np.allclose(points[0], points[-1]):
        points = points[:-1]

    return points


def _polygon_area(points: list[list[float]]) -> float:
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    return abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _mm_to_pattern_units(points_mm: list[list[float]]) -> list[list[float]]:
    # Forma's existing pattern loader uses units_in_meter=100, i.e. centimetres.
    return [[round(x / 10.0, 6), round(y / 10.0, 6)] for x, y in points_mm]


def _build_edges(vertices: list[list[float]]) -> list[dict]:
    edges = []
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        p0 = vertices[i]
        p1 = vertices[j]
        arc_length_mm = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)) * 10.0)
        edges.append(
            {
                "edge_index": i,
                "endpoints": [i, j],
                "polyline": [p0],
                "arc_length_mm": arc_length_mm,
            }
        )
    return edges


def _translation_for_label(label: str) -> list[float]:
    if label in {"front_body", "back_body"}:
        return [0.0, 0.0, 12.0 if label == "front_body" else -12.0]
    if label == "left_sleeve":
        return [28.0, 4.0, 10.0]
    if label == "right_sleeve":
        return [-28.0, 4.0, 10.0]
    return [0.0, 0.0, 0.0]


def load_dxf(
    path: str | Path,
    *,
    garment_id: str | None = None,
    fabric_id: str = "cotton_jersey_default",
    auto_label: bool = True,
) -> dict:
    """
    Load a DXF file into Forma's canonical pattern dict shape.

    Returns:
      {
        "garment_id": str,
        "source_path": str,
        "source_format": "dxf",
        "units_in_meter": 100,
        "panels": {...},
        "stitches": [],
        "fabric_id": str,
      }
    """
    path = Path(path)
    if not path.exists():
        raise DxfLoadError(f"DXF file not found: {path}")

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    entities = []
    for entity in msp:
        if entity.dxftype() not in {"LWPOLYLINE", "POLYLINE"}:
            continue
        if entity.dxftype() == "LWPOLYLINE" and not entity.closed:
            continue
        if entity.dxftype() == "POLYLINE" and not entity.is_closed:
            continue
        vertices_mm = _as_closed_vertices_mm(entity)
        area_mm2 = _polygon_area(vertices_mm)
        if area_mm2 <= 0.0:
            continue
        entities.append((entity, vertices_mm, area_mm2))

    if not entities:
        raise DXFLoadError(f"No closed DXF polylines found in modelspace: {path}")

    # Deterministic order: larger panels first, then layer name.
    entities.sort(
        key=lambda item: (-item[2], item[0].dxf.layer.lower(), item[0].dxf.handle.lower())
    )

    raw_panels = {}
    for idx, (entity, vertices_mm, _) in enumerate(entities):
        layer = str(entity.dxf.layer or "").strip()
        panel_id = layer if layer and layer != "0" else f"panel_{idx:02d}"
        if panel_id in raw_panels:
            panel_id = f"{panel_id}_{idx:02d}"
        vertices = _mm_to_pattern_units(vertices_mm)
        raw_panels[panel_id] = vertices

    panel_names = label_tshirt_panels(raw_panels) if auto_label else {
        panel_id: panel_id for panel_id in raw_panels
    }

    panels = {}
    for raw_id, vertices in raw_panels.items():
        semantic_id = panel_names.get(raw_id)
        if semantic_id is None:
            # Extra annotation polyline dropped by labeler (not one of the 4
            # canonical shirt panels).
            continue
        panels[semantic_id] = {
            "vertices": vertices,
            "edges": _build_edges(vertices),
            "translation": _translation_for_label(semantic_id),
            "rotation": [0.0, 0.0, 0.0],
            "source_layer": raw_id,
        }

    return {
        "garment_id": garment_id or path.stem,
        "source_path": str(path),
        "source_format": "dxf",
        "units_in_meter": 100,
        "panels": panels,
        "stitches": [],
        "fabric_id": fabric_id,
    }


def load_dxf_pattern(
    path: str | Path,
    garment_id: str | None = None,
    fabric_id: str = "cotton_jersey_default",
    units_in_meter: int = 100,
    auto_label: bool = True,
) -> dict:
    """
    Compatibility wrapper with a loader name that mirrors load_pattern().
    """
    pattern = load_dxf(
        path,
        garment_id=garment_id,
        fabric_id=fabric_id,
        auto_label=auto_label,
    )
    pattern["units_in_meter"] = units_in_meter
    return pattern
