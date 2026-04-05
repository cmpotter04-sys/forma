"""
Pattern Maker — AC-2

Loads GarmentCode JSON pattern files directly (no pygarment dependency).
Returns a structured dict consumed by the Tailor seam converter.

GarmentCode JSON structure:
  pattern.panels: dict[panel_name -> {vertices, edges, translation, rotation}]
  pattern.stitches: list of [{panel, edge}, {panel, edge}] pairs
  properties.units_in_meter: int (100 = coords in cm, multiply *10 for mm)
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional

# units_in_meter = 100 means 1 unit = 1 cm → multiply by 10 to get mm
_MM_PER_UNIT = 10.0


class PatternLoadError(Exception):
    """Raised when a GarmentCode JSON fails schema validation."""


def _eval_quadratic_bezier(
    p0: list[float],
    p1: list[float],
    p2: list[float],
    n_samples: int,
) -> list[list[float]]:
    """Sample n_samples points on a quadratic Bézier (inclusive of p2, exclusive of p0)."""
    pts = []
    for i in range(1, n_samples + 1):
        t = i / n_samples
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append([x, y])
    return pts


def _eval_cubic_bezier(
    p0: list[float],
    p1: list[float],
    p2: list[float],
    p3: list[float],
    n_samples: int,
) -> list[list[float]]:
    """Sample n_samples points on a cubic Bézier (inclusive of p3, exclusive of p0)."""
    pts = []
    for i in range(1, n_samples + 1):
        t = i / n_samples
        mt = 1 - t
        x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append([x, y])
    return pts


def _resolve_control_point(
    p0: list[float],
    p1: list[float],
    relative_param: list[float],
) -> list[float]:
    """
    GarmentCode stores Bézier control points as relative parameters [t_along, t_perp].
    t_along: fraction along the chord from p0→p1
    t_perp: fraction of chord length, perpendicular to chord

    Returns absolute control point coordinates.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]

    # Unit vectors along and perpendicular to chord
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular (rotated 90° CCW)

    t_along, t_perp = relative_param
    cx = p0[0] + t_along * dx + t_perp * length * px
    cy = p0[1] + t_along * dy + t_perp * length * py
    return [cx, cy]


def _discretize_edge(
    vertices: list[list[float]],
    edge: dict,
    target_segment_mm: float = 5.0,
    units_to_mm: float = _MM_PER_UNIT,
) -> list[list[float]]:
    """
    Convert a GarmentCode edge (straight or curved) to an ordered list of 2D points.
    Returns points from start vertex up to (but NOT including) the end vertex,
    so callers can chain edges without duplicating vertices.

    target_segment_mm: target spacing in mm for sampling curved edges.
    """
    ep = edge["endpoints"]
    p0 = vertices[ep[0]]
    p1 = vertices[ep[1]]

    curvature = edge.get("curvature")
    if curvature is None:
        # Straight edge — just return start point
        return [p0]

    curve_type = curvature["type"]
    params = curvature["params"]

    # Estimate chord length to choose number of samples
    chord_len_mm = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * units_to_mm
    n_samples = max(4, int(chord_len_mm / target_segment_mm))

    if curve_type == "quadratic":
        cp = _resolve_control_point(p0, p1, params[0])
        pts = _eval_quadratic_bezier(p0, cp, p1, n_samples)
        return [p0] + pts[:-1]  # exclude endpoint (shared with next edge)

    elif curve_type == "cubic":
        cp1 = _resolve_control_point(p0, p1, params[0])
        cp2 = _resolve_control_point(p0, p1, params[1])
        pts = _eval_cubic_bezier(p0, cp1, cp2, p1, n_samples)
        return [p0] + pts[:-1]

    else:
        raise PatternLoadError(f"Unknown curvature type: {curve_type!r}")


def _arc_length_mm(points: list[list[float]], units_to_mm: float = _MM_PER_UNIT) -> float:
    """Compute total arc length in mm of a polyline."""
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(
            points[i][0] - points[i-1][0],
            points[i][1] - points[i-1][1],
        ) * units_to_mm
    return total


def load_pattern(path: str | Path) -> dict:
    """
    Load a GarmentCode JSON pattern file.

    Returns:
        {
          "garment_id": str,
          "source_path": str,
          "units_in_meter": int,
          "panels": {
            panel_name: {
              "vertices": [[x,y], ...],           # raw 2D coords (pattern units)
              "edges": [                           # one per GarmentCode edge
                {
                  "edge_index": int,
                  "endpoints": [int, int],
                  "polyline": [[x,y], ...],        # discretized, excl. final vertex
                  "arc_length_mm": float,
                }
              ],
              "translation": [x,y,z],
              "rotation": [rx,ry,rz],
            }
          },
          "stitches": [                            # raw stitch pairs from JSON
            [{"panel": str, "edge": int}, {"panel": str, "edge": int}]
          ],
          "fabric_id": str,
        }
    """
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    _validate_raw(raw, path)

    units_in_meter = raw.get("properties", {}).get("units_in_meter", 100)
    units_to_mm = (1.0 / units_in_meter) * 1000.0  # 100 → 10mm/unit

    meta = raw.get("_forma_metadata", {})
    garment_id = meta.get("garment_id", path.stem)
    fabric_id = meta.get("fabric_id", "cotton_jersey_default")

    panels = {}
    for panel_name, panel_raw in raw["pattern"]["panels"].items():
        verts = panel_raw["vertices"]
        edges_raw = panel_raw["edges"]

        parsed_edges = []
        for idx, edge in enumerate(edges_raw):
            polyline = _discretize_edge(verts, edge, target_segment_mm=5.0,
                                        units_to_mm=units_to_mm)
            # arc length: polyline + final endpoint
            ep = edge["endpoints"]
            full_poly = polyline + [verts[ep[1]]]
            arc_mm = _arc_length_mm(full_poly, units_to_mm)
            parsed_edges.append({
                "edge_index": idx,
                "endpoints": ep,
                "polyline": polyline,
                "arc_length_mm": arc_mm,
            })

        panels[panel_name] = {
            "vertices": verts,
            "edges": parsed_edges,
            "translation": panel_raw.get("translation", [0.0, 0.0, 0.0]),
            "rotation": panel_raw.get("rotation", [0.0, 0.0, 0.0]),
        }

    return {
        "garment_id": garment_id,
        "source_path": str(path),
        "units_in_meter": units_in_meter,
        "panels": panels,
        "stitches": raw["pattern"].get("stitches", []),
        "fabric_id": fabric_id,
    }


def _validate_raw(raw: dict, path: Path) -> None:
    """Raise PatternLoadError if the JSON is missing required top-level keys."""
    if "pattern" not in raw:
        raise PatternLoadError(f"{path}: missing top-level 'pattern' key")
    pattern = raw["pattern"]
    if "panels" not in pattern:
        raise PatternLoadError(f"{path}: missing pattern.panels")
    if not isinstance(pattern["panels"], dict):
        raise PatternLoadError(f"{path}: pattern.panels must be a dict")
    if not pattern["panels"]:
        raise PatternLoadError(f"{path}: pattern.panels is empty")


def load_all_sizes(patterns_dir: str | Path = "data/patterns") -> dict[str, dict]:
    """
    Load XS, S, M, L, XL T-shirt patterns from the patterns directory.

    Returns: {"XS": pattern_dict, "S": pattern_dict, "M": pattern_dict,
              "L": pattern_dict, "XL": pattern_dict}
    """
    patterns_dir = Path(patterns_dir)
    result = {}
    for size in ("XS", "S", "M", "L", "XL"):
        fname = patterns_dir / f"tshirt_size_{size}.json"
        result[size] = load_pattern(fname)
    return result
