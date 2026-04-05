"""
Pattern Maker — AC-2

Loads GarmentCode JSON pattern files directly (no pygarment dependency).
Returns a structured dict consumed by the Tailor seam converter.

GarmentCode JSON structure:
  pattern.panels: dict[panel_name -> {vertices, edges, translation, rotation}]
  pattern.stitches: list of [{panel, edge}, {panel, edge}] pairs
  properties.units_in_meter: int (100 = coords in cm, multiply *10 for mm)

Multi-garment support:
  Use load_garment(garment_type, size) to load any supported garment.
  SUPPORTED_GARMENTS maps garment type → {sizes, pattern_prefix, patterns_dir}.
  Sized garments (tshirt) live in data/patterns/.
  Single-size specification garments (dress, shirt, hoody, jumpsuit) live in
  data/garmentcode_assets/Patterns/ and expose only "default" as their size.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

# units_in_meter = 100 means 1 unit = 1 cm → multiply by 10 to get mm
_MM_PER_UNIT = 10.0

# ---------------------------------------------------------------------------
# Supported garment registry
#
# Each entry describes where to find pattern files and which sizes are valid.
#
# "pattern_prefix"  : filename prefix; the size string is appended before ".json"
#                     e.g. "tshirt_size_" + "M" → "tshirt_size_M.json"
# "sizes"           : list of valid size strings
# "patterns_dir"    : path relative to the repo root where the files live
#
# For single-spec garments (dress, shirt, etc.) the only valid size is "default"
# and the full filename is just pattern_prefix + "default" (no suffix appended) —
# see load_garment() for the exact resolution logic.
# ---------------------------------------------------------------------------
SUPPORTED_GARMENTS: dict[str, dict] = {
    "tshirt": {
        "sizes": ["XS", "S", "M", "L", "XL"],
        "pattern_prefix": "tshirt_size_",
        "patterns_dir": "data/patterns",
    },
    # ---- Single-size specification garments (GarmentCode mean bodies) -------
    # These ship as one reference-size JSON in data/garmentcode_assets/Patterns/.
    # Per-size variants are deferred until GarmentCode parametric generation is
    # wired into the pipeline (Phase 2 Stage 3+).
    "dress": {
        "sizes": ["default"],
        "pattern_prefix": "dress_pencil_specification",
        "patterns_dir": "data/garmentcode_assets/Patterns",
    },
    "shirt": {
        "sizes": ["default"],
        "pattern_prefix": "shirt_mean_specification",
        "patterns_dir": "data/garmentcode_assets/Patterns",
    },
    "hoody": {
        "sizes": ["default"],
        "pattern_prefix": "hoody_mean_specification",
        "patterns_dir": "data/garmentcode_assets/Patterns",
    },
    "jumpsuit": {
        "sizes": ["default"],
        "pattern_prefix": "js_mean_all_specification",
        "patterns_dir": "data/garmentcode_assets/Patterns",
    },
    # ---- Hand-crafted sized fixtures (data/patterns/) -----------------------
    "tank_top": {
        "sizes": ["M"],
        "pattern_prefix": "tank_top_size_",
        "patterns_dir": "data/patterns",
    },
    "dress_v1": {
        "sizes": ["M"],
        "pattern_prefix": "dress_size_",
        "patterns_dir": "data/patterns",
    },
    "trousers": {
        "sizes": ["M"],
        "pattern_prefix": "trousers_size_",
        "patterns_dir": "data/patterns",
    },
}


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


def load_garment(
    garment_type: str,
    size: str,
    repo_root: str | Path | None = None,
) -> dict:
    """
    Load the pattern for any supported garment type + size combination.

    Parameters
    ----------
    garment_type : str
        Must be a key in SUPPORTED_GARMENTS (e.g. "tshirt", "dress", "shirt").
    size : str
        Must be in SUPPORTED_GARMENTS[garment_type]["sizes"].
        For single-size spec garments the only valid value is "default".
    repo_root : str | Path | None
        Absolute path to the repository root.  If None, the path is resolved
        relative to this source file's location (../../.. from src/pattern_maker/).

    Returns
    -------
    dict — same structure as load_pattern().

    Raises
    ------
    PatternLoadError
        If garment_type is unsupported, size is invalid, or the pattern file
        does not exist or fails schema validation.
    """
    if garment_type not in SUPPORTED_GARMENTS:
        supported = ", ".join(sorted(SUPPORTED_GARMENTS))
        raise PatternLoadError(
            f"Unsupported garment type {garment_type!r}. "
            f"Supported types: {supported}"
        )

    spec = SUPPORTED_GARMENTS[garment_type]
    valid_sizes = spec["sizes"]
    if size not in valid_sizes:
        raise PatternLoadError(
            f"Invalid size {size!r} for garment {garment_type!r}. "
            f"Valid sizes: {valid_sizes}"
        )

    if repo_root is None:
        # src/pattern_maker/load_patterns.py → go up two levels to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent

    repo_root = Path(repo_root)
    patterns_dir = repo_root / spec["patterns_dir"]
    prefix = spec["pattern_prefix"]

    # Single-size spec garments: filename IS the prefix (no size suffix)
    # Sized garments: filename = prefix + size + ".json"
    if valid_sizes == ["default"]:
        fname = patterns_dir / f"{prefix}.json"
    else:
        fname = patterns_dir / f"{prefix}{size}.json"

    if not fname.exists():
        raise PatternLoadError(
            f"Pattern file not found for {garment_type!r} size {size!r}: {fname}"
        )

    return load_pattern(fname)


def load_all_sizes(
    garment_type_or_dir: "str | Path" = "tshirt",
    patterns_dir: "str | Path | None" = None,
) -> dict[str, dict]:
    """
    Load all size variants for a supported garment type.

    Parameters
    ----------
    garment_type_or_dir : str | Path
        Either a garment type key from SUPPORTED_GARMENTS (e.g. "tshirt"),
        OR a Path / str pointing to a patterns directory — the legacy calling
        convention (load_all_sizes("data/patterns")).  When a Path or a string
        that is not a SUPPORTED_GARMENTS key is supplied, the function treats it
        as a patterns directory and loads T-shirt sizes from that directory,
        preserving backwards compatibility.
    patterns_dir : str | Path | None
        Explicit override for the patterns directory.  When supplied together
        with a garment_type string, this directory is used instead of the
        registry default.

    Returns
    -------
    dict mapping size string → pattern dict
    e.g. {"XS": {...}, "S": {...}, "M": {...}, "L": {...}, "XL": {...}}

    Notes
    -----
    Calling load_all_sizes() with no arguments still loads all T-shirt sizes
    from data/patterns/, identical to the Phase 1 behaviour.
    """
    # --- Backwards-compatibility shim ----------------------------------------
    # Old callers pass a Path or a directory string as the first positional arg.
    first = garment_type_or_dir
    if isinstance(first, Path) or (
        isinstance(first, str) and first not in SUPPORTED_GARMENTS
    ):
        # Legacy call: load_all_sizes(patterns_dir)
        base_dir = Path(first)
        spec = SUPPORTED_GARMENTS["tshirt"]
        result = {}
        for size in spec["sizes"]:
            fname = base_dir / f"{spec['pattern_prefix']}{size}.json"
            result[size] = load_pattern(fname)
        return result
    # -------------------------------------------------------------------------

    garment_type = str(first)
    if garment_type not in SUPPORTED_GARMENTS:
        supported = ", ".join(sorted(SUPPORTED_GARMENTS))
        raise PatternLoadError(
            f"Unsupported garment type {garment_type!r}. "
            f"Supported types: {supported}"
        )

    spec = SUPPORTED_GARMENTS[garment_type]

    # Resolve the patterns directory
    if patterns_dir is not None:
        base_dir = Path(patterns_dir)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        base_dir = repo_root / spec["patterns_dir"]

    result = {}
    for size in spec["sizes"]:
        prefix = spec["pattern_prefix"]
        if spec["sizes"] == ["default"]:
            fname = base_dir / f"{prefix}.json"
        else:
            fname = base_dir / f"{prefix}{size}.json"
        result[size] = load_pattern(fname)
    return result
