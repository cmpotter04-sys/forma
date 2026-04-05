"""
Tailor — AC-2

Converts a loaded GarmentCode pattern (from load_patterns.py) into
seam_manifest.json format per FORMA_SEAM_MANIFEST_SCHEMA.md.

Pipeline:
  loaded_pattern
    ↓  build_panel_records()   — discretize edges, compute arc lengths
    ↓  build_seam_pairs()      — map GarmentCode stitches → seam pairs
    ↓  validate_seam_pairs()   — enforce tolerance, orphan, duplicate rules
    ↓  resample_seam_edges()   — equalize vertex counts for XPBD constraints
    ↓  write_seam_manifest()   — serialize to JSON
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional


TOLERANCE_MM = 2.0
GATHER_TOLERANCE_MM = 10.0


class SeamValidationError(Exception):
    """Raised when seam manifest validation fails. Never silently proceed."""
    def __init__(self, message: str, failing_seams: Optional[list] = None):
        super().__init__(message)
        self.failing_seams = failing_seams or []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample_polyline(polyline: list[list[float]], n: int) -> list[list[float]]:
    """
    Resample a polyline to exactly n evenly-spaced points using linear interpolation.
    polyline must have at least 2 points.
    """
    if len(polyline) < 2:
        return polyline * n

    # Build cumulative arc-length table
    lengths = [0.0]
    for i in range(1, len(polyline)):
        d = math.hypot(polyline[i][0] - polyline[i-1][0],
                       polyline[i][1] - polyline[i-1][1])
        lengths.append(lengths[-1] + d)
    total = lengths[-1]
    if total < 1e-12:
        return [polyline[0]] * n

    result = []
    for k in range(n):
        target = total * k / (n - 1) if n > 1 else 0.0
        # Binary search for segment
        lo, hi = 0, len(lengths) - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if lengths[mid + 1] < target:
                lo = mid + 1
            else:
                hi = mid
        seg_start = lengths[lo]
        seg_end = lengths[lo + 1]
        seg_len = seg_end - seg_start
        if seg_len < 1e-12:
            result.append(polyline[lo])
        else:
            t = (target - seg_start) / seg_len
            p0 = polyline[lo]
            p1 = polyline[lo + 1]
            result.append([p0[0] + t * (p1[0] - p0[0]),
                           p0[1] + t * (p1[1] - p0[1])])
    return result


def _edge_full_polyline(panel: dict, edge_idx: int) -> list[list[float]]:
    """Return full polyline for an edge (including final endpoint)."""
    edge = panel["edges"][edge_idx]
    verts = panel["vertices"]
    ep = edge["endpoints"]
    return edge["polyline"] + [verts[ep[1]]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_seam_manifest(pattern: dict) -> dict:
    """
    Convert a loaded GarmentCode pattern dict to seam_manifest format.

    Raises SeamValidationError if any validation rule fails.
    Returns the manifest dict (not yet written to disk).
    """
    garment_id = pattern["garment_id"]
    panels_raw = pattern["panels"]
    stitches = pattern["stitches"]

    # --- 1. Build panel records -------------------------------------------
    panels_out = []
    # Map edge_id → arc_length_mm for quick lookup
    edge_arc: dict[str, float] = {}
    # Map edge_id → full polyline (for resampling)
    edge_polylines: dict[str, list[list[float]]] = {}

    for panel_name, panel in panels_raw.items():
        edges_out = []
        for edge in panel["edges"]:
            idx = edge["edge_index"]
            edge_id = f"{panel_name}_e{idx}"
            arc_mm = edge["arc_length_mm"]
            full_poly = _edge_full_polyline(panel, idx)
            # vertex indices are panel-local; use polyline point count as proxy
            vert_indices = list(range(len(full_poly)))

            edges_out.append({
                "edge_id": edge_id,
                "vertices": vert_indices,
                "arc_length_mm": round(arc_mm, 3),
                "label": _infer_label(panel_name, idx, len(panel["edges"])),
            })
            edge_arc[edge_id] = arc_mm
            edge_polylines[edge_id] = full_poly

        vertex_count = sum(len(e["polyline"]) for e in panel["edges"]) + 1
        panels_out.append({
            "panel_id": panel_name,
            "vertex_count": vertex_count,
            "edge_count": len(panel["edges"]),
            "edges": edges_out,
        })

    # --- 2. Build seam pairs from GarmentCode stitches --------------------
    seam_pairs = []
    seen_edges: set[str] = set()

    for stitch_idx, stitch in enumerate(stitches):
        if len(stitch) != 2:
            raise SeamValidationError(
                f"Stitch {stitch_idx} does not have exactly 2 endpoints: {stitch}"
            )
        a, b = stitch
        edge_a_id = f"{a['panel']}_e{a['edge']}"
        edge_b_id = f"{b['panel']}_e{b['edge']}"

        # Validate edge references exist
        if edge_a_id not in edge_arc:
            raise SeamValidationError(
                f"Stitch {stitch_idx}: edge_a={edge_a_id!r} not found in panels",
                failing_seams=[edge_a_id],
            )
        if edge_b_id not in edge_arc:
            raise SeamValidationError(
                f"Stitch {stitch_idx}: edge_b={edge_b_id!r} not found in panels",
                failing_seams=[edge_b_id],
            )

        # Check for duplicate assignments
        for eid in (edge_a_id, edge_b_id):
            if eid in seen_edges:
                raise SeamValidationError(
                    f"Edge {eid!r} appears in more than one seam pair (duplicate assignment)",
                    failing_seams=[eid],
                )
        seen_edges.add(edge_a_id)
        seen_edges.add(edge_b_id)

        arc_diff = abs(edge_arc[edge_a_id] - edge_arc[edge_b_id])
        stitch_type = _infer_stitch_type(a["panel"], a["edge"])
        tol = GATHER_TOLERANCE_MM if stitch_type == "gather" else TOLERANCE_MM
        valid = arc_diff < tol

        seam_pairs.append({
            "seam_id": f"seam_{stitch_idx + 1:03d}",
            "edge_a": edge_a_id,
            "edge_b": edge_b_id,
            "arc_length_diff_mm": round(arc_diff, 3),
            "valid": valid,
            "stitch_type": stitch_type,
        })

    # --- 3. Validate all seam pairs ----------------------------------------
    failing = [sp for sp in seam_pairs if not sp["valid"]]
    if failing:
        raise SeamValidationError(
            f"{len(failing)} seam pair(s) exceed arc-length tolerance:\n"
            + "\n".join(
                f"  {sp['seam_id']}: {sp['edge_a']} ↔ {sp['edge_b']} "
                f"diff={sp['arc_length_diff_mm']:.2f}mm "
                f"(type={sp['stitch_type']})"
                for sp in failing
            ),
            failing_seams=[sp["seam_id"] for sp in failing],
        )

    # --- 4. Identify unmatched edges --------------------------------------
    all_edge_ids = set(edge_arc.keys())
    unmatched = sorted(all_edge_ids - seen_edges)

    # --- 5. Resample paired edges to equal vertex counts -------------------
    _resample_paired_edges(edge_polylines, seam_pairs)

    # --- 5b. Sync vertices arrays in panels_out to post-resample lengths ----
    # Build a lookup: edge_id → panel index + edge index in panels_out
    _edge_to_panel_edge: dict[str, tuple[int, int]] = {}
    for pi, p in enumerate(panels_out):
        for ei, e in enumerate(p["edges"]):
            _edge_to_panel_edge[e["edge_id"]] = (pi, ei)

    for eid, poly in edge_polylines.items():
        if eid not in _edge_to_panel_edge:
            continue
        pi, ei = _edge_to_panel_edge[eid]
        panels_out[pi]["edges"][ei]["vertices"] = list(range(len(poly)))

    # --- 6. Build validation record ----------------------------------------
    max_diff = max((sp["arc_length_diff_mm"] for sp in seam_pairs), default=0.0)
    validation = {
        "all_seams_valid": all(sp["valid"] for sp in seam_pairs),
        "max_arc_length_diff_mm": round(max_diff, 3),
        "tolerance_mm": TOLERANCE_MM,
        "unmatched_edges": unmatched,
        "total_seam_pairs": len(seam_pairs),
    }

    return {
        "garment_id": garment_id,
        "panel_count": len(panels_out),
        "panels": panels_out,
        "seam_pairs": seam_pairs,
        "validation": validation,
    }


def write_seam_manifest(manifest: dict, out_path: str | Path) -> None:
    """Serialize manifest to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_and_validate_manifest(path: str | Path) -> dict:
    """
    Load a seam_manifest.json from disk and re-validate it.
    Raises SeamValidationError if any rule fails.
    """
    with open(path) as f:
        manifest = json.load(f)
    _validate_manifest_schema(manifest, path)
    return manifest


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _infer_label(panel_name: str, edge_idx: int, total_edges: int) -> str:
    """Heuristic label based on panel name and edge position."""
    name = panel_name.lower()
    if "sleeve" in name:
        if edge_idx == 0:
            return "sleeve_cap"
        elif edge_idx == total_edges - 1:
            return "sleeve_hem"
        return f"sleeve_seam_{edge_idx}"
    # Torso panels: edges 0=bottom_hem, last=neckline, others=side/shoulder seams
    if edge_idx == 0:
        return "bottom_hem"
    if edge_idx == total_edges - 1:
        return "neckline"
    if "front" in name or "ftorso" in name:
        return f"front_seam_{edge_idx}"
    if "back" in name or "btorso" in name:
        return f"back_seam_{edge_idx}"
    return f"edge_{edge_idx}"


def _infer_stitch_type(panel_name: str, edge_idx: int) -> str:
    """Sleeve armhole attachments are gather seams; everything else is standard."""
    if "sleeve" in panel_name.lower():
        return "gather"
    return "standard"


def _resample_paired_edges(
    edge_polylines: dict[str, list[list[float]]],
    seam_pairs: list[dict],
) -> None:
    """
    Ensure each paired edge has equal vertex count by resampling shorter one.
    Mutates edge_polylines in-place.
    """
    for sp in seam_pairs:
        poly_a = edge_polylines[sp["edge_a"]]
        poly_b = edge_polylines[sp["edge_b"]]
        n = max(len(poly_a), len(poly_b))
        if len(poly_a) != n:
            edge_polylines[sp["edge_a"]] = _resample_polyline(poly_a, n)
        if len(poly_b) != n:
            edge_polylines[sp["edge_b"]] = _resample_polyline(poly_b, n)


def _validate_manifest_schema(manifest: dict, path) -> None:
    """Re-validate a loaded manifest dict against all schema rules."""
    required_top = ["garment_id", "panel_count", "panels", "seam_pairs", "validation"]
    for field in required_top:
        if field not in manifest:
            raise SeamValidationError(f"{path}: missing field '{field}'")

    # Rule: panel_count must match actual panels array length
    actual_panel_count = len(manifest["panels"])
    if manifest["panel_count"] != actual_panel_count:
        raise SeamValidationError(
            f"{path}: panel_count={manifest['panel_count']} does not match "
            f"actual panels array length={actual_panel_count}"
        )

    # Rule: total_seam_pairs must match actual seam_pairs array length
    actual_pair_count = len(manifest["seam_pairs"])
    stated_pair_count = manifest["validation"].get("total_seam_pairs")
    if stated_pair_count != actual_pair_count:
        raise SeamValidationError(
            f"{path}: validation.total_seam_pairs={stated_pair_count} does not match "
            f"actual seam_pairs array length={actual_pair_count}"
        )

    # Build edge id → vertex count map; validate edge_id format
    import re as _re
    edge_ids: dict[str, int] = {}
    for panel in manifest["panels"]:
        panel_id = panel["panel_id"]
        # Validate required panel fields
        for pf in ("panel_id", "vertex_count", "edge_count", "edges"):
            if pf not in panel:
                raise SeamValidationError(f"{path}: panel missing field '{pf}'")
        for edge in panel["edges"]:
            # Validate required edge fields
            for ef in ("edge_id", "vertices", "arc_length_mm", "label"):
                if ef not in edge:
                    raise SeamValidationError(
                        f"{path}: edge in panel '{panel_id}' missing field '{ef}'"
                    )
            eid = edge["edge_id"]
            # Rule: edge_id must follow {panel_id}_e{index} format
            if not _re.fullmatch(r".+_e\d+", eid):
                raise SeamValidationError(
                    f"{path}: edge_id {eid!r} does not match expected format "
                    f"'{{panel_id}}_e{{index}}'"
                )
            edge_ids[eid] = len(edge["vertices"])

    seen: set[str] = set()
    failing = []
    computed_max_diff = 0.0
    for sp in manifest["seam_pairs"]:
        # Validate required seam_pair fields
        for sf in ("seam_id", "edge_a", "edge_b", "arc_length_diff_mm", "valid", "stitch_type"):
            if sf not in sp:
                raise SeamValidationError(f"{path}: seam pair missing field '{sf}'")

        # Rule: seam_id must follow seam_NNN format
        if not _re.fullmatch(r"seam_\d{3,}", sp["seam_id"]):
            raise SeamValidationError(
                f"{path}: seam_id {sp['seam_id']!r} does not match expected format 'seam_NNN'"
            )

        # Rule 2: no orphan references
        if sp["edge_a"] not in edge_ids:
            raise SeamValidationError(f"Orphan edge reference: {sp['edge_a']!r}")
        if sp["edge_b"] not in edge_ids:
            raise SeamValidationError(f"Orphan edge reference: {sp['edge_b']!r}")

        # Rule 3: no duplicate assignments
        for eid in (sp["edge_a"], sp["edge_b"]):
            if eid in seen:
                raise SeamValidationError(f"Duplicate edge in seam pairs: {eid!r}")
            seen.add(eid)

        # Rule 1: arc length tolerance
        tol = GATHER_TOLERANCE_MM if sp["stitch_type"] == "gather" else TOLERANCE_MM
        if sp["arc_length_diff_mm"] >= tol:
            failing.append(sp["seam_id"])

        # Rule: stitch_type must be a known value
        if sp["stitch_type"] not in ("standard", "gather"):
            raise SeamValidationError(
                f"{path}: seam {sp['seam_id']!r} has unknown stitch_type "
                f"{sp['stitch_type']!r} (must be 'standard' or 'gather')"
            )

        # Rule 5: vertex count compatibility between paired edges
        vcount_a = edge_ids[sp["edge_a"]]
        vcount_b = edge_ids[sp["edge_b"]]
        if vcount_a != vcount_b:
            raise SeamValidationError(
                f"{path}: seam {sp['seam_id']!r} vertex count mismatch — "
                f"{sp['edge_a']} has {vcount_a} vertices, "
                f"{sp['edge_b']} has {vcount_b} vertices"
            )

        diff = sp["arc_length_diff_mm"]
        if diff > computed_max_diff:
            computed_max_diff = diff

    if failing:
        raise SeamValidationError(
            f"Seams exceed tolerance: {failing}", failing_seams=failing
        )

    # Rule: all_seams_valid must be consistent with actual pair states
    actual_all_valid = len(failing) == 0
    if manifest["validation"].get("all_seams_valid") != actual_all_valid:
        raise SeamValidationError(
            f"{path}: validation.all_seams_valid={manifest['validation']['all_seams_valid']} "
            f"is inconsistent with actual seam pair states (expected {actual_all_valid})"
        )

    # Rule: max_arc_length_diff_mm must be consistent (within floating-point rounding)
    stated_max = manifest["validation"].get("max_arc_length_diff_mm", 0.0)
    if abs(stated_max - computed_max_diff) > 0.001:
        raise SeamValidationError(
            f"{path}: validation.max_arc_length_diff_mm={stated_max} does not match "
            f"computed maximum={computed_max_diff:.3f}"
        )
