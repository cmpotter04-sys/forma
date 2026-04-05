"""
src/geometer/garment_assembly.py

Shared garment assembly module extracted from xpbd_simulate.py (Phase 1).
Used by both the CPU (xpbd_simulate) and GPU (warp_simulate) backends.

This module handles:
  - Panel geometry: building 2D outlines, placing in 3D
  - Body-conformal wrapping: radius maps, bilinear interpolation
  - Sleeve placement: armhole identification, cylindrical wrapping
  - Triangulation: Delaunay with interior grid points
  - Seam constraint building: edge index maps, resampling, pairing
  - Garment assembly: full pipeline from pattern + manifest → 3D mesh + constraints
  - Strain ratio computation: post-sim per-region strain analysis

Coordinate system:
  Body: Y-up, metres. Feet at y=0, head at y=1.8m. Facing +Z. Wearer's left = +X.
  Pattern: cm units. GarmentCode translation/rotation fields give 3D placement.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.spatial import Delaunay, KDTree


REQUIRED_REGIONS = [
    "chest_front", "chest_side", "shoulder_left",
    "shoulder_right", "upper_back", "waist",
]


# ---------------------------------------------------------------------------
# Panel geometry helpers
# ---------------------------------------------------------------------------

def build_panel_outline_2d(panel: dict) -> list[list[float]]:
    """
    Chain all edge polylines into a closed 2D polygon.
    Returns list of [x, y] points in cm (no duplicate closing vertex).
    """
    outline: list[list[float]] = []
    for edge in panel["edges"]:
        outline.extend(edge["polyline"])
    return outline


def place_panel_in_3d(
    pts_2d: list[list[float]],
    translation: list[float],
    rotation_deg: list[float],
) -> np.ndarray:
    """
    Apply GarmentCode rotation (rz around Z axis) + translation to 2D panel points.

    GarmentCode 2D panels lie in the local XY plane. Rotation is around the
    Z axis only (rx=ry=0 for all T-shirt panels). Translation (tx, ty, tz)
    places the panel in 3D world space (cm).

    Returns (N, 3) float array in METRES.
    """
    tx, ty, tz = translation
    rz = math.radians(rotation_deg[2])
    cos_rz, sin_rz = math.cos(rz), math.sin(rz)

    pts = np.array(pts_2d, dtype=float)  # (N, 2) in cm
    x2 = pts[:, 0]
    y2 = pts[:, 1]

    # Rotate in XY plane, then translate
    x3 = x2 * cos_rz - y2 * sin_rz + tx
    y3 = x2 * sin_rz + y2 * cos_rz + ty
    z3 = np.full(len(pts), tz)

    return np.column_stack([x3, y3, z3]) / 100.0   # cm → metres


def place_panel_cylindrical(
    pts_2d: np.ndarray,
    panel: dict,
    r_garment_cm: float,
    y_shift_cm: float,
) -> np.ndarray:
    """
    Place a flat torso panel onto a cylinder of radius r_garment.

    The panel's 2D x coordinate (cm) represents arc length around the body.
    It is mapped to angle θ on the cylinder so that adjacent panel seam edges
    land at the same 3D position (zero seam gap).

    Front panels (tz > 0 in GarmentCode):  θ = x_2d / r_garment
    Back  panels (tz < 0 in GarmentCode):  θ = ±π − x_2d / r_garment
      where sign = +1 for left-back (x_2d ≥ 0) and −1 for right-back (x_2d ≤ 0)

    Returns (N, 3) float array in METRES.
    """
    pts = np.array(pts_2d, dtype=float)   # (N, 2) in cm
    x2 = pts[:, 0]
    y2 = pts[:, 1]

    tz = panel["translation"][2]           # positive = front panel
    ty = panel["translation"][1]

    # Height: keep GarmentCode y, just shift up to cover correct body zones
    y3_cm = y2 + ty + y_shift_cm

    # Circumferential mapping → (x_3d, z_3d)
    if tz >= 0:
        # Front panel: arc from center-front (θ=0) outward to side
        theta = x2 / r_garment_cm
    else:
        # Back panel: arc inward from side to center-back (θ=±π)
        # Determine side: positive x_2d → wearer's left (+π), negative → right (−π)
        x_mean = float(np.mean(x2))
        back_sign = 1.0 if x_mean >= 0 else -1.0
        theta = back_sign * math.pi - x2 / r_garment_cm

    x3_cm = r_garment_cm * np.sin(theta)
    z3_cm = r_garment_cm * np.cos(theta)

    return np.column_stack([x3_cm, y3_cm, z3_cm]) / 100.0   # cm → metres


def place_sleeve_near_shoulder(
    pts_2d: np.ndarray,
    panel: dict,
    target_top_y_cm: float = 140.0,
) -> np.ndarray:
    """
    Position a sleeve panel at the correct shoulder height (legacy fallback).

    Returns (N, 3) float array in METRES.
    """
    tx, ty, tz = panel["translation"]
    rz = math.radians(panel["rotation"][2])
    cos_rz, sin_rz = math.cos(rz), math.sin(rz)

    pts = np.array(pts_2d, dtype=float)   # (N, 2) in cm
    x2 = pts[:, 0]
    y2 = pts[:, 1]

    x3_cm = x2 * cos_rz - y2 * sin_rz + tx
    z3_cm = np.full(len(pts), tz)

    y2_max = float(pts[:, 1].max())
    sleeve_y_shift = target_top_y_cm - (y2_max + ty)
    y3_cm = x2 * sin_rz + y2 * cos_rz + ty + sleeve_y_shift

    return np.column_stack([x3_cm, y3_cm, z3_cm]) / 100.0   # cm → metres


def place_sleeve_at_armhole(
    pts_2d: np.ndarray,
    panel: dict,
    armhole_centroid: np.ndarray,
    armhole_radius: float,
    arm_direction: np.ndarray,
    cap_edge_local_indices: list[int],
) -> np.ndarray:
    """
    Place a sleeve panel wrapped around the arm cylinder at the armhole.

    1. Apply GarmentCode rotation to orient the 2D panel
    2. Identify the cap edge direction (circumferential) and sleeve axis (axial)
    3. Project all vertices onto circumferential / axial coordinates
    4. Wrap onto a cylinder aligned with arm_direction, centered at armhole_centroid

    Parameters
    ----------
    pts_2d              : (N, 2) — triangulated panel points in cm
    panel               : GarmentCode panel dict (translation, rotation)
    armhole_centroid    : (3,) — armhole center position in metres
    armhole_radius      : float — arm cylinder radius in metres
    arm_direction       : (3,) — unit vector along arm (+X or -X)
    cap_edge_local_indices : local vertex indices forming the sleeve cap edge

    Returns (N, 3) float in metres.
    """
    rz = math.radians(panel["rotation"][2])
    cos_rz, sin_rz = math.cos(rz), math.sin(rz)

    pts = np.array(pts_2d, dtype=float)  # (N, 2) in cm
    x2 = pts[:, 0]
    y2 = pts[:, 1]

    # Apply GarmentCode rotation (orients the panel shape correctly)
    x_rot = x2 * cos_rz - y2 * sin_rz
    y_rot = x2 * sin_rz + y2 * cos_rz

    # Cap edge in rotated 2D space
    valid_cap = [i for i in cap_edge_local_indices if i < len(pts)]
    if len(valid_cap) < 2:
        # Fallback: use top-most vertices as cap
        valid_cap = list(np.argsort(-y_rot)[:max(2, len(pts) // 4)])

    cap_pts = np.column_stack([x_rot[valid_cap], y_rot[valid_cap]])
    cap_centroid_2d = cap_pts.mean(axis=0)

    # Cap edge direction (circumferential = tangent along the cap)
    cap_dir = cap_pts[-1] - cap_pts[0]
    cap_len = np.linalg.norm(cap_dir)
    if cap_len < 1e-6:
        cap_dir = np.array([1.0, 0.0])
    else:
        cap_dir = cap_dir / cap_len

    # Axial direction = perpendicular to cap edge, pointing toward cuff
    axial_dir = np.array([-cap_dir[1], cap_dir[0]])
    all_centroid_2d = np.array([x_rot.mean(), y_rot.mean()])
    if np.dot(axial_dir, all_centroid_2d - cap_centroid_2d) < 0:
        axial_dir = -axial_dir

    # Project all vertices onto circumferential and axial coordinates
    pts_rot = np.column_stack([x_rot, y_rot])
    rel = pts_rot - cap_centroid_2d
    circ_cm = np.dot(rel, cap_dir)     # circumferential displacement
    axial_cm = np.dot(rel, axial_dir)  # along-arm displacement (positive = toward cuff)

    # Cylindrical wrapping: theta = circ / r_arm
    r_arm_cm = armhole_radius * 100.0
    if r_arm_cm < 1.0:
        r_arm_cm = 5.0  # fallback minimum
    theta = circ_cm / r_arm_cm

    # Determine theta offset based on front/back panel
    # Front panels (tz > 0): wrap around front of arm (theta_offset = 0)
    # Back panels (tz < 0): wrap around back of arm (theta_offset = pi)
    tz = panel["translation"][2]
    theta_offset = 0.0 if tz >= 0 else math.pi

    # 3D positions: arm cylinder in YZ plane, axis along arm_direction
    arm_dir = arm_direction / (np.linalg.norm(arm_direction) + 1e-10)
    r_arm_m = armhole_radius

    positions = np.empty((len(pts), 3), dtype=float)
    for i in range(len(pts)):
        angle = theta[i] + theta_offset
        along = axial_cm[i] / 100.0  # cm to m
        # Position = armhole center + along arm + cylinder wrapping in YZ
        positions[i, 0] = armhole_centroid[0] + along * arm_dir[0]
        positions[i, 1] = armhole_centroid[1] + r_arm_m * math.sin(angle)
        positions[i, 2] = armhole_centroid[2] + r_arm_m * math.cos(angle)

    return positions


# ---------------------------------------------------------------------------
# Body-conformal wrapping helpers
# ---------------------------------------------------------------------------

def measure_body_radii_at_height(
    body_vertices: np.ndarray,
    y_center: float,
    centroid: np.ndarray,
    ref_angles: np.ndarray,
    y_band: float = 0.02,
) -> tuple[np.ndarray, float]:
    """
    Measure the body's surface radius at each reference angle using the
    actual outermost vertex (not convex hull interpolation).

    Returns (radii, perimeter).  radii[k] = max distance from centroid
    among body vertices whose angle from centroid is near ref_angles[k].
    """
    # Arm gap detection for x_max
    band_mask = np.abs(body_vertices[:, 1] - y_center) < y_band
    x_max = 0.20
    if np.sum(band_mask) >= 10:
        abs_x = np.sort(np.abs(body_vertices[band_mask, 0]))
        gaps = np.diff(abs_x)
        for i, g in enumerate(gaps):
            if abs_x[i] > 0.10 and g > 0.015:
                x_max = float(abs_x[i]) + 0.005
                break

    mask = band_mask & (np.abs(body_vertices[:, 0]) < x_max)
    pts_xz = body_vertices[mask][:, [0, 2]]

    if len(pts_xz) < 3:
        return None, 0.0

    # Compute angle and radius from fixed centroid for ALL torso vertices
    angles = np.arctan2(pts_xz[:, 0] - centroid[0], pts_xz[:, 1] - centroid[1])
    radii = np.linalg.norm(pts_xz - centroid, axis=1)

    # Bin vertices by angle and take the MAX radius per bin (outermost vertex)
    n_a = len(ref_angles)
    a_step = ref_angles[1] - ref_angles[0]
    binned_radii = np.zeros(n_a)

    for k in range(n_a):
        lo = ref_angles[k] - a_step / 2
        hi = ref_angles[k] + a_step / 2
        # Handle wrap-around at ±π
        if lo < -math.pi:
            in_bin = (angles >= lo + 2 * math.pi) | (angles < hi)
        elif hi > math.pi:
            in_bin = (angles >= lo) | (angles < hi - 2 * math.pi)
        else:
            in_bin = (angles >= lo) & (angles < hi)
        if np.any(in_bin):
            binned_radii[k] = float(np.max(radii[in_bin]))

    # Fill empty bins by interpolation from neighbors
    empty = binned_radii == 0
    if np.any(empty) and not np.all(empty):
        filled = ~empty
        filled_angles = ref_angles[filled]
        filled_radii = binned_radii[filled]
        ext_a = np.concatenate([filled_angles - 2 * math.pi, filled_angles, filled_angles + 2 * math.pi])
        ext_r = np.concatenate([filled_radii, filled_radii, filled_radii])
        binned_radii[empty] = np.interp(ref_angles[empty], ext_a, ext_r)

    # Approximate perimeter from the binned radii
    pts_xz_approx = np.column_stack([
        centroid[0] + binned_radii * np.sin(ref_angles),
        centroid[1] + binned_radii * np.cos(ref_angles),
    ])
    segments = np.diff(np.vstack([pts_xz_approx, pts_xz_approx[0:1]]), axis=0)
    perimeter = float(np.sum(np.linalg.norm(segments, axis=1)))

    return binned_radii, perimeter


def build_body_radius_map(
    body_vertices: np.ndarray,
    body_height_m: float,
    n_heights: int = 10,
    n_ref_angles: int = 60,
) -> dict:
    """
    Build a 2D body radius profile: radius(y, θ).

    Extracts contours at n_heights between waist and upper-chest, resamples
    each to n_ref_angles uniform angles, and stores them in a lookup table
    for fast bilinear interpolation during panel placement.

    Uses a FIXED reference centroid (computed from all torso vertices) to
    prevent centroid drift between heights that would distort the radius
    profile.

    Returns dict with heights (n_heights,), ref_angles (n_ref_angles,),
    radii (n_heights, n_ref_angles), centroid (2,),
    and chest_perimeter_m (for garment scale computation).
    """
    y_lo = 0.55 * body_height_m     # below waist
    y_hi = 0.78 * body_height_m     # above chest
    heights = np.linspace(y_lo, y_hi, n_heights)
    ref_angles = np.linspace(-math.pi, math.pi, n_ref_angles, endpoint=False)

    # Compute a FIXED reference centroid from all torso vertices (Y in range, |X| < 0.20)
    torso_mask = ((body_vertices[:, 1] >= y_lo) &
                  (body_vertices[:, 1] <= y_hi) &
                  (np.abs(body_vertices[:, 0]) < 0.20))
    torso_xz = body_vertices[torso_mask][:, [0, 2]]
    if len(torso_xz) > 0:
        centroid = torso_xz.mean(axis=0)
    else:
        centroid = np.array([0.0, 0.0])

    radii = np.zeros((n_heights, n_ref_angles))
    hull_perimeters = np.zeros(n_heights)

    for hi, y in enumerate(heights):
        # Use vertex-based max-per-bin for accurate radius profile
        height_radii, _ = measure_body_radii_at_height(
            body_vertices, y, centroid, ref_angles,
        )
        if height_radii is not None:
            radii[hi] = height_radii

        # Use convex hull for perimeter (accurate circumference measurement)
        from scipy.spatial import ConvexHull
        band_mask = np.abs(body_vertices[:, 1] - y) < 0.02
        x_max = 0.20
        if np.sum(band_mask) >= 10:
            abs_x = np.sort(np.abs(body_vertices[band_mask, 0]))
            gaps = np.diff(abs_x)
            for i, g in enumerate(gaps):
                if abs_x[i] > 0.10 and g > 0.015:
                    x_max = float(abs_x[i]) + 0.005
                    break
        mask = band_mask & (np.abs(body_vertices[:, 0]) < x_max)
        pts_xz = body_vertices[mask][:, [0, 2]]
        if len(pts_xz) >= 3:
            hull = ConvexHull(pts_xz)
            hp = pts_xz[hull.vertices]
            segs = np.diff(np.vstack([hp, hp[0:1]]), axis=0)
            hull_perimeters[hi] = float(np.sum(np.linalg.norm(segs, axis=1)))

    # Use the maximum hull perimeter as "chest" for garment scale
    chest_perimeter_m = float(np.max(hull_perimeters)) if np.any(hull_perimeters > 0) else 0.96

    # Fill any missing heights by nearest-neighbor
    valid = np.any(radii > 0, axis=1)
    if not np.all(valid) and np.any(valid):
        for hi in range(n_heights):
            if not valid[hi]:
                nearest = np.argmin(np.abs(heights[valid] - heights[hi]))
                valid_idx = np.where(valid)[0][nearest]
                radii[hi] = radii[valid_idx]

    return {
        "heights": heights,
        "ref_angles": ref_angles,
        "radii": radii,
        "centroid": centroid,           # single fixed centroid
        "chest_perimeter_m": chest_perimeter_m,
    }


def query_body_radius(
    y_arr: np.ndarray,
    theta_arr: np.ndarray,
    body_map: dict,
) -> np.ndarray:
    """
    Bilinear interpolation of body radius at (y, θ).

    Returns radius array in metres. Centroid is fixed (stored in body_map).
    """
    heights = body_map["heights"]
    ref_angles = body_map["ref_angles"]
    radii = body_map["radii"]           # (n_h, n_a)
    n_h = len(heights)
    n_a = len(ref_angles)

    # Y interpolation
    y_frac = (y_arr - heights[0]) / (heights[-1] - heights[0]) * (n_h - 1)
    y_frac = np.clip(y_frac, 0, n_h - 1)
    y_lo = np.floor(y_frac).astype(int)
    y_hi = np.minimum(y_lo + 1, n_h - 1)
    y_t = y_frac - y_lo

    # Angle interpolation (wrap-around)
    a_step = ref_angles[1] - ref_angles[0]
    a_frac = (theta_arr - ref_angles[0]) / a_step
    a_frac = a_frac % n_a
    a_lo = np.floor(a_frac).astype(int) % n_a
    a_hi = (a_lo + 1) % n_a
    a_t = a_frac - np.floor(a_frac)

    # Bilinear interpolation of radius
    r00 = radii[y_lo, a_lo]
    r01 = radii[y_lo, a_hi]
    r10 = radii[y_hi, a_lo]
    r11 = radii[y_hi, a_hi]
    r_interp_lo = r00 * (1 - a_t) + r01 * a_t
    r_interp_hi = r10 * (1 - a_t) + r11 * a_t

    return r_interp_lo * (1 - y_t) + r_interp_hi * y_t


def place_panel_on_body_contour(
    pts_2d: np.ndarray,
    panel: dict,
    body_map: dict,
    y_shift_cm: float,
    r_garment_cm: float,
    garment_scale: float,
) -> np.ndarray:
    """
    Place a torso panel using body-conformal wrapping with Y-dependent radius.

    Uses the same angle mapping as cylindrical wrapping (θ = x_2d / r_garment)
    but replaces the constant cylinder radius with the body's actual radius
    at each vertex's (y, θ), scaled by garment_circ / body_perimeter.

    The body radius is looked up from a pre-built 2D profile (Y × θ), so the
    garment follows the body's taper at waist, chest, and shoulders.

    Returns (N, 3) float in metres.
    """
    pts = np.array(pts_2d, dtype=float)
    x2 = pts[:, 0]     # cm
    y2 = pts[:, 1]     # cm

    tz = panel["translation"][2]
    ty = panel["translation"][1]

    y3_m = (y2 + ty + y_shift_cm) / 100.0

    # Same angle mapping as place_panel_cylindrical
    if tz >= 0:
        theta = x2 / r_garment_cm
    else:
        x_mean = float(np.mean(x2))
        back_sign = 1.0 if x_mean >= 0 else -1.0
        theta = back_sign * math.pi - x2 / r_garment_cm

    # Body radius at each vertex's (y, θ)
    centroid = body_map["centroid"]   # fixed (2,) XZ
    r_body = query_body_radius(y3_m, theta, body_map)

    # Scale body radius by garment/body ratio
    r_garment = r_body * garment_scale

    # Convert to XZ coordinates (relative to fixed centroid)
    x3 = centroid[0] + r_garment * np.sin(theta)
    z3 = centroid[1] + r_garment * np.cos(theta)

    return np.column_stack([x3, y3_m, z3])


def triangulate_panel(outline_2d: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Delaunay-triangulate a 2D panel polygon.

    Adds interior grid points at ~5 cm spacing for a denser, more simulation-
    friendly mesh. Filters triangles whose centroids fall outside the boundary.

    Returns (pts, faces):
        pts   — (M, 2) float, augmented vertex array including grid points
        faces — (K, 3) int, triangle indices into pts
    """
    pts = np.array(outline_2d, dtype=float)   # (N, 2)

    # Bounding box for interior point grid
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    # Add interior grid points at 5 cm spacing
    grid_step = 5.0
    xs = np.arange(xmin + grid_step, xmax, grid_step)
    ys = np.arange(ymin + grid_step, ymax, grid_step)
    if len(xs) > 0 and len(ys) > 0:
        gx, gy = np.meshgrid(xs, ys)
        interior_candidates = np.column_stack([gx.ravel(), gy.ravel()])
        # Keep only candidates inside the polygon
        inside = points_inside_polygon(interior_candidates, pts)
        if np.any(inside):
            pts = np.vstack([pts, interior_candidates[inside]])

    tri = Delaunay(pts)
    faces = tri.simplices   # (M, 3)

    # Filter: keep only triangles whose centroid is inside the outline polygon
    outline_pts = np.array(outline_2d, dtype=float)
    centroids = pts[faces].mean(axis=1)   # (M, 2)
    inside_mask = points_inside_polygon(centroids, outline_pts)
    faces = faces[inside_mask]

    return pts, faces   # augmented vertex array (includes grid pts), filtered faces


def points_inside_polygon(pts: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Ray-casting test: returns bool array, True for pts inside the polygon.

    pts     : (N, 2) float
    polygon : (M, 2) float — closed polygon (first != last)
    """
    n_poly = len(polygon)
    n_pts = len(pts)
    inside = np.zeros(n_pts, dtype=bool)

    px = pts[:, 0]
    py = pts[:, 1]

    j = n_poly - 1
    for i in range(n_poly):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Edge i → j: does ray from (px, py) in +x direction cross?
        cond = ((yi > py) != (yj > py)) & \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi)
        inside ^= cond
        j = i

    return inside


# ---------------------------------------------------------------------------
# Seam constraint builder
# ---------------------------------------------------------------------------

def resample_1d(indices: list[int], vertices: np.ndarray, n: int) -> list[int]:
    """
    Resample a sequence of vertex indices to exactly n evenly-spaced samples
    by arc length. Returns new list of n indices (pointing into vertices).
    """
    if n <= 1:
        return [indices[0]] * n
    if len(indices) == n:
        return list(indices)

    pts = vertices[indices]                            # (K, 3)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_len[-1]
    if total < 1e-12:
        return [indices[0]] * n

    targets = np.linspace(0.0, total, n)
    resampled = []
    for t in targets:
        j = np.searchsorted(cum_len, t, side="right") - 1
        j = int(np.clip(j, 0, len(indices) - 1))
        resampled.append(indices[j])
    return resampled


def build_edge_global_indices(
    panel_name: str,
    panel: dict,
    panel_vertex_offset: int,
) -> dict[str, list[int]]:
    """
    Map each edge_id to its list of global vertex indices (full polyline
    including the endpoint, which is the start of the next edge).

    Panel vertices are stored consecutively in the global array starting
    at panel_vertex_offset. Each edge's polyline occupies sequential slots.
    The endpoint of edge_i is the start slot of edge_{i+1}
    (or slot 0 of the panel for the last edge, since the polygon is closed).
    """
    edges = panel["edges"]
    n_edges = len(edges)

    # Compute where each edge's polyline starts in the panel's local array
    edge_starts: list[int] = []
    pos = 0
    for edge in edges:
        edge_starts.append(pos)
        pos += len(edge["polyline"])
    # pos now equals total outline vertex count (= vertex_count - 1)

    result: dict[str, list[int]] = {}
    for i, edge in enumerate(edges):
        edge_id = f"{panel_name}_e{edge['edge_index']}"
        start = edge_starts[i]
        end = edge_starts[i] + len(edge["polyline"])

        # Polyline global indices (exclusive of endpoint)
        poly_global = list(range(panel_vertex_offset + start,
                                 panel_vertex_offset + end))

        # Endpoint: start of next edge (or edge_0 for the last edge = index 0)
        if i + 1 < n_edges:
            endpoint_global = panel_vertex_offset + edge_starts[i + 1]
        else:
            endpoint_global = panel_vertex_offset + 0   # closed polygon

        result[edge_id] = poly_global + [endpoint_global]

    return result


# ---------------------------------------------------------------------------
# Body-surface projection (pre-simulation) — panel-aware
# ---------------------------------------------------------------------------

def _classify_panel(panel_name: str) -> str:
    """
    Classify a panel by its name into a target body region.

    Returns one of: 'front_torso', 'back_torso', 'left_sleeve', 'right_sleeve'
    """
    name = panel_name.lower()
    if 'sleeve' in name:
        if 'left' in name:
            return 'left_sleeve'
        else:
            return 'right_sleeve'
    elif 'ftorso' in name:
        return 'front_torso'
    elif 'btorso' in name:
        return 'back_torso'
    # Fallback: use translation Z sign if available, default front
    return 'front_torso'


def _get_body_subregion(
    region: str,
    body_vertices: np.ndarray,
    body_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the subset of body vertices corresponding to a target region.

    Returns (sub_vertices, sub_normals, sub_indices) where sub_indices
    maps back into the original body_vertices array.
    """
    body_y = body_vertices[:, 1]
    body_y_max = float(body_y.max())
    body_y_min = float(body_y.min())
    body_height = body_y_max - body_y_min

    # Torso Y range: ~50% to ~82% of body height (waist to shoulders)
    torso_y_lo = body_y_min + 0.48 * body_height
    torso_y_hi = body_y_min + 0.85 * body_height

    if region == 'front_torso':
        mask = ((body_y >= torso_y_lo) & (body_y <= torso_y_hi) &
                (body_vertices[:, 2] > 0) &
                (np.abs(body_vertices[:, 0]) < 0.20))
    elif region == 'back_torso':
        mask = ((body_y >= torso_y_lo) & (body_y <= torso_y_hi) &
                (body_vertices[:, 2] <= 0) &
                (np.abs(body_vertices[:, 0]) < 0.20))
    elif region == 'left_sleeve':
        # Left arm: +X side
        mask = ((body_y >= torso_y_lo) & (body_y <= torso_y_hi) &
                (body_vertices[:, 0] > 0.10))
    elif region == 'right_sleeve':
        # Right arm: -X side
        mask = ((body_y >= torso_y_lo) & (body_y <= torso_y_hi) &
                (body_vertices[:, 0] < -0.10))
    else:
        mask = np.ones(len(body_vertices), dtype=bool)

    indices = np.where(mask)[0]
    if len(indices) < 10:
        # Fallback: use all body vertices in torso Y range
        mask = (body_y >= torso_y_lo) & (body_y <= torso_y_hi)
        indices = np.where(mask)[0]

    return body_vertices[indices], body_normals[indices], indices


def project_garment_onto_body(
    garment: dict,
    body_vertices: np.ndarray,
    body_normals: np.ndarray,
    body_faces: np.ndarray | None = None,
    standoff_m: float = 0.005,
) -> dict:
    """
    Panel-aware projection of garment vertices onto the body surface.

    Uses panel names (stored in garment["panel_vertex_ranges"]) to determine
    which body region each panel should map to:
      - ftorso panels → front body surface (Z > 0)
      - btorso panels → back body surface (Z ≤ 0)
      - left_sleeve panels → left arm (X > 0.10)
      - right_sleeve panels → right arm (X < -0.10)

    For each panel:
      1. Compute panel centroid and target body-region centroid
      2. Translate the entire panel to align centroids
      3. Project each vertex onto the nearest point in the target sub-region
      4. Offset by standoff_m along the surface normal

    Topology (faces, edges, seam constraints) is NOT modified.
    Stretch constraint rest lengths are PRESERVED from the flat pattern.

    Parameters
    ----------
    garment         : dict from assemble_garment() with "panel_vertex_ranges"
    body_vertices   : (B, 3) body mesh vertices in metres
    body_normals    : (B, 3) per-vertex body normals (unit vectors)
    body_faces      : (F, 3) body mesh face indices (unused, reserved)
    standoff_m      : offset along outward normal (default 5 mm)

    Returns
    -------
    garment dict with updated "vertices" (rest lengths unchanged)
    """
    vertices = garment["vertices"].copy()
    N = len(vertices)
    if N == 0:
        return garment

    panel_ranges = garment.get("panel_vertex_ranges", {})

    if not panel_ranges:
        # Fallback: no panel info, project all vertices onto whole body
        body_tree = KDTree(body_vertices)
        _, nearest_idx = body_tree.query(vertices)
        garment["vertices"] = (body_vertices[nearest_idx]
                               + body_normals[nearest_idx] * standoff_m)
        return garment

    projected = vertices.copy()

    for panel_name, (start_idx, end_idx) in panel_ranges.items():
        region = _classify_panel(panel_name)
        sub_verts, sub_normals, _ = _get_body_subregion(
            region, body_vertices, body_normals,
        )

        if len(sub_verts) < 3:
            continue

        panel_vertices = vertices[start_idx:end_idx]  # (P, 3)
        if len(panel_vertices) == 0:
            continue

        # Translate panel to target body region.
        # Sleeve panels: full 3D translation to position at arm location.
        # Torso panels: skip — already body-conformal from assembly.
        #   The assembly's cylindrical wrapping encodes size-dependent depth
        #   and lateral spread which must be preserved for correct clearance.
        panel_centroid = panel_vertices.mean(axis=0)
        target_centroid = sub_verts.mean(axis=0)

        if 'sleeve' in region:
            shift = target_centroid - panel_centroid
            projected[start_idx:end_idx] = panel_vertices + shift
        # else: keep torso panels as-is from assembly

    # Final pass: fix any penetrating vertices (push outside body surface).
    # Use a small margin (1mm) — just enough to prevent actual penetrations,
    # without distorting the assembly's intended placement.
    body_tree = KDTree(body_vertices)
    _, nearest_idx = body_tree.query(projected)
    nearest_pts = body_vertices[nearest_idx]
    nearest_nrm = body_normals[nearest_idx]
    displacements = projected - nearest_pts
    signed_dists = np.einsum("ij,ij->i", displacements, nearest_nrm)
    margin = 0.001  # 1mm collision margin
    penetrating = signed_dists < margin
    if np.any(penetrating):
        correction = (margin - signed_dists[penetrating])[:, np.newaxis] \
                     * nearest_nrm[penetrating]
        projected[penetrating] += correction

    garment["vertices"] = projected
    return garment


# ---------------------------------------------------------------------------
# Garment assembly
# ---------------------------------------------------------------------------

def _is_vertex_from_torso(
    vertex_idx: int,
    panel_offsets: dict[str, int],
    panel_is_torso: dict[str, bool],
    pattern: dict,
) -> bool:
    """Check if a global vertex index belongs to a torso panel."""
    # Find which panel contains this vertex
    sorted_panels = sorted(panel_offsets.items(), key=lambda x: x[1])
    for i, (pname, offset) in enumerate(sorted_panels):
        next_offset = sorted_panels[i + 1][1] if i + 1 < len(sorted_panels) else float('inf')
        if offset <= vertex_idx < next_offset:
            return panel_is_torso.get(pname, True)
    return True


def identify_armholes(
    manifest: dict,
    edge_global_indices: dict[str, list[int]],
    panel_is_torso: dict[str, bool],
    vertices: np.ndarray,
    body_vertices: np.ndarray | None = None,
) -> dict[str, dict]:
    """
    Identify armhole boundary from the seam manifest.

    Finds sleeve-to-torso seam pairs, collects the torso-side and sleeve-side
    edge vertices, and computes armhole geometry (centroid, radius, arm direction)
    for each side (left / right).

    Returns dict with keys "left" and "right", each containing:
        armhole_indices: set of global vertex indices on the torso armhole boundary
        sleeve_cap_indices: set of global vertex indices on the sleeve cap
        centroid: (3,) armhole centroid position
        radius: float, approximate arm cylinder radius
        arm_direction: (3,) unit vector along arm
        seam_pairs: list of (edge_a_id, edge_b_id, torso_edge_id, sleeve_edge_id)
    """
    sides: dict[str, dict] = {
        "left": {"armhole_indices": set(), "sleeve_cap_indices": set(),
                 "sleeve_cap_by_panel": {}, "seam_pairs": []},
        "right": {"armhole_indices": set(), "sleeve_cap_indices": set(),
                  "sleeve_cap_by_panel": {}, "seam_pairs": []},
    }

    for sp in manifest["seam_pairs"]:
        edge_a_id = sp["edge_a"]
        edge_b_id = sp["edge_b"]

        if edge_a_id not in edge_global_indices or edge_b_id not in edge_global_indices:
            continue

        panel_a = "_".join(edge_a_id.split("_")[:-1])
        panel_b = "_".join(edge_b_id.split("_")[:-1])

        a_is_torso = panel_is_torso.get(panel_a, True)
        b_is_torso = panel_is_torso.get(panel_b, True)

        # We want seams where exactly one side is torso, the other is sleeve
        if a_is_torso == b_is_torso:
            continue

        torso_edge_id = edge_a_id if a_is_torso else edge_b_id
        sleeve_edge_id = edge_b_id if a_is_torso else edge_a_id
        torso_panel = panel_a if a_is_torso else panel_b
        sleeve_panel = panel_b if a_is_torso else panel_a

        side = "left" if "left_" in torso_panel else "right"
        sides[side]["armhole_indices"].update(edge_global_indices[torso_edge_id])
        sides[side]["sleeve_cap_indices"].update(edge_global_indices[sleeve_edge_id])
        sides[side]["seam_pairs"].append(
            (edge_a_id, edge_b_id, torso_edge_id, sleeve_edge_id)
        )

        # Track which local indices form the cap for each sleeve panel
        if sleeve_panel not in sides[side]["sleeve_cap_by_panel"]:
            sides[side]["sleeve_cap_by_panel"][sleeve_panel] = []
        sides[side]["sleeve_cap_by_panel"][sleeve_panel].extend(
            edge_global_indices[sleeve_edge_id]
        )

    # Compute geometry for each side
    for side_name, side_data in sides.items():
        ah_idx = sorted(side_data["armhole_indices"])
        if not ah_idx:
            side_data["centroid"] = np.array([0.0, 1.4, 0.0])
            side_data["radius"] = 0.05
            side_data["arm_direction"] = np.array([1.0, 0.0, 0.0] if side_name == "left"
                                                  else [-1.0, 0.0, 0.0])
            continue

        ah_pts = vertices[ah_idx]
        centroid = ah_pts.mean(axis=0)

        # Arm direction: lateral outward
        arm_dir = np.array([1.0, 0.0, 0.0] if centroid[0] > 0 else [-1.0, 0.0, 0.0])

        # Compute arm wrapping radius from body mesh cross-section
        # (NOT from armhole boundary arc length — the armhole is an oval, not a circle)
        radius = 0.05  # fallback 5cm
        if body_vertices is not None:
            # Sample arm cross-section slightly outward from armhole centroid
            sample_x = centroid[0] + 0.05 * arm_dir[0]
            arm_mask = ((np.abs(body_vertices[:, 0] - sample_x) < 0.03) &
                        (np.abs(body_vertices[:, 1] - centroid[1]) < 0.05))
            arm_pts = body_vertices[arm_mask]
            if len(arm_pts) >= 4:
                yz = arm_pts[:, [1, 2]]
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(yz)
                    hp = yz[hull.vertices]
                    segs = np.diff(np.vstack([hp, hp[0:1]]), axis=0)
                    arm_perim = float(np.sum(np.linalg.norm(segs, axis=1)))
                    radius = arm_perim / (2 * math.pi)
                except Exception:
                    pass

        side_data["centroid"] = centroid
        side_data["radius"] = radius
        side_data["arm_direction"] = arm_dir

    return sides


def assemble_garment(
    pattern: dict,
    manifest: dict,
    body_vertices: np.ndarray,
    n_seam_pts: int = 20,
) -> dict:
    """
    Build 3D garment mesh from GarmentCode pattern + seam manifest.

    Returns
    -------
    {
      "vertices": (N, 3) float — initial 3D positions in metres
      "faces":    (M, 3) int   — triangle indices
      "stretch_i/j/rest": stretch constraint arrays
      "seam_i/j": torso seam constraint arrays (rest length = 0)
      "sleeve_seam_i/j": sleeve-to-torso + sleeve-to-sleeve seam arrays
      "garment_scale", "body_map": placement metadata
    }
    """
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    panel_offsets: dict[str, int] = {}
    edge_global_indices: dict[str, list[int]] = {}
    panel_pts_2d: dict[str, np.ndarray] = {}     # for re-placement

    vertex_offset = 0
    stretch_edges: list[tuple[int, int, float]] = []

    # ---- Pre-pass: compute placement parameters ------------------------------
    total_circ_cm = 0.0
    max_torso_y_cm = -1e9
    _TARGET_TOP_Y_CM = 140.0

    _panel_is_torso: dict[str, bool] = {}
    for _pname, _panel in pattern["panels"].items():
        _tx = _panel["translation"][0]
        _is_torso = abs(_tx) < 20.0
        _panel_is_torso[_pname] = _is_torso
        if _is_torso:
            _outline = build_panel_outline_2d(_panel)
            _pts = np.array(_outline, dtype=float)
            total_circ_cm += float(_pts[:, 0].max() - _pts[:, 0].min())
            _ty = _panel["translation"][1]
            max_torso_y_cm = max(max_torso_y_cm, float(_pts[:, 1].max()) + _ty)

    if total_circ_cm < 1.0:
        total_circ_cm = 100.0
    r_garment_cm = total_circ_cm / (2.0 * math.pi)

    y_shift_cm = _TARGET_TOP_Y_CM - max_torso_y_cm if max_torso_y_cm > -1e8 else 0.0

    body_height_m = float(body_vertices[:, 1].max() - body_vertices[:, 1].min())
    body_map = build_body_radius_map(body_vertices, body_height_m)
    garment_scale = (total_circ_cm / 100.0) / body_map["chest_perimeter_m"]

    # ---- PASS 1: Place ALL panels (torso conformal, sleeve near-shoulder) ----
    for panel_name, panel in pattern["panels"].items():
        outline_2d = build_panel_outline_2d(panel)
        pts_2d, faces_local = triangulate_panel(outline_2d)
        panel_pts_2d[panel_name] = pts_2d

        if _panel_is_torso[panel_name]:
            verts_3d = place_panel_on_body_contour(
                pts_2d, panel, body_map, y_shift_cm,
                r_garment_cm, garment_scale,
            )
        else:
            # Temporary placement — will be replaced after armhole identification
            verts_3d = place_sleeve_near_shoulder(pts_2d, panel)

        panel_offsets[panel_name] = vertex_offset
        n_panel_verts = len(verts_3d)

        # Collect stretch constraints
        edge_set: set[tuple[int, int]] = set()
        for face in faces_local:
            for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
                key = (min(a, b), max(a, b))
                if key not in edge_set:
                    edge_set.add(key)
                    rest = float(np.linalg.norm(verts_3d[b] - verts_3d[a]))
                    stretch_edges.append((
                        a + vertex_offset,
                        b + vertex_offset,
                        rest,
                    ))

        edge_global_indices.update(
            build_edge_global_indices(panel_name, panel, vertex_offset)
        )

        all_verts.append(verts_3d)
        offset_faces = faces_local + vertex_offset
        all_faces.append(offset_faces)
        vertex_offset += n_panel_verts

    vertices = np.vstack(all_verts)
    faces = np.vstack(all_faces)

    # ---- PASS 2: Identify armholes and re-place sleeves ----------------------
    armholes = identify_armholes(
        manifest, edge_global_indices, _panel_is_torso, vertices,
        body_vertices=body_vertices,
    )

    for side_name, side_data in armholes.items():
        ah_centroid = side_data["centroid"]
        ah_radius = side_data["radius"]
        arm_dir = side_data["arm_direction"]
        cap_by_panel = side_data.get("sleeve_cap_by_panel", {})

        for panel_name, cap_global_indices in cap_by_panel.items():
            if panel_name not in panel_offsets:
                continue
            offset = panel_offsets[panel_name]
            pts_2d = panel_pts_2d[panel_name]
            panel = pattern["panels"][panel_name]

            # Convert cap global indices to local
            cap_local = [gi - offset for gi in cap_global_indices
                         if 0 <= gi - offset < len(pts_2d)]

            new_verts = place_sleeve_at_armhole(
                pts_2d, panel, ah_centroid, ah_radius, arm_dir, cap_local,
            )

            # Update vertices in-place
            n_verts = len(new_verts)
            vertices[offset:offset + n_verts] = new_verts

    # ---- Build seam constraints from manifest --------------------------------
    # Now separate into torso seams and sleeve seams
    torso_seam_pairs: list[tuple[int, int]] = []
    sleeve_seam_pairs: list[tuple[int, int]] = []

    for sp in manifest["seam_pairs"]:
        edge_a_id = sp["edge_a"]
        edge_b_id = sp["edge_b"]

        if edge_a_id not in edge_global_indices or edge_b_id not in edge_global_indices:
            continue

        panel_a = "_".join(edge_a_id.split("_")[:-1])
        panel_b = "_".join(edge_b_id.split("_")[:-1])

        a_is_torso = _panel_is_torso.get(panel_a, True)
        b_is_torso = _panel_is_torso.get(panel_b, True)

        # Classify: both torso → torso seam; any sleeve involved → sleeve seam
        is_torso_seam = a_is_torso and b_is_torso
        target_list = torso_seam_pairs if is_torso_seam else sleeve_seam_pairs

        ids_a = edge_global_indices[edge_a_id]
        ids_b = edge_global_indices[edge_b_id]

        ids_a = resample_1d(ids_a, vertices, n_seam_pts)
        ids_b = resample_1d(ids_b, vertices, n_seam_pts)

        # Auto-orient using endpoint matching
        pt_a0 = vertices[ids_a[0]]
        pt_a1 = vertices[ids_a[-1]]
        pt_b0 = vertices[ids_b[0]]
        pt_b1 = vertices[ids_b[-1]]
        gap_endpoint_fwd = (float(np.linalg.norm(pt_a0 - pt_b0))
                            + float(np.linalg.norm(pt_a1 - pt_b1)))
        gap_endpoint_rev = (float(np.linalg.norm(pt_a0 - pt_b1))
                            + float(np.linalg.norm(pt_a1 - pt_b0)))
        if gap_endpoint_rev < gap_endpoint_fwd:
            ids_b = ids_b[::-1]
            gap_endpoint_fwd = gap_endpoint_rev

        # For torso seams: skip if endpoint gap > 3cm (structural seams)
        # For sleeve seams: use a larger threshold since initial gaps may be bigger
        gap_threshold = 0.03 if is_torso_seam else 0.10
        if gap_endpoint_fwd > gap_threshold:
            continue

        # Per-vertex gap threshold: torso=2cm, sleeve=5cm
        vertex_gap_threshold = 0.02 if is_torso_seam else 0.05
        for ia, ib in zip(ids_a, ids_b):
            if ia == ib:
                continue
            if float(np.linalg.norm(vertices[ia] - vertices[ib])) > vertex_gap_threshold:
                continue
            target_list.append((ia, ib))

    torso_seam_pairs = list(dict.fromkeys(torso_seam_pairs))
    sleeve_seam_pairs = list(dict.fromkeys(sleeve_seam_pairs))

    # Pack constraints into arrays and recompute rest lengths from final positions
    if stretch_edges:
        stretch_arr = np.array(stretch_edges, dtype=object)
        stretch_i = stretch_arr[:, 0].astype(np.int32)
        stretch_j = stretch_arr[:, 1].astype(np.int32)
        # Recompute rest lengths from FINAL vertex positions (after sleeve
        # re-placement and snapping). Pass 1 rest lengths are stale for
        # sleeve panels that were re-placed in Pass 2/3.
        stretch_rest = np.linalg.norm(
            vertices[stretch_j] - vertices[stretch_i], axis=1
        ).astype(np.float64)
    else:
        stretch_i = np.array([], dtype=np.int32)
        stretch_j = np.array([], dtype=np.int32)
        stretch_rest = np.array([], dtype=np.float64)

    def _pack_seam_arrays(pairs):
        if pairs:
            arr = np.array(pairs, dtype=np.int32)
            return arr[:, 0], arr[:, 1]
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    seam_i, seam_j = _pack_seam_arrays(torso_seam_pairs)
    sleeve_seam_i, sleeve_seam_j = _pack_seam_arrays(sleeve_seam_pairs)

    # Estimate garment area
    if len(faces) > 0:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        total_area = float(np.sum(np.linalg.norm(cross, axis=1))) * 0.5
    else:
        total_area = 0.1

    # Build panel vertex ranges for panel-aware projection
    panel_vertex_ranges: dict[str, tuple[int, int]] = {}
    sorted_panels = sorted(panel_offsets.items(), key=lambda x: x[1])
    for i, (pname, offset) in enumerate(sorted_panels):
        if i + 1 < len(sorted_panels):
            end = sorted_panels[i + 1][1]
        else:
            end = len(vertices)
        panel_vertex_ranges[pname] = (offset, end)

    return {
        "vertices": vertices,
        "faces": faces,
        "total_area_m2": total_area,
        "stretch_i": stretch_i,
        "stretch_j": stretch_j,
        "stretch_rest": stretch_rest,
        "seam_i": seam_i,
        "seam_j": seam_j,
        "sleeve_seam_i": sleeve_seam_i,
        "sleeve_seam_j": sleeve_seam_j,
        "garment_scale": garment_scale,
        "body_map": body_map,
        "panel_vertex_ranges": panel_vertex_ranges,
    }


# ---------------------------------------------------------------------------
# Post-simulation analysis
# ---------------------------------------------------------------------------

def compute_strain_ratios(
    positions: np.ndarray,
    stretch_i: np.ndarray,
    stretch_j: np.ndarray,
    stretch_rest: np.ndarray,
    garment_regions: dict[str, list[int]],
) -> dict[str, float]:
    """
    Compute per-region median strain ratio from stretch constraints.

    strain_ratio = current_edge_length / rest_length
      > 1.0 = stretched (tension)
      < 1.0 = compressed
      = 1.0 = at rest

    Only counts constraints where BOTH endpoints are in the same region.
    """
    if len(stretch_i) == 0:
        return {r: 1.0 for r in garment_regions}

    # Compute current edge lengths
    d = positions[stretch_j] - positions[stretch_i]
    current_lengths = np.linalg.norm(d, axis=1)
    safe_rest = np.where(stretch_rest > 1e-10, stretch_rest, 1e-10)
    strain_ratios = current_lengths / safe_rest

    # Build vertex → region lookup
    vertex_to_region: dict[int, str] = {}
    for region, vids in garment_regions.items():
        for vid in vids:
            vertex_to_region[vid] = region

    # Assign each constraint to a region (both endpoints must match)
    region_strains: dict[str, list[float]] = {r: [] for r in garment_regions}
    for k in range(len(stretch_i)):
        vi = int(stretch_i[k])
        vj = int(stretch_j[k])
        ri = vertex_to_region.get(vi)
        rj = vertex_to_region.get(vj)
        if ri is not None and ri == rj:
            region_strains[ri].append(float(strain_ratios[k]))

    # Compute median per region (require ≥5 edges for meaningful median)
    result: dict[str, float] = {}
    for region, strains in region_strains.items():
        if len(strains) >= 5:
            result[region] = float(np.median(strains))
        else:
            result[region] = 1.0  # insufficient data → assume rest

    return result
