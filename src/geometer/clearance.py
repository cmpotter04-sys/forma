"""
src/geometer/clearance.py

Signed clearance computation between a draped garment and the body surface.

Terminology (from FORMA_GEOMETER_SPEC.md Sub-Problem 4):
    delta_mm > 0  → garment is further from body than body surface (ease / loose)
    delta_mm < 0  → garment is compressed against / penetrating body (tight)
    delta_mm ≈ 0  → garment resting at body surface

All distances are in millimetres. Inputs are in metres.

Clearance approach:
    Uses KDTree nearest-neighbor for garments at or outside the body surface.
    For undersized garments placed inside the body, supplements with a
    circumference-based radial clearance derived from the garment_scale
    (garment_circ / body_circ) and the body's radius at the region height.
    This guarantees monotonic clearance as garment size changes.
"""

from __future__ import annotations
import math
from typing import Optional
import numpy as np
from scipy.spatial import KDTree


def compute_region_clearance(
    garment_vertices: np.ndarray,
    body_vertices: np.ndarray,
    body_normals: np.ndarray,
    garment_vertex_ids: list[int],
    body_vertex_ids: list[int],
    garment_scale: Optional[float] = None,
    body_map: Optional[dict] = None,
    bend_offset_m: float = 0.0,
) -> float:
    """
    Compute median signed clearance for a single body region.

    Uses KDTree nearest-neighbor signed distance as the primary metric.
    When garment_scale and body_map are provided (from garment assembly),
    supplements with a circumference-based radial clearance to ensure
    physically meaningful negative values for undersized garments.

    Positive = garment is outside body (ease).
    Negative = garment is inside body (tight).

    Parameters
    ----------
    garment_vertices   : (M, 3) float — draped garment vertex positions (metres)
    body_vertices      : (N, 3) float — body mesh vertex positions (metres)
    body_normals       : (N, 3) float — per-vertex outward normals (unit)
    garment_vertex_ids : indices into garment_vertices for this region
    body_vertex_ids    : indices into body_vertices for this region
    garment_scale      : garment_circ / body_circ from assembly (optional)
    body_map           : body radius profile from assembly (optional)

    Returns
    -------
    delta_mm : float — median signed clearance in millimetres
    """
    if not garment_vertex_ids or not body_vertex_ids:
        return 0.0

    body_region_verts = body_vertices[body_vertex_ids]
    body_region_normals = body_normals[body_vertex_ids]
    gv = garment_vertices[garment_vertex_ids]      # (K, 3)

    # KDTree signed distance (accurate when garment is near/outside body)
    tree = KDTree(body_region_verts)
    _, nearest_idx = tree.query(gv)
    nearest_pts = body_region_verts[nearest_idx]
    nearest_nrm = body_region_normals[nearest_idx]
    displacement = gv - nearest_pts
    kdtree_signed = np.einsum("ij,ij->i", displacement, nearest_nrm)

    if garment_scale is not None and body_map is not None:
        # Circumference-based radial clearance at the region's height.
        # delta_r = body_r(y) × (garment_scale - 1)
        # This is guaranteed monotonic with garment size.
        y_center = float(np.median(body_region_verts[:, 1]))
        heights = body_map["heights"]
        centroid = body_map["centroid"]

        # Interpolate mean body radius at this height
        hi = int(np.clip(
            np.searchsorted(heights, y_center) - 1,
            0, len(heights) - 2,
        ))
        t = (y_center - heights[hi]) / max(heights[hi + 1] - heights[hi], 1e-10)
        t = np.clip(t, 0.0, 1.0)
        radii_lo = body_map["radii"][hi]
        radii_hi = body_map["radii"][min(hi + 1, len(heights) - 1)]
        mean_body_r = float(np.mean(radii_lo * (1 - t) + radii_hi * t))

        # Circumference-based radial clearance, adjusted for bending resistance.
        # Stiffer fabrics bridge body concavities (positive offset); softer
        # fabrics conform more closely (negative offset).  Cotton reference
        # has bend_offset_m=0, so existing results are unchanged.
        circ_delta_m = mean_body_r * (garment_scale - 1.0) + bend_offset_m

        # Use circumference-based value as a floor for the clearance.
        # KDTree may underestimate tightness when garment is inside the body,
        # so take the minimum (most negative) of the two per vertex.
        circ_clearance = np.full(len(gv), circ_delta_m)
        signed_dists = np.minimum(kdtree_signed, circ_clearance)
    else:
        signed_dists = kdtree_signed

    delta_m = float(np.median(signed_dists))
    return delta_m * 1000.0   # metres → millimetres


def detect_tunnel_through(
    garment_vertices: np.ndarray,
    body_vertices: np.ndarray,
    body_normals: np.ndarray,
) -> tuple[int, float]:
    """
    Count garment vertices that have tunnelled inside the body mesh.

    A vertex is considered tunnelled if:
    - Its nearest body vertex is within 2 mm
    - The displacement is in the OPPOSITE direction of the body normal
      (meaning the garment vertex is on the inside).

    Returns
    -------
    (tunnel_count, tunnel_pct)
    """
    tree = KDTree(body_vertices)
    dists, nearest_idx = tree.query(garment_vertices)

    displacements = garment_vertices - body_vertices[nearest_idx]   # (M, 3)
    normals_at_nearest = body_normals[nearest_idx]                  # (M, 3)

    dot_products = np.einsum("ij,ij->i", displacements, normals_at_nearest)

    # Tunnelled: close to body AND on the inside (dot < 0)
    tunnelled = (dists < 0.002) & (dot_products < 0)
    tunnel_count = int(np.sum(tunnelled))
    tunnel_pct = tunnel_count / max(len(garment_vertices), 1) * 100.0

    return tunnel_count, tunnel_pct


def classify_severity(
    delta_mm: float,
    median_strain_ratio: float | None = None,
) -> str:
    """
    Classify signed clearance into a fit severity level.

    Post-collision (AC-2), strain ratio provides a secondary metric:
        red    : delta_mm < -25  OR  median_strain_ratio > 1.15
        yellow : delta_mm < -10  OR  median_strain_ratio > 1.08
        green  : otherwise
    """
    sr = median_strain_ratio if median_strain_ratio is not None else 1.0

    if delta_mm <= -25.0 or sr > 1.15:
        return "red"
    elif delta_mm <= -10.0 or sr > 1.08:
        return "yellow"
    else:
        return "green"


def classify_ease(delta_mm: float) -> tuple[float, str]:
    """
    Classify clearance for the ease_map.

    Returns (excess_mm, verdict).
    excess_mm counts only positive clearance (ease); tight regions have excess=0.
    """
    excess_mm = max(0.0, delta_mm)

    if delta_mm < 0.0:
        verdict = "tight_fit"
    elif delta_mm <= 20.0:
        verdict = "standard_fit"
    elif delta_mm <= 50.0:
        verdict = "relaxed_fit"
    else:
        verdict = "oversized"

    return excess_mm, verdict
