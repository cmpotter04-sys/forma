"""
src/geometer/region_map.py

Body region segmentation for the parametric torso mesh.
No SMPL-X — uses vertex height and normal direction to assign each body
vertex to one of the 6 Forma regions.

Supports two body coordinate systems automatically:
    MakeHuman  — Y-up  (col 1 is height), facing +Z (col 2 normal = front)
    Anny       — Z-up  (col 2 is height), facing +Y (col 1 normal = front)

The up axis is auto-detected as the column with the greatest vertex extent.
Height bands are expressed as proportions of total body height so they work
for any mesh scale or coordinate orientation.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree


REQUIRED_REGIONS = [
    "chest_front",
    "chest_side",
    "shoulder_left",
    "shoulder_right",
    "upper_back",
    "waist",
]

# ---------------------------------------------------------------------------
# Proportional height bands (fraction of total body height)
# Derived from standard anthropometric tables (Pheasant "Bodyspace" 2nd ed.;
# Winter "Biomechanics" 4th ed.) to match both MakeHuman and Anny geometry.
# ---------------------------------------------------------------------------
_SHOULDER_LO = 0.79   # shoulder / armscye base
_SHOULDER_HI = 0.87   # top of shoulder region / base of neck
_CHEST_LO    = 0.68   # bottom of chest / bust band
_CHEST_HI    = 0.84   # top of chest band (overlaps shoulder slightly)
_WAIST_LO    = 0.55   # bottom of natural-waist band
_WAIST_HI    = 0.68   # top of natural-waist band (= chest bottom)


def classify_body_vertices(
    vertices: np.ndarray,
    normals: np.ndarray,
) -> dict[str, list[int]]:
    """
    Assign each body vertex to exactly one Forma region.

    Works with both Y-up (MakeHuman) and Z-up (Anny) meshes by auto-detecting
    the up axis from vertex extents.  Height bands are proportional so they
    are independent of absolute mesh scale.

    Priority order: shoulders first (they overlap the chest height band),
    then front/back chest, then chest_side, then waist.

    Parameters
    ----------
    vertices : (N, 3) float array — body mesh vertices in meters
    normals  : (N, 3) float array — per-vertex outward normals (unit)

    Returns
    -------
    dict mapping region_name → list of vertex indices
    """
    n = len(vertices)

    # --- Auto-detect coordinate axes from vertex extents --------------------
    extents = vertices.max(axis=0) - vertices.min(axis=0)

    # The axis with the greatest span is "up" (height).
    up_axis = int(np.argmax(extents))

    # Among the two remaining axes, the wider one is lateral (left-right),
    # the narrower one is the depth / forward axis.
    non_up = [a for a in (0, 1, 2) if a != up_axis]
    lat_axis = non_up[int(np.argmax([extents[a] for a in non_up]))]
    fwd_axis = [a for a in non_up if a != lat_axis][0]

    h  = vertices[:, up_axis]    # height coordinate (feet = min, head = max)
    x  = vertices[:, lat_axis]   # lateral coordinate (+ = wearer's left)
    nf = normals[:, fwd_axis]    # forward normal component (+ = front-facing)

    H = h.max()   # total body height (used to compute proportional bands)

    assigned = np.zeros(n, dtype=bool)
    regions: dict[str, list[int]] = {r: [] for r in REQUIRED_REGIONS}

    def _assign(mask: np.ndarray, region: str) -> None:
        idx = np.where(mask & ~assigned)[0]
        regions[region].extend(idx.tolist())
        assigned[idx] = True

    # Shoulder zones (proportional height band), classified by lateral position
    _assign(
        (h >= H * _SHOULDER_LO) & (h <= H * _SHOULDER_HI) & (x < -0.05),
        "shoulder_right",
    )
    _assign(
        (h >= H * _SHOULDER_LO) & (h <= H * _SHOULDER_HI) & (x >  0.05),
        "shoulder_left",
    )

    # Chest / upper-back zone, classified by forward normal direction
    chest_band = (h >= H * _CHEST_LO) & (h <= H * _CHEST_HI)
    _assign(chest_band & (nf >  0.25), "chest_front")
    _assign(chest_band & (nf < -0.25), "upper_back")
    _assign(chest_band,                 "chest_side")   # remainder in band

    # Waist zone
    _assign((h >= H * _WAIST_LO) & (h <= H * _WAIST_HI), "waist")

    return regions


def assign_garment_to_body_regions(
    garment_vertices: np.ndarray,
    body_vertices: np.ndarray,
    body_regions: dict[str, list[int]],
    max_distance_m: float = 0.10,
) -> dict[str, list[int]]:
    """
    For each garment vertex, find its nearest body vertex and inherit
    that body vertex's region assignment.

    Garment vertices further than max_distance_m from the body surface are
    excluded (typically sleeve panels placed far from the torso).  The default
    threshold of 10 cm accommodates very large garment sizes while excluding
    sleeve panels that produce large spurious clearance values.

    Parameters
    ----------
    garment_vertices : (M, 3) float array
    body_vertices    : (N, 3) float array
    body_regions     : output of classify_body_vertices()
    max_distance_m   : garment vertices farther than this are ignored

    Returns
    -------
    dict mapping region_name → list of garment vertex indices
    """
    tree = KDTree(body_vertices)
    dists, nearest_body_idx = tree.query(garment_vertices)

    # Build reverse map: body_vertex_idx → region name
    body_vertex_to_region: dict[int, str] = {}
    for region, vids in body_regions.items():
        for vid in vids:
            body_vertex_to_region[vid] = region

    garment_regions: dict[str, list[int]] = {r: [] for r in REQUIRED_REGIONS}
    for gv_idx, (bv_idx, dist) in enumerate(zip(nearest_body_idx, dists)):
        if dist > max_distance_m:
            continue   # sleeve or other far-out vertex — skip
        region = body_vertex_to_region.get(int(bv_idx))
        if region is not None:
            garment_regions[region].append(gv_idx)

    return garment_regions
