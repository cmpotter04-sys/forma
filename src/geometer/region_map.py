"""
src/geometer/region_map.py

Body region segmentation for the parametric torso mesh.
No SMPL-X — uses vertex height (Y coordinate) and normal direction to assign
each body vertex to one of the 6 Forma regions.

Body coordinate system:
    Y-up, feet at y=0, head at y=1.8m
    Facing +Z (front of body has normal_z > 0)
    Left-right along X (wearer's left = +X)
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


def classify_body_vertices(
    vertices: np.ndarray,
    normals: np.ndarray,
) -> dict[str, list[int]]:
    """
    Assign each body vertex to exactly one Forma region.

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
    y = vertices[:, 1]   # height (up axis, 0 = sole)
    x = vertices[:, 0]   # lateral (+ = wearer's left)
    nz = normals[:, 2]   # forward component (+Z = front-facing)

    assigned = np.zeros(n, dtype=bool)
    regions: dict[str, list[int]] = {r: [] for r in REQUIRED_REGIONS}

    def _assign(mask: np.ndarray, region: str) -> None:
        idx = np.where(mask & ~assigned)[0]
        regions[region].extend(idx.tolist())
        assigned[idx] = True

    # Shoulder zones (y 1.32–1.50 m), classified by lateral position
    _assign((y >= 1.32) & (y <= 1.50) & (x < -0.05), "shoulder_right")
    _assign((y >= 1.32) & (y <= 1.50) & (x >  0.05), "shoulder_left")

    # Chest / upper-back zone (y 1.12–1.38 m), classified by normal direction
    chest_band = (y >= 1.12) & (y <= 1.38)
    _assign(chest_band & (nz >  0.25), "chest_front")
    _assign(chest_band & (nz < -0.25), "upper_back")
    _assign(chest_band,                 "chest_side")   # remainder in band

    # Waist zone (y 0.95–1.12 m)
    _assign((y >= 0.95) & (y <= 1.12), "waist")

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
