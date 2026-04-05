"""
src/geometer/subdivide.py

Loop subdivision utility for mesh refinement.
Subdivides a triangle mesh until vertex count meets or exceeds a target,
producing a smoother, denser mesh suitable for high-fidelity simulation.

Uses trimesh.remesh.subdivide_loop (Loop subdivision) when available,
falling back to trimesh.remesh.subdivide (midpoint) otherwise.
"""

from __future__ import annotations

import numpy as np

# Try Loop subdivision first (better quality), fall back to midpoint
try:
    from trimesh.remesh import subdivide_loop as _subdivide_fn

    _LOOP_AVAILABLE = True
except ImportError:
    from trimesh.remesh import subdivide as _subdivide_fn

    _LOOP_AVAILABLE = False


def subdivide_to_target(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_verts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subdivide a triangle mesh until vertex count >= target_verts.

    Each Loop subdivision pass roughly 4× the face count:
        655 verts → ~2600 → ~10K → ~40K → ~160K.
    Two passes reach ~10K, three reach ~40K.

    Parameters
    ----------
    vertices     : (V, 3) float — vertex positions
    faces        : (F, 3) int   — triangle indices
    target_verts : minimum desired vertex count

    Returns
    -------
    (vertices, faces) — subdivided mesh arrays
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)

    max_passes = 6  # safety cap: each pass ≈4× faces, 6 passes → ~4M faces

    for _ in range(max_passes):
        if len(v) >= target_verts:
            break
        if _LOOP_AVAILABLE:
            v, f = _subdivide_fn(v, f)
        else:
            v, f = _subdivide_fn(v, f)

    return v, f.astype(np.int32)


def subdivide_garment(garment: dict, target_verts: int) -> dict:
    """
    Subdivide a garment mesh and rebuild constraint arrays.

    Takes the garment dict from assemble_garment() and subdivides
    vertices/faces to reach target_verts. Recomputes stretch constraints
    (edges + rest lengths) from the new mesh topology. Seam constraints
    are cleared (they reference original vertex indices).

    Parameters
    ----------
    garment      : dict from assemble_garment()
    target_verts : minimum desired vertex count

    Returns
    -------
    Updated garment dict with subdivided mesh and new constraints
    """
    vertices = garment["vertices"]
    faces = garment["faces"]

    if len(vertices) >= target_verts:
        return garment  # already at or above target

    new_verts, new_faces = subdivide_to_target(vertices, faces, target_verts)

    # Rebuild stretch constraints from new mesh edges
    edge_set: set[tuple[int, int]] = set()
    stretch_edges: list[tuple[int, int, float]] = []

    for face in new_faces:
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key not in edge_set:
                edge_set.add(key)
                rest = float(np.linalg.norm(new_verts[b] - new_verts[a]))
                stretch_edges.append((key[0], key[1], rest))

    if stretch_edges:
        arr = np.array(stretch_edges, dtype=object)
        stretch_i = arr[:, 0].astype(np.int32)
        stretch_j = arr[:, 1].astype(np.int32)
        stretch_rest = arr[:, 2].astype(np.float64)
    else:
        stretch_i = np.array([], dtype=np.int32)
        stretch_j = np.array([], dtype=np.int32)
        stretch_rest = np.array([], dtype=np.float64)

    # Estimate area from new faces
    if len(new_faces) > 0:
        v0 = new_verts[new_faces[:, 0]]
        v1 = new_verts[new_faces[:, 1]]
        v2 = new_verts[new_faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        total_area = float(np.sum(np.linalg.norm(cross, axis=1))) * 0.5
    else:
        total_area = garment.get("total_area_m2", 0.1)

    result = dict(garment)  # shallow copy
    result["vertices"] = new_verts
    result["faces"] = new_faces
    result["total_area_m2"] = total_area
    result["stretch_i"] = stretch_i
    result["stretch_j"] = stretch_j
    result["stretch_rest"] = stretch_rest
    # Clear seam constraints (original vertex indices are invalid after subdivision)
    result["seam_i"] = np.array([], dtype=np.int32)
    result["seam_j"] = np.array([], dtype=np.int32)
    result["sleeve_seam_i"] = np.array([], dtype=np.int32)
    result["sleeve_seam_j"] = np.array([], dtype=np.int32)
    # Clear panel_vertex_ranges (no longer valid)
    result["panel_vertex_ranges"] = {}

    return result
