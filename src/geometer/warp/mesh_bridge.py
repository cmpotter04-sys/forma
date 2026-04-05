# Forma kernel: mesh_bridge.py
# Derived from:
#   - Forma Phase 1 garment assembly (src/geometer/garment_assembly.py)
#   - NVIDIA Warp official documentation (nvidia.github.io/warp/, Apache 2.0)
#   - Baraff & Witkin, "Large Steps in Cloth Simulation" (SIGGRAPH 1998)
#     for per-vertex mass from triangle area
# No code from NvidiaWarp-GarmentCode was referenced.

"""
src/geometer/warp/mesh_bridge.py

Data translation layer between Forma's numpy/trimesh world and Warp's
wp.array world.

This module handles:
  - Converting numpy vertex arrays to/from Warp vec3 arrays
  - Converting numpy face arrays to flattened Warp int32 arrays
  - Building Warp collision meshes from body mesh data
  - Computing per-vertex particle masses from triangle areas

All functions degrade gracefully when Warp is not installed, raising
ImportError with a clear message.
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    HAS_WARP = False


def _require_warp() -> None:
    """Raise ImportError with a clear message if warp is not available."""
    if not HAS_WARP:
        raise ImportError(
            "warp-lang is required for the Warp backend. "
            "Install it with: pip install warp-lang"
        )


def numpy_to_warp_verts(
    verts_np: np.ndarray,
    dtype=None,
    device: str = None,
) -> "wp.array":
    """
    Convert (N, 3) numpy float array to Warp vec3 array.

    Parameters
    ----------
    verts_np : (N, 3) numpy array — vertex positions
    dtype    : Warp float type (default: wp.float32)
    device   : Warp device string (default: wp.get_preferred_device())

    Returns
    -------
    wp.array of shape (N,) with dtype wp.vec3 (using the specified float type)
    """
    _require_warp()
    if dtype is None:
        dtype = wp.float32
    if device is None:
        device = str(wp.get_preferred_device())

    verts = np.ascontiguousarray(verts_np, dtype=np.float32 if dtype == wp.float32 else np.float64)
    vec_type = wp.vec3 if dtype == wp.float32 else wp.types.vector(3, dtype)

    return wp.array(verts, dtype=vec_type, device=device)


def numpy_to_warp_indices(
    faces_np: np.ndarray,
    device: str = None,
) -> "wp.array":
    """
    Convert (M, 3) numpy int32 face array to flattened Warp int32 array.

    Warp sim meshes use flat index arrays: [i0, j0, k0, i1, j1, k1, ...].

    Parameters
    ----------
    faces_np : (M, 3) numpy int32 array — triangle face indices
    device   : Warp device string (default: wp.get_preferred_device())

    Returns
    -------
    wp.array of shape (M*3,) with dtype wp.int32
    """
    _require_warp()
    if device is None:
        device = str(wp.get_preferred_device())

    flat = np.ascontiguousarray(faces_np.flatten(), dtype=np.int32)
    return wp.array(flat, dtype=wp.int32, device=device)


def warp_to_numpy_verts(
    warp_arr: "wp.array",
) -> np.ndarray:
    """
    Extract vertex positions from a Warp array back to (N, 3) numpy.

    Works with Warp state particle positions (wp.vec3 arrays).

    Parameters
    ----------
    warp_arr : wp.array of vec3

    Returns
    -------
    (N, 3) numpy float64 array
    """
    _require_warp()
    return warp_arr.numpy().astype(np.float64)


def build_warp_mesh(
    verts_np: np.ndarray,
    faces_np: np.ndarray,
    device: str = None,
) -> "wp.sim.Mesh":
    """
    Build a Warp sim Mesh suitable for collision detection.

    Parameters
    ----------
    verts_np  : (N, 3) numpy float array — body mesh vertices (metres)
    faces_np  : (M, 3) numpy int32 array — body mesh triangle faces
    device    : Warp device string

    Returns
    -------
    wp.sim.Mesh instance
    """
    _require_warp()
    import warp.sim

    # Warp 1.6.2: wp.sim.Mesh() expects raw numpy arrays, not wp.array.
    # It handles GPU transfer internally.
    verts = np.ascontiguousarray(verts_np, dtype=np.float32)
    indices = np.ascontiguousarray(faces_np.flatten(), dtype=np.int32)

    mesh = wp.sim.Mesh(
        vertices=verts,
        indices=indices,
    )
    return mesh


def compute_particle_masses(
    garment_verts: np.ndarray,
    garment_faces: np.ndarray,
    density_kg_m2: float,
) -> np.ndarray:
    """
    Compute per-vertex mass from triangle area and fabric density.

    Each vertex's mass = sum(area_of_adjacent_triangles / 3) × density.
    This distributes each triangle's mass equally to its 3 vertices.

    Parameters
    ----------
    garment_verts   : (N, 3) float — garment vertex positions (metres)
    garment_faces   : (M, 3) int   — triangle face indices
    density_kg_m2   : fabric surface density (kg/m²)

    Returns
    -------
    (N,) float array — per-vertex mass in kg
    """
    N = len(garment_verts)
    masses = np.zeros(N, dtype=np.float64)

    if len(garment_faces) == 0:
        # Fallback: uniform mass distribution
        total_area = 0.1  # m²
        return np.full(N, density_kg_m2 * total_area / max(N, 1))

    # Compute triangle areas
    v0 = garment_verts[garment_faces[:, 0]]
    v1 = garment_verts[garment_faces[:, 1]]
    v2 = garment_verts[garment_faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = np.linalg.norm(cross, axis=1) * 0.5  # (M,)

    # Distribute each triangle's mass to its 3 vertices
    tri_mass = tri_areas * density_kg_m2 / 3.0
    np.add.at(masses, garment_faces[:, 0], tri_mass)
    np.add.at(masses, garment_faces[:, 1], tri_mass)
    np.add.at(masses, garment_faces[:, 2], tri_mass)

    # Ensure no zero-mass vertices (can happen at isolated vertices)
    min_mass = density_kg_m2 * 1e-6
    masses = np.maximum(masses, min_mass)

    return masses
