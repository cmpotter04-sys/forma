# Forma kernel: body_collision.py
# Derived from:
#   - Macklin et al., "XPBD: Position-Based Simulation of Compliant
#     Constrained Dynamics" (MIG 2016)
#   - Forma Phase 1 body collision logic (src/geometer/xpbd_simulate.py)
#   - Baraff & Witkin, "Large Steps in Cloth Simulation" (SIGGRAPH 1998)
# No code from NvidiaWarp-GarmentCode was referenced.

"""
src/geometer/warp/kernels/body_collision.py

GPU kernel for body-surface collision resolution.

Phase 1 used a scipy KDTree nearest-vertex search to detect and resolve
garment-body penetration.  This module replaces that with a Warp kernel
that queries the body BVH (built from the triangulated surface) via
wp.mesh_query_point(), giving true nearest-surface-point resolution
rather than nearest-vertex resolution.

Public API
----------
build_body_collision(body_mesh_trimesh) -> wp.Mesh
    Convert a trimesh body mesh to a Warp BVH collision mesh.

resolve_body_collision_kernel  (wp.kernel)
    Per-particle kernel: find nearest body surface point, push particle
    out along the surface normal if penetration depth > 0.

apply_body_collision(particle_q, body_wp_mesh, margin_m, device) -> None
    Python helper that launches resolve_body_collision_kernel.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Lazy Warp import — mirrors the pattern in warp_simulate.py so this module
# can be imported on CPU-only machines without crashing.
# ---------------------------------------------------------------------------
try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    HAS_WARP = False


def _require_warp() -> None:
    if not HAS_WARP:
        raise ImportError(
            "warp-lang is required for the Warp backend. "
            "Install it with: pip install warp-lang"
        )


# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------
# The kernel is defined at module level so Warp's JIT compiler can cache the
# CUDA binary across calls.  The @wp.kernel decorator is applied only when
# Warp is available; the module-level guard ensures this never runs on a
# CPU-only machine.

if HAS_WARP:
    @wp.kernel
    def resolve_body_collision_kernel(
        particle_q:      wp.array(dtype=wp.vec3),   # (N,) particle positions — modified in-place
        body_mesh_id:    wp.uint64,                  # Warp BVH mesh handle
        margin_m:        float,                      # standoff distance in metres (e.g. 0.001)
        max_query_dist:  float,                      # BVH search radius in metres (e.g. 0.05)
    ):
        """
        For each garment particle, find the nearest point on the body surface
        using the Warp BVH and push the particle out if it is inside (or too
        close to) the surface.

        Penetration sign convention (matches Phase 1):
          - signed_dist < margin_m  →  particle is inside or within standoff
          - correction pushes particle along the surface normal to reach margin_m

        The surface normal is recovered from the barycentric coordinates of the
        nearest triangle: n = normalize(cross(e01, e02)) where e01/e02 are the
        triangle edge vectors.  The sign is oriented so that the normal points
        outward (away from the body interior), verified by checking that the
        dot product of (particle - nearest_point) and the normal is non-negative
        at the true exterior.

        Sources:
          - wp.mesh_query_point() — NVIDIA Warp docs, Apache 2.0
          - Baraff & Witkin 1998 §3 (velocity filter / normal recovery)
          - Macklin et al. 2016 §4 (position-level constraint)
          - Forma Phase 1: _apply_body_collision() in xpbd_simulate.py
        """
        tid = wp.tid()  # one thread per particle

        p = particle_q[tid]

        # --- Query BVH for nearest surface point ----------------------------
        # wp.mesh_query_point returns (hit, face_index, u, v, sign)
        # u, v are barycentric coordinates; sign > 0 outside, < 0 inside.
        # max_query_dist limits the BVH traversal radius; particles far from
        # the body are skipped cheaply.
        query_result = wp.mesh_query_point(body_mesh_id, p, max_query_dist)

        if not query_result.result:
            # Particle is farther than max_query_dist from the body — no action.
            return

        face_index = query_result.face
        u = query_result.u
        v = query_result.v

        # --- Recover nearest surface point from barycentric coords ----------
        # wp.mesh_eval_position interpolates the triangle at (u, v).
        nearest_pt = wp.mesh_eval_position(body_mesh_id, face_index, u, v)

        # --- Recover outward face normal from barycentric coords ------------
        # wp.mesh_eval_face_normal returns the geometric (area-weighted) normal
        # for the triangle at face_index.  The sign convention in Warp follows
        # the winding order stored in the mesh, which matches trimesh's
        # outward-facing convention for a closed watertight body mesh.
        face_normal = wp.mesh_eval_face_normal(body_mesh_id, face_index)

        # Ensure the normal is unit length (it is for well-formed meshes, but
        # degenerate triangles can produce near-zero normals).
        n_len = wp.length(face_normal)
        if n_len < 1.0e-10:
            return
        outward_normal = face_normal / n_len

        # --- Signed distance (positive = outside body, negative = inside) ---
        displacement = p - nearest_pt
        signed_dist = wp.dot(displacement, outward_normal)

        # --- Push particle out if inside or within standoff ------------------
        if signed_dist < margin_m:
            correction = (margin_m - signed_dist) * outward_normal
            particle_q[tid] = p + correction


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------

def build_body_collision(body_mesh_trimesh) -> "wp.Mesh":
    """
    Convert a trimesh body mesh into a Warp collision mesh (BVH).

    Parameters
    ----------
    body_mesh_trimesh : trimesh.Trimesh
        Body surface mesh, Y-up, metres.  Must be watertight for correct
        inside/outside sign detection.  The MakeHuman CC0 base mesh used by
        Forma satisfies this requirement.

    Returns
    -------
    wp.Mesh
        Warp BVH collision mesh.  Pass mesh.id (wp.uint64) to the kernel.

    Notes
    -----
    - Vertices are cast to float32 because Warp's BVH is single-precision.
    - Faces are flattened to a 1-D int32 array: [i0, j0, k0, i1, j1, k1, ...].
    - This function must be called BEFORE launching the kernel.  The returned
      wp.Mesh object owns the BVH; keep it alive for the duration of the sim.
    """
    _require_warp()
    import warp.sim  # noqa: F401  (ensures wp.sim.Mesh is available)

    verts = np.ascontiguousarray(body_mesh_trimesh.vertices, dtype=np.float32)
    indices = np.ascontiguousarray(
        body_mesh_trimesh.faces.flatten(), dtype=np.int32
    )

    # wp.sim.Mesh builds the BVH internally.  The constructor signature is
    # stable across Warp 1.x: Mesh(vertices, indices).
    mesh = wp.sim.Mesh(vertices=verts, indices=indices)
    return mesh


def apply_body_collision(
    particle_q:    "wp.array",
    body_wp_mesh:  "wp.Mesh",
    margin_m:      float = 0.001,
    max_query_dist: float = 0.05,
    device:        str = None,
) -> None:
    """
    Launch resolve_body_collision_kernel over all garment particles.

    Modifies particle_q in-place on the GPU.  Call this once per simulation
    step, after all constraint projections have been applied (same ordering as
    the Phase 1 CPU backend in _run_xpbd()).

    Parameters
    ----------
    particle_q      : wp.array(dtype=wp.vec3, shape=(N,))
        Current garment particle positions.  Modified in-place.
    body_wp_mesh    : wp.Mesh
        Warp BVH mesh built by build_body_collision().
    margin_m        : float
        Standoff distance in metres.  Particles closer than this to the body
        surface are pushed out.  Default 0.001 (1 mm) matches Phase 1.
    max_query_dist  : float
        BVH search radius.  Particles beyond this distance from the body are
        skipped.  Default 0.05 (5 cm) is well beyond the expected clearance
        range for a well-fitted garment while pruning distant particles cheaply.
    device          : str or None
        Warp device string.  Defaults to the device of particle_q.

    Raises
    ------
    ImportError  : if warp-lang is not installed.
    ValueError   : if particle_q is not a 1-D wp.vec3 array.
    """
    _require_warp()

    if particle_q.dtype != wp.vec3:
        raise ValueError(
            f"particle_q must have dtype wp.vec3, got {particle_q.dtype}"
        )

    if device is None:
        device = str(particle_q.device)

    n_particles = particle_q.shape[0]

    wp.launch(
        kernel=resolve_body_collision_kernel,
        dim=n_particles,
        inputs=[
            particle_q,
            body_wp_mesh.id,
            float(margin_m),
            float(max_query_dist),
        ],
        device=device,
    )

    # Synchronize so the caller can safely read particle_q on CPU if needed.
    wp.synchronize_device(device)
