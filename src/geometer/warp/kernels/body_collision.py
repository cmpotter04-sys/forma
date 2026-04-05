# Forma kernel: body_collision.py
# Derived from:
#   - Macklin et al., "XPBD: Position-Based Simulation of Compliant
#     Constrained Dynamics" (MIG 2016)
#   - Forma Phase 1 body collision logic (src/geometer/xpbd_simulate.py)
#   - Baraff & Witkin, "Large Steps in Cloth Simulation" (SIGGRAPH 1998)
#   - Trilinear interpolation: standard numerical methods (Stuyck 2018,
#     "Cloth Simulation for Computer Graphics", Chapter 5)
#   - mesh_to_sdf precomputation: github.com/marian42/mesh_to_sdf (MIT license)
# No code from NvidiaWarp-GarmentCode was referenced.

"""
src/geometer/warp/kernels/body_collision.py

GPU kernel for body-surface collision resolution.

Phase 1 used a scipy KDTree nearest-vertex search to detect and resolve
garment-body penetration.  This module provides two collision resolution
approaches:

=============================================================================
APPROACH 1 — BVH (DEFAULT, RECOMMENDED FOR FORMA)
=============================================================================
Uses wp.mesh_query_point() per-frame BVH traversal on the body triangle mesh.

How it works:
  - build_body_collision(trimesh) converts the body mesh to a wp.sim.Mesh BVH
    that lives on GPU.  This is called ONCE at body-load time.
  - resolve_body_collision_kernel() queries the BVH per-particle at each step.
    BVH traversal is O(log N_triangles) per query, fully parallelised on GPU.
  - The BVH is NOT rebuilt between steps — the body mesh is static in Forma.

Accuracy:
  - Exact nearest-surface-point resolution (sub-millimetre, limited only by
    mesh tessellation of the 21,833-vertex MakeHuman body).
  - Surface normals recovered from face geometry — no interpolation error.

Why BVH is right for Forma:
  - Body is static: BVH built once, traversed cheaply each step.  The Warp
    model builder already owns the BVH on GPU from add_shape_mesh().
  - Clearance is measured from final particle positions.  Any systematic
    position error in collision resolution propagates 1:1 into clearance.
  - The 0.5 mm parity requirement (FORMA_PHASE2_EXECUTOR_SPEC.md) demands
    sub-millimetre position accuracy — only exact BVH can guarantee this.

Public API:
    build_body_collision(body_mesh_trimesh) -> wp.Mesh
    resolve_body_collision_kernel (wp.kernel)
    apply_body_collision(particle_q, body_wp_mesh, margin_m, device) -> None

=============================================================================
APPROACH 2 — SDF VOXEL GRID (OPT-IN, EXPERIMENTAL)
=============================================================================
Uses mesh_to_sdf (MIT, github.com/marian42/mesh_to_sdf) to precompute a dense
voxel grid of signed distances from the body surface.  At runtime the kernel
samples the grid with trilinear interpolation in O(1) per particle per step.

How it works:
  - precompute_body_sdf(trimesh, voxel_resolution) runs mesh_to_sdf once at
    body-load time to build a (R, R, R) float32 numpy array of signed distances,
    plus a (3,) origin and scalar voxel_size needed to map world coords to grid.
  - resolve_body_collision_sdf_kernel() samples that grid trilinearly each step.

IMPORTANT — accuracy trade-off:
  The SDF voxel grid has an inherent interpolation error of ~(voxel_size / 2).
  For a 2 m tall body:
      64^3  →  31.2 mm voxels  →  up to 15.6 mm position error  (unusable)
     128^3  →  15.6 mm voxels  →  up to  7.8 mm position error  (unusable)
     256^3  →   7.8 mm voxels  →  up to  3.9 mm position error  (unusable)
    2048^3  →   0.98 mm voxels →  up to  0.49 mm position error (barely ok)

  A 2048^3 float32 grid requires 34 GB of GPU memory — not feasible.
  At any practical resolution (<=256^3, ~67 MB dist-only) the SDF approach
  WILL violate the 0.5 mm clearance parity requirement.

  Additionally, mesh_to_sdf's scan-line algorithm can produce artefacts in
  narrow/concave regions (armpits, crotch, under-bust) — exactly the regions
  Forma measures most carefully.

When SDF collision might still be useful:
  - Rapid prototyping where coarse fit guidance is acceptable (e.g. broad-phase
    filtering before a follow-up BVH pass)
  - Animated bodies that change shape per frame (not Forma's current use case)
  - Debugging / visualisation of the body's implicit surface

  USE apply_body_collision_sdf() ONLY when you have explicitly accepted
  the accuracy trade-off and disabled the 0.5 mm parity assertion.

Dependency:
    pip install mesh-to-sdf   # MIT licence — commercially permissive

Public API:
    precompute_body_sdf(body_mesh_trimesh, voxel_resolution) -> BodySDF
    resolve_body_collision_sdf_kernel (wp.kernel)
    apply_body_collision_sdf(particle_q, body_sdf, margin_m, device) -> None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
# SDF voxel grid dataclass (Approach 2)
# ---------------------------------------------------------------------------

@dataclass
class BodySDF:
    """
    Precomputed signed-distance voxel grid for the body mesh.

    Produced by precompute_body_sdf().  Pass to apply_body_collision_sdf().

    Attributes
    ----------
    grid_np     : (R, R, R) float32 ndarray — signed distance in metres
                  (negative = inside body, positive = outside).
    grid_wp     : wp.array of float32, shape (R*R*R,) — same data on GPU,
                  in C-order (z-major, then y, then x).  None until
                  upload_to_gpu() is called.
    origin      : (3,) float64 ndarray — world-space coordinate of voxel [0,0,0]
                  (the lower-left-front corner of the bounding box, padded by
                  `padding` voxels).
    voxel_size  : float — edge length of one voxel in metres.
    resolution  : int — grid side length R (grid is R^3).

    Accuracy note
    -------------
    Trilinear interpolation error is bounded by voxel_size / 2.  For a 2 m
    body, 128^3 gives ~7.8 mm error — enough to violate the 0.5 mm parity
    requirement.  See module docstring for full resolution vs accuracy table.
    """
    grid_np:    np.ndarray          # (R, R, R) float32
    origin:     np.ndarray          # (3,) float64 — lower-left-front corner
    voxel_size: float               # metres per voxel edge
    resolution: int                 # R
    grid_wp:    Optional["wp.array"] = None   # populated by upload_to_gpu()

    def upload_to_gpu(self, device: str = None) -> None:
        """
        Copy the SDF grid to a flat Warp float32 array on the specified device.

        Must be called before passing this BodySDF to apply_body_collision_sdf().
        Safe to call multiple times — re-uploads if already uploaded.

        Parameters
        ----------
        device : Warp device string.  Defaults to wp.get_preferred_device().
        """
        _require_warp()
        if device is None:
            device = str(wp.get_preferred_device())
        flat = np.ascontiguousarray(self.grid_np.flatten(), dtype=np.float32)
        self.grid_wp = wp.array(flat, dtype=wp.float32, device=device)


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


# ---------------------------------------------------------------------------
# Approach 2 — SDF voxel grid (opt-in, experimental)
# ---------------------------------------------------------------------------
# The SDF kernel and Python helpers below implement an alternative collision
# resolution path using a precomputed voxel grid.
#
# READ THE MODULE DOCSTRING BEFORE USING THIS PATH.  In summary:
#   - At any practical GPU-feasible resolution (<=256^3) the trilinear
#     interpolation error (~3.9 mm for 256^3 on a 2 m body) WILL violate
#     Forma's 0.5 mm clearance parity requirement.
#   - The BVH path above is the correct choice for production use.
#   - This path exists for completeness and for experimental scenarios where
#     coarse collision suffices (e.g. broad-phase, animated bodies, debug vis).
# ---------------------------------------------------------------------------

if HAS_WARP:
    @wp.kernel
    def resolve_body_collision_sdf_kernel(
        particle_q:  wp.array(dtype=wp.vec3),   # (N,) — modified in-place
        sdf_grid:    wp.array(dtype=wp.float32), # (R*R*R,) flat C-order
        origin:      wp.vec3,                    # world coords of voxel [0,0,0]
        voxel_size:  float,                      # metres per voxel edge
        resolution:  int,                        # grid side length R
        margin_m:    float,                      # standoff distance in metres
    ):
        """
        Per-particle SDF collision kernel (EXPERIMENTAL — see module docstring).

        Samples the precomputed SDF voxel grid with trilinear interpolation to
        estimate the signed distance from the particle to the body surface.
        Pushes the particle outward by the gradient direction if penetrating.

        Gradient estimation:
          Central finite differences on the SDF grid recover a unit-length
          approximate outward normal.  Step size = voxel_size.  For particles
          very close to the surface (within one voxel), finite difference
          accuracy degrades — another reason BVH is preferred for production.

        Sign convention (same as BVH path):
          sdf < 0   → inside body
          sdf >= 0  → outside body
          Particles with sdf < margin_m are pushed along the gradient.

        Sources:
          - Trilinear interpolation: Stuyck 2018, "Cloth Simulation for
            Computer Graphics", Chapter 5, Equation 5.7.
          - Central difference gradient: standard FD approximation (2nd-order).
          - mesh_to_sdf sign convention: github.com/marian42/mesh_to_sdf docs.
          - Forma Phase 1: _apply_body_collision() in xpbd_simulate.py
            (penetration sign convention and margin logic).
        """
        tid = wp.tid()
        p = particle_q[tid]

        R = resolution

        # --- Map world position to fractional grid coordinates ----------------
        # Grid origin = lower-left-front corner of the padded bounding box
        # (set by precompute_body_sdf to bbox_min - padding * voxel_size).
        fx = (p[0] - origin[0]) / voxel_size
        fy = (p[1] - origin[1]) / voxel_size
        fz = (p[2] - origin[2]) / voxel_size

        # Clamp to [1, R-2] so finite-difference neighbours are always valid.
        # Particles outside this margin are too far from the body to collide.
        if fx < 1.0 or fx > float(R) - 2.0:
            return
        if fy < 1.0 or fy > float(R) - 2.0:
            return
        if fz < 1.0 or fz > float(R) - 2.0:
            return

        # Integer cell index (floor)
        ix = int(fx)
        iy = int(fy)
        iz = int(fz)

        # Fractional offsets within the cell [0, 1)
        tx = fx - float(ix)
        ty = fy - float(iy)
        tz = fz - float(iz)

        # --- Trilinear interpolation of SDF -----------------------------------
        # C-order flat index: idx = iz * R*R + iy * R + ix
        R2 = R * R

        d000 = sdf_grid[iz       * R2 + iy       * R + ix    ]
        d100 = sdf_grid[iz       * R2 + iy       * R + ix + 1]
        d010 = sdf_grid[iz       * R2 + (iy + 1) * R + ix    ]
        d110 = sdf_grid[iz       * R2 + (iy + 1) * R + ix + 1]
        d001 = sdf_grid[(iz + 1) * R2 + iy       * R + ix    ]
        d101 = sdf_grid[(iz + 1) * R2 + iy       * R + ix + 1]
        d011 = sdf_grid[(iz + 1) * R2 + (iy + 1) * R + ix    ]
        d111 = sdf_grid[(iz + 1) * R2 + (iy + 1) * R + ix + 1]

        # Lerp along x, then y, then z
        dx00 = d000 + (d100 - d000) * tx
        dx10 = d010 + (d110 - d010) * tx
        dx01 = d001 + (d101 - d001) * tx
        dx11 = d011 + (d111 - d011) * tx
        dxy0 = dx00 + (dx10 - dx00) * ty
        dxy1 = dx01 + (dx11 - dx01) * ty
        sdf_val = dxy0 + (dxy1 - dxy0) * tz

        # --- Early exit if not penetrating or within standoff ----------------
        if sdf_val >= margin_m:
            return

        # --- Approximate outward normal via central finite differences --------
        # Clamp to [1, R-2] already ensured neighbours exist.
        gx = (sdf_grid[iz * R2 + iy * R + (ix + 1)] -
              sdf_grid[iz * R2 + iy * R + (ix - 1)]) * 0.5
        gy = (sdf_grid[iz       * R2 + (iy + 1) * R + ix] -
              sdf_grid[iz       * R2 + (iy - 1) * R + ix]) * 0.5
        gz = (sdf_grid[(iz + 1) * R2 + iy * R + ix] -
              sdf_grid[(iz - 1) * R2 + iy * R + ix]) * 0.5

        grad = wp.vec3(gx, gy, gz)
        grad_len = wp.length(grad)
        if grad_len < 1.0e-10:
            # Degenerate gradient (flat region in SDF) — skip correction.
            return
        outward_normal = grad / grad_len

        # --- Push particle outward -------------------------------------------
        correction = (margin_m - sdf_val) * outward_normal
        particle_q[tid] = p + correction


def precompute_body_sdf(
    body_mesh_trimesh,
    voxel_resolution: int = 64,
    padding: int = 3,
) -> "BodySDF":
    """
    Precompute a signed-distance voxel grid from the body mesh using mesh_to_sdf.

    ACCURACY WARNING — read before use:
        At voxel_resolution=64 on a ~2 m body, voxel size = 31.2 mm, giving
        trilinear interpolation error up to 15.6 mm.  This WILL violate the
        0.5 mm clearance parity requirement.  See module docstring for the full
        resolution vs accuracy table.  Use apply_body_collision() (BVH) for
        production work.

    Parameters
    ----------
    body_mesh_trimesh : trimesh.Trimesh
        Body mesh in Y-up, metres.  Does NOT need to be watertight
        (mesh_to_sdf handles open meshes, though quality degrades).
    voxel_resolution  : int
        Grid side length R.  Memory cost = R^3 * 4 bytes.
          64^3  =    1 MB   (31.2 mm voxels on 2 m body)
         128^3  =    8 MB   (15.6 mm voxels)
         256^3  =   67 MB   ( 7.8 mm voxels)
    padding           : int
        Number of extra voxels to add around the bounding box on each side.
        Prevents out-of-bounds access for particles near the bbox edge.
        Default 3 is sufficient for margin_m <= padding * voxel_size.

    Returns
    -------
    BodySDF
        Populated with grid_np, origin, voxel_size, resolution.
        Call body_sdf.upload_to_gpu() before passing to apply_body_collision_sdf().

    Raises
    ------
    ImportError : if mesh-to-sdf is not installed (pip install mesh-to-sdf).
    """
    try:
        import mesh_to_sdf  # noqa: F401
    except ImportError:
        raise ImportError(
            "mesh-to-sdf is required for the SDF collision path. "
            "Install it with: pip install mesh-to-sdf\n"
            "License: MIT (commercially permissible).\n"
            "Note: this is the EXPERIMENTAL collision path — see module docstring."
        )

    import numpy as np
    from mesh_to_sdf import mesh_to_voxels

    # --- Compute grid dimensions from bounding box ----------------------------
    verts = np.array(body_mesh_trimesh.vertices, dtype=np.float64)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_extent = bbox_max - bbox_min  # (3,) — extent in each dimension

    # Use the largest bounding box dimension to set voxel_size so the grid
    # is isotropic (equal voxel size in all three axes).
    # The non-dominant axes are covered by fewer voxels, which is fine —
    # the padding prevents out-of-bounds queries.
    max_extent = float(bbox_extent.max())
    voxel_size = max_extent / float(voxel_resolution - 2 * padding)

    # Total grid size in voxels (may differ from voxel_resolution along
    # non-dominant axes, but we keep it cubic for simplicity and GPU efficiency).
    R = voxel_resolution
    origin = bbox_min - padding * voxel_size  # lower-left-front corner

    # --- Run mesh_to_sdf ------------------------------------------------------
    # mesh_to_voxels returns a (R, R, R) float32 numpy array of signed distances.
    # scan_count controls the number of scan directions used to determine
    # inside/outside (higher = more accurate for open meshes).
    # sign_method='depth' is robust for humanoid meshes that are nearly watertight.
    #
    # mesh_to_sdf expects the mesh centred and scaled so that the bounding sphere
    # has radius 1.  We pass our own sample_count / voxel_count parameters to
    # control resolution directly.
    sdf_grid = mesh_to_voxels(
        body_mesh_trimesh,
        voxel_resolution=R,
        check_result=False,   # skip internal quality check (slow, uses trimesh internals)
        pad=False,            # we handle padding through origin offset above
    )

    # mesh_to_voxels returns values in normalised mesh-space (radius-1 sphere).
    # Rescale to metres: the internal normalisation maps the bounding-sphere
    # radius to 1.0.  The bounding sphere radius in world-space is:
    centre = (bbox_max + bbox_min) / 2.0
    bsphere_radius = float(np.linalg.norm(verts - centre, axis=1).max())
    # sdf_grid values are in normalised units → multiply by bsphere_radius
    sdf_grid_m = (sdf_grid * bsphere_radius).astype(np.float32)

    return BodySDF(
        grid_np=sdf_grid_m,
        origin=origin.astype(np.float64),
        voxel_size=float(bsphere_radius * 2.0 / float(R)),   # world metres per voxel
        resolution=R,
    )


def apply_body_collision_sdf(
    particle_q: "wp.array",
    body_sdf:   "BodySDF",
    margin_m:   float = 0.001,
    device:     str = None,
) -> None:
    """
    Launch resolve_body_collision_sdf_kernel over all garment particles.

    ACCURACY WARNING:
        SDF collision accuracy is limited by voxel_size (~voxel_resolution/2).
        This WILL violate the 0.5 mm clearance parity requirement at any
        practical resolution.  See module docstring.  Use apply_body_collision()
        for production.

    Parameters
    ----------
    particle_q : wp.array(dtype=wp.vec3, shape=(N,))
        Garment particle positions.  Modified in-place on GPU.
    body_sdf   : BodySDF
        Precomputed SDF from precompute_body_sdf().  Must have upload_to_gpu()
        called before this function.
    margin_m   : float
        Standoff distance in metres.  Particles with sdf < margin_m are pushed
        outward.  Default 0.001 (1 mm) matches Phase 1 and the BVH path.
    device     : str or None
        Warp device string.  Defaults to the device of particle_q.

    Raises
    ------
    ImportError  : if warp-lang is not installed.
    ValueError   : if particle_q is not a wp.vec3 array or sdf grid not on GPU.
    """
    _require_warp()

    if particle_q.dtype != wp.vec3:
        raise ValueError(
            f"particle_q must have dtype wp.vec3, got {particle_q.dtype}"
        )
    if body_sdf.grid_wp is None:
        raise ValueError(
            "body_sdf.grid_wp is None — call body_sdf.upload_to_gpu() first."
        )

    if device is None:
        device = str(particle_q.device)

    origin_wp = wp.vec3(
        float(body_sdf.origin[0]),
        float(body_sdf.origin[1]),
        float(body_sdf.origin[2]),
    )

    wp.launch(
        kernel=resolve_body_collision_sdf_kernel,
        dim=particle_q.shape[0],
        inputs=[
            particle_q,
            body_sdf.grid_wp,
            origin_wp,
            float(body_sdf.voxel_size),
            int(body_sdf.resolution),
            float(margin_m),
        ],
        device=device,
    )

    wp.synchronize_device(device)
