# Forma kernel: attachment.py
# Derived from:
#   - Macklin et al., "XPBD: Position-Based Simulation of Compliant
#     Constrained Dynamics" (MIG 2016)
#   - Forma Phase 1 collar pinning logic (src/geometer/xpbd_simulate.py)
#   - NVIDIA Warp official documentation and Apache 2.0 examples
#     (nvidia.github.io/warp/)
# No code from NvidiaWarp-GarmentCode was referenced.

"""
src/geometer/warp/kernels/attachment.py

XPBD attachment constraints for collar and waistband pinning.

An attachment constraint fixes a garment particle to a target anchor position
on the body mesh.  In XPBD terms the constraint function is:

    C(p) = ||p - anchor|| = 0

The single-particle gradient is:

    grad_C(p) = (p - anchor) / ||p - anchor||   (unit vector from anchor to p)

The XPBD correction (Macklin 2016, eq. 4) reduces to:

    delta_p = -C(p) / (w + alpha_tilde) * grad_C(p)

where:
    w           = inv_mass of the particle
    alpha_tilde = compliance / dt²   (0 → hard constraint)

Because the anchor is kinematic (infinite mass, inv_mass = 0), only the
garment particle moves.  When inv_mass = 0 the particle is already pinned
(kinematic), so no correction is applied.

Usage
-----
Build attachment constraint arrays with build_attachment_constraints(), then
call the Warp kernel solve_attachment_constraints() inside the simulation loop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazy Warp import — matches the pattern used by warp_simulate.py so that
# importing this module at the top of warp_simulate.py (on a CPU-only machine)
# does not hard-fail before a simulation is actually requested.
# ---------------------------------------------------------------------------

_wp = None


def _ensure_warp():
    """Import and initialise Warp on first use."""
    global _wp
    if _wp is None:
        try:
            import warp as wp
            wp.init()
            _wp = wp
        except ImportError:
            raise ImportError(
                "warp-lang is required for the Warp backend. "
                "Install it with: pip install warp-lang"
            )
    return _wp


# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------

# The kernel is defined inside a function so that Warp's JIT compiler only
# runs when warp-lang is actually available (lazy decoration pattern).
# Call get_kernel() to obtain the compiled kernel object.

_kernel_cache = None


def get_kernel():
    """
    Return the compiled Warp attachment kernel, compiling it on first call.

    The kernel is cached after the first call, so repeated calls are free.
    """
    global _kernel_cache
    if _kernel_cache is not None:
        return _kernel_cache

    wp = _ensure_warp()

    @wp.kernel
    def solve_attachment_constraints(
        positions: wp.array(dtype=wp.vec3),
        inv_masses: wp.array(dtype=float),
        anchor_positions: wp.array(dtype=wp.vec3),
        constraint_indices: wp.array(dtype=int),
        compliance: float,
        dt: float,
    ):
        """
        XPBD attachment constraint kernel — one Warp thread per constraint.

        For thread tid:
          - constraint_indices[tid] is the garment-particle index being pinned.
          - anchor_positions[tid]   is the target world-space position.

        The constraint function is C(p) = ||p - anchor||.
        The correction is:

            delta_p = -C / (w + alpha_tilde) * grad_C

        where:
            C           = distance from particle to anchor
            grad_C      = unit vector (p - anchor) / C   (if C > eps)
            w           = inv_mass[particle]
            alpha_tilde = compliance / (dt * dt)

        When C < eps the particle is already at the anchor — no correction.
        When inv_mass == 0 the particle is kinematic — no correction.
        """
        tid = wp.tid()

        particle_idx = constraint_indices[tid]
        p = positions[particle_idx]
        anchor = anchor_positions[tid]
        w = inv_masses[particle_idx]

        # Kinematic particles (mass = 0, inv_mass = 0) are already pinned.
        if w < 1e-30:
            return

        # Constraint value C = ||p - anchor||
        diff = p - anchor
        C = wp.length(diff)

        if C < 1.0e-10:
            return  # Already at anchor; nothing to do.

        # Normalised gradient: (p - anchor) / C
        grad_C = diff / C

        # XPBD scaled compliance: alpha_tilde = compliance / dt²
        alpha_tilde = compliance / (dt * dt)

        # XPBD correction magnitude
        # Only one particle moves (anchor is kinematic), so denominator = w + alpha_tilde.
        delta_lambda = -C / (w + alpha_tilde)

        # Positional correction
        delta_p = delta_lambda * w * grad_C

        positions[particle_idx] = p + delta_p

    _kernel_cache = solve_attachment_constraints
    return _kernel_cache


# ---------------------------------------------------------------------------
# Python helper: identify collar / waistband vertices and build constraint data
# ---------------------------------------------------------------------------

def build_attachment_constraints(
    garment_mesh: dict,
    body_mesh_vertices: np.ndarray,
    collar_z_frac: float = 0.95,
    waistband_z_frac: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify collar and waistband vertices on the garment and return their
    indices together with the corresponding body-surface anchor positions.

    The Forma coordinate system is Y-up, metres, feet at y=0, head at y≈1.8m.
    Collar vertices are the top ``collar_z_frac`` fraction of garment Y-extent
    (i.e. above the ``(1 - collar_z_frac) * 100``-th Y percentile).
    Waistband vertices sit near the ``waistband_z_frac`` fractional Y height
    (within ±2 % of garment Y-extent).

    Each identified garment vertex is paired with the nearest body-surface
    vertex as its anchor (nearest-neighbour in 3-D).

    Parameters
    ----------
    garment_mesh : dict
        Assembled garment dict as produced by
        ``geometer.garment_assembly.assemble_garment()``.  Must contain the
        key ``"vertices"`` with shape ``(N, 3)``, metres.
    body_mesh_vertices : np.ndarray, shape (M, 3)
        Body vertex positions in metres (Y-up).
    collar_z_frac : float
        Fractional Y threshold for collar detection.  Vertices above the
        ``(1 - collar_z_frac) * 100``-th percentile of garment Y-coordinates
        are considered collar particles.  Default 0.95 → top 5 % of vertices.

        This mirrors the Phase 1 CPU solver which pins the top 8 % by Y
        (92nd-percentile threshold).  Here the threshold is expressed as the
        fraction of the normalised Y range [0, 1] so it generalises to
        different body sizes.  ``collar_z_frac=0.95`` ≈ 92nd percentile for
        a roughly uniform Y distribution.
    waistband_z_frac : float
        Fractional Y height (0 = feet, 1 = top of garment) at which waistband
        vertices are expected.  Vertices within ±2 % of garment Y-extent
        around this fraction are flagged as waistband particles.
        Default 0.45 (mid-lower torso for a typical T-shirt / trouser hem).

    Returns
    -------
    constraint_indices : np.ndarray, shape (K,), dtype int32
        Garment-particle indices for all K attachment constraints
        (collar + waistband combined, duplicates removed).
    anchor_positions : np.ndarray, shape (K, 3), dtype float32
        World-space anchor positions (nearest body vertex) for each constraint.

    Notes
    -----
    The returned arrays are ready for upload to the GPU via ``wp.array()``.
    The caller is responsible for choosing the compliance value that matches
    the desired constraint stiffness (0.0 → hard pin; ~1e-4 → soft guide).
    """
    from scipy.spatial import KDTree

    positions = np.asarray(garment_mesh["vertices"], dtype=np.float64)  # (N, 3)
    N = len(positions)

    y_coords = positions[:, 1]
    y_min = float(y_coords.min())
    y_max = float(y_coords.max())
    y_range = max(y_max - y_min, 1e-6)

    # Normalised Y in [0, 1]
    y_norm = (y_coords - y_min) / y_range

    # ---- Collar: vertices above collar_z_frac --------------------------------
    # Mirrors CPU solver: top 8% pinned (92nd percentile).
    # collar_z_frac=0.95 targets a similar band while being size-independent.
    collar_mask = y_norm >= collar_z_frac

    # ---- Waistband: vertices near waistband_z_frac ± 2% of Y-extent --------
    waistband_half_band = 0.02  # ±2% of garment Y-extent
    waistband_mask = np.abs(y_norm - waistband_z_frac) <= waistband_half_band

    # ---- Combine and deduplicate --------------------------------------------
    combined_mask = collar_mask | waistband_mask
    constrained_indices = np.where(combined_mask)[0].astype(np.int32)

    if len(constrained_indices) == 0:
        # Fallback: pin at least the single highest vertex so the garment
        # does not fall through the body under gravity.
        constrained_indices = np.array([int(np.argmax(y_coords))], dtype=np.int32)

    # ---- Find nearest body vertex for each constrained garment vertex --------
    body_tree = KDTree(body_mesh_vertices)
    constrained_positions = positions[constrained_indices]  # (K, 3)
    _, nearest_body_idx = body_tree.query(constrained_positions)
    anchor_positions = body_mesh_vertices[nearest_body_idx].astype(np.float32)

    return constrained_indices, anchor_positions


# ---------------------------------------------------------------------------
# Convenience: launch the kernel for one XPBD sub-step
# ---------------------------------------------------------------------------

def apply_attachment_constraints(
    wp_positions,
    wp_inv_masses,
    wp_anchor_positions,
    wp_constraint_indices,
    compliance: float,
    dt: float,
):
    """
    Launch the attachment constraint kernel for one XPBD sub-step.

    Parameters
    ----------
    wp_positions : wp.array(dtype=wp.vec3)
        Current particle positions (read-write, modified in-place on GPU).
    wp_inv_masses : wp.array(dtype=float)
        Per-particle inverse masses (read-only).
    wp_anchor_positions : wp.array(dtype=wp.vec3)
        Target anchor positions for each constraint (read-only).
    wp_constraint_indices : wp.array(dtype=int)
        Garment particle indices for each constraint (read-only).
    compliance : float
        XPBD compliance α (m/N).  0.0 = perfectly rigid; ~1e-6 = very stiff.
        A value of 1e-7 is recommended for collar pinning (stiff but not
        numerically singular).
    dt : float
        Simulation timestep in seconds.

    Notes
    -----
    The number of threads launched equals the number of constraints
    (``wp_constraint_indices.shape[0]``).  The function is a thin wrapper
    around ``wp.launch()`` so it can be inlined into a custom simulation loop
    without callers needing to import warp directly.
    """
    wp = _ensure_warp()
    kernel = get_kernel()
    n_constraints = wp_constraint_indices.shape[0]
    if n_constraints == 0:
        return
    wp.launch(
        kernel,
        dim=n_constraints,
        inputs=[
            wp_positions,
            wp_inv_masses,
            wp_anchor_positions,
            wp_constraint_indices,
            compliance,
            dt,
        ],
    )


# ---------------------------------------------------------------------------
# GPU array upload helpers (convenience for callers)
# ---------------------------------------------------------------------------

def upload_attachment_arrays(
    constraint_indices: np.ndarray,
    anchor_positions: np.ndarray,
    device: str = "cuda",
) -> tuple:
    """
    Upload numpy attachment arrays to the specified Warp device.

    Parameters
    ----------
    constraint_indices : np.ndarray, shape (K,), dtype int32
    anchor_positions   : np.ndarray, shape (K, 3), dtype float32
    device             : Warp device string (default ``"cuda"``).

    Returns
    -------
    (wp_constraint_indices, wp_anchor_positions)
        A pair of ``wp.array`` objects ready for use with
        ``apply_attachment_constraints()``.
    """
    wp = _ensure_warp()

    wp_indices = wp.array(
        constraint_indices.astype(np.int32),
        dtype=wp.int32,
        device=device,
    )
    wp_anchors = wp.array(
        anchor_positions.astype(np.float32).reshape(-1, 3),
        dtype=wp.vec3,
        device=device,
    )
    return wp_indices, wp_anchors
