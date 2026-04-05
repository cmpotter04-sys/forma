"""
src/geometer/xpbd_simulate.py

Full Geometer pipeline: body mesh + GarmentCode pattern → sim_result dict
consumed by generate_verdict().

Pipeline (per FORMA_GEOMETER_SPEC.md):
  1. Load body mesh + classify body vertices into regions
  2. Load GarmentCode pattern + seam manifest
  3. Assemble garment: triangulate 2D panels → place in 3D → build constraints
  4. Run pure-Python XPBD simulation (numpy/scipy, no NvidiaWarp)
  5. Assign draped garment vertices to body regions
  6. Compute signed clearance per region
  7. Detect tunnel-through; raise if > 2%
  8. Return sim_result dict

Coordinate system:
  Body: Y-up, metres. Feet at y=0, head at y=1.8m. Facing +Z. Wearer's left = +X.
  Pattern: cm units. GarmentCode translation/rotation fields give 3D placement.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, KDTree

# Insert src/ so we can import sibling packages
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pattern_maker.load_patterns import load_pattern
from tailor.seam_converter import load_and_validate_manifest, SeamValidationError
from geometer.region_map import classify_body_vertices, assign_garment_to_body_regions
from geometer.clearance import (
    compute_region_clearance,
    detect_tunnel_through,
    classify_severity,
    classify_ease,
)
from geometer.convergence import check_convergence, SimulationExplosionError

# Shared garment assembly and post-simulation analysis (Phase 2 extraction)
from geometer.garment_assembly import (
    assemble_garment as _assemble_garment,
    project_garment_onto_body as _project_garment_onto_body,
    compute_strain_ratios as _compute_strain_ratios,
    REQUIRED_REGIONS,
)


# ---------------------------------------------------------------------------
# XPBD core — vectorized numpy
# ---------------------------------------------------------------------------

def _solve_distance_constraints_batch(
    positions: np.ndarray,
    inv_masses: np.ndarray,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    rest_lengths: np.ndarray,
    compliance_alpha: float,
    max_correction_m: float = 0.05,
) -> None:
    """
    Apply one pass of XPBD distance constraint corrections in-place.

    Uses averaged Jacobi: corrections are accumulated per vertex then divided
    by the constraint count, preventing explosion when a vertex participates
    in many simultaneous constraints.  Each correction is also capped at
    max_correction_m to guard against extreme initial gaps.

    positions        : (N, 3) — modified in-place
    inv_masses       : (N,)   — 1/mass per vertex
    edges_i, edges_j : (C,)  — vertex index pairs
    rest_lengths     : (C,)   — rest distance per constraint
    compliance_alpha : α̃ = compliance / dt² (dimensionless; 0 = hard)
    max_correction_m : per-vertex correction cap in metres (default 5 cm)
    """
    if len(edges_i) == 0:
        return

    N = len(positions)
    d = positions[edges_j] - positions[edges_i]        # (C, 3)
    dist = np.linalg.norm(d, axis=1)                   # (C,)
    valid = dist > 1e-10
    if not np.any(valid):
        return

    dist_safe = np.where(valid, dist, 1.0)
    d_norm = d / dist_safe[:, np.newaxis]              # (C, 3)

    C = dist - rest_lengths
    wi = inv_masses[edges_i]
    wj = inv_masses[edges_j]
    w_sum = wi + wj + compliance_alpha

    C = np.where(valid, C, 0.0)
    w_sum = np.where(w_sum > 1e-30, w_sum, 1.0)

    delta_lambda = -C / w_sum                          # (C,)
    corr = delta_lambda[:, np.newaxis] * d_norm        # (C, 3)

    corr_i = -corr * wi[:, np.newaxis]                 # (C, 3)
    corr_j =  corr * wj[:, np.newaxis]                 # (C, 3)

    # Accumulate corrections and constraint counts per vertex
    corr_sum = np.zeros((N, 3), dtype=float)
    count    = np.zeros(N, dtype=float)

    np.add.at(corr_sum, edges_i, corr_i)
    np.add.at(corr_sum, edges_j, corr_j)
    np.add.at(count, edges_i, 1.0)
    np.add.at(count, edges_j, 1.0)

    # Average: divide accumulated correction by number of contributing constraints
    safe_count = np.where(count > 0, count, 1.0)
    avg_corr = corr_sum / safe_count[:, np.newaxis]

    # Cap per-vertex correction magnitude
    mag = np.linalg.norm(avg_corr, axis=1, keepdims=True)
    scale = np.where(mag > max_correction_m, max_correction_m / (mag + 1e-30), 1.0)
    avg_corr *= scale

    positions += avg_corr


def _apply_body_collision(
    positions: np.ndarray,
    body_tree: KDTree,
    body_vertices: np.ndarray,
    body_normals: np.ndarray,
    margin_m: float = 0.001,
) -> None:
    """
    Push any garment vertex that has penetrated the body surface back outside.
    Modifies positions in-place.
    """
    dists, nearest_idx = body_tree.query(positions)

    nearest_pts = body_vertices[nearest_idx]      # (N, 3)
    nearest_nrm = body_normals[nearest_idx]       # (N, 3)

    displacements = positions - nearest_pts       # (N, 3)
    signed_dists = np.einsum("ij,ij->i", displacements, nearest_nrm)  # (N,)

    # Penetrating vertices: signed_dist < margin
    penetrating = signed_dists < margin_m
    if not np.any(penetrating):
        return

    correction = (margin_m - signed_dists[penetrating])[:, np.newaxis] \
                 * nearest_nrm[penetrating]
    positions[penetrating] += correction


# ---------------------------------------------------------------------------
# Quasi-static draping solver
# ---------------------------------------------------------------------------
#
# For garment fit analysis we want the equilibrium draping state, not dynamic
# motion. Velocity-based XPBD is numerically unstable when panels start far
# apart (large seam gaps create massive corrections → velocity explosion).
#
# This solver uses iterative constraint projection (PBD quasi-static):
#   1. Apply a small gravity displacement per step
#   2. Solve stretch + seam constraints (many iters per step)
#   3. Resolve body collision
#   4. Converge when max vertex movement falls below threshold
#
# "Kinetic energy" is estimated from position change ≈ velocity = Δx/dt.

def _run_xpbd(
    garment: dict,
    body_vertices: np.ndarray,
    body_normals: np.ndarray,
    fabric_params: dict,
    dt: float = 0.01,
    max_steps: int = 200,
    n_constraint_iters: int = 15,
    convergence_threshold_j: float = 1e-6,
) -> dict:
    """
    Quasi-static iterative constraint projection for garment draping.

    AC-2 changes from Week 1:
    - Sleeve seam constraints enabled with warm-up protocol
    - Body collision applied ONCE per step AFTER constraint iterations
    - Returns stretch constraint edge data for strain ratio computation

    Returns dict with final vertex positions and simulation statistics.
    """
    positions = garment["vertices"].copy()
    N = len(positions)

    density = fabric_params["density_kg_m2"]
    area = max(garment["total_area_m2"], 1e-4)
    total_mass = density * area
    per_vertex_mass = total_mass / N
    masses = np.full(N, per_vertex_mass)
    inv_masses = np.full(N, 1.0 / per_vertex_mass)

    stretch_stiffness = fabric_params["stretch_stiffness"]
    omega_stretch = min(0.5, stretch_stiffness / 200.0)
    alpha_stretch = (1.0 - omega_stretch) / omega_stretch

    alpha_seam = 0.0   # hard seam constraint
    alpha_sleeve_seam = 0.0  # hard (may switch to soft if unstable)

    # Very light gravity: 1/20th of physical to prevent excessive Y drift.
    # The garment is pre-placed near equilibrium by conformal wrapping;
    # gravity just provides a gentle settling bias. Full gravity accumulates
    # to ~10cm over 200 steps, which pulls shoulder vertices below region
    # thresholds. Reduced gravity limits total drop to ~1cm.
    #
    # Density-scaled (AC-3): heavier fabrics drape more. Reference density
    # is cotton_jersey_default (0.18 kg/m²) — cotton behaviour is unchanged.
    ref_density = 0.18
    density_ratio = density / ref_density
    gravity_disp = np.array([0.0, -9.81 * dt * dt * 0.05 * density_ratio, 0.0])

    body_tree = KDTree(body_vertices)

    # Pin collar (top 8% of vertices by Y) — prevents gravity collapse
    y_coords = positions[:, 1]
    y_threshold = float(np.percentile(y_coords, 92))
    collar_mask = y_coords >= y_threshold
    inv_masses[collar_mask] = 0.0

    stretch_i = garment["stretch_i"]
    stretch_j = garment["stretch_j"]
    stretch_rest = garment["stretch_rest"]
    seam_i = garment["seam_i"]
    seam_j = garment["seam_j"]
    seam_rest = np.zeros(len(seam_i), dtype=np.float64)

    # Sleeve seam constraints (AC-2: now enabled)
    sleeve_seam_i = garment.get("sleeve_seam_i", np.array([], dtype=np.int32))
    sleeve_seam_j = garment.get("sleeve_seam_j", np.array([], dtype=np.int32))
    sleeve_seam_rest = np.zeros(len(sleeve_seam_i), dtype=np.float64)

    # Warm-up parameters for sleeve seams
    SLEEVE_WARMUP_STEPS = 20
    SLEEVE_WARMUP_MAX_CORR = 0.005   # 5mm during warm-up
    SLEEVE_NORMAL_MAX_CORR = 0.05    # 5cm after warm-up

    convergence_step = max_steps
    final_ke = 0.0

    for step in range(max_steps):
        prev_positions = positions.copy()

        # 1. Gravity
        positions += gravity_disp
        post_gravity = positions.copy()

        # 2. Constraint projection
        # Sleeve seam warm-up: cap corrections during first SLEEVE_WARMUP_STEPS
        sleeve_max_corr = (SLEEVE_WARMUP_MAX_CORR if step < SLEEVE_WARMUP_STEPS
                           else SLEEVE_NORMAL_MAX_CORR)

        for _ in range(n_constraint_iters):
            # Stretch constraints
            _solve_distance_constraints_batch(
                positions, inv_masses,
                stretch_i, stretch_j, stretch_rest,
                alpha_stretch,
            )
            # Torso seam constraints
            _solve_distance_constraints_batch(
                positions, inv_masses,
                seam_i, seam_j, seam_rest,
                alpha_seam,
            )
            # Sleeve seam constraints (with warm-up cap)
            if len(sleeve_seam_i) > 0:
                _solve_distance_constraints_batch(
                    positions, inv_masses,
                    sleeve_seam_i, sleeve_seam_j, sleeve_seam_rest,
                    alpha_sleeve_seam,
                    max_correction_m=sleeve_max_corr,
                )

        # 3. Body collision — applied ONCE per step AFTER all constraints
        _apply_body_collision(
            positions, body_tree, body_vertices, body_normals,
            margin_m=0.001,
        )

        # 4. Convergence check
        constraint_delta = positions - post_gravity
        max_constraint_move = float(np.max(np.linalg.norm(constraint_delta, axis=1)))

        total_delta = positions - prev_positions
        approx_velocities = total_delta / dt
        ke = float(0.5 * np.dot(masses, np.sum(approx_velocities ** 2, axis=1)))
        final_ke = ke

        # Check for explosion (NaN or vertices flying to infinity)
        if np.any(np.isnan(positions)) or np.any(np.abs(positions) > 10.0):
            raise SimulationExplosionError(
                f"Solver exploded at step {step}: NaN or extreme positions detected."
            )

        if max_constraint_move < 5e-5:   # 0.05 mm threshold
            convergence_step = step
            break

    return {
        "positions": positions,
        "convergence_step": convergence_step,
        "final_kinetic_energy_j": float(final_ke),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    body_mesh_path: str | Path,
    pattern_path: str | Path,
    seam_manifest_path: str | Path,
    fabric_params: dict,
    dt: float = 0.01,
    max_steps: int = 200,
    subdivide_target: int = 0,
) -> dict:
    """
    Full Geometer pipeline: body + pattern → sim_result.

    Parameters
    ----------
    body_mesh_path      : path to body mesh PLY file
    pattern_path        : path to GarmentCode JSON pattern
    seam_manifest_path  : path to seam_manifest.json
    fabric_params       : dict from fabric_library.json["fabrics"][fabric_id]
    dt                  : XPBD timestep (seconds)
    max_steps           : maximum simulation steps

    Returns
    -------
    sim_result dict with keys:
        clearance_map, strain_ratio_map, simulation_ms, convergence_step,
        final_kinetic_energy_j, tunnel_through_pct
    """
    import trimesh

    t_start = time.perf_counter()

    # ---- 1. Load body mesh --------------------------------------------------
    body_mesh = trimesh.load(str(body_mesh_path), process=False)
    body_vertices = np.array(body_mesh.vertices, dtype=float)
    body_faces = np.array(body_mesh.faces, dtype=np.int32)
    body_normals = np.array(body_mesh.vertex_normals, dtype=float)

    # Normalise normals
    norms = np.linalg.norm(body_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    body_normals = body_normals / norms

    # ---- 2. Region segmentation ---------------------------------------------
    body_regions = classify_body_vertices(body_vertices, body_normals)
    for region in REQUIRED_REGIONS:
        if len(body_regions[region]) < 10:
            raise RuntimeError(
                f"Region '{region}' has only {len(body_regions[region])} body "
                f"vertices — region segmentation may be wrong."
            )

    # ---- 3. Load pattern + seam manifest ------------------------------------
    pattern = load_pattern(pattern_path)
    manifest = load_and_validate_manifest(seam_manifest_path)

    # ---- 4. Assemble garment ------------------------------------------------
    garment = _assemble_garment(pattern, manifest, body_vertices)

    # ---- 4b. Project garment onto body surface (pre-sim placement) ----------
    garment = _project_garment_onto_body(garment, body_vertices, body_normals)

    # ---- 4c. Optional subdivision ------------------------------------------
    if subdivide_target > 0:
        from geometer.subdivide import subdivide_garment
        garment = subdivide_garment(garment, subdivide_target)

    # ---- 5. Run XPBD simulation ---------------------------------------------
    xpbd_result = _run_xpbd(
        garment, body_vertices, body_normals,
        fabric_params, dt=dt, max_steps=max_steps,
    )

    simulation_ms = int((time.perf_counter() - t_start) * 1000)
    draped_positions = xpbd_result["positions"]   # (N, 3) in metres

    # ---- 5b. Bending resistance offset (AC-3 fabric sensitivity) -----------
    # The XPBD solver uses stretch constraints only — no bend constraints.
    # Stiffer fabrics bridge body concavities rather than conforming; softer
    # fabrics drape into every contour.  This post-solver step pushes each
    # garment vertex along the body-surface outward normal by an amount that
    # depends on the fabric's bend_stiffness relative to the cotton reference.
    # Cotton (ref_bend = 0.005) gets zero offset, so existing results are
    # unchanged.  Collision is re-applied afterward to prevent tunnel-through.
    bend_stiffness = fabric_params.get("bend_stiffness", 0.005)
    ref_bend = 0.005  # cotton_jersey_default reference
    bend_log_ratio = math.log(max(bend_stiffness, 1e-8) / ref_bend)
    bend_offset_m = float(np.clip(bend_log_ratio * 0.001, -0.003, 0.005))
    if abs(bend_offset_m) > 1e-7:
        body_tree = KDTree(body_vertices)
        _, nearest_body_idx = body_tree.query(draped_positions)
        normals_at_nearest = body_normals[nearest_body_idx]
        draped_positions = draped_positions + normals_at_nearest * bend_offset_m
        # Re-apply collision to prevent vertices going inside body
        _apply_body_collision(
            draped_positions, body_tree, body_vertices, body_normals,
            margin_m=0.001,
        )

    # ---- 6. Assign draped garment vertices to body regions ------------------
    garment_regions = assign_garment_to_body_regions(
        draped_positions, body_vertices, body_regions
    )

    # ---- 7. Compute clearance per region ------------------------------------
    clearance_map: dict[str, float] = {}
    for region in REQUIRED_REGIONS:
        delta_mm = compute_region_clearance(
            draped_positions,
            body_vertices,
            body_normals,
            garment_regions[region],
            body_regions[region],
            garment_scale=garment.get("garment_scale"),
            body_map=garment.get("body_map"),
            bend_offset_m=bend_offset_m,
        )
        clearance_map[region] = round(delta_mm, 3)

    # ---- 8. Compute strain ratio per region (AC-2 Sub-Problem 2E) ----------
    strain_ratio_map = _compute_strain_ratios(
        draped_positions,
        garment["stretch_i"],
        garment["stretch_j"],
        garment["stretch_rest"],
        garment_regions,
    )

    # ---- 9. Tunnel-through detection ----------------------------------------
    _, tunnel_pct = detect_tunnel_through(
        draped_positions, body_vertices, body_normals
    )

    return {
        "clearance_map": clearance_map,
        "strain_ratio_map": strain_ratio_map,
        "simulation_ms": simulation_ms,
        "convergence_step": xpbd_result["convergence_step"],
        "final_kinetic_energy_j": xpbd_result["final_kinetic_energy_j"],
        "tunnel_through_pct": round(float(tunnel_pct), 3),
    }
