# Forma kernel: warp_simulate.py
# Derived from:
#   - Macklin et al., "XPBD: Position-Based Simulation of Compliant
#     Constrained Dynamics" (MIG 2016)
#   - Forma Phase 1 XPBD solver (src/geometer/xpbd_simulate.py)
#   - NVIDIA Warp official documentation and Apache 2.0 examples
#     (nvidia.github.io/warp/)
#   - Baraff & Witkin, "Large Steps in Cloth Simulation" (SIGGRAPH 1998)
# No code from NvidiaWarp-GarmentCode was referenced.

"""
src/geometer/warp/warp_simulate.py

Warp GPU simulation backend for Forma.

Mirrors the CPU pipeline in xpbd_simulate.run_simulation() but replaces
the numpy XPBD solver with NVIDIA Warp's wp.sim.XPBDIntegrator running
on GPU.

Public API:
    run_simulation_warp(body_mesh_path, pattern_path, seam_manifest_path,
                        fabric_params, dt, max_steps) → sim_result dict

The return dict has identical keys to the CPU backend so the verdict
generator, clearance, and strain tools work unchanged.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

# Insert src/ so we can import sibling packages
_SRC = Path(__file__).parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pattern_maker.load_patterns import load_pattern
from tailor.seam_converter import load_and_validate_manifest, SeamValidationError
from geometer.region_map import classify_body_vertices, assign_garment_to_body_regions
from geometer.clearance import (
    compute_region_clearance,
    detect_tunnel_through,
)
from geometer.convergence import SimulationExplosionError
from geometer.garment_assembly import (
    assemble_garment,
    project_garment_onto_body,
    compute_strain_ratios,
    REQUIRED_REGIONS,
)

# Lazy Warp import — only fails when actually called, not at import time.
# This lets pipeline.py import both backends without requiring warp locally.
_wp = None
_wp_sim = None


def _ensure_warp():
    """Import and initialize Warp on first use."""
    global _wp, _wp_sim
    if _wp is None:
        try:
            import warp as wp
            import warp.sim
            wp.init()
            _wp = wp
            _wp_sim = warp.sim
        except ImportError:
            raise ImportError(
                "warp-lang is required for the Warp backend. "
                "Install it with: pip install warp-lang"
            )
    return _wp, _wp_sim


def _build_warp_model(
    garment: dict,
    body_verts_np: np.ndarray,
    body_faces_np: np.ndarray,
    fabric_params: dict,
    n_constraint_iters: int = 15,
):
    """
    Build a wp.sim.Model from Forma's assembled garment data.

    Maps Forma's constraint system to Warp's ModelBuilder:
      - Garment vertices → particles (with mass from density × area)
      - Stretch constraints → springs
      - Seam constraints → springs with rest_length=0
      - Collar pinning → kinematic particles (mass=0)
      - Body mesh → collision shape

    Returns (model, garment_particle_offset) where garment_particle_offset
    is the index offset for garment particles in the model.
    """
    wp, sim = _ensure_warp()
    from geometer.warp.mesh_bridge import compute_particle_masses, build_warp_mesh

    builder = sim.ModelBuilder()

    positions = garment["vertices"]       # (N, 3) metres
    faces = garment["faces"]              # (M, 3) int
    N = len(positions)

    # ---- Compute per-vertex mass from triangle areas ----
    density = fabric_params["density_kg_m2"]
    masses = compute_particle_masses(positions, faces, density)

    # Clamp minimum particle mass to prevent explosion from tiny triangles.
    # Particles at panel edges/corners get vanishingly small masses from
    # small triangle areas. Even modest spring forces (ke ~0.01 N/m) cause
    # catastrophic acceleration when mass < 1e-6 kg.
    # Floor ensures stiffness/mass ratio stays below ~1000 (stable for XPBD).
    avg_mass = float(masses[masses > 0].mean())
    mass_floor = max(avg_mass * 0.1, 1e-5)
    masses = np.maximum(masses, mass_floor)

    # ---- Pin collar (top 8% by Y) — set mass=0 for kinematic ----
    y_coords = positions[:, 1]
    y_threshold = float(np.percentile(y_coords, 92))
    collar_mask = y_coords >= y_threshold

    # ---- Add garment particles ----
    for i in range(N):
        m = 0.0 if collar_mask[i] else float(masses[i])
        builder.add_particle(
            pos=tuple(positions[i].astype(float)),
            vel=(0.0, 0.0, 0.0),
            mass=m,
        )

    # ---- Add stretch constraints as springs ----
    stretch_i = garment["stretch_i"]
    stretch_j = garment["stretch_j"]
    stretch_rest = garment["stretch_rest"]

    stretch_stiffness = fabric_params["stretch_stiffness"]
    # Warp XPBD spring stiffness is force-based (N/m). Scale proportional
    # to mass for stability. Higher ratio = stiffer fabric response.
    warp_stretch_ke = avg_mass * 10000.0  # force-based spring stiffness scaled to particle mass
    warp_stretch_kd = avg_mass * 10.0  # damping must be mass-proportional to avoid explosion

    for k in range(len(stretch_i)):
        builder.add_spring(
            int(stretch_i[k]),
            int(stretch_j[k]),
            ke=warp_stretch_ke,
            kd=warp_stretch_kd,
            control=0.0,
        )

    # ---- Add seam constraints (slightly stiffer than stretch) ----
    seam_ke = warp_stretch_ke * 2.0
    seam_kd = warp_stretch_kd  # same damping for seams

    seam_i = garment["seam_i"]
    seam_j = garment["seam_j"]
    for k in range(len(seam_i)):
        builder.add_spring(
            int(seam_i[k]),
            int(seam_j[k]),
            ke=seam_ke,
            kd=seam_kd,
            control=0.0,
        )

    # ---- Add sleeve seam constraints ----
    sleeve_seam_i = garment.get("sleeve_seam_i", np.array([], dtype=np.int32))
    sleeve_seam_j = garment.get("sleeve_seam_j", np.array([], dtype=np.int32))
    for k in range(len(sleeve_seam_i)):
        builder.add_spring(
            int(sleeve_seam_i[k]),
            int(sleeve_seam_j[k]),
            ke=seam_ke,
            kd=seam_kd,
            control=0.0,
        )

    # ---- Add body as collision shape ----
    body_mesh = build_warp_mesh(body_verts_np, body_faces_np)
    builder.add_body(origin=wp.transform_identity())
    builder.add_shape_mesh(
        body=0,
        mesh=body_mesh,
        density=0.0,  # kinematic (infinite mass)
    )

    # ---- Finalize ----
    model = builder.finalize()

    # Collision standoff: Warp defaults particle_radius to 0.1m (100mm) which
    # pushes the garment far from the body. Set to 1mm to match physical standoff.
    model.particle_radius = wp.array(
        np.full(N, 0.001, dtype=np.float32),
        device=model.particle_radius.device,
    )
    model.soft_contact_margin = 0.002  # 2mm collision detection margin

    # Set gravity AFTER finalize: 1/20th physical (quasi-static settling)
    # 9.81 / 20 = 0.4905 m/s²
    model.gravity = np.array([0.0, -0.49, 0.0])
    model.ground = False  # no ground plane — garment is on a body

    return model


def _run_warp_simulation(
    model,
    garment: dict,
    fabric_params: dict,
    dt: float = 0.01,
    max_steps: int = 200,
    n_constraint_iters: int = 15,
) -> dict:
    """
    Run the Warp XPBD simulation loop.

    Returns dict with final positions, convergence step, and kinetic energy.
    """
    wp, sim = _ensure_warp()

    N = len(garment["vertices"])

    integrator = sim.XPBDIntegrator(iterations=n_constraint_iters)

    state_0 = model.state()
    state_1 = model.state()

    # Sleeve warm-up parameters (match CPU backend)
    SLEEVE_WARMUP_STEPS = 20

    convergence_step = max_steps
    final_ke = 0.0

    prev_positions = None

    for step in range(max_steps):
        # Extract current positions for convergence check
        curr_pos = state_0.particle_q.numpy()[:N].copy()

        if prev_positions is not None:
            # Convergence: max vertex movement < 0.05mm
            delta = curr_pos - prev_positions
            max_move = float(np.max(np.linalg.norm(delta, axis=1)))

            # Estimate kinetic energy
            velocities = delta / dt
            # Use uniform mass estimate for KE (exact masses aren't easily
            # accessible from the model after finalize)
            density = fabric_params["density_kg_m2"]
            area = max(garment["total_area_m2"], 1e-4)
            per_vertex_mass = density * area / N
            ke = float(0.5 * per_vertex_mass * np.sum(velocities ** 2))
            final_ke = ke

            # Check for explosion
            if np.any(np.isnan(curr_pos)) or np.any(np.abs(curr_pos) > 10.0):
                raise SimulationExplosionError(
                    f"Warp solver exploded at step {step}: "
                    f"NaN or extreme positions detected."
                )

            if max_move < 5e-5:  # 0.05mm threshold
                convergence_step = step
                break

        prev_positions = curr_pos.copy()

        # Simulate one step
        state_0.clear_forces()
        sim.collide(model, state_0)
        integrator.simulate(model, state_0, state_1, dt=dt)

        # Swap states
        state_0, state_1 = state_1, state_0

    # Extract final positions
    final_pos = state_0.particle_q.numpy()[:N].astype(np.float64)

    return {
        "positions": final_pos,
        "convergence_step": convergence_step,
        "final_kinetic_energy_j": float(final_ke),
    }


def run_simulation_warp(
    body_mesh_path: str | Path,
    pattern_path: str | Path,
    seam_manifest_path: str | Path,
    fabric_params: dict,
    dt: float = 0.001,
    max_steps: int = 200,
    subdivide_target: int = 0,
) -> dict:
    """
    Full Geometer pipeline via Warp GPU: body + pattern → sim_result.

    Identical signature and return schema to xpbd_simulate.run_simulation().

    Parameters
    ----------
    body_mesh_path      : path to body mesh PLY file
    pattern_path        : path to GarmentCode JSON pattern
    seam_manifest_path  : path to seam_manifest.json
    fabric_params       : dict from fabric_library.json["fabrics"][fabric_id]
    dt                  : simulation timestep (seconds)
    max_steps           : maximum simulation steps

    Returns
    -------
    sim_result dict with keys:
        clearance_map, strain_ratio_map, simulation_ms, convergence_step,
        final_kinetic_energy_j, tunnel_through_pct
    """
    import trimesh

    _ensure_warp()
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

    # ---- 4. Assemble garment (shared with CPU backend) ----------------------
    garment = assemble_garment(pattern, manifest, body_vertices)

    # ---- 4b. Project garment onto body surface (pre-sim placement) ----------
    garment = project_garment_onto_body(garment, body_vertices, body_normals)

    # ---- 4c. Optional subdivision ------------------------------------------
    if subdivide_target > 0:
        from geometer.subdivide import subdivide_garment
        garment = subdivide_garment(garment, subdivide_target)

    # ---- 5. Build Warp model and run simulation -----------------------------
    model = _build_warp_model(
        garment, body_vertices, body_faces, fabric_params,
    )

    warp_result = _run_warp_simulation(
        model, garment, fabric_params, dt=dt, max_steps=max_steps,
    )

    simulation_ms = int((time.perf_counter() - t_start) * 1000)
    draped_positions = warp_result["positions"]   # (N, 3) in metres

    # ---- 5b. Bending resistance offset (same as CPU backend) ----------------
    bend_stiffness = fabric_params.get("bend_stiffness", 0.005)
    ref_bend = 0.005  # cotton_jersey_default reference
    bend_log_ratio = math.log(max(bend_stiffness, 1e-8) / ref_bend)
    bend_offset_m = float(np.clip(bend_log_ratio * 0.001, -0.003, 0.005))
    if abs(bend_offset_m) > 1e-7:
        body_tree = KDTree(body_vertices)
        _, nearest_body_idx = body_tree.query(draped_positions)
        normals_at_nearest = body_normals[nearest_body_idx]
        draped_positions = draped_positions + normals_at_nearest * bend_offset_m
        # Re-apply collision via KDTree (same as CPU path)
        nearest_pts = body_vertices[nearest_body_idx]
        nearest_nrm = body_normals[nearest_body_idx]
        displacements = draped_positions - nearest_pts
        signed_dists = np.einsum("ij,ij->i", displacements, nearest_nrm)
        penetrating = signed_dists < 0.001
        if np.any(penetrating):
            correction = (0.001 - signed_dists[penetrating])[:, np.newaxis] \
                         * nearest_nrm[penetrating]
            draped_positions[penetrating] += correction

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

    # ---- 8. Compute strain ratio per region ---------------------------------
    strain_ratio_map = compute_strain_ratios(
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
        "convergence_step": warp_result["convergence_step"],
        "final_kinetic_energy_j": warp_result["final_kinetic_energy_j"],
        "tunnel_through_pct": round(float(tunnel_pct), 3),
    }
