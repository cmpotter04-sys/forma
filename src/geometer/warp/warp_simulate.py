# Forma kernel: warp_simulate.py
# Derived from:
#   - Macklin et al., "XPBD: Position-Based Simulation of Compliant
#     Constrained Dynamics" (MIG 2016)
#   - Forma Phase 1 XPBD solver (src/geometer/xpbd_simulate.py)
#   - NVIDIA Warp official documentation and Apache 2.0 examples
#     (nvidia.github.io/warp/)
#   - Baraff & Witkin, "Large Steps in Cloth Simulation" (SIGGRAPH 1998)
#   - Grinspun et al., "Discrete Shells" (SCA 2003) — dihedral bending
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
from geometer.warp.contact_pressure import compute_contact_pressure

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
                "Install it with: pip install warp-lang\n"
                "NOTE: warp.sim requires a GPU runtime (e.g. Google Colab with GPU). "
                "The CPU-only warp-lang PyPI build does not include warp.sim."
            )
    return _wp, _wp_sim


def _build_warp_model(
    garment: dict,
    body_verts_np: np.ndarray,
    body_faces_np: np.ndarray,
    fabric_params: dict,
    n_constraint_iters: int = 32,
):
    """
    Build a wp.sim.Model from Forma's assembled garment data.

    Maps Forma's constraint system to Warp's ModelBuilder:
      - Garment vertices → particles (with mass from density × area)
      - Stretch constraints → springs
      - Seam constraints → springs with rest_length=0
      - Collar pinning → kinematic particles (mass=0)
      - Body mesh → collision shape

    All configuration that Warp reads at finalize-time (gravity, ground,
    particle radius, contact margins) is set on the builder BEFORE finalize().
    Post-finalize attributes on the model are read-only in Warp 1.x.

    Returns the finalized wp.sim.Model.
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

    # ---- Gravity: 1/20th physical (quasi-static settling, matches CPU path) ----
    # Full 9.81 m/s² causes shoulder vertices to drift below region thresholds
    # over 200 steps. Reduced gravity limits total drop to ~1cm. Density-scaled
    # so heavier fabrics drape more (cotton reference density = 0.18 kg/m²).
    # builder.gravity and builder.ground must be set BEFORE finalize().
    ref_density = 0.18
    density_ratio = density / ref_density
    gravity_y = -9.81 * 0.05 * density_ratio  # ~-0.49 m/s² for cotton
    builder.gravity = wp.vec3(0.0, float(gravity_y), 0.0)

    # ---- No ground plane — garment is on a body, not falling to floor ----
    # builder.ground is a ModelBuilder attribute that must be set pre-finalize.
    builder.ground = False

    # ---- Particle radius: 1mm standoff (Warp default 0.1m = 100mm is too large) ----
    # default_particle_radius sets the radius for all subsequently added particles.
    # Must be set before add_particle() calls.
    builder.default_particle_radius = 0.001  # 1mm

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

    # ---- Spring stiffness mapping ----
    # Warp XPBD spring stiffness is force-based (N/m). Scale proportional to
    # avg particle mass for numerical stability. The ratio ke/m must stay < ~1e4
    # to avoid stiff integrator instability at dt=0.001s.
    #   warp_stretch_ke = avg_mass × 10000  → ratio = 1e4 (borderline stable)
    #   warp_stretch_kd = avg_mass × 10     → ratio = 10  (well-damped)
    # Seam constraints are 2× stiffer to pull panels together tightly.
    warp_stretch_ke = avg_mass * 10000.0
    warp_stretch_kd = avg_mass * 10.0
    seam_ke = warp_stretch_ke * 2.0
    seam_kd = warp_stretch_kd

    # ---- Add stretch constraints as springs ----
    # control=0.0 means "use natural rest length computed from initial positions"
    # (the distance between vertex i and vertex j at t=0). This matches
    # the Phase 1 CPU solver which stores stretch_rest as initial edge lengths.
    stretch_i = garment["stretch_i"]
    stretch_j = garment["stretch_j"]

    for k in range(len(stretch_i)):
        builder.add_spring(
            int(stretch_i[k]),
            int(stretch_j[k]),
            ke=warp_stretch_ke,
            kd=warp_stretch_kd,
            control=0.0,
        )

    # ---- Add seam constraints (target rest_length = 0, so panels pull together) ----
    # Seam pairs are matched vertices from adjacent panels that should coincide.
    # We model them as springs with control=0.0 (use initial separation as rest).
    # The initial separation is ~0 because project_garment_onto_body already
    # places seam vertices at the same body-surface location.
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

    # ---- Add dihedral bending constraints (Grinspun 2003 Discrete Shells) ----
    # References:
    #   - Grinspun et al., "Discrete Shells" (SCA 2003)
    #   - Macklin et al., "XPBD" (MIG 2016) — XPBD dihedral formulation
    #   - NVIDIA Warp ModelBuilder.add_edge() Apache 2.0 docs
    #
    # For each interior edge shared by two triangles, add an edge bending constraint
    # that penalises changes in dihedral angle.  This replaces the post-hoc
    # bend_offset_m heuristic with proper physics.
    #
    # Stiffness scaling: fabric bend_stiffness (N·m) → Warp edge ke (N/m):
    #   ke_edge = bend_stiffness / avg_edge_length²
    # This keeps force magnitudes consistent with the stretch ke above.
    bend_stiffness_nm = float(fabric_params.get("bend_stiffness", 0.005))

    # Build edge → adjacent opposite-vertex list
    edge_to_opp: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        for ea, eb, opp in ((i0, i1, i2), (i1, i2, i0), (i2, i0, i1)):
            key = (min(ea, eb), max(ea, eb))
            if key not in edge_to_opp:
                edge_to_opp[key] = []
            edge_to_opp[key].append(opp)

    if garment.get("stretch_rest") is not None and len(garment["stretch_rest"]) > 0:
        avg_edge_len = float(np.mean(garment["stretch_rest"]))
        avg_edge_len = max(avg_edge_len, 1e-4)
        # ke in N/m; cap to 10% of stretch ke to keep bending softer than stretch
        bend_ke = min(bend_stiffness_nm / (avg_edge_len ** 2), warp_stretch_ke * 0.1)
        for (ea, eb), opps in edge_to_opp.items():
            if len(opps) == 2:
                builder.add_edge(ea, eb, opps[0], opps[1], ke=float(bend_ke), kd=0.0)

    # ---- Add body as kinematic collision shape ----
    # add_body returns the body index (0 for the first body).
    # add_shape_mesh attaches the collision geometry to that body.
    # density=0.0 makes the body kinematic (infinite effective mass — it doesn't move).
    body_mesh = build_warp_mesh(body_verts_np, body_faces_np)
    body_idx = builder.add_body(origin=wp.transform_identity())
    builder.add_shape_mesh(
        body=body_idx,
        mesh=body_mesh,
        density=0.0,
    )

    # ---- Finalize model ----
    model = builder.finalize()

    # ---- Post-finalize: soft contact parameters ----
    # model.soft_contact_margin: distance (m) within which soft contact forces activate.
    # model.soft_contact_ke: contact spring stiffness (N/m).
    # These are model-level float attributes readable/writable after finalize in Warp 1.x.
    model.soft_contact_margin = 0.002   # 2mm activation distance
    model.soft_contact_ke = 1e4         # contact spring stiffness

    return model


def _run_warp_simulation(
    model,
    garment: dict,
    fabric_params: dict,
    dt: float = 0.01,
    max_steps: int = 200,
    n_constraint_iters: int = 32,
) -> dict:
    """
    Run the Warp XPBD simulation loop.

    Simulation strategy mirrors the CPU quasi-static draping solver:
      1. XPBDIntegrator applies gravity + solves springs (n_constraint_iters per step)
      2. wp.sim.collide() resolves soft body-collision contacts
      3. Convergence detected when max vertex movement < 0.05mm between steps
      4. Velocity damping applied each step via the fabric damping coefficient

    The XPBDIntegrator.simulate() call advances state_0 → state_1 in one step.
    States are then swapped so state_0 always holds the "current" positions.

    Returns dict with final positions (N, 3), convergence step, and kinetic energy.
    """
    wp, sim = _ensure_warp()

    N = len(garment["vertices"])

    # XPBDIntegrator: iterations = constraint solver passes per timestep.
    # 32 passes (power-of-2, fits GPU warp scheduling) — 2× the prior 15.
    # On GPU this costs ~0 extra wall-clock; it eliminates solver drift at
    # the cost of slightly more VRAM bandwidth per step.
    integrator = sim.XPBDIntegrator(iterations=n_constraint_iters)

    state_0 = model.state()
    state_1 = model.state()

    # Fabric damping coefficient (0.995 for cotton = 0.5% velocity loss per step).
    # Applied by scaling particle velocities after each simulate() call.
    fabric_damping = float(fabric_params.get("damping", 0.995))

    # Pre-compute uniform mass estimate for kinetic energy (used in convergence check).
    # Per-vertex masses from model aren't easily recovered post-finalize;
    # this approximation is only used for the KE metric (not for physics).
    area = max(garment["total_area_m2"], 1e-4)
    per_vertex_mass = fabric_params["density_kg_m2"] * area / N

    convergence_step = max_steps
    final_ke = 0.0
    prev_positions = None

    for step in range(max_steps):
        # ---- Extract current particle positions for convergence check ----
        # state_0.particle_q is a wp.array of vec3 containing all particles.
        # The first N entries are garment particles; any remaining entries are
        # body/rigid-body DOFs which we ignore.
        curr_pos = state_0.particle_q.numpy()[:N].copy()

        if prev_positions is not None:
            delta = curr_pos - prev_positions
            max_move = float(np.max(np.linalg.norm(delta, axis=1)))

            # Estimate kinetic energy from positional change ≈ velocity = Δx/dt.
            # Uses uniform mass approximation (sufficient for convergence metric).
            approx_velocities = delta / dt
            ke = float(0.5 * per_vertex_mass * np.sum(approx_velocities ** 2))
            final_ke = ke

            # Explosion guard: NaN or extreme positions (> 10m from origin)
            if np.any(np.isnan(curr_pos)) or np.any(np.abs(curr_pos) > 10.0):
                raise SimulationExplosionError(
                    f"Warp solver exploded at step {step}: "
                    f"NaN or extreme positions detected."
                )

            # Convergence: max vertex movement < 0.05mm between steps.
            # Matches Phase 1 CPU solver threshold (5e-5 metres = 0.05mm).
            if max_move < 5e-5:
                convergence_step = step
                break

        prev_positions = curr_pos.copy()

        # ---- Simulate one step ----
        # 1. Clear external forces accumulated from previous step
        state_0.clear_forces()

        # 2. Resolve soft contacts (garment particles vs body collision shape).
        #    wp.sim.collide() populates soft contact impulses in state_0
        #    which XPBDIntegrator applies during simulate().
        sim.collide(model, state_0)

        # 3. Advance physics: gravity + spring constraints + contact response.
        #    XPBDIntegrator runs `iterations` passes of constraint projection.
        integrator.simulate(model, state_0, state_1, dt=dt)

        # 4. Apply velocity damping: multiply particle velocities by fabric_damping.
        #    In Warp 1.x, state.particle_qd is a wp.array of vec3 holding
        #    garment particle velocities only (rigid body DOFs are in body_qd).
        #    We apply damping in numpy and write back via assign() to avoid
        #    requiring a custom GPU kernel for this scalar multiply.
        if fabric_damping < 1.0:
            vels = state_1.particle_qd.numpy()      # (N, 3) float32
            vels_damped = (vels * fabric_damping).astype(np.float32)
            state_1.particle_qd.assign(vels_damped)

        # 5. Swap states: state_1 becomes the new current state
        state_0, state_1 = state_1, state_0

    # Extract final particle positions as float64 numpy array
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
    subdivide_target    : if > 0, subdivide garment mesh to this vertex count

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

    # ---- 5b. Bending resistance offset (matches CPU backend) ----------------
    # The Warp solver uses spring constraints only — no explicit bend constraints.
    # Stiffer fabrics bridge body concavities rather than conforming; softer
    # fabrics drape into every contour. This post-solver step pushes each
    # garment vertex along the body-surface outward normal by an amount that
    # depends on the fabric's bend_stiffness relative to the cotton reference.
    # Cotton (ref_bend = 0.005) gets zero offset, preserving existing results.
    # Collision is re-applied afterward to prevent tunnel-through.
    bend_stiffness = fabric_params.get("bend_stiffness", 0.005)
    ref_bend = 0.005  # cotton_jersey_default reference
    bend_log_ratio = math.log(max(bend_stiffness, 1e-8) / ref_bend)
    bend_offset_m = float(np.clip(bend_log_ratio * 0.001, -0.003, 0.005))
    if abs(bend_offset_m) > 1e-7:
        body_tree = KDTree(body_vertices)
        _, nearest_body_idx = body_tree.query(draped_positions)
        normals_at_nearest = body_normals[nearest_body_idx]
        draped_positions = draped_positions + normals_at_nearest * bend_offset_m
        # Re-apply collision via KDTree — mirror of CPU path
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

    # ---- 10. Contact pressure per region ------------------------------------
    # Builds a partial sim_result dict to pass context (soft_contact_ke).
    # garment_faces and body_normals are passed explicitly so compute_contact_pressure
    # can use accurate per-vertex areas and signed-distance projections.
    _partial_result = {"soft_contact_ke": 1e4}
    pressure_map = compute_contact_pressure(
        sim_result=_partial_result,
        body_vertices=body_vertices,
        garment_vertices=draped_positions,
        region_labels=garment_regions,
        fabric_params=fabric_params,
        dt=dt,
        garment_faces=garment["faces"],
        body_normals=body_normals,
    )

    return {
        "clearance_map": clearance_map,
        "strain_ratio_map": strain_ratio_map,
        "pressure_map": pressure_map,
        "simulation_ms": simulation_ms,
        "convergence_step": warp_result["convergence_step"],
        "final_kinetic_energy_j": warp_result["final_kinetic_energy_j"],
        "tunnel_through_pct": round(float(tunnel_pct), 3),
    }
