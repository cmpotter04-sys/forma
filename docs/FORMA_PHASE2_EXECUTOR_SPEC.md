# FORMA — Phase 2 Executor Spec: Stage 1 ("The Rosetta Stone")
**Version:** 1.0
**Stage:** 1 of 4
**Goal:** Prove NVIDIA Warp (Apache 2.0) reproduces Forma Phase 1 physics at exact parity.

---

## Read This First

You are implementing Stage 1 of Forma's GPU transition. Phase 1 built a working
XPBD cloth simulation on CPU (numpy/scipy). Your job is to integrate that pipeline
with NVIDIA Warp's GPU simulation infrastructure and prove the results match.

**You are NOT rewriting the physics engine from scratch.** Warp ships with
`wp.sim.XPBDIntegrator` — a built-in XPBD solver. Your work is bridging Forma's
data model into Warp's simulation framework and validating parity.

**You are NOT modifying Phase 1 code.** The CPU solver (`xpbd_simulate.py`) is
the regression baseline. Do not edit it. If you need shared utilities (mesh loading,
region classification, etc.), extract them into shared modules.

---

## Acceptance Criteria

### AC-1: Warp Integration (Week 1)
**Input:** M-size t-shirt pattern + M-size body mesh + cotton_jersey_default fabric
**Process:** Run simulation through Warp's wp.sim.XPBDIntegrator on GPU
**Output:** fit_verdict.json (v1.2 schema)
**Gate:** Per-region clearance delta ≤ 0.5mm vs CPU baseline verdict

Tasks:
1. Load MakeHuman PLY body mesh into `wp.sim.ModelBuilder` as a kinematic body
2. Load triangulated garment panels as cloth particles with mass from fabric density
3. Map Forma stretch constraints to Warp's spring/distance constraint system
4. Map seam constraints to Warp's attachment/spring constraints
5. Implement collar pinning (top 8% of vertices by Y) as kinematic particle constraints
6. Configure quasi-static solver: 1/20th gravity, damping from fabric_library.json
7. Implement sleeve warm-up protocol: 20 steps with 5mm correction cap
8. Run simulation to convergence (max vertex movement < 0.05mm)
9. Extract deformed garment mesh, compute clearance via existing clearance.py
10. Generate verdict via existing generate_verdict.py

### AC-2: 100K Smoke Test (Week 2)
**Input:** Same as AC-1 but with Loop-subdivided meshes (~100K vertices)
**Process:** Run identical Warp pipeline on higher-density meshes
**Output:** Performance profile + accuracy comparison

Tasks:
1. Implement Loop subdivision for body and garment meshes (or use trimesh.remesh)
2. Run Warp simulation at 100K vertices on Colab GPU
3. Record: VRAM usage, wall-clock time, per-kernel timing breakdown
4. Compare verdict against 22K baseline (accuracy should improve or hold, not degrade)
5. Document bottleneck: is it solver iterations, collision, or memory?

### AC-3: Regression Test Suite
**Output:** `tests/test_warp_parity.py` passing on Colab

The parity test must:
1. Run CPU backend on reference case (M-on-M, cotton_jersey_default)
2. Run Warp backend on identical inputs
3. Assert per-region clearance delta ≤ 0.5mm
4. Assert per-region strain ratio delta ≤ 0.02
5. Assert verdict agreement (fit: true/false matches)
6. Run in < 60 seconds on Colab T4

---

## Technical Reference

### Warp Simulation Setup Pattern
```python
import warp as wp
import warp.sim

wp.init()

builder = wp.sim.ModelBuilder()

# Add kinematic body mesh (does not move)
builder.add_body(origin=wp.transform_identity())
body_mesh = wp.sim.Mesh(
    vertices=body_verts,    # wp.array of vec3
    indices=body_faces,     # wp.array of int32 (flattened triangles)
)

# Add cloth particles (garment vertices)
for i, pos in enumerate(garment_verts):
    builder.add_particle(pos=pos, vel=(0,0,0), mass=particle_mass)

# Add stretch constraints (distance springs)
for edge in garment_edges:
    builder.add_spring(edge[0], edge[1], stiffness=stretch_k, damping=damp)

# Build model
model = builder.finalize()
state_0 = model.state()
state_1 = model.state()

# Create integrator
integrator = wp.sim.XPBDIntegrator(iterations=solver_iterations)

# Simulation loop
for step in range(max_steps):
    state_0.clear_forces()
    wp.sim.collide(model, state_0)
    integrator.simulate(model, state_0, state_1, dt=sim_dt)
    state_0, state_1 = state_1, state_0
    # Check convergence...
```

**IMPORTANT:** This is illustrative pseudocode. Consult the official Warp docs
(nvidia.github.io/warp/) and `warp/examples/sim/example_cloth.py` for exact API.
The API may have changed — verify parameter names and signatures.

### Fabric Parameter Mapping

Forma's fabric_library.json uses these parameters:
```json
{
  "fabric_id": "cotton_jersey_default",
  "density_kg_m2": 0.18,
  "stretch_stiffness": 40,
  "bend_stiffness": 0.005,
  "shear_stiffness": 3.0,
  "damping": 0.995
}
```

These must be mapped to Warp's constraint system:
- `density_kg_m2` → particle mass = density × triangle_area / 3 (per vertex)
- `stretch_stiffness` → spring stiffness (may need scaling — validate via parity test)
- `bend_stiffness` → bending constraint compliance (1/stiffness)
- `damping` → velocity damping per step

The mapping is NOT guaranteed to be 1:1. Use the parity test to calibrate.

### Float Precision Strategy
- **Start with float32** (Warp default on GPU, 2x faster than float64)
- **Measure delta** at AC-1 gate. If ≤ 0.5mm, stay on float32.
- **Fallback to float64** if precision is insufficient. Warp supports `wp.float64`.
- Do NOT mix precisions in the same kernel.

### Key Phase 1 Parameters to Preserve
| Parameter | Value | Where It Lives |
|-----------|-------|----------------|
| Gravity scale | 1/20th physical (0.49 m/s²) | xpbd_simulate.py |
| Collar pin | Top 8% of verts by Y coordinate | xpbd_simulate.py |
| Convergence | Max vertex movement < 0.05mm | convergence.py |
| Sleeve warm-up | 20 steps, 5mm correction cap | xpbd_simulate.py |
| Collision | KDTree nearest-neighbor push-out | xpbd_simulate.py |
| Bending offset | Post-sim, fabric-dependent | xpbd_simulate.py |

---

## File Deliverables

By end of Stage 1, these files must exist:

```
src/geometer/warp/
├── __init__.py
├── warp_simulate.py          # Main Warp simulation function
│                              # Signature: simulate_warp(body_mesh, garment, fabric_params) -> deformed_verts
├── mesh_bridge.py            # Convert trimesh/numpy meshes ↔ Warp arrays
└── README.md                 # Setup instructions for Colab

src/pipeline.py               # MODIFIED: add backend="cpu"|"warp" parameter

tests/
├── test_warp_parity.py       # AC-3: side-by-side regression test
└── test_warp_smoke_100k.py   # AC-2: high vertex count smoke test

output/
└── stage1_profile.json       # Performance profile (22K vs 100K timing)
```

---

## What NOT To Build in Stage 1

- Custom CUDA kernels (use Warp's built-in integrator)
- VBD solver (that's Stage 3)
- Spatial hashing / HashGrid (that's Stage 3)
- Body-part collision resolution kernels (that's Stage 2)
- Attachment constraint kernels (that's Stage 2)
- Any rendering, DLSS, or visualization work
- Supabase integration

---

## Environment Setup (Colab)

```python
# First cell of every Colab notebook
!pip install warp-lang trimesh scipy numpy pytest

import warp as wp
wp.init()
print(f"Warp {wp.__version__}, Device: {wp.get_preferred_device()}")
```

Verify GPU is available. If `wp.get_preferred_device()` returns `cpu`, the Colab
runtime is not GPU-enabled — go to Runtime > Change runtime type > GPU.

---

*Forma Phase 2 Executor Spec v1.0 — March 2026*
