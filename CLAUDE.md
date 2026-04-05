# FORMA — Claude Code Context

## What This Project Is
Forma is a physics-based garment fit verification system. Given a body mesh and a
garment sewing pattern, it runs cloth simulation and returns a fit_verdict.json
stating whether the garment fits, with per-region signed clearance measurements
in millimeters.

## Current Phase
**Phase 2, Stage 1 — GPU Transition ("The Rosetta Stone").**
Phase 1 (Python/CPU XPBD) is complete and serves as the regression baseline.
Phase 2 migrates the physics engine to GPU via NVIDIA Warp.
Read FORMA_PHASE2_EXECUTOR_SPEC.md for current acceptance criteria.

## Architecture (v2.0)

### Unchanged from Phase 1
- **Sculptor:** generates sized body meshes from MakeHuman CC0 base mesh (21,833 verts)
  with morph targets + programmatic XZ scaling. Three sizes: makehuman_male_{S,M,XL}.ply
- **Pattern Maker:** loads GarmentCode JSON patterns directly (NOT via pygarment)
- **Tailor:** seam manifest validation and assembly
- **Verdict:** assembles fit_verdict.json (v1.2 schema)

### Changed in Phase 2
- **Geometer:** now has two backends:
  - `cpu` — Phase 1 pure-Python XPBD solver (numpy/scipy). **Untouched. This is the regression baseline.**
  - `warp` — Phase 2 GPU solver using NVIDIA Warp (Apache 2.0). **This is what you are building.**
- **Pipeline:** `run_fit_check()` accepts a `backend` parameter (`"cpu"` or `"warp"`)

### Directory Structure (Phase 2 additions)
```
src/geometer/
├── xpbd_simulate.py          # Phase 1 CPU solver — DO NOT MODIFY
├── clearance.py               # Shared (works with any solver output)
├── convergence.py             # Shared
├── region_map.py              # Shared
└── warp/                      # NEW — Phase 2 GPU path
    ├── warp_simulate.py       # Warp integration layer (wp.sim.ModelBuilder)
    ├── kernels/               # Forma-owned custom kernels (clean room)
    │   ├── attachment.py      # Collar/waistband pinning constraints
    │   ├── body_collision.py  # Body-part collision resolution
    │   └── self_collision.py  # Point-triangle self-collision (future)
    └── calibration/           # Stiffness conversion tables
        └── xpbd_to_vbd.json  # Fabric parameter mapping (Stage 3)
```

## Non-Negotiable Rules
1. NEVER use a fixed confidence scalar. confidence = 1.0 for synthetic_mannequin only.
2. ALWAYS validate seam_manifest.json before passing to Geometer.
   If seam validation fails, raise SeamValidationError — never silently proceed.
3. body_source field is REQUIRED in every body_profile.json and fit_verdict.json.
4. Negative delta_mm = body larger than garment (too tight / compressed).
   Positive delta_mm = garment larger than body (ease / loose). Never invert.
5. fit = True if and only if zero regions have severity "red". No exceptions.
6. NEVER hardcode verdict values. All values come from actual geometry pipeline.
7. All fabric parameters must come from data/fabrics/fabric_library.json.
8. verdict_id must use 12 hex chars (vrd_{12 hex}).
9. **NEVER modify Phase 1 files.** xpbd_simulate.py is the regression baseline.
   If you need shared utilities, extract them — don't edit the originals.
10. **OWNERSHIP GATE:** Every external dependency must be Apache 2.0, MIT, BSD,
    or equivalent. No exceptions. Check before importing anything new.

## CLEAN ROOM PROTOCOL — MANDATORY
You are building garment-specific Warp kernels. The following rules are legally binding:

### DO NOT — under any circumstances:
- Clone, read, browse, or reference the NvidiaWarp-GarmentCode repository
  (github.com/maria-korosteleva/NvidiaWarp-GarmentCode)
- Reference any code derived from that repository
- The repository is NVSCL licensed (non-commercial only). ANY exposure to its
  source code creates derivative work risk for Forma.

### PERMITTED sources for kernel development:
- NVIDIA Warp official docs and Apache 2.0 examples (nvidia.github.io/warp/)
- GarmentCodeData ECCV 2024 paper (published academic research — descriptions only)
- Macklin & Müller published XPBD/PBD papers (ACM SIGGRAPH proceedings)
- Standard cloth simulation textbooks (Stuyck 2018, Baraff & Witkin 1998)
- Forma's own Phase 1 codebase (xpbd_simulate.py — your own prior work)

### REQUIRED audit trail:
Every Forma-owned kernel file must include a header comment listing the specific
published sources used for derivation. Example:
```python
# Forma kernel: attachment.py
# Derived from:
#   - Macklin et al., "XPBD: Position-Based Simulation of Compliant
#     Constrained Dynamics" (MIG 2016)
#   - Forma Phase 1 collar pinning logic (src/geometer/xpbd_simulate.py)
# No code from NvidiaWarp-GarmentCode was referenced.
```

## Dependency Constraints
### Allowed (commercially licensable)
- **warp-lang** (Apache 2.0) — GPU simulation framework. Use wp.sim module.
- **numpy** (BSD) — linear algebra, array operations
- **scipy** (BSD) — KDTree, Delaunay, optimization
- **trimesh** (MIT) — mesh I/O, triangulation, normals
- **pytest** (MIT) — test suite

### Blocked (license or technical)
- **NvidiaWarp-GarmentCode** — NVSCL, non-commercial. DO NOT USE.
- **pygarment** — Cannot be imported (libcairo missing, cgal broken)
- **smplx / SMPL-X** — Licensing not cleared for commercial use
- **pytorch3d** — Heavy dependency, deferred to Phase 3

## Environment
- **Python 3.11** (NOT 3.12+ — Warp AST compatibility)
- **Primary dev environment:** Google Colab with GPU runtime (T4 or A100)
- **Local dev:** CPU-only for non-simulation work (pattern loading, verdict assembly)
- **Warp version:** Pin to latest stable release on PyPI (currently 1.12.0)

## Key Spec Documents (read before implementing)
- **FORMA_PHASE2_EXECUTOR_SPEC.md** — Stage 1 acceptance criteria and tasks (READ FIRST)
- **FORMA_CODEBASE_OVERVIEW.md** — full Phase 1 architecture reference
- **FORMA_GEOMETER_SPEC.md** — geometry pipeline: region segmentation, clearance computation
- **FORMA_SEAM_MANIFEST_SCHEMA.md** — seam manifest schema and validation rules
- **FORMA_FABRIC_LIBRARY.md** — XPBD material parameters

## Output Schema
All outputs must match v1.2 schema. The Warp backend must produce identical
verdict JSON structure to the CPU backend. The ONLY difference is the simulation
engine — inputs and outputs are identical.

## Regression Testing
Every code change must pass:
```bash
# Phase 1 baseline (must still pass — you didn't break anything)
pytest tests/

# Phase 2 parity (new — Warp vs CPU comparison)
pytest tests/test_warp_parity.py
```

The parity test runs both backends on identical inputs and asserts:
- Per-region clearance delta ≤ 0.5mm
- Per-region strain ratio delta ≤ 0.02
- Verdict match (fit: true/false agrees)
