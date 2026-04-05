# Forma — Complete Codebase Overview

> **What is Forma?** A physics-based garment fit verification system. Given a body mesh and a 2D sewing pattern, it runs cloth simulation (XPBD) and returns a machine-readable **fit verdict** with per-region signed clearance in millimetres.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Directory Structure](#directory-structure)
3. [Pipeline Flow](#pipeline-flow)
4. [Source Modules](#source-modules)
   - [Pipeline Orchestrator](#pipeline-orchestrator)
   - [Geometer (Physics Engine)](#geometer-physics-engine)
   - [Pattern Maker](#pattern-maker)
   - [Sculptor (Body Generation)](#sculptor-body-generation)
   - [Tailor (Seam Conversion)](#tailor-seam-conversion)
   - [Verdict Generator](#verdict-generator)
5. [Data Assets](#data-assets)
6. [Fabric Library](#fabric-library)
7. [Seam Manifests](#seam-manifests)
8. [Scripts](#scripts)
9. [Test Suite](#test-suite)
10. [Output Artifacts](#output-artifacts)
11. [Specification Documents](#specification-documents)
12. [Technology Stack & Constraints](#technology-stack--constraints)
13. [Key Design Decisions](#key-design-decisions)

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        pipeline.py                               │
│        run_fit_check(body, pattern, manifest, fabric)            │
├────────────┬──────────────┬──────────────┬───────────────────────┤
│  Sculptor  │ Pattern Maker│    Tailor    │       Geometer        │
│ Body mesh  │ Load pattern │ Seam manifest│  XPBD simulation      │
│ generation │ from JSON    │ validation   │  + clearance compute  │
├────────────┴──────────────┴──────────────┼───────────────────────┤
│                                          │     Verdict           │
│                                          │  fit_verdict.json     │
│                                          │  assembly (v1.2)      │
└──────────────────────────────────────────┴───────────────────────┘
```

**Roles** (named in the codebase's internal ontology):

| Role        | Responsibility                                  | Location            |
|-------------|--------------------------------------------------|---------------------|
| Sculptor    | Generate sized body meshes (MakeHuman)          | `src/sculptor/`     |
| Pattern Maker | Parse GarmentCode JSON patterns               | `src/pattern_maker/`|
| Tailor      | Build & validate seam manifests                 | `src/tailor/`       |
| Geometer    | XPBD physics sim + clearance measurement        | `src/geometer/`     |
| Verdict     | Assemble fit_verdict.json (v1.2 schema)         | `src/verdict/`      |
| Archivist   | Persist to Supabase (planned — Week 3)          | `db/migrations/`    |
| Strategist  | Architecture decisions (Claude via claude.ai)   | —                   |
| Executor    | Implementation (Claude Code)                    | —                   |

---

## Directory Structure

```
Forma.Mar10Go/
├── src/                          # Core source code
│   ├── pipeline.py               # Single entry-point orchestrator
│   ├── geometer/                 # Physics engine + clearance
│   │   ├── xpbd_simulate.py      # 1,493 lines — full XPBD pipeline
│   │   ├── clearance.py          # Signed clearance computation
│   │   ├── convergence.py        # Energy-based convergence detection
│   │   └── region_map.py         # Body region segmentation (6 zones)
│   ├── pattern_maker/            # Pattern loading & scaling
│   │   ├── load_patterns.py      # GarmentCode JSON parser (Bézier)
│   │   └── scale_pattern.py      # Uniform pattern scaling
│   ├── sculptor/                 # Body mesh generation
│   │   ├── generate_mannequin.py # Parametric torso (legacy)
│   │   └── makehuman_body.py     # MakeHuman CC0 base + morph targets
│   ├── tailor/                   # Seam processing
│   │   └── seam_converter.py     # GarmentCode stitches → seam manifest
│   └── verdict/                  # Output assembly
│       └── generate_verdict.py   # fit_verdict.json v1.2 builder
├── data/                         # Input data assets
│   ├── bodies/                   # Body meshes (PLY) + profiles (JSON)
│   ├── fabrics/                  # fabric_library.json
│   ├── patterns/                 # GarmentCode JSON patterns (XS–XL)
│   └── garmentcode/              # GarmentCode library (submodule)
├── seam_manifests/               # Pre-built seam manifests (XS–XL)
├── output/                       # Simulation output
│   ├── verdicts/                 # fit_verdict JSON files
│   └── *.png                     # Visualization images
├── scripts/                      # Utility & batch scripts
├── tests/                        # pytest test suite (8 files)
├── db/migrations/                # Supabase schema (planned)
├── docs/                         # Spike results & research
├── memory/                       # MEMORY.md (project context)
└── *.md                          # Specification documents
```

---

## Pipeline Flow

The core pipeline executes in this order:

```
1. VALIDATE INPUTS
   ├── Body mesh PLY exists
   ├── Pattern JSON exists
   └── Seam manifest JSON exists

2. LOAD FABRIC PARAMS
   └── fabric_library.json → {density, stretch, bend, shear, damping}

3. RUN SIMULATION (Geometer)
   ├── Load body mesh (trimesh)
   ├── Classify body vertices into 6 regions
   ├── Load GarmentCode pattern (JSON → structured dict)
   ├── Load & validate seam manifest
   ├── Assemble garment:
   │   ├── Triangulate 2D panels (Delaunay + interior grid)
   │   ├── Place torso panels (body-conformal wrapping)
   │   ├── Identify armholes from seam manifest
   │   ├── Place sleeves at armholes (cylindrical wrapping)
   │   └── Build constraints (stretch + seam)
   ├── Run XPBD simulation:
   │   ├── Quasi-static iterative constraint projection
   │   ├── Gravity + stretch + seam constraints
   │   ├── Sleeve seam warm-up protocol (20 steps)
   │   ├── Body collision (KDTree nearest-neighbor)
   │   └── Convergence detection (max vertex movement < 0.05mm)
   ├── Post-sim bending resistance offset (fabric-dependent)
   ├── Assign garment vertices to body regions
   ├── Compute signed clearance per region
   ├── Compute strain ratios per region
   └── Detect tunnel-through (< 2% threshold)

4. GENERATE VERDICT
   ├── Classify severity: green / yellow / red
   ├── Classify ease: tight / standard / relaxed / oversized
   ├── fit = True iff zero red regions
   └── Output fit_verdict.json (v1.2 schema)
```

---

## Source Modules

### Pipeline Orchestrator

**File:** `src/pipeline.py` (156 lines)

The single entry point for the fit-check pipeline. Provides two public functions:

- **`run_fit_check(body, pattern, manifest, fabric_id)`** — Full pipeline for one garment. Validates paths, loads fabric params, runs simulation, generates verdict.
- **`run_batch_fit_check(body, patterns, manifests, fabric_id)`** — Sequential fit checks for multiple garments on the same body.

---

### Geometer (Physics Engine)

**Directory:** `src/geometer/` — The heart of the system.

#### `xpbd_simulate.py` (1,493 lines)

The largest and most complex file. Implements the full geometry + physics pipeline:

| Component | What It Does |
|-----------|-------------|
| **Panel geometry** | Builds 2D outlines from edge polylines, places panels in 3D |
| **Body-conformal wrapping** | Measures body radius profile at multiple heights/angles, wraps garment panels to follow body contour instead of a simple cylinder |
| **Cylindrical placement** | Maps panel 2D x-coordinate to angle θ on a cylinder for initial placement |
| **Sleeve placement** | Identifies armholes from seam manifest, wraps sleeves onto arm cylinders |
| **Triangulation** | Delaunay triangulation with interior grid points (~5cm spacing) |
| **Seam constraints** | Resamples paired edges to equal vertex count, auto-orients via endpoint matching |
| **XPBD solver** | Quasi-static iterative constraint projection: stretch + seam + collision |
| **Strain computation** | Per-region median strain ratio from stretch constraints |

Key physics parameters:
- **Gravity:** 1/20th physical (pre-placed garment, just settling bias)
- **Collar pinning:** Top 8% of vertices by Y are fixed (prevents gravity collapse)
- **Sleeve warm-up:** 20-step protocol with 5mm correction cap
- **Convergence:** Max constraint movement < 0.05mm

#### `clearance.py` (189 lines)

Signed clearance between draped garment and body:

- **KDTree signed distance** — Primary metric using nearest-neighbor + body normal dot product
- **Circumference-based radial clearance** — Secondary metric guaranteeing monotonic clearance with garment size
- **Tunnel-through detection** — Vertices within 2mm of body surface on the inside
- **Severity classification** — `red` (< -25mm or strain > 1.15), `yellow` (< -10mm or strain > 1.08), `green`
- **Ease classification** — `tight_fit`, `standard_fit` (0–20mm), `relaxed_fit` (20–50mm), `oversized` (> 50mm)

#### `convergence.py` (70 lines)

Energy-based convergence detection with explosion protection:
- Primary: kinetic energy < threshold (1e-6 J)
- Explosion: energy spike > 10× previous step → `SimulationExplosionError`
- Secondary: vertex movement check (< 0.4mm for ≥ 98.5% of vertices)

#### `region_map.py` (121 lines)

Body region segmentation using vertex height (Y) and normal direction (Z):

| Region | Height Range | Criteria |
|--------|-------------|----------|
| `shoulder_left` | 1.32–1.50m | x > 0.05 |
| `shoulder_right` | 1.32–1.50m | x < -0.05 |
| `chest_front` | 1.12–1.38m | normal_z > 0.25 |
| `upper_back` | 1.12–1.38m | normal_z < -0.25 |
| `chest_side` | 1.12–1.38m | remainder |
| `waist` | 0.95–1.12m | all |

Garment vertices are assigned to body regions via KDTree nearest-neighbor with 10cm max distance.

---

### Pattern Maker

**Directory:** `src/pattern_maker/`

#### `load_patterns.py` (249 lines)

Parses GarmentCode JSON directly (no pygarment dependency). Handles:

- **Straight edges** — Direct vertex-to-vertex
- **Quadratic Bézier curves** — Relative control point resolution
- **Cubic Bézier curves** — Two relative control points
- **Discretization** — Curves sampled at ~5mm target spacing
- **Arc length computation** — Per-edge polyline measurement

Returns a structured dict with panels, edges (as discretized polylines), stitches, translation/rotation vectors.

#### `scale_pattern.py` (68 lines)

Uniform scaling of GarmentCode patterns:
- Scales vertex coordinates and translations by a factor
- Leaves rotations and Bézier control points (relative) unchanged
- Updates `_forma_metadata` with scale lineage

---

### Sculptor (Body Generation)

**Directory:** `src/sculptor/`

#### `generate_mannequin.py` (312 lines) — *Legacy*

Generates a parametric torso mesh from circular cross-sections:
- 18 control points defining circumference at body landmarks
- Cubic spline interpolation between heights
- 120 rings × 80 vertices per ring
- Validates against target measurements (size M male: 180cm, 96cm chest, 80cm waist, 95cm hip)

#### `makehuman_body.py` (353 lines) — *Current*

Generates sized body meshes from the MakeHuman CC0 base mesh (21,833 vertices):

1. Loads `makehuman_base.obj`
2. Applies `caucasian-male-young` morph target
3. Scales to target height, recenters
4. Iteratively applies height-dependent XZ torso scaling to hit circumference targets (up to 8 rounds)
5. Exports PLY + body profile JSON

Pre-built sizes:

| Size | Height | Chest | Waist | Hip |
|------|--------|-------|-------|-----|
| S | 170cm | 88cm | 72cm | 87cm |
| M | 180cm | 96cm | 80cm | 95cm |
| XL | 185cm | 108cm | 96cm | 105cm |

---

### Tailor (Seam Conversion)

**Directory:** `src/tailor/`

#### `seam_converter.py` (328 lines)

Converts GarmentCode stitch definitions into the Forma seam manifest format:

1. **Build panel records** — Discretize edges, compute arc lengths
2. **Build seam pairs** — Map GarmentCode stitches → edge pairs
3. **Validate** — Arc-length tolerance (2mm standard, 10mm gather seams), orphan/duplicate edge checks
4. **Resample** — Equalize paired edge vertex counts for XPBD constraints
5. **Serialize** — Write `seam_manifest.json`

Raises `SeamValidationError` on any validation failure — never silently proceeds.

---

### Verdict Generator

**Directory:** `src/verdict/`

#### `generate_verdict.py` (326 lines)

Assembles the final `fit_verdict.json` document (v1.2 schema):

- **strain_map** — Per-region: `{region, delta_mm, severity, median_strain_ratio}`
- **ease_map** — Per-region: `{region, excess_mm, verdict}`
- **fit** — Boolean: `True` iff zero red regions
- **confidence** — `1.0` for synthetic mannequin (formula-based for photo bodies in Phase 2)
- **fabric_params_used** — Complete fabric parameter record
- **Validation** — Full schema validator checking 15+ top-level fields, verdict_id format, region completeness, severity/ease label enums, fit/red consistency

---

## Data Assets

### Body Meshes (`data/bodies/`)

| File | Description |
|------|-------------|
| `makehuman_base.obj` | CC0 base mesh (21,833 verts) |
| `makehuman_targets.npz` | Morph target library (31MB) |
| `makehuman_male_S.ply` + `.json` | Size S body (170cm) |
| `makehuman_male_M.ply` + `.json` | Size M body (180cm) |
| `makehuman_male_XL.ply` + `.json` | Size XL body (185cm) |
| `mannequin_sizeM_180cm.ply` + `_betas.json` | Legacy parametric mannequin |
| `body_profile_mannequin.json` | Legacy body profile |

### Patterns (`data/patterns/`)

GarmentCode JSON files for T-shirt patterns in 5 sizes:
`tshirt_size_{XS, S, M, L, XL}.json`

Each contains: panels (vertices, edges with Bézier curvature, translation, rotation), stitches, and `_forma_metadata`.

### GarmentCode Library (`data/garmentcode/`)

A bundled copy/submodule of the GarmentCode project, including its own GUI, samplers, and processing scripts.

---

## Fabric Library

**File:** `data/fabrics/fabric_library.json`

Five fabric definitions with XPBD material parameters:

| Fabric | Density | Stretch | Bend | Shear | Damping |
|--------|---------|---------|------|-------|---------|
| `cotton_jersey_default` | 0.18 kg/m² | 40 | 0.005 | 3.0 | 0.995 |
| `cotton_jersey_heavy` | 0.30 kg/m² | 55 | 0.015 | 5.0 | 0.993 |
| `polyester_woven` | 0.12 kg/m² | 80 | 0.008 | 8.0 | 0.997 |
| `denim_12oz` | 0.40 kg/m² | 120 | 0.08 | 15.0 | 0.990 |
| `silk_charmeuse` | 0.07 kg/m² | 25 | 0.001 | 1.5 | 0.998 |

---

## Seam Manifests

**Directory:** `seam_manifests/`

Pre-built seam manifests for each pattern size: `tshirt_size_{XS,S,M,L,XL}_manifest.json`

Each contains: panel records with edge IDs, vertices, and arc lengths; seam pairs with arc-length-diff validation; stitch type classification (standard vs gather).

---

## Scripts

**Directory:** `scripts/` (8 files)

| Script | Purpose |
|--------|---------|
| `generate_tshirt_patterns.py` | Generate T-shirt patterns for S, M, XL |
| `generate_xs_l_patterns.py` | Generate additional XS, L sizes |
| `run_5size_sweep.py` | Batch simulation across all 5 sizes |
| `run_all_simulations.py` | Run full simulation suite |
| `run_crossbody_sweep.py` | Cross-body size sweep (garment × body) |
| `render_week2_viz.py` | Generate Week 2 visualization |
| `spike_garmentcode_sim.py` | GarmentCode integration spike |
| `spike_pygarment.py` | pygarment integration spike (deprecated) |

---

## Test Suite

**Directory:** `tests/` (8 test files)

| Test File | What It Tests |
|-----------|--------------|
| `test_geometer.py` | XPBD simulation, clearance computation, convergence |
| `test_patterns.py` | Pattern loading, Bézier curves, edge discretization |
| `test_mannequin.py` | Body mesh generation, measurements, validation |
| `test_schema_compliance.py` | fit_verdict.json v1.2 schema validation |
| `test_week2_pipeline.py` | End-to-end pipeline with fabric parameters |
| `test_week2_sizes.py` | Multi-size simulation correctness |
| `test_week2_fabrics.py` | Fabric sensitivity (different materials) |
| `test_ac2_sleeve_collision_strain.py` | Sleeve collision and strain ratio validation |

---

## Output Artifacts

### Verdicts (`output/verdicts/`)

- `vrd_S_on_M.json` — Size S garment on size M body
- `vrd_M_on_M.json` — Size M garment on size M body
- `vrd_XL_on_M.json` — Size XL garment on size M body
- `fabric_test_*.json` — Fabric sensitivity test results
- `crossbody/` — Cross-body sweep results
- `extended/` — Extended test results

### Visualizations (`output/`)

- `week1_visualization.png` — Week 1 results visualization
- `week2_visualization.png` — Week 2 results visualization

---

## Specification Documents

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Project context, architecture summary, non-negotiable rules, dependency constraints |
| `FORMA_CONTEXT.md` | Full project context and background |
| `FORMA_GEOMETER_SPEC.md` | Geometry pipeline specification (region segmentation, garment assembly, clearance) |
| `FORMA_SEAM_MANIFEST_SCHEMA.md` | Seam manifest JSON schema and validation rules |
| `FORMA_FABRIC_LIBRARY.md` | XPBD material parameter documentation |
| `FORMA_PROMPT_PLAYBOOK.md` | Development workflow playbook |
| `FORMA_WEEK1_SPEC.md` | Week 1 acceptance criteria and output schemas |
| `FORMA_WEEK2_SPEC_PHYSICS.md` | Week 2 physics spec (initial) |
| `FORMA_WEEK2_SPEC_PHYSICS_v2.2.md` | Week 2 physics spec (current, v2.2) |
| `FORMA_WEEK3_SPEC_SUPABASE.md` | Week 3 Supabase persistence spec |
| `FORMA_WEEK4_SPEC.md` | Week 4 spec |
| `FORMA_POST_SPIKE_ADDENDUM.md` | Post-dependency-spike decisions |
| `docs/spike_results.md` | AC-0 dependency spike findings |

---

## Technology Stack & Constraints

### Stack

| Tool | Use |
|------|-----|
| Python 3.11 | Runtime (not 3.12+ due to NvidiaWarp AST compat) |
| numpy | XPBD solver, linear algebra |
| scipy | KDTree (collision/clearance), Delaunay (triangulation), spline interpolation |
| trimesh | Mesh I/O (PLY/OBJ), triangulation, normals |
| pytest | Test suite |

### Hard Constraints

- **No pygarment** — Cannot be imported (libcairo missing, cgal broken)
- **No SMPL-X** — Licensing not cleared for commercial use
- **No CUDA locally** — Pure-Python XPBD solver; NvidiaWarp only on Colab
- **No Python 3.12+** — NvidiaWarp fork uses `ast.Str` (removed in 3.12)

---

## Key Design Decisions

1. **Body-conformal wrapping** — Garment panels are placed using the body's actual radius profile (Y × θ), not a simple cylinder. This produces realistic fit from the initial placement and reduces simulation time.

2. **Quasi-static XPBD** — Instead of dynamic cloth simulation, uses iterative constraint projection with heavily reduced gravity. The garment is pre-placed near equilibrium; physics just settles it.

3. **Dual clearance metrics** — KDTree signed distance for garments near the body surface, supplemented by circumference-based radial clearance to handle undersized garments placed inside the body.

4. **Strain ratio (AC-2)** — Post-collision strain metric (`current_length / rest_length`) provides a secondary fit signal beyond clearance alone. Thresholds: `> 1.15 → red`, `> 1.08 → yellow`.

5. **Bending resistance offset (AC-3)** — Post-sim adjustment pushing garment vertices along body normals based on fabric bend stiffness. Stiffer fabrics bridge concavities; softer fabrics conform.

6. **Sleeve warm-up protocol** — Sleeve seam constraints start with a 5mm correction cap for 20 steps to prevent velocity explosion from large initial gaps, then relax to 5cm.

7. **Seam manifest validation** — Arc-length tolerance (2mm standard, 10mm gather), orphan edge detection, duplicate assignment checks. Validation errors halt the pipeline.

8. **MakeHuman CC0 body** — Uses the CC0-licensed MakeHuman base mesh with morph targets and iterative XZ scaling instead of SMPL-X (licensing concerns). Achieves ≤ 2mm measurement accuracy.

---

*Generated from codebase review — March 2026*
