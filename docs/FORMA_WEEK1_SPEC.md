# FORMA — Week 1 Spec: XPBD Smoke Test
**Version:** 1.1  
**Phase:** 1 / Week 1  
**Owner:** Founder  
**Executor:** Claude Code CLI  
**Status:** READY TO BUILD  
**Companion Docs:**  
- `FORMA_GEOMETER_SPEC.md` — geometry pipeline implementation (measurement-to-beta, region segmentation, clearance computation)  
- `FORMA_SEAM_MANIFEST_SCHEMA.md` — seam_manifest.json schema and validation rules  
- `FORMA_FABRIC_LIBRARY.md` — material parameter definitions for XPBD simulation

---

## Objective

Validate the end-to-end physics pipeline using a synthetic mannequin body mesh and a
GarmentCode T-shirt pattern. Produce a `fit_verdict.json` that is physically correct
and matches the v1.2 output schema.

This is a **go/no-go gate**. If the strain maps at the end of Week 1 do not reflect
physical reality (size S on size M body → red everywhere, size XL → green with ease),
the pipeline has a bug that must be resolved before any further work begins.

---

## Acceptance Criteria

All criteria must pass for Week 1 to be considered complete.

### AC-0: Dependency Spike (DO THIS FIRST — 30 minutes)
Before writing any production code, run these two spike scripts to verify assumptions:

**Spike 1 — pygarment API surface:**
```bash
python scripts/spike_pygarment.py
```
Determines whether pygarment can generate SMPL-X meshes with controlled shape
parameters. If it cannot, the BetaSolver from FORMA_GEOMETER_SPEC.md is required.

**Spike 2 — GarmentCode simulation pipeline:**
```bash
python scripts/spike_garmentcode_sim.py
```
Determines whether NvidiaWarp-GarmentCode handles garment assembly and placement
internally, or if we need to build panel triangulation + seam stitching + initial
positioning ourselves.

**Pass conditions:**
- Both spike scripts run without crash
- A written note in `docs/spike_results.md` documenting:
  - Whether pygarment handles measurement-to-beta conversion (yes/no)
  - Whether GarmentCode handles garment assembly (yes/no)
  - Any API surprises or missing functionality
  - Decision on which fallback path to take if either answer is "no"

**If pygarment cannot generate bodies from measurements:**
- Install `smplx` and `scipy` directly
- Download SMPL-X model files from https://smpl-x.is.tue.mpg.de/
- Use BetaSolver from `src/sculptor/beta_solver.py` (see FORMA_GEOMETER_SPEC.md)

### AC-1: Mannequin Generation
- Generate a synthetic body mesh using SMPL-X shape parameters via `pygarment`
  (or via smplx + BetaSolver if AC-0 spike indicates pygarment cannot do this)
- Parameters: male, height=180cm, chest=96cm, waist=80cm, hip=95cm (standard size M)
- **Verify generated measurements:** after mesh generation, re-measure the mesh
  (chest circumference, waist circumference, hip circumference, height) and confirm
  all are within ±2mm of target. Save achieved measurements to body_profile.
- Output: `mannequin_sizeM_180cm.ply` saved to `data/bodies/`
- Output: `body_profile_mannequin.json` matching the v1.2 schema (see schema section)
- Output: `data/bodies/mannequin_sizeM_180cm_betas.json` (SMPL-X beta coefficients for reproducibility)
- `scan_method` must be `"synthetic_mannequin"`
- `scan_accuracy_mm` must be `0`
- `confidence` must be `1.0`

### AC-2: Pattern Loading
- Load T-shirt patterns for sizes S, M, and XL from GarmentCodeData via `pygarment`
- Each pattern must load cleanly with no schema errors
- Each pattern must produce a valid `seam_manifest.json` via the Tailor converter
- Seam manifest must conform to FORMA_SEAM_MANIFEST_SCHEMA.md (see companion doc)
- Seam pairs must validate within ±2mm tolerance
- Raise `SeamValidationError` on any validation failure — never silently proceed
- Output: `patterns/tshirt_size_S.json`, `tshirt_size_M.json`, `tshirt_size_XL.json`
- Output: `seam_manifests/tshirt_size_S_manifest.json` (and M, XL equivalents)

### AC-3: XPBD Simulation Runs to Convergence
- Run NvidiaWarp-GarmentCode XPBD simulation for each size (S, M, XL) on the size M mannequin
- **Garment assembly:** if AC-0 spike shows GarmentCode handles assembly internally, use it.
  If not, implement panel triangulation + seam stitching + initial placement per FORMA_GEOMETER_SPEC.md.
- **Primary convergence criterion:** total kinetic energy drops below 1e-6 J (see FORMA_GEOMETER_SPEC.md)
- **Secondary convergence check:** fewer than 1.5% of vertices moving more than 0.4mm per frame
- Maximum 200 timesteps per simulation run
- No cloth explosion: if kinetic energy spikes >10x between frames, raise `SimulationExplosionError`
- No tunnel-through: fewer than 2% of garment vertices may be inside body mesh (signed distance check)
- Vertices must remain within 500mm of body surface (sanity bound)
- Log simulation time in milliseconds per run
- Log convergence step number and final kinetic energy for each run

### AC-4: Clearance Map Physical Correctness
This is the core validation. The schema field is named `strain_map` for historical
reasons, but the values represent **signed clearance** (garment-to-body surface
distance), not mechanical fabric strain. See FORMA_GEOMETER_SPEC.md Sub-Problem 4
for the computation algorithm.

**All clearance values must come from the actual simulation geometry pipeline —
not from hardcoded numbers.** The Geometer must compute signed distance between
draped garment vertices and body surface per region, using body region segmentation
defined in FORMA_GEOMETER_SPEC.md Sub-Problem 3.

The following conditions must all be true:

| Test | Expected Result | Pass Condition |
|------|----------------|----------------|
| Size S on size M mannequin | Tight — body larger than garment | chest_front delta_mm < −20mm (RED) |
| Size S on size M mannequin | Tight | shoulder delta_mm < −15mm (RED) |
| Size M on size M mannequin | Correct fit — small positive ease | all regions delta_mm between −5mm and +30mm |
| Size M on size M mannequin | Correct fit | no region severity = "red" |
| Size XL on size M mannequin | Loose — garment larger than body | chest_front delta_mm > +40mm (GREEN) |
| Size XL on size M mannequin | Loose | waist excess_mm > +50mm |

Negative delta_mm = body is larger than garment (strain / too tight)
Positive delta_mm = garment is larger than body (ease / too loose)

### AC-5: Output Schema Compliance
- Produce `fit_verdict.json` for each of the 3 test runs (S, M, XL)
- Each file must match the v1.2 schema exactly (see below)
- All required fields must be present and correctly typed
- Save to `output/verdicts/`

---

## Output Schema — fit_verdict.json (v1.2)

```json
{
  "verdict_id": "string — vrd_{12 char hex}",
  "fit": "boolean",
  "confidence": "float — 1.0 for synthetic_mannequin",
  "body_source": "string — enum: synthetic_mannequin | standard_photo | precision_suit",
  "scan_method": "string — synthetic_mannequin",
  "scan_accuracy_mm": "integer — 0 for synthetic",
  "garment_id": "string — e.g. tshirt_gc_v1_size_S",
  "body_profile_id": "string — e.g. mannequin_sizeM_180cm",
  "strain_map": [
    {
      "region": "string — chest_front | chest_side | shoulder_left | shoulder_right | upper_back | waist",
      "delta_mm": "float — signed clearance: negative = too tight, positive = ease",
      "severity": "string — enum: green | yellow | red"
    }
  ],
  "ease_map": [
    {
      "region": "string",
      "excess_mm": "float",
      "verdict": "string — enum: tight_fit | standard_fit | relaxed_fit | oversized"
    }
  ],
  "simulation_ms": "integer — wall clock time of XPBD run",
  "convergence_step": "integer — timestep at which simulation converged",
  "final_kinetic_energy_j": "float — kinetic energy at convergence",
  "tunnel_through_pct": "float — percentage of garment vertices inside body mesh",
  "fabric_params_used": {
    "fabric_id": "string — e.g. cotton_jersey_default",
    "type": "string — cotton_jersey | polyester_woven | denim | silk_charmeuse | etc.",
    "density_kg_m2": "float — mass per unit area (e.g. 0.15 for light jersey)",
    "stretch_stiffness": "float — Young's modulus analog for XPBD (e.g. 50.0)",
    "bend_stiffness": "float — bending resistance (e.g. 0.01)",
    "shear_stiffness": "float — shear resistance (e.g. 5.0)",
    "damping": "float — velocity damping coefficient (e.g. 0.99)"
  }
}
```

**NOTE on verdict_id:** Uses 12 hex chars (~281 trillion unique IDs) instead of the
original 6 chars (~16 million). Generate via `uuid.uuid4().hex[:12]`.
```

### Severity Thresholds
| delta_mm | severity |
|----------|----------|
| > −10mm | green |
| −10mm to −25mm | yellow |
| < −25mm | red |

### fit Boolean Logic
`fit = true` if and only if zero regions have severity `"red"`

---

## Body Profile Schema — body_profile.json (v1.2)

```json
{
  "body_profile_id": "mannequin_sizeM_180cm",
  "body_source": "synthetic_mannequin",
  "scan_method": "synthetic_mannequin",
  "scan_accuracy_mm": 0,
  "confidence": 1.0,
  "measurements": {
    "height_cm": 180,
    "chest_cm": 96,
    "waist_cm": 80,
    "hip_cm": 95,
    "shoulder_width_cm": 44,
    "inseam_cm": 82
  },
  "achieved_measurements": {
    "height_cm": 180.1,
    "chest_cm": 95.9,
    "waist_cm": 80.2,
    "hip_cm": 95.1,
    "shoulder_width_cm": 44.0,
    "inseam_cm": 82.1
  },
  "smplx_betas": [0.23, -0.41, 0.87, 0.12, -0.33, 0.05, -0.11, 0.28, -0.07, 0.15],
  "max_measurement_error_mm": 2.0,
  "mesh_path": "data/bodies/mannequin_sizeM_180cm.ply",
  "created_at": "ISO 8601 timestamp"
}
```

---

## Repository Structure

```
forma/
├── CLAUDE.md                          ← Claude Code reads this first, every session
├── data/
│   ├── bodies/
│   │   ├── mannequin_sizeM_180cm.ply
│   │   └── mannequin_sizeM_180cm_betas.json
│   ├── patterns/
│   │   ├── tshirt_size_S.json
│   │   ├── tshirt_size_M.json
│   │   └── tshirt_size_XL.json
│   └── fabrics/
│       └── fabric_library.json        ← material params for XPBD
├── seam_manifests/
│   ├── tshirt_size_S_manifest.json
│   ├── tshirt_size_M_manifest.json
│   └── tshirt_size_XL_manifest.json
├── output/
│   ├── verdicts/
│   │   ├── vrd_S_on_M.json
│   │   ├── vrd_M_on_M.json
│   │   └── vrd_XL_on_M.json
│   └── preflight/
│       └── mannequin_preflight.json   ← AC-0 mesh validation
├── docs/
│   └── spike_results.md               ← AC-0 spike findings
├── src/
│   ├── sculptor/
│   │   ├── generate_mannequin.py      ← AC-1
│   │   └── beta_solver.py             ← measurement-to-SMPL-X beta conversion
│   ├── pattern_maker/
│   │   └── load_patterns.py           ← AC-2
│   ├── tailor/
│   │   └── seam_converter.py          ← AC-2 (see FORMA_SEAM_MANIFEST_SCHEMA.md)
│   ├── geometer/
│   │   ├── xpbd_simulate.py           ← AC-3 (simulation orchestration)
│   │   ├── region_map.py              ← body region segmentation
│   │   ├── clearance.py               ← signed distance computation
│   │   └── convergence.py             ← energy-based convergence checks
│   └── verdict/
│       └── generate_verdict.py        ← AC-5
├── tests/
│   ├── test_mannequin.py
│   ├── test_patterns.py
│   ├── test_simulation.py
│   ├── test_geometer.py               ← region segmentation + clearance tests
│   └── test_schema_compliance.py
├── scripts/
│   ├── spike_pygarment.py             ← AC-0: dependency verification
│   ├── spike_garmentcode_sim.py       ← AC-0: pipeline verification
│   ├── preflight_check.py             ← mesh validation before simulation
│   └── run_smoke_test.sh              ← runs full Week 1 pipeline end-to-end
└── requirements.txt
```

---

## CLAUDE.md — Paste This Into Your Repo Root

```markdown
# FORMA — Claude Code Context

## What This Project Is
Forma is a physics-based garment fit verification system. Given a body mesh and a
garment pattern, it runs XPBD cloth simulation and returns a fit_verdict.json
stating whether the garment fits, with per-region signed clearance measurements
in millimeters.

## Architecture (v1.2)
- Sculptor: generates body meshes (MVP: synthetic SMPL-X mannequin via pygarment or smplx + BetaSolver)
- Pattern Maker: loads GarmentCode parametric patterns (pygarment + GarmentCodeData)
- Tailor: converts GarmentCode JSON to seam_manifest.json (schema converter, no LLM). See FORMA_SEAM_MANIFEST_SCHEMA.md.
- Geometer: runs XPBD simulation via NvidiaWarp-GarmentCode, extracts signed clearance per region. See FORMA_GEOMETER_SPEC.md.
- Archivist: persists body profiles, patterns, verdicts (Supabase — Week 3)
- Strategist: Claude Sonnet 4.6 — architecture and decisions only
- Executor: you (Claude Code) — implementation

## Non-Negotiable Rules
1. NEVER use a fixed confidence scalar. confidence = 1.0 for synthetic_mannequin only.
   For photo-based bodies: f(scan_accuracy_mm, strain_magnitude, body_region).
2. ALWAYS validate seam_manifest.json before passing to Geometer.
   If seam validation fails, raise SeamValidationError — never silently proceed.
3. body_source field is REQUIRED in every body_profile.json and fit_verdict.json.
   Enum: ["synthetic_mannequin", "standard_photo", "precision_suit"]
4. Negative delta_mm = body larger than garment (too tight / compressed).
   Positive delta_mm = garment larger than body (ease / loose). Never invert this.
   delta_mm is SIGNED CLEARANCE, not mechanical fabric strain.
5. fit = True if and only if zero regions have severity "red". No exceptions.
6. NEVER hardcode verdict values. All strain_map and ease_map values must come
   from the actual Geometer geometry pipeline (signed distance computation).
7. All fabric parameters must come from data/fabrics/fabric_library.json.
   Never use magic numbers like "stretch: 0.7" — use the full param set.
8. verdict_id must use 12 hex chars (vrd_{12 hex}), not 6.

## Key Spec Documents (read before implementing)
- FORMA_WEEK1_SPEC.md — acceptance criteria and output schemas
- FORMA_GEOMETER_SPEC.md — geometry pipeline: beta solver, region segmentation, clearance computation
- FORMA_SEAM_MANIFEST_SCHEMA.md — seam_manifest.json schema and validation rules
- FORMA_FABRIC_LIBRARY.md — XPBD material parameters

## Output Schema
All outputs must match v1.2 schema defined in FORMA_WEEK1_SPEC.md.
Key addition: fabric_params_used now requires fabric_id, density_kg_m2,
stretch_stiffness, bend_stiffness, shear_stiffness, damping.

## Stack
- Python 3.11+
- pygarment — pattern generation (+ mannequin if it supports beta control)
- smplx, scipy — SMPL-X body model + BetaSolver (fallback if pygarment lacks body generation)
- NvidiaWarp-GarmentCode — XPBD simulation (build from source, see README)
- ezdxf, shapely — DXF geometry
- trimesh, scipy — mesh I/O, KDTree for clearance computation
- pytest — all tests

## Current Phase
Phase 1, Week 1 — XPBD smoke test. See FORMA_WEEK1_SPEC.md for full spec.
```

---

## Stack & Installation

```bash
# Python environment
python -m venv forma-env
source forma-env/bin/activate  # or forma-env\Scripts\activate on Windows

# Core dependencies — pinned for reproducibility
pip install pygarment==0.5.5        # or latest — check PyPI
pip install ezdxf==0.18.1 shapely==2.0.6
pip install trimesh==4.5.3          # mesh I/O + operations (replaces pytorch3d for Phase 1)
pip install scipy==1.14.1           # optimization (BetaSolver) + KDTree (clearance)
pip install open3d==0.18.0          # point cloud visualization (optional for debugging)
pip install smplx==0.1.28           # SMPL-X body model (fallback if pygarment lacks beta control)
pip install pytest==8.3.4

# NOTE: pytorch3d is NOT needed for Week 1.
# trimesh + scipy handle all mesh operations the Geometer requires.
# Defer pytorch3d to Phase 2 (mesh fusion).

# NvidiaWarp-GarmentCode — build from source
git clone https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode
cd NvidiaWarp-GarmentCode
python build_lib.py             # requires CUDA Toolkit 11.5+ and GCC 7.2+
pip install -e .
cd ..
```

**Note for M3 Mac:** NvidiaWarp-GarmentCode requires CUDA. On Apple Silicon, build
will run CPU-only (no CUDA available). This is acceptable for smoke test development.
Run final validation on Google Colab (A100) where CUDA is available.
Expect simulation times of **2–10 seconds on M3 CPU** vs ~100–300ms on A100 GPU.
(The original estimate of ~100ms on M3 was too optimistic for CPU-only XPBD with
body collision + self-collision.)

---

## Test Suite

### test_simulation.py — Core Validation

```python
import pytest
import json
from pathlib import Path

VERDICT_DIR = Path("output/verdicts")

def load_verdict(filename):
    with open(VERDICT_DIR / filename) as f:
        return json.load(f)

def get_region(verdict, region_name):
    for r in verdict["strain_map"]:
        if r["region"] == region_name:
            return r
    return None

class TestSizeSOnSizeMBody:
    """Size S garment on size M body — should be RED everywhere tight"""

    def setup_method(self):
        self.v = load_verdict("vrd_S_on_M.json")

    def test_fit_is_false(self):
        assert self.v["fit"] == False

    def test_chest_front_is_red(self):
        r = get_region(self.v, "chest_front")
        assert r["delta_mm"] < -20, f"Expected < -20mm, got {r['delta_mm']}"
        assert r["severity"] == "red"

    def test_shoulder_is_red(self):
        r = get_region(self.v, "shoulder_left")
        assert r["delta_mm"] < -15
        assert r["severity"] == "red"

    def test_confidence_is_1(self):
        assert self.v["confidence"] == 1.0

    def test_body_source_is_synthetic(self):
        assert self.v["body_source"] == "synthetic_mannequin"


class TestSizeMOnSizeMBody:
    """Size M garment on size M body — should FIT with small positive ease"""

    def setup_method(self):
        self.v = load_verdict("vrd_M_on_M.json")

    def test_fit_is_true(self):
        assert self.v["fit"] == True

    def test_no_red_regions(self):
        for r in self.v["strain_map"]:
            assert r["severity"] != "red", f"Region {r['region']} is red — should fit"

    def test_chest_in_tolerance(self):
        r = get_region(self.v, "chest_front")
        assert -5 <= r["delta_mm"] <= 30, f"Unexpected delta: {r['delta_mm']}mm"

    def test_confidence_is_1(self):
        assert self.v["confidence"] == 1.0


class TestSizeXLOnSizeMBody:
    """Size XL garment on size M body — should be green/loose everywhere"""

    def setup_method(self):
        self.v = load_verdict("vrd_XL_on_M.json")

    def test_fit_is_true(self):
        assert self.v["fit"] == True

    def test_chest_front_has_large_ease(self):
        r = get_region(self.v, "chest_front")
        assert r["delta_mm"] > 40, f"Expected > 40mm ease, got {r['delta_mm']}"

    def test_waist_has_large_ease(self):
        for e in self.v["ease_map"]:
            if e["region"] == "waist":
                assert e["excess_mm"] > 50


class TestSchemaCompliance:
    """All verdicts must match v1.2 schema"""

    REQUIRED_FIELDS = [
        "verdict_id", "fit", "confidence", "body_source",
        "scan_method", "scan_accuracy_mm", "garment_id",
        "body_profile_id", "strain_map", "ease_map",
        "simulation_ms", "convergence_step", "final_kinetic_energy_j",
        "tunnel_through_pct", "fabric_params_used"
    ]

    REQUIRED_FABRIC_FIELDS = [
        "fabric_id", "type", "density_kg_m2",
        "stretch_stiffness", "bend_stiffness", "shear_stiffness", "damping"
    ]

    REQUIRED_REGIONS = [
        "chest_front", "chest_side", "shoulder_left",
        "shoulder_right", "upper_back", "waist"
    ]

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_required_fields_present(self, filename):
        v = load_verdict(filename)
        for field in self.REQUIRED_FIELDS:
            assert field in v, f"Missing field: {field} in {filename}"

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_verdict_id_format(self, filename):
        v = load_verdict(filename)
        vid = v["verdict_id"]
        assert vid.startswith("vrd_"), f"verdict_id must start with 'vrd_': {vid}"
        hex_part = vid[4:]
        assert len(hex_part) == 12, f"verdict_id hex part must be 12 chars, got {len(hex_part)}"
        int(hex_part, 16)  # raises ValueError if not valid hex

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_fabric_params_complete(self, filename):
        v = load_verdict(filename)
        fp = v["fabric_params_used"]
        for field in self.REQUIRED_FABRIC_FIELDS:
            assert field in fp, f"Missing fabric field: {field} in {filename}"

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_all_regions_present(self, filename):
        v = load_verdict(filename)
        regions = [r["region"] for r in v["strain_map"]]
        for required in self.REQUIRED_REGIONS:
            assert required in regions, f"Missing region: {required} in {filename}"

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_ease_map_has_all_regions(self, filename):
        v = load_verdict(filename)
        regions = [r["region"] for r in v["ease_map"]]
        for required in self.REQUIRED_REGIONS:
            assert required in regions, f"Missing ease_map region: {required} in {filename}"

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_body_source_is_valid_enum(self, filename):
        v = load_verdict(filename)
        valid = ["synthetic_mannequin", "standard_photo", "precision_suit"]
        assert v["body_source"] in valid

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_severity_values_valid(self, filename):
        v = load_verdict(filename)
        for r in v["strain_map"]:
            assert r["severity"] in ["green", "yellow", "red"]

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_tunnel_through_within_bounds(self, filename):
        v = load_verdict(filename)
        assert v["tunnel_through_pct"] < 2.0, f"Tunnel-through too high: {v['tunnel_through_pct']}%"
```

---

## Run Command for Claude Code

Hand Claude Code this exact instruction on Day 1:

```
Read CLAUDE.md first. Then read FORMA_WEEK1_SPEC.md, FORMA_GEOMETER_SPEC.md,
and FORMA_SEAM_MANIFEST_SCHEMA.md.

Your task is to implement all acceptance criteria in order:
AC-0 (spike) → AC-1 → AC-2 → AC-3 → AC-4 → AC-5.

AC-0 is a 30-minute dependency spike. Run the two spike scripts and document
findings in docs/spike_results.md BEFORE writing any production code.

Do not proceed to the next AC until the current one passes its tests.
After each AC, run the relevant test file and show me the results.
After all ACs pass, run scripts/run_smoke_test.sh end-to-end and
show me the three fit_verdict.json outputs side by side.

Do not hardcode any verdict values. All outputs must come from
real simulation runs through the Geometer pipeline.
```

---

## Go / No-Go Decision Criteria

At the end of Week 1, review the three verdict outputs manually.

**GO if:**
- All 5 ACs pass
- pytest suite is green
- Size S verdict shows red on chest and shoulders
- Size M verdict shows no red regions
- Size XL verdict shows large positive ease values
- Simulation runs to convergence without explosion or tunneling

**NO-GO if:**
- Size M on size M body shows red regions (physics is wrong)
- Size S shows positive ease on chest (delta direction inverted)
- Simulation explodes or fails to converge
- Any required schema field is missing

If NO-GO: bring the failing verdict JSON and simulation logs back to
claude.ai (Strategist layer) for diagnosis before continuing.

---

*FORMA Architecture v1.2 — Phase 1 Week 1 Spec*  
*Do not modify acceptance criteria without founder sign-off*
