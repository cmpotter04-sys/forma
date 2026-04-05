# AC-0 Dependency Spike Results
**Date:** 2026-03-10
**Environment:** M3 MacBook, Python 3.14.0, macOS Darwin 24.6.0 (arm64)
**Spec version:** FORMA_WEEK1_SPEC.md v1.1 / FORMA_GEOMETER_SPEC.md v1.0

---

## Executive Summary

| Question | Answer |
|----------|--------|
| Can pygarment generate SMPL-X meshes with controlled measurements? | **NO** |
| Does pygarment expose GarmentCode pattern loading? | **PARTIALLY** (blocked by libcairo) |
| Does NvidiaWarp-GarmentCode handle garment assembly internally? | **NO** (assembly is in pygarment 2.x which cannot be installed) |
| Does NvidiaWarp-GarmentCode include simulation capabilities? | **YES, but not buildable on M3 Mac + Python 3.14** |
| Can we run the full pipeline locally on this machine right now? | **NO — 3 hard blockers (see below)** |

---

## Spike 1: pygarment API Surface

### Installation

```bash
pip install pygarment          # resolved to 1.0.0.post1
                                # 2.x blocked — cgal dependency has corrupt PyPI metadata
                                # (every 2.x cgal wheel version-mismatch errors during resolve)
```

pygarment 2.0.x depends on `cgal`, `libigl`, `pyrender`, `nicegui`, `trimesh`. The `cgal` PyPI package has persistent metadata version mismatches that prevent pip from resolving pygarment ≥ 2.0.0. **pygarment 1.0.0.post1 installed.**

### Import Failure

```
OSError: no library called "cairo-2" was found
         cannot load library 'libcairo.2.dylib'
```

**Root cause:** pygarment → component.py → pattern.wrappers → cairosvg → cairocffi → libcairo (system C library). libcairo is not present on this machine. No package manager available (no brew, apt, port, conda) to install it.

**pygarment cannot be imported at all on this machine without installing libcairo.**

### Source-Level Inspection (grep-based, no import required)

The installed package contains 12 Python source files. Key findings:

#### What pygarment IS
A **parametric sewing pattern design framework**. It provides building blocks for defining garment patterns in code:
- `Panel` — a single flat piece of fabric with edges
- `Component` — a garment composed of panels with stitching rules
- `Edge`, `EdgeSeqFactory` — 2D curve geometry for panel outlines
- `Interface`, `Stitches` — seam edge connectivity
- `BodyParametrizationBase` — a YAML loader for body measurement files (NOT a mesh generator)

#### What pygarment is NOT
- **No SMPL-X body mesh generation** — zero references to SMPL, smplx, beta coefficients, or body mesh generation in any source file
- **No 3D simulation** — zero references to warp, XPBD, cloth simulation, or physics
- **No measurement-to-beta conversion** — `BodyParametrizationBase` is purely a YAML key-value store for driving pattern *dimensions* (e.g., chest=96cm → sleeve length formula), not for generating a 3D body

#### component.assembly() — what it actually does
```python
def assembly(self):
    """Returns simulator friendly description of component sewing pattern"""
    # Produces a VisPattern: a JSON spec with 'panels' dict and 'stitches' list
    # This is a 2D PATTERN DESCRIPTION, not a 3D assembled/draped mesh
    # Comment in panel.py: "SIM Note that Qualoth simulator does not support
    #   internal loops in panels" — references Qualoth (Maya plugin), not NvidiaWarp
```

The `assembly()` output is the INPUT to a simulator, not the output of simulation. pygarment is to simulation what a tailor's pattern is to a finished garment.

#### GarmentCode pattern loading
`pattern.core.BasicPattern` and `ParametrizedPattern` are installed alongside pygarment and CAN load GarmentCode JSON files. However:
- `ParametrizedPattern` can load `.json` pattern files and apply parameter values
- Import is blocked by `pattern.wrappers` → cairosvg → libcairo chain

**Workaround:** GarmentCode JSON patterns can be loaded with `pattern.core` directly (not via `pygarment`), but this requires patching or mocking the cairosvg import at the module level.

### VERDICT: Spike 1

**Can pygarment generate SMPL-X meshes with controlled shape parameters?**
**NO.** pygarment has no body mesh generation capability at any version.

**Fallback decision:**
Use `smplx` + `scipy` BetaSolver from `FORMA_GEOMETER_SPEC.md` exactly as specified. Both are installed and importable.

---

## Spike 2: GarmentCode / NvidiaWarp Simulation Pipeline

### NvidiaWarp-GarmentCode — Installation Attempt

**PyPI warp-lang (official NVIDIA warp):**
```bash
pip install warp-lang           # installed 1.12.0 (arm64 wheel, CPU-only)
```
- Imports successfully: `import warp as wp; wp.init()` → `CUDA not enabled in this build`
- Provides only: `warp.array`, `warp.kernel`, `warp.func` — the JIT compilation layer
- **Does NOT include `warp.sim`** (the simulation/ModelBuilder module). The sim module is separate from the core JIT layer.

**NvidiaWarp-GarmentCode fork (build from source):**
```bash
git clone --depth 1 https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode /tmp/NvidiaWarp-GarmentCode
cd /tmp/NvidiaWarp-GarmentCode
python build_lib.py             # NOT attempted — pre-screened: requires CUDA Toolkit + GCC
```

**Build would fail because:** CUDA Toolkit is required. M3 Mac has no CUDA support (Apple Silicon uses Metal, not CUDA).

**Import from source also fails (Python 3.14 incompatible):**
```
AttributeError: module 'ast' has no attribute 'Str'
```
The GarmentCode fork is based on Warp v1.0.0-beta.6 which uses `ast.Str` (removed in Python 3.12). **Incompatible with Python 3.14.**

### NvidiaWarp-GarmentCode — Source-Level Inspection

Repository was cloned successfully. Key structure:
```
NvidiaWarp-GarmentCode/
├── warp/sim/
│   ├── integrator_xpbd.py     ← XPBDIntegrator class with simulate() method
│   ├── model.py               ← ModelBuilder for simulation state
│   ├── collide.py             ← collision detection kernels
│   └── ...
├── warp/collision/
│   └── panel_assignment.py    ← GarmentCode-specific: SMPL body segmentation,
│                                  panel-to-body-part assignment, mesh filtering
└── README.md
```

**Changes introduced for GarmentCodeData** (from README):
- Self-collisions (point-triangle + edge-edge)
- Attachment constraints
- **Body-part drag for initial collision resolution** — drags intersecting garment panels toward their corresponding body parts until collisions are resolved
- Body model collision constraint (pushes cloth vertices inside body to outside)

**Does GarmentCode handle garment assembly internally?**

The NvidiaWarp-GarmentCode fork provides **only the physics solver**. It handles:
- XPBD simulation (springs, bending, shear constraints)
- Body collision resolution
- Self-collision detection
- Body-part-based initial placement drag

It does **NOT** handle:
- Panel triangulation (converting 2D polygon outlines to triangle meshes)
- Seam stitching (adding distance constraints between matched seam edge vertices)
- Initial 3D placement (positioning flat panels around the body before simulation)

Per the README: "This simulator version is used by **PyGarment v2.0.0+**". The garment assembly pipeline lives in pygarment 2.x (which we cannot install due to cgal issues). The warp fork is the physics backend; pygarment 2.x is the frontend that does triangulation + placement + seam setup.

**`panel_assignment.py` (GarmentCode-specific):**
```python
def process_body_seg(seg, smpl_parts=False, limbs_merge=False):
    """Merge SMPL segmentation into bigger chunks for body collision filtering"""
    smpl_parts = {
        "left_arm": ['leftArm', 'leftForeArm', 'leftHand', 'leftHandIndex1'],
        "right_arm": [...],
        "body": ['spine', 'spine1', 'spine2', 'neck', 'leftShoulder', ...],
    }
```
This is the SMPL body-part segmentation used for initial collision drag. It expects SMPL vertex segmentation as input — meaning the body mesh must be SMPL/SMPL-X format for the GarmentCode-specific collision drag to work correctly.

### CPU-Only Simulation Path

**warp-lang 1.12.0 (PyPI) on M3 Mac:**
- CPU device confirmed working: `wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device='cpu')` ✅
- Does NOT include `warp.sim` module
- **Cannot run XPBD simulation** with the PyPI version alone

**Alternative CPU path:**
The GarmentCode-specific XPBD additions (body-part drag, self-collision) are only in the fork. The standard warp.sim XPBD integrator exists in the PyPI warp at a different path (not confirmed available in 1.12.0). For smoke test purposes, a pure-Python XPBD implementation using numpy/scipy would be feasible (~200 timesteps, ~10,000 vertices, expected runtime 2-30s on M3 CPU).

### VERDICT: Spike 2

**Does NvidiaWarp-GarmentCode handle garment assembly internally?**
**NO.** Assembly (triangulation + seam stitching + initial placement) is in pygarment 2.x, not in the warp fork.

**What must we build ourselves:**
1. Panel triangulation: convert 2D polygon panel outlines → triangle meshes (edge length ~5mm)
2. Seam stitching: sample matched points along seam edge pairs, add distance constraints (rest_length=0)
3. Initial placement: position flat panels around body with ~20mm clearance, aligned by body torso center
4. XPBD physics loop: either use the GarmentCode warp fork on Colab/Linux, or implement a CPU-only XPBD in numpy/scipy for local dev

---

## Hard Blockers Summary

### Blocker 1: libcairo missing
- **Affects:** pygarment import, pattern.wrappers import
- **Impact:** Cannot use pygarment API at runtime on this machine
- **Workaround:** Load GarmentCode JSON patterns directly via `pattern.core.BasicPattern` with a mock/stub for cairosvg, OR run on a system with cairo installed (Linux with apt, or Colab)

### Blocker 2: Python 3.14 — too new
- **Affects:** NvidiaWarp-GarmentCode fork (uses ast.Str removed in 3.12)
- **Impact:** Cannot import the GarmentCode warp fork locally
- **Workaround:** Use Python 3.11 environment. The spec says "Python 3.11+" — use 3.11 not 3.14.

### Blocker 3: No CUDA on M3 Mac
- **Affects:** NvidiaWarp-GarmentCode build
- **Impact:** GPU simulation unavailable; expected 2-10s CPU simulation per run
- **Workaround:** CPU-only mode acceptable for smoke test development. Spec explicitly notes this. Run final validation on Google Colab (A100).

---

## Decisions for AC-1 onward

### Body Generation (AC-1)
**Decision: Use smplx + scipy BetaSolver as specified in FORMA_GEOMETER_SPEC.md**

```bash
# Already installed:
smplx==0.1.28
scipy==1.17.1
torch==2.10.0

# Still required (manual download — not pip-installable):
# SMPL-X model files from https://smpl-x.is.tue.mpg.de/
# Place at: data/smplx_models/SMPLX_MALE.npz (and FEMALE, NEUTRAL)
```

### Pattern Loading (AC-2)
**Decision: Load GarmentCode JSON patterns directly via pattern.core, bypassing VisPattern**

```python
# Use pattern.core.ParametrizedPattern directly (no cairo dependency)
# Mock or skip cairosvg import at module level
# GarmentCode pattern files are plain JSON — can be loaded without pygarment
```

### Simulation (AC-3)
**Decision: Two-track approach**

- **Local dev (M3 Mac):** Pure-Python XPBD using numpy/scipy. Slower but runnable. Used for unit tests and schema validation.
- **Full validation:** Google Colab A100, Python 3.10, NvidiaWarp-GarmentCode fork from source. Run final smoke test there.

### Garment Assembly
**Decision: Build the assembly pipeline ourselves** (panel triangulation + seam stitching + initial placement), as specified in FORMA_GEOMETER_SPEC.md Sub-Problem 2.

---

## API Surprises

1. **pygarment 1.x ≠ pygarment 2.x**: These are substantially different. v1 is pattern-design-only; v2 integrates with NvidiaWarp for simulation. The spec likely assumed v2, but cgal blocks v2 installation.

2. **warp-lang (PyPI) ≠ NvidiaWarp-GarmentCode (GitHub fork)**: The PyPI package is the official NVIDIA warp. The GarmentCode fork is a separate build with different capabilities. Using `pip install warp-lang` does NOT give you the GarmentCode-specific simulation features.

3. **BodyParametrizationBase is just a YAML loader**: The name suggests body parameterization, but it only reads key-value measurement files. It does not connect to any 3D model.

4. **pygarment references Qualoth, not NvidiaWarp in v1**: The `panel.assembly()` docstring says "SIM Note that Qualoth simulator does not support..." — indicating v1 was designed for Qualoth (a commercial Maya cloth sim plugin). NvidiaWarp integration came in v2.

5. **shapely 2.0.6 has no Python 3.14 wheel**: Must use 2.1.2+ for Python 3.14. shapely 2.0.6 requires GEOS system library to build from source (unavailable on this machine).

6. **open3d has no Python 3.14 wheel**: Entirely unavailable. Spec lists it as "optional for debugging" — omit from requirements until Python 3.11 environment is established.

---

## Exact pip Commands That Worked

```bash
python -m venv forma-env
source forma-env/bin/activate
pip install --upgrade pip

pip install pygarment           # → 1.0.0.post1 (2.x blocked by cgal)
pip install shapely             # → 2.1.2 (NOT 2.0.6 — no py3.14 wheel for 2.0.6)
pip install ezdxf               # → 1.4.3 (NOT 0.18.1 — the .18.1 has no pure-py wheel)
pip install trimesh             # → 4.11.3
pip install scipy               # → 1.17.1
pip install smplx               # → 0.1.28
pip install pytest              # → 9.0.2
pip install warp-lang           # → 1.12.0

# FAILED:
# pip install open3d            # No Python 3.14 wheel available
# pip install shapely==2.0.6    # No Python 3.14 wheel, GEOS missing for source build
# pip install ezdxf==0.18.1     # Built fine actually (pure python), superseded by 1.4.3
```

---

## Exact Package Versions Installed

```
CairoSVG==2.8.2
Jinja2==3.1.6
MarkupSafe==3.0.3
PySimpleGUI==5.0.8.3
PyYAML==6.0.3
Pygments==2.19.2
cairocffi==1.7.1
cffi==2.0.0
contourpy==1.3.3
cssselect2==0.9.0
cycler==0.12.1
defusedxml==0.7.1
ezdxf==1.4.3
filelock==3.25.1
fonttools==4.62.0
fsspec==2026.2.0
iniconfig==2.3.0
kiwisolver==1.5.0
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.3
packaging==26.0
pillow==12.1.1
psutil==7.2.2
pycparser==3.0
pygarment==1.0.0.post1
pyparsing==3.3.2
pytest==9.0.2
python-dateutil==2.9.0.post0
scipy==1.17.1
setuptools==82.0.1
shapely==2.1.2
six==1.17.0
smplx==0.1.28
svgpathtools==1.7.2
svgwrite==1.4.3
sympy==1.14.0
tinycss2==1.5.1
torch==2.10.0
trimesh==4.11.3
typing_extensions==4.15.0
warp-lang==1.12.0
webencodings==0.5.1
```

**Python version:** 3.14.0
**Platform:** darwin arm64 (M3 Mac)

**NOT installed (failures):**
- `open3d` — no Python 3.14 wheel on any platform
- `shapely==2.0.6` — no Python 3.14 wheel; GEOS not available for source build
- `pygarment>=2.0.0` — cgal PyPI metadata corrupt (version string mismatches block all 2.x versions)

---

*FORMA AC-0 Spike Results — stop here and await founder review before AC-1*
