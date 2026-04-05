# FORMA – Post-Spike Build Approach (AC-0 Addendum)
**Date:** 2026-03-10
**Triggered by:** AC-0 spike results (docs/spike_results.md)
**Status:** ACTIVE – supersedes assumptions in original specs where noted

---

## What the Spike Found

Three of the original spec's assumptions were wrong:

1. **pygarment cannot generate body meshes.** It's a pattern design framework,
   not a body generator. No SMPL-X capability at any version.

2. **pygarment 2.x cannot be installed.** The cgal PyPI dependency has corrupt
   metadata. Only pygarment 1.0.0.post1 is installable, and even that requires
   libcairo (unavailable without a package manager on the dev machine).

3. **NvidiaWarp-GarmentCode does not handle garment assembly.** Assembly
   (triangulation, seam stitching, initial placement) lives in pygarment 2.x,
   which is uninstallable. The warp fork is physics-only.

Additionally: Python 3.14 is too new for the NvidiaWarp fork (uses removed
`ast.Str`) and lacks wheels for open3d and older shapely.

---

## Revised Build Decisions

### Environment: Python 3.11 (mandatory)

Recreate the virtual environment with Python 3.11 before writing any code.
Python 3.14 breaks NvidiaWarp-GarmentCode, open3d, and shapely 2.0.6.

```bash
# Install Python 3.11 if not present (pyenv or direct download)
pyenv install 3.11.11   # or download from python.org
pyenv local 3.11.11

python3.11 -m venv forma-env
source forma-env/bin/activate

# Reinstall all dependencies under 3.11
pip install smplx==0.1.28 scipy trimesh shapely ezdxf pytest torch warp-lang
pip install open3d         # should work on 3.11
```

### AC-1 Body Generation: smplx + BetaSolver

Use the `smplx` library directly with the BetaSolver from FORMA_GEOMETER_SPEC.md.
pygarment is not involved in body generation at all.

**Required manual step:** Download SMPL-X model files from
https://smpl-x.is.tue.mpg.de/ (requires registration). Place at:
```
data/smplx_models/SMPLX_MALE.npz
```

This is a one-time download. The model files are ~300MB and cannot be
pip-installed.

### AC-2 Pattern Loading: Direct JSON loading, bypass pygarment

GarmentCode patterns are plain JSON files. Load them directly with Python's
`json` module. Do not depend on pygarment's import chain (blocked by libcairo).

Two options, in order of preference:

**Option A (preferred): Use GarmentCodeData directly.**
Clone the GarmentCodeData repository and load pattern JSON files with stdlib.
The pattern files contain panel outlines, edge definitions, and stitch annotations.
Parse these into the seam_manifest.json format defined in FORMA_SEAM_MANIFEST_SCHEMA.md.

```bash
git clone https://github.com/maria-korosteleva/GarmentCodeData data/garmentcode
```

**Option B (fallback): Mock cairosvg and use pattern.core.**
Patch the cairosvg import in pygarment's pattern.wrappers to allow
`pattern.core.ParametrizedPattern` to load GarmentCode JSON without cairo.
This is fragile and depends on pygarment internals.

### AC-3 Simulation: Two-track approach

**Track 1 – Local development (M3 Mac, CPU-only):**
Pure-Python XPBD implementation using numpy/scipy. This is slower (2–30s per
run) but runnable without CUDA or the NvidiaWarp fork. Used for:
- Unit testing
- Schema validation
- Iterating on the clearance computation pipeline
- Running the Week 1 smoke test locally

The pure-Python XPBD must implement:
- Distance constraints (stretch)
- Bending constraints (dihedral angle)
- Body collision (project cloth vertices outside body surface)
- Self-collision (basic – can be simplified for smoke test)
- Gravity
- Damping

**Track 2 – Full validation (Google Colab A100):**
NvidiaWarp-GarmentCode fork built from source with CUDA. Python 3.10 or 3.11.
Used for:
- Final 200-timestep validation
- Performance benchmarking
- A100 notebook (FORMA_Week1_A100_Validation.ipynb)

### Garment Assembly: Build from scratch

The full assembly pipeline must be built per FORMA_GEOMETER_SPEC.md Sub-Problem 2:

1. **Panel triangulation:** Take 2D polygon outlines from GarmentCode JSON,
   discretize curves into polylines, triangulate with triangle or scipy.spatial.Delaunay.
   Target edge length ~5mm.

2. **Seam stitching:** Read stitch annotations from GarmentCode JSON, match
   edge pairs, resample to equal vertex counts, add XPBD distance constraints
   with rest_length=0.

3. **Initial placement:** Position panels around body torso center with ~20mm
   clearance. Align front panel to body front, back panel to body back.
   Sleeves positioned at shoulder height.

This is the most significant scope increase from the spike findings.

---

## Impact on Timeline

The spike revealed more work than originally planned. Realistic impact:

| Item | Original Estimate | Revised Estimate | Delta |
|------|------------------|------------------|-------|
| Body generation | 2–3 hours (pygarment) | 4–6 hours (smplx + BetaSolver) | +3 hours |
| Pattern loading | 1–2 hours (pygarment) | 3–4 hours (direct JSON parsing) | +2 hours |
| Garment assembly | 0 hours (assumed handled) | 8–12 hours (build from scratch) | +10 hours |
| XPBD simulation | 2–3 hours (NvidiaWarp) | 8–12 hours (pure-Python XPBD) | +9 hours |
| Environment setup | 30 min | 1–2 hours (Python 3.11 + SMPL-X download) | +1 hour |

**Total additional effort: ~25 hours.** Week 1 will likely take 5–7 days of
focused work instead of 2–3. This is acceptable within the Phase 1 timeline
if you're running parallel agents.

---

## Updated Prompt Adjustments

The prompts in FORMA_PROMPT_PLAYBOOK.md should be adjusted with these prefixes
for the Week 1 agents:

**Add to Agent A (Body + Simulation) prompt:**
```
IMPORTANT – AC-0 spike results:
- Use smplx + BetaSolver for body generation. pygarment cannot generate meshes.
- Download SMPL-X model files first: https://smpl-x.is.tue.mpg.de/
  Place SMPLX_MALE.npz at data/smplx_models/
- Use Python 3.11 environment (not 3.14).
- Build a pure-Python XPBD simulator using numpy/scipy for local dev.
  NvidiaWarp is not available on this machine.
- You must build the garment assembly pipeline yourself:
  panel triangulation, seam stitching, initial placement.
  Follow FORMA_GEOMETER_SPEC.md Sub-Problem 2 exactly.
```

**Add to Agent B (Patterns + Verdicts) prompt:**
```
IMPORTANT – AC-0 spike results:
- Do NOT use pygarment for pattern loading. It cannot be imported (libcairo missing).
- Load GarmentCode JSON patterns directly with Python's json module.
- Clone GarmentCodeData: git clone https://github.com/maria-korosteleva/GarmentCodeData data/garmentcode
- Parse panel outlines, edge definitions, and stitch annotations from the JSON.
- Convert to seam_manifest.json format per FORMA_SEAM_MANIFEST_SCHEMA.md.
- Use Python 3.11 environment (not 3.14).
```

---

## What Does NOT Change

- All output schemas (fit_verdict.json, body_profile.json, seam_manifest.json) – unchanged
- Acceptance criteria values (size S red, size M green, size XL loose) – unchanged
- The Geometer clearance computation pipeline – unchanged (already specified for this scenario)
- The severity thresholds, fit boolean logic, verdict_id format – unchanged
- Week 2, 3, 4 specs – unchanged (they consume outputs, don't care how they're produced)
- The competitive positioning, revenue model, data flywheel – unchanged

The spike changed *how* we build the engine, not *what* the engine produces.

---

*FORMA Post-Spike Addendum v1.0 – companion to docs/spike_results.md*
