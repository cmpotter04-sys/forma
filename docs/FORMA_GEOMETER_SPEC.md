# FORMA — Geometer Implementation Spec
**Version:** 1.0  
**Phase:** 1 / Week 1  
**Owner:** Founder  
**Executor:** Claude Code CLI  
**Depends on:** FORMA_WEEK1_SPEC.md (AC-3, AC-4)  
**Status:** READY TO BUILD — implement alongside Week 1 AC-3

---

## Why This Document Exists

The Week 1 spec defines *what* the Geometer must output (strain_map, ease_map,
fit_verdict.json) but not *how* the 3D geometry processing works. Without this
spec, the executor will either hallucinate the implementation or hardcode values.

This document specifies the four geometry sub-problems that sit between
"load the body mesh and garment panels" and "write fit_verdict.json":

1. Measurement-to-Beta Conversion (body mesh generation)
2. Garment Assembly & Initial Placement
3. Body Region Segmentation
4. Clearance Computation (what the schema calls `delta_mm`)

---

## Sub-Problem 1: Measurement-to-Beta Conversion

### The Problem
SMPL-X body meshes are controlled by **shape coefficients** (betas) — a vector
of 10 floats in PCA space. The Week 1 spec asks for a body with specific tape
measurements (chest=96cm, waist=80cm, etc.). There is no closed-form mapping
from centimeters to betas.

### The Solution: Iterative Optimization

```
INPUT:  target_measurements = {chest_cm: 96, waist_cm: 80, hip_cm: 95, height_cm: 180, ...}
OUTPUT: betas[10] such that SMPL-X(betas).measurements ≈ target_measurements
```

**Algorithm:**
1. Start with betas = zeros(10) (average body)
2. Forward pass: generate SMPL-X mesh from current betas
3. Measure: extract chest/waist/hip/height from mesh vertices
4. Compute loss: L2 distance between measured and target
5. Backprop through SMPL-X model (differentiable) or use scipy.minimize
6. Repeat until all measurements within ±2mm of target
7. Save final betas to body_profile.json for reproducibility

**Measurement Extraction from Mesh:**
SMPL-X provides vertex indices for standard landmarks. To measure chest circumference:
- Identify the chest cross-section plane (horizontal plane at nipple-height vertices)
- Extract the convex hull of vertices within ±5mm of that plane
- Compute perimeter of the hull = chest circumference

Repeat analogously for waist (narrowest torso cross-section) and hip (widest
cross-section below waist).

### Implementation

```python
# src/sculptor/beta_solver.py

import numpy as np
from scipy.optimize import minimize

class BetaSolver:
    """
    Solves for SMPL-X beta parameters that produce a body mesh
    matching target tape measurements.
    """

    # SMPL-X vertex indices for measurement landmarks
    # These come from the SMPL-X body model registration
    CHEST_PLANE_VERTEX = 3076   # approximate sternum midpoint
    WAIST_PLANE_VERTEX = 3500   # approximate navel height
    HIP_PLANE_VERTEX = 6540     # approximate widest hip point

    def __init__(self, smplx_model):
        self.model = smplx_model

    def solve(self, target: dict, max_iter: int = 200, tol_mm: float = 2.0) -> np.ndarray:
        """
        Find betas that produce a mesh matching target measurements.

        Args:
            target: dict with keys height_cm, chest_cm, waist_cm, hip_cm
            max_iter: maximum optimization iterations
            tol_mm: convergence tolerance in millimeters

        Returns:
            betas: np.ndarray of shape (10,)

        Raises:
            BetaSolverError: if optimization fails to converge within tol_mm
        """
        def objective(betas):
            mesh = self.model.forward(betas=betas)
            measured = self._extract_measurements(mesh.vertices)
            residuals = []
            for key in target:
                residuals.append((measured[key] - target[key]) * 10)  # cm -> mm
            return np.sum(np.array(residuals) ** 2)

        x0 = np.zeros(10)
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter})

        # Verify convergence
        final_mesh = self.model.forward(betas=result.x)
        final_measurements = self._extract_measurements(final_mesh.vertices)
        for key in target:
            error_mm = abs(final_measurements[key] - target[key]) * 10
            if error_mm > tol_mm:
                raise BetaSolverError(
                    f"Failed to converge: {key} off by {error_mm:.1f}mm (tol={tol_mm}mm)"
                )

        return result.x

    def _extract_measurements(self, vertices: np.ndarray) -> dict:
        """Extract tape measurements from mesh vertices."""
        # Height: max_y - min_y of all vertices
        height_cm = (vertices[:, 1].max() - vertices[:, 1].min()) * 100

        # Circumferences via cross-section perimeter
        chest_cm = self._circumference_at_vertex(vertices, self.CHEST_PLANE_VERTEX)
        waist_cm = self._circumference_at_vertex(vertices, self.WAIST_PLANE_VERTEX)
        hip_cm = self._circumference_at_vertex(vertices, self.HIP_PLANE_VERTEX)

        return {
            "height_cm": height_cm,
            "chest_cm": chest_cm,
            "waist_cm": waist_cm,
            "hip_cm": hip_cm,
        }

    def _circumference_at_vertex(self, vertices, ref_vertex_idx, band_mm=5.0):
        """
        Compute circumference of body cross-section at the height of a
        reference vertex. Takes all vertices within ±band_mm of that height,
        projects to 2D (x, z), computes convex hull perimeter.
        """
        ref_height = vertices[ref_vertex_idx, 1]
        band_m = band_mm / 1000.0
        mask = np.abs(vertices[:, 1] - ref_height) < band_m
        cross_section = vertices[mask][:, [0, 2]]  # project to x-z plane

        from scipy.spatial import ConvexHull
        hull = ConvexHull(cross_section)
        perimeter = 0.0
        for simplex in hull.simplices:
            p0, p1 = cross_section[simplex[0]], cross_section[simplex[1]]
            perimeter += np.linalg.norm(p1 - p0)

        return perimeter * 100  # meters -> cm


class BetaSolverError(Exception):
    pass
```

### pygarment Compatibility Check

**CRITICAL — Do this before writing any other code:**

```python
# scripts/spike_pygarment.py — run this on Day 1 before anything else
#
# Purpose: verify pygarment can generate SMPL-X meshes with controlled shape params
# If this fails, we need the BetaSolver above. If pygarment handles it internally,
# we can skip the solver and use pygarment's API directly.

try:
    import pygarment
    print(f"pygarment version: {pygarment.__version__}")

    # Attempt 1: Check if pygarment exposes SMPL-X generation
    # Look for body generation functions
    import inspect
    body_modules = [m for m in dir(pygarment) if 'body' in m.lower() or 'smpl' in m.lower()]
    print(f"Body-related modules: {body_modules}")

    # Attempt 2: Check if GarmentCode simulation includes body generation
    sim_modules = [m for m in dir(pygarment) if 'sim' in m.lower()]
    print(f"Simulation modules: {sim_modules}")

    # Attempt 3: Check pygarment.body or pygarment.bodies namespace
    for ns in ['body', 'bodies', 'meshgen', 'smplx']:
        if hasattr(pygarment, ns):
            print(f"Found pygarment.{ns}: {dir(getattr(pygarment, ns))}")

except ImportError:
    print("pygarment not installed. Run: pip install pygarment")
except Exception as e:
    print(f"Error: {e}")

print("\n--- If pygarment cannot generate SMPL-X meshes directly ---")
print("Fallback: pip install smplx torch scipy")
print("Use BetaSolver from src/sculptor/beta_solver.py")
print("Download SMPL-X model from https://smpl-x.is.tue.mpg.de/")
```

### Output Contract

```json
{
  "body_profile_id": "mannequin_sizeM_180cm",
  "betas": [0.23, -0.41, 0.87, ...],
  "target_measurements": {
    "height_cm": 180, "chest_cm": 96, "waist_cm": 80, "hip_cm": 95
  },
  "achieved_measurements": {
    "height_cm": 180.1, "chest_cm": 95.9, "waist_cm": 80.2, "hip_cm": 95.1
  },
  "max_error_mm": 2.0,
  "mesh_path": "data/bodies/mannequin_sizeM_180cm.ply"
}
```

---

## Sub-Problem 2: Garment Assembly & Initial Placement

### The Problem
GarmentCode patterns are 2D panel geometries with seam edge correspondences.
Before XPBD simulation, these flat panels must be:
1. Triangulated into simulation-ready meshes
2. Stitched together along seam edges (creating a 3D tube/shell)
3. Positioned around the body at a physically reasonable starting location

### Expected Behavior

If using NvidiaWarp-GarmentCode's full pipeline, steps 1–3 are handled
internally. **Verify this by running GarmentCode's own example simulation
first before writing any custom code.**

If using Warp as a raw XPBD solver only, implement:

**Panel Triangulation:**
```
INPUT:  panel_outline (2D polygon vertices + curves)
OUTPUT: triangle_mesh (vertices, faces) with target edge length ~5mm
TOOL:   triangle or meshpy library
```

**Seam Stitching:**
Seam pairs from seam_manifest.json define which edges on different panels
must be joined. During simulation, these are enforced as distance constraints:
```
For each seam pair (edge_A on panel_1, edge_B on panel_2):
    Sample N points along each edge (matched by arc length)
    Add XPBD distance constraint: point_A_i ↔ point_B_i, rest_length = 0
```

**Initial Placement:**
Position the assembled garment around the body with ~20mm clearance:
1. Compute the garment's bounding box center
2. Align garment center with body torso center (average of chest vertices)
3. Scale garment outward by 20mm along surface normals
4. This gives the simulator a "reasonable start" — gravity + constraints do the rest

### What to Check First

```python
# scripts/spike_garmentcode_sim.py — verify GarmentCode handles assembly

# If GarmentCode's simulation pipeline handles panel assembly and placement
# internally, we do NOT need to build steps 1-3 ourselves.
# Run one of GarmentCode's example simulations to check.

import pygarment
# ... attempt to run a built-in example simulation
# If it works end-to-end (pattern → draped garment on body), we're good.
# If it only gives us panels and we need to assemble ourselves, flag this.
```

---

## Sub-Problem 3: Body Region Segmentation

### The Problem
The fit_verdict requires per-region measurements (chest_front, chest_side,
shoulder_left, shoulder_right, upper_back, waist). We need to know which
vertices on the body mesh belong to which region, and which vertices on the
draped garment correspond to which body region.

### SMPL-X Vertex Segmentation

SMPL-X provides a part segmentation map: each of its ~10,475 vertices is
assigned to one of 14 body parts. We map these to Forma's 6 regions:

```python
# src/geometer/region_map.py

# SMPL-X part labels → Forma region mapping
SMPLX_TO_FORMA = {
    # SMPL-X part index : Forma region name
    # Torso front (below neck, above waist, front-facing)
    "spine1_front":     "chest_front",
    "spine2_front":     "chest_front",

    # Torso sides
    "spine1_left":      "chest_side",
    "spine1_right":     "chest_side",
    "spine2_left":      "chest_side",
    "spine2_right":     "chest_side",

    # Shoulders
    "left_shoulder":    "shoulder_left",
    "right_shoulder":   "shoulder_right",
    "left_collar":      "shoulder_left",
    "right_collar":     "shoulder_right",

    # Upper back
    "spine1_back":      "upper_back",
    "spine2_back":      "upper_back",

    # Waist
    "pelvis_front":     "waist",
    "pelvis_left":      "waist",
    "pelvis_right":     "waist",
    "pelvis_back":      "waist",
}
```

**NOTE:** SMPL-X doesn't natively split front/back/side. We determine
front vs back vs side using **vertex normals**:
- If vertex normal dot product with (0, 0, 1) > 0.5 → front-facing
- If vertex normal dot product with (0, 0, 1) < −0.5 → back-facing
- Otherwise → side-facing

```python
def classify_body_vertices(vertices, normals, smplx_labels):
    """
    Assign each body vertex to a Forma region.

    Returns:
        dict mapping region_name → list of vertex indices
    """
    forward = np.array([0, 0, 1])  # assumes body facing +Z
    regions = {r: [] for r in [
        "chest_front", "chest_side", "shoulder_left",
        "shoulder_right", "upper_back", "waist"
    ]}

    for vid, label in enumerate(smplx_labels):
        facing = np.dot(normals[vid], forward)

        if label in TORSO_LABELS:
            if facing > 0.5:
                regions["chest_front"].append(vid)
            elif facing < -0.5:
                regions["upper_back"].append(vid)
            else:
                regions["chest_side"].append(vid)

        elif label in SHOULDER_LEFT_LABELS:
            regions["shoulder_left"].append(vid)
        elif label in SHOULDER_RIGHT_LABELS:
            regions["shoulder_right"].append(vid)
        elif label in WAIST_LABELS:
            regions["waist"].append(vid)

    return regions
```

### Garment-to-Body Region Assignment

For each garment vertex in the draped state, assign it to the body region
of its **nearest body surface point**:

```python
def assign_garment_to_body_regions(garment_vertices, body_vertices, body_regions, body_faces):
    """
    For each garment vertex, find nearest body surface point,
    look up which Forma region that body point belongs to.
    """
    from scipy.spatial import KDTree

    tree = KDTree(body_vertices)
    _, nearest_body_idx = tree.query(garment_vertices)

    # Build reverse map: body vertex index → region
    body_vertex_to_region = {}
    for region, vids in body_regions.items():
        for vid in vids:
            body_vertex_to_region[vid] = region

    garment_regions = {}
    for gv_idx, bv_idx in enumerate(nearest_body_idx):
        region = body_vertex_to_region.get(bv_idx, None)
        if region:
            garment_regions.setdefault(region, []).append(gv_idx)

    return garment_regions
```

---

## Sub-Problem 4: Clearance Computation (delta_mm)

### Terminology Clarification

What the v1.2 schema calls `strain_map.delta_mm` is actually **signed clearance**
between the garment surface and the body surface, not mechanical fabric strain.

```
delta_mm > 0  →  garment is FURTHER from body than rest (ease / loose)
delta_mm < 0  →  garment is CLOSER to body than rest, or compressed (tight)
delta_mm ≈ 0  →  garment sits at natural rest distance from body
```

True fabric strain (triangle deformation ratio) is a separate quantity that
could be added later for detecting where a garment would literally tear.

### Algorithm

For each body region:

```python
def compute_region_clearance(
    garment_vertices,      # draped garment vertex positions
    body_vertices,         # body mesh vertex positions
    body_faces,            # body mesh triangle faces
    garment_vertex_ids,    # which garment vertices belong to this region
    body_vertex_ids,       # which body vertices belong to this region
    body_normals,          # per-vertex body surface normals
):
    """
    Compute signed clearance for a single body region.

    Returns:
        delta_mm: float — median signed distance, negative = tight
    """
    from scipy.spatial import KDTree

    # For each garment vertex in this region, compute signed distance to body
    signed_distances = []

    body_region_verts = body_vertices[body_vertex_ids]
    body_region_normals = body_normals[body_vertex_ids]
    tree = KDTree(body_region_verts)

    for gv_idx in garment_vertex_ids:
        gv = garment_vertices[gv_idx]

        # Find nearest body point
        dist, nearest_idx = tree.query(gv)
        nearest_body_point = body_region_verts[nearest_idx]
        nearest_normal = body_region_normals[nearest_idx]

        # Signed distance: positive if garment is outside body surface
        # (in direction of outward normal), negative if inside/compressed
        displacement = gv - nearest_body_point
        signed_dist = np.dot(displacement, nearest_normal)
        signed_distances.append(signed_dist)

    signed_distances = np.array(signed_distances)

    # Use median to reject outlier vertices (e.g. at panel edges)
    delta_m = np.median(signed_distances)
    delta_mm = delta_m * 1000.0  # meters → millimeters

    return delta_mm
```

### Severity Classification

```python
def classify_severity(delta_mm: float) -> str:
    """
    Classify signed clearance into severity level.
    Negative delta = tight (body larger than garment).
    """
    if delta_mm > -10:
        return "green"
    elif delta_mm >= -25:
        return "yellow"
    else:
        return "red"
```

### Ease Map Computation

The ease_map uses the same clearance values but classifies differently:

```python
def classify_ease(delta_mm: float) -> tuple:
    """
    Returns (excess_mm, verdict) for the ease_map.
    excess_mm = max(0, delta_mm) — only counts positive clearance as "excess"
    """
    excess_mm = max(0.0, delta_mm)

    if delta_mm < 0:
        verdict = "tight_fit"
    elif delta_mm <= 20:
        verdict = "standard_fit"
    elif delta_mm <= 50:
        verdict = "relaxed_fit"
    else:
        verdict = "oversized"

    return excess_mm, verdict
```

---

## Convergence Criterion — Improved

### Replace Vertex-Movement Check with Energy-Based Convergence

The Week 1 spec defines convergence as "fewer than 1.5% of vertices moving
more than 0.4mm per frame." This is fragile — wrinkle shuffling can prevent
convergence even when the global fit has stabilized.

**Use total kinetic energy instead:**

```python
def check_convergence(
    velocities: np.ndarray,    # per-vertex velocities, shape (N, 3)
    masses: np.ndarray,        # per-vertex masses, shape (N,)
    threshold_j: float = 1e-6, # energy threshold in joules
    prev_energy: float = None,
) -> tuple:
    """
    Energy-based convergence check.

    Returns:
        (converged: bool, current_energy: float)
    """
    # Kinetic energy = 0.5 * m * v^2 for each vertex
    speed_sq = np.sum(velocities ** 2, axis=1)
    kinetic_energy = 0.5 * np.sum(masses * speed_sq)

    # Converged if energy is below absolute threshold
    converged = kinetic_energy < threshold_j

    # Also check energy is not increasing (explosion detection)
    if prev_energy is not None and kinetic_energy > prev_energy * 10:
        raise SimulationExplosionError(
            f"Energy spike detected: {prev_energy:.6f} → {kinetic_energy:.6f}"
        )

    return converged, kinetic_energy


class SimulationExplosionError(Exception):
    pass
```

**Keep the vertex-movement check as a secondary safety net**, not as the
primary convergence criterion:

```python
def check_vertex_movement(positions, prev_positions, threshold_mm=0.4, max_pct=1.5):
    """Secondary convergence check — original spec criterion."""
    movement = np.linalg.norm(positions - prev_positions, axis=1) * 1000  # m → mm
    pct_moving = np.sum(movement > threshold_mm) / len(movement) * 100
    return pct_moving < max_pct
```

---

## Tunnel-Through Detection

```python
def detect_tunnel_through(garment_vertices, body_vertices, body_faces, body_normals):
    """
    Detect if any garment vertices have passed through the body mesh.

    A garment vertex is "tunneled" if:
    1. Its nearest body point is very close (< 2mm)
    2. AND the displacement is in the OPPOSITE direction of the body normal
       (meaning it's inside the body, not outside)

    Returns:
        tunnel_count: int — number of garment vertices inside body
        tunnel_pct: float — percentage of garment vertices tunneled
    """
    from scipy.spatial import KDTree
    tree = KDTree(body_vertices)

    dists, nearest_idx = tree.query(garment_vertices)

    tunnel_count = 0
    for i in range(len(garment_vertices)):
        if dists[i] < 0.002:  # within 2mm of body
            displacement = garment_vertices[i] - body_vertices[nearest_idx[i]]
            normal = body_normals[nearest_idx[i]]
            if np.dot(displacement, normal) < 0:  # inside body
                tunnel_count += 1

    tunnel_pct = tunnel_count / len(garment_vertices) * 100
    return tunnel_count, tunnel_pct
```

---

## Full Geometer Pipeline — Orchestration

```python
# src/geometer/xpbd_simulate.py — high-level flow

def run_simulation(body_mesh_path, pattern_path, seam_manifest_path, fabric_params):
    """
    Full Geometer pipeline: body + pattern → fit_verdict fields.

    Returns:
        dict with keys: strain_map, ease_map, simulation_ms, converged
    """
    import time

    # 1. Load body mesh + compute region segmentation
    body = load_body_mesh(body_mesh_path)
    body_regions = classify_body_vertices(body.vertices, body.normals, body.smplx_labels)

    # 2. Load pattern + seam manifest
    pattern = load_pattern(pattern_path)
    seams = load_seam_manifest(seam_manifest_path)

    # 3. Assemble garment (triangulate panels, position around body)
    #    NOTE: If using GarmentCode's full pipeline, this is handled internally.
    #    If using raw Warp, call assemble_garment() here.
    garment = assemble_garment(pattern, seams, body)

    # 4. Run XPBD simulation
    t0 = time.perf_counter()
    draped_garment = xpbd_drape(
        garment_mesh=garment,
        body_mesh=body,
        fabric_params=fabric_params,
        max_timesteps=200,
    )
    simulation_ms = int((time.perf_counter() - t0) * 1000)

    # 5. Assign draped garment vertices to body regions
    garment_regions = assign_garment_to_body_regions(
        draped_garment.vertices, body.vertices, body_regions, body.faces
    )

    # 6. Compute clearance per region
    strain_map = []
    ease_map = []

    for region_name in ["chest_front", "chest_side", "shoulder_left",
                         "shoulder_right", "upper_back", "waist"]:
        delta_mm = compute_region_clearance(
            draped_garment.vertices, body.vertices, body.faces,
            garment_regions.get(region_name, []),
            body_regions.get(region_name, []),
            body.normals,
        )
        severity = classify_severity(delta_mm)
        strain_map.append({
            "region": region_name,
            "delta_mm": round(delta_mm, 1),
            "severity": severity,
        })

        excess_mm, verdict = classify_ease(delta_mm)
        ease_map.append({
            "region": region_name,
            "excess_mm": round(excess_mm, 1),
            "verdict": verdict,
        })

    # 7. Detect tunnel-through
    tunnel_count, tunnel_pct = detect_tunnel_through(
        draped_garment.vertices, body.vertices, body.faces, body.normals
    )
    if tunnel_pct > 2.0:
        raise SimulationError(
            f"Tunnel-through detected: {tunnel_count} vertices ({tunnel_pct:.1f}%) inside body"
        )

    return {
        "strain_map": strain_map,
        "ease_map": ease_map,
        "simulation_ms": simulation_ms,
        "converged": draped_garment.converged,
        "tunnel_pct": tunnel_pct,
    }
```

---

## Test Additions for Geometer

```python
# tests/test_geometer.py

class TestBetaSolver:
    """Verify measurement-to-beta conversion produces correct body dimensions."""

    def test_chest_within_tolerance(self):
        """Generated mesh chest circumference must be within ±2mm of target."""
        # This test runs the actual solver — slow but essential
        pass  # Implement after pygarment spike confirms approach

    def test_betas_are_reproducible(self):
        """Same target measurements must produce identical betas."""
        pass


class TestRegionSegmentation:
    """Verify body region classification is physically correct."""

    def test_all_six_regions_have_vertices(self):
        """Every required region must contain at least 50 vertices."""
        pass

    def test_chest_front_faces_forward(self):
        """All chest_front vertices must have normals facing +Z."""
        pass

    def test_no_region_overlap(self):
        """No vertex should belong to more than one region."""
        pass


class TestClearanceComputation:
    """Verify signed clearance math is correct."""

    def test_garment_outside_body_gives_positive(self):
        """Garment vertices outside body surface → positive delta_mm."""
        pass

    def test_garment_compressed_gives_negative(self):
        """Garment vertices compressed against body → negative delta_mm."""
        pass

    def test_severity_thresholds(self):
        assert classify_severity(5.0) == "green"
        assert classify_severity(-5.0) == "green"
        assert classify_severity(-15.0) == "yellow"
        assert classify_severity(-30.0) == "red"


class TestConvergence:
    """Verify convergence detection works correctly."""

    def test_zero_velocity_converges(self):
        vels = np.zeros((100, 3))
        masses = np.ones(100)
        converged, _ = check_convergence(vels, masses)
        assert converged

    def test_high_velocity_does_not_converge(self):
        vels = np.random.randn(100, 3)
        masses = np.ones(100)
        converged, _ = check_convergence(vels, masses)
        assert not converged

    def test_energy_spike_raises_explosion(self):
        vels = np.ones((100, 3)) * 100  # extreme velocity
        masses = np.ones(100)
        with pytest.raises(SimulationExplosionError):
            check_convergence(vels, masses, prev_energy=0.001)
```

---

## Dependencies Added

```
# Add to requirements.txt
scipy>=1.11        # BetaSolver optimization + KDTree
trimesh>=4.0       # mesh I/O and operations (lighter than pytorch3d)
smplx>=0.1.28      # SMPL-X body model (if pygarment doesn't handle beta solving)
```

**NOTE:** `trimesh` replaces `pytorch3d` for Week 1. pytorch3d is heavy,
hard to install, and unnecessary for distance queries. trimesh + scipy
handle everything the Geometer needs. Defer pytorch3d to Phase 2 mesh fusion.

---

*FORMA Geometer Implementation Spec v1.0*  
*Companion to FORMA_WEEK1_SPEC.md — covers the geometry pipeline gap*
