"""
tests/test_geometer.py

Tests for AC-3 (XPBD simulation) and AC-4 (region segmentation, clearance).
Covers: region segmentation, clearance math, convergence detection,
        garment assembly, and end-to-end simulation output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BODY_PLY = ROOT / "data" / "bodies" / "makehuman_male_M.ply"
MANIFEST_M = ROOT / "seam_manifests" / "tshirt_size_M_manifest.json"
MANIFEST_S = ROOT / "seam_manifests" / "tshirt_size_S_manifest.json"
PATTERN_M = ROOT / "data" / "patterns" / "tshirt_size_M.json"
PATTERN_S = ROOT / "data" / "patterns" / "tshirt_size_S.json"
FABRIC_LIB = ROOT / "data" / "fabrics" / "fabric_library.json"


@pytest.fixture(scope="module")
def body_mesh():
    import trimesh
    mesh = trimesh.load(str(BODY_PLY), process=False)
    return mesh


@pytest.fixture(scope="module")
def body_verts_normals(body_mesh):
    verts = np.array(body_mesh.vertices, dtype=float)
    normals = np.array(body_mesh.vertex_normals, dtype=float)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normals = normals / norms
    return verts, normals


@pytest.fixture(scope="module")
def fabric_params():
    with open(FABRIC_LIB) as f:
        lib = json.load(f)
    return lib["fabrics"]["cotton_jersey_default"]


@pytest.fixture(scope="module")
def sim_result_M(fabric_params):
    """Run size-M simulation once; reuse across tests (slow fixture)."""
    from src.geometer.xpbd_simulate import run_simulation
    return run_simulation(
        BODY_PLY, PATTERN_M, MANIFEST_M, fabric_params
    )


# ---------------------------------------------------------------------------
# TestRegionSegmentation
# ---------------------------------------------------------------------------

class TestRegionSegmentation:
    """Body vertex region classification (AC-4 Sub-Problem 3)."""

    def test_all_six_regions_populated(self, body_verts_normals):
        from src.geometer.region_map import classify_body_vertices, REQUIRED_REGIONS
        verts, normals = body_verts_normals
        regions = classify_body_vertices(verts, normals)
        for region in REQUIRED_REGIONS:
            assert len(regions[region]) >= 50, \
                f"Region '{region}' has only {len(regions[region])} vertices"

    def test_no_region_overlap(self, body_verts_normals):
        from src.geometer.region_map import classify_body_vertices, REQUIRED_REGIONS
        verts, normals = body_verts_normals
        regions = classify_body_vertices(verts, normals)
        all_assigned: list[int] = []
        for vids in regions.values():
            all_assigned.extend(vids)
        assert len(all_assigned) == len(set(all_assigned)), \
            "Some vertices appear in more than one region"

    def test_chest_front_normals_face_forward(self, body_verts_normals):
        from src.geometer.region_map import classify_body_vertices
        verts, normals = body_verts_normals
        regions = classify_body_vertices(verts, normals)
        ids = regions["chest_front"]
        assert len(ids) > 0
        chest_normals = normals[ids]
        # At least 80% of chest_front normals have nz > 0
        fwd_pct = float(np.mean(chest_normals[:, 2] > 0)) * 100
        assert fwd_pct >= 80.0, \
            f"Only {fwd_pct:.1f}% of chest_front normals face +Z"

    def test_upper_back_normals_face_backward(self, body_verts_normals):
        from src.geometer.region_map import classify_body_vertices
        verts, normals = body_verts_normals
        regions = classify_body_vertices(verts, normals)
        ids = regions["upper_back"]
        assert len(ids) > 0
        back_normals = normals[ids]
        bwd_pct = float(np.mean(back_normals[:, 2] < 0)) * 100
        assert bwd_pct >= 80.0, \
            f"Only {bwd_pct:.1f}% of upper_back normals face -Z"

    def test_shoulder_left_on_positive_x_side(self, body_verts_normals):
        from src.geometer.region_map import classify_body_vertices
        verts, normals = body_verts_normals
        regions = classify_body_vertices(verts, normals)
        ids = regions["shoulder_left"]
        assert len(ids) > 0
        xs = verts[ids, 0]
        assert float(np.median(xs)) > 0.0, \
            "shoulder_left should be on the +X (wearer's left) side"

    def test_shoulder_right_on_negative_x_side(self, body_verts_normals):
        from src.geometer.region_map import classify_body_vertices
        verts, normals = body_verts_normals
        regions = classify_body_vertices(verts, normals)
        ids = regions["shoulder_right"]
        assert len(ids) > 0
        xs = verts[ids, 0]
        assert float(np.median(xs)) < 0.0, \
            "shoulder_right should be on the -X (wearer's right) side"

    def test_waist_at_correct_height(self, body_verts_normals):
        from src.geometer.region_map import classify_body_vertices
        verts, normals = body_verts_normals
        regions = classify_body_vertices(verts, normals)
        ids = regions["waist"]
        ys = verts[ids, 1]
        assert float(np.median(ys)) == pytest.approx(1.05, abs=0.12), \
            "Waist region median Y should be near 1.05 m"


# ---------------------------------------------------------------------------
# TestClearanceComputation
# ---------------------------------------------------------------------------

class TestClearanceComputation:
    """Signed clearance math (AC-4 Sub-Problem 4)."""

    def test_garment_outside_body_gives_positive(self):
        from src.geometer.clearance import compute_region_clearance
        body_verts = np.array([[0.0, 0.0, 0.0]])
        body_normals = np.array([[0.0, 0.0, 1.0]])  # points +Z
        # Garment vertex 50 mm outside (in +Z direction)
        garment_verts = np.array([[0.0, 0.0, 0.050]])
        delta_mm = compute_region_clearance(
            garment_verts, body_verts, body_normals, [0], [0]
        )
        assert delta_mm == pytest.approx(50.0, abs=1.0), \
            f"Expected ~50mm ease, got {delta_mm:.1f}mm"

    def test_garment_compressed_gives_negative(self):
        from src.geometer.clearance import compute_region_clearance
        body_verts = np.array([[0.0, 0.0, 0.0]])
        body_normals = np.array([[0.0, 0.0, 1.0]])
        # Garment vertex 15 mm inside body (negative signed distance)
        garment_verts = np.array([[0.0, 0.0, -0.015]])
        delta_mm = compute_region_clearance(
            garment_verts, body_verts, body_normals, [0], [0]
        )
        assert delta_mm == pytest.approx(-15.0, abs=1.0), \
            f"Expected -15mm tight, got {delta_mm:.1f}mm"

    def test_empty_region_returns_zero(self):
        from src.geometer.clearance import compute_region_clearance
        body_verts = np.array([[0.0, 0.0, 0.0]])
        body_normals = np.array([[0.0, 0.0, 1.0]])
        garment_verts = np.array([[0.0, 0.0, 0.05]])
        delta_mm = compute_region_clearance(
            garment_verts, body_verts, body_normals, [], [0]
        )
        assert delta_mm == 0.0

    def test_severity_thresholds(self):
        from src.geometer.clearance import classify_severity
        assert classify_severity(15.0)   == "green"
        assert classify_severity(-5.0)   == "green"
        assert classify_severity(-9.9)   == "green"   # just above -10 boundary
        assert classify_severity(-10.0)  == "yellow"  # exactly -10 → not > -10
        assert classify_severity(-10.1)  == "yellow"
        assert classify_severity(-15.0)  == "yellow"
        assert classify_severity(-24.9)  == "yellow"  # just above -25 boundary
        assert classify_severity(-25.0)  == "red"    # exactly -25 → not > -25
        assert classify_severity(-25.1)  == "red"
        assert classify_severity(-40.0)  == "red"

    def test_ease_classification(self):
        from src.geometer.clearance import classify_ease
        excess, verdict = classify_ease(-5.0)
        assert excess == 0.0 and verdict == "tight_fit"

        excess, verdict = classify_ease(10.0)
        assert excess == pytest.approx(10.0) and verdict == "standard_fit"

        excess, verdict = classify_ease(35.0)
        assert excess == pytest.approx(35.0) and verdict == "relaxed_fit"

        excess, verdict = classify_ease(60.0)
        assert excess == pytest.approx(60.0) and verdict == "oversized"

    def test_tunnel_through_detection(self):
        from src.geometer.clearance import detect_tunnel_through
        # Body at origin, normal pointing +Z
        body_verts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        body_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        # One vertex inside body (z=-0.001 < 0, close), one outside
        garment = np.array([
            [0.0, 0.0, -0.001],   # inside → tunnel
            [0.0, 0.0,  0.050],   # outside → ok
        ])
        count, pct = detect_tunnel_through(garment, body_verts, body_normals)
        assert count == 1
        assert pct == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# TestConvergence
# ---------------------------------------------------------------------------

class TestConvergence:
    """Energy-based convergence detection."""

    def test_zero_velocity_converges(self):
        from src.geometer.convergence import check_convergence
        vels = np.zeros((100, 3))
        masses = np.ones(100)
        converged, ke = check_convergence(vels, masses)
        assert converged
        assert ke == pytest.approx(0.0, abs=1e-15)

    def test_high_velocity_does_not_converge(self):
        from src.geometer.convergence import check_convergence
        rng = np.random.default_rng(42)
        vels = rng.standard_normal((100, 3))
        masses = np.ones(100)
        converged, ke = check_convergence(vels, masses)
        assert not converged
        assert ke > 1e-6

    def test_energy_spike_raises_explosion(self):
        from src.geometer.convergence import check_convergence, SimulationExplosionError
        vels = np.ones((100, 3)) * 50.0   # extreme velocity
        masses = np.ones(100)
        with pytest.raises(SimulationExplosionError):
            check_convergence(vels, masses, prev_energy=0.001)

    def test_custom_threshold(self):
        from src.geometer.convergence import check_convergence
        vels = np.ones((10, 3)) * 0.001   # small but non-zero
        masses = np.ones(10)
        ke_approx = 0.5 * 10 * 3 * (0.001 ** 2)   # ≈ 1.5e-5 J

        converged_tight, _ = check_convergence(vels, masses, threshold_j=1e-6)
        assert not converged_tight

        converged_loose, _ = check_convergence(vels, masses, threshold_j=1.0)
        assert converged_loose

    def test_vertex_movement_check(self):
        from src.geometer.convergence import check_vertex_movement
        pos = np.zeros((100, 3))
        prev = np.zeros((100, 3))

        # All stationary
        assert check_vertex_movement(pos, prev)

        # 5% moving more than 0.4mm
        prev_modified = prev.copy()
        prev_modified[:5, 0] = 0.001   # 1mm movement for 5% of vertices
        assert not check_vertex_movement(pos, prev_modified, max_pct=1.5)

        # Only 2% moving — within 1.5% limit? No, 2 > 1.5
        prev2 = prev.copy()
        prev2[:2, 0] = 0.001
        assert not check_vertex_movement(pos, prev2, max_pct=1.5)

        # 1% moving — within limit
        prev1 = prev.copy()
        prev1[:1, 0] = 0.001
        assert check_vertex_movement(pos, prev1, max_pct=1.5)


# ---------------------------------------------------------------------------
# TestGarmentAssembly
# ---------------------------------------------------------------------------

class TestGarmentAssembly:
    """Garment assembly: triangulation + 3D placement + seam constraints."""

    @pytest.fixture(scope="class")
    def garment(self):
        import trimesh as tm
        from src.geometer.xpbd_simulate import _assemble_garment
        from src.pattern_maker.load_patterns import load_pattern
        from src.tailor.seam_converter import load_and_validate_manifest

        pattern = load_pattern(PATTERN_M)
        manifest = load_and_validate_manifest(MANIFEST_M)
        body_mesh = tm.load(str(BODY_PLY), process=False)
        body_vertices = np.array(body_mesh.vertices, dtype=float)
        return _assemble_garment(pattern, manifest, body_vertices)

    def test_assembly_produces_vertices(self, garment):
        assert garment["vertices"].shape[1] == 3
        assert len(garment["vertices"]) > 100, "Expected >100 garment vertices"

    def test_assembly_produces_faces(self, garment):
        assert garment["faces"].shape[1] == 3
        assert len(garment["faces"]) > 50, "Expected >50 garment faces"
        n = len(garment["vertices"])
        assert garment["faces"].max() < n
        assert garment["faces"].min() >= 0

    def test_garment_panels_placed_near_body_height(self, garment):
        verts = garment["vertices"]
        y_med = float(np.median(verts[:, 1]))
        assert 0.7 <= y_med <= 1.5, \
            f"Garment median height {y_med:.2f} m is out of expected torso range"

    def test_seam_constraints_exist(self, garment):
        assert len(garment["seam_i"]) > 0, "Expected seam constraints"
        n = len(garment["vertices"])
        assert garment["seam_i"].max() < n
        assert garment["seam_j"].max() < n

    def test_stretch_constraints_exist(self, garment):
        assert len(garment["stretch_i"]) > 100, "Expected many stretch constraints"
        assert np.all(garment["stretch_rest"] > 0), "Rest lengths must be positive"


# ---------------------------------------------------------------------------
# TestSimulationOutput  (uses the module-scoped sim_result_M fixture)
# ---------------------------------------------------------------------------

class TestSimulationOutput:
    """End-to-end simulation output for size M on size M body."""

    def test_sim_result_has_required_keys(self, sim_result_M):
        required = {
            "clearance_map", "simulation_ms", "convergence_step",
            "final_kinetic_energy_j", "tunnel_through_pct",
        }
        assert required.issubset(sim_result_M.keys())

    def test_all_six_regions_in_clearance_map(self, sim_result_M):
        from src.geometer.xpbd_simulate import REQUIRED_REGIONS
        cm = sim_result_M["clearance_map"]
        for region in REQUIRED_REGIONS:
            assert region in cm, f"clearance_map missing region: {region}"

    def test_clearance_values_are_floats(self, sim_result_M):
        for region, val in sim_result_M["clearance_map"].items():
            assert isinstance(val, (int, float)), \
                f"clearance_map[{region!r}] is not a number: {val!r}"

    def test_size_M_on_M_body_clearance_is_positive(self, sim_result_M):
        """Size M garment on size M body should have positive ease (not too tight)."""
        cm = sim_result_M["clearance_map"]
        # At least one torso region should show positive clearance
        torso_regions = ["chest_front", "upper_back", "waist"]
        any_positive = any(cm[r] > 0.0 for r in torso_regions)
        assert any_positive, \
            f"Expected some positive clearance for size M on M body; got {cm}"

    def test_tunnel_through_below_limit(self, sim_result_M):
        assert sim_result_M["tunnel_through_pct"] < 2.0, \
            f"Tunnel-through {sim_result_M['tunnel_through_pct']:.2f}% exceeds 2%"

    def test_simulation_ms_is_positive_int(self, sim_result_M):
        assert isinstance(sim_result_M["simulation_ms"], int)
        assert sim_result_M["simulation_ms"] > 0

    def test_convergence_step_in_range(self, sim_result_M):
        step = sim_result_M["convergence_step"]
        assert 0 <= step <= 200, f"convergence_step={step} out of range [0, 200]"

    def test_final_kinetic_energy_non_negative(self, sim_result_M):
        assert sim_result_M["final_kinetic_energy_j"] >= 0.0

    def test_clearance_values_in_physical_range(self, sim_result_M):
        """Clearance values should be physically reasonable (-200mm to +200mm)."""
        cm = sim_result_M["clearance_map"]
        for region, val in cm.items():
            assert -200.0 <= val <= 200.0, \
                f"clearance_map[{region!r}] = {val:.1f}mm is outside physical range"
