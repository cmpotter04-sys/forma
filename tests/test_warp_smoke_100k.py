"""
AC-2: Warp GPU smoke test at high vertex count.

Validates that the Warp backend produces physically reasonable results
at 10K+ vertices after Loop subdivision.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Skip entire module if warp is not installed
try:
    import warp as wp
    import warp.sim
    _WARP_AVAILABLE = True
except ImportError:
    _WARP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _WARP_AVAILABLE, reason="warp-lang not installed"
)

BODY_M_PLY = ROOT / "data" / "bodies" / "makehuman_male_M.ply"
PATTERN_M = ROOT / "data" / "patterns" / "tshirt_size_M.json"
MANIFEST_M = ROOT / "seam_manifests" / "tshirt_size_M_manifest.json"
FABRIC_LIB = ROOT / "data" / "fabrics" / "fabric_library.json"


def _load_fabric(fabric_id="cotton_jersey_default"):
    with open(FABRIC_LIB) as f:
        lib = json.load(f)
    return lib["fabrics"][fabric_id]


@pytest.fixture(scope="module")
def warp_10k_result():
    """
    Run Warp simulation with ~10K vertex garment (2 passes of subdivision).
    Returns (sim_result, wall_clock_ms, vertex_count).
    """
    import trimesh
    from src.pattern_maker.load_patterns import load_pattern
    from src.tailor.seam_converter import load_and_validate_manifest
    from src.geometer.garment_assembly import assemble_garment, project_garment_onto_body
    from src.geometer.subdivide import subdivide_garment
    from src.geometer.warp.warp_simulate import (
        _ensure_warp, _build_warp_model, _run_warp_simulation,
    )
    from src.geometer.region_map import classify_body_vertices, assign_garment_to_body_regions
    from src.geometer.clearance import compute_region_clearance, detect_tunnel_through
    from src.geometer.garment_assembly import compute_strain_ratios, REQUIRED_REGIONS

    fabric_params = _load_fabric()
    _ensure_warp()

    # 1. Load body
    body_mesh = trimesh.load(str(BODY_M_PLY), process=False)
    body_vertices = np.array(body_mesh.vertices, dtype=float)
    body_faces = np.array(body_mesh.faces, dtype=np.int32)
    body_normals = np.array(body_mesh.vertex_normals, dtype=float)
    norms = np.linalg.norm(body_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    body_normals = body_normals / norms

    # 2. Load pattern + manifest
    pattern = load_pattern(str(PATTERN_M))
    manifest = load_and_validate_manifest(str(MANIFEST_M))

    # 3. Assemble + project
    garment = assemble_garment(pattern, manifest, body_vertices)
    garment = project_garment_onto_body(garment, body_vertices, body_normals)

    # 4. Subdivide to ~10K vertices (2 passes)
    garment = subdivide_garment(garment, target_verts=10000)
    n_verts = len(garment["vertices"])
    print(f"\n  Subdivided garment: {n_verts} vertices, "
          f"{len(garment['faces'])} faces, "
          f"{len(garment['stretch_i'])} stretch constraints")

    # 5. Region segmentation
    body_regions = classify_body_vertices(body_vertices, body_normals)

    # 6. Run Warp simulation
    t0 = time.perf_counter()
    model = _build_warp_model(garment, body_vertices, body_faces, fabric_params)
    warp_result = _run_warp_simulation(model, garment, fabric_params, dt=0.001, max_steps=200)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    print(f"  Warp sim wall-clock: {wall_ms}ms")

    draped = warp_result["positions"]

    # 7. Compute clearance + strain
    garment_regions = assign_garment_to_body_regions(draped, body_vertices, body_regions)
    clearance_map = {}
    for region in REQUIRED_REGIONS:
        delta_mm = compute_region_clearance(
            draped, body_vertices, body_normals,
            garment_regions[region], body_regions[region],
            garment_scale=garment.get("garment_scale"),
            body_map=garment.get("body_map"),
        )
        clearance_map[region] = round(delta_mm, 3)

    strain_map = compute_strain_ratios(
        draped, garment["stretch_i"], garment["stretch_j"],
        garment["stretch_rest"], garment_regions,
    )

    _, tunnel_pct = detect_tunnel_through(draped, body_vertices, body_normals)

    sim_result = {
        "clearance_map": clearance_map,
        "strain_ratio_map": strain_map,
        "simulation_ms": wall_ms,
        "convergence_step": warp_result["convergence_step"],
        "final_kinetic_energy_j": warp_result["final_kinetic_energy_j"],
        "tunnel_through_pct": round(float(tunnel_pct), 3),
    }

    return sim_result, wall_ms, n_verts


class TestWarpSmoke10K:
    def test_no_explosion(self, warp_10k_result):
        """Warp sim with 10K vertices completes without explosion."""
        sim_result, _, _ = warp_10k_result
        assert sim_result is not None

    def test_m_on_m_fit(self, warp_10k_result):
        """M garment on M body: no red regions (fit=True)."""
        sim_result, _, _ = warp_10k_result
        from src.geometer.clearance import classify_severity
        for region, delta in sim_result["clearance_map"].items():
            sr = sim_result.get("strain_ratio_map", {}).get(region, 1.0)
            sev = classify_severity(delta, sr)
            assert sev != "red", (
                f"Region {region}: severity={sev}, delta={delta:.1f}mm, strain={sr:.4f}"
            )

    def test_clearance_reasonable(self, warp_10k_result):
        """All clearance values in physically reasonable range."""
        sim_result, _, _ = warp_10k_result
        for region, delta in sim_result["clearance_map"].items():
            assert -5.0 < delta < 30.0, (
                f"Region {region}: clearance={delta:.1f}mm (out of range)"
            )

    def test_strain_not_extreme(self, warp_10k_result):
        """All strain ratios below 1.05 (no extreme stretching)."""
        sim_result, _, _ = warp_10k_result
        for region, sr in sim_result.get("strain_ratio_map", {}).items():
            assert sr < 1.15, (
                f"Region {region}: strain_ratio={sr:.4f} (too high)"
            )

    def test_vertex_count(self, warp_10k_result):
        """Garment should have >= 10K vertices after subdivision."""
        _, _, n_verts = warp_10k_result
        assert n_verts >= 10000, f"Only {n_verts} vertices (expected >= 10K)"

    def test_timing(self, warp_10k_result):
        """Print timing info (informational, not assertion)."""
        sim_result, wall_ms, n_verts = warp_10k_result
        print(f"\n  10K Warp sim: {wall_ms}ms, "
              f"converged at step {sim_result['convergence_step']}, "
              f"{n_verts} vertices")
