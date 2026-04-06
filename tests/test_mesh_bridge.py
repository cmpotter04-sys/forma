"""
tests/test_mesh_bridge.py

Unit tests for src/geometer/warp/mesh_bridge.py.

compute_particle_masses() is pure numpy — no Warp required.
All Warp-dependent functions are tested via pytest.importorskip("warp")
so they skip cleanly on CPU-only machines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from geometer.warp.mesh_bridge import compute_particle_masses


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_triangle():
    """One unit-right-triangle in the XY plane (area = 0.5 m²)."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, faces


@pytest.fixture
def two_triangles():
    """Two triangles sharing an edge — a unit square split diagonally.
    Total area = 1.0 m².
    Verts: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
    Tri 0: 0-1-2  Tri 1: 0-2-3
    """
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces


# ---------------------------------------------------------------------------
# compute_particle_masses — basic correctness
# ---------------------------------------------------------------------------

class TestComputeParticleMasses:

    def test_returns_array_same_length_as_verts(self, single_triangle):
        verts, faces = single_triangle
        masses = compute_particle_masses(verts, faces, density_kg_m2=1.0)
        assert len(masses) == len(verts)

    def test_total_mass_equals_area_times_density(self, single_triangle):
        """For density=1.0, total mass should equal total surface area."""
        verts, faces = single_triangle
        # Area of right triangle with legs 1m = 0.5 m²
        masses = compute_particle_masses(verts, faces, density_kg_m2=1.0)
        assert abs(masses.sum() - 0.5) < 1e-9

    def test_uniform_triangle_equal_vertex_masses(self, single_triangle):
        """All three vertices of a single triangle should get equal mass."""
        verts, faces = single_triangle
        masses = compute_particle_masses(verts, faces, density_kg_m2=1.0)
        assert np.allclose(masses[0], masses[1])
        assert np.allclose(masses[1], masses[2])

    def test_density_scales_mass_linearly(self, single_triangle):
        verts, faces = single_triangle
        m1 = compute_particle_masses(verts, faces, density_kg_m2=1.0)
        m2 = compute_particle_masses(verts, faces, density_kg_m2=3.5)
        assert np.allclose(m2, m1 * 3.5)

    def test_two_triangles_total_mass(self, two_triangles):
        """Unit square (area=1) with density=0.2 → total mass = 0.2 kg."""
        verts, faces = two_triangles
        masses = compute_particle_masses(verts, faces, density_kg_m2=0.2)
        assert abs(masses.sum() - 0.2) < 1e-9

    def test_shared_edge_vertices_heavier(self, two_triangles):
        """Vertices shared by both triangles (0 and 2) get more mass than
        corner-only vertices (1 and 3)."""
        verts, faces = two_triangles
        masses = compute_particle_masses(verts, faces, density_kg_m2=1.0)
        # Vert 0 and 2 each appear in 2 triangles; verts 1 and 3 in 1 each
        assert masses[0] > masses[1]
        assert masses[2] > masses[3]
        assert np.isclose(masses[0], masses[2])
        assert np.isclose(masses[1], masses[3])

    def test_no_zero_mass_vertices(self, two_triangles):
        """All vertex masses must be positive (min_mass floor enforced)."""
        verts, faces = two_triangles
        masses = compute_particle_masses(verts, faces, density_kg_m2=0.001)
        assert np.all(masses > 0)

    def test_empty_faces_fallback(self):
        """Empty face array triggers uniform fallback — should not raise."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int32)
        masses = compute_particle_masses(verts, faces, density_kg_m2=0.3)
        assert len(masses) == 3
        assert np.all(masses > 0)

    def test_realistic_garment_size(self):
        """A realistic tshirt panel (~0.6m²) with cotton density 0.22 kg/m²
        should give total mass ~0.13 kg."""
        rng = np.random.default_rng(0)
        N = 500
        verts = np.zeros((N, 3))
        verts[:, 0] = rng.uniform(0, 0.6, N)  # 60cm wide
        verts[:, 1] = rng.uniform(0, 1.0, N)  # 100cm tall
        # Simple triangulation: use scipy Delaunay
        from scipy.spatial import Delaunay
        tri = Delaunay(verts[:, :2])
        faces = tri.simplices.astype(np.int32)

        masses = compute_particle_masses(verts, faces, density_kg_m2=0.22)
        total_mass = masses.sum()
        # Area ≈ 0.6 m² (convex hull of random points in 0.6×1.0 rectangle)
        assert 0.05 < total_mass < 0.20, f"Unexpected total mass: {total_mass:.4f}"


# ---------------------------------------------------------------------------
# Warp-dependent functions — skip on CPU-only machines
# ---------------------------------------------------------------------------

warp = pytest.importorskip("warp", reason="warp-lang not installed")


class TestWarpConversions:

    def test_numpy_to_warp_verts_shape(self):
        from geometer.warp.mesh_bridge import numpy_to_warp_verts
        verts = np.random.rand(100, 3).astype(np.float32)
        wa = numpy_to_warp_verts(verts, device="cpu")
        assert wa.shape == (100,)

    def test_numpy_to_warp_indices_shape(self):
        from geometer.warp.mesh_bridge import numpy_to_warp_indices
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        wi = numpy_to_warp_indices(faces, device="cpu")
        assert wi.shape == (6,)

    def test_roundtrip_verts(self):
        from geometer.warp.mesh_bridge import numpy_to_warp_verts, warp_to_numpy_verts
        verts_np = np.random.rand(50, 3).astype(np.float32)
        wa = numpy_to_warp_verts(verts_np, device="cpu")
        back = warp_to_numpy_verts(wa)
        assert back.shape == (50, 3)
        assert np.allclose(back.astype(np.float32), verts_np, atol=1e-5)

    def test_roundtrip_indices(self):
        from geometer.warp.mesh_bridge import numpy_to_warp_indices
        faces = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)
        wi = numpy_to_warp_indices(faces, device="cpu")
        flat = wi.numpy()
        assert np.array_equal(flat, faces.flatten())
