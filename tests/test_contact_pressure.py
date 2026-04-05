from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Shared synthetic geometry helpers
# ---------------------------------------------------------------------------

def _make_cylinder_patch(n_verts: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (body_verts, body_normals, garment_verts) for a tiny cylinder patch.

    Body verts sit on a unit circle in XZ at Y=0.
    Body normals point radially outward (already unit length).
    Garment verts are placed 10 mm outside the body surface (slight ease).
    """
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    body_verts = np.column_stack([
        np.cos(angles),
        np.zeros(n_verts),
        np.sin(angles),
    ]).astype(np.float64)
    body_normals = body_verts.copy()  # already unit radial

    # Garment at radius 1.010 m → 10 mm outside
    garment_verts = body_verts * 1.010
    return body_verts, body_normals, garment_verts


def _make_contact_garment(body_verts: np.ndarray) -> np.ndarray:
    """Return garment verts that are 5 mm inside (penetrating) the body."""
    # Body is on unit circle; shrink to 0.995 m radius → signed dist ≈ -0.005 m
    return body_verts * 0.995


def _make_faces(n: int) -> np.ndarray:
    """Build a simple fan triangulation for n_verts arranged in a ring."""
    faces = []
    for i in range(n - 2):
        faces.append([0, i + 1, i + 2])
    return np.array(faces, dtype=np.int64)


def _region_labels_all(n: int) -> dict[str, list[int]]:
    return {
        "chest_front": list(range(0, n // 2)),
        "upper_back": list(range(n // 2, n)),
    }


def _minimal_fabric_params() -> dict:
    return {"density_kg_m2": 0.2, "total_area_m2": 0.5}


def _minimal_sim_result() -> dict:
    return {"soft_contact_ke": 1e4}


# ---------------------------------------------------------------------------
# TestCpuFallbackReturnsZeros
# ---------------------------------------------------------------------------

class TestCpuFallbackReturnsZeros:

    def test_cpu_fallback_returns_zeros(self):
        """When _HAS_WARP is patched to False, every region must return 0.0."""
        import geometer.warp.contact_pressure as cp_mod

        body_verts, body_normals, garment_verts = _make_cylinder_patch(16)
        region_labels = _region_labels_all(16)

        with mock.patch.object(cp_mod, "_HAS_WARP", False):
            result = cp_mod.compute_contact_pressure(
                sim_result=_minimal_sim_result(),
                body_vertices=body_verts,
                garment_vertices=garment_verts,
                region_labels=region_labels,
                fabric_params=_minimal_fabric_params(),
                body_normals=body_normals,
            )

        assert set(result.keys()) == set(region_labels.keys())
        for region, val in result.items():
            assert val == 0.0, f"Expected 0.0 for {region!r} in CPU fallback, got {val}"


# ---------------------------------------------------------------------------
# TestNoContactRegionReturnsZero
# ---------------------------------------------------------------------------

class TestNoContactRegionReturnsZero:

    def test_no_contact_region_returns_zero(self):
        """Garment vertices outside the body (positive clearance) → pressure 0.0."""
        import geometer.warp.contact_pressure as cp_mod

        if not cp_mod._HAS_WARP:
            pytest.skip("warp not installed — positive-clearance path still exercised via fallback")

        body_verts, body_normals, garment_verts = _make_cylinder_patch(16)
        # Garment is 10 mm outside: no contact expected
        region_labels = _region_labels_all(16)
        faces = _make_faces(16)

        result = cp_mod.compute_contact_pressure(
            sim_result=_minimal_sim_result(),
            body_vertices=body_verts,
            garment_vertices=garment_verts,
            region_labels=region_labels,
            fabric_params=_minimal_fabric_params(),
            garment_faces=faces,
            body_normals=body_normals,
        )

        for region, val in result.items():
            assert val == 0.0, (
                f"Region {region!r} has positive clearance but pressure={val} > 0"
            )

    def test_no_contact_without_warp_also_returns_zero(self):
        """Even when Warp is absent the no-contact path still yields zeros."""
        import geometer.warp.contact_pressure as cp_mod

        body_verts, body_normals, garment_verts = _make_cylinder_patch(12)
        region_labels = _region_labels_all(12)

        with mock.patch.object(cp_mod, "_HAS_WARP", False):
            result = cp_mod.compute_contact_pressure(
                sim_result=_minimal_sim_result(),
                body_vertices=body_verts,
                garment_vertices=garment_verts,
                region_labels=region_labels,
                fabric_params=_minimal_fabric_params(),
                body_normals=body_normals,
            )

        assert all(v == 0.0 for v in result.values())


# ---------------------------------------------------------------------------
# TestContactRegionReturnsNonzeroWhenWarp
# ---------------------------------------------------------------------------

class TestContactRegionReturnsNonzeroWhenWarp:

    def test_contact_region_returns_nonzero_when_warp(self):
        """Region with garment penetrating body → pressure > 0 (requires warp)."""
        import geometer.warp.contact_pressure as cp_mod

        if not cp_mod._HAS_WARP:
            pytest.skip("warp not installed")

        body_verts, body_normals, _ = _make_cylinder_patch(16)
        garment_verts = _make_contact_garment(body_verts)  # 5 mm inside
        faces = _make_faces(16)

        # Put all vertices in one region so the median is over all contact verts
        region_labels = {"contact_zone": list(range(16))}

        result = cp_mod.compute_contact_pressure(
            sim_result=_minimal_sim_result(),
            body_vertices=body_verts,
            garment_vertices=garment_verts,
            region_labels=region_labels,
            fabric_params=_minimal_fabric_params(),
            garment_faces=faces,
            body_normals=body_normals,
        )

        assert result["contact_zone"] > 0.0, (
            f"Expected pressure > 0 for penetrating garment, got {result['contact_zone']}"
        )


# ---------------------------------------------------------------------------
# TestAllRegionsPresentInOutput
# ---------------------------------------------------------------------------

class TestAllRegionsPresentInOutput:

    @pytest.mark.parametrize("region_names", [
        ["chest_front", "upper_back", "waist", "shoulder_left", "shoulder_right"],
        ["only_region"],
        [],
    ])
    def test_all_regions_present_in_output(self, region_names):
        """Output dict keys must exactly match the region_labels input keys."""
        import geometer.warp.contact_pressure as cp_mod

        body_verts, body_normals, garment_verts = _make_cylinder_patch(20)
        n = len(body_verts)

        if region_names:
            chunk = max(1, n // len(region_names))
            region_labels = {
                name: list(range(i * chunk, min((i + 1) * chunk, n)))
                for i, name in enumerate(region_names)
            }
        else:
            region_labels = {}

        with mock.patch.object(cp_mod, "_HAS_WARP", False):
            result = cp_mod.compute_contact_pressure(
                sim_result=_minimal_sim_result(),
                body_vertices=body_verts,
                garment_vertices=garment_verts,
                region_labels=region_labels,
                fabric_params=_minimal_fabric_params(),
                body_normals=body_normals,
            )

        assert set(result.keys()) == set(region_labels.keys()), (
            f"Output keys {set(result.keys())} != input keys {set(region_labels.keys())}"
        )

    def test_all_regions_present_with_warp(self):
        """Key parity holds when Warp is actually available."""
        import geometer.warp.contact_pressure as cp_mod

        if not cp_mod._HAS_WARP:
            pytest.skip("warp not installed")

        body_verts, body_normals, garment_verts = _make_cylinder_patch(18)
        faces = _make_faces(18)
        region_labels = _region_labels_all(18)

        result = cp_mod.compute_contact_pressure(
            sim_result=_minimal_sim_result(),
            body_vertices=body_verts,
            garment_vertices=garment_verts,
            region_labels=region_labels,
            fabric_params=_minimal_fabric_params(),
            garment_faces=faces,
            body_normals=body_normals,
        )

        assert set(result.keys()) == set(region_labels.keys())


# ---------------------------------------------------------------------------
# TestPressureNonNegative
# ---------------------------------------------------------------------------

class TestPressureNonNegative:

    def test_pressure_non_negative_no_contact(self):
        """Ease geometry: every value must be >= 0.0."""
        import geometer.warp.contact_pressure as cp_mod

        if not cp_mod._HAS_WARP:
            pytest.skip("warp not installed")

        body_verts, body_normals, garment_verts = _make_cylinder_patch(16)
        faces = _make_faces(16)
        region_labels = _region_labels_all(16)

        result = cp_mod.compute_contact_pressure(
            sim_result=_minimal_sim_result(),
            body_vertices=body_verts,
            garment_vertices=garment_verts,
            region_labels=region_labels,
            fabric_params=_minimal_fabric_params(),
            garment_faces=faces,
            body_normals=body_normals,
        )

        for region, val in result.items():
            assert val >= 0.0, f"Pressure for {region!r} is negative: {val}"

    def test_pressure_non_negative_with_contact(self):
        """Contact geometry: values must still be >= 0.0 (sign-convention check)."""
        import geometer.warp.contact_pressure as cp_mod

        if not cp_mod._HAS_WARP:
            pytest.skip("warp not installed")

        body_verts, body_normals, _ = _make_cylinder_patch(16)
        garment_verts = _make_contact_garment(body_verts)
        faces = _make_faces(16)
        region_labels = _region_labels_all(16)

        result = cp_mod.compute_contact_pressure(
            sim_result=_minimal_sim_result(),
            body_vertices=body_verts,
            garment_vertices=garment_verts,
            region_labels=region_labels,
            fabric_params=_minimal_fabric_params(),
            garment_faces=faces,
            body_normals=body_normals,
        )

        for region, val in result.items():
            assert val >= 0.0, f"Pressure for {region!r} is negative: {val}"

    def test_pressure_non_negative_fallback(self):
        """Fallback path (no warp) must also satisfy non-negativity."""
        import geometer.warp.contact_pressure as cp_mod

        body_verts, body_normals, garment_verts = _make_cylinder_patch(14)
        region_labels = _region_labels_all(14)

        with mock.patch.object(cp_mod, "_HAS_WARP", False):
            result = cp_mod.compute_contact_pressure(
                sim_result=_minimal_sim_result(),
                body_vertices=body_verts,
                garment_vertices=garment_verts,
                region_labels=region_labels,
                fabric_params=_minimal_fabric_params(),
                body_normals=body_normals,
            )

        for region, val in result.items():
            assert val >= 0.0


# ---------------------------------------------------------------------------
# TestMissingPressureMapInSimResult (integration with verdict generator)
# ---------------------------------------------------------------------------

class TestMissingPressureMapInSimResult:

    def _make_minimal_sim_result_for_verdict(self) -> dict:
        """A sim_result dict that mimics the CPU solver — no pressure_map key."""
        return {
            "clearance_map": {
                "chest_front": 12.0,
                "chest_side": 10.0,
                "upper_back": 8.0,
                "waist": 5.0,
                "shoulder_left": 15.0,
                "shoulder_right": 14.0,
            },
            "strain_ratio_map": {
                "chest_front": 1.02,
                "chest_side": 1.01,
                "upper_back": 1.01,
                "waist": 1.00,
                "shoulder_left": 1.03,
                "shoulder_right": 1.02,
            },
            "simulation_ms": 1234,
            "convergence_step": 80,
            "final_kinetic_energy_j": 1e-5,
            "tunnel_through_pct": 0.1,
            # NOTE: no 'pressure_map' key — this is what we are testing
        }

    def test_missing_pressure_map_does_not_crash_verdict(self):
        """generate_verdict must not raise when sim_result has no 'pressure_map'."""
        from verdict.generate_verdict import generate_verdict

        sim_result = self._make_minimal_sim_result_for_verdict()
        assert "pressure_map" not in sim_result

        verdict = generate_verdict(
            sim_result=sim_result,
            garment_id="tshirt_size_M",
            body_profile_id="mannequin_M",
            fabric_id="cotton_jersey_default",
            body_source="synthetic_mannequin",
        )

        # Verdict must be produced
        assert isinstance(verdict, dict)
        assert "fit" in verdict
        # pressure_map should be an empty list (not missing)
        assert "pressure_map" in verdict
        assert verdict["pressure_map"] == [], (
            f"Expected empty pressure_map list, got {verdict['pressure_map']!r}"
        )

    def test_present_pressure_map_is_propagated(self):
        """When sim_result does carry 'pressure_map', verdict includes its entries."""
        from verdict.generate_verdict import generate_verdict

        sim_result = self._make_minimal_sim_result_for_verdict()
        sim_result["pressure_map"] = {
            "chest_front": 0.0042,
            "chest_side": 0.0,
            "upper_back": 0.0,
            "waist": 0.0,
            "shoulder_left": 0.0,
            "shoulder_right": 0.0,
        }

        verdict = generate_verdict(
            sim_result=sim_result,
            garment_id="tshirt_size_M",
            body_profile_id="mannequin_M",
            fabric_id="cotton_jersey_default",
            body_source="synthetic_mannequin",
        )

        assert len(verdict["pressure_map"]) == len(sim_result["pressure_map"])
        regions_in_verdict = {e["region"] for e in verdict["pressure_map"]}
        assert regions_in_verdict == set(sim_result["pressure_map"].keys())
        chest_entry = next(
            e for e in verdict["pressure_map"] if e["region"] == "chest_front"
        )
        assert chest_entry["pressure_n_mm2"] == pytest.approx(0.0042, rel=1e-5)


# ---------------------------------------------------------------------------
# TestPerVertexAreaHelper
# ---------------------------------------------------------------------------

class TestPerVertexAreaHelper:
    """Unit tests for the internal _compute_per_vertex_areas helper."""

    def test_single_triangle_area(self):
        from geometer.warp.contact_pressure import _compute_per_vertex_areas

        # Right triangle with legs 1 m — area = 0.5 m²
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)

        areas = _compute_per_vertex_areas(verts, faces)

        assert areas.shape == (3,)
        # Each vertex gets 1/3 of 0.5 m² ≈ 0.1667 m²
        expected = 0.5 / 3.0
        for i, a in enumerate(areas):
            assert a == pytest.approx(expected, rel=1e-6), (
                f"Vertex {i} area {a} != expected {expected}"
            )

    def test_no_faces_returns_fallback(self):
        from geometer.warp.contact_pressure import _compute_per_vertex_areas

        verts = np.zeros((10, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int64)

        areas = _compute_per_vertex_areas(verts, faces)

        assert areas.shape == (10,)
        assert np.all(areas == pytest.approx(1e-4))

    def test_area_floor_applied(self):
        """Isolated vertices (not referenced by any face) must not be zero."""
        from geometer.warp.contact_pressure import _compute_per_vertex_areas

        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],   # isolated — not in any face
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)

        areas = _compute_per_vertex_areas(verts, faces)

        assert areas[3] >= 1e-8, "Isolated vertex area must be >= floor 1e-8"


# ---------------------------------------------------------------------------
# TestSignedDistanceHelper
# ---------------------------------------------------------------------------

class TestSignedDistanceHelper:

    def test_outside_body_positive(self):
        from geometer.warp.contact_pressure import _compute_signed_distances

        body_verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        body_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        # Garment 50 mm ahead of body surface
        garment_verts = np.array([[0.0, 0.0, 0.05]], dtype=np.float64)

        dists = _compute_signed_distances(garment_verts, body_verts, body_normals)

        assert dists[0] == pytest.approx(0.05, rel=1e-6)

    def test_inside_body_negative(self):
        from geometer.warp.contact_pressure import _compute_signed_distances

        body_verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        body_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        # Garment 5 mm inside body
        garment_verts = np.array([[0.0, 0.0, -0.005]], dtype=np.float64)

        dists = _compute_signed_distances(garment_verts, body_verts, body_normals)

        assert dists[0] == pytest.approx(-0.005, rel=1e-6)

    def test_on_body_surface_zero(self):
        from geometer.warp.contact_pressure import _compute_signed_distances

        body_verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        body_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        garment_verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        dists = _compute_signed_distances(garment_verts, body_verts, body_normals)

        assert dists[0] == pytest.approx(0.0, abs=1e-12)
