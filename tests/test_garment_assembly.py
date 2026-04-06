"""
tests/test_garment_assembly.py

Unit tests for pure-numpy/scipy functions in src/geometer/garment_assembly.py.
No GPU, no CUDA, no warp-lang required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from geometer.garment_assembly import (
    build_panel_outline_2d,
    points_inside_polygon,
    triangulate_panel,
    resample_1d,
    compute_strain_ratios,
    REQUIRED_REGIONS,
)


# ---------------------------------------------------------------------------
# points_inside_polygon
# ---------------------------------------------------------------------------

class TestPointsInsidePolygon:

    def _unit_square(self):
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def test_center_inside(self):
        poly = self._unit_square()
        result = points_inside_polygon(np.array([[0.5, 0.5]]), poly)
        assert result[0] is np.bool_(True)

    def test_outside_point(self):
        poly = self._unit_square()
        result = points_inside_polygon(np.array([[2.0, 2.0]]), poly)
        assert not result[0]

    def test_corner_point(self):
        poly = self._unit_square()
        # Corners are edge cases — just check no crash
        points_inside_polygon(np.array([[0.0, 0.0]]), poly)

    def test_multiple_points(self):
        poly = self._unit_square()
        pts = np.array([[0.5, 0.5], [2.0, 2.0], [0.1, 0.1], [-0.1, 0.5]])
        result = points_inside_polygon(pts, poly)
        assert result[0]   # inside
        assert not result[1]  # outside
        assert result[2]   # inside
        assert not result[3]  # outside

    def test_triangle_polygon(self):
        # Equilateral triangle
        poly = np.array([[0, 0], [2, 0], [1, 2]], dtype=float)
        inside = points_inside_polygon(np.array([[1.0, 0.5]]), poly)
        outside = points_inside_polygon(np.array([[0.0, 1.5]]), poly)
        assert inside[0]
        assert not outside[0]


# ---------------------------------------------------------------------------
# triangulate_panel
# ---------------------------------------------------------------------------

class TestTriangulatePanel:

    def _square_outline(self, size=20.0):
        """20cm square outline as [x, y] list."""
        return [[0, 0], [size, 0], [size, size], [0, size]]

    def test_returns_arrays(self):
        pts, faces = triangulate_panel(self._square_outline())
        assert isinstance(pts, np.ndarray)
        assert isinstance(faces, np.ndarray)

    def test_pts_shape(self):
        pts, faces = triangulate_panel(self._square_outline())
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_faces_shape(self):
        pts, faces = triangulate_panel(self._square_outline())
        assert faces.ndim == 2
        assert faces.shape[1] == 3

    def test_has_interior_points(self):
        """20cm square should get interior grid points at 5cm spacing."""
        pts, _ = triangulate_panel(self._square_outline(20.0))
        # Outline has 4 points; with 5cm grid we expect interior points
        assert len(pts) > 4

    def test_face_indices_in_range(self):
        pts, faces = triangulate_panel(self._square_outline())
        assert faces.min() >= 0
        assert faces.max() < len(pts)

    def test_all_triangle_centroids_inside(self):
        """All returned triangles must have centroids inside the outline."""
        outline = self._square_outline(20.0)
        pts, faces = triangulate_panel(outline)
        centroids = pts[faces].mean(axis=1)
        poly = np.array(outline, dtype=float)
        inside = points_inside_polygon(centroids, poly)
        assert np.all(inside), f"{(~inside).sum()} triangles have outside centroids"

    def test_small_panel_still_works(self):
        """A 3×3cm panel (smaller than grid step) should triangulate without error."""
        pts, faces = triangulate_panel([[0, 0], [3, 0], [3, 3], [0, 3]])
        assert len(faces) >= 1


# ---------------------------------------------------------------------------
# resample_1d
# ---------------------------------------------------------------------------

class TestResample1d:

    def _line_vertices(self, n=10):
        """Evenly-spaced vertices along the x-axis: [0,0,0] to [9,0,0]."""
        return np.array([[float(i), 0.0, 0.0] for i in range(n)])

    def test_same_count_returns_same(self):
        verts = self._line_vertices(5)
        indices = list(range(5))
        result = resample_1d(indices, verts, 5)
        assert result == indices

    def test_upsample_returns_correct_count(self):
        verts = self._line_vertices(5)
        indices = list(range(5))
        result = resample_1d(indices, verts, 9)
        assert len(result) == 9

    def test_downsample_returns_correct_count(self):
        verts = self._line_vertices(10)
        indices = list(range(10))
        result = resample_1d(indices, verts, 4)
        assert len(result) == 4

    def test_single_point_output(self):
        verts = self._line_vertices(5)
        result = resample_1d(list(range(5)), verts, 1)
        assert len(result) == 1

    def test_degenerate_zero_length_returns_fill(self):
        """All-same-position vertices → zero arc length → fill with first index."""
        verts = np.zeros((5, 3))
        result = resample_1d(list(range(5)), verts, 4)
        assert len(result) == 4
        assert all(r == 0 for r in result)

    def test_indices_in_range(self):
        verts = self._line_vertices(10)
        result = resample_1d(list(range(10)), verts, 7)
        assert all(0 <= r < 10 for r in result)


# ---------------------------------------------------------------------------
# build_panel_outline_2d
# ---------------------------------------------------------------------------

class TestBuildPanelOutline2d:

    def _simple_panel(self):
        """Minimal panel with two straight edges covering a square."""
        return {
            "edges": [
                {"polyline": [[0, 0], [10, 0]]},
                {"polyline": [[10, 0], [10, 10], [0, 10], [0, 0]]},
            ]
        }

    def test_returns_list(self):
        result = build_panel_outline_2d(self._simple_panel())
        assert isinstance(result, list)

    def test_all_points_are_pairs(self):
        result = build_panel_outline_2d(self._simple_panel())
        for pt in result:
            assert len(pt) == 2

    def test_chains_all_edges(self):
        panel = self._simple_panel()
        total = sum(len(e["polyline"]) for e in panel["edges"])
        result = build_panel_outline_2d(panel)
        assert len(result) == total

    def test_first_point_matches_first_edge_start(self):
        result = build_panel_outline_2d(self._simple_panel())
        assert result[0] == [0, 0]


# ---------------------------------------------------------------------------
# compute_strain_ratios
# ---------------------------------------------------------------------------

class TestComputeStrainRatios:

    def _minimal_garment_regions(self):
        """All required regions mapped to vertex indices [0, 1, 2]."""
        return {r: [0, 1, 2] for r in REQUIRED_REGIONS}

    def _per_region_edges(self, n_edges=10, rest_len=1.0, scale=1.0):
        """Build edges where each region owns its OWN disjoint set of vertices.
        vertex_to_region maps each vertex to exactly one region, so all n_edges
        per region are counted correctly.

        Returns (positions, si, sj, sr, regions).
        """
        n_regions = len(REQUIRED_REGIONS)
        verts_per_region = n_edges + 1  # chain of n_edges edges
        total_verts = n_regions * verts_per_region

        positions = np.zeros((total_verts, 3))
        si_list, sj_list, sr_list = [], [], []
        regions = {}

        for r_idx, region in enumerate(REQUIRED_REGIONS):
            base = r_idx * verts_per_region
            vids = list(range(base, base + verts_per_region))
            regions[region] = vids
            # Place vertices in a line with current spacing = rest_len * scale
            for k, vid in enumerate(vids):
                positions[vid] = [k * rest_len * scale, float(r_idx), 0]
            # Edges: base → base+1, base+1 → base+2, ...
            for k in range(n_edges):
                si_list.append(vids[k])
                sj_list.append(vids[k + 1])
                sr_list.append(rest_len)

        si = np.array(si_list, dtype=np.int32)
        sj = np.array(sj_list, dtype=np.int32)
        sr = np.array(sr_list)
        return positions, si, sj, sr, regions

    def test_no_deformation_strain_one(self):
        """No deformation → all regions report strain ≈ 1.0."""
        positions, si, sj, sr, regions = self._per_region_edges(n_edges=10, scale=1.0)
        ratios = compute_strain_ratios(positions, si, sj, sr, regions)
        for region in REQUIRED_REGIONS:
            assert region in ratios
            assert abs(ratios[region] - 1.0) < 0.01, f"Region {region}: ratio={ratios[region]}"

    def test_stretched_garment_strain_above_one(self):
        """Edges stretched to 2× rest → all regions report strain ≈ 2.0."""
        positions, si, sj, sr, regions = self._per_region_edges(n_edges=10, scale=2.0)
        ratios = compute_strain_ratios(positions, si, sj, sr, regions)
        for region in REQUIRED_REGIONS:
            assert ratios[region] > 1.5, f"Region {region}: expected ~2.0, got {ratios[region]}"

    def test_returns_all_required_regions(self):
        positions, si, sj, sr, regions = self._per_region_edges()
        ratios = compute_strain_ratios(positions, si, sj, sr, regions)
        assert set(ratios.keys()) == set(REQUIRED_REGIONS)

    def test_empty_region_returns_one(self):
        """Regions with no vertices return 1.0 (fallback)."""
        positions, si, sj, sr, regions = self._per_region_edges(n_edges=10)
        regions["chest_front"] = []
        ratios = compute_strain_ratios(positions, si, sj, sr, regions)
        assert ratios["chest_front"] == 1.0

    def test_compressed_garment_strain_below_one(self):
        """Edges compressed to 0.5× rest → all regions report strain ≈ 0.5."""
        positions, si, sj, sr, regions = self._per_region_edges(n_edges=10, scale=0.5)
        ratios = compute_strain_ratios(positions, si, sj, sr, regions)
        for region in REQUIRED_REGIONS:
            assert ratios[region] < 0.7, f"Region {region}: expected ~0.5, got {ratios[region]}"

    def test_insufficient_edges_returns_one(self):
        """Fewer than 5 edges in a region → fallback to 1.0."""
        positions, si, sj, sr, regions = self._per_region_edges(n_edges=4, scale=2.0)
        ratios = compute_strain_ratios(positions, si, sj, sr, regions)
        for region in REQUIRED_REGIONS:
            assert ratios[region] == 1.0, f"Region {region}: expected 1.0 (fallback), got {ratios[region]}"
