"""
tests/test_mannequin.py

Acceptance tests for the MakeHuman full-body mesh (AC-0, Week 2).

Tests verify:
1. Mesh files exist and load as valid trimesh objects
2. Meshes are non-degenerate (vertex count, face count, no NaN)
3. All three sizes (S, M, XL) are within ±2cm of target circumferences
4. body_profile JSON matches v1.2 schema requirements
5. Meshes have realistic body proportions (arms, shoulders)
"""

import json
import numpy as np
import pytest
import trimesh
from pathlib import Path
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
BODIES_DIR = ROOT / "data" / "bodies"

BODY_CONFIGS = {
    "S":  {"path": BODIES_DIR / "makehuman_male_S.ply",
           "profile": BODIES_DIR / "makehuman_male_S.json",
           "height_cm": 170.0, "chest_cm": 88.0, "waist_cm": 72.0, "hip_cm": 87.0},
    "M":  {"path": BODIES_DIR / "makehuman_male_M.ply",
           "profile": BODIES_DIR / "makehuman_male_M.json",
           "height_cm": 180.0, "chest_cm": 96.0, "waist_cm": 80.0, "hip_cm": 95.0},
    "XL": {"path": BODIES_DIR / "makehuman_male_XL.ply",
           "profile": BODIES_DIR / "makehuman_male_XL.json",
           "height_cm": 185.0, "chest_cm": 108.0, "waist_cm": 96.0, "hip_cm": 105.0},
}

# Tolerance: ±2cm for circumferences, ±1cm for height
CIRC_TOL_CM = 2.0
HEIGHT_TOL_CM = 1.0

# Height bands for measurement (fraction of body height)
CHEST_FRAC = (0.67, 0.76)
WAIST_FRAC = (0.56, 0.63)
HIP_FRAC = (0.48, 0.56)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_arm_gap_x(vertices, y_height, band=0.015):
    """Find X threshold separating torso from arm vertices."""
    mask = np.abs(vertices[:, 1] - y_height) < band
    if np.sum(mask) < 10:
        return 0.20
    abs_x = np.sort(np.abs(vertices[mask, 0]))
    gaps = np.diff(abs_x)
    for i, g in enumerate(gaps):
        if abs_x[i] > 0.10 and g > 0.03:
            return float(abs_x[i]) + 0.005
    return 0.20


def measure_circumference(vertices, y_height, band=0.015):
    """Measure torso circumference at y_height using convex hull perimeter."""
    x_limit = _find_arm_gap_x(vertices, y_height, band)
    mask = (np.abs(vertices[:, 1] - y_height) < band) & (np.abs(vertices[:, 0]) < x_limit)
    pts = vertices[mask][:, [0, 2]]
    if len(pts) < 3:
        return 0.0
    hull = ConvexHull(pts)
    hp = pts[hull.vertices]
    hpc = np.vstack([hp, hp[0]])
    return float(np.sum(np.linalg.norm(np.diff(hpc, axis=0), axis=1))) * 100.0


def scan_for_landmark(vertices, y_lo, y_hi, maximize=True):
    """Scan heights to find max (or min) circumference."""
    best_y = (y_lo + y_hi) / 2
    best_circ = 0.0 if maximize else 1e9
    for y in np.linspace(y_lo, y_hi, 25):
        circ = measure_circumference(vertices, y)
        if circ < 1.0:
            continue
        if (maximize and circ > best_circ) or (not maximize and circ < best_circ):
            best_circ = circ
            best_y = y
    return best_y, best_circ


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestMeshExists:
    """Basic file existence checks for all three body sizes."""

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_ply_exists(self, size):
        assert BODY_CONFIGS[size]["path"].exists()

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_profile_json_exists(self, size):
        assert BODY_CONFIGS[size]["profile"].exists()


class TestMeshGeometry:
    """Mesh structural validity for all sizes."""

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_has_vertices(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        assert len(mesh.vertices) >= 10000, f"Expected ≥10000 verts, got {len(mesh.vertices)}"

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_vertex_count_under_50k(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        assert len(mesh.vertices) < 50000, f"Vertex count {len(mesh.vertices)} exceeds 50k limit"

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_has_faces(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        assert len(mesh.faces) >= 10000

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_no_nan_vertices(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        assert not np.any(np.isnan(mesh.vertices))

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_has_arms(self, size):
        """Full body mesh should have arms extending beyond ±0.3m in X."""
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        x_extent = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
        assert x_extent > 0.6, f"X extent {x_extent:.2f}m — no arms detected"

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_vertices_within_body_bounds(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        verts = mesh.vertices
        height_m = BODY_CONFIGS[size]["height_cm"] / 100.0
        assert verts[:, 1].min() >= -0.01
        assert verts[:, 1].max() <= height_m + 0.01


class TestMeasurements:
    """Verify measurements within ±2cm of targets."""

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_height(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        v = mesh.vertices
        height_cm = (v[:, 1].max() - v[:, 1].min()) * 100.0
        target = BODY_CONFIGS[size]["height_cm"]
        err_cm = abs(height_cm - target)
        assert err_cm <= HEIGHT_TOL_CM, (
            f"Size {size} height: {height_cm:.2f}cm (target {target}, err {err_cm:.2f}cm)"
        )

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_chest(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        v = mesh.vertices
        height_m = BODY_CONFIGS[size]["height_cm"] / 100.0
        _, circ = scan_for_landmark(v, CHEST_FRAC[0] * height_m, CHEST_FRAC[1] * height_m)
        target = BODY_CONFIGS[size]["chest_cm"]
        err_cm = abs(circ - target)
        assert err_cm <= CIRC_TOL_CM, (
            f"Size {size} chest: {circ:.2f}cm (target {target}, err {err_cm:.2f}cm)"
        )

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_waist(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        v = mesh.vertices
        height_m = BODY_CONFIGS[size]["height_cm"] / 100.0
        _, circ = scan_for_landmark(
            v, WAIST_FRAC[0] * height_m, WAIST_FRAC[1] * height_m, maximize=False,
        )
        target = BODY_CONFIGS[size]["waist_cm"]
        err_cm = abs(circ - target)
        assert err_cm <= CIRC_TOL_CM, (
            f"Size {size} waist: {circ:.2f}cm (target {target}, err {err_cm:.2f}cm)"
        )

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_hip(self, size):
        mesh = trimesh.load(str(BODY_CONFIGS[size]["path"]), process=False)
        v = mesh.vertices
        height_m = BODY_CONFIGS[size]["height_cm"] / 100.0
        _, circ = scan_for_landmark(v, HIP_FRAC[0] * height_m, HIP_FRAC[1] * height_m)
        target = BODY_CONFIGS[size]["hip_cm"]
        err_cm = abs(circ - target)
        assert err_cm <= CIRC_TOL_CM, (
            f"Size {size} hip: {circ:.2f}cm (target {target}, err {err_cm:.2f}cm)"
        )

    def test_chest_larger_than_waist_M(self):
        mesh = trimesh.load(str(BODY_CONFIGS["M"]["path"]), process=False)
        v = mesh.vertices
        _, chest = scan_for_landmark(v, 1.20, 1.37)
        _, waist = scan_for_landmark(v, 1.00, 1.13, maximize=False)
        assert chest > waist

    def test_hip_larger_than_waist_M(self):
        mesh = trimesh.load(str(BODY_CONFIGS["M"]["path"]), process=False)
        v = mesh.vertices
        _, hip = scan_for_landmark(v, 0.86, 1.00)
        _, waist = scan_for_landmark(v, 1.00, 1.13, maximize=False)
        assert hip > waist


class TestBodyProfileSchema:
    """body_profile JSON matches v1.2 schema."""

    REQUIRED_FIELDS = [
        "body_profile_id", "body_source", "scan_method", "scan_accuracy_mm",
        "confidence", "measurements", "achieved_measurements",
        "max_measurement_error_mm", "mesh_path", "created_at",
    ]

    REQUIRED_MEASUREMENTS = [
        "height_cm", "chest_cm", "waist_cm", "hip_cm",
        "shoulder_width_cm", "inseam_cm",
    ]

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_all_required_fields_present(self, size):
        with open(BODY_CONFIGS[size]["profile"]) as f:
            profile = json.load(f)
        for field in self.REQUIRED_FIELDS:
            assert field in profile, f"Size {size}: missing field {field}"

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_body_source_is_synthetic_mannequin(self, size):
        with open(BODY_CONFIGS[size]["profile"]) as f:
            profile = json.load(f)
        assert profile["body_source"] == "synthetic_mannequin"

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_confidence_is_1(self, size):
        with open(BODY_CONFIGS[size]["profile"]) as f:
            profile = json.load(f)
        assert profile["confidence"] == 1.0

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_all_measurement_fields_present(self, size):
        with open(BODY_CONFIGS[size]["profile"]) as f:
            profile = json.load(f)
        for field in self.REQUIRED_MEASUREMENTS:
            assert field in profile["measurements"]
            assert field in profile["achieved_measurements"]

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_max_error_within_20mm(self, size):
        """Max measurement error should be within 20mm (±2cm tolerance)."""
        with open(BODY_CONFIGS[size]["profile"]) as f:
            profile = json.load(f)
        err = profile["max_measurement_error_mm"]
        assert err <= 20.0, f"Size {size}: max_error {err:.1f}mm exceeds 20mm"

    @pytest.mark.parametrize("size", ["S", "M", "XL"])
    def test_created_at_is_iso8601(self, size):
        from datetime import datetime
        with open(BODY_CONFIGS[size]["profile"]) as f:
            profile = json.load(f)
        datetime.fromisoformat(profile["created_at"])
