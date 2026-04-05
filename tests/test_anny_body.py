"""
tests/test_anny_body.py

Unit tests for src/sculptor/anny_body.py — Forma's parametric body generator.

Covers:
    1. Return type is trimesh.Trimesh
    2. Face count within expected range
    3. Z extent matches height_cm input within 10%
    4. Standard female size M runs without error
    5. Large male dimensions run without error
    6. PLY is written to disk and is loadable when output_path is given
    7. body_source field equals "anny_parametric"
    8. S and XL bodies are geometrically distinct
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.sculptor.anny_body import generate_anny_body, generate_anny_size  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixture — size M female (test requirement 4 measurements)
# ---------------------------------------------------------------------------

SIZE_M_KWARGS = dict(
    height_cm=168,
    chest_cm=88,
    waist_cm=70,
    hips_cm=94,
    inseam_cm=76,
    shoulder_width_cm=37,
)

SIZE_XL_MALE_KWARGS = dict(
    height_cm=185,
    chest_cm=108,
    waist_cm=96,
    hips_cm=106,
    inseam_cm=82,
    shoulder_width_cm=46,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_trimesh():
    """generate_anny_body() must return a trimesh.Trimesh instance."""
    mesh = generate_anny_body(**SIZE_M_KWARGS)
    assert isinstance(mesh, trimesh.Trimesh), (
        f"Expected trimesh.Trimesh, got {type(mesh)}"
    )


def test_mesh_has_faces():
    """Face count must be between 1 000 and 20 000."""
    mesh = generate_anny_body(**SIZE_M_KWARGS)
    fc = len(mesh.faces)
    assert 1000 <= fc <= 20000, (
        f"Face count {fc} is outside the expected range [1000, 20000]"
    )


def test_approximate_height():
    """
    Mesh Z extent must be within 10% of height_cm / 100.

    Anny places feet at Z=0 and head top at Z≈height_m, so
    bounds[1][2] - bounds[0][2] should be close to height_m.
    """
    height_cm = SIZE_M_KWARGS["height_cm"]
    mesh = generate_anny_body(**SIZE_M_KWARGS)
    z_min, z_max = mesh.bounds[:, 2]
    z_extent = z_max - z_min
    height_m = height_cm / 100.0
    tolerance = height_m * 0.10
    assert abs(z_extent - height_m) <= tolerance, (
        f"Z extent {z_extent:.4f}m is more than 10% off target {height_m:.4f}m"
    )


def test_size_m_female():
    """Standard female size M (168/88/70/94/76/37) must run without error."""
    mesh = generate_anny_body(**SIZE_M_KWARGS)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_size_xl_male():
    """Large male dimensions (185/108/96/106/82/46) must run without error."""
    mesh = generate_anny_body(**SIZE_XL_MALE_KWARGS)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_output_path_saves_ply(tmp_path):
    """
    When output_path is provided the PLY file must be created on disk
    and must be loadable as a valid trimesh.Trimesh with faces.
    """
    out = tmp_path / "anny_test.ply"
    generate_anny_body(**SIZE_M_KWARGS, output_path=str(out))

    assert out.exists(), f"Expected PLY at {out} but file was not created"

    loaded = trimesh.load(str(out), process=False)
    assert isinstance(loaded, trimesh.Trimesh), (
        f"Loaded object is {type(loaded)}, not trimesh.Trimesh"
    )
    assert len(loaded.faces) > 0, "Loaded PLY has no faces"


def test_body_source_field(tmp_path):
    """
    generate_anny_body() must set body_source = "anny_parametric".

    Verified via the companion body_profile.json written alongside the PLY,
    which is the authoritative output carrier for this field.
    """
    out = tmp_path / "anny_source_test.ply"
    generate_anny_body(**SIZE_M_KWARGS, output_path=str(out))

    profile_path = out.with_suffix(".json")
    assert profile_path.exists(), (
        f"Expected body_profile JSON at {profile_path} but it was not created"
    )

    with open(profile_path) as fh:
        profile = json.load(fh)

    assert "body_source" in profile, "body_profile JSON missing 'body_source' field"
    assert profile["body_source"] == "anny_parametric", (
        f"Expected body_source='anny_parametric', got '{profile['body_source']}'"
    )


def test_different_sizes_differ():
    """
    S and XL Anny bodies must be geometrically distinct: they must differ in
    vertex count OR in Z extent (a uniform-scale mesh could have the same face
    topology but different heights).
    """
    # Use the built-in size presets (S: 166 cm, XL: 172 cm female)
    mesh_s  = generate_anny_size("S")
    mesh_xl = generate_anny_size("XL")

    same_verts = len(mesh_s.vertices) == len(mesh_xl.vertices)
    z_extent_s  = mesh_s.bounds[1][2]  - mesh_s.bounds[0][2]
    z_extent_xl = mesh_xl.bounds[1][2] - mesh_xl.bounds[0][2]
    same_z = abs(z_extent_xl - z_extent_s) < 1e-6

    assert not (same_verts and same_z), (
        "S and XL bodies have identical vertex count and Z extent — "
        "size scaling is not working"
    )
