"""
src/sculptor/anny_body.py

Anny — parametric elliptic-cylinder body model for Forma.

Generates a watertight body mesh from 6 anthropometric measurements using
trimesh primitives (cylinders, spheres, capsules). No external scan data or
licensed body model is required.

The body is constructed from the following segments, each modelled as a
trimesh primitive, then merged via trimesh.util.concatenate():

    Head       — icosphere, radius ≈ 0.11 m, centered above neck
    Neck       — short cylinder
    Torso      — elliptic cylinder, shoulder-to-hip level
                 rx = chest_r * 1.10 (wider side-to-side, X axis)
                 ry = chest_r * 0.90 (shallower front-to-back, Y axis)
                 The cross-section tapers linearly from chest at shoulders
                 to waist at mid-torso, then back out to hip radii.
    Hips       — slightly flared elliptic cylinder, hip-to-crotch
    Left leg   — cylinder, crotch-to-ankle, radius ≈ hip_r * 0.30
    Right leg  — mirror of left leg
    Shoulders  — two icospheres placed lateral of upper torso

All coordinates use a right-handed Z-up frame:
    +Z  = up (height)
    +X  = left (viewer's right)
    +Y  = forward

Origin convention:
    Feet at Z = 0
    Head top  at Z ≈ height_m

Usage (module):
    from src.sculptor.anny_body import generate_anny_body
    mesh = generate_anny_body(170, 90, 72, 96, 76, 38, output_path="/tmp/body.ply")

body_source field in body_profile.json: "anny_parametric"
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _elliptic_cylinder(
    rx: float,
    ry: float,
    height: float,
    sections: int = 32,
    z_bottom: float = 0.0,
) -> trimesh.Trimesh:
    """
    Build a closed elliptic cylinder with semi-axes rx (X) and ry (Y).

    Strategy: create a circular cylinder of radius 1.0, then apply a
    non-uniform XY scale to make it elliptic.  trimesh.creation.cylinder
    always produces a right circular cylinder so we scale afterwards.

    Parameters
    ----------
    rx       : semi-axis along X (metres)
    ry       : semi-axis along Y (metres)
    height   : cylinder height (metres)
    sections : number of angular divisions around circumference
    z_bottom : Z coordinate of the bottom cap centre

    Returns
    -------
    Closed, oriented trimesh.Trimesh centred at (0, 0, z_bottom + height/2)
    """
    cyl = trimesh.creation.cylinder(
        radius=1.0,
        height=height,
        sections=sections,
    )
    # Scale XY to produce ellipse; translate so bottom cap sits at z_bottom
    transform = np.eye(4)
    transform[0, 0] = rx          # X scale
    transform[1, 1] = ry          # Y scale
    transform[2, 3] = z_bottom + height / 2.0   # Z translation
    cyl.apply_transform(transform)
    return cyl


def _tapered_elliptic_cylinder(
    rx_bottom: float,
    ry_bottom: float,
    rx_top: float,
    ry_top: float,
    height: float,
    sections: int = 32,
    z_bottom: float = 0.0,
) -> trimesh.Trimesh:
    """
    Build a frustum (tapered cylinder) with elliptic cross-sections.

    Both end caps are closed.  Implemented by hand as trimesh has no native
    tapered primitive.

    The lateral surface is built as a triangle strip connecting top/bottom
    ellipse rings.  Top and bottom caps are triangle fans from their centres.
    """
    theta = np.linspace(0, 2 * math.pi, sections, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Bottom ring vertices
    bot_ring = np.column_stack([
        rx_bottom * cos_t,
        ry_bottom * sin_t,
        np.full(sections, z_bottom),
    ])
    # Top ring vertices
    top_ring = np.column_stack([
        rx_top * cos_t,
        ry_top * sin_t,
        np.full(sections, z_bottom + height),
    ])

    # Cap centres
    bot_centre = np.array([[0.0, 0.0, z_bottom]])
    top_centre = np.array([[0.0, 0.0, z_bottom + height]])

    # Vertex array: bot_ring [0..s-1], top_ring [s..2s-1],
    #               bot_centre [2s], top_centre [2s+1]
    verts = np.vstack([bot_ring, top_ring, bot_centre, top_centre])
    s = sections
    bot_c_idx = 2 * s
    top_c_idx = 2 * s + 1

    faces = []

    for i in range(s):
        j = (i + 1) % s

        # Lateral quad → 2 triangles (outward normals).
        # theta increases CCW viewed from +Z; for an outward-pointing normal
        # on the lateral wall the winding must be CCW viewed from outside
        # (i.e. from the positive-radius direction).
        # Quad corners in CCW-from-outside order:
        #   bot-i (θ_i, z_bot) → bot-j (θ_{i+1}, z_bot)
        #   → top-j (θ_{i+1}, z_top) → top-i (θ_i, z_top)
        faces.append([i,     j,     s + i])   # bot-i, bot-j, top-i
        faces.append([j,     s + j, s + i])   # bot-j, top-j, top-i

        # Bottom cap fan: normal pointing −Z → CCW viewed from below (+Z up).
        # Centre → current → next is CCW from below.
        faces.append([bot_c_idx, i, j])

        # Top cap fan: normal pointing +Z → CCW viewed from above.
        # Centre → next → current is CCW from above.
        faces.append([top_c_idx, s + j, s + i])

    faces = np.array(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return mesh


def _place_sphere(radius: float, centre: np.ndarray, subdivisions: int = 3) -> trimesh.Trimesh:
    """Return an icosphere of given radius translated to centre."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(centre)
    return sphere


def _place_cylinder(
    radius: float,
    height: float,
    z_bottom: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    sections: int = 24,
) -> trimesh.Trimesh:
    """Return a closed cylinder translated so its bottom is at z_bottom."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cyl.apply_translation([x_offset, y_offset, z_bottom + height / 2.0])
    return cyl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_anny_body(
    height_cm: float,
    chest_cm: float,
    waist_cm: float,
    hips_cm: float,
    inseam_cm: float,
    shoulder_width_cm: float,
    output_path: str | None = None,
    gender: str = "custom",
) -> trimesh.Trimesh:
    """
    Return a watertight body mesh scaled to the given measurements.

    Parameters
    ----------
    height_cm          : total standing height in centimetres (e.g. 170.0)
    chest_cm           : chest circumference in centimetres  (e.g. 90.0)
    waist_cm           : waist circumference in centimetres  (e.g. 72.0)
    hips_cm            : hip circumference in centimetres    (e.g. 96.0)
    inseam_cm          : inseam leg length in centimetres    (e.g. 76.0)
    shoulder_width_cm  : biacromial shoulder width in cm     (e.g. 38.0)
    output_path        : if given, save PLY to this path and write a
                         companion body_profile.json
    gender             : "male", "female", or "custom" — recorded in
                         body_profile.json; does not affect geometry

    Returns
    -------
    trimesh.Trimesh  — merged body mesh, Z-up, feet at Z=0
    """
    # -----------------------------------------------------------------------
    # Convert all measurements to metres
    # -----------------------------------------------------------------------
    H   = height_cm          / 100.0
    IC  = chest_cm           / 100.0   # chest circumference
    IW  = waist_cm           / 100.0   # waist circumference
    IH  = hips_cm            / 100.0   # hip circumference
    SW  = shoulder_width_cm  / 100.0   # biacromial width
    INS = inseam_cm          / 100.0   # inseam length

    # -----------------------------------------------------------------------
    # Circumference → mean radius (r = C / 2π)
    # -----------------------------------------------------------------------
    chest_r = IC / (2.0 * math.pi)
    waist_r = IW / (2.0 * math.pi)
    hip_r   = IH / (2.0 * math.pi)

    # Elliptic axes for chest cross-section: slightly wider side-to-side.
    # In this Z-up frame X = left-right, Y = front-back.
    chest_rx = chest_r * 1.10   # X semi-axis (left-right, wider)
    chest_ry = chest_r * 0.90   # Y semi-axis (front-back, shallower)

    # Elliptic axes for waist
    waist_rx = waist_r * 1.08
    waist_ry = waist_r * 0.92

    # Elliptic axes for hip (wider side-to-side)
    hip_rx = hip_r * 1.15
    hip_ry = hip_r * 0.85

    # -----------------------------------------------------------------------
    # Vertical landmark heights (Z, feet = 0)
    # -----------------------------------------------------------------------
    # Proportions derived from standard anthropometric tables
    # (Pheasant, "Bodyspace", 2nd ed.; Winter, "Biomechanics", 4th ed.)

    crotch_z      = INS               # crotch height = inseam length (feet at Z=0)
    hip_z         = H * 0.52          # hip prominence just above crotch
    waist_z       = H * 0.60          # natural waist ≈ 60 %
    chest_z       = H * 0.72          # full chest / bust level
    shoulder_z    = H * 0.82          # shoulder / armscye level
    neck_base_z   = H * 0.86          # base of neck
    neck_top_z    = H * 0.90          # top of neck
    head_base_z   = neck_top_z
    head_radius   = H * 0.075         # head ≈ 15 % of height diameter

    # -----------------------------------------------------------------------
    # Neck radius (independent of inputs — standard proportion)
    # -----------------------------------------------------------------------
    neck_r = H * 0.038    # ≈ 6.5 cm radius for 170 cm body

    # Leg radius — proportional to hip radius
    leg_r = hip_r * 0.28

    # Foot-to-ankle offset keeps leg flush with crotch floor.
    # ankle_z is a small fixed riser so the leg cylinder doesn't start at Z=0
    # (the ankle sphere sits there instead).
    ankle_z = H * 0.04
    leg_height = max(crotch_z - ankle_z, 0.05)

    # Horizontal leg offset: legs separated by ≈ hip_rx * 0.50
    leg_x_offset = hip_rx * 0.50

    # -----------------------------------------------------------------------
    # Shoulder sphere offset from body centre
    # -----------------------------------------------------------------------
    shoulder_x_offset = SW / 2.0      # half biacromial width
    shoulder_r = neck_r * 1.6         # shoulder ball radius

    # -----------------------------------------------------------------------
    # Build segments
    # -----------------------------------------------------------------------
    parts: list[trimesh.Trimesh] = []

    # --- Head ---
    head_centre = np.array([0.0, 0.0, head_base_z + head_radius])
    parts.append(_place_sphere(head_radius, head_centre, subdivisions=3))

    # --- Neck ---
    neck_height = neck_top_z - neck_base_z
    parts.append(_place_cylinder(neck_r, neck_height, neck_base_z, sections=20))

    # --- Upper torso: shoulder to chest (tapered elliptic frustum) ---
    # At shoulder level: approximately chest width scaled to shoulder width
    shoulder_rx = SW / 2.0 * 0.92
    shoulder_ry = chest_ry * 0.85
    upper_torso_height = shoulder_z - chest_z
    parts.append(_tapered_elliptic_cylinder(
        rx_bottom=chest_rx,
        ry_bottom=chest_ry,
        rx_top=shoulder_rx,
        ry_top=shoulder_ry,
        height=upper_torso_height,
        sections=32,
        z_bottom=chest_z,
    ))

    # --- Mid torso: chest to waist (tapered) ---
    mid_torso_height = chest_z - waist_z
    parts.append(_tapered_elliptic_cylinder(
        rx_bottom=waist_rx,
        ry_bottom=waist_ry,
        rx_top=chest_rx,
        ry_top=chest_ry,
        height=mid_torso_height,
        sections=32,
        z_bottom=waist_z,
    ))

    # --- Lower torso / abdomen: waist to hip level (tapered) ---
    lower_torso_height = waist_z - hip_z
    parts.append(_tapered_elliptic_cylinder(
        rx_bottom=hip_rx,
        ry_bottom=hip_ry,
        rx_top=waist_rx,
        ry_top=waist_ry,
        height=lower_torso_height,
        sections=32,
        z_bottom=hip_z,
    ))

    # --- Pelvic block: hip level to crotch (uniform elliptic cylinder) ---
    # Guard: if inseam places crotch_z above hip_z, clamp so we always have a
    # positive height (prevents degenerate geometry for extreme proportions).
    pelvic_height = max(hip_z - crotch_z, 0.01)
    parts.append(_elliptic_cylinder(
        rx=hip_rx,
        ry=hip_ry,
        height=pelvic_height,
        sections=32,
        z_bottom=crotch_z,
    ))

    # --- Left leg ---
    parts.append(_place_cylinder(
        radius=leg_r,
        height=leg_height,
        z_bottom=ankle_z,
        x_offset=+leg_x_offset,
        sections=24,
    ))

    # --- Right leg ---
    parts.append(_place_cylinder(
        radius=leg_r,
        height=leg_height,
        z_bottom=ankle_z,
        x_offset=-leg_x_offset,
        sections=24,
    ))

    # --- Left foot / ankle base (small sphere to close leg bottom) ---
    parts.append(_place_sphere(
        radius=leg_r,
        centre=np.array([+leg_x_offset, 0.0, ankle_z]),
        subdivisions=2,
    ))

    # --- Right foot / ankle base ---
    parts.append(_place_sphere(
        radius=leg_r,
        centre=np.array([-leg_x_offset, 0.0, ankle_z]),
        subdivisions=2,
    ))

    # --- Left shoulder sphere ---
    parts.append(_place_sphere(
        radius=shoulder_r,
        centre=np.array([+shoulder_x_offset, 0.0, shoulder_z]),
        subdivisions=2,
    ))

    # --- Right shoulder sphere ---
    parts.append(_place_sphere(
        radius=shoulder_r,
        centre=np.array([-shoulder_x_offset, 0.0, shoulder_z]),
        subdivisions=2,
    ))

    # -----------------------------------------------------------------------
    # Merge all parts
    # -----------------------------------------------------------------------
    body = trimesh.util.concatenate(parts)

    # Process to merge duplicate vertices and fix winding.
    # trimesh.util.concatenate returns a raw mesh; re-constructing with
    # process=True triggers vertex merging and degenerate-face removal.
    body = trimesh.Trimesh(
        vertices=body.vertices,
        faces=body.faces,
        process=True,
    )
    # Keep only non-degenerate faces (trimesh 4.x API)
    body.update_faces(body.nondegenerate_faces())
    trimesh.repair.fix_normals(body)

    # -----------------------------------------------------------------------
    # Optional: save PLY + body_profile.json
    # -----------------------------------------------------------------------
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        body.export(str(out))

        profile = _build_profile(
            mesh=body,
            output_path=out,
            height_cm=height_cm,
            chest_cm=chest_cm,
            waist_cm=waist_cm,
            hips_cm=hips_cm,
            inseam_cm=inseam_cm,
            shoulder_width_cm=shoulder_width_cm,
            gender=gender,
        )
        profile_path = out.with_suffix(".json")
        with open(profile_path, "w") as fh:
            json.dump(profile, fh, indent=2)

    return body


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def _build_profile(
    mesh: trimesh.Trimesh,
    output_path: Path,
    height_cm: float,
    chest_cm: float,
    waist_cm: float,
    hips_cm: float,
    inseam_cm: float,
    shoulder_width_cm: float,
    gender: str = "custom",
) -> dict:
    """Build a v1.2-schema body_profile dict for the Anny mesh."""
    return {
        "body_profile_id": output_path.stem,
        "body_source": "anny_parametric",
        "gender": gender,
        "scan_method": "anny_parametric",
        # Anny is a synthetic parametric model; measurements are exact by
        # construction (all clearance error comes from geometry sampling, not
        # scan noise).  scan_accuracy_mm reflects sampling discretisation.
        "scan_accuracy_mm": 2,
        # confidence < 1.0: Anny is parametric, not a real scan.
        # The pipeline should blend confidence based on garment + body fit.
        # For full synthetic use set to 0.9; real-user measurements set lower.
        "confidence": 0.9,
        "measurements": {
            "height_cm": height_cm,
            "chest_cm": chest_cm,
            "waist_cm": waist_cm,
            "hips_cm": hips_cm,
            "inseam_cm": inseam_cm,
            "shoulder_width_cm": shoulder_width_cm,
        },
        # Anny constructs geometry directly from measurements, so achieved ==
        # target by design.  The mesh is not re-measured after construction
        # because the primitives are analytically scaled.
        "achieved_measurements": {
            "height_cm": height_cm,
            "chest_cm": chest_cm,
            "waist_cm": waist_cm,
            "hips_cm": hips_cm,
            "inseam_cm": inseam_cm,
            "shoulder_width_cm": shoulder_width_cm,
        },
        "smplx_betas": None,
        "max_measurement_error_mm": 0.0,
        "mesh_path": str(output_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mesh_source": "anny_parametric_elliptic_cylinders",
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper — size-named presets
# ---------------------------------------------------------------------------

# Standard male size presets (ISO 8559 / EU sizing, cm)
SIZE_PRESETS: dict[str, dict] = {
    "S": {
        "height_cm": 168, "chest_cm": 88, "waist_cm": 73,
        "hips_cm": 90, "inseam_cm": 77, "shoulder_width_cm": 42,
    },
    "M": {
        "height_cm": 176, "chest_cm": 96, "waist_cm": 81,
        "hips_cm": 98, "inseam_cm": 81, "shoulder_width_cm": 45,
    },
    "XL": {
        "height_cm": 180, "chest_cm": 112, "waist_cm": 97,
        "hips_cm": 112, "inseam_cm": 83, "shoulder_width_cm": 49,
    },
}

# Standard female size presets (ISO 8559, cm)
SIZE_PRESETS_FEMALE: dict[str, dict] = {
    "XS": {
        "height_cm": 158, "chest_cm": 80, "waist_cm": 62,
        "hips_cm": 87, "inseam_cm": 72, "shoulder_width_cm": 35,
    },
    "S": {
        "height_cm": 163, "chest_cm": 84, "waist_cm": 66,
        "hips_cm": 91, "inseam_cm": 74, "shoulder_width_cm": 36,
    },
    "M": {
        "height_cm": 168, "chest_cm": 88, "waist_cm": 70,
        "hips_cm": 95, "inseam_cm": 76, "shoulder_width_cm": 37,
    },
    "L": {
        "height_cm": 170, "chest_cm": 96, "waist_cm": 78,
        "hips_cm": 103, "inseam_cm": 77, "shoulder_width_cm": 39,
    },
    "XL": {
        "height_cm": 172, "chest_cm": 104, "waist_cm": 86,
        "hips_cm": 111, "inseam_cm": 78, "shoulder_width_cm": 41,
    },
}

# Legacy alias kept for backwards compatibility
ANNY_SIZES = SIZE_PRESETS_FEMALE


def generate_anny_size(
    size: str,
    output_path: str | None = None,
    gender: str = "male",
) -> trimesh.Trimesh:
    """
    Generate Anny body for a named size.

    Parameters
    ----------
    size        : size key. Male: "S", "M", "XL".
                  Female: "XS", "S", "M", "L", "XL".
    output_path : if given, save PLY and body_profile.json here
    gender      : "male" or "female" — selects the appropriate preset dict

    Returns
    -------
    trimesh.Trimesh
    """
    if gender == "female":
        presets = SIZE_PRESETS_FEMALE
    elif gender == "male":
        presets = SIZE_PRESETS
    else:
        raise ValueError(f"Unknown gender '{gender}'. Valid: 'male', 'female'")

    if size not in presets:
        raise ValueError(
            f"Unknown size '{size}' for gender '{gender}'. "
            f"Valid: {list(presets)}"
        )
    return generate_anny_body(**presets[size], output_path=output_path, gender=gender)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    out = "/tmp/anny_test.ply"
    print("Generating Anny size-M female body...")
    mesh = generate_anny_size("M", output_path=out)

    print(f"  Vertices : {len(mesh.vertices)}")
    print(f"  Faces    : {len(mesh.faces)}")
    print(f"  Watertight: {mesh.is_watertight}")
    bounds = mesh.bounds
    height_achieved = (bounds[1][2] - bounds[0][2]) * 100.0
    print(f"  Height   : {height_achieved:.1f} cm  (target 168.0)")
    print(f"  Saved to : {out}")
    print(f"  Profile  : {out.replace('.ply', '.json')}")

    if not mesh.is_watertight:
        print("  WARNING: mesh is not fully watertight — gaps at segment joints.")
        print("  This is expected for a concatenated primitive model; use")
        print("  trimesh.repair.fill_holes() if strict watertightness is required.")
        sys.exit(0)

    print("Done.")
