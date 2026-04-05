"""
src/sculptor/makehuman_body.py

Generate sized body meshes from the MakeHuman CC0 base mesh.

The MakeHuman base mesh (21,833 vertices, CC0 license) is loaded, the
caucasian-male-young macro morph is applied for male proportions, then
height-dependent XZ scaling deforms the torso to hit target circumferences.

Circumference is measured using convex hull perimeter on torso-only vertices
(arm vertices excluded via |X| threshold derived from the arm-torso gap).

Usage:
    python src/sculptor/makehuman_body.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
BASE_OBJ = ROOT / "data" / "bodies" / "makehuman_base.obj"
TARGETS_NPZ = ROOT / "data" / "bodies" / "makehuman_targets.npz"

MH_UNIT_SCALE = 0.1  # MakeHuman decimetres → metres


# ---------------------------------------------------------------------------
# Morph target helpers
# ---------------------------------------------------------------------------

def _apply_target(
    vertices: np.ndarray,
    targets: dict,
    target_name: str,
    weight: float = 1.0,
) -> None:
    """Apply a morph target to vertices IN-PLACE (in MakeHuman dm space)."""
    idx = targets[target_name + ".index"].astype(int)
    vec = targets[target_name + ".vector"].astype(float) / 1000.0
    vertices[idx] += vec * weight


# ---------------------------------------------------------------------------
# Circumference measurement
# ---------------------------------------------------------------------------

def _find_arm_gap_x(vertices: np.ndarray, y_height: float, band: float = 0.015) -> float:
    """
    Find the X-coordinate where the arm separates from the torso at a given
    height. Returns the X threshold for torso-only filtering.
    """
    mask = np.abs(vertices[:, 1] - y_height) < band
    if np.sum(mask) < 10:
        return 0.20  # fallback
    abs_x = np.sort(np.abs(vertices[mask, 0]))
    gaps = np.diff(abs_x)
    # Find the first large gap (> 3cm) after X > 0.10
    for i, g in enumerate(gaps):
        if abs_x[i] > 0.10 and g > 0.03:
            return float(abs_x[i]) + 0.005  # just inside the gap
    return 0.20  # fallback


def measure_circumference(
    vertices: np.ndarray,
    y_height: float,
    band: float = 0.015,
    x_limit: float | None = None,
) -> float:
    """
    Measure torso circumference at y_height using convex hull perimeter.

    Arm vertices are excluded via |X| < x_limit (auto-detected if None).
    Returns circumference in centimetres.
    """
    if x_limit is None:
        x_limit = _find_arm_gap_x(vertices, y_height, band)

    mask = (np.abs(vertices[:, 1] - y_height) < band) & (np.abs(vertices[:, 0]) < x_limit)
    pts = vertices[mask][:, [0, 2]]

    if len(pts) < 3:
        return 0.0

    hull = ConvexHull(pts)
    hp = pts[hull.vertices]
    hpc = np.vstack([hp, hp[0]])
    return float(np.sum(np.linalg.norm(np.diff(hpc, axis=0), axis=1))) * 100.0


def _scan_for_landmark(
    vertices: np.ndarray,
    y_lo: float,
    y_hi: float,
    maximize: bool = True,
) -> tuple[float, float]:
    """
    Scan heights to find the one with max (or min) circumference.
    Returns (best_y, circ_cm).
    """
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
# Programmatic XZ scaling
# ---------------------------------------------------------------------------

def _scale_torso_xz(
    vertices: np.ndarray,
    landmarks: list[tuple[float, float, float]],
) -> None:
    """
    Apply height-dependent XZ scaling to hit target circumferences.

    landmarks: list of (y_height, current_circ_cm, target_circ_cm).
    Vertices near each landmark height get an XZ scale = target/current.
    Scale is interpolated between landmarks and held constant outside.
    Only torso vertices are scaled; arm vertices are left untouched.
    """
    if not landmarks:
        return

    landmarks = sorted(landmarks, key=lambda t: t[0])

    ys = np.array([l[0] for l in landmarks])
    scales = np.array([l[2] / l[1] if l[1] > 1.0 else 1.0 for l in landmarks])

    # Per-vertex interpolated scale factor based on Y height
    vertex_scales = np.interp(vertices[:, 1], ys, scales)

    # Per-vertex arm gap: compute at each landmark height, interpolate
    arm_gaps = np.array([_find_arm_gap_x(vertices, y) for y in ys])
    vertex_arm_gaps = np.interp(vertices[:, 1], ys, arm_gaps)

    # Blend: full scale (1.0) for |X| < 95% of arm gap,
    # linear taper to no-scale (0.0) from 95% to 105% of arm gap
    abs_x = np.abs(vertices[:, 0])
    inner = vertex_arm_gaps * 0.95
    outer = vertex_arm_gaps * 1.05
    # blend = 1.0 when abs_x <= inner, 0.0 when abs_x >= outer, linear between
    blend = np.clip((outer - abs_x) / (outer - inner + 1e-6), 0.0, 1.0)
    effective_scale = 1.0 + (vertex_scales - 1.0) * blend

    x_center = 0.0
    z_center = float(np.median(vertices[:, 2]))

    vertices[:, 0] = x_center + (vertices[:, 0] - x_center) * effective_scale
    vertices[:, 2] = z_center + (vertices[:, 2] - z_center) * effective_scale


# ---------------------------------------------------------------------------
# Body generation
# ---------------------------------------------------------------------------

# Landmark height bands (fraction of total height)
_BANDS = {
    "chest": (0.67, 0.76),   # ~1.20-1.37m for 180cm
    "waist": (0.56, 0.63),   # ~1.00-1.13m for 180cm
    "hip":   (0.48, 0.56),   # ~0.86-1.00m for 180cm
}


def generate_body(
    chest_cm: float,
    waist_cm: float,
    hip_cm: float,
    height_cm: float,
    output_path: str | Path,
    shoulder_width_cm: float = 44.0,
    inseam_cm: float = 82.0,
    gender: str = "male",
) -> dict:
    """
    Generate a MakeHuman body mesh at the given measurements.

    1. Load base.obj
    2. Apply caucasian-male-young morph
    3. Scale to target height, recenter
    4. Measure current torso circumferences at chest/waist/hip
    5. Apply height-dependent XZ scaling to hit targets
    6. Re-measure and iterate if needed
    7. Export as PLY + body_profile JSON

    Returns body_profile dict (v1.2 schema).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load base mesh
    mesh = trimesh.load(str(BASE_OBJ), process=False)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    # 2. Apply gender macro morph (caucasian-{gender}-young)
    targets = np.load(str(TARGETS_NPZ), allow_pickle=True)
    _apply_target(vertices, targets, f"targets/macrodetails/caucasian-{gender}-young", 1.0)

    # 3. Convert dm → m, scale to target height, recenter
    vertices *= MH_UNIT_SCALE
    current_h = float(vertices[:, 1].max() - vertices[:, 1].min())
    vertices *= (height_cm / 100.0) / current_h
    vertices[:, 1] -= float(vertices[:, 1].min())
    vertices[:, 0] -= float(vertices[:, 0].mean())
    vertices[:, 2] -= float(vertices[:, 2].mean())

    height_m = height_cm / 100.0
    targets_map = {"chest": chest_cm, "waist": waist_cm, "hip": hip_cm}

    # 4-5. Iterate: measure → scale → re-measure (up to 8 rounds)
    for iteration in range(8):
        landmarks = []
        for region, target_cm in targets_map.items():
            frac_lo, frac_hi = _BANDS[region]
            maximize = region != "waist"
            best_y, current_circ = _scan_for_landmark(
                vertices, frac_lo * height_m, frac_hi * height_m, maximize=maximize,
            )
            landmarks.append((best_y, current_circ, target_cm))

        # Check if all within tolerance
        max_err = max(abs(l[1] - l[2]) for l in landmarks)
        if max_err <= 2.0:
            break

        _scale_torso_xz(vertices, landmarks)

    # 6. Final measurements
    achieved_circs = {}
    for region, target_cm in targets_map.items():
        frac_lo, frac_hi = _BANDS[region]
        maximize = region != "waist"
        _, circ = _scan_for_landmark(
            vertices, frac_lo * height_m, frac_hi * height_m, maximize=maximize,
        )
        achieved_circs[region] = circ

    # Final height
    final_h_cm = float(vertices[:, 1].max() - vertices[:, 1].min()) * 100.0

    # 7. Export PLY
    out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out_mesh.export(str(output_path))

    # 8. Build profile
    achieved = {
        "height_cm": round(final_h_cm, 2),
        "chest_cm": round(achieved_circs["chest"], 2),
        "waist_cm": round(achieved_circs["waist"], 2),
        "hip_cm": round(achieved_circs["hip"], 2),
        "shoulder_width_cm": shoulder_width_cm,
        "inseam_cm": inseam_cm,
    }

    max_error_mm = max(
        abs(achieved["chest_cm"] - chest_cm) * 10.0,
        abs(achieved["waist_cm"] - waist_cm) * 10.0,
        abs(achieved["hip_cm"] - hip_cm) * 10.0,
        abs(achieved["height_cm"] - height_cm) * 10.0,
    )

    profile = {
        "body_profile_id": output_path.stem,
        "body_source": "synthetic_mannequin",
        "scan_method": "synthetic_mannequin",
        "scan_accuracy_mm": 0,
        "confidence": 1.0,
        "measurements": {
            "height_cm": height_cm,
            "chest_cm": chest_cm,
            "waist_cm": waist_cm,
            "hip_cm": hip_cm,
            "shoulder_width_cm": shoulder_width_cm,
            "inseam_cm": inseam_cm,
        },
        "achieved_measurements": achieved,
        "smplx_betas": None,
        "max_measurement_error_mm": round(max_error_mm, 3),
        "mesh_path": str(output_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mesh_source": "makehuman_cc0_base_with_xz_scaling",
        "vertex_count": len(vertices),
        "face_count": len(faces),
    }

    profile_path = output_path.with_suffix(".json")
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    return profile


# ---------------------------------------------------------------------------
# Generate all three body sizes
# ---------------------------------------------------------------------------

BODY_SIZES = {
    "S":  {"chest_cm": 88,  "waist_cm": 72, "hip_cm": 87,  "height_cm": 170},
    "M":  {"chest_cm": 96,  "waist_cm": 80, "hip_cm": 95,  "height_cm": 180},
    "XL": {"chest_cm": 108, "waist_cm": 96, "hip_cm": 105, "height_cm": 185},
}

# ISO 8559 female reference measurements
BODY_SIZES_FEMALE = {
    "S":  {"chest_cm": 84,  "waist_cm": 66, "hip_cm": 90,  "height_cm": 163},
    "M":  {"chest_cm": 92,  "waist_cm": 74, "hip_cm": 98,  "height_cm": 167},
    "XL": {"chest_cm": 104, "waist_cm": 86, "hip_cm": 110, "height_cm": 172},
}


def generate_all_bodies(output_dir: Path | None = None, include_female: bool = True) -> dict[str, dict]:
    """Generate S, M, XL body meshes for male (and optionally female). Returns dict of key → profile."""
    if output_dir is None:
        output_dir = ROOT / "data" / "bodies"

    results = {}
    sizes_by_gender = [("male", BODY_SIZES), ("female", BODY_SIZES_FEMALE)] if include_female else [("male", BODY_SIZES)]

    for gender, sizes in sizes_by_gender:
        for size, params in sizes.items():
            output_path = output_dir / f"makehuman_{gender}_{size}.ply"
            print(f"\n{'='*60}")
            print(f"  Generating {gender} size {size}: {params}")
            print(f"{'='*60}")

            profile = generate_body(output_path=output_path, gender=gender, **params)

            ach = profile["achieved_measurements"]
            print(f"  Height:  {ach['height_cm']:.2f} cm  (target {params['height_cm']})")
            print(f"  Chest:   {ach['chest_cm']:.2f} cm  (target {params['chest_cm']})")
            print(f"  Waist:   {ach['waist_cm']:.2f} cm  (target {params['waist_cm']})")
            print(f"  Hip:     {ach['hip_cm']:.2f} cm  (target {params['hip_cm']})")
            print(f"  Max err: {profile['max_measurement_error_mm']:.1f} mm")
            print(f"  Saved:   {output_path}")

            results[f"{gender}_{size}"] = profile

    return results


if __name__ == "__main__":
    generate_all_bodies()
