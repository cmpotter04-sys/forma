"""
src/pipeline.py — AC-5

Single entry point for the Forma fit-check pipeline.

    run_fit_check(body, pattern, manifest, fabric_id) → fit_verdict dict
    run_batch_fit_check(body, patterns, manifests, fabric_id) → [fit_verdict, ...]

All inputs are file paths. Returns v1.2 schema verdict dicts.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geometer.xpbd_simulate import run_simulation
from geometer.convergence import SimulationExplosionError
from tailor.seam_converter import SeamValidationError
from verdict.generate_verdict import generate_verdict

FABRIC_LIBRARY_PATH = Path(__file__).parent.parent / "data" / "fabrics" / "fabric_library.json"


def _validate_path(path: str | Path, label: str) -> Path:
    """Raise FileNotFoundError if the path does not exist."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _load_fabric_params(fabric_id: str, library_path: Path | None = None) -> dict:
    """
    Load fabric parameters from fabric_library.json.

    Raises ValueError if fabric_id is not in the library.
    """
    lib_path = library_path or FABRIC_LIBRARY_PATH
    with open(lib_path) as f:
        library = json.load(f)
    if fabric_id not in library["fabrics"]:
        raise ValueError(
            f"Unknown fabric_id: {fabric_id!r}. "
            f"Available: {list(library['fabrics'].keys())}"
        )
    return library["fabrics"][fabric_id]


def run_fit_check(
    body_mesh_path: str = None,
    pattern_path: str = None,
    seam_manifest_path: str = None,
    fabric_id: str = "cotton_jersey_default",
    fabric_library_path: str | None = None,
    backend: str = "cpu",
    subdivide_target: int = 0,
    anny_measurements: dict = None,
) -> dict:
    """
    Full pipeline: body + pattern + fabric → fit_verdict dict (v1.2 schema).

    Parameters
    ----------
    body_mesh_path      : path to pre-built body mesh PLY file.
                          Mutually exclusive with anny_measurements.
    pattern_path        : path to GarmentCode JSON pattern
    seam_manifest_path  : path to seam_manifest.json
    fabric_id           : key into fabric_library.json (default: cotton_jersey_default)
    fabric_library_path : override for fabric_library.json location (optional)
    backend             : simulation backend — "cpu" (Phase 1 XPBD), "warp" (NVIDIA Warp GPU),
                          or "hood" (ContourCraft/HOOD GPU — requires CUDA + pretrained weights)
    subdivide_target    : if > 0, subdivide garment mesh to this vertex count
                          after assembly (default: 0 = no subdivision)
    anny_measurements   : if given, generate body on the fly via generate_anny_body().
                          Mutually exclusive with body_mesh_path. Expected keys:
                            height_cm, chest_cm, waist_cm, hips_cm,
                            inseam_cm, shoulder_width_cm  (all float, in cm)

    Returns
    -------
    dict : complete fit_verdict.json v1.2 document

    Raises
    ------
    ValueError              — if both or neither of body_mesh_path / anny_measurements are given,
                              if fabric_id not found in fabric library, or invalid backend
    FileNotFoundError       — if body_mesh_path, pattern_path, or seam_manifest_path don't exist
    SeamValidationError     — if seam manifest fails validation
    SimulationExplosionError — if solver diverges
    RuntimeError            — if tunnel-through exceeds limit or other simulation failure
    ImportError             — if backend="warp" but warp-lang is not installed,
                              or backend="hood" but ContourCraft / CUDA not available
    """
    if backend not in ("cpu", "warp", "hood"):
        raise ValueError(f"Unknown backend: {backend!r}. Must be 'cpu', 'warp', or 'hood'.")

    # 0. Resolve body source — exactly one of body_mesh_path / anny_measurements required
    if anny_measurements is not None and body_mesh_path is not None:
        raise ValueError(
            "Provide either body_mesh_path or anny_measurements, not both."
        )
    if anny_measurements is None and body_mesh_path is None:
        raise ValueError(
            "One of body_mesh_path or anny_measurements must be provided."
        )

    _tmp_file = None  # holds the NamedTemporaryFile so it stays alive during simulation
    body_source_override = None

    if anny_measurements is not None:
        from sculptor.anny_body import generate_anny_body
        import numpy as np
        import trimesh as _trimesh
        _tmp_file = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        _tmp_file.close()  # close so generate_anny_body can write to it on all platforms
        anny_mesh = generate_anny_body(**anny_measurements)

        # Anny uses Z-up (X=left-right, Y=forward, Z=height). The rest of the
        # pipeline (garment_assembly, xpbd_simulate) assumes Y-up (X=left-right,
        # Y=height, Z=forward).  Swap Y and Z axes here so the saved PLY is
        # already in the expected coordinate frame.
        v = np.array(anny_mesh.vertices)
        v_yup = v[:, [0, 2, 1]]      # X unchanged; new Y = old Z; new Z = old Y
        mesh_yup = _trimesh.Trimesh(vertices=v_yup, faces=anny_mesh.faces, process=True)
        _trimesh.repair.fix_normals(mesh_yup)
        mesh_yup.export(_tmp_file.name)

        body_mesh_path = _tmp_file.name
        body_source_override = "anny_parametric"

    # 1. Validate inputs exist
    body_path = _validate_path(body_mesh_path, "Body mesh")
    pat_path = _validate_path(pattern_path, "Pattern")
    manifest_path = _validate_path(seam_manifest_path, "Seam manifest")

    # 2. Load fabric params (raises ValueError if unknown)
    lib_path = Path(fabric_library_path) if fabric_library_path else None
    fabric_params = _load_fabric_params(fabric_id, lib_path)

    # 3. Run simulation
    if backend == "warp":
        from geometer.warp.warp_simulate import run_simulation_warp
        sim_result = run_simulation_warp(
            str(body_path),
            str(pat_path),
            str(manifest_path),
            fabric_params,
            subdivide_target=subdivide_target,
        )
    elif backend == "hood":
        from geometer.hood.hood_simulate import run_simulation_hood
        sim_result = run_simulation_hood(
            str(body_path),
            str(pat_path),
            str(manifest_path),
            fabric_params,
            subdivide_target=subdivide_target,
        )
    else:
        sim_result = run_simulation(
            str(body_path),
            str(pat_path),
            str(manifest_path),
            fabric_params,
            subdivide_target=subdivide_target,
        )

    # 4. Derive garment_id and body_profile_id from filenames
    garment_id = pat_path.stem  # e.g. "tshirt_size_M"
    body_profile_id = body_path.stem  # e.g. "makehuman_male_M"

    # 5. Generate verdict
    verdict = generate_verdict(sim_result, garment_id, body_profile_id, fabric_id)

    # 6. Override body_source when body was generated from Anny measurements
    if body_source_override is not None:
        verdict["body_source"] = body_source_override

    return verdict


def run_batch_fit_check(
    body_mesh_path: str,
    pattern_paths: list[str],
    seam_manifest_paths: list[str],
    fabric_id: str = "cotton_jersey_default",
    fabric_library_path: str | None = None,
    backend: str = "cpu",
    subdivide_target: int = 0,
) -> list[dict]:
    """
    Run fit checks for multiple garments on the same body.

    The body mesh is validated once. Each pattern+manifest pair is run
    sequentially, reusing the same fabric parameters.

    Parameters
    ----------
    body_mesh_path      : path to body mesh PLY file
    pattern_paths       : list of GarmentCode JSON pattern paths
    seam_manifest_paths : list of seam_manifest.json paths (same order as pattern_paths)
    fabric_id           : key into fabric_library.json
    fabric_library_path : override for fabric_library.json location (optional)
    backend             : simulation backend — "cpu", "warp", or "hood"
    subdivide_target    : if > 0, subdivide garment mesh to this vertex count (default: 0)

    Returns
    -------
    list[dict] : verdict dicts in the same order as pattern_paths
    """
    if len(pattern_paths) != len(seam_manifest_paths):
        raise ValueError(
            f"pattern_paths ({len(pattern_paths)}) and seam_manifest_paths "
            f"({len(seam_manifest_paths)}) must have the same length"
        )

    # Validate body path once
    _validate_path(body_mesh_path, "Body mesh")

    verdicts = []
    for pat, manifest in zip(pattern_paths, seam_manifest_paths):
        verdict = run_fit_check(
            body_mesh_path, pat, manifest,
            fabric_id=fabric_id,
            fabric_library_path=fabric_library_path,
            backend=backend,
            subdivide_target=subdivide_target,
        )
        verdicts.append(verdict)

    return verdicts
