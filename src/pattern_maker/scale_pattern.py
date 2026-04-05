"""
Pattern Maker — Pattern Scaling

Scales a GarmentCode pattern JSON by a uniform factor.
Multiplies all vertex coordinates and translation values by scale_factor.
Does NOT scale rotation values.
Edge curvature control points are relative (fractions), so they don't need scaling.

scale_pattern_to_measurements() bridges body measurements → scale_factor by
comparing chest_cm against the pattern's reference_measurements (fallback: 88 cm).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

__all__ = ["scale_pattern", "scale_pattern_to_measurements"]

# Default M-size reference chest used when the pattern JSON lacks
# a reference_measurements block (all current tshirt patterns).
_DEFAULT_REFERENCE_CHEST_CM = 88.0


def scale_pattern(input_path: str | Path, output_path: str | Path, scale_factor: float) -> None:
    """
    Scale a GarmentCode pattern by scale_factor.

    Multiplies all vertex coordinates and translation values by scale_factor.
    Does NOT scale rotation values.
    Edge curvature control points are relative (fractions), so they don't need scaling.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path) as f:
        pattern = json.load(f)

    scaled = copy.deepcopy(pattern)

    for panel_name, panel in scaled["pattern"]["panels"].items():
        # Scale vertex coordinates
        panel["vertices"] = [
            [v[0] * scale_factor, v[1] * scale_factor]
            for v in panel["vertices"]
        ]
        # Scale translation (X, Y, Z all scaled)
        panel["translation"] = [
            t * scale_factor for t in panel["translation"]
        ]
        # Do NOT scale rotation — angles are invariant
        # Do NOT scale curvature params — they are relative fractions

    # Update metadata
    source_meta = pattern.get("_forma_metadata", {})
    source_scale = source_meta.get("scale_factor", 1.0)
    source_size = source_meta.get("size", "M")

    # Infer new size from output filename (e.g. "tshirt_size_XS" → "XS")
    stem = output_path.stem
    new_size = stem.split("_")[-1] if "_" in stem else "scaled"

    # Derive garment_id prefix from source metadata or input filename rather
    # than hardcoding "tshirt_gc_v1".  This lets scale_pattern() work for any
    # garment type (shirt, trouser, dress, …).
    source_garment_id = source_meta.get("garment_id", input_path.stem)
    # Strip any trailing size token from the source garment_id
    # e.g. "tshirt_gc_v1_size_M" → "tshirt_gc_v1_size"
    # We want the prefix up to (but not including) the old size value.
    if source_garment_id.endswith(f"_{source_size}"):
        garment_id_prefix = source_garment_id[: -len(f"_{source_size}")]
    else:
        garment_id_prefix = source_garment_id

    # Similarly, derive the scaled_from filename from the actual input file.
    scaled_from_filename = input_path.name

    scaled["_forma_metadata"] = {
        "garment_id": f"{garment_id_prefix}_{new_size}",
        "size": new_size,
        "scale_factor": round(source_scale * scale_factor, 6),
        "source": source_meta.get("source", "unknown"),
        "scaled_from": scaled_from_filename,
        "relative_scale": scale_factor,
        "fabric_id": source_meta.get("fabric_id", "cotton_jersey_default"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scaled, f, indent=2)

    return scaled.get("_forma_metadata", {})


def scale_pattern_to_measurements(
    input_path: str,
    output_path: str,
    chest_cm: float,
    waist_cm: float = None,   # optional, used for waist-dominant garments
    inseam_cm: float = None,  # optional, used for trousers  # noqa: ARG001
) -> dict:
    """
    Scale a pattern to fit target body measurements.

    Derives scale_factor from chest_cm vs the pattern's reference chest measurement.
    Falls back to waist_cm or a 1.0 factor if reference measurements not found.

    Returns the metadata dict from the scaled pattern.
    """
    input_path = Path(input_path)

    with open(input_path) as f:
        pattern = json.load(f)

    ref = pattern.get("_forma_metadata", {}).get("reference_measurements", {})

    if ref.get("chest_cm"):
        scale_factor = chest_cm / ref["chest_cm"]
    elif waist_cm is not None and ref.get("waist_cm"):
        scale_factor = waist_cm / ref["waist_cm"]
    else:
        scale_factor = chest_cm / _DEFAULT_REFERENCE_CHEST_CM

    return scale_pattern(input_path, output_path, scale_factor)
