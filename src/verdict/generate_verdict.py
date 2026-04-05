"""
Verdict Generator — AC-5

Assembles fit_verdict.json (v1.2 schema) from Geometer simulation results.
All clearance values come from the Geometer pipeline — nothing is hardcoded.

Entry point:
    generate_verdict(sim_result, garment_id, body_profile_id, fabric_id) -> dict

Writes to:
    output/verdicts/vrd_{size}_on_M.json

Schema: FORMA_WEEK1_SPEC.md § "Output Schema — fit_verdict.json (v1.2)"
"""

from __future__ import annotations
import json
import sys
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from geometer.clearance import classify_ease

_REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = _REPO_ROOT / "output" / "verdicts"
FABRIC_LIBRARY_PATH = _REPO_ROOT / "data" / "fabrics" / "fabric_library.json"

# Severity thresholds (from FORMA_WEEK1_SPEC.md)
# delta_mm > -10  → green
# -25 <= delta_mm <= -10 → yellow
# delta_mm < -25  → red
_SEVERITY_THRESHOLDS = [
    (-10.0, "green"),
    (-25.0, "yellow"),
]


def _severity(delta_mm: float, median_strain_ratio: float | None = None) -> str:
    """Classify signed clearance into green / yellow / red (with strain ratio)."""
    sr = median_strain_ratio if median_strain_ratio is not None else 1.0
    if delta_mm <= -25.0 or sr > 1.15:
        return "red"
    elif delta_mm <= -10.0 or sr > 1.08:
        return "yellow"
    else:
        return "green"



def _load_fabric(fabric_id: str) -> dict:
    """
    Load fabric parameters from fabric_library.json.
    Raises ValueError for unknown fabric_id. Never use magic numbers.
    """
    with open(FABRIC_LIBRARY_PATH) as f:
        library = json.load(f)
    if fabric_id not in library["fabrics"]:
        raise ValueError(
            f"Unknown fabric_id: {fabric_id!r}. "
            f"Available: {list(library['fabrics'].keys())}"
        )
    return library["fabrics"][fabric_id]


_VALID_BODY_SOURCES = {"synthetic_mannequin", "standard_photo", "precision_suit"}


def generate_verdict(
    sim_result: dict,
    garment_id: str,
    body_profile_id: str,
    fabric_id: str = "cotton_jersey_default",
    body_source: str = "synthetic_mannequin",
    confidence: Optional[float] = None,
) -> dict:
    """
    Assemble a fit_verdict.json v1.2 dict from simulation results.

    Parameters
    ----------
    sim_result : dict
        Output from the Geometer pipeline. Expected keys:
          clearance_map: dict[region_name → delta_mm (float)]
            Signed clearance per region. Negative = too tight, positive = ease.
          simulation_ms: int
          convergence_step: int
          final_kinetic_energy_j: float
          tunnel_through_pct: float

    garment_id : str
        e.g. "tshirt_gc_v1_size_S"

    body_profile_id : str
        e.g. "mannequin_sizeM_180cm"

    fabric_id : str
        Key into fabric_library.json. Default: cotton_jersey_default.

    body_source : str
        One of "synthetic_mannequin", "standard_photo", "precision_suit".
        REQUIRED — must be set explicitly for non-synthetic bodies.
        (CLAUDE.md rule #3)

    confidence : float | None
        Caller-supplied confidence score. Must be None (auto) or in [0.0, 1.0].
        Auto-rules (CLAUDE.md rule #1):
          - synthetic_mannequin → always 1.0
          - all other sources    → caller MUST supply a value < 1.0;
                                   passing confidence=1.0 raises ValueError.

    Returns
    -------
    dict : complete fit_verdict.json v1.2 document
    """
    _validate_sim_result(sim_result)

    # --- body_source validation (rule #3) ---
    if body_source not in _VALID_BODY_SOURCES:
        raise ValueError(
            f"body_source must be one of {sorted(_VALID_BODY_SOURCES)}, "
            f"got {body_source!r}"
        )

    # --- confidence enforcement (rule #1) ---
    if body_source == "synthetic_mannequin":
        if confidence is not None and confidence != 1.0:
            raise ValueError(
                "confidence must be 1.0 (or omitted) for synthetic_mannequin; "
                f"got {confidence!r}"
            )
        resolved_confidence = 1.0
    else:
        # Non-synthetic body: caller must supply confidence and it must be < 1.0
        if confidence is None:
            raise ValueError(
                f"confidence is required for body_source={body_source!r}. "
                "Pass a float in [0.0, 1.0) derived from scan accuracy."
            )
        if confidence == 1.0:
            raise ValueError(
                f"confidence=1.0 is only valid for synthetic_mannequin, "
                f"not for body_source={body_source!r} (CLAUDE.md rule #1)"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {confidence!r}"
            )
        resolved_confidence = float(confidence)

    clearance_map: dict[str, float] = sim_result["clearance_map"]
    strain_ratio_map: dict[str, float] = sim_result.get("strain_ratio_map", {})
    fabric_params = _load_fabric(fabric_id)

    # Build strain_map (one entry per region, now includes median_strain_ratio)
    strain_map = []
    for region, delta_mm in clearance_map.items():
        sr = strain_ratio_map.get(region)
        entry = {
            "region": region,
            "delta_mm": round(float(delta_mm), 3),
            "severity": _severity(float(delta_mm), sr),
        }
        if sr is not None:
            entry["median_strain_ratio"] = round(sr, 4)
        strain_map.append(entry)

    # Build ease_map (same regions, same delta_mm reinterpreted as ease)
    ease_map = []
    for region, delta_mm in clearance_map.items():
        excess_mm, ease_label = classify_ease(float(delta_mm))
        ease_map.append({
            "region": region,
            "excess_mm": round(excess_mm, 3),
            "verdict": ease_label,
        })

    # fit = True iff zero regions are red (non-negotiable rule)
    fit = all(e["severity"] != "red" for e in strain_map)

    verdict = {
        "verdict_id": f"vrd_{uuid.uuid4().hex[:12]}",
        "fit": fit,
        "confidence": resolved_confidence,
        "body_source": body_source,
        "scan_method": body_source,
        "scan_accuracy_mm": 0,
        "garment_id": garment_id,
        "body_profile_id": body_profile_id,
        "strain_map": strain_map,
        "ease_map": ease_map,
        "simulation_ms": int(sim_result["simulation_ms"]),
        "convergence_step": int(sim_result["convergence_step"]),
        "final_kinetic_energy_j": float(sim_result["final_kinetic_energy_j"]),
        "tunnel_through_pct": float(sim_result["tunnel_through_pct"]),
        "fabric_params_used": {
            "fabric_id": fabric_params["fabric_id"],
            "type": fabric_params["type"],
            "density_kg_m2": fabric_params["density_kg_m2"],
            "stretch_stiffness": fabric_params["stretch_stiffness"],
            "bend_stiffness": fabric_params["bend_stiffness"],
            "shear_stiffness": fabric_params["shear_stiffness"],
            "damping": fabric_params["damping"],
        },
    }

    return verdict


def save_verdict(verdict: dict, filename: str) -> Path:
    """
    Write verdict dict to output/verdicts/<filename>.
    Creates directory if needed. Returns the output path.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2)
    return out_path


def generate_and_save(
    sim_result: dict,
    garment_id: str,
    body_profile_id: str,
    output_filename: str,
    fabric_id: str = "cotton_jersey_default",
) -> tuple[dict, Path]:
    """
    Convenience wrapper: generate + save in one call.
    Returns (verdict_dict, output_path).
    """
    verdict = generate_verdict(sim_result, garment_id, body_profile_id, fabric_id)
    path = save_verdict(verdict, output_filename)
    return verdict, path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_SIM_FIELDS = [
    "clearance_map", "simulation_ms", "convergence_step",
    "final_kinetic_energy_j", "tunnel_through_pct",
]

REQUIRED_REGIONS = [
    "chest_front", "chest_side", "shoulder_left",
    "shoulder_right", "upper_back", "waist",
]


def _validate_sim_result(sim_result: dict) -> None:
    """Raise ValueError if the sim_result is missing required fields."""
    for field in REQUIRED_SIM_FIELDS:
        if field not in sim_result:
            raise ValueError(f"sim_result missing required field: {field!r}")

    clearance_map = sim_result["clearance_map"]
    if not isinstance(clearance_map, dict):
        raise ValueError("sim_result['clearance_map'] must be a dict")

    missing_regions = [r for r in REQUIRED_REGIONS if r not in clearance_map]
    if missing_regions:
        raise ValueError(
            f"clearance_map missing required regions: {missing_regions}"
        )

    for region, delta in clearance_map.items():
        if not isinstance(delta, (int, float)):
            raise ValueError(
                f"clearance_map[{region!r}] must be a float, got {type(delta)}"
            )


def validate_verdict_schema(verdict: dict) -> list[str]:
    """
    Validate a verdict dict against the v1.2 schema.
    Returns list of error strings (empty = valid).
    """
    errors = []

    required_top = [
        "verdict_id", "fit", "confidence", "body_source",
        "scan_method", "scan_accuracy_mm", "garment_id",
        "body_profile_id", "strain_map", "ease_map",
        "simulation_ms", "convergence_step", "final_kinetic_energy_j",
        "tunnel_through_pct", "fabric_params_used",
    ]
    for field in required_top:
        if field not in verdict:
            errors.append(f"Missing top-level field: {field!r}")

    # verdict_id format: vrd_{12 hex chars}
    vid = verdict.get("verdict_id", "")
    if not vid.startswith("vrd_"):
        errors.append(f"verdict_id must start with 'vrd_': {vid!r}")
    else:
        hex_part = vid[4:]
        if len(hex_part) != 12:
            errors.append(
                f"verdict_id hex part must be 12 chars, got {len(hex_part)}: {vid!r}"
            )
        else:
            try:
                int(hex_part, 16)
            except ValueError:
                errors.append(f"verdict_id hex part is not valid hex: {hex_part!r}")

    # body_source enum
    valid_body_sources = ["synthetic_mannequin", "standard_photo", "precision_suit"]
    if verdict.get("body_source") not in valid_body_sources:
        errors.append(
            f"body_source must be one of {valid_body_sources}, "
            f"got {verdict.get('body_source')!r}"
        )

    # strain_map regions
    if "strain_map" in verdict:
        regions_present = {r["region"] for r in verdict["strain_map"]}
        for req in REQUIRED_REGIONS:
            if req not in regions_present:
                errors.append(f"strain_map missing region: {req!r}")
        valid_severities = {"green", "yellow", "red"}
        for r in verdict["strain_map"]:
            if r.get("severity") not in valid_severities:
                errors.append(
                    f"strain_map region {r.get('region')!r}: "
                    f"invalid severity {r.get('severity')!r}"
                )

    # ease_map regions
    if "ease_map" in verdict:
        regions_present = {r["region"] for r in verdict["ease_map"]}
        for req in REQUIRED_REGIONS:
            if req not in regions_present:
                errors.append(f"ease_map missing region: {req!r}")
        valid_verdicts = {"tight_fit", "standard_fit", "relaxed_fit", "oversized"}
        for r in verdict["ease_map"]:
            if r.get("verdict") not in valid_verdicts:
                errors.append(
                    f"ease_map region {r.get('region')!r}: "
                    f"invalid verdict {r.get('verdict')!r}"
                )

    # fit boolean consistency: fit=True iff no red regions
    if "fit" in verdict and "strain_map" in verdict:
        has_red = any(r.get("severity") == "red" for r in verdict["strain_map"])
        if verdict["fit"] is True and has_red:
            errors.append("fit=True but there are red regions — violates spec rule #5")
        if verdict["fit"] is False and not has_red:
            errors.append("fit=False but no red regions — violates spec rule #5")

    # fabric_params_used fields
    required_fabric = [
        "fabric_id", "type", "density_kg_m2",
        "stretch_stiffness", "bend_stiffness", "shear_stiffness", "damping",
    ]
    fp = verdict.get("fabric_params_used", {})
    for field in required_fabric:
        if field not in fp:
            errors.append(f"fabric_params_used missing field: {field!r}")

    # tunnel_through_pct < 2.0
    ttp = verdict.get("tunnel_through_pct")
    if ttp is not None and ttp >= 2.0:
        errors.append(
            f"tunnel_through_pct={ttp} exceeds 2.0% limit (AC-3 violation)"
        )

    return errors
