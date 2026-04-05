"""
Measurement validator — compares user-supplied target measurements against
the achieved measurements stored in a body_profile.json.

For Forma synthetic mannequins the sculptor already measures itself and saves
achieved_measurements to the JSON profile, so no mesh re-analysis is needed.

Entry point:
    validate_measurements(profile_path, targets) -> ValidationResult
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


MEASUREMENT_KEYS = [
    "height_cm",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_width_cm",
    "inseam_cm",
]

# Tolerance per measurement (mm).  Height is more lenient.
_TOLERANCES_MM = {
    "height_cm": 20.0,
    "chest_cm": 15.0,
    "waist_cm": 15.0,
    "hip_cm": 15.0,
    "shoulder_width_cm": 20.0,
    "inseam_cm": 20.0,
}


@dataclass
class MeasurementDelta:
    key: str
    target_cm: float
    achieved_cm: float
    delta_mm: float
    within_tolerance: bool
    tolerance_mm: float


@dataclass
class ValidationResult:
    valid: bool
    max_delta_mm: float
    deltas: list[MeasurementDelta]

    def summary(self) -> str:
        lines = [f"valid={self.valid}  max_delta={self.max_delta_mm:.1f}mm"]
        for d in self.deltas:
            status = "OK" if d.within_tolerance else "WARN"
            lines.append(
                f"  [{status}] {d.key}: target={d.target_cm:.1f}  "
                f"achieved={d.achieved_cm:.1f}  Δ={d.delta_mm:+.1f}mm"
            )
        return "\n".join(lines)


def validate_measurements(
    profile_path: str | Path,
    targets: dict[str, float],
    strict: bool = False,
) -> ValidationResult:
    """
    Compare target measurements against achieved_measurements in body_profile.json.

    Parameters
    ----------
    profile_path : str | Path
        Path to body_profile.json (saved alongside the PLY by generate_body()).
    targets : dict[str, float]
        Target measurements in cm.  Only keys present in both targets and the
        profile are compared; missing keys are skipped.
    strict : bool
        If True, ValidationResult.valid is False if ANY measurement exceeds
        its tolerance.  If False (default), valid is True as long as no
        measurement exceeds 2× tolerance.

    Returns
    -------
    ValidationResult
    """
    profile_path = Path(profile_path)
    with open(profile_path) as f:
        profile = json.load(f)

    achieved = profile.get("achieved_measurements", {})

    deltas = []
    for key, target_cm in targets.items():
        if key not in achieved:
            continue
        achieved_cm = float(achieved[key])
        delta_mm = (achieved_cm - target_cm) * 10.0
        tol = _TOLERANCES_MM.get(key, 15.0)
        threshold = tol if strict else tol * 2.0
        deltas.append(MeasurementDelta(
            key=key,
            target_cm=target_cm,
            achieved_cm=achieved_cm,
            delta_mm=delta_mm,
            within_tolerance=abs(delta_mm) <= tol,
            tolerance_mm=tol,
        ))

    max_delta = max((abs(d.delta_mm) for d in deltas), default=0.0)
    threshold = max(d.tolerance_mm for d in deltas) if deltas else 15.0
    if strict:
        valid = all(d.within_tolerance for d in deltas)
    else:
        valid = all(abs(d.delta_mm) <= d.tolerance_mm * 2.0 for d in deltas)

    return ValidationResult(valid=valid, max_delta_mm=max_delta, deltas=deltas)
