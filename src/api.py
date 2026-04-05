# requires: pip install fastapi uvicorn

"""
src/api.py — Forma REST API

Wraps run_fit_check() as a FastAPI application.

Endpoints
---------
POST /fit-check          — full pipeline; body mesh path OR anny_measurements
POST /fit-check/quick    — size preset shortcut (XS/S/M/L/XL + gender + garment)
GET  /health             — liveness probe
GET  /sizes              — all SIZE_PRESETS and SIZE_PRESETS_FEMALE
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

# Ensure src/ is on the import path when run directly
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from pipeline import run_fit_check
from sculptor.anny_body import SIZE_PRESETS, SIZE_PRESETS_FEMALE

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Forma Fit-Check API",
    version="2.0",
    description="Physics-based garment fit verification.",
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnnnyMeasurements(BaseModel):
    height_cm: float = Field(..., gt=0, description="Total standing height in cm")
    chest_cm: float = Field(..., gt=0, description="Chest circumference in cm")
    waist_cm: float = Field(..., gt=0, description="Waist circumference in cm")
    hips_cm: float = Field(..., gt=0, description="Hip circumference in cm")
    inseam_cm: float = Field(..., gt=0, description="Inseam leg length in cm")
    shoulder_width_cm: float = Field(..., gt=0, description="Biacromial shoulder width in cm")


class FitCheckRequest(BaseModel):
    """
    Full fit-check request.

    Provide exactly one of:
      - anny_measurements  — generate body on the fly from anthropometric measurements
      - body_mesh_path     — path to a pre-built PLY body mesh

    pattern_path and seam_manifest_path are always required.
    """
    # Body source — mutually exclusive; validated below
    anny_measurements: Optional[AnnnyMeasurements] = Field(
        None,
        description="Generate body from measurements (mutually exclusive with body_mesh_path)",
    )
    body_mesh_path: Optional[str] = Field(
        None,
        description="Path to pre-built body mesh PLY (mutually exclusive with anny_measurements)",
    )

    # Garment inputs — always required
    pattern_path: str = Field(..., description="Path to GarmentCode JSON pattern file")
    seam_manifest_path: str = Field(..., description="Path to seam_manifest.json")

    # Simulation options
    fabric_id: str = Field(
        "cotton_jersey_default",
        description="Key into data/fabrics/fabric_library.json",
    )
    backend: Literal["cpu", "warp", "hood"] = Field(
        "cpu",
        description="Simulation backend: 'cpu' (Phase 1 XPBD), 'warp' (GPU), or 'hood'",
    )
    subdivide_target: int = Field(
        0,
        ge=0,
        description="Subdivide garment mesh to this vertex count (0 = no subdivision)",
    )

    @model_validator(mode="after")
    def _check_body_source(self) -> "FitCheckRequest":
        has_measurements = self.anny_measurements is not None
        has_path = self.body_mesh_path is not None
        if has_measurements and has_path:
            raise ValueError(
                "Provide either anny_measurements or body_mesh_path, not both."
            )
        if not has_measurements and not has_path:
            raise ValueError(
                "One of anny_measurements or body_mesh_path must be provided."
            )
        return self


class QuickFitCheckRequest(BaseModel):
    """
    Quick fit-check using a named size preset.

    Resolves SIZE_PRESETS (male) or SIZE_PRESETS_FEMALE from size + gender,
    then runs the full pipeline with anny_measurements.
    """
    size: str = Field(
        ...,
        description=(
            "Size key. Male: S, M, XL. Female: XS, S, M, L, XL."
        ),
    )
    gender: Literal["male", "female"] = Field(
        ...,
        description="Select male or female size preset table",
    )
    pattern_path: str = Field(..., description="Path to GarmentCode JSON pattern file")
    seam_manifest_path: str = Field(..., description="Path to seam_manifest.json")
    fabric_id: str = Field(
        "cotton_jersey_default",
        description="Key into data/fabrics/fabric_library.json",
    )
    backend: Literal["cpu", "warp", "hood"] = Field(
        "cpu",
        description="Simulation backend",
    )
    subdivide_target: int = Field(0, ge=0)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness probe")
def health() -> dict:
    """Return API status and version."""
    return {"status": "ok", "version": "2.0"}


@app.get("/sizes", summary="List all size presets")
def sizes() -> dict:
    """Return SIZE_PRESETS (male) and SIZE_PRESETS_FEMALE."""
    return {
        "male": SIZE_PRESETS,
        "female": SIZE_PRESETS_FEMALE,
    }


@app.post("/fit-check", summary="Run a full fit-check")
def fit_check(request: FitCheckRequest) -> JSONResponse:
    """
    Run a full pipeline fit-check.

    Supply either anny_measurements (body generated on the fly) or a
    body_mesh_path to an existing PLY file, plus pattern_path,
    seam_manifest_path, fabric_id, and backend.

    Returns the complete fit_verdict.json v1.2 document.
    """
    try:
        measurements_dict = (
            request.anny_measurements.model_dump()
            if request.anny_measurements is not None
            else None
        )
        verdict = run_fit_check(
            body_mesh_path=request.body_mesh_path,
            pattern_path=request.pattern_path,
            seam_manifest_path=request.seam_manifest_path,
            fabric_id=request.fabric_id,
            backend=request.backend,
            subdivide_target=request.subdivide_target,
            anny_measurements=measurements_dict,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Backend unavailable: {exc}",
        )
    except Exception as exc:  # SimulationExplosionError, RuntimeError, SeamValidationError, etc.
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content=verdict)


@app.post("/fit-check/quick", summary="Quick fit-check via size preset")
def fit_check_quick(request: QuickFitCheckRequest) -> JSONResponse:
    """
    Run a fit-check using a named size preset.

    Resolves the preset measurements from size + gender, generates the body
    via Anny, and runs the full pipeline.  Useful for automated grading runs.
    """
    # Resolve preset
    presets = SIZE_PRESETS_FEMALE if request.gender == "female" else SIZE_PRESETS
    if request.size not in presets:
        valid = list(presets.keys())
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown size '{request.size}' for gender '{request.gender}'. "
                f"Valid sizes: {valid}"
            ),
        )

    measurements = presets[request.size]

    try:
        verdict = run_fit_check(
            pattern_path=request.pattern_path,
            seam_manifest_path=request.seam_manifest_path,
            fabric_id=request.fabric_id,
            backend=request.backend,
            subdivide_target=request.subdivide_target,
            anny_measurements=measurements,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Backend unavailable: {exc}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content=verdict)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
