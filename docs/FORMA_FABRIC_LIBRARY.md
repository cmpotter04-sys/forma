# FORMA — Fabric Library Spec
**Version:** 1.0  
**Phase:** 1 / Week 1 (cotton_jersey_default only), expanded in Phase 2  
**Status:** READY TO BUILD

---

## Purpose

The original spec used `{"type": "cotton_jersey", "stretch": 0.7}` which gives the
XPBD solver almost no information. Real cloth simulation needs at minimum five
parameters: density, stretch stiffness, bend stiffness, shear stiffness, and damping.

This document defines the Fabric Library — a simple JSON lookup table that the
Geometer loads before simulation. For Week 1, only `cotton_jersey_default` is needed.
Additional fabrics are added in Phase 2 when the Pattern Maker expands beyond T-shirts.

---

## Schema: fabric_library.json

```json
{
  "version": "1.0",
  "fabrics": {
    "cotton_jersey_default": {
      "fabric_id": "cotton_jersey_default",
      "type": "cotton_jersey",
      "description": "Standard weight cotton jersey knit, typical for basic T-shirts",
      "density_kg_m2": 0.18,
      "stretch_stiffness": 40.0,
      "bend_stiffness": 0.005,
      "shear_stiffness": 3.0,
      "damping": 0.995,
      "gravity_scale": 1.0,
      "notes": "Values calibrated to GarmentCode default simulation parameters"
    },
    "cotton_jersey_heavy": {
      "fabric_id": "cotton_jersey_heavy",
      "type": "cotton_jersey",
      "description": "Heavy cotton jersey, thick T-shirts and sweatshirts",
      "density_kg_m2": 0.30,
      "stretch_stiffness": 55.0,
      "bend_stiffness": 0.015,
      "shear_stiffness": 5.0,
      "damping": 0.993,
      "gravity_scale": 1.0,
      "notes": "Phase 2 addition"
    },
    "polyester_woven": {
      "fabric_id": "polyester_woven",
      "type": "polyester_woven",
      "description": "Lightweight polyester plain weave, dress shirts",
      "density_kg_m2": 0.12,
      "stretch_stiffness": 80.0,
      "bend_stiffness": 0.008,
      "shear_stiffness": 8.0,
      "damping": 0.997,
      "gravity_scale": 1.0,
      "notes": "Phase 2 addition"
    },
    "denim_12oz": {
      "fabric_id": "denim_12oz",
      "type": "denim",
      "description": "Medium weight denim, standard jeans",
      "density_kg_m2": 0.40,
      "stretch_stiffness": 120.0,
      "bend_stiffness": 0.08,
      "shear_stiffness": 15.0,
      "damping": 0.990,
      "gravity_scale": 1.0,
      "notes": "Phase 2 addition"
    },
    "silk_charmeuse": {
      "fabric_id": "silk_charmeuse",
      "type": "silk_charmeuse",
      "description": "Lightweight silk charmeuse, dresses and blouses",
      "density_kg_m2": 0.07,
      "stretch_stiffness": 25.0,
      "bend_stiffness": 0.001,
      "shear_stiffness": 1.5,
      "damping": 0.998,
      "gravity_scale": 1.0,
      "notes": "Phase 2 addition — drapes very differently from jersey"
    }
  }
}
```

---

## Parameter Definitions

| Parameter | Unit | Description | Range (typical) |
|-----------|------|-------------|-----------------|
| density_kg_m2 | kg/m² | Mass per unit area of the fabric. Heavier fabrics drape differently and respond more slowly to constraints. | 0.05 – 0.50 |
| stretch_stiffness | unitless | Resistance to in-plane stretching. Higher = stiffer fabric that resists extension. Maps to XPBD distance constraint compliance. | 20 – 200 |
| bend_stiffness | unitless | Resistance to bending/folding. Higher = stiffer, less drapey. Silk is very low, denim is high. Maps to XPBD dihedral constraint compliance. | 0.001 – 0.1 |
| shear_stiffness | unitless | Resistance to in-plane shearing (parallelogram deformation). Prevents fabric from collapsing. | 1.0 – 20.0 |
| damping | unitless | Velocity damping per timestep. 1.0 = no damping, 0.99 = moderate. Prevents oscillation. | 0.98 – 1.0 |
| gravity_scale | unitless | Multiplier on gravity (9.81 m/s²). Normally 1.0. Can be reduced to simulate lighter drape behavior. | 0.5 – 1.5 |

---

## How the Geometer Uses These Parameters

```python
def load_fabric(fabric_id: str) -> dict:
    """Load fabric parameters from the fabric library."""
    import json
    with open("data/fabrics/fabric_library.json") as f:
        library = json.load(f)
    if fabric_id not in library["fabrics"]:
        raise ValueError(f"Unknown fabric_id: {fabric_id}")
    return library["fabrics"][fabric_id]
```

The Geometer passes these values to the XPBD solver:
- `density_kg_m2` → per-vertex mass = density × vertex_voronoi_area
- `stretch_stiffness` → XPBD distance constraint compliance = 1 / stretch_stiffness
- `bend_stiffness` → XPBD dihedral constraint compliance = 1 / bend_stiffness
- `shear_stiffness` → XPBD triangle shear constraint compliance
- `damping` → velocity *= damping each timestep

---

## Week 1 Scope

For the smoke test, only `cotton_jersey_default` is used. The fabric_id is
specified per pattern or defaults to `cotton_jersey_default` if not specified.

All three test runs (S, M, XL on size M body) use the same fabric parameters
to isolate the variable being tested (garment size vs body size).

---

## Calibration Notes (Phase 2)

These initial values are reasonable starting points based on published cloth
simulation literature and GarmentCode defaults. In Phase 2:

1. Validate against GarmentCode's own default parameters (their solver may use
   different units or scaling — normalize during integration)
2. Create a calibration test: simulate a 30cm × 30cm fabric square falling
   under gravity, compare drape profile to real fabric photos
3. Allow per-garment fabric overrides in the pattern metadata

---

*FORMA Fabric Library Spec v1.0*
