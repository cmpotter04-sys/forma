# Forma — Warp GPU Backend

## Setup (Google Colab)

```python
# First cell of every Colab notebook
!pip install warp-lang trimesh scipy numpy pytest

import warp as wp
wp.init()
print(f"Warp {wp.__version__}, Device: {wp.get_preferred_device()}")
```

Verify GPU is available. If `wp.get_preferred_device()` returns `cpu`, the Colab
runtime is not GPU-enabled — go to **Runtime > Change runtime type > GPU**.

## Usage

```python
from pipeline import run_fit_check

# CPU backend (Phase 1 — regression baseline)
cpu_verdict = run_fit_check(body, pattern, manifest, backend="cpu")

# Warp GPU backend (Phase 2)
warp_verdict = run_fit_check(body, pattern, manifest, backend="warp")
```

## Running Parity Tests

```bash
# Phase 1 baseline (must still pass)
pytest tests/ --ignore=tests/test_warp_parity.py

# Phase 2 parity (Warp vs CPU — requires GPU)
pytest tests/test_warp_parity.py -v
```

## Architecture

```
src/geometer/warp/
├── __init__.py          # Package init
├── mesh_bridge.py       # numpy/trimesh ↔ Warp array conversion
├── warp_simulate.py     # Warp XPBDIntegrator integration
└── README.md            # This file
```

The Warp backend uses these shared modules from Phase 1:
- `garment_assembly.py` — Panel geometry, placement, triangulation, seam constraints
- `clearance.py` — Signed clearance computation
- `convergence.py` — Energy/movement convergence detection
- `region_map.py` — Body region segmentation

## Float Precision

Uses **float32** by default (Warp GPU default, 2× faster than float64).
If parity test fails at ≤ 0.5mm threshold, switch to float64 in `mesh_bridge.py`.

## Dependencies

- **warp-lang** (Apache 2.0) — GPU simulation framework
- All other deps inherited from Phase 1 (numpy, scipy, trimesh)
