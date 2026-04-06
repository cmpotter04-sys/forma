# Forma — Session Handoff

**Date:** 2026-04-05  
**Local test count:** 475 passing, 7 skipped  
**GitHub:** cmpotter04-sys/forma (main, commit 3656860)

---

## Where We Are

Phase 2 Stage 1 ("The Rosetta Stone") — GPU transition. CPU solver done, HOOD/ContourCraft validation in progress on Kaggle.

**The one blocking thing:** HOOD validation on Kaggle must pass before Warp parity tests or the Three.js visualizer can be built. Kaggle v21 is RUNNING right now.

---

## Kaggle HOOD Validation Status

**Kernel:** calvinpotter/forma-hood-validation  
**Current version:** v21 (RUNNING as of this handoff)  
**Push command:**
```bash
cd ~/Desktop/Forma\ 4.4.26/notebooks && python3 -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); api.kernels_push('.')"
```
**Check status:**
```bash
python3 -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); print(api.kernels_status('calvinpotter/forma-hood-validation'))"
```
**Download logs:**
```bash
mkdir -p /tmp/kaggle_v21_output && python3 -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); api.kernels_output('calvinpotter/forma-hood-validation', path='/tmp/kaggle_v21_output')"
```

---

## Iteration History (v13–v21)

| Version | Fixed | Error |
|---------|-------|-------|
| v13 | CPU solver working | HOOD: `cugraph` not installed |
| v14 | `cudf`/`cugraph` stubbed | `cccollisions` not installed |
| v15 | `cccollisions` stubbed | `torch_scatter` not installed |
| v16 | `torch_scatter` stub + `warp-lang` install | `runners/ccraft.py` Python 3.12 dataclass error |
| v17 | `warp-lang` + dataclass patch (utils only) | Same dataclass error in `runners/` |
| v18 | Dataclass patch extended to ALL CC files | `torch.load` fails — missing `version` ZIP record |
| v19 | Wrote `version` to zip (wrong: wrote `'version'` not `'{prefix}/version'`) | Same ZIP version error |
| v20 | Correct prefix detection → wrote `'ccraft_data/version'` | `PytorchStreamReader failed locating file data.pkl` |
| v21 | **TWO fixes applied** (see below) — RUNNING | TBD |

---

## What v21 Fixes (Two Independent Changes)

### Fix A — Root cause: wrong checkpoint file (notebook cell `6151f015`)

**v13–v20 bug:** The Google Drive ID `1NfxAeaC2va8TWMjiO_gbAcVPnZ8BYFPD` downloads `ccraft_data.zip` — the **full data bundle**, not a model checkpoint. Previous versions saved this 842 MB ZIP directly as `ContourCraft/trained_models/contourcraft.pth` then called `torch.load()` on it. It has no `data.pkl` because it's a data archive, not a PyTorch model file.

**v21 fix:** Cell `6151f015` now:
1. Downloads as `ccraft_data.zip`
2. Extracts to `/kaggle/working/` → creates `ccraft_data/` directory
3. Sets `CHECKPOINT_PATH = ccraft_data/trained_models/contourcraft.pth` (the real model, ~tens of MB)

Cell `1084a544` now sets `DEFAULTS.data_root = CCRAFT_DATA_DIR` (the `ccraft_data/` dir with `aux_data/` subdirectory that ContourCraft needs for SMPL aux data).

### Fix B — Defense: `data.pkl` alias for old-format checkpoints (`hood_simulate.py` + `ckpt_version_patch` cell)

Even if the real `contourcraft.pth` uses non-standard archive naming (e.g. `contourcraft/contourcraft.pkl` instead of `contourcraft/data.pkl`), the patch cell and `_build_contourcraft_runner()` now:
1. Detect the archive prefix
2. Add `{prefix}/version` if missing
3. **NEW:** Find the actual `.pkl` file under that prefix and copy its content as `{prefix}/data.pkl`

This handles the old PyTorch ZIP format where the pickle was named after the archive.

### Fix C — Defense: tensor type handling (`hood_simulate.py:335`)

`valid_rollout` may return a CUDA torch tensor. The old code called `.astype(np.float64)` on it directly (torch tensors don't have `.astype()`). Fixed to:
```python
_pred_frame = trajectories_dict["pred"][-1]
if hasattr(_pred_frame, "detach"):
    final_positions = _pred_frame.detach().cpu().numpy().astype(np.float64)
else:
    final_positions = np.asarray(_pred_frame, dtype=np.float64)
```

---

## If v21 Fails

Check the log for the new error. After extraction, the `ckpt_version_patch` cell prints the ZIP entries of the real `contourcraft.pth` — inspect that output. Likely candidates:

1. **ContourCraft inference fails accessing `aux_data/`** — check whether `DEFAULTS.data_root` is set to `ccraft_data/` and that `ccraft_data/aux_data/smpl_aux.pkl` exists after extraction.
2. **`make_fromanypose_dataloader` import error** — this function in `utils/datasets.py` has not been audited. Check what it imports at the top of the file.
3. **CUDA OOM** — extraction + pytorch3d build + model load is tight on T4. Unlikely but possible.
4. **The real `contourcraft.pth` also has a non-standard pickle name** — the `ckpt_version_patch` cell will print the ZIP entries. If it says `WARNING: no .pkl found`, you need to inspect the actual ZIP structure.

To debug, add a print of all ZIP entries to the patch cell:
```python
with zipfile.ZipFile(CHECKPOINT_PATH, 'r') as zf:
    print('\n'.join(zf.namelist()))
```

---

## If v21 Passes

You'll see output like:
```
CPU clearance_map:
  chest_front          : +7.1 mm
  waist                : +14.6 mm
  ...
HOOD clearance_map:
  chest_front          : +X.X mm
  ...
OVERALL: PASS — all regions within 2.0 mm tolerance
```
Record those HOOD values as the GPU baseline. Then proceed to **next steps** below.

---

## Next Steps (Ordered)

1. **✅ Confirm HOOD validation passes** — v21 RUNNING. If it errors, fix and push v22.

2. **Run Warp parity tests on Colab T4**
   - `tests/test_warp_parity.py` is written, skips locally (no `warp.sim`)
   - Colab setup (run these cells in order):
     ```python
     # Cell 1 — install
     !git clone https://github.com/cmpotter04-sys/forma.git
     %cd forma
     !pip install -q warp-lang==1.12.0 trimesh scipy numpy ezdxf shapely
     
     # Cell 2 — run parity tests (needs GPU runtime!)
     !python3.11 -m pytest tests/test_warp_parity.py -v
     ```
   - Gate: per-region clearance delta ≤ 0.5mm vs CPU, strain ratio delta ≤ 0.02, verdict match

3. **Fill `output/stage1_profile.json`**
   - Run `scripts/generate_stage1_profile.py` on Colab/Kaggle with GPU
   - Fills AC-2 deliverable (GPU timing benchmarks)
   - Command: `python scripts/generate_stage1_profile.py`

4. **Three.js visualizer** — user chose to hold until GPU baseline confirmed

5. **Precompute batch pipeline** — 20-body measurement grid → HOOD → stored results

---

## Local Development

**Run tests:**
```bash
cd ~/Desktop/Forma\ 4.4.26
python3.11 -m pytest tests/ --tb=short -q
```

**Key source files:**
- `src/geometer/hood/hood_simulate.py` — HOOD backend (commit 3656860)
- `notebooks/hood_validation.ipynb` — Kaggle validation notebook
- `src/geometer/warp/warp_simulate.py` — Warp GPU backend (527+ lines)
- `src/geometer/warp/mesh_bridge.py` — numpy↔Warp data bridge

---

## Stubs Installed in Kaggle Notebook (all still in place)

| Cell ID | What it stubs/installs |
|---------|----------------------|
| `8fb896f1` | `smplx` (SMPL, SMPLX, lbs, utils, body_models) |
| `rapids_stub_00` | `cudf`, `cugraph`, `cccollisions` |
| `ceb23c56` | `torch-geometric`, `torch_scatter`, `torch_sparse` (pip) |
| `scatter_stub_00` | `torch_scatter`/`torch_sparse` fallback stubs if pip fails |
| `1bf69aee` | `warp-lang`, `omegaconf`, `einops`, `networkx`, `munch`, etc. |
| `589511df` | `pytorch3d` (builds from source, ~5 min) |
| `9c90980a` | CCCollisions actual build attempt (falls back to rapids_stub_00 if fails) |
| `ckpt_version_patch` | Patches real `contourcraft.pth` ZIP: adds version + data.pkl alias |
| `0c131a1e` | Python 3.12 dataclass fix (all ContourCraft `.py` files) |

---

## Architecture Reminder

```
pipeline.run_fit_check(backend="cpu"|"warp"|"hood")
    → cpu:  src/geometer/xpbd_simulate.py       (Phase 1, regression baseline)
    → warp: src/geometer/warp/warp_simulate.py  (Phase 2, GPU via NVIDIA Warp)
    → hood: src/geometer/hood/hood_simulate.py  (Phase 2, ContourCraft GNN)
```

All three backends return identical `sim_result` dict schema. `verdict` assembly is shared.

CPU clearance baseline (v13, male M + tshirt M, cotton_jersey_default):
- `chest_front`: +7.1mm
- `waist`: +14.6mm

---

## What This Session Did

1. Identified v20 error: `PytorchStreamReader failed locating file data.pkl`
2. Identified true root cause: Google Drive file is `ccraft_data.zip` data bundle, not model checkpoint — previous sessions were calling `torch.load()` on the entire data archive
3. Confirmed v21 notebook already had the extraction fix (from linter/previous session)
4. Added defense fix: `data.pkl` alias for old-format PyTorch checkpoints
5. Added defense fix: `detach().cpu().numpy()` for CUDA tensor output from `valid_rollout`
6. Committed all fixes (commit `3656860`) and pushed to GitHub + Kaggle v21
7. v21 is RUNNING as of handoff — should be ~10–15 min into its run
