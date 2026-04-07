# Forma — Immediate Next Steps

## 1. Re-verify the current ingestion state

Run:

```bash
python3 -m pytest \
  tests/test_dxf_ingestion.py \
  tests/test_dxf_seam_inference.py \
  tests/test_dxf_stitch_plan.py \
  tests/test_dxf_bundle.py \
  tests/test_pipeline_dxf.py \
  tests/test_cli.py -q
```

Then run:

```bash
python3 -m pytest tests/test_api.py tests/test_stage1_profile.py -q
```

Expected from this session:

- DXF/CLI slice: `17 passed`
- API/stage-profile slice: `53 passed`

## 2. Continue the DXF ingestion close-out

Highest-priority code to read first:

- `src/pattern_maker/load_dxf.py`
- `src/tailor/panel_labeler.py`
- `src/tailor/dxf_seam_inference.py`
- `src/tailor/dxf_stitch_plan.py`
- `src/tailor/dxf_bundle.py`

Highest-priority open problem:

- The DXF path is still shirt-specific and heuristic.
- It now behaves safely, but it still needs stronger inference on messier real shirt DXFs.

Most useful next improvements:

1. Harden edge-role inference in `dxf_seam_inference.py`.
2. Add messier-but-supported DXF fixtures.
3. Decide whether `validation.dropped_seam_candidates` should remain diagnostic or become a hard failure for MVP shirt intake.

## 3. Monitor HOOD / Kaggle without blocking ingestion

Poll status:

```bash
python3 -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); print(api.kernels_status('calvinpotter/forma-hood-validation'))"
```

Last known status from this session:

- `{"status": "ERROR", "failureMessage": ""}`

Try pulling output bundle:

```bash
python3 -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); api.kernels_output('calvinpotter/forma-hood-validation', path='/tmp/forma_kaggle_output')"
```

Goal:

- get the real failure trace
- do not guess at the root cause without the artifact

## 4. Respect the current verification boundary

Important user instruction:

- the CPU engine is only acceptable for ingestion / schema / local contract checks
- the CPU engine is **not** acceptable as proof that GPU physics are solved
- treat CPU as a cheap plumbing harness, not as a meaningful physics validator

GPU verification should be grounded in:

- Kaggle HOOD results
- open-source, free-for-commercial-use garment / cloth physics implementations already relevant to this repo

## 5. Do not disturb the dirty worktree

The repo already has unrelated in-progress edits, including:

- `src/api.py`
- `src/cli.py`
- `src/pipeline.py`
- `docs/FORMA_CODEBASE_OVERVIEW.md`
- `notebooks/hood_validation.ipynb`
- `scripts/generate_stage1_profile.py`
- `output/stage1_profile.json`

Do not revert anything unless explicitly asked.

## 6. Read this after starting

For full context, read:

- `HANDOFF.md`
