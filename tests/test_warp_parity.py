"""
tests/test_warp_parity.py — AC-3: Warp vs CPU regression test

Runs both backends on identical inputs (M-on-M, cotton_jersey_default)
and asserts that results match within tolerance:
  - Per-region clearance delta ≤ 0.5mm
  - Per-region strain ratio delta ≤ 0.02
  - Verdict agreement (fit: true/false matches)

GPU tests are decorated with @pytest.mark.skipif(not _warp_available(), ...)
so they skip cleanly on CPU-only machines (local dev, CI without CUDA).
On Colab with GPU runtime, all tests must pass.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

# Ensure src/ is importable
_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Warp availability probe
# ---------------------------------------------------------------------------

def _warp_available() -> bool:
    """
    Return True if warp-lang is installed AND warp.init() succeeds
    (i.e. a CUDA device or CPU fallback is actually usable).

    Called at decoration time, so it must not raise.
    """
    try:
        import warp as wp
        import warp.sim  # noqa: F401
        wp.init()
        return True
    except Exception:
        return False


# Evaluate once at module load so all decorators share the same result.
# This avoids redundant warp.init() calls and keeps collection fast.
_WARP_OK = _warp_available()

# ---------------------------------------------------------------------------
# Test asset paths
# ---------------------------------------------------------------------------

DATA_DIR = _ROOT / "data"
BODY_M    = str(DATA_DIR / "bodies"   / "makehuman_male_M.ply")
PATTERN_M = str(DATA_DIR / "patterns" / "tshirt_size_M.json")
MANIFEST_M = str(_ROOT / "seam_manifests" / "tshirt_size_M_manifest.json")

REQUIRED_REGIONS = [
    "chest_front",
    "chest_side",
    "shoulder_left",
    "shoulder_right",
    "upper_back",
    "waist",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_exists() -> bool:
    return (
        Path(BODY_M).exists()
        and Path(PATTERN_M).exists()
        and Path(MANIFEST_M).exists()
    )


def _strain_by_region(verdict: dict) -> dict:
    """Convert verdict["strain_map"] list → dict keyed by region name."""
    return {entry["region"]: entry for entry in verdict["strain_map"]}


# ---------------------------------------------------------------------------
# Module-scoped verdict fixtures (run pipelines once, reused by all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpu_verdict():
    """Run CPU backend once; shared across all tests in this module."""
    if not _data_exists():
        pytest.skip("Test data not found")
    from pipeline import run_fit_check
    return run_fit_check(
        BODY_M, PATTERN_M, MANIFEST_M,
        fabric_id="cotton_jersey_default",
        backend="cpu",
    )


@pytest.fixture(scope="module")
def warp_verdict():
    """Run Warp backend once; shared across all GPU tests in this module."""
    if not _WARP_OK:
        pytest.skip("warp/CUDA not available")
    if not _data_exists():
        pytest.skip("Test data not found")
    from pipeline import run_fit_check
    return run_fit_check(
        BODY_M, PATTERN_M, MANIFEST_M,
        fabric_id="cotton_jersey_default",
        backend="warp",
    )


# ---------------------------------------------------------------------------
# 1. CPU smoke test — always runs (no GPU required)
# ---------------------------------------------------------------------------

def test_cpu_backend_smoke(cpu_verdict):
    """
    CPU backend runs without error and returns a structurally valid verdict.
    Always runs; no GPU required.
    """
    v = cpu_verdict
    assert isinstance(v, dict), "verdict must be a dict"
    assert "fit" in v, "verdict missing 'fit'"
    assert isinstance(v["fit"], bool), "fit must be a bool"
    assert "strain_map" in v, "verdict missing 'strain_map'"
    assert "ease_map" in v, "verdict missing 'ease_map'"
    assert "verdict_id" in v, "verdict missing 'verdict_id'"
    assert v["verdict_id"].startswith("vrd_"), "verdict_id must start with 'vrd_'"
    assert len(v["verdict_id"]) == 16, (
        f"verdict_id must be 'vrd_' + 12 hex chars, got {v['verdict_id']!r}"
    )

    sm = _strain_by_region(v)
    for region in REQUIRED_REGIONS:
        assert region in sm, f"strain_map missing region: {region!r}"
        entry = sm[region]
        assert "delta_mm" in entry, f"{region}: missing 'delta_mm'"
        assert "severity" in entry, f"{region}: missing 'severity'"
        assert "median_strain_ratio" in entry, f"{region}: missing 'median_strain_ratio'"
        assert entry["severity"] in ("green", "yellow", "red"), (
            f"{region}: invalid severity {entry['severity']!r}"
        )


# ---------------------------------------------------------------------------
# 2. Warp smoke test — GPU skip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _WARP_OK, reason="warp/CUDA not available")
def test_warp_backend_smoke(warp_verdict):
    """
    Warp backend runs without error and returns a structurally valid verdict.
    Skips on CPU-only machines.
    """
    v = warp_verdict
    assert isinstance(v, dict), "verdict must be a dict"
    assert "fit" in v, "verdict missing 'fit'"
    assert isinstance(v["fit"], bool), "fit must be a bool"
    assert "strain_map" in v, "verdict missing 'strain_map'"
    assert "ease_map" in v, "verdict missing 'ease_map'"
    assert "verdict_id" in v, "verdict missing 'verdict_id'"
    assert v["verdict_id"].startswith("vrd_"), "verdict_id must start with 'vrd_'"
    assert len(v["verdict_id"]) == 16, (
        f"verdict_id must be 'vrd_' + 12 hex chars, got {v['verdict_id']!r}"
    )

    sm = _strain_by_region(v)
    for region in REQUIRED_REGIONS:
        assert region in sm, f"strain_map missing region: {region!r}"
        entry = sm[region]
        assert "delta_mm" in entry, f"{region}: missing 'delta_mm'"
        assert "severity" in entry, f"{region}: missing 'severity'"
        assert "median_strain_ratio" in entry, f"{region}: missing 'median_strain_ratio'"
        assert entry["severity"] in ("green", "yellow", "red"), (
            f"{region}: invalid severity {entry['severity']!r}"
        )


# ---------------------------------------------------------------------------
# 3. Clearance parity — GPU skip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _WARP_OK, reason="warp/CUDA not available")
def test_clearance_parity(cpu_verdict, warp_verdict):
    """
    Per-region |cpu_delta_mm - warp_delta_mm| ≤ 0.5mm for all required regions.
    Tolerance spec: FORMA_PHASE2_EXECUTOR_SPEC.md AC-3.
    """
    cpu_sm = _strain_by_region(cpu_verdict)
    warp_sm = _strain_by_region(warp_verdict)

    failures = []
    for region in REQUIRED_REGIONS:
        cpu_delta  = cpu_sm[region]["delta_mm"]
        warp_delta = warp_sm[region]["delta_mm"]
        diff = abs(cpu_delta - warp_delta)
        if diff > 0.5:
            failures.append(
                f"  {region}: cpu={cpu_delta:.3f}mm  warp={warp_delta:.3f}mm  "
                f"delta={diff:.3f}mm  (limit 0.5mm)"
            )

    assert not failures, (
        "Clearance parity failures (|cpu - warp| > 0.5mm):\n"
        + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 4. Strain ratio parity — GPU skip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _WARP_OK, reason="warp/CUDA not available")
def test_strain_parity(cpu_verdict, warp_verdict):
    """
    Per-region |cpu_median_strain_ratio - warp_median_strain_ratio| ≤ 0.02
    for all required regions.
    Tolerance spec: FORMA_PHASE2_EXECUTOR_SPEC.md AC-3.
    """
    cpu_sm = _strain_by_region(cpu_verdict)
    warp_sm = _strain_by_region(warp_verdict)

    failures = []
    for region in REQUIRED_REGIONS:
        cpu_sr  = cpu_sm[region].get("median_strain_ratio")
        warp_sr = warp_sm[region].get("median_strain_ratio")

        if cpu_sr is None or warp_sr is None:
            failures.append(
                f"  {region}: median_strain_ratio missing "
                f"(cpu={cpu_sr!r}, warp={warp_sr!r})"
            )
            continue

        diff = abs(cpu_sr - warp_sr)
        if diff > 0.02:
            failures.append(
                f"  {region}: cpu={cpu_sr:.4f}  warp={warp_sr:.4f}  "
                f"delta={diff:.4f}  (limit 0.02)"
            )

    assert not failures, (
        "Strain ratio parity failures (|cpu - warp| > 0.02):\n"
        + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 5. Verdict parity (fit bool) — GPU skip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _WARP_OK, reason="warp/CUDA not available")
def test_verdict_parity(cpu_verdict, warp_verdict):
    """
    fit: true/false must agree between CPU and Warp backends.
    Tolerance spec: FORMA_PHASE2_EXECUTOR_SPEC.md AC-3.
    """
    cpu_fit  = cpu_verdict["fit"]
    warp_fit = warp_verdict["fit"]
    assert cpu_fit == warp_fit, (
        f"Verdict mismatch: cpu fit={cpu_fit}, warp fit={warp_fit}.\n"
        "Both backends must agree on whether the garment fits."
    )


# ---------------------------------------------------------------------------
# 6. Warp timing — warns (not fails) if Warp is slower — GPU skip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _WARP_OK, reason="warp/CUDA not available")
def test_simulation_ms_warp_faster(cpu_verdict, warp_verdict):
    """
    Warp backend should be faster than CPU backend on GPU hardware.
    Emits a warning (not a failure) if Warp is slower — the test always passes
    as long as both backends complete successfully.

    Rationale: on Colab T4/A100 Warp should be substantially faster for
    large meshes.  On small test meshes or shared GPU instances the overhead
    of Warp initialisation can exceed the simulation time, so we do not
    assert a strict speedup to avoid flaky failures on CI.
    """
    cpu_ms  = cpu_verdict["simulation_ms"]
    warp_ms = warp_verdict["simulation_ms"]

    if warp_ms >= cpu_ms:
        warnings.warn(
            f"Warp backend ({warp_ms}ms) is not faster than CPU backend "
            f"({cpu_ms}ms).  This is expected on small meshes or CPU-fallback "
            f"Warp builds, but should be investigated on GPU hardware.",
            stacklevel=1,
        )
    else:
        speedup = cpu_ms / max(warp_ms, 1)
        # Not an assertion — just informational output captured by pytest -v
        print(
            f"\n  Warp speedup: {speedup:.1f}x  "
            f"(cpu={cpu_ms}ms, warp={warp_ms}ms)"
        )


# ---------------------------------------------------------------------------
# 7. Invalid backend raises ValueError — always runs
# ---------------------------------------------------------------------------

def test_invalid_backend_raises():
    """ValueError for unknown backend name."""
    if not _data_exists():
        pytest.skip("Test data not found")
    from pipeline import run_fit_check
    with pytest.raises(ValueError, match="Unknown backend"):
        run_fit_check(
            BODY_M, PATTERN_M, MANIFEST_M,
            backend="bogus_backend",
        )


# ---------------------------------------------------------------------------
# 8. CPU pipeline signature guard — always runs (import-only, no simulation)
# ---------------------------------------------------------------------------

def test_cpu_pipeline_signature():
    """
    run_fit_check must accept a 'backend' parameter defaulting to 'cpu'.
    This guards against accidental API regressions — runs without GPU or data.
    """
    import inspect
    from pipeline import run_fit_check
    sig = inspect.signature(run_fit_check)
    assert "backend" in sig.parameters, (
        "pipeline.run_fit_check is missing the 'backend' parameter"
    )
    assert sig.parameters["backend"].default == "cpu", (
        "pipeline.run_fit_check 'backend' parameter default must be 'cpu'"
    )
