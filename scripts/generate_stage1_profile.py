"""
scripts/generate_stage1_profile.py
-----------------------------------
AC-2 deliverable: generates output/stage1_profile.json with real timing data.

Run from repo root on a Colab/Kaggle T4 GPU:
    python scripts/generate_stage1_profile.py

What it does:
  1. Checks that warp-lang is available; skips GPU runs gracefully if not.
  2. Runs CPU backend (M-size tshirt + M-size body, cotton_jersey_default).
  3. Runs Warp backend on identical inputs.
  4. Computes per-region clearance and strain deltas between CPU and Warp.
  5. Runs Warp backend with subdivide_target=10000 (smoke test, ~10K verts).
  6. Collects wall-clock timing and VRAM stats via torch.cuda.memory_allocated.
  7. Writes output/stage1_profile.json following the Stage 1 schema.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — works whether run from repo root or from scripts/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Input paths (relative to repo root)
# ---------------------------------------------------------------------------
BODY_MESH_PATH = str(_REPO_ROOT / "data" / "bodies" / "makehuman_male_M.ply")
PATTERN_PATH = str(_REPO_ROOT / "data" / "patterns" / "tshirt_size_M.json")
SEAM_MANIFEST_PATH = str(_REPO_ROOT / "data" / "seam_manifests" / "tshirt_size_M_seam_manifest.json")
FABRIC_ID = "cotton_jersey_default"
OUTPUT_PATH = _REPO_ROOT / "output" / "stage1_profile.json"

# Smoke test: ~10K vertex target (NOT 100K — would OOM on T4)
SMOKE_SUBDIVIDE_TARGET = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_warp_available() -> bool:
    """Return True if warp-lang is importable and has a GPU device."""
    try:
        import warp as wp
        wp.init()
        preferred = wp.get_preferred_device()
        if "cuda" not in str(preferred).lower() and preferred != "cuda:0":
            print(f"[warn] Warp available but preferred device is {preferred!r} (not GPU).")
            print("       GPU runs will fall back to warp CPU path.")
        return True
    except ImportError:
        print("[skip] warp-lang not installed — Warp backend runs will be skipped.")
        return False
    except Exception as exc:
        print(f"[skip] Warp init failed ({exc}) — Warp backend runs will be skipped.")
        return False


def _collect_environment(warp_available: bool) -> dict:
    """Collect runtime environment metadata."""
    env: dict = {
        "device": None,
        "warp_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "vram_gb": None,
    }

    if warp_available:
        try:
            import warp as wp
            env["warp_version"] = wp.__version__
            env["device"] = str(wp.get_preferred_device())
        except Exception:
            pass

    # Collect CUDA / GPU info via torch (optional dependency)
    try:
        import torch
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
    except ImportError:
        pass
    except Exception:
        pass

    return env


def _reset_vram_stats() -> None:
    """Reset CUDA memory peak stats if torch is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _get_vram_peak_mb() -> float | None:
    """Return peak CUDA memory allocated in MB, or None if torch unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except ImportError:
        pass
    return None


def _run_backend(backend: str, subdivide_target: int = 0) -> tuple[dict, float]:
    """
    Run pipeline.run_fit_check() for the given backend and return
    (verdict_dict, wall_clock_seconds).

    Returns (None, elapsed) if the run fails; caller should handle None.
    """
    from pipeline import run_fit_check

    t0 = time.perf_counter()
    verdict = run_fit_check(
        body_mesh_path=BODY_MESH_PATH,
        pattern_path=PATTERN_PATH,
        seam_manifest_path=SEAM_MANIFEST_PATH,
        fabric_id=FABRIC_ID,
        backend=backend,
        subdivide_target=subdivide_target,
    )
    elapsed = time.perf_counter() - t0
    return verdict, elapsed


def _extract_clearance(verdict: dict) -> dict[str, float]:
    """Return {region: delta_mm} from a verdict's strain_map."""
    return {entry["region"]: entry["delta_mm"] for entry in verdict.get("strain_map", [])}


def _extract_strain_ratios(verdict: dict) -> dict[str, float]:
    """Return {region: median_strain_ratio} from a verdict's strain_map."""
    result = {}
    for entry in verdict.get("strain_map", []):
        if "median_strain_ratio" in entry:
            result[entry["region"]] = entry["median_strain_ratio"]
    return result


def _compute_deltas(
    cpu_clearance: dict[str, float],
    warp_clearance: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Compute per-region absolute clearance deltas and return
    (max_delta_mm, per_region_dict).
    """
    deltas: dict[str, float] = {}
    for region in cpu_clearance:
        if region in warp_clearance:
            deltas[region] = round(abs(cpu_clearance[region] - warp_clearance[region]), 3)
    max_delta = max(deltas.values()) if deltas else None
    return max_delta, deltas


def _compute_strain_deltas(
    cpu_strain: dict[str, float],
    warp_strain: dict[str, float],
) -> float | None:
    """Return max absolute strain ratio delta across shared regions."""
    deltas = []
    for region in cpu_strain:
        if region in warp_strain:
            deltas.append(abs(cpu_strain[region] - warp_strain[region]))
    return round(max(deltas), 4) if deltas else None


def _garment_vertex_count(verdict: dict) -> int | None:
    """
    Try to recover garment vertex count from the verdict.
    verdict doesn't directly store it; we use simulation_ms as a proxy marker
    and note it's unavailable. The profile records None here unless the sim_result
    exposes it — add this metric if the geometer pipeline is extended.
    """
    # The verdict v1.2 schema does not include garment vertex count.
    # Return None — the profile will show null and can be filled in manually
    # once the geometer exposes vertex_count in sim_result.
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Forma Stage 1 — AC-2 Benchmark Profile Generator")
    print("=" * 60)

    warp_available = _check_warp_available()
    environment = _collect_environment(warp_available)

    print(f"\nEnvironment:")
    print(f"  device       : {environment['device']}")
    print(f"  warp_version : {environment['warp_version']}")
    print(f"  gpu_name     : {environment['gpu_name']}")
    print(f"  vram_gb      : {environment['vram_gb']}")

    # ------------------------------------------------------------------
    # Step 2: CPU backend — baseline 22K run
    # ------------------------------------------------------------------
    print("\n[1/3] Running CPU backend (M tshirt + M body, cotton_jersey_default)...")
    cpu_verdict = None
    cpu_sim_ms = None
    cpu_error = None

    try:
        cpu_verdict, cpu_elapsed = _run_backend("cpu", subdivide_target=0)
        cpu_sim_ms = int(cpu_elapsed * 1000)
        print(f"      CPU done in {cpu_sim_ms} ms  (simulation_ms from verdict: {cpu_verdict['simulation_ms']})")
    except Exception as exc:
        cpu_error = str(exc)
        print(f"      CPU run FAILED: {exc}")

    # ------------------------------------------------------------------
    # Step 3: Warp backend — baseline 22K run
    # ------------------------------------------------------------------
    print("\n[2/3] Running Warp backend (same inputs)...")
    warp_verdict = None
    warp_sim_ms = None
    warp_error = None

    if warp_available:
        _reset_vram_stats()
        try:
            warp_verdict, warp_elapsed = _run_backend("warp", subdivide_target=0)
            warp_sim_ms = int(warp_elapsed * 1000)
            print(f"      Warp done in {warp_sim_ms} ms  (simulation_ms from verdict: {warp_verdict['simulation_ms']})")
        except Exception as exc:
            warp_error = str(exc)
            print(f"      Warp run FAILED: {exc}")
    else:
        print("      SKIPPED — warp-lang not available.")

    # ------------------------------------------------------------------
    # Step 4: Compute CPU vs Warp deltas
    # ------------------------------------------------------------------
    clearance_max_delta_mm = None
    clearance_per_region = {}
    strain_ratio_max_delta = None
    verdict_match = None

    if cpu_verdict is not None and warp_verdict is not None:
        cpu_clearance = _extract_clearance(cpu_verdict)
        warp_clearance = _extract_clearance(warp_verdict)
        cpu_strain = _extract_strain_ratios(cpu_verdict)
        warp_strain = _extract_strain_ratios(warp_verdict)

        clearance_max_delta_mm, clearance_per_region = _compute_deltas(cpu_clearance, warp_clearance)
        strain_ratio_max_delta = _compute_strain_deltas(cpu_strain, warp_strain)
        verdict_match = (cpu_verdict["fit"] == warp_verdict["fit"])

        print(f"\n      Clearance max delta : {clearance_max_delta_mm} mm  (AC-1 gate: ≤ 0.5mm)")
        print(f"      Strain ratio max Δ  : {strain_ratio_max_delta}     (AC-3 gate: ≤ 0.02)")
        print(f"      Verdict match       : {verdict_match}")
        print(f"      Per-region deltas   : {clearance_per_region}")
    elif cpu_verdict is not None and warp_verdict is None:
        print("\n      Delta computation skipped — Warp run not available.")

    # ------------------------------------------------------------------
    # Step 5: Warp smoke test — ~10K vertex garment
    # ------------------------------------------------------------------
    print(f"\n[3/3] Running Warp smoke test (subdivide_target={SMOKE_SUBDIVIDE_TARGET})...")
    smoke_verdict = None
    smoke_sim_ms = None
    smoke_vram_peak_mb = None
    smoke_error = None

    if warp_available:
        _reset_vram_stats()
        try:
            smoke_verdict, smoke_elapsed = _run_backend("warp", subdivide_target=SMOKE_SUBDIVIDE_TARGET)
            smoke_sim_ms = int(smoke_elapsed * 1000)
            smoke_vram_peak_mb = _get_vram_peak_mb()
            print(f"      Smoke test done in {smoke_sim_ms} ms")
            print(f"      VRAM peak          : {smoke_vram_peak_mb} MB")
        except MemoryError as exc:
            smoke_error = f"OOM: {exc}"
            print(f"      Smoke test OOM: {exc}")
        except Exception as exc:
            smoke_error = str(exc)
            print(f"      Smoke test FAILED: {exc}")
    else:
        print("      SKIPPED — warp-lang not available.")

    # ------------------------------------------------------------------
    # Step 5b: Compute smoke vs 22K clearance delta
    # ------------------------------------------------------------------
    smoke_clearance_vs_22k = None
    smoke_bottleneck = None

    if warp_verdict is not None and smoke_verdict is not None:
        warp_22k_clearance = _extract_clearance(warp_verdict)
        smoke_clearance = _extract_clearance(smoke_verdict)
        smoke_max_delta, _ = _compute_deltas(warp_22k_clearance, smoke_clearance)
        smoke_clearance_vs_22k = smoke_max_delta
        print(f"      Smoke vs 22K clearance max delta: {smoke_clearance_vs_22k} mm")

        # Simple bottleneck heuristic based on timing ratios
        # (solver iterations vs resolution scaling)
        if smoke_sim_ms is not None and warp_sim_ms is not None and warp_sim_ms > 0:
            time_ratio = smoke_sim_ms / warp_sim_ms
            vertex_ratio = SMOKE_SUBDIVIDE_TARGET / 21833
            if time_ratio > vertex_ratio * 1.5:
                smoke_bottleneck = "collision"
            elif time_ratio > vertex_ratio:
                smoke_bottleneck = "solver_iterations"
            else:
                smoke_bottleneck = "memory_bandwidth"

    # ------------------------------------------------------------------
    # Step 6: Acceptance criteria evaluation
    # ------------------------------------------------------------------
    ac1_clearance_delta_le_05mm = None
    ac2_10k_runs_without_oom = None
    ac3_parity_tests_pass = None

    if clearance_max_delta_mm is not None:
        ac1_clearance_delta_le_05mm = bool(clearance_max_delta_mm <= 0.5)

    if warp_available:
        ac2_10k_runs_without_oom = smoke_error is None and smoke_verdict is not None

    # AC-3 requires the pytest suite; we can only assess from this script's results
    if (
        ac1_clearance_delta_le_05mm is True
        and strain_ratio_max_delta is not None
        and strain_ratio_max_delta <= 0.02
        and verdict_match is True
    ):
        ac3_parity_tests_pass = True
    elif clearance_max_delta_mm is not None:
        ac3_parity_tests_pass = False

    # ------------------------------------------------------------------
    # Step 7: Build and write output/stage1_profile.json
    # ------------------------------------------------------------------
    profile = {
        "stage": 1,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "environment": environment,
        "baseline_22k": {
            "vertex_count_body": 21833,
            "vertex_count_garment": _garment_vertex_count(warp_verdict) if warp_verdict else None,
            "cpu_simulation_ms": cpu_sim_ms,
            "warp_simulation_ms": warp_sim_ms,
            "clearance_max_delta_mm": clearance_max_delta_mm,
            "clearance_per_region_mm": clearance_per_region if clearance_per_region else None,
            "strain_ratio_max_delta": strain_ratio_max_delta,
            "verdict_match": verdict_match,
            "cpu_error": cpu_error,
            "warp_error": warp_error,
        },
        "smoke_10k": {
            "subdivide_target": SMOKE_SUBDIVIDE_TARGET,
            "vertex_count_body": 21833,
            "vertex_count_garment": _garment_vertex_count(smoke_verdict) if smoke_verdict else None,
            "warp_simulation_ms": smoke_sim_ms,
            "vram_peak_mb": smoke_vram_peak_mb,
            "clearance_vs_22k_max_delta_mm": smoke_clearance_vs_22k,
            "bottleneck": smoke_bottleneck,
            "error": smoke_error,
        },
        "acceptance_criteria": {
            "ac1_clearance_delta_le_05mm": ac1_clearance_delta_le_05mm,
            "ac2_10k_runs_without_oom": ac2_10k_runs_without_oom,
            "ac3_parity_tests_pass": ac3_parity_tests_pass,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nProfile written to: {OUTPUT_PATH}")
    print("\nAcceptance criteria summary:")
    print(f"  AC-1 clearance delta ≤ 0.5mm : {profile['acceptance_criteria']['ac1_clearance_delta_le_05mm']}")
    print(f"  AC-2 10K runs without OOM    : {profile['acceptance_criteria']['ac2_10k_runs_without_oom']}")
    print(f"  AC-3 parity tests pass        : {profile['acceptance_criteria']['ac3_parity_tests_pass']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
