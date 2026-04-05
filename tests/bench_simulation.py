#!/usr/bin/env python3
"""
bench_simulation.py — Performance benchmark for Forma simulation backends.

Standalone script (not pytest). Run with:
    python3 bench_simulation.py

Benchmarks:
  - CPU backend (Phase 1 XPBD): N=5 runs, reports mean/min/max ms
  - Warp backend (GPU): if CUDA available, same N=5, reports speedup vs CPU
  - HOOD backend (ContourCraft): if CONTOURCRAFT_ROOT env var set, same N=5, reports speedup

Output: formatted results table + JSON save to .claude-flow/data/bench_latest.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Setup paths
ROOT = Path(__file__).parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Test assets
BODY_PLY = ROOT / "data" / "bodies" / "makehuman_male_M.ply"
PATTERN_M = ROOT / "data" / "patterns" / "tshirt_size_M.json"
MANIFEST_M = ROOT / "seam_manifests" / "tshirt_size_M_manifest.json"
FABRIC_ID = "cotton_jersey_default"

# Output
OUTPUT_DIR = ROOT / ".claude-flow" / "data"


def load_fabric_params(fabric_id: str) -> dict:
    """Load fabric params from fabric_library.json."""
    fabric_lib = ROOT / "data" / "fabrics" / "fabric_library.json"
    with open(fabric_lib) as f:
        lib = json.load(f)
    if fabric_id not in lib["fabrics"]:
        raise ValueError(f"Unknown fabric_id: {fabric_id}")
    return lib["fabrics"][fabric_id]


def benchmark_backend(
    backend: str,
    n_runs: int = 5,
) -> dict:
    """
    Benchmark a single backend.

    Parameters
    ----------
    backend : str
        "cpu", "warp", or "hood"
    n_runs : int
        Number of runs to average (default 5)

    Returns
    -------
    dict with keys: backend, status, n_runs, times_ms, mean_ms, min_ms, max_ms
           If status != "success", only backend and status are set.
    """
    result = {
        "backend": backend,
        "status": "pending",
        "n_runs": n_runs,
        "times_ms": [],
        "mean_ms": None,
        "min_ms": None,
        "max_ms": None,
    }

    # Try to import the necessary modules first
    if backend == "warp":
        try:
            import warp as wp
            wp.force_load()  # Ensure GPU is available
        except ImportError:
            result["status"] = "skip_missing_warp"
            return result
        except Exception as e:
            result["status"] = f"skip_no_cuda"
            return result

    elif backend == "hood":
        contourcraft_root = os.environ.get("CONTOURCRAFT_ROOT")
        if not contourcraft_root:
            result["status"] = "skip_no_contourcraft_root"
            return result
        try:
            # Verify HOOD module is available
            from geometer.hood.hood_simulate import run_simulation_hood  # noqa
        except ImportError:
            result["status"] = "skip_missing_hood"
            return result

    elif backend != "cpu":
        result["status"] = f"error_unknown_backend"
        return result

    # Load assets
    try:
        from pipeline import run_fit_check

        fabric_params = load_fabric_params(FABRIC_ID)

        # Verify assets exist
        for path, label in [
            (BODY_PLY, "Body mesh"),
            (PATTERN_M, "Pattern"),
            (MANIFEST_M, "Seam manifest"),
        ]:
            if not path.exists():
                result["status"] = f"error_missing_{label.lower().replace(' ', '_')}"
                return result

    except Exception as e:
        result["status"] = f"error_import: {str(e)}"
        return result

    # Run benchmark
    times_ms = []
    for i in range(n_runs):
        try:
            start = time.perf_counter()
            verdict = run_fit_check(
                body_mesh_path=str(BODY_PLY),
                pattern_path=str(PATTERN_M),
                seam_manifest_path=str(MANIFEST_M),
                fabric_id=FABRIC_ID,
                backend=backend,
            )
            elapsed_s = time.perf_counter() - start
            times_ms.append(elapsed_s * 1000.0)

            # Validate verdict has required fields
            if not isinstance(verdict, dict):
                raise ValueError("verdict is not a dict")
            if "fit" not in verdict:
                raise ValueError("verdict missing 'fit' field")

        except Exception as e:
            result["status"] = f"error_run_{i+1}: {str(e)[:60]}"
            return result

    # Compute statistics
    result["times_ms"] = times_ms
    result["mean_ms"] = sum(times_ms) / len(times_ms)
    result["min_ms"] = min(times_ms)
    result["max_ms"] = max(times_ms)
    result["status"] = "success"

    return result


def format_results_table(results: list[dict]) -> str:
    """Format benchmark results as a table."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("FORMA PERFORMANCE BENCHMARK")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Test Assets:")
    lines.append(f"  Body:     {BODY_PLY.name}")
    lines.append(f"  Pattern:  {PATTERN_M.name}")
    lines.append(f"  Manifest: {MANIFEST_M.name}")
    lines.append(f"  Fabric:   {FABRIC_ID}")
    lines.append("")
    lines.append("-" * 80)
    lines.append(
        f"{'Backend':<12} {'Status':<18} {'Runs':<6} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Speedup':<8}"
    )
    lines.append("-" * 80)

    cpu_mean = None
    for result in results:
        backend = result["backend"]
        status = result["status"]
        n_runs = result.get("n_runs", "-")
        mean_ms = result.get("mean_ms")
        min_ms = result.get("min_ms")
        max_ms = result.get("max_ms")

        if status == "success":
            cpu_mean = mean_ms if backend == "cpu" else cpu_mean
            mean_str = f"{mean_ms:.2f}"
            min_str = f"{min_ms:.2f}"
            max_str = f"{max_ms:.2f}"
            speedup_str = (
                "1.0x"
                if backend == "cpu"
                else f"{cpu_mean / mean_ms:.2f}x" if cpu_mean else "n/a"
            )
        else:
            mean_str = "—"
            min_str = "—"
            max_str = "—"
            speedup_str = "—"

        lines.append(
            f"{backend:<12} {status:<18} {str(n_runs):<6} {mean_str:<12} {min_str:<12} {max_str:<12} {speedup_str:<8}"
        )

    lines.append("-" * 80)
    lines.append("")

    return "\n".join(lines)


def main():
    """Run benchmarks and report results."""
    print("\nInitializing Forma performance benchmark...\n")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    print("Running CPU backend benchmark (N=5)...")
    cpu_result = benchmark_backend("cpu", n_runs=5)
    print(f"  Status: {cpu_result['status']}")

    print("Attempting Warp backend benchmark (N=5)...")
    warp_result = benchmark_backend("warp", n_runs=5)
    print(f"  Status: {warp_result['status']}")

    print("Attempting HOOD backend benchmark (N=5)...")
    hood_result = benchmark_backend("hood", n_runs=5)
    print(f"  Status: {hood_result['status']}")

    results = [cpu_result, warp_result, hood_result]

    # Print formatted table
    table = format_results_table(results)
    print(table)

    # Save to JSON
    output_file = OUTPUT_DIR / "bench_latest.json"
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_assets": {
            "body": str(BODY_PLY),
            "pattern": str(PATTERN_M),
            "manifest": str(MANIFEST_M),
            "fabric_id": FABRIC_ID,
        },
        "results": results,
    }

    try:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"WARNING: Failed to save results: {e}", file=sys.stderr)

    # Exit with error if CPU benchmark failed
    if cpu_result["status"] != "success":
        print(
            f"\nERROR: CPU backend benchmark failed. Cannot proceed.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
