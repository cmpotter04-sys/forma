import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline import run_fit_check
from tailor.seam_converter import SeamValidationError
from geometer.convergence import SimulationExplosionError


def _build_summary(verdict: dict) -> str:
    parts = [f"fit={verdict['fit']}"]
    for entry in verdict.get("strain_map", []):
        region = entry["region"]
        delta = entry["delta_mm"]
        severity = entry["severity"]
        sign = "+" if delta >= 0 else ""
        parts.append(f"{region}={sign}{delta:.1f}mm({severity})")
    return "  ".join(parts)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forma",
        description="Forma fit-check CLI",
    )
    sub = p.add_subparsers(dest="command")

    fc = sub.add_parser("fit-check", help="Run a garment fit check")
    fc.add_argument("--body", required=True, metavar="PATH",
                    help="Path to body PLY mesh")
    fc.add_argument("--pattern", required=True, metavar="PATH",
                    help="Path to GarmentCode JSON pattern")
    fc.add_argument("--seam-manifest", required=True, metavar="PATH",
                    help="Path to seam manifest JSON")
    fc.add_argument("--fabric", default="cotton_jersey_default", metavar="FABRIC_ID",
                    help="Fabric ID from fabric_library.json (default: cotton_jersey_default)")
    fc.add_argument("--backend", default="cpu", choices=["cpu", "warp", "hood"],
                    help="Simulation backend (default: cpu)")
    fc.add_argument("--output", default=None, metavar="PATH",
                    help="Write verdict JSON to this path (default: stdout)")
    fc.add_argument("--quiet", "-q", action="store_true",
                    help="Suppress progress output; only emit JSON result")

    return p


def main():
    parser = _parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "fit-check":
        try:
            verdict = run_fit_check(
                body_mesh_path=args.body,
                pattern_path=args.pattern,
                seam_manifest_path=args.seam_manifest,
                fabric_id=args.fabric,
                backend=args.backend,
            )
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        except SeamValidationError as e:
            print(f"ERROR: seam validation failed — {e}", file=sys.stderr)
            sys.exit(3)
        except SimulationExplosionError as e:
            print(f"ERROR: simulation diverged — {e}", file=sys.stderr)
            sys.exit(4)
        except ImportError as e:
            print(f"ERROR: backend not available — {e}", file=sys.stderr)
            sys.exit(5)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(4)

        verdict_json = json.dumps(verdict, indent=2)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(verdict_json)
            if not args.quiet:
                print(f"Verdict written to {out_path}", file=sys.stderr)
        else:
            print(verdict_json)

        print(_build_summary(verdict), file=sys.stderr)


if __name__ == "__main__":
    main()
