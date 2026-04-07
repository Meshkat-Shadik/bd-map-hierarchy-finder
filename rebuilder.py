#!/usr/bin/env python3
"""
Orchestrator: runs the full V6 build pipeline.

  1. Validates input CSV and V5 data
  2. Runs 2_ingest.py  (geocoding + segments)
  3. Runs 3_build_index.py (K-way merge + binary index)
  4. Atomically swaps new files into place
  5. Creates data/.reload_signal so a running server hot-reloads

Usage (standalone):
    python rebuilder.py --input data/input.csv [--workers 2]

Called programmatically by the server's /upload endpoint.
"""

import os
import sys
import json
import time
import argparse
import csv as csv_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V5_DATA_DIR = os.environ.get("V5_DATA_DIR", os.path.join(BASE_DIR, "v5_data"))
DATA_DIR = os.path.join(BASE_DIR, "data")

REQUIRED_V5_FILES = [
    "meta.json",
    "sparse_grid.bin",
    "sparse_index.bin",
    "master_dict.json",
]


def validate_inputs(csv_path: str) -> tuple[bool, str]:
    """Return (ok, error_message)."""
    if not os.path.exists(csv_path):
        return False, f"Input CSV not found: {csv_path}"

    # Check V5 data files
    for fname in REQUIRED_V5_FILES:
        fpath = os.path.join(V5_DATA_DIR, fname)
        if not os.path.exists(fpath):
            return False, (
                f"V5 data file missing: {fpath}\n"
                f"Run the V5 pipeline first (see v5/README.md)."
            )

    # Check CSV has lat/lng columns
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv_mod.DictReader(f)
            fieldnames = [c.lower() for c in (reader.fieldnames or [])]
            has_lat = any(c in fieldnames for c in ('lat', 'latitude'))
            has_lng = any(c in fieldnames for c in ('lng', 'lon', 'long', 'longitude'))
            if not has_lat or not has_lng:
                return False, (
                    f"CSV must have lat and lng columns. "
                    f"Found: {reader.fieldnames}"
                )
    except Exception as e:
        return False, f"Could not read CSV: {e}"

    return True, ""


def signal_reload():
    """Create .reload_signal so the running server picks up new files."""
    signal_path = os.path.join(DATA_DIR, ".reload_signal")
    with open(signal_path, 'w') as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def run_pipeline(
    csv_path: str,
    n_workers: int = 2,
    chunk_size: int = 100_000,
    keep_segments: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full build pipeline.

    Returns a summary dict with timing and entity counts.
    Raises on validation failure or build error.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Validate
    ok, err = validate_inputs(csv_path)
    if not ok:
        raise ValueError(err)

    n_workers = max(1, min(int(n_workers), 4))
    chunk_size = max(10_000, int(chunk_size))

    if verbose:
        print(f"[rebuilder] Input: {csv_path}")
        print(f"[rebuilder] Workers: {n_workers}  Chunk size: {chunk_size:,}")
        print()

    t_total = time.perf_counter()

    # 2. Ingest
    from importlib.util import spec_from_file_location, module_from_spec
    ingest_mod_path = os.path.join(BASE_DIR, "2_ingest.py")
    spec = spec_from_file_location("ingest", ingest_mod_path)
    ingest_mod = module_from_spec(spec)
    spec.loader.exec_module(ingest_mod)

    ingest_result = ingest_mod.run_ingest(
        csv_path=csv_path,
        n_workers=n_workers,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    if ingest_result["valid"] == 0:
        raise ValueError(
            f"Ingest produced 0 valid records from {ingest_result['total_rows']:,} rows. "
            f"Check that coordinates are inside Bangladesh."
        )

    # 3. Build index
    build_mod_path = os.path.join(BASE_DIR, "3_build_index.py")
    spec2 = spec_from_file_location("build_index", build_mod_path)
    build_mod = module_from_spec(spec2)
    spec2.loader.exec_module(build_mod)

    build_result = build_mod.run_build(
        keep_segments=keep_segments,
        verbose=verbose,
    )

    # 4. Signal server to hot-reload
    signal_reload()

    total_elapsed = time.perf_counter() - t_total

    summary = {
        "success": True,
        "total_rows": ingest_result["total_rows"],
        "valid": ingest_result["valid"],
        "skipped": ingest_result["skipped"],
        "n_entities": build_result["n_entities"],
        "n_prefixes": build_result["n_prefixes"],
        "built_at": build_result["built_at"],
        "total_elapsed_sec": round(total_elapsed, 2),
        "ingest_sec": ingest_result["elapsed_sec"],
        "build_sec": build_result["build_duration_sec"],
    }

    if verbose:
        print()
        print("=" * 60)
        print(f"  Build complete in {total_elapsed:.1f}s")
        print(f"  Entities: {build_result['n_entities']:,}")
        print(f"  Valid:    {ingest_result['valid']:,} / {ingest_result['total_rows']:,} rows")
        print(f"  Skipped:  {ingest_result['skipped']:,} (outside Bangladesh or bad coords)")
        print(f"  Prefixes: {build_result['n_prefixes']:,} (all queryable geocode areas)")
        print("=" * 60)

    return summary


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full V6 build pipeline: ingest CSV → binary index"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV (id,lat,lng)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel geocoding workers (default: 2)")
    parser.add_argument("--chunk", type=int, default=100_000,
                        help="CSV rows per chunk (default: 100,000)")
    parser.add_argument("--keep-segments", action="store_true",
                        help="Keep intermediate segment files")
    args = parser.parse_args()

    try:
        result = run_pipeline(
            csv_path=args.input,
            n_workers=args.workers,
            chunk_size=args.chunk,
            keep_segments=args.keep_segments,
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n[rebuilder] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
