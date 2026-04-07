#!/usr/bin/env python3
"""
Orchestrator: runs the full V6 build pipeline.

  1. Validates input CSV and V5 data
  2. Runs 2_ingest.py  (geocoding + segments, captures extra fields)
  3. Runs 3_build_index.py (K-way merge + binary index + metadata)
  4. Atomically swaps new files into place
  5. Creates data/.reload_signal so a running server hot-reloads

Usage (standalone):
    python rebuilder.py --input data/input.csv [--workers 2] [--extra field1,field2]
"""

import os
import sys
import json
import time
import argparse
import csv as csv_mod

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
V5_DATA_DIR = os.environ.get("V5_DATA_DIR", os.path.join(BASE_DIR, "v5_data"))
DATA_DIR    = os.path.join(BASE_DIR, "data")

REQUIRED_V5_FILES = [
    "meta.json",
    "sparse_grid.bin",
    "sparse_index.bin",
    "master_dict.json",
]


def validate_inputs(csv_path: str) -> tuple[bool, str]:
    if not os.path.exists(csv_path):
        return False, f"Input CSV not found: {csv_path}"

    for fname in REQUIRED_V5_FILES:
        fpath = os.path.join(V5_DATA_DIR, fname)
        if not os.path.exists(fpath):
            return False, f"V5 data file missing: {fpath}"

    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader     = csv_mod.DictReader(f)
            fieldnames = [c.lower() for c in (reader.fieldnames or [])]
            has_lat = any(c in fieldnames for c in ('lat', 'latitude'))
            has_lng = any(c in fieldnames for c in ('lng', 'lon', 'long', 'longitude'))
            if not has_lat or not has_lng:
                return False, f"CSV must have lat and lng columns. Found: {reader.fieldnames}"
    except Exception as e:
        return False, f"Could not read CSV: {e}"

    return True, ""


def signal_reload():
    signal_path = os.path.join(DATA_DIR, ".reload_signal")
    with open(signal_path, 'w') as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def run_pipeline(
    csv_path: str,
    n_workers: int = 2,
    chunk_size: int = 100_000,
    extra_fields: list = None,
    display_field: str = 'id',
    keep_segments: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full build pipeline.
    extra_fields: list of CSV column names to store as entity metadata.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    ok, err = validate_inputs(csv_path)
    if not ok:
        raise ValueError(err)

    extra_fields = list(extra_fields or [])
    n_workers    = max(1, min(int(n_workers), 4))
    chunk_size   = max(10_000, int(chunk_size))

    if verbose:
        print(f"[rebuilder] Input:        {csv_path}")
        print(f"[rebuilder] Workers:      {n_workers}  Chunk size: {chunk_size:,}")
        print(f"[rebuilder] Extra fields: {extra_fields or '(none)'}")
        print()

    t_total = time.perf_counter()

    # ── 2. Ingest ─────────────────────────────────────────────────────────────
    from importlib.util import spec_from_file_location, module_from_spec

    spec       = spec_from_file_location("ingest", os.path.join(BASE_DIR, "2_ingest.py"))
    ingest_mod = module_from_spec(spec)
    spec.loader.exec_module(ingest_mod)

    ingest_result = ingest_mod.run_ingest(
        csv_path=csv_path,
        n_workers=n_workers,
        chunk_size=chunk_size,
        extra_fields=extra_fields,
        verbose=verbose,
    )

    if ingest_result["valid"] == 0:
        raise ValueError(
            f"Ingest produced 0 valid records from {ingest_result['total_rows']:,} rows. "
            f"Check that coordinates are inside Bangladesh."
        )

    # ── 3. Build index ────────────────────────────────────────────────────────
    spec2      = spec_from_file_location("build_index", os.path.join(BASE_DIR, "3_build_index.py"))
    build_mod  = module_from_spec(spec2)
    spec2.loader.exec_module(build_mod)

    build_result = build_mod.run_build(
        keep_segments=keep_segments,
        schema_fields=extra_fields,
        display_field=display_field,
        verbose=verbose,
    )

    # ── 4. Signal hot-reload ──────────────────────────────────────────────────
    signal_reload()

    total_elapsed = time.perf_counter() - t_total

    summary = {
        "success":           True,
        "total_rows":        ingest_result["total_rows"],
        "valid":             ingest_result["valid"],
        "skipped":           ingest_result["skipped"],
        "extra_fields":      extra_fields,
        "n_entities":        build_result["n_entities"],
        "n_prefixes":        build_result["n_prefixes"],
        "built_at":          build_result["built_at"],
        "total_elapsed_sec": round(total_elapsed, 2),
        "ingest_sec":        ingest_result["elapsed_sec"],
        "build_sec":         build_result["build_duration_sec"],
    }

    if verbose:
        print()
        print("=" * 60)
        print(f"  Build complete in {total_elapsed:.1f}s")
        print(f"  Entities:     {build_result['n_entities']:,}")
        print(f"  Valid:        {ingest_result['valid']:,} / {ingest_result['total_rows']:,} rows")
        print(f"  Skipped:      {ingest_result['skipped']:,}")
        print(f"  Extra fields: {extra_fields or '(none)'}")
        print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk",   type=int, default=100_000)
    parser.add_argument("--extra",   default="", help="Comma-separated extra field names")
    parser.add_argument("--keep-segments", action="store_true")
    args = parser.parse_args()

    extra = [f.strip() for f in args.extra.split(',') if f.strip()] if args.extra else []

    try:
        run_pipeline(csv_path=args.input, n_workers=args.workers,
                     chunk_size=args.chunk, extra_fields=extra,
                     keep_segments=args.keep_segments)
        sys.exit(0)
    except Exception as e:
        print(f"\n[rebuilder] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
