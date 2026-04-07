#!/usr/bin/env python3
"""



SMOKE003,22.3569,91.7832SMOKE002,24.3636,88.6241Step 2: Ingest CSV and geocode each row.

Reads id,lat,lng CSV in streaming 1M-row chunks, uses V5 rasterized grid to
convert each coordinate to a Bangladesh administrative GEO_CODE, then writes
sorted binary segment files for the K-way merge step.

Usage:
    python 2_ingest.py --input data/input.csv [--workers 2] [--chunk 100000]
"""

import os
import sys
import csv
import json
import mmap
import math
import struct
import argparse
import time
import multiprocessing as mp
import threading

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V5_DATA_DIR = os.environ.get("V5_DATA_DIR", os.path.join(BASE_DIR, "v5_data"))
DATA_DIR = os.path.join(BASE_DIR, "data")
SEGMENTS_DIR = os.path.join(DATA_DIR, "segments")

# Numpy dtype for a segment record
SEGMENT_DTYPE = np.dtype([
    ('geocode', 'S24'),   # null-padded ASCII geocode, max 22 chars
    ('lat', 'f4'),
    ('lng', 'f4'),
])


# =============================================================================
# Worker initializer — opens V5 mmap handles once per worker process
# =============================================================================

_worker_meta = None
_worker_grid = None
_worker_sparse_index = None
_worker_master_dict = None


def _worker_init():
    global _worker_meta, _worker_grid, _worker_sparse_index, _worker_master_dict

    with open(os.path.join(V5_DATA_DIR, "meta.json")) as f:
        _worker_meta = json.load(f)

    f_grid = open(os.path.join(V5_DATA_DIR, "sparse_grid.bin"), "rb")
    _worker_grid = mmap.mmap(f_grid.fileno(), 0, access=mmap.ACCESS_READ)

    f_idx = open(os.path.join(V5_DATA_DIR, "sparse_index.bin"), "rb")
    idx_mm = mmap.mmap(f_idx.fileno(), 0, access=mmap.ACCESS_READ)
    tiles_rows = _worker_meta["tiles_rows"]
    tiles_cols = _worker_meta["tiles_cols"]
    _worker_sparse_index = np.ndarray(
        shape=(tiles_rows, tiles_cols), dtype=np.int32, buffer=idx_mm
    )

    with open(os.path.join(V5_DATA_DIR, "master_dict.json")) as f:
        _worker_master_dict = json.load(f)


def _geocode_one(lat: float, lng: float) -> str:
    """Convert a coordinate to GEO_CODE using V5 rasterized grid."""
    m = _worker_meta
    row = int((m["max_lat"] - lat) / m["cell_size"])
    col = int((lng - m["min_lng"]) / m["cell_size"])

    if row < 0 or row >= m["rows"] or col < 0 or col >= m["cols"]:
        return ""

    tile_size = m["tile_size"]
    tile_r = row // tile_size
    tile_c = col // tile_size
    tile_offset = int(_worker_sparse_index[tile_r, tile_c])

    if tile_offset == -1:
        return ""

    sub_r = row % tile_size
    sub_c = col % tile_size
    byte_start = tile_offset * (tile_size * tile_size * 4) + ((sub_r * tile_size + sub_c) * 4)
    loc_id = int.from_bytes(_worker_grid[byte_start:byte_start + 4], 'little')

    if loc_id == 0 or loc_id >= len(_worker_master_dict):
        return ""

    item = _worker_master_dict[loc_id]
    return item.get("GEO_CODE", "") if item else ""


def _geocode_batch(args):
    """
    Process one chunk of CSV rows.

    Args:
        args: (chunk_idx, rows) where rows is list of (raw_id, lat_str, lng_str)

    Returns:
        (chunk_idx, geocodes_array, ids_list, valid_count, skipped_count)
    """
    chunk_idx, rows = args
    valid = []
    skipped = 0

    for raw_id, lat_str, lng_str in rows:
        try:
            lat = float(lat_str)
            lng = float(lng_str)
        except (ValueError, TypeError):
            skipped += 1
            continue

        geocode = _geocode_one(lat, lng)
        if not geocode:
            skipped += 1
            continue

        gc_bytes = geocode.encode('ascii').ljust(24, b'\x00')
        valid.append((gc_bytes, np.float32(lat), np.float32(lng), raw_id))

    if not valid:
        return chunk_idx, None, [], 0, skipped

    # Sort by geocode bytes within chunk — reduces work in K-way merge
    valid.sort(key=lambda x: x[0])

    n = len(valid)
    arr = np.empty(n, dtype=SEGMENT_DTYPE)
    ids = []
    for i, (gc_bytes, lat, lng, raw_id) in enumerate(valid):
        arr[i]['geocode'] = gc_bytes
        arr[i]['lat'] = lat
        arr[i]['lng'] = lng
        ids.append(raw_id)

    return chunk_idx, arr, ids, n, skipped


# =============================================================================
# CSV chunk reader (streaming, never loads whole file)
# =============================================================================

def _read_chunks(csv_path: str, chunk_size: int, lat_col: str, lng_col: str, id_col: str):
    """Yield (chunk_idx, [(id, lat_str, lng_str), ...]) tuples."""
    chunk_idx = 0
    buf = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Flexible column detection
        actual_lat = lat_col if lat_col in fieldnames else next(
            (c for c in fieldnames if c.lower() in ('lat', 'latitude')), None)
        actual_lng = lng_col if lng_col in fieldnames else next(
            (c for c in fieldnames if c.lower() in ('lng', 'lon', 'long', 'longitude')), None)
        actual_id = id_col if id_col in fieldnames else next(
            (c for c in fieldnames if c.lower() in ('id', 'school_id', 'hospital_id', 'entity_id')), None)

        if not actual_lat or not actual_lng:
            raise ValueError(f"Could not find lat/lng columns in {fieldnames}")

        for i, row in enumerate(reader):
            raw_id = row.get(actual_id, f"ITEM{i:08d}") if actual_id else f"ITEM{i:08d}"
            buf.append((raw_id, row.get(actual_lat, ''), row.get(actual_lng, '')))

            if len(buf) >= chunk_size:
                yield chunk_idx, buf
                chunk_idx += 1
                buf = []

    if buf:
        yield chunk_idx, buf


# =============================================================================
# Main ingest function
# =============================================================================

def run_ingest(csv_path: str, n_workers: int = 2, chunk_size: int = 100_000,
               lat_col: str = 'lat', lng_col: str = 'lng', id_col: str = 'id',
               verbose: bool = True) -> dict:
    """
    Ingest a CSV file and write sorted segment files.

    Returns a summary dict with n_segments, total_rows, valid, skipped.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    # Remove any existing segments
    for f in os.listdir(SEGMENTS_DIR):
        if f.startswith('seg_') or f == 'manifest.json':
            os.remove(os.path.join(SEGMENTS_DIR, f))

    t0 = time.perf_counter()
    total_rows = 0
    total_valid = 0
    total_skipped = 0
    n_segments = 0

    # Keep multiprocessing conservative by default to avoid memory pressure.
    n_workers = max(1, min(int(n_workers), 4))
    chunk_size = max(10_000, int(chunk_size))

    # Stream chunks directly from disk to keep peak RAM stable on large CSVs.
    chunk_gen = _read_chunks(csv_path, chunk_size, lat_col, lng_col, id_col)

    # Multiprocessing requires this module to be importable by child processes.
    # When loaded dynamically by rebuilder (module name: ingest), Python 3.13
    # can fail to pickle worker functions; run single-process in that mode.
    use_pool = (
        n_workers > 1
        and threading.current_thread() is threading.main_thread()
        and __name__ == "__main__"
    )

    if not use_pool:
        # FastAPI background jobs run in threads; avoid forking there on macOS.
        _worker_init()
        for chunk_idx, rows in chunk_gen:
            _, arr, ids, valid, skipped = _geocode_batch((chunk_idx, rows))
            total_rows += valid + skipped
            total_valid += valid
            total_skipped += skipped

            if arr is None:
                continue

            seg_path = os.path.join(SEGMENTS_DIR, f"seg_{n_segments:06d}")
            np.save(seg_path + ".npy", arr)
            with open(seg_path + ".ids", 'w', encoding='utf-8') as f:
                f.write('\n'.join(ids))
            n_segments += 1

            if verbose:
                elapsed = time.perf_counter() - t0
                rate = total_rows / elapsed if elapsed > 0 else 0
                print(f"\r[ingest] {total_rows:>10,} rows  {total_valid:>10,} valid  "
                      f"{total_skipped:>8,} skipped  {rate:>8,.0f} rows/s",
                      end='', flush=True)
    else:
        # Use 'fork' in main-thread CLI mode to avoid spawn pickling issues.
        ctx = mp.get_context('fork')

        with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
            for chunk_idx, arr, ids, valid, skipped in pool.imap_unordered(
                _geocode_batch, chunk_gen, chunksize=1
            ):
                total_rows += valid + skipped
                total_valid += valid
                total_skipped += skipped

                if arr is None:
                    continue

                seg_path = os.path.join(SEGMENTS_DIR, f"seg_{n_segments:06d}")
                np.save(seg_path + ".npy", arr)
                with open(seg_path + ".ids", 'w', encoding='utf-8') as f:
                    f.write('\n'.join(ids))
                n_segments += 1

                if verbose:
                    elapsed = time.perf_counter() - t0
                    rate = total_rows / elapsed if elapsed > 0 else 0
                    print(f"\r[ingest] {total_rows:>10,} rows  {total_valid:>10,} valid  "
                          f"{total_skipped:>8,} skipped  {rate:>8,.0f} rows/s", end='', flush=True)

    if verbose:
        print()

    manifest = {
        "n_segments": n_segments,
        "total_rows": total_rows,
        "valid": total_valid,
        "skipped": total_skipped,
        "source_csv": os.path.abspath(csv_path),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    with open(os.path.join(SEGMENTS_DIR, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"[ingest] Done: {total_valid:,} valid / {total_rows:,} total rows "
              f"in {manifest['elapsed_sec']:.1f}s → {n_segments} segments")

    return manifest


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CSV into V6 segment files")
    parser.add_argument("--input", required=True, help="Path to input CSV (id,lat,lng)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--chunk", type=int, default=100_000, help="Rows per chunk")
    parser.add_argument("--lat", default="lat", help="Latitude column name")
    parser.add_argument("--lng", default="lng", help="Longitude column name")
    parser.add_argument("--id", default="id", help="ID column name")
    args = parser.parse_args()

    run_ingest(
        csv_path=args.input,
        n_workers=args.workers,
        chunk_size=args.chunk,
        lat_col=args.lat,
        lng_col=args.lng,
        id_col=args.id,
    )
