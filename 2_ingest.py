#!/usr/bin/env python3
"""
Step 2: Ingest CSV and geocode each row.

Reads id,lat,lng (+optional extra fields) CSV in streaming chunks, uses V5
rasterized grid to convert each coordinate to a Bangladesh administrative
GEO_CODE, then writes sorted binary segment files for the K-way merge step.

Extra fields (if any) are saved as id→{field:value} JSON alongside each segment.

Usage:
    python 2_ingest.py --input data/input.csv [--workers 2] [--extra field1,field2]
"""

import os
import csv
import json
import mmap
import argparse
import time
import multiprocessing as mp
import threading

import numpy as np

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
V5_DATA_DIR  = os.environ.get("V5_DATA_DIR", os.path.join(BASE_DIR, "v5_data"))
DATA_DIR     = os.path.join(BASE_DIR, "data")
SEGMENTS_DIR = os.path.join(DATA_DIR, "segments")   # module-level default only

SEGMENT_DTYPE = np.dtype([
    ('geocode', 'S24'),
    ('lat', 'f4'),
    ('lng', 'f4'),
])

# =============================================================================
# Worker initializer
# =============================================================================

_worker_meta         = None
_worker_grid         = None
_worker_sparse_index = None
_worker_master_dict  = None


def _worker_init():
    global _worker_meta, _worker_grid, _worker_sparse_index, _worker_master_dict

    with open(os.path.join(V5_DATA_DIR, "meta.json")) as f:
        _worker_meta = json.load(f)

    f_grid = open(os.path.join(V5_DATA_DIR, "sparse_grid.bin"), "rb")
    _worker_grid = mmap.mmap(f_grid.fileno(), 0, access=mmap.ACCESS_READ)

    f_idx = open(os.path.join(V5_DATA_DIR, "sparse_index.bin"), "rb")
    idx_mm = mmap.mmap(f_idx.fileno(), 0, access=mmap.ACCESS_READ)
    _worker_sparse_index = np.ndarray(
        shape=(_worker_meta["tiles_rows"], _worker_meta["tiles_cols"]),
        dtype=np.int32, buffer=idx_mm
    )

    with open(os.path.join(V5_DATA_DIR, "master_dict.json")) as f:
        _worker_master_dict = json.load(f)


def _geocode_one(lat: float, lng: float) -> str:
    m         = _worker_meta
    row       = int((m["max_lat"] - lat) / m["cell_size"])
    col       = int((lng - m["min_lng"]) / m["cell_size"])

    if row < 0 or row >= m["rows"] or col < 0 or col >= m["cols"]:
        return ""

    tile_size   = m["tile_size"]
    tile_r      = row // tile_size
    tile_c      = col // tile_size
    tile_offset = int(_worker_sparse_index[tile_r, tile_c])

    if tile_offset == -1:
        return ""

    sub_r      = row % tile_size
    sub_c      = col % tile_size
    byte_start = tile_offset * (tile_size * tile_size * 4) + ((sub_r * tile_size + sub_c) * 4)
    loc_id     = int.from_bytes(_worker_grid[byte_start:byte_start + 4], 'little')

    if loc_id == 0 or loc_id >= len(_worker_master_dict):
        return ""

    item = _worker_master_dict[loc_id]
    return item.get("GEO_CODE", "") if item else ""


def _geocode_batch(args):
    """
    Process one chunk of CSV rows.
    args: (chunk_idx, rows)
    rows: list of (raw_id, lat_str, lng_str, extra_dict)
    Returns: (chunk_idx, arr, ids, extras, valid_count, skipped_count)
    """
    chunk_idx, rows = args
    valid   = []
    skipped = 0

    for raw_id, lat_str, lng_str, extra in rows:
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
        valid.append((gc_bytes, np.float32(lat), np.float32(lng), raw_id, extra))

    if not valid:
        return chunk_idx, None, [], [], 0, skipped

    valid.sort(key=lambda x: x[0])

    n      = len(valid)
    arr    = np.empty(n, dtype=SEGMENT_DTYPE)
    ids    = []
    extras = []
    for i, (gc_bytes, lat, lng, raw_id, extra) in enumerate(valid):
        arr[i]['geocode'] = gc_bytes
        arr[i]['lat']     = lat
        arr[i]['lng']     = lng
        ids.append(raw_id)
        extras.append(extra)

    return chunk_idx, arr, ids, extras, n, skipped


# =============================================================================
# CSV chunk reader
# =============================================================================

def _read_chunks(csv_path, chunk_size, lat_col, lng_col, id_col, extra_fields):
    """Yield (chunk_idx, [(id, lat_str, lng_str, extra_dict), ...]) tuples."""
    chunk_idx = 0
    buf       = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        actual_lat = lat_col if lat_col in fieldnames else next(
            (c for c in fieldnames if c.lower() in ('lat', 'latitude')), None)
        actual_lng = lng_col if lng_col in fieldnames else next(
            (c for c in fieldnames if c.lower() in ('lng', 'lon', 'long', 'longitude')), None)
        actual_id  = id_col if id_col in fieldnames else next(
            (c for c in fieldnames if c.lower() in ('id', 'school_id', 'hospital_id', 'entity_id')), None)

        if not actual_lat or not actual_lng:
            raise ValueError(f"Could not find lat/lng columns in {fieldnames}")

        # Only include extra fields that exist in the CSV
        resolved_extras = [f for f in (extra_fields or []) if f in fieldnames]

        for i, row in enumerate(reader):
            raw_id = row.get(actual_id, f"ITEM{i:08d}") if actual_id else f"ITEM{i:08d}"
            extra  = {f: row.get(f, '') for f in resolved_extras}
            buf.append((raw_id, row.get(actual_lat, ''), row.get(actual_lng, ''), extra))

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
               extra_fields: list = None, data_dir: str = None, verbose: bool = True) -> dict:
    """
    Ingest a CSV file and write sorted segment files.
    extra_fields: list of column names to capture as entity metadata.
    data_dir: override destination directory (defaults to module-level DATA_DIR).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    effective_data_dir = data_dir or DATA_DIR
    segments_dir = os.path.join(effective_data_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    for fname in os.listdir(segments_dir):
        if fname.startswith('seg_') or fname == 'manifest.json':
            os.remove(os.path.join(segments_dir, fname))

    extra_fields  = list(extra_fields or [])
    t0            = time.perf_counter()
    total_valid   = 0
    total_skipped = 0
    n_segments    = 0

    n_workers  = max(1, min(int(n_workers), 4))
    chunk_size = max(10_000, int(chunk_size))

    chunk_gen = _read_chunks(csv_path, chunk_size, lat_col, lng_col, id_col, extra_fields)

    use_pool = (
        n_workers > 1
        and threading.current_thread() is threading.main_thread()
        and __name__ == "__main__"
    )

    def _save_segment(arr, ids, extras):
        nonlocal n_segments
        seg_path = os.path.join(segments_dir, f"seg_{n_segments:06d}")
        np.save(seg_path + ".npy", arr)
        with open(seg_path + ".ids", 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(ids))
        if extra_fields and extras:
            with open(seg_path + ".meta", 'w', encoding='utf-8') as fh:
                json.dump(dict(zip(ids, extras)), fh)
        n_segments += 1

    if not use_pool:
        _worker_init()
        for chunk_idx, rows in chunk_gen:
            _, arr, ids, extras, valid, skipped = _geocode_batch((chunk_idx, rows))
            total_valid   += valid
            total_skipped += skipped
            if arr is None:
                continue
            _save_segment(arr, ids, extras)
            if verbose:
                elapsed = time.perf_counter() - t0
                total   = total_valid + total_skipped
                rate    = total / elapsed if elapsed > 0 else 0
                print(f"\r[ingest] {total:>10,} rows  {total_valid:>10,} valid  "
                      f"{total_skipped:>8,} skipped  {rate:>8,.0f} rows/s",
                      end='', flush=True)
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
            for chunk_idx, arr, ids, extras, valid, skipped in pool.imap_unordered(
                _geocode_batch, chunk_gen, chunksize=1
            ):
                total_valid   += valid
                total_skipped += skipped
                if arr is None:
                    continue
                _save_segment(arr, ids, extras)
                if verbose:
                    elapsed = time.perf_counter() - t0
                    total   = total_valid + total_skipped
                    rate    = total / elapsed if elapsed > 0 else 0
                    print(f"\r[ingest] {total:>10,} rows  {total_valid:>10,} valid  "
                          f"{total_skipped:>8,} skipped  {rate:>8,.0f} rows/s", end='', flush=True)

    if verbose:
        print()

    total_rows = total_valid + total_skipped
    manifest   = {
        "n_segments":   n_segments,
        "total_rows":   total_rows,
        "valid":        total_valid,
        "skipped":      total_skipped,
        "extra_fields": extra_fields,
        "source_csv":   os.path.abspath(csv_path),
        "elapsed_sec":  round(time.perf_counter() - t0, 2),
    }
    with open(os.path.join(segments_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"[ingest] Done: {total_valid:,} valid / {total_rows:,} total rows "
              f"in {manifest['elapsed_sec']:.1f}s → {n_segments} segments "
              f"({len(extra_fields)} extra fields)")

    return manifest


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CSV into V6 segment files")
    parser.add_argument("--input",   required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk",   type=int, default=100_000)
    parser.add_argument("--lat",     default="lat")
    parser.add_argument("--lng",     default="lng")
    parser.add_argument("--id",      default="id")
    parser.add_argument("--extra",   default="", help="Comma-separated extra field names")
    args = parser.parse_args()

    extra = [f.strip() for f in args.extra.split(',') if f.strip()] if args.extra else []
    run_ingest(csv_path=args.input, n_workers=args.workers, chunk_size=args.chunk,
               lat_col=args.lat, lng_col=args.lng, id_col=args.id, extra_fields=extra)
