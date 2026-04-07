#!/usr/bin/env python3
"""
Step 3: K-way merge all segments and build production binary index files.

Reads sorted segment files from data/segments/, merges them in a single pass,
and produces:
  data/entities_sorted.bin   — 40-byte fixed records sorted by geocode
  data/entity_id_pool.bin    — flat byte pool of entity ID strings
  data/entity_id_index.bin   — FNV1a-64 hash index for reverse lookup
  data/geocode_keys.bin      — sorted geocode prefixes (parallel arrays)
  data/prefix_starts.bin     — start record index for each prefix
  data/prefix_counts.bin     — count of records under each prefix
  data/build_meta.json       — build metadata

Usage:
    python 3_build_index.py [--keep-segments]
"""

import os
import sys
import json
import heapq
import struct
import argparse
import time
import shutil

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SEGMENTS_DIR = os.path.join(DATA_DIR, "segments")

# Canonical hierarchy levels by GEO_CODE prefix length
CANONICAL_BREAKS = [2, 4, 8, 13, 16, 20, 22]

LEVEL_BY_LEN = {
    2: 'division',
    4: 'district',
    6: 'citycorporation',
    8: 'upazila',
    10: 'municipality',
    13: 'union',
    16: 'mauza',
    20: 'village',
    22: 'ea',
}

# Entity record: 40 bytes fixed
RECORD_SIZE = 40
RECORD_FMT = '24s f f I H H'   # geocode(24) lat(4) lng(4) id_offset(4) id_len(2) pad(2)

# Entity ID index slot: 20 bytes fixed
SLOT_SIZE = 20
SLOT_DTYPE = np.dtype([
    ('h', np.uint64),
    ('pos', np.uint32),
    ('id_offset', np.uint32),
    ('id_len', np.uint16),
    ('pad', np.uint16),
])


# =============================================================================
# FNV1a-64 hash (pure Python, no deps)
# =============================================================================

_FNV_PRIME = 0x00000100000001B3
_FNV_OFFSET = 0xcbf29ce484222325


def fnv1a_64(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


# =============================================================================
# Segment iterator
# =============================================================================

def _iter_segment(npy_path: str, ids_path: str):
    """Yield (geocode_bytes, lat_f32, lng_f32, entity_id_str) from one segment."""
    arr = np.load(npy_path)
    with open(ids_path, 'r', encoding='utf-8') as f:
        ids = f.read().splitlines()

    for i in range(len(arr)):
        gc_bytes = bytes(arr[i]['geocode'])  # 24-byte null-padded
        yield gc_bytes, arr[i]['lat'], arr[i]['lng'], ids[i]


# =============================================================================
# Main build function
# =============================================================================

def run_build(keep_segments: bool = False, verbose: bool = True) -> dict:
    """
    K-way merge all segment files and write production binary index.
    Returns build_meta dict.
    """
    manifest_path = os.path.join(SEGMENTS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Segments manifest not found at {manifest_path}. Run 2_ingest.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_segments = manifest["n_segments"]
    if n_segments == 0:
        raise ValueError("No segments found. Ingest produced 0 valid records.")

    if verbose:
        print(f"[build] Merging {n_segments} segments "
              f"({manifest['valid']:,} valid entities)...")

    t0 = time.perf_counter()

    # Output files (write to .tmp first for atomic swap)
    entities_path = os.path.join(DATA_DIR, "entities_sorted.bin.tmp")
    pool_path = os.path.join(DATA_DIR, "entity_id_pool.bin.tmp")
    id_index_path = os.path.join(DATA_DIR, "entity_id_index.bin.tmp")
    keys_path = os.path.join(DATA_DIR, "geocode_keys.bin.tmp")
    starts_path = os.path.join(DATA_DIR, "prefix_starts.bin.tmp")
    counts_path = os.path.join(DATA_DIR, "prefix_counts.bin.tmp")

    # Collect all segment iterators
    segment_iters = []
    for i in range(n_segments):
        npy = os.path.join(SEGMENTS_DIR, f"seg_{i:06d}.npy")
        ids = os.path.join(SEGMENTS_DIR, f"seg_{i:06d}.ids")
        if os.path.exists(npy) and os.path.exists(ids):
            segment_iters.append(_iter_segment(npy, ids))

    if not segment_iters:
        raise FileNotFoundError("Segment files missing. Re-run 2_ingest.py.")

    # K-way merge: heapq.merge sorts by first element (geocode_bytes)
    merged = heapq.merge(*segment_iters)

    # Separate lists for building the numpy slots array at end
    # ~18 bytes per entry as Python ints (CPython overhead ~28B each — acceptable)
    id_slots_h = []
    id_slots_pos = []
    id_slots_id_offset = []
    id_slots_id_len = []

    # Prefix index: {prefix_str: [start_pos, count]}
    prefix_dict = {}

    pos = 0
    pool_offset = 0
    report_interval = 100_000

    with open(entities_path, 'wb') as f_ent, open(pool_path, 'wb') as f_pool:
        for gc_bytes, lat, lng, entity_id in merged:
            # Strip null padding for string operations
            geocode = gc_bytes.rstrip(b'\x00').decode('ascii')

            # Encode entity ID (cap at uint16 max)
            id_bytes = entity_id.encode('utf-8')
            id_len = min(len(id_bytes), 65535)
            id_bytes = id_bytes[:id_len]

            # Write 40-byte entity record
            f_ent.write(struct.pack(RECORD_FMT,
                                    gc_bytes,
                                    float(lat),
                                    float(lng),
                                    pool_offset,
                                    id_len,
                                    0))

            # Write entity ID to pool
            f_pool.write(id_bytes)

            # Accumulate ID index slot
            h = fnv1a_64(id_bytes)
            id_slots_h.append(h)
            id_slots_pos.append(pos)
            id_slots_id_offset.append(pool_offset)
            id_slots_id_len.append(id_len)

            # Update prefix index for all canonical parent levels
            gc_len = len(geocode)
            for b in CANONICAL_BREAKS:
                if b > gc_len:
                    break
                prefix = geocode[:b]
                if prefix not in prefix_dict:
                    prefix_dict[prefix] = [pos, 0]
                prefix_dict[prefix][1] += 1

            # Also index at exact length if non-canonical (municipality, citycorp, etc.)
            if gc_len not in CANONICAL_BREAKS:
                if geocode not in prefix_dict:
                    prefix_dict[geocode] = [pos, 0]
                prefix_dict[geocode][1] += 1

            pos += 1
            pool_offset += id_len

            if verbose and pos % report_interval == 0:
                elapsed = time.perf_counter() - t0
                rate = pos / elapsed if elapsed > 0 else 0
                print(f"\r[build] {pos:>10,} records written  {rate:>8,.0f} rec/s",
                      end='', flush=True)

    if verbose:
        print()

    n_entities = pos

    # -----------------------------------------------------------------
    # Build entity_id_index.bin: sorted by hash for binary search
    # -----------------------------------------------------------------
    if verbose:
        print(f"[build] Sorting {n_entities:,} ID hash slots...")

    slots = np.empty(n_entities, dtype=SLOT_DTYPE)
    slots['h'] = np.array(id_slots_h, dtype=np.uint64)
    slots['pos'] = np.array(id_slots_pos, dtype=np.uint32)
    slots['id_offset'] = np.array(id_slots_id_offset, dtype=np.uint32)
    slots['id_len'] = np.array(id_slots_id_len, dtype=np.uint16)
    slots['pad'] = 0

    # Free source lists to reclaim RAM before argsort
    del id_slots_h, id_slots_pos, id_slots_id_offset, id_slots_id_len

    sort_idx = np.argsort(slots['h'], kind='stable')
    slots = slots[sort_idx]
    del sort_idx

    slots.tofile(id_index_path)
    del slots

    # -----------------------------------------------------------------
    # Build prefix index parallel arrays
    # -----------------------------------------------------------------
    if verbose:
        print(f"[build] Writing prefix index ({len(prefix_dict):,} prefixes)...")

    prefixes_sorted = sorted(prefix_dict.keys())
    n_prefixes = len(prefixes_sorted)

    keys_arr = np.array(
        [p.encode('ascii').ljust(24, b'\x00') for p in prefixes_sorted],
        dtype='S24'
    )
    starts_arr = np.array([prefix_dict[p][0] for p in prefixes_sorted], dtype=np.uint32)
    counts_arr = np.array([prefix_dict[p][1] for p in prefixes_sorted], dtype=np.uint32)

    keys_arr.tofile(keys_path)
    starts_arr.tofile(starts_path)
    counts_arr.tofile(counts_path)
    del keys_arr, starts_arr, counts_arr, prefix_dict

    # -----------------------------------------------------------------
    # Atomic rename: .tmp -> final
    # -----------------------------------------------------------------
    final_pairs = [
        (entities_path, os.path.join(DATA_DIR, "entities_sorted.bin")),
        (pool_path,     os.path.join(DATA_DIR, "entity_id_pool.bin")),
        (id_index_path, os.path.join(DATA_DIR, "entity_id_index.bin")),
        (keys_path,     os.path.join(DATA_DIR, "geocode_keys.bin")),
        (starts_path,   os.path.join(DATA_DIR, "prefix_starts.bin")),
        (counts_path,   os.path.join(DATA_DIR, "prefix_counts.bin")),
    ]
    for tmp, final in final_pairs:
        if os.path.exists(final):
            os.remove(final)
        os.rename(tmp, final)

    elapsed = time.perf_counter() - t0
    build_meta = {
        "n_entities": n_entities,
        "n_valid": manifest["valid"],
        "n_skipped": manifest["skipped"],
        "n_prefixes": n_prefixes,
        "source_csv": manifest.get("source_csv", ""),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_duration_sec": round(elapsed, 2),
        "record_size": RECORD_SIZE,
        "slot_size": SLOT_SIZE,
        "geocode_pad_len": 24,
    }
    with open(os.path.join(DATA_DIR, "build_meta.json"), 'w') as f:
        json.dump(build_meta, f, indent=2)

    if not keep_segments:
        shutil.rmtree(SEGMENTS_DIR, ignore_errors=True)
        if verbose:
            print("[build] Segments cleaned up.")

    if verbose:
        ent_mb = (n_entities * RECORD_SIZE) / 1024 / 1024
        print(f"[build] Complete: {n_entities:,} entities, "
              f"{n_prefixes:,} prefixes, {ent_mb:.1f} MB entities file "
              f"in {elapsed:.1f}s")

    return build_meta


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build V6 binary index from segments")
    parser.add_argument("--keep-segments", action="store_true",
                        help="Keep segment files after build (default: delete)")
    args = parser.parse_args()

    run_build(keep_segments=args.keep_segments)
