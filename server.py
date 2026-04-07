#!/usr/bin/env python3
"""
V6 Location Aggregation Server

Upload any CSV (id,lat,lng) for schools, hospitals, mosques, or any entities.
Then query: given entity X, return ALL entities at the same division / district /
upazila / union / mauza level.

Architecture:
  - V5Geocoder: O(1) lat,lng -> GEO_CODE via mmap'd rasterized grid
  - V6MmapStore: entities stored as sorted binary file; prefix index in RAM
  - Build pipeline: 2_ingest.py + 3_build_index.py (run via /upload or rebuilder.py)

Endpoints:
  POST /upload                          Upload CSV, trigger background build
  GET  /status                          System status and entity counts
  GET  /entity/{id}                     Entity geocode + full hierarchy
  GET  /entity/{id}/peers?level=...     All entities at same hierarchy level
  GET  /area/{geocode}                  All entities in area by geocode prefix
  GET  /areas?level=division            Summary counts per area
"""

import os
import sys
import json
import mmap
import struct
import time
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query, Form, UploadFile, File, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
V5_DATA_DIR = os.environ.get("V5_DATA_DIR", os.path.join(BASE_DIR, "v5_data"))

RECORD_FMT = '24s f f I H H'   # geocode(24) lat(4) lng(4) id_offset(4) id_len(2) pad(2)
RECORD_SIZE = 40

SLOT_DTYPE = np.dtype([
    ('h', np.uint64),
    ('pos', np.uint32),
    ('id_offset', np.uint32),
    ('id_len', np.uint16),
    ('pad', np.uint16),
])
SLOT_SIZE = 20

# Canonical GEO_CODE prefix lengths per hierarchy level
LEVEL_TO_PREFIX_LEN = {
    'division':        2,
    'district':        4,
    'upazila':         8,
    'union':          13,
    'mauza':          16,
    'village':        20,
    'ea':             22,
    # non-canonical but queryable
    'citycorporation': 6,
    'municipality':   10,
}

LEVEL_NAMES = list(LEVEL_TO_PREFIX_LEN.keys())

_FNV_PRIME  = 0x00000100000001B3
_FNV_OFFSET = 0xcbf29ce484222325


def fnv1a_64(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


# =============================================================================
# V5 Geocoder (unchanged from v6 original — keep verbatim)
# =============================================================================

class V5Geocoder:
    """Uses V5 data to convert lat,lng -> geocode"""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("[v6] Loading V5 geocoder...")

        with open(os.path.join(V5_DATA_DIR, "meta.json")) as f:
            meta = json.load(f)

        self.min_lat = meta["min_lat"]
        self.max_lat = meta["max_lat"]
        self.min_lng = meta["min_lng"]
        self.max_lng = meta["max_lng"]
        self.cell_size = meta["cell_size"]
        self.rows = meta["rows"]
        self.cols = meta["cols"]
        self.tile_size = meta.get("tile_size", 64)
        self.tiles_rows = meta.get("tiles_rows", 0)
        self.tiles_cols = meta.get("tiles_cols", 0)

        # mmap grid
        f = open(os.path.join(V5_DATA_DIR, "sparse_grid.bin"), "rb")
        self.grid = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # mmap index
        f2 = open(os.path.join(V5_DATA_DIR, "sparse_index.bin"), "rb")
        idx_mm = mmap.mmap(f2.fileno(), 0, access=mmap.ACCESS_READ)
        self.sparse_index = np.ndarray(
            shape=(self.tiles_rows, self.tiles_cols),
            dtype=np.int32, buffer=idx_mm
        )

        # Load master dict for geocode lookup
        with open(os.path.join(V5_DATA_DIR, "master_dict.json")) as f:
            self.master_dict = json.load(f)

        # Build geocode -> name/level map
        self.geocode_info = {}
        for item in self.master_dict:
            if item:
                gc = item.get("GEO_CODE", "")
                layer = item.get("__layer__", "")
                name = ""
                if layer == "division":   name = item.get("DIVISION_NAME", "")
                elif layer == "district": name = item.get("DISTRICT_NAME", "")
                elif layer == "upazila":  name = item.get("UPAZILA_NAME", "")
                elif layer == "union":    name = item.get("UNION_NAME", "")
                elif layer == "mauza":    name = item.get("MAUZA_NAME", "")
                elif layer == "village":  name = item.get("VILLAGE_NAME", "")
                if gc:
                    self.geocode_info[gc] = {"name": name, "level": layer}

        print(f"[v6] V5 geocoder ready: {len(self.master_dict):,} locations")

    def get_geocode(self, lat: float, lng: float) -> str:
        """Convert lat,lng to geocode string"""
        row = int((self.max_lat - lat) / self.cell_size)
        col = int((lng - self.min_lng) / self.cell_size)

        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return ""

        tile_r = row // self.tile_size
        tile_c = col // self.tile_size
        tile_offset = int(self.sparse_index[tile_r, tile_c])

        if tile_offset == -1:
            return ""

        sub_r = row % self.tile_size
        sub_c = col % self.tile_size
        byte_start = (tile_offset * (self.tile_size * self.tile_size * 4)
                      + ((sub_r * self.tile_size + sub_c) * 4))
        loc_id = int.from_bytes(self.grid[byte_start:byte_start + 4], 'little')

        if loc_id == 0 or loc_id >= len(self.master_dict):
            return ""

        item = self.master_dict[loc_id]
        return item.get("GEO_CODE", "") if item else ""

    def get_info(self, geocode: str) -> dict:
        """Get human-readable info for geocode"""
        if geocode in self.geocode_info:
            return self.geocode_info[geocode]

        length = len(geocode)
        level = "unknown"
        if length == 2:   level = "division"
        elif length == 4: level = "district"
        elif length == 6: level = "citycorporation"
        elif length == 8: level = "upazila"
        elif length == 10: level = "municipality"
        elif length == 13: level = "union"
        elif length == 16: level = "mauza"
        elif length >= 20: level = "village_or_ea"

        return {"name": geocode, "level": level}

    def build_hierarchy(self, geocode: str) -> dict:
        """Return a dict of all parent geocodes at each level."""
        h = {}
        for level, prefix_len in LEVEL_TO_PREFIX_LEN.items():
            if len(geocode) >= prefix_len:
                prefix = geocode[:prefix_len]
                info = self.get_info(prefix)
                h[level] = {"geocode": prefix, "name": info["name"]}
        return h


# =============================================================================
# V6 MmapStore — mmap'd binary files, prefix index in RAM
# =============================================================================

class V6MmapStore:
    """
    Serves entity queries from mmap'd binary files.

    Data files (built by 3_build_index.py):
      entities_sorted.bin  — 40-byte records sorted by geocode
      entity_id_pool.bin   — flat byte pool of entity ID strings
      entity_id_index.bin  — FNV1a-64 hash slots sorted by hash
      geocode_keys.bin     — sorted geocode prefix strings (S24)
      prefix_starts.bin    — start record index per prefix (uint32)
      prefix_counts.bin    — entity count per prefix (uint32)
      build_meta.json      — build metadata
    """

    def __init__(self):
        self.is_loaded = False
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        meta_path = os.path.join(DATA_DIR, "build_meta.json")
        entities_path = os.path.join(DATA_DIR, "entities_sorted.bin")

        if not os.path.exists(meta_path) or not os.path.exists(entities_path):
            print("[v6] No index found — upload a CSV to build it.")
            return

        print("[v6] Loading V6 binary index...")
        t0 = time.perf_counter()

        with open(meta_path) as f:
            self.meta = json.load(f)

        n = self.meta["n_entities"]

        # mmap entity records (40B each)
        f_ent = open(entities_path, "rb")
        self._mm_entities = mmap.mmap(f_ent.fileno(), 0, access=mmap.ACCESS_READ)

        # mmap entity ID pool
        f_pool = open(os.path.join(DATA_DIR, "entity_id_pool.bin"), "rb")
        self._mm_pool = mmap.mmap(f_pool.fileno(), 0, access=mmap.ACCESS_READ)

        # mmap ID index slots
        f_idx = open(os.path.join(DATA_DIR, "entity_id_index.bin"), "rb")
        self._mm_id_idx = mmap.mmap(f_idx.fileno(), 0, access=mmap.ACCESS_READ)
        self._id_slots = np.ndarray(n, dtype=SLOT_DTYPE, buffer=self._mm_id_idx)

        # Load hash column into RAM for fast np.searchsorted (8B × N)
        self._id_hashes = np.array(self._id_slots['h'], dtype=np.uint64)

        # Load prefix index into RAM dict  (~65 MB max)
        keys  = np.fromfile(os.path.join(DATA_DIR, "geocode_keys.bin"),   dtype='S24')
        starts = np.fromfile(os.path.join(DATA_DIR, "prefix_starts.bin"), dtype=np.uint32)
        counts = np.fromfile(os.path.join(DATA_DIR, "prefix_counts.bin"), dtype=np.uint32)
        self._prefix_index = {
            k.rstrip(b'\x00').decode('ascii'): (int(s), int(c))
            for k, s, c in zip(keys, starts, counts)
        }

        # ── Extra-field metadata (optional) ──────────────────────────────────
        self.metadata      = {}
        self.schema        = []
        self.display_field = 'id'
        meta_file   = os.path.join(DATA_DIR, "entity_metadata.json")
        schema_file = os.path.join(DATA_DIR, "schema.json")
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                self.metadata = json.load(f)
        if os.path.exists(schema_file):
            with open(schema_file) as f:
                sch = json.load(f)
                self.schema        = sch.get("fields", [])
                self.display_field = sch.get("display_field", "id")

        elapsed = time.perf_counter() - t0
        self.is_loaded = True
        print(f"[v6] Index loaded: {n:,} entities, "
              f"{len(self._prefix_index):,} prefixes, "
              f"{len(self.schema)} extra fields in {elapsed:.2f}s")

    def reload(self):
        """Hot-reload all data files without restarting the process."""
        with self._lock:
            if self.is_loaded:
                try:
                    self._mm_entities.close()
                    self._mm_pool.close()
                    self._mm_id_idx.close()
                except Exception:
                    pass
            self.is_loaded     = False
            self.metadata      = {}
            self.schema        = []
            self.display_field = 'id'
            self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_entity_pos(self, entity_id: str) -> Optional[int]:
        """Return record index for entity_id, or None if not found."""
        id_bytes = entity_id.encode('utf-8')
        h = fnv1a_64(id_bytes)
        lo = int(np.searchsorted(self._id_hashes, h))
        n = len(self._id_hashes)
        MAX_WALK = 16  # max hash collisions to probe
        for i in range(lo, min(lo + MAX_WALK, n)):
            slot = self._id_slots[i]
            if int(slot['h']) != h:
                break
            stored_off = int(slot['id_offset'])
            stored_len = int(slot['id_len'])
            stored = bytes(self._mm_pool[stored_off: stored_off + stored_len])
            if stored == id_bytes:
                return int(slot['pos'])
        return None

    def _read_record(self, pos: int) -> dict:
        """Read one 40-byte entity record by position index."""
        offset = pos * RECORD_SIZE
        raw = bytes(self._mm_entities[offset: offset + RECORD_SIZE])
        gc_bytes, lat, lng, id_offset, id_len, _ = struct.unpack(RECORD_FMT, raw)
        geocode = gc_bytes.rstrip(b'\x00').decode('ascii')
        entity_id = bytes(self._mm_pool[id_offset: id_offset + id_len]).decode('utf-8')
        return {
            'id': entity_id,
            'geocode': geocode,
            'lat': round(float(lat), 6),
            'lng': round(float(lng), 6),
        }

    def _read_range(self, start: int, offset: int, limit: int) -> list[dict]:
        """Read `limit` entity records starting at (start + offset)."""
        actual_start = (start + offset) * RECORD_SIZE
        actual_end = actual_start + limit * RECORD_SIZE
        chunk = bytes(self._mm_entities[actual_start: actual_end])
        records = []
        for i in range(limit):
            raw = chunk[i * RECORD_SIZE: (i + 1) * RECORD_SIZE]
            gc_bytes, lat, lng, id_offset, id_len, _ = struct.unpack(RECORD_FMT, raw)
            entity_id = bytes(self._mm_pool[id_offset: id_offset + id_len]).decode('utf-8')
            records.append({
                'id': entity_id,
                'lat': round(float(lat), 6),
                'lng': round(float(lng), 6),
            })
        return records

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[dict]:
        """Return entity record + geocode + metadata, or None."""
        pos = self._get_entity_pos(entity_id)
        if pos is None:
            return None
        rec = self._read_record(pos)
        rec['metadata'] = self.metadata.get(entity_id, {})
        if self.display_field and self.display_field != 'id' and self.metadata:
            label = self.metadata.get(entity_id, {}).get(self.display_field, '')
            if label:
                rec['label'] = str(label)
        return rec

    def get_area(self, geocode_prefix: str, offset: int = 0, limit: int = 1000
                 ) -> Optional[tuple[list, int]]:
        """
        Return (records_list, total_count) for geocode prefix, or None if unknown.
        """
        entry = self._prefix_index.get(geocode_prefix)
        if entry is None:
            return None
        start, count = entry
        if offset >= count:
            return [], count
        actual_limit = min(limit, count - offset)
        records = self._read_range(start, offset, actual_limit)
        if self.display_field and self.display_field != 'id' and self.metadata:
            for r in records:
                label = self.metadata.get(r['id'], {}).get(self.display_field, '')
                if label:
                    r['label'] = str(label)
        return records, count

    def list_areas(self, prefix_len: int) -> list[dict]:
        """Return all prefixes of exactly `prefix_len` chars with counts."""
        return [
            {'geocode': gc, 'count': cnt, 'start': st}
            for gc, (st, cnt) in self._prefix_index.items()
            if len(gc) == prefix_len
        ]

    def get_all_entities(self) -> list[dict]:
        """Return every entity in one shot — used by /entities/all for single-request loading."""
        n = self.meta.get("n_entities", 0)
        if n == 0:
            return []
        records = self._read_range(0, 0, n)
        if self.display_field and self.display_field != 'id' and self.metadata:
            for r in records:
                label = self.metadata.get(r['id'], {}).get(self.display_field, '')
                if label:
                    r['label'] = str(label)
        return records

    def get_stats(self) -> dict:
        if not self.is_loaded:
            return {'is_loaded': False, 'n_entities': 0}
        return {
            'is_loaded': True,
            'n_entities': self.meta.get('n_entities', 0),
            'n_valid': self.meta.get('n_valid', 0),
            'n_skipped': self.meta.get('n_skipped', 0),
            'n_prefixes': self.meta.get('n_prefixes', 0),
            'source_csv': self.meta.get('source_csv', ''),
            'built_at': self.meta.get('built_at', ''),
        }


# =============================================================================
# Global instances (populated in lifespan)
# =============================================================================

geocoder: V5Geocoder = None
store: V6MmapStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global geocoder, store
    geocoder = V5Geocoder.get()
    store = V6MmapStore()
    # Background task: watch for hot-reload signal
    task = asyncio.create_task(_watch_reload_signal())
    yield
    task.cancel()


async def _watch_reload_signal():
    """Check every 2 seconds for data/.reload_signal and hot-reload if found."""
    signal_path = os.path.join(DATA_DIR, ".reload_signal")
    while True:
        await asyncio.sleep(2)
        if os.path.exists(signal_path):
            try:
                os.remove(signal_path)
            except OSError:
                pass
            print("[v6] Reload signal detected — reloading index...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, store.reload)
            print("[v6] Hot-reload complete.")


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(
    title="V6 Location Aggregator",
    version="6.1.0",
    description="Upload any CSV (id,lat,lng) and query entities by administrative hierarchy",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Background build runner
# =============================================================================

_build_status = {"running": False, "last_result": None, "started_at": None}
_build_lock = threading.Lock()


def _run_build_background(csv_path: str, n_workers: int, extra_fields: list = None, display_field: str = 'id'):
    global _build_status
    with _build_lock:
        _build_status["running"] = True
        _build_status["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _build_status["last_result"] = None

    try:
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("rebuilder", os.path.join(BASE_DIR, "rebuilder.py"))
        rebuilder = module_from_spec(spec)
        spec.loader.exec_module(rebuilder)
        result = rebuilder.run_pipeline(
            csv_path=csv_path,
            n_workers=n_workers,
            extra_fields=extra_fields or [],
            display_field=display_field,
            verbose=True,
        )
        with _build_lock:
            _build_status["last_result"] = result
    except Exception as e:
        print(f"[v6] Build failed: {e}")
        with _build_lock:
            _build_status["last_result"] = {"success": False, "error": str(e)}
    finally:
        with _build_lock:
            _build_status["running"] = False


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
def root():
    return {
        "name": "V6 Location Aggregator",
        "version": "6.1.0",
        "status": "ready" if (store and store.is_loaded) else "no_data",
        "endpoints": {
            "upload":   "POST /upload (CSV file: id,lat,lng)",
            "status":   "GET /status",
            "entity":   "GET /entity/{id}",
            "peers":    "GET /entity/{id}/peers?level=division|district|upazila|union",
            "area":     "GET /area/{geocode}?offset=0&limit=1000",
            "areas":    "GET /areas?level=division",
        }
    }


@app.get("/status")
def status():
    stats = store.get_stats() if store else {"is_loaded": False}
    return {
        **stats,
        "build_running": _build_status["running"],
        "build_started": _build_status["started_at"],
        "build_result":  _build_status["last_result"],
    }


@app.get("/schema")
def get_schema():
    """Return the list of extra metadata fields and display_field for the current dataset."""
    if not store or not store.is_loaded:
        return {"fields": [], "display_field": "id", "loaded": False}
    return {"fields": store.schema, "display_field": store.display_field, "loaded": True}


@app.get("/area/{geocode}/stats")
def area_stats(geocode: str):
    """
    Return aggregate statistics for numeric extra fields across all entities
    in the given geographic area.
    """
    _require_loaded()
    result = store.get_area(geocode, offset=0, limit=999_999)
    if result is None:
        raise HTTPException(404, f"No data for geocode: {geocode!r}")

    entities, total = result

    if not store.schema or not store.metadata:
        return {"geocode": geocode, "total": total, "stats": {}}

    # Compute per-field numeric stats
    buckets: dict[str, list[float]] = {f: [] for f in store.schema}
    for e in entities:
        meta = store.metadata.get(e["id"], {})
        for field in store.schema:
            raw = meta.get(field, "")
            if raw != "":
                try:
                    buckets[field].append(float(raw))
                except (ValueError, TypeError):
                    pass

    stats = {}
    for field, vals in buckets.items():
        if vals:
            stats[field] = {
                "count": len(vals),
                "sum":   round(sum(vals), 2),
                "avg":   round(sum(vals) / len(vals), 2),
                "min":   round(min(vals), 2),
                "max":   round(max(vals), 2),
            }

    return {"geocode": geocode, "total": total, "stats": stats}


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    workers: int = Query(2, ge=1, le=32),
    extra_fields: str = Form('[]'),
    display_field: str = Form('id'),
):
    """
    Upload a CSV file and trigger async build pipeline.

    extra_fields: JSON array of column names to store as entity metadata.
    Example: extra_fields='["manager_name","today_revenue","medicine_stock"]'

    The build runs in the background. Poll /status to check progress.
    When complete, the server hot-reloads the new index automatically.
    """
    if _build_status["running"]:
        raise HTTPException(409, "A build is already running. Check /status.")

    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "File must be .csv")

    # Parse extra_fields JSON
    try:
        schema = json.loads(extra_fields)
        if not isinstance(schema, list):
            schema = []
        schema = [str(f).strip() for f in schema if str(f).strip()]
    except Exception:
        schema = []

    os.makedirs(DATA_DIR, exist_ok=True)
    dest     = os.path.join(DATA_DIR, "input.csv")
    dest_tmp = dest + ".tmp"

    with open(dest_tmp, 'wb') as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    os.replace(dest_tmp, dest)

    safe_workers = max(1, min(workers, 4))
    display_field = display_field.strip() or 'id'
    background_tasks.add_task(_run_build_background, dest, safe_workers, schema, display_field)

    return {
        "accepted":      True,
        "message":       "Build pipeline started. Poll /status for progress.",
        "input_file":    dest,
        "workers":       safe_workers,
        "extra_fields":  schema,
        "display_field": display_field,
    }


@app.get("/entity/{entity_id}")
def get_entity(entity_id: str):
    """
    Return an entity's coordinates, geocode, and full administrative hierarchy.

    Example response:
    {
      "id": "SCHOOL001",
      "lat": 23.8103, "lng": 90.4125,
      "geocode": "302600470102034",
      "level": "village",
      "hierarchy": {
        "division":  {"geocode": "30",   "name": "Dhaka"},
        "district":  {"geocode": "3026", "name": "Dhaka"},
        "upazila":   {"geocode": "30260047", "name": "Savar"},
        ...
      }
    }
    """
    _require_loaded()
    record = store.get_entity(entity_id)
    if record is None:
        raise HTTPException(404, f"Entity not found: {entity_id!r}")

    gc_info = geocoder.get_info(record['geocode'])
    hierarchy = geocoder.build_hierarchy(record['geocode'])

    return {
        **record,
        'level': gc_info['level'],
        'name': gc_info['name'],
        'hierarchy': hierarchy,
        'geocode_precision_m': 55,
    }


@app.get("/entity/{entity_id}/peers")
def get_peers(
    entity_id: str,
    level: str = Query('division', description="Hierarchy level to aggregate at"),
    offset: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
):
    """
    Return all entities in the same administrative area as `entity_id` at `level`.

    Supported levels: division, district, upazila, union, mauza,
                      village, ea, citycorporation, municipality

    Example: /entity/SCHOOL001/peers?level=district&limit=100
    Returns all entities (schools, or whatever was uploaded) in the same district.
    """
    _require_loaded()

    prefix_len = LEVEL_TO_PREFIX_LEN.get(level)
    if prefix_len is None:
        raise HTTPException(
            400,
            f"Invalid level: {level!r}. Valid levels: {LEVEL_NAMES}"
        )

    # Find the entity
    record = store.get_entity(entity_id)
    if record is None:
        raise HTTPException(404, f"Entity not found: {entity_id!r}")

    geocode = record['geocode']
    if len(geocode) < prefix_len:
        entity_level = geocoder.get_info(geocode)['level']
        raise HTTPException(
            400,
            f"Entity is at level={entity_level!r} (geocode length {len(geocode)}); "
            f"cannot aggregate at {level!r} (requires length {prefix_len})"
        )

    prefix = geocode[:prefix_len]
    result = store.get_area(prefix, offset, limit)
    if result is None:
        raise HTTPException(404, f"No indexed data for geocode prefix: {prefix!r}")

    peers, total = result
    area_info = geocoder.get_info(prefix)

    return {
        'entity_id': entity_id,
        'entity_geocode': geocode,
        'level': level,
        'area_geocode': prefix,
        'area_name': area_info['name'],
        'total': total,
        'offset': offset,
        'returned': len(peers),
        'has_more': offset + len(peers) < total,
        'peers': peers,
    }


@app.get("/area/{geocode}")
def get_area(
    geocode: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
):
    """
    Return all entities in the geographic area identified by `geocode`.

    geocode examples:
      30         = Dhaka Division (all entities)
      3026       = Dhaka District
      30260047   = Specific upazila
    """
    _require_loaded()

    result = store.get_area(geocode, offset, limit)
    if result is None:
        raise HTTPException(404, f"No data for geocode: {geocode!r}")

    entities, total = result
    gc_info = geocoder.get_info(geocode)

    return {
        'geocode': geocode,
        'name': gc_info['name'],
        'level': gc_info['level'],
        'total': total,
        'offset': offset,
        'returned': len(entities),
        'has_more': offset + len(entities) < total,
        'entities': entities,
    }


@app.get("/areas")
def list_areas(
    level: str = Query('division',
                       description="Hierarchy level: division, district, upazila, union, ..."),
):
    """
    List all geographic areas at the given level with entity counts.

    Useful for building dropdowns or summary statistics.
    """
    _require_loaded()

    prefix_len = LEVEL_TO_PREFIX_LEN.get(level)
    if prefix_len is None:
        raise HTTPException(
            400,
            f"Invalid level: {level!r}. Valid levels: {LEVEL_NAMES}"
        )

    raw = store.list_areas(prefix_len)
    result = []
    for entry in raw:
        gc = entry['geocode']
        info = geocoder.get_info(gc)
        result.append({
            'geocode': gc,
            'name': info['name'],
            'level': level,
            'count': entry['count'],
        })

    result.sort(key=lambda x: x['geocode'])

    return {
        'level': level,
        'total_areas': len(result),
        'areas': result,
    }


@app.get("/entities/all")
def all_entities():
    """
    Return ALL entities in a single request.

    Replaces the old N+1 pattern (GET /areas → N × GET /area/{geocode}).
    One round trip instead of two → dramatically faster initial map load.
    Each record includes id, lat, lng, and label (if a display_field is set).
    """
    _require_loaded()
    records = store.get_all_entities()
    return {
        "n_entities":    len(records),
        "display_field": store.display_field,
        "entities":      records,
    }


class _GeocodeBatchItem(BaseModel):
    id:  str   = ''
    lat: float = Field(..., ge=5.0,  le=35.0)
    lng: float = Field(..., ge=80.0, le=100.0)

class _GeocodeBatchInput(BaseModel):
    coords: list[_GeocodeBatchItem] = Field(..., min_length=1, max_length=50_000)


@app.post("/geocode/batch")
def geocode_batch(body: _GeocodeBatchInput):
    """
    Batch reverse-geocode coordinates → geocode string + full hierarchy.

    Used by the browser when processing a CSV locally (no server-side entity
    storage required).  The V5 raster lookup runs in <1 ms for 5 000 points,
    so even 50 000-row CSVs finish in a few seconds.
    """
    results = []
    for item in body.coords:
        gc = geocoder.get_geocode(item.lat, item.lng)
        h  = geocoder.build_hierarchy(gc) if gc else {}
        results.append({'id': item.id, 'geocode': gc, 'hierarchy': h})
    return {'results': results, 'count': len(results)}


# =============================================================================
# Helpers
# =============================================================================

def _require_loaded():
    if not store or not store.is_loaded:
        raise HTTPException(
            503,
            "No data loaded. Upload a CSV via POST /upload and wait for the build to complete."
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("""
================================================================================
                    V6 LOCATION AGGREGATOR  v6.1.0
================================================================================

QUICK START:
  1. Start server:
       python server.py

  2. Upload CSV (any entity type — schools, hospitals, mosques, etc.):
       curl -X POST -F "file=@schools.csv" http://localhost:7861/upload

  3. Watch build progress:
       curl http://localhost:7861/status

  4. Query all entities in same division as an entity:
       curl "http://localhost:7861/entity/SCHOOL001/peers?level=division"

  5. Query all entities in a specific area:
       curl "http://localhost:7861/area/30?limit=100"

  6. List all divisions with entity counts:
       curl "http://localhost:7861/areas?level=division"

CSV FORMAT:
  id,lat,lng
  SCHOOL001,23.8103,90.4125
  HOSPITAL002,24.1234,90.5678
  (id column is optional — auto-generated if missing)

================================================================================
""")
    uvicorn.run(app, host="0.0.0.0", port=7861)
