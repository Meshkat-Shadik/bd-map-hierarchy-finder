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
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_auth_requests

GOOGLE_CLIENT_ID = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "476786425809-4ju4v578ae38vl0hu8sraoe1d61r230v.apps.googleusercontent.com",
)

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

    def __init__(self, data_dir: str = None):
        self._data_dir = data_dir or DATA_DIR
        self.is_loaded = False
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        d = self._data_dir
        meta_path     = os.path.join(d, "build_meta.json")
        entities_path = os.path.join(d, "entities_sorted.bin")

        if not os.path.exists(meta_path) or not os.path.exists(entities_path):
            print(f"[v6] No index in {d} — upload a CSV to build it.")
            return

        print(f"[v6] Loading index from {d}…")
        t0 = time.perf_counter()

        with open(meta_path) as f:
            self.meta = json.load(f)

        n = self.meta["n_entities"]

        f_ent = open(entities_path, "rb")
        self._mm_entities = mmap.mmap(f_ent.fileno(), 0, access=mmap.ACCESS_READ)

        f_pool = open(os.path.join(d, "entity_id_pool.bin"), "rb")
        self._mm_pool = mmap.mmap(f_pool.fileno(), 0, access=mmap.ACCESS_READ)

        f_idx = open(os.path.join(d, "entity_id_index.bin"), "rb")
        self._mm_id_idx = mmap.mmap(f_idx.fileno(), 0, access=mmap.ACCESS_READ)
        self._id_slots  = np.ndarray(n, dtype=SLOT_DTYPE, buffer=self._mm_id_idx)
        self._id_hashes = np.array(self._id_slots['h'], dtype=np.uint64)

        keys   = np.fromfile(os.path.join(d, "geocode_keys.bin"),   dtype='S24')
        starts = np.fromfile(os.path.join(d, "prefix_starts.bin"),  dtype=np.uint32)
        counts = np.fromfile(os.path.join(d, "prefix_counts.bin"),  dtype=np.uint32)
        self._prefix_index = {
            k.rstrip(b'\x00').decode('ascii'): (int(s), int(c))
            for k, s, c in zip(keys, starts, counts)
        }

        self.metadata      = {}
        self.schema        = []
        self.display_field = 'id'
        meta_file   = os.path.join(d, "entity_metadata.json")
        schema_file = os.path.join(d, "schema.json")
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
              f"{len(self._prefix_index):,} prefixes in {elapsed:.2f}s")

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
        """
        Return up to ENTITY_RENDER_LIMIT entities.

        50M entities × 50 B/entity ≈ 2.5 GB — not feasible as a single JSON response.
        Callers check .meta["n_entities"] to detect truncation.
        """
        n      = self.meta.get("n_entities", 0)
        actual = min(n, ENTITY_RENDER_LIMIT)
        if actual == 0:
            return []
        records = self._read_range(0, 0, actual)
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
# Session Manager — one V6MmapStore per browser session (UUID)
# =============================================================================

SESSIONS_DIR    = os.path.join(BASE_DIR, "data", "sessions")
USERS_DIR       = os.path.join(BASE_DIR, "data", "users")   # persistent, never evicted
MAX_SESSIONS    = 30        # LRU eviction when limit reached (anonymous only)
SESSION_TTL_SEC = 86400     # 24 h of inactivity → evict (anonymous only)
# Cap for /entities/all to prevent gigantic JSON responses (50M rows × 50B ≈ 2.5 GB)
ENTITY_RENDER_LIMIT = 200_000


def _session_data_dir(session_id: str) -> str:
    """Map a session ID to its data directory.
    'default'  → data/
    'u_{sub}'  → data/users/{sub}/   (Google-authenticated, permanent)
    '{uuid}'   → data/sessions/{uuid}/  (anonymous, LRU-evicted)
    """
    if session_id == 'default':
        return DATA_DIR
    if session_id.startswith('u_'):
        return os.path.join(USERS_DIR, session_id[2:])
    return os.path.join(SESSIONS_DIR, session_id)


class _SessionManager:
    def __init__(self):
        self._sessions: dict[str, dict] = {}   # sid → {store, last_seen}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> 'V6MmapStore':
        with self._lock:
            self._evict()
            if session_id not in self._sessions:
                d = _session_data_dir(session_id)
                os.makedirs(d, exist_ok=True)
                self._sessions[session_id] = {
                    'store':     V6MmapStore(d),
                    'last_seen': time.time(),
                }
            else:
                self._sessions[session_id]['last_seen'] = time.time()
            return self._sessions[session_id]['store']

    def reload(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]['store'].reload()
            else:
                # Session not cached yet — create fresh
                self.get(session_id)

    def _evict(self):
        now = time.time()
        # Only evict anonymous sessions (u_ prefix = authenticated, never evict)
        expired = [sid for sid, s in self._sessions.items()
                   if not sid.startswith('u_') and now - s['last_seen'] > SESSION_TTL_SEC]
        for sid in expired:
            del self._sessions[sid]
        anon = [sid for sid in list(self._sessions) if not sid.startswith('u_')]
        while len(anon) >= MAX_SESSIONS:
            oldest = min(anon, key=lambda s: self._sessions[s]['last_seen'])
            del self._sessions[oldest]
            anon.remove(oldest)


# =============================================================================
# Global instances (populated in lifespan)
# =============================================================================

geocoder: V5Geocoder = None
session_mgr: _SessionManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global geocoder, session_mgr
    geocoder    = V5Geocoder.get()
    session_mgr = _SessionManager()
    # Pre-warm the default session (existing data/ directory)
    session_mgr.get('default')
    task = asyncio.create_task(_watch_reload_signals())
    yield
    task.cancel()


async def _watch_reload_signals():
    """
    Every 2 s scan for .reload_signal files in data/ and data/sessions/*/.
    When found, hot-reload the corresponding session's store.
    """
    while True:
        await asyncio.sleep(2)
        dirs_to_check = [('default', DATA_DIR)]
        if os.path.isdir(SESSIONS_DIR):
            for sid in os.listdir(SESSIONS_DIR):
                dirs_to_check.append((sid, os.path.join(SESSIONS_DIR, sid)))
        for session_id, d in dirs_to_check:
            sig = os.path.join(d, ".reload_signal")
            if os.path.exists(sig):
                try:
                    os.remove(sig)
                except OSError:
                    pass
                print(f"[v6] Reload signal for session={session_id!r}")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, session_mgr.reload, session_id)


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


_build_status: dict[str, dict] = {}   # session_id → status dict


def _get_or_init_build_status(session_id: str) -> dict:
    if session_id not in _build_status:
        _build_status[session_id] = {"running": False, "last_result": None, "started_at": None}
    return _build_status[session_id]


def _run_build_background(csv_path: str, n_workers: int, session_id: str = 'default',
                           extra_fields: list = None, display_field: str = 'id'):
    with _build_lock:
        st = _get_or_init_build_status(session_id)
        st["running"]    = True
        st["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        st["last_result"]= None

    data_dir = _session_data_dir(session_id)
    os.makedirs(data_dir, exist_ok=True)

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
            data_dir=data_dir,
            verbose=True,
        )
        with _build_lock:
            _build_status[session_id]["last_result"] = result
    except Exception as e:
        print(f"[v6] Build failed (session={session_id}): {e}")
        with _build_lock:
            _build_status[session_id]["last_result"] = {"success": False, "error": str(e)}
    finally:
        with _build_lock:
            _build_status[session_id]["running"] = False


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
def root():
    return {
        "name": "V6 Location Aggregator",
        "version": "6.1.0",
        "status": "ready",
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
def status(session_id: str = Query('default')):
    st    = _get_or_init_build_status(session_id)
    s     = session_mgr.get(session_id)
    stats = s.get_stats() if s else {"is_loaded": False}
    return {
        **stats,
        "session_id":    session_id,
        "build_running": st["running"],
        "build_started": st["started_at"],
        "build_result":  st["last_result"],
    }


@app.get("/schema")
def get_schema(session_id: str = Query('default')):
    """Return the list of extra metadata fields and display_field for the current session."""
    s = session_mgr.get(session_id)
    if not s or not s.is_loaded:
        return {"fields": [], "display_field": "id", "loaded": False}
    return {"fields": s.schema, "display_field": s.display_field, "loaded": True}


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    workers:       int = Query(2, ge=1, le=32),
    session_id:    str = Form('default'),
    extra_fields:  str = Form('[]'),
    display_field: str = Form('id'),
):
    """
    Upload a CSV file and trigger async build pipeline.

    session_id: browser UUID — each user uploads to their own slot (no conflicts).
    extra_fields: JSON array of column names to store as entity metadata.
    The build runs in the background. Poll /status?session_id=... to check progress.
    """
    st = _get_or_init_build_status(session_id)
    if st["running"]:
        raise HTTPException(409, f"A build is already running for session {session_id!r}.")

    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "File must be .csv")

    try:
        schema = json.loads(extra_fields)
        if not isinstance(schema, list):
            schema = []
        schema = [str(f).strip() for f in schema if str(f).strip()]
    except Exception:
        schema = []

    data_dir = _session_data_dir(session_id)
    os.makedirs(data_dir, exist_ok=True)
    dest     = os.path.join(data_dir, "input.csv")
    dest_tmp = dest + ".tmp"

    with open(dest_tmp, 'wb') as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(dest_tmp, dest)

    safe_workers  = max(1, min(workers, 4))
    display_field = display_field.strip() or 'id'
    background_tasks.add_task(
        _run_build_background, dest, safe_workers, session_id, schema, display_field
    )

    return {
        "accepted":      True,
        "session_id":    session_id,
        "message":       "Build pipeline started. Poll /status?session_id=... for progress.",
        "workers":       safe_workers,
        "extra_fields":  schema,
        "display_field": display_field,
    }


@app.get("/entity/{entity_id}")
def get_entity(entity_id: str, session_id: str = Query('default')):
    """Return an entity's coordinates, geocode, and full administrative hierarchy."""
    _require_loaded(session_id)
    s      = session_mgr.get(session_id)
    record = s.get_entity(entity_id)
    if record is None:
        raise HTTPException(404, f"Entity not found: {entity_id!r}")

    gc_info   = geocoder.get_info(record['geocode'])
    hierarchy = geocoder.build_hierarchy(record['geocode'])

    return {
        **record,
        'level':              gc_info['level'],
        'name':               gc_info['name'],
        'hierarchy':          hierarchy,
        'geocode_precision_m': 55,
    }


@app.get("/entity/{entity_id}/peers")
def get_peers(
    entity_id: str,
    level:     str = Query('division', description="Hierarchy level to aggregate at"),
    offset:    int = Query(0, ge=0),
    limit:     int = Query(1000, ge=1, le=10000),
    session_id:str = Query('default'),
):
    _require_loaded(session_id)
    s = session_mgr.get(session_id)

    prefix_len = LEVEL_TO_PREFIX_LEN.get(level)
    if prefix_len is None:
        raise HTTPException(400, f"Invalid level: {level!r}. Valid: {LEVEL_NAMES}")

    record = s.get_entity(entity_id)
    if record is None:
        raise HTTPException(404, f"Entity not found: {entity_id!r}")

    geocode = record['geocode']
    if len(geocode) < prefix_len:
        raise HTTPException(400,
            f"Entity geocode too short for level {level!r} (need {prefix_len} chars, got {len(geocode)})")

    prefix = geocode[:prefix_len]
    result = s.get_area(prefix, offset, limit)
    if result is None:
        raise HTTPException(404, f"No indexed data for geocode prefix: {prefix!r}")

    peers, total = result
    area_info    = geocoder.get_info(prefix)

    return {
        'entity_id':     entity_id,
        'entity_geocode':geocode,
        'level':         level,
        'area_geocode':  prefix,
        'area_name':     area_info['name'],
        'total':         total,
        'offset':        offset,
        'returned':      len(peers),
        'has_more':      offset + len(peers) < total,
        'peers':         peers,
    }


@app.get("/area/{geocode}")
def get_area(
    geocode:    str,
    offset:     int = Query(0, ge=0),
    limit:      int = Query(1000, ge=1, le=10000),
    session_id: str = Query('default'),
):
    _require_loaded(session_id)
    s      = session_mgr.get(session_id)
    result = s.get_area(geocode, offset, limit)
    if result is None:
        raise HTTPException(404, f"No data for geocode: {geocode!r}")

    entities, total = result
    gc_info = geocoder.get_info(geocode)

    return {
        'geocode':  geocode,
        'name':     gc_info['name'],
        'level':    gc_info['level'],
        'total':    total,
        'offset':   offset,
        'returned': len(entities),
        'has_more': offset + len(entities) < total,
        'entities': entities,
    }


@app.get("/areas")
def list_areas(
    level:      str = Query('division'),
    session_id: str = Query('default'),
):
    _require_loaded(session_id)
    s          = session_mgr.get(session_id)
    prefix_len = LEVEL_TO_PREFIX_LEN.get(level)
    if prefix_len is None:
        raise HTTPException(400, f"Invalid level: {level!r}. Valid: {LEVEL_NAMES}")

    raw    = s.list_areas(prefix_len)
    result = []
    for entry in raw:
        gc   = entry['geocode']
        info = geocoder.get_info(gc)
        result.append({'geocode': gc, 'name': info['name'], 'level': level, 'count': entry['count']})
    result.sort(key=lambda x: x['geocode'])
    return {'level': level, 'total_areas': len(result), 'areas': result}


@app.get("/entities/all")
def all_entities(session_id: str = Query('default')):
    """
    Return ALL entities in one request (capped at ENTITY_RENDER_LIMIT).

    For datasets > ENTITY_RENDER_LIMIT the response includes truncated=true and
    n_total so the frontend can warn the user.  Use /entities/clustered for a
    zoom-aware cluster view of very large datasets.
    """
    _require_loaded(session_id)
    s       = session_mgr.get(session_id)
    n_total = s.meta.get("n_entities", 0)
    records = s.get_all_entities()          # already capped inside the method
    return {
        "n_entities":    n_total,
        "returned":      len(records),
        "truncated":     len(records) < n_total,
        "display_field": s.display_field,
        "entities":      records,
    }


@app.get("/entities/clustered")
def entities_clustered(
    zoom:       int = Query(7, ge=1, le=18),
    session_id: str = Query('default'),
):
    """
    Return cluster centroids sized by entity count — ideal for large datasets.

    zoom < 8  → division-level clusters   (prefix len 2)
    zoom 8-9  → district-level clusters   (prefix len 4)
    zoom 10-11→ upazila-level clusters    (prefix len 8)
    zoom 12+  → union-level clusters      (prefix len 13)

    Each cluster: {geocode, name, lat, lng, count}
    """
    _require_loaded(session_id)
    s = session_mgr.get(session_id)

    if   zoom < 8:  prefix_len = 2
    elif zoom < 10: prefix_len = 4
    elif zoom < 12: prefix_len = 8
    else:           prefix_len = 13

    areas   = s.list_areas(prefix_len)
    results = []
    for area in areas:
        # Sample up to 200 records to estimate centroid quickly
        recs, total = s.get_area(area['geocode'], 0, min(200, area['count']))
        if not recs:
            continue
        avg_lat = sum(r['lat'] for r in recs) / len(recs)
        avg_lng = sum(r['lng'] for r in recs) / len(recs)
        info    = geocoder.get_info(area['geocode'])
        results.append({
            'geocode': area['geocode'],
            'name':    info['name'],
            'lat':     round(avg_lat, 5),
            'lng':     round(avg_lng, 5),
            'count':   total,
        })

    return {'clusters': results, 'prefix_len': prefix_len, 'zoom': zoom}


class _GeocodeBatchItem(BaseModel):
    id:  str   = ''
    lat: float = Field(..., ge=5.0,  le=35.0)
    lng: float = Field(..., ge=80.0, le=100.0)

class _GeocodeBatchInput(BaseModel):
    coords: list[_GeocodeBatchItem] = Field(..., min_length=1, max_length=50_000)


@app.post("/geocode/batch")
def geocode_batch(body: _GeocodeBatchInput):
    """
    Stateless batch reverse-geocode: lat/lng → geocode + hierarchy.

    Used for the local upload flow (small CSVs < ~50K rows processed in
    the browser).  The V5 raster lookup is O(1) per point, so 5 000-item
    chunks complete in well under 100 ms.  No session needed.
    """
    results = []
    for item in body.coords:
        gc = geocoder.get_geocode(item.lat, item.lng)
        h  = geocoder.build_hierarchy(gc) if gc else {}
        results.append({'id': item.id, 'geocode': gc, 'hierarchy': h})
    return {'results': results, 'count': len(results)}


# =============================================================================
# Google Auth + User Profile endpoints
# =============================================================================

@app.post("/auth/google")
async def auth_google(credential: str = Body(..., embed=True)):
    """Verify a Google ID token. Returns {session_id, sub, email, name, picture}."""
    try:
        idinfo = google_id_token.verify_oauth2_token(
            credential,
            google_auth_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError as exc:
        raise HTTPException(401, f"Invalid Google token: {exc}")

    sub     = idinfo["sub"]
    email   = idinfo.get("email", "")
    name    = idinfo.get("name", "")
    picture = idinfo.get("picture", "")

    user_dir = os.path.join(USERS_DIR, sub)
    os.makedirs(user_dir, exist_ok=True)

    profile = {"sub": sub, "email": email, "name": name, "picture": picture}
    with open(os.path.join(user_dir, "profile.json"), "w") as f:
        json.dump(profile, f)

    session_id = f"u_{sub}"
    session_mgr.get(session_id)   # pre-warm so directory + store exist immediately

    return {**profile, "session_id": session_id}


@app.get("/user/settings")
def get_user_settings(session_id: str = Query(...)):
    """Return saved display/label preferences for an authenticated user."""
    if not session_id.startswith("u_"):
        raise HTTPException(400, "Not an authenticated session")
    path = os.path.join(USERS_DIR, session_id[2:], "settings.json")
    if not os.path.exists(path):
        return {"display_field": "id"}
    with open(path) as f:
        return json.load(f)


@app.put("/user/settings")
async def put_user_settings(session_id: str = Query(...), body: dict = Body(...)):
    """Persist display/label preferences for an authenticated user."""
    if not session_id.startswith("u_"):
        raise HTTPException(400, "Not an authenticated session")
    user_dir = os.path.join(USERS_DIR, session_id[2:])
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, "settings.json")
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update({k: v for k, v in body.items()})
    with open(path, "w") as f:
        json.dump(existing, f)
    return existing


@app.get("/user/csv")
def download_user_csv(session_id: str = Query(...)):
    """Download the CSV that was uploaded to this authenticated account."""
    if not session_id.startswith("u_"):
        raise HTTPException(400, "Not an authenticated session")
    csv_path = os.path.join(USERS_DIR, session_id[2:], "input.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(404, "No CSV uploaded yet for this account")
    return FileResponse(csv_path, media_type="text/csv", filename="my_data.csv")


# =============================================================================
# Helpers
# =============================================================================

def _require_loaded(session_id: str = 'default'):
    s = session_mgr.get(session_id) if session_mgr else None
    if not s or not s.is_loaded:
        raise HTTPException(503,
            f"No data loaded for session {session_id!r}. "
            "Upload a CSV via POST /upload and wait for the build to complete.")


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
