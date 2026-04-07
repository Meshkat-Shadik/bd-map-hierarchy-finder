---
title: BD Location Aggregator V6
emoji: 📍
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
---

# BD Location Aggregator V6

Upload any CSV of locations (schools, hospitals, mosques, clinics — anything with `id,lat,lng`) and instantly query them by Bangladesh administrative hierarchy.

## How it works

1. **Upload** a CSV → server geocodes each row using the V5 rasterized grid
2. **Query** by any area code: get all entities in a division, district, upazila, union, or mauza

## Endpoints

### POST `/upload`
Upload a CSV file and trigger index build (runs in background).
```
curl -X POST -F "file=@schools.csv" https://<space>/upload
```
CSV format: `id,lat,lng` (id column is optional)

### GET `/status`
Check build progress and entity counts.

### GET `/entity/{id}`
Get a single entity with its full administrative hierarchy.
```json
{
  "id": "SCHOOL001",
  "lat": 23.8103, "lng": 90.4125,
  "geocode": "302600470102034",
  "hierarchy": {
    "division":  {"geocode": "30",   "name": "Dhaka"},
    "district":  {"geocode": "3026", "name": "Dhaka"},
    "upazila":   {"geocode": "30260047", "name": "Savar"}
  }
}
```

### GET `/entity/{id}/peers?level=district`
All entities in the same area as `{id}` at the given level.

Supported levels: `division`, `district`, `upazila`, `union`, `mauza`, `village`, `ea`

### GET `/area/{geocode}?limit=1000`
All entities in a specific area by geocode.

### GET `/areas?level=division`
List all areas at a given level with entity counts.

### GET `/health`
Basic health check.
# bd-map-hierarchy-finder
