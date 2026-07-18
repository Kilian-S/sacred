"""gen28 v3-theatre: build a REAL-terrain lattice from OSM land cover (Kilian 2026-07-18:
'take a section of a real map', 'base and demand node must not be static').

Pulls OSM land-cover polygons for a real corridor bbox, projects to UTM (metres), rasterises
to a ~1 km terrain-class grid by cell-centroid priority (water > urban > forest > field >
open), and records two REAL settlements as arbitrary base/target cells (not centred). Output
is a cached JSON the env loads (no repeat network calls); the raw GeoJSON is cached too.

Terrain -> game mechanics (set in the env, documented here):
  water : flyable, NO interceptor emplacement (cannot site a ground SAM on water), high exposure
  urban : flyable, emplacement BLOCKED + line-of-sight blocked (dense cores shield the corridor)
  forest: flyable (small cost), SHORT-range concealed ambush can emplace (high p, small r)
  field : flyable, LONG-range emplacement (high p, large r): direct but exposed open ground
  open  : default/unknown ground, field-like

Run: PYTHONPATH=. .venv/bin/python scratch/fetch_theatre.py --name kgd_gvardeysk
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# named real corridors (lon/lat); base/target are REAL settlements, arbitrary positions.
CORRIDORS = {
    "kgd_gvardeysk": dict(
        bbox=(20.42, 54.60, 21.12, 54.78),          # W,S,E,N ~ 45 x 20 km
        base=("Kaliningrad centre", 20.5100, 54.7100),
        target=("Gvardeysk", 21.0500, 54.6500),
    ),
    "kgd_baltiysk": dict(
        bbox=(19.85, 54.60, 20.60, 54.78),
        base=("Kaliningrad centre", 20.5100, 54.7100),
        target=("Baltiysk naval base", 19.9150, 54.6500),
    ),
}

CLASS_TAGS = {
    "water": {"natural": ["water", "wetland", "bay", "strait"],
              "waterway": ["riverbank", "river", "dock"],
              "landuse": ["reservoir", "basin"]},
    "urban": {"landuse": ["residential", "commercial", "industrial", "retail", "port",
                          "military"]},
    "forest": {"natural": ["wood"], "landuse": ["forest"]},
    "field": {"landuse": ["farmland", "meadow", "farmyard", "orchard", "grass",
                          "greenfield", "vineyard"],
              "natural": ["grassland", "scrub", "heath"]},
}
PRIORITY = ["water", "urban", "forest", "field"]     # highest first; default = "open"
CLASS_ID = {"open": 0, "field": 1, "forest": 2, "urban": 3, "water": 4}


def main():
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="kgd_gvardeysk", choices=list(CORRIDORS))
    ap.add_argument("--cell-km", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    spec = CORRIDORS[args.name]
    W, S, E, N = spec["bbox"]
    ox.settings.requests_timeout = 180

    print(f"[theatre] {args.name}: pulling OSM land cover for bbox {spec['bbox']} ...", flush=True)
    per_class = {}
    for cls, tags in CLASS_TAGS.items():
        try:
            g = ox.features_from_bbox((W, S, E, N), tags=tags)
            g = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])]
            per_class[cls] = g
            print(f"    {cls}: {len(g)} polygons", flush=True)
        except Exception as e:
            print(f"    {cls}: none ({type(e).__name__})", flush=True)
            per_class[cls] = None

    # UTM 34N (Kaliningrad) so cells are metric squares and base/target project cleanly
    crs = "EPSG:32634"
    corners = gpd.GeoSeries([Point(W, S), Point(E, N)], crs="EPSG:4326").to_crs(crs)
    x0, y0 = corners.iloc[0].x, corners.iloc[0].y
    x1, y1 = corners.iloc[1].x, corners.iloc[1].y
    cell = args.cell_km * 1000.0
    ncol = int(np.ceil((x1 - x0) / cell))     # x = easting
    nrow = int(np.ceil((y1 - y0) / cell))     # y = northing
    print(f"[theatre] grid {ncol} x {nrow} cells @ {args.cell_km} km "
          f"(~{ncol*args.cell_km:.0f} x {nrow*args.cell_km:.0f} km)", flush=True)

    trees = {}
    for cls in PRIORITY:
        g = per_class.get(cls)
        if g is None or len(g) == 0:
            trees[cls] = None
            continue
        geoms = list(g.to_crs(crs).geometry.values)
        trees[cls] = (STRtree(geoms), geoms)

    grid = np.zeros((nrow, ncol), dtype=int)          # class id per cell (row=north idx, col=east)
    for r in range(nrow):
        for c in range(ncol):
            cx = x0 + (c + 0.5) * cell
            cy = y0 + (r + 0.5) * cell
            pt = Point(cx, cy)
            klass = "open"
            for cls in PRIORITY:                       # priority order
                tr = trees[cls]
                if tr is None:
                    continue
                st, geoms = tr
                hit = any(geoms[i].contains(pt) for i in st.query(pt))
                if hit:
                    klass = cls
                    break
            grid[r, c] = CLASS_ID[klass]

    def to_cell(lon, lat):
        p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
        c = int((p.x - x0) / cell); r = int((p.y - y0) / cell)
        return (max(0, min(nrow - 1, r)), max(0, min(ncol - 1, c)))

    base_rc = to_cell(spec["base"][1], spec["base"][2])
    tgt_rc = to_cell(spec["target"][1], spec["target"][2])
    counts = {k: int((grid == v).sum()) for k, v in CLASS_ID.items()}
    tot = grid.size
    print(f"[theatre] terrain mix: " +
          " ".join(f"{k} {100*counts[k]/tot:.0f}%" for k in CLASS_ID), flush=True)
    print(f"[theatre] base {spec['base'][0]} -> cell {base_rc}; "
          f"target {spec['target'][0]} -> cell {tgt_rc}", flush=True)

    out = args.out or f"data/maps/theatre_{args.name}.json"
    json.dump({
        "name": args.name, "cell_km": args.cell_km, "crs": crs,
        "origin_xy": [x0, y0], "cell_m": cell, "nrow": nrow, "ncol": ncol,
        "class_id": CLASS_ID, "grid": grid.tolist(),
        "base": {"label": spec["base"][0], "cell": list(base_rc)},
        "target": {"label": spec["target"][0], "cell": list(tgt_rc)},
        "bbox_lonlat": list(spec["bbox"]),
    }, open(out, "w"))
    print(f"[theatre] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
