"""gen28 v3-theatre VECTOR substrate (Kilian 2026-07-18: 'why are we rasterising so much?').

Keeps the terrain as the REAL OSM POLYGONS (smooth forest/water/urban/farmland shapes), not a
raster grid. The game only needs point-in-polygon (can a hazard emplace here?) and
segment-crosses-polygon (does urban mask line of sight?) tests, which shapely does directly on
vector geometry, and the policy reads per-route features, not pixels: map detail is decoupled
from training cost entirely. Output = one JSON of simplified exterior rings per class (in km,
origin at the bbox SW corner) + the two real settlements as continuous km points.

Run: PYTHONPATH=. .venv/bin/python scratch/fetch_theatre_vector.py --name kgd_gvardeysk
"""
from __future__ import annotations

import argparse
import json

import numpy as np

CORRIDORS = {
    "kgd_gvardeysk": dict(
        bbox=(20.42, 54.60, 21.12, 54.78),
        base=("KALININGRAD", 20.5100, 54.7100),
        target=("GVARDEYSK", 21.0500, 54.6500)),
    "kgd_baltiysk": dict(
        bbox=(19.80, 54.58, 20.60, 54.80),
        base=("KALININGRAD", 20.5100, 54.7100),
        target=("BALTIYSK", 19.9150, 54.6500)),
}
CLASS_TAGS = {
    "water": {"natural": ["water", "wetland", "bay", "strait"],
              "waterway": ["riverbank", "dock"], "landuse": ["reservoir", "basin"]},
    "urban": {"landuse": ["residential", "commercial", "industrial", "retail", "port",
                          "military", "construction"]},
    "forest": {"natural": ["wood"], "landuse": ["forest"]},
    "field": {"landuse": ["farmland", "meadow", "farmyard", "orchard", "greenfield", "vineyard"],
              "natural": ["grassland", "scrub", "heath"]},
}
MIN_AREA_KM2 = {"water": 0.03, "urban": 0.05, "forest": 0.08, "field": 0.15}
SIMPLIFY_M = 90.0


def main():
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import Point
    from shapely.ops import unary_union

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="kgd_gvardeysk", choices=list(CORRIDORS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    spec = CORRIDORS[args.name]
    W, S, E, N = spec["bbox"]
    ox.settings.requests_timeout = 180
    crs = "EPSG:32634"

    corners = gpd.GeoSeries([Point(W, S), Point(E, N)], crs="EPSG:4326").to_crs(crs)
    x0, y0 = corners.iloc[0].x, corners.iloc[0].y
    Wkm = (corners.iloc[1].x - x0) / 1000.0
    Hkm = (corners.iloc[1].y - y0) / 1000.0
    from shapely.geometry import box
    clip = box(x0, y0, corners.iloc[1].x, corners.iloc[1].y)

    print(f"[vec] {args.name}: {Wkm:.0f} x {Hkm:.0f} km; pulling OSM polygons ...", flush=True)
    classes = {}
    for cls, tags in CLASS_TAGS.items():
        try:
            g = ox.features_from_bbox((W, S, E, N), tags=tags)
            g = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(crs)
            geom = unary_union(list(g.geometry.values)).buffer(0).intersection(clip)
            geom = geom.simplify(SIMPLIFY_M, preserve_topology=True)
            rings = []
            polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
            for p in polys:
                if p.is_empty or p.area / 1e6 < MIN_AREA_KM2[cls]:
                    continue
                xs, ys = p.exterior.coords.xy
                ring = [[round((x - x0) / 1000.0, 3), round((y - y0) / 1000.0, 3)]
                        for x, y in zip(xs, ys)]
                rings.append(ring)
            classes[cls] = rings
            print(f"    {cls}: {len(rings)} polygons (>= {MIN_AREA_KM2[cls]} km2)", flush=True)
        except Exception as e:
            print(f"    {cls}: none ({type(e).__name__})", flush=True)
            classes[cls] = []

    def to_km(lon, lat):
        p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
        return [round((p.x - x0) / 1000.0, 3), round((p.y - y0) / 1000.0, 3)]

    out = args.out or f"data/maps/theatre_{args.name}_vec.json"
    json.dump({
        "name": args.name, "W_km": round(Wkm, 3), "H_km": round(Hkm, 3),
        "classes": classes,
        "base": {"label": spec["base"][0], "xy_km": to_km(spec["base"][1], spec["base"][2])},
        "target": {"label": spec["target"][0], "xy_km": to_km(spec["target"][1], spec["target"][2])},
        "bbox_lonlat": list(spec["bbox"]),
    }, open(out, "w"))
    print(f"[vec] base {out.split('/')[-1]}: base {spec['base'][0]} @ "
          f"{to_km(spec['base'][1], spec['base'][2])}, target {spec['target'][0]} @ "
          f"{to_km(spec['target'][1], spec['target'][2])} km", flush=True)
    print(f"[vec] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
