"""Fetch the REAL OSM terrain for the two agreed theatres and save game-ready vector data +
a detailed map. Run on a machine with a stable network (osmnx caches, so re-runs resume).

  Hormuz (maritime): carrier drone resupply to the Bandar Abbas bridgehead. Land (Iran coast,
    strait islands) = emplaceable/hostile; sea = traversable. Fetches coastline + islands + urban.
  Dnipro  (land): supply Dnipro -> Zaporizhzhia along the river. Fetches water/urban/forest/field.

Outputs per theatre:
  data/maps/theatre_<name>_vec.json   (simplified rings in km, + base/target/bridgehead/bbox)
  assets/theatre_<name>_osm.png       (detailed review render)

Run BOTH:  PYTHONPATH=. .venv/bin/python scratch/fetch_new_theatres.py --name both
Run one:   PYTHONPATH=. .venv/bin/python scratch/fetch_new_theatres.py --name hormuz
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

CORRIDORS = {
    "hormuz": dict(
        bbox=(54.90, 25.75, 57.45, 27.42), epsg="EPSG:32640", maritime=True,
        base=("CARRIER GROUP", 57.05, 25.90), target=("BANDAR ABBAS", 56.28, 27.18),
        bridgehead=(56.28, 27.18, 14.0),
        title="Hormuz — carrier drone resupply to the Bandar Abbas bridgehead"),
    "ukraine": dict(
        bbox=(34.80, 47.75, 35.45, 48.55), epsg="EPSG:32636", maritime=False,
        base=("DNIPRO", 35.045, 48.465), target=("ZAPORIZHZHIA", 35.145, 47.840),
        title="Dnipro -> Zaporizhzhia resupply corridor"),
}
TAGS_SEA = {                              # land (coast+islands) = emplaceable + LOS block;
    "coast": {"natural": ["coastline"]},  # sea = traversable. urban kept but LIGHT (residential/
    "island": {"place": ["island", "islet"]},  # industrial/port only); the box now ends at RAK,
    "urban": {"landuse": ["residential", "industrial", "port"]},  # cropping Dubai/Sharjah.
}
TAGS_LAND = {
    "water": {"natural": ["water"], "waterway": ["riverbank"],
              "landuse": ["reservoir", "basin"]},
    "urban": {"landuse": ["residential", "commercial", "industrial", "retail", "port"]},
    "forest": {"natural": ["wood"], "landuse": ["forest"]},
    "field": {"landuse": ["farmland", "meadow", "orchard"], "natural": ["grassland", "scrub"]},
}
SIMPLIFY_M = 80.0
COL = dict(sea="#a3bccf", water="#8fb0c6", land="#e7dec9", island="#ddd3ba", coast="#6b6350",
           urban="#c3b9a6", forest="#a9c089", field="#dfe0b8", friend="#274c86",
           hostile="#b23524", secured="#3f7a4e", ink="#2c2820", grid="#8a806a")


def fetch_layer(ox, bbox, tags, tries=3):
    for k in range(tries):
        try:
            return ox.features_from_bbox(bbox, tags=tags)
        except Exception as e:
            print(f"      attempt {k+1}/{tries} failed ({type(e).__name__}); retrying...",
                  flush=True)
            time.sleep(8)
    return None


def run(name):
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import Point, box
    from shapely.ops import unary_union

    spec = CORRIDORS[name]
    W, S, E, N = spec["bbox"]; crs = spec["epsg"]
    ox.settings.requests_timeout = 1500
    ox.settings.use_cache = True

    corners = gpd.GeoSeries([Point(W, S), Point(E, N)], crs="EPSG:4326").to_crs(crs)
    x0, y0, x1, y1 = corners.iloc[0].x, corners.iloc[0].y, corners.iloc[1].x, corners.iloc[1].y
    Wkm, Hkm = (x1 - x0) / 1000, (y1 - y0) / 1000
    clip = box(x0, y0, x1, y1)

    def kmxy(lon, lat):
        p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
        return [(p.x - x0) / 1000, (p.y - y0) / 1000]

    print(f"[{name}] {Wkm:.0f} x {Hkm:.0f} km ({crs}); fetching OSM (patient, cached)...",
          flush=True)
    tagset = TAGS_SEA if spec["maritime"] else TAGS_LAND
    poly_layers, line_layers = {}, {}
    for cls, tags in tagset.items():
        print(f"    {cls} ...", flush=True)
        g = fetch_layer(ox, (W, S, E, N), tags)
        if g is None:
            print(f"    {cls}: FAILED after retries (will be empty)", flush=True)
            poly_layers[cls], line_layers[cls] = [], []
            continue
        g = g.to_crs(crs)
        polys = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])]
        rings = []
        if len(polys):
            geom = unary_union(list(polys.geometry)).buffer(0).intersection(clip)
            geom = geom.simplify(SIMPLIFY_M, preserve_topology=True)
            for p in ([geom] if geom.geom_type == "Polygon" else getattr(geom, "geoms", [])):
                if p.is_empty or p.area / 1e6 < 0.02:
                    continue
                xs, ys = p.exterior.coords.xy
                rings.append([[round((x - x0) / 1000, 3), round((y - y0) / 1000, 3)]
                              for x, y in zip(xs, ys)])
        poly_layers[cls] = rings
        lines = g[g.geometry.type.isin(["LineString", "MultiLineString"])]
        lrs = []
        if len(lines):
            geom = unary_union(list(lines.geometry)).intersection(clip)
            geom = geom.simplify(SIMPLIFY_M, preserve_topology=True)
            for ln in ([geom] if geom.geom_type == "LineString" else getattr(geom, "geoms", [])):
                if ln.is_empty:
                    continue
                xs, ys = ln.coords.xy
                lrs.append([[round((x - x0) / 1000, 3), round((y - y0) / 1000, 3)]
                            for x, y in zip(xs, ys)])
        line_layers[cls] = lrs
        print(f"    {cls}: {len(rings)} polygons, {len(lrs)} lines", flush=True)

    out = dict(name=name, W_km=round(Wkm, 3), H_km=round(Hkm, 3), epsg=crs,
               maritime=spec["maritime"], bbox_lonlat=list(spec["bbox"]),
               poly=poly_layers, line=line_layers,
               base=dict(label=spec["base"][0], xy_km=kmxy(*spec["base"][1:])),
               target=dict(label=spec["target"][0], xy_km=kmxy(*spec["target"][1:])))
    if "bridgehead" in spec:
        blo, bla, br = spec["bridgehead"]
        out["bridgehead"] = dict(xy_km=kmxy(blo, bla), radius_km=br)
    path = f"data/maps/theatre_{name}_vec.json"
    json.dump(out, open(path, "w"))
    print(f"[{name}] wrote {path}", flush=True)

    # detailed render
    fig, ax = plt.subplots(figsize=(11, 11 * Hkm / Wkm))
    ax.set_facecolor(COL["sea"] if spec["maritime"] else COL["land"])
    hostile = spec["maritime"]

    def fill(rings, color, z, ec=None, lw=0, alpha=1):
        for r in rings:
            ax.fill([p[0] for p in r], [p[1] for p in r], color=color, zorder=z,
                    ec=ec or "none", lw=lw, alpha=alpha)

    if spec["maritime"]:
        fill(poly_layers["island"], COL["island"], 2, ec=COL["coast"], lw=0.6)
        fill(poly_layers["island"], COL["hostile"], 3, alpha=0.28)
        for r in line_layers["coast"]:
            ax.plot([p[0] for p in r], [p[1] for p in r], color=COL["coast"], lw=1.3, zorder=3)
        fill(poly_layers.get("urban", []), COL["urban"], 4, ec="#9c917d", lw=0.3)
        if "bridgehead" in out:
            bx, by = out["bridgehead"]["xy_km"]
            ax.add_patch(Circle((bx, by), out["bridgehead"]["radius_km"], facecolor=COL["secured"],
                                alpha=0.32, edgecolor=COL["secured"], lw=1.6, zorder=5))
    else:
        fill(poly_layers["field"], COL["field"], 1)
        fill(poly_layers["forest"], COL["forest"], 2)
        fill(poly_layers["water"], COL["water"], 3)
        fill(poly_layers["urban"], COL["urban"], 4, ec="#9c917d", lw=0.3)

    bx, by = out["base"]["xy_km"]; tx, ty = out["target"]["xy_km"]
    ax.plot([bx, tx], [by, ty], "--", color=COL["friend"], lw=1.8, zorder=8, alpha=.85)
    for (x, y), lab, mk in (((bx, by), out["base"]["label"], "s"),
                            ((tx, ty), out["target"]["label"], "*")):
        ax.scatter([x], [y], marker=mk, s=200 if mk == "*" else 95, zorder=9,
                   facecolor=COL["friend"], edgecolor="white", lw=1.6)
        ax.annotate(lab, (x, y), (8, 8), textcoords="offset points", zorder=9,
                    fontsize=10, fontweight="bold", color="#1c3a66")
    ax.plot([3, 23], [3, 3], color=COL["ink"], lw=2.5, zorder=10)
    ax.annotate("20 km", (13, 3.8), ha="center", fontsize=8, zorder=10)
    ax.set_xlim(0, Wkm); ax.set_ylim(0, Hkm); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{spec['title']}   ·   {Wkm:.0f} x {Hkm:.0f} km (real OSM)", fontsize=11.5, loc="left")
    fig.tight_layout()
    png = f"assets/theatre_{name}_osm.png"
    fig.savefig(png, dpi=150)
    print(f"[{name}] wrote {png}\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="both", choices=list(CORRIDORS) + ["both"])
    args = ap.parse_args()
    names = list(CORRIDORS) if args.name == "both" else [args.name]
    for nm in names:
        run(nm)
    print("DONE. Commit the new data/maps/theatre_*_vec.json + assets/theatre_*_osm.png.")


if __name__ == "__main__":
    main()
