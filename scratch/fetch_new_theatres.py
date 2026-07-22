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
        bbox=(55.15, 25.85, 57.20, 27.35), epsg="EPSG:32640", maritime=True,
        base=("CARRIER GROUP", 57.05, 25.95), target=("BANDAR ABBAS", 56.28, 27.18),
        bridgehead=(56.28, 27.18, 14.0),
        # open-water seeds (km) to classify coastline faces as sea vs land (carrier auto-added)
        sea_seeds=[[130, 55], [40, 55], [150, 105], [95, 40], [175, 70]],
        title="Hormuz — carrier drone resupply to the Bandar Abbas bridgehead"),
    "ukraine": dict(
        bbox=(34.80, 47.75, 35.45, 48.55), epsg="EPSG:32636", maritime=False,
        base=("DNIPRO", 35.045, 48.465), target=("ZAPORIZHZHIA", 35.145, 47.840),
        title="Dnipro -> Zaporizhzhia resupply corridor"),
    # the original WIDE strait (retry with good network): ~360x300 km, ~43x the Overpass
    # per-query limit so it fragments into MANY sub-queries; urban over Dubai/Sharjah is the
    # heavy one. Kept separate from the tight carrier theatre so both survive.
    "hormuz_wide": dict(
        bbox=(54.20, 24.90, 57.80, 27.60), epsg="EPSG:32640", maritime=True,
        base=("CARRIER GROUP", 57.50, 25.30), target=("BANDAR ABBAS", 56.28, 27.18),
        bridgehead=(56.28, 27.18, 14.0),
        sea_seeds_ll=[(56.4, 26.0), (57.4, 25.4), (54.5, 26.7), (56.0, 25.3), (57.0, 26.4)],
        title="Hormuz WIDE — full strait (Iran / Qeshm / Musandam / UAE)"),
}
TAGS_SEA = {                              # land (coast + islands) = emplaceable + LOS block,
    "coast": {"natural": ["coastline"]},  # sea = traversable. urban kept (the tighter box tops
    "island": {"place": ["island", "islet"]},  # out at Musandam, N of Dubai/Sharjah, so the
    "urban": {"landuse": ["residential", "industrial", "port", "military"]},  # layer is light.
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


def coastline_to_land(coast_km_lines, Wkm, Hkm, sea_seeds, island_rings=None):
    """Turn OSM coastline LINES into filled LAND polygons: close them against the bbox, then a
    face is SEA if it contains any open-water seed, else LAND. Returns (exterior rings, hole rings
    = sea inlets). Islands (place=island polygons) are unioned in for good measure."""
    from shapely.geometry import LineString, Point, Polygon, box as shbox
    from shapely.ops import polygonize, unary_union
    segs = [LineString(l) for l in coast_km_lines if len(l) >= 2]
    merged = unary_union(segs + [shbox(0, 0, Wkm, Hkm).boundary])
    seeds = [Point(s) for s in sea_seeds]
    land = [f for f in polygonize(merged) if not any(f.contains(sp) for sp in seeds)]
    land += [Polygon(r) for r in (island_rings or []) if len(r) >= 3]
    u = unary_union(land)
    polys = [u] if u.geom_type == "Polygon" else list(getattr(u, "geoms", []))
    ext = [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords] for p in polys]
    holes = [[[round(x, 3), round(y, 3)] for x, y in h.coords] for p in polys for h in p.interiors]
    return ext, holes


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

    if spec["maritime"]:                      # coastline lines -> filled land (+ sea inlets)
        seeds = ([kmxy(*spec["base"][1:])] + spec.get("sea_seeds", [])
                 + [kmxy(lo, la) for lo, la in spec.get("sea_seeds_ll", [])])
        ext, holes = coastline_to_land(line_layers.get("coast", []), Wkm, Hkm, seeds,
                                       poly_layers.get("island"))
        poly_layers["land"] = ext
        poly_layers["land_holes"] = holes
        print(f"    land: {len(ext)} polygons ({len(holes)} sea inlets) from coastline",
              flush=True)

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
        fill(poly_layers.get("land", []), COL["land"], 2, ec=COL["coast"], lw=0.8)
        fill(poly_layers.get("land", []), COL["hostile"], 3, alpha=0.22)
        fill(poly_layers.get("land_holes", []), COL["sea"], 4)     # sea inlets punched back
        fill(poly_layers.get("urban", []), COL["urban"], 5, ec="#9c917d", lw=0.3)
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
