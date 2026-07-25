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
    # --- added 2026-07-22 (Kilian's theatre list; all trimmed to the Ukraine/Kaliningrad size band) ---
    "singapore": dict(
        # Sisters' Islands / southern anchorage -> Changi naval area: urban archipelago + reservoirs
        bbox=(103.60, 1.18, 104.10, 1.48), epsg="EPSG:32648", maritime=True,
        base=("SOUTHERN GROUP", 103.74, 1.21), target=("CHANGI", 103.99, 1.39),
        bridgehead=(103.99, 1.39, 6.0),
        sea_seeds=[[10, 5], [45, 6], [25, 3], [8, 30], [50, 28]],
        title="Singapore - archipelago resupply across the southern anchorage to Changi"),
    "hongkong": dict(
        # South Lamma/Lantau approach -> Victoria Harbour: dense urban + country-park forest + sea
        bbox=(113.85, 22.15, 114.30, 22.45), epsg="EPSG:32650", maritime=True,
        base=("SOUTH LANTAU", 113.92, 22.19), target=("VICTORIA HARBOUR", 114.17, 22.30),
        bridgehead=(114.17, 22.30, 5.0),
        sea_seeds=[[8, 6], [42, 5], [22, 4], [10, 28], [40, 30]],
        title="Hong Kong - approach through the islands to Victoria Harbour"),
    "taiwan": dict(
        # Xiamen coast -> Kinmen (the Strait chokepoint at its narrowest, island vs mainland)
        bbox=(118.15, 24.30, 118.65, 24.60), epsg="EPSG:32651", maritime=True,
        base=("XIAMEN COAST", 118.20, 24.55), target=("KINMEN", 118.42, 24.43),
        bridgehead=(118.42, 24.43, 6.0),
        sea_seeds=[[25, 15], [40, 10], [12, 8], [45, 25], [30, 30]],
        title="Taiwan Strait - Xiamen coast to the Kinmen bridgehead"),
    "narva": dict(
        # Kohtla-Jarve -> Kingisepp: the Narva river border crossing is the natural pinch.
        # coast=True: the Gulf of Finland fills the north; OSM does NOT tag open sea, so without
        # this the sea reads as `open` (emplaceable!) - seeds mark the sea faces -> water.
        bbox=(27.20, 59.15, 29.10, 59.68), epsg="EPSG:32635", maritime=False, coast=True,
        base=("KOHTLA-JARVE", 27.28, 59.40), target=("KINGISEPP", 28.61, 59.37),
        sea_seeds=[[15, 57], [40, 57], [65, 57], [90, 57], [105, 52]],
        title="Kohtla-Jarve -> Kingisepp corridor (the Narva river crossing)"),
    "karelia": dict(
        # Kotka -> Lappeenranta: coastal port to inland lakeland (the Karelian lake maze)
        bbox=(26.85, 60.45, 28.20, 61.20), epsg="EPSG:32635", maritime=False,
        base=("KOTKA", 26.945, 60.467), target=("LAPPEENRANTA", 28.19, 61.058),
        title="Kotka -> Lappeenranta corridor (Finnish lakeland)"),
    "alps": dict(
        # Innsbruck -> Trento: the Brenner axis, deep alpine valleys (mountains = no-fly walls)
        bbox=(11.05, 46.05, 11.75, 47.30), epsg="EPSG:32632", maritime=False,
        base=("INNSBRUCK", 11.393, 47.267), target=("TRENTO", 11.121, 46.070),
        title="Innsbruck -> Trento (the Brenner alpine corridor)"),
    "fulda": dict(
        # THE FULDA GAP: the Cold War Warsaw-Pact invasion axis, from the inner-German border
        # observation post Point Alpha SW to Frankfurt, flanked by the forested Vogelsberg (S)
        # and Rhon (NE) uplands. Deep inland (no coast); heavy forest cover. Biased large.
        bbox=(8.45, 49.90, 10.20, 50.90), epsg="EPSG:32632", maritime=False,
        base=("POINT ALPHA", 9.938, 50.686), target=("FRANKFURT", 8.682, 50.110),
        title="The Fulda Gap - Point Alpha to Frankfurt (the Cold War invasion axis)"),
}
TAGS_SEA = {                              # land (coast + islands) = emplaceable + LOS block,
    "coast": {"natural": ["coastline"]},  # sea = traversable. urban kept (the tighter box tops
    "island": {"place": ["island", "islet"]},  # out at Musandam, N of Dubai/Sharjah, so the
    "urban": {"landuse": ["residential", "industrial", "port", "military"]},  # layer is light.
    "forest": {"natural": ["wood"], "landuse": ["forest"]},   # country-park concealment (HK/SG)
}
TAGS_LAND = {
    "water": {"natural": ["water"], "waterway": ["riverbank"],
              "landuse": ["reservoir", "basin"]},
    "urban": {"landuse": ["residential", "commercial", "industrial", "retail", "port"]},
    "forest": {"natural": ["wood"], "landuse": ["forest"]},
    "field": {"landuse": ["farmland", "meadow", "orchard"], "natural": ["grassland", "scrub"]},
    # bare high terrain = flight walls / no emplacement; only populated in alpine boxes, empty elsewhere
    "alpine": {"natural": ["bare_rock", "scree", "glacier", "cliff", "ridge"]},
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


def coastline_sea(coast_km_lines, Wkm, Hkm, base_xy, target_xy):
    """Derive the open-SEA polygon(s) from OSM coastline lines (OSM does NOT tag open sea as
    water). OSM coastline arrives as many disconnected fragments that DON'T span the box, so
    polygonize cannot close them (it collapses to one all-box face). Instead: linemerge the
    fragments into the main coast, extend its endpoints past the box edges, and SPLIT the box by
    it. The SEA side is the piece holding neither the base nor the target (both on land). Falls
    back to a buffered-coast barrier if the split degenerates. Source stays OSM."""
    from shapely.geometry import LineString, Point, box as shbox
    from shapely.ops import linemerge, split, unary_union
    segs = [LineString(l) for l in coast_km_lines if len(l) >= 2]
    if not segs:
        return []
    bx = shbox(0, 0, Wkm, Hkm)
    base, tgt = Point(*base_xy), Point(*target_xy)
    merged = linemerge(unary_union(segs))
    lines = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    main = max(lines, key=lambda l: l.length)
    cs = list(main.coords)

    def _ext(p_from, p_to, d=15.0):               # push an endpoint d km past the box edge
        vx, vy = p_to[0] - p_from[0], p_to[1] - p_from[1]
        n = (vx * vx + vy * vy) ** 0.5 or 1.0
        return (p_to[0] + vx / n * d, p_to[1] + vy / n * d)

    crossing = LineString([_ext(cs[1], cs[0])] + cs + [_ext(cs[-2], cs[-1])])
    sea = []
    try:
        pieces = [g for g in split(bx, crossing).geoms if g.area > 1.0]
        sea = [p for p in pieces if not p.contains(base) and not p.contains(tgt)]
    except Exception:
        sea = []
    if not sea:                                    # fallback: buffer the fragments into a barrier
        band = unary_union(segs).buffer(2.0)
        comps = bx.difference(band)
        comps = list(comps.geoms) if comps.geom_type == "MultiPolygon" else [comps]
        sea = [c for c in comps if c.area > 1.0 and not c.contains(base) and not c.contains(tgt)]
    if not sea:
        return []
    u = unary_union(sea)
    polys = [u] if u.geom_type == "Polygon" else list(getattr(u, "geoms", []))
    return [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords]
            for p in polys if p.area > 1.0]


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
        seeds = [kmxy(*spec["base"][1:])] + spec.get("sea_seeds", [])
        ext, holes = coastline_to_land(line_layers.get("coast", []), Wkm, Hkm, seeds,
                                       poly_layers.get("island"))
        poly_layers["land"] = ext
        poly_layers["land_holes"] = holes
        print(f"    land: {len(ext)} polygons ({len(holes)} sea inlets) from coastline",
              flush=True)
    elif spec.get("coast"):                   # coastal LAND theatre: capture the open sea as water
        print("    coast (sea capture) ...", flush=True)
        gc = fetch_layer(ox, (W, S, E, N), {"natural": ["coastline"]})
        coast_lines = []
        if gc is not None:
            gc = gc.to_crs(crs)
            lines = gc[gc.geometry.type.isin(["LineString", "MultiLineString"])]
            if len(lines):
                geom = unary_union(list(lines.geometry)).intersection(clip)
                geom = geom.simplify(SIMPLIFY_M, preserve_topology=True)
                for ln in ([geom] if geom.geom_type == "LineString"
                           else getattr(geom, "geoms", [])):
                    if ln.is_empty:
                        continue
                    xs, ys = ln.coords.xy
                    coast_lines.append([[(x - x0) / 1000, (y - y0) / 1000]
                                        for x, y in zip(xs, ys)])
        from shapely.geometry import Polygon as _Poly
        sea = coastline_sea(coast_lines, Wkm, Hkm,
                            kmxy(*spec["base"][1:]), kmxy(*spec["target"][1:]))
        poly_layers["sea"] = sea                    # own layer: sea backdrop + non-emplaceable
        line_layers.setdefault("coast", coast_lines)
        seapct = 100 * sum(_Poly(r).area for r in sea) / (Wkm * Hkm) if sea else 0.0
        print(f"    sea: {len(sea)} polygon(s), {seapct:.0f}% of box -> non-emplaceable "
              f"(was reading as emplaceable 'open')", flush=True)

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
        fill(poly_layers.get("sea", []), COL["sea"], 0)        # open-sea backdrop (below land)
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
