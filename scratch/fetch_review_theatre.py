"""Fetch + render candidate theatres from OSM for review (Hormuz maritime, Ukraine land).
Pulls real polygons/coastlines, projects to the local UTM zone, renders a review PNG with the
candidate base/target and corridor axis. NO game build yet - this is for aligning on the area.

Run: PYTHONPATH=. .venv/bin/python scratch/fetch_review_theatre.py --name hormuz
"""
from __future__ import annotations

import argparse
import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CORRIDORS = {
    # Strait of Hormuz core chokepoint: sea = highway, coasts/islands = threat emplacement.
    "hormuz": dict(
        bbox=(55.75, 26.05, 56.85, 27.00), epsg="EPSG:32640", maritime=True,
        base=("GULF OF OMAN", 56.65, 26.25), target=("PERSIAN GULF", 55.95, 26.85),
        title="Strait of Hormuz — contested transit corridor"),
    # Dnipro river corridor Dnipro city <-> Zaporizhzhia: river + urban = strong structure.
    "ukraine": dict(
        bbox=(34.80, 47.75, 35.45, 48.55), epsg="EPSG:32636", maritime=False,
        base=("DNIPRO", 35.045, 48.465), target=("ZAPORIZHZHIA", 35.145, 47.840),
        title="Dnipro corridor (Dnipro - Zaporizhzhia)"),
}
# maritime: sea is the BACKGROUND, so we fetch only LAND (coastline + islands + urban).
# land: full terrain classes. Open-sea `natural=water` is deliberately NOT fetched (it hangs
# Overpass and is the background anyway).
TAGS_SEA = {
    "coast": {"natural": ["coastline"]},
    "island": {"place": ["island", "islet"]},
    "urban": {"landuse": ["residential", "industrial", "port", "military"]},
}
TAGS_LAND = {
    "water": {"natural": ["water"], "waterway": ["riverbank"],
              "landuse": ["reservoir", "basin"]},
    "urban": {"landuse": ["residential", "commercial", "industrial", "retail", "port"]},
    "forest": {"natural": ["wood"], "landuse": ["forest"]},
    "field": {"landuse": ["farmland", "meadow", "orchard"], "natural": ["grassland", "scrub"]},
}
COL = dict(sea="#a3bccf", water="#8fb0c6", land="#e7dec9", island="#ddd3ba",
           urban="#c3b9a6", forest="#a9c089", field="#dfe0b8", coast="#6b6350")


def main():
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import Point, box
    from shapely.ops import unary_union

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(CORRIDORS))
    args = ap.parse_args()
    spec = CORRIDORS[args.name]
    W, S, E, N = spec["bbox"]
    crs = spec["epsg"]
    ox.settings.requests_timeout = 90

    corners = gpd.GeoSeries([Point(W, S), Point(E, N)], crs="EPSG:4326").to_crs(crs)
    x0, y0, x1, y1 = corners.iloc[0].x, corners.iloc[0].y, corners.iloc[1].x, corners.iloc[1].y
    Wkm, Hkm = (x1 - x0) / 1000, (y1 - y0) / 1000
    clip = box(x0, y0, x1, y1)
    print(f"[{args.name}] {Wkm:.0f} x {Hkm:.0f} km ({crs}); fetching OSM ...", flush=True)

    def _alarm(sig, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)

    tagset = TAGS_SEA if spec["maritime"] else TAGS_LAND
    layers = {}
    for cls, tags in tagset.items():
        signal.alarm(75)                              # hard cap per layer; skip if it hangs
        try:
            g = ox.features_from_bbox((W, S, E, N), tags=tags).to_crs(crs)
            layers[cls] = g[~g.geometry.is_empty]
            print(f"    {cls}: {len(layers[cls])} features", flush=True)
        except Exception as e:
            print(f"    {cls}: SKIPPED ({type(e).__name__})", flush=True)
            layers[cls] = None
        finally:
            signal.alarm(0)
    for cls in ("water", "coast", "island", "urban", "forest", "field"):
        layers.setdefault(cls, None)

    fig, ax = plt.subplots(figsize=(11, 11 * Hkm / Wkm))
    ax.set_facecolor(COL["sea"] if spec["maritime"] else COL["land"])

    def draw_polys(g, color, z, ec=None, lw=0):
        if g is None:
            return
        polys = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])]
        for geom in polys.geometry:
            geom = geom.intersection(clip)
            for p in ([geom] if geom.geom_type == "Polygon" else getattr(geom, "geoms", [])):
                if p.is_empty:
                    continue
                xs, ys = p.exterior.coords.xy
                ax.fill([(x - x0) / 1000 for x in xs], [(y - y0) / 1000 for y in ys],
                        color=color, zorder=z, ec=ec or "none", lw=lw)

    if spec["maritime"]:
        # sea background already set; draw land where OSM gives it (islands + coastline polys)
        draw_polys(layers["island"], COL["island"], 2, ec=COL["coast"], lw=0.4)
        # coastline: OSM ways; draw as lines to show the shores
        c = layers["coast"]
        if c is not None:
            lines = c[c.geometry.type.isin(["LineString", "MultiLineString"])]
            for geom in lines.geometry:
                geom = geom.intersection(clip)
                for ln in ([geom] if geom.geom_type == "LineString" else getattr(geom, "geoms", [])):
                    if ln.is_empty:
                        continue
                    xs, ys = ln.coords.xy
                    ax.plot([(x - x0) / 1000 for x in xs], [(y - y0) / 1000 for y in ys],
                            color=COL["coast"], lw=1.3, zorder=3)
        draw_polys(layers["water"], COL["water"], 1)
    else:
        draw_polys(layers["field"], COL["field"], 1)
        draw_polys(layers["forest"], COL["forest"], 2)
        draw_polys(layers["water"], COL["water"], 3)
        draw_polys(layers["urban"], COL["urban"], 4, ec="#9c917d", lw=0.3)

    # endpoints + corridor axis
    def km(lon, lat):
        p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
        return (p.x - x0) / 1000, (p.y - y0) / 1000
    bx, by = km(*spec["base"][1:]); tx, ty = km(*spec["target"][1:])
    ax.plot([bx, tx], [by, ty], "--", color="#274c86", lw=1.6, zorder=8, alpha=.8)
    for (x, y), lab, mk in (((bx, by), spec["base"][0], "s"), ((tx, ty), spec["target"][0], "*")):
        ax.scatter([x], [y], marker=mk, s=180 if mk == "*" else 90, zorder=9,
                   facecolor="#274c86", edgecolor="white", lw=1.5)
        ax.annotate(lab, (x, y), (7, 7), textcoords="offset points", zorder=9,
                    fontsize=10, fontweight="bold", color="#1c3a66")
    # graticule
    for gx in np.arange(0, Wkm, 20):
        ax.axvline(gx, color="#8a806a", lw=.3, alpha=.4, zorder=6)
    for gy in np.arange(0, Hkm, 20):
        ax.axhline(gy, color="#8a806a", lw=.3, alpha=.4, zorder=6)
    ax.plot([2, 22], [2, 2], color="#2c2820", lw=2.5, zorder=10)
    ax.annotate("20 km", (12, 2.8), ha="center", fontsize=8, zorder=10)
    ax.set_xlim(0, Wkm); ax.set_ylim(0, Hkm); ax.set_aspect("equal")
    ax.set_title(f"{spec['title']}   ·   {Wkm:.0f} x {Hkm:.0f} km", fontsize=12, loc="left")
    ax.set_xlabel("km E of SW corner"); ax.set_ylabel("km N of SW corner")
    fig.tight_layout()
    out = f"assets/review_{args.name}.png"
    fig.savefig(out, dpi=150)
    print(f"[{args.name}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
