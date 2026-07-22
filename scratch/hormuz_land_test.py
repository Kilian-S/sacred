"""Polygonise the fetched coastline into filled LAND (fixes the 'mainland is just squiggles'
render). Reads the saved theatre_hormuz_vec.json, closes the coastline against the bbox,
classifies faces as sea (containing a sea seed) vs land, and re-renders. No re-fetch."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from shapely.geometry import LineString, Point, box as shbox
from shapely.ops import polygonize, unary_union

COL = dict(sea="#a3bccf", land="#e7dec9", coast="#6b6350", urban="#c3b9a6",
           friend="#274c86", hostile="#b23524", secured="#3f7a4e", ink="#2c2820")

d = json.load(open("data/maps/theatre_hormuz_vec.json"))
Wkm, Hkm = d["W_km"], d["H_km"]
carrier = d["base"]["xy_km"]
# sea seeds in obvious open water (km); a face containing any of these is sea
SEEDS = [carrier, [130, 55], [40, 55], [150, 105], [95, 40], [175, 70]]


def polygonise_land():
    segs = [LineString(l) for l in d["line"]["coast"] if len(l) >= 2]
    bnd = shbox(0, 0, Wkm, Hkm).boundary
    merged = unary_union(segs + [bnd])
    faces = list(polygonize(merged))
    seeds = [Point(s) for s in SEEDS]
    land = [f for f in faces if not any(f.contains(sp) for sp in seeds)]
    # union with the place=island polygons (belt-and-braces)
    isl = [Point  # placeholder
           ] if False else []
    for r in d["poly"]["island"]:
        if len(r) >= 3:
            from shapely.geometry import Polygon
            land.append(Polygon(r))
    return unary_union(land)


land = polygonise_land()
print(f"faces->land area: {land.area:.0f} km^2 of {Wkm*Hkm:.0f} km^2 "
      f"({100*land.area/(Wkm*Hkm):.0f}% land)")

# save the land layer (exteriors + sea-inlet holes) into the game data
polys_ = [land] if land.geom_type == "Polygon" else list(land.geoms)
d["poly"]["land"] = [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords] for p in polys_]
d["poly"]["land_holes"] = [[[round(x, 3), round(y, 3)] for x, y in h.coords]
                           for p in polys_ for h in p.interiors]
d["line"].pop("coast", None)                       # coast lines superseded by the land polygons
json.dump(d, open("data/maps/theatre_hormuz_vec.json", "w"))
print(f"saved poly[land]: {len(d['poly']['land'])} polygons, "
      f"{len(d['poly']['land_holes'])} sea-inlet holes")

fig, ax = plt.subplots(figsize=(11, 11 * Hkm / Wkm))
ax.set_facecolor(COL["sea"])
polys = [land] if land.geom_type == "Polygon" else list(land.geoms)
for p in polys:
    xs, ys = p.exterior.coords.xy
    ax.fill(xs, ys, color=COL["land"], ec=COL["coast"], lw=0.8, zorder=2)
    ax.fill(xs, ys, color=COL["hostile"], alpha=0.22, zorder=3, lw=0)
    for hole in p.interiors:                       # sea inlets
        hx, hy = hole.coords.xy
        ax.fill(hx, hy, color=COL["sea"], zorder=4, lw=0)
for r in d["poly"]["urban"]:
    ax.fill([p[0] for p in r], [p[1] for p in r], color=COL["urban"], zorder=5, lw=0)
bx, by = d["bridgehead"]["xy_km"]
ax.add_patch(Circle((bx, by), d["bridgehead"]["radius_km"], facecolor=COL["secured"],
                    alpha=0.32, edgecolor=COL["secured"], lw=1.6, zorder=6))
cx, cy = carrier; tx, ty = d["target"]["xy_km"]
ax.plot([cx, tx], [cy, ty], "--", color=COL["friend"], lw=1.8, zorder=8, alpha=.85)
for (x, y), lab, mk in (((cx, cy), d["base"]["label"], "s"), ((tx, ty), d["target"]["label"], "*")):
    ax.scatter([x], [y], marker=mk, s=200 if mk == "*" else 95, zorder=9,
               facecolor=COL["friend"], edgecolor="white", lw=1.6)
    ax.annotate(lab, (x, y), (8, 8), textcoords="offset points", zorder=9, fontsize=10,
                fontweight="bold", color="#1c3a66")
ax.plot([3, 23], [3, 3], color=COL["ink"], lw=2.5, zorder=10)
ax.annotate("20 km", (13, 4), ha="center", fontsize=8, zorder=10)
ax.set_xlim(0, Wkm); ax.set_ylim(0, Hkm); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Hormuz — carrier drone resupply to the Bandar Abbas bridgehead  ·  "
             f"{Wkm:.0f} x {Hkm:.0f} km (real OSM)", fontsize=11.5, loc="left")
fig.tight_layout()
fig.savefig("assets/theatre_hormuz_osm.png", dpi=150)
print("wrote assets/theatre_hormuz_osm.png")
