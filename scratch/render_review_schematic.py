"""Offline AREA-REVIEW maps for the two proposed theatres, from bundled Natural Earth coastlines
(no network needed). Coarse by design: the goal is to agree on WHERE the theatre sits and how the
corridor is framed. Fine OSM terrain (islands, urban, forest, exact river banks) is fetched at
BUILD time on a machine with reliable Overpass access (see note in the reply).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box, LineString

NE = ("/Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/lib/python3.13/site-packages/"
      "pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp")
COL = dict(sea="#9fbdd0", land="#e7dec9", coast="#6b6350", river="#7fa6c4",
           friend="#274c86", grid="#8a806a", ink="#2c2820")

# approximate Dnipro course through the Dnipro->Zaporizhzhia corridor (lon,lat; for framing only)
DNIPRO = [(35.02, 48.55), (35.05, 48.47), (35.12, 48.40), (35.08, 48.31), (34.98, 48.22),
          (35.02, 48.12), (35.09, 48.02), (35.12, 47.92), (35.09, 47.84), (35.14, 47.76)]

SPECS = {
    "hormuz": dict(bbox=(55.75, 26.05, 56.85, 27.00), epsg="EPSG:32640", maritime=True,
                   land=["Iran", "Oman", "United Arab Emirates"],
                   base=("GULF OF OMAN", 56.65, 26.25), target=("PERSIAN GULF", 55.95, 26.85),
                   title="PROPOSAL A — Strait of Hormuz (maritime chokepoint)",
                   note="sea = transit corridor · coasts/islands = threat emplacement · "
                        "narrowest ~40 km (Iran coast vs Oman's Musandam)"),
    "ukraine": dict(bbox=(34.80, 47.75, 35.45, 48.55), epsg="EPSG:32636", maritime=False,
                    land=["Ukraine"], river=DNIPRO,
                    base=("DNIPRO", 35.045, 48.465), target=("ZAPORIZHZHIA", 35.145, 47.840),
                    title="PROPOSAL B — Dnipro corridor (land, river-structured)",
                    note="Dnipro river (approx.) = the natural chokepoint · crossings + urban "
                         "= where routing matters · exact banks/terrain from OSM at build"),
}


def render(name):
    spec = SPECS[name]
    W, S, E, N = spec["bbox"]; crs = spec["epsg"]
    ne = gpd.read_file(NE)
    corners = gpd.GeoSeries([Point(W, S), Point(E, N)], crs="EPSG:4326").to_crs(crs)
    x0, y0, x1, y1 = corners.iloc[0].x, corners.iloc[0].y, corners.iloc[1].x, corners.iloc[1].y
    Wkm, Hkm = (x1 - x0) / 1000, (y1 - y0) / 1000
    clip = box(x0, y0, x1, y1)

    def km_xy(lon, lat):
        p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
        return (p.x - x0) / 1000, (p.y - y0) / 1000

    fig, ax = plt.subplots(figsize=(11, 11 * Hkm / Wkm + 0.6))
    ax.set_facecolor(COL["sea"] if spec["maritime"] else COL["land"])
    land = ne[ne["name"].isin(spec["land"])].to_crs(crs)
    for geom in land.geometry:
        geom = geom.intersection(clip)
        for p in ([geom] if geom.geom_type == "Polygon" else getattr(geom, "geoms", [])):
            if p.is_empty:
                continue
            xs, ys = p.exterior.coords.xy
            ax.fill([(x - x0) / 1000 for x in xs], [(y - y0) / 1000 for y in ys],
                    color=COL["land"] if spec["maritime"] else "#ded4bd",
                    ec=COL["coast"], lw=1.1, zorder=2)
    if not spec["maritime"]:                              # land theatre: draw the river
        pts = [km_xy(lo, la) for lo, la in spec["river"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COL["river"], lw=6, zorder=3,
                solid_capstyle="round", alpha=.9)
        ax.annotate("Dnipro R.", pts[len(pts) // 2], (10, 0), textcoords="offset points",
                    color="#3f6f95", fontsize=10, fontstyle="italic", zorder=6)

    # graticule (lon/lat ticks)
    for gx in np.arange(0, Wkm, 20):
        ax.axvline(gx, color=COL["grid"], lw=.3, alpha=.35, zorder=4)
    for gy in np.arange(0, Hkm, 20):
        ax.axhline(gy, color=COL["grid"], lw=.3, alpha=.35, zorder=4)
    # endpoints + corridor axis
    bx, by = km_xy(*spec["base"][1:]); tx, ty = km_xy(*spec["target"][1:])
    ax.plot([bx, tx], [by, ty], "--", color=COL["friend"], lw=2, zorder=7, alpha=.85)
    for (x, y), lab, mk in (((bx, by), spec["base"][0], "s"), ((tx, ty), spec["target"][0], "*")):
        ax.scatter([x], [y], marker=mk, s=210 if mk == "*" else 100, zorder=8,
                   facecolor=COL["friend"], edgecolor="white", lw=1.6)
        ax.annotate(lab, (x, y), (8, 8), textcoords="offset points", zorder=8,
                    fontsize=10.5, fontweight="bold", color="#1c3a66")
    ax.plot([3, 23], [3, 3], color=COL["ink"], lw=3, zorder=9)
    ax.annotate("20 km", (13, 4), ha="center", fontsize=8.5, zorder=9, color=COL["ink"])
    ax.set_xlim(0, Wkm); ax.set_ylim(0, Hkm); ax.set_aspect("equal")
    ax.set_title(f"{spec['title']}\n{Wkm:.0f} x {Hkm:.0f} km   ·   {spec['note']}",
                 fontsize=11.5, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = f"assets/review_{name}.png"
    fig.savefig(out, dpi=150)
    print(f"[{name}] wrote {out}  ({Wkm:.0f} x {Hkm:.0f} km)")


if __name__ == "__main__":
    for nm in ("hormuz", "ukraine"):
        render(nm)
