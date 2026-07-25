#!/usr/bin/env python3
"""gen39: one review figure per theatre (Kilian 2026-07-25).

Renders exactly the game the screen runs: real OSM terrain, the flight-path menu, every candidate
emplacement coloured by the ground it stands on, and the interception reach of each terrain class
drawn to scale as rings on the map. Oracle-only, no training, no model calls.

    PYTHONPATH=. python scratch/gen39_maps.py            # all four
    PYTHONPATH=. python scratch/gen39_maps.py --maps fulda

Writes assets/gen39_theatre_<name>.png.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from src.envs.aerial_conceal import ConcealBase
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

MAPS = ["kgd_gvardeysk", "ukraine", "narva", "fulda"]
PATH = "data/maps/theatre_%s_vec.json"

FILL = {"water": "#a8c8e0", "sea": "#a8c8e0", "urban": "#b9b3ad",
        "forest": "#9dbb8e", "field": "#e6dfba", "alpine": "#cfc7bd"}
SITE = {"open": "#c0392b", "field": "#d35400", "forest": "#1e6f3c", "urban": "#4a3f8f"}
OPEN_BG = "#f4f1e6"
LABEL = {"open": "open ground", "field": "farmland", "forest": "forest", "urban": "urban"}


def polys_of(th, cls, min_area):
    out = []
    for p in th.polys.get(cls, []):
        for g in (p.geoms if hasattr(p, "geoms") else [p]):
            if g.area >= min_area and g.exterior is not None:
                out.append(np.asarray(g.exterior.coords))
    return out


def draw(name, ax_all=None):
    d = json.load(open(PATH % name))
    ref = lateral_width(load_vec_theatre(PATH % "kgd_gvardeysk"))
    th = load_vec_theatre(PATH % name)
    sc = lateral_width(th) / ref
    base = ConcealBase(PATH % name, terrain=terrain_v2(), range_scale=sc,
                       spacing_km=2.0 * sc, standoff_km=4.0 * sc)
    terr = base.terrain

    # size by aspect but cap the long side, or the tall corridors come out a metre high
    asp = th.H / th.W
    fig_w, fig_h = ((15.0, max(4.5, 15.0 * asp * 0.92)) if asp <= 1.0
                    else (max(5.5, 15.0 / asp * 1.15), 15.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor(OPEN_BG)
    min_area = 2e-4 * th.W * th.H                     # drop specks that cannot be seen anyway

    for cls in ("field", "forest", "urban", "water", "sea", "alpine"):
        # urban is kept at full detail: it is the line-of-sight blocker, so its small blocks are
        # load-bearing in the game even when they are visually tiny
        rings = polys_of(th, cls, min_area * (0.02 if cls == "urban" else 1.0))
        if rings:
            ax.add_collection(PolyCollection(rings, facecolors=FILL[cls], edgecolors="none",
                                             zorder=1, rasterized=True))

    for i, r_ in enumerate(base.menu):                # the flight-path menu
        lane = i in base.lane_idx
        ax.plot(r_[:, 0], r_[:, 1], lw=0.7, alpha=0.55, zorder=3,
                color="#2b2b2b" if lane else "#7b2d8e",
                ls="-" if lane else (0, (4, 2)))

    for cls in ("open", "field", "forest", "urban"):  # candidate emplacements
        idx = [i for i, c in enumerate(base.cls) if c == cls]
        if idx:
            ax.scatter(base.coords[idx, 0], base.coords[idx, 1], s=9, c=SITE[cls],
                       edgecolors="white", linewidths=0.25, zorder=4)

    # interception reach, drawn to scale, on a real site of each class near the run
    for cls, frac in (("open", 0.30), ("field", 0.46), ("forest", 0.62), ("urban", 0.78)):
        idx = [i for i, c in enumerate(base.cls) if c == cls]
        if not idx:
            continue
        anchor = th.base + frac * (th.target - th.base)
        j = idx[int(np.argmin(np.linalg.norm(base.coords[idx] - anchor, axis=1)))]
        r = float(base.rr[j])
        ax.add_patch(Circle(base.coords[j], r, fill=False, ec=SITE[cls], lw=1.6,
                            ls=(0, (5, 3)), zorder=5))
        ax.annotate(f"{LABEL[cls]}  {r:.1f} km", base.coords[j] + np.array([0, r]),
                    color=SITE[cls], fontsize=8.5, weight="bold", ha="center", va="bottom",
                    zorder=6, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))

    for pt, lab, mk in ((th.base, d["base"]["label"], "^"), (th.target, d["target"]["label"], "s")):
        ax.plot(*pt, mk, ms=13, mfc="#0b6fa4", mec="white", mew=1.6, zorder=7)
        ax.annotate(lab, pt + np.array([0, -0.028 * th.H]), fontsize=10, weight="bold",
                    ha="center", va="top", zorder=7,
                    bbox=dict(fc="white", ec="#0b6fa4", alpha=0.9, pad=1.6))
    if "waypoint_frankfurt_km" in d:                  # the retargeted Fulda run
        ax.plot(*d["waypoint_frankfurt_km"], "*", ms=15, mfc="#f0c419", mec="#333", mew=0.8,
                zorder=7)
        ax.annotate("FRANKFURT (on the direct path)", np.array(d["waypoint_frankfurt_km"])
                    + np.array([0, 0.02 * th.H]), fontsize=9, ha="center", va="bottom", zorder=7,
                    bbox=dict(fc="white", ec="#f0c419", alpha=0.9, pad=1.4))
    ax.plot([th.base[0], th.target[0]], [th.base[1], th.target[1]], color="#0b6fa4", lw=1.0,
            alpha=0.5, ls=(0, (1, 3)), zorder=2)

    rows = [f"{LABEL[c]}: reach {terr[c]['r_km'] * sc:4.1f} km, lethality {terr[c]['p_max']:.2f}, "
            f"{'gives itself away' if terr[c]['reveal'] else 'stays hidden'}"
            f"{', blocks sight' if terr[c]['los'] else ''}" for c in ("open", "field", "forest",
                                                                     "urban")]
    ax.set_title(f"{name}   {d['base']['label']} to {d['target']['label']}   "
                 f"{np.linalg.norm(th.target - th.base):.0f} km   |   "
                 f"{th.W:.0f} x {th.H:.0f} km, range scale x{sc:.2f}, "
                 f"{base.H} emplacements ({int(base.concealed.sum())} concealed), "
                 f"{base.R} flight paths", fontsize=11, weight="bold", pad=9)
    fig.text(0.5, 0.005, "   |   ".join(rows), fontsize=8.6, va="bottom", ha="center",
             color="#222")

    handles = [Line2D([], [], color="#2b2b2b", lw=1.2, label="flight path (lane)"),
               Line2D([], [], color="#7b2d8e", lw=1.2, ls=(0, (4, 2)),
                      label="flight path (terrain-following)"),
               Line2D([], [], color="#0b6fa4", lw=1.2, ls=(0, (1, 3)), label="direct line")]
    handles += [Line2D([], [], marker="o", ls="", mfc=SITE[c], mec="white",
                       label=f"emplacement: {LABEL[c]}") for c in ("open", "field", "forest",
                                                                  "urban")]
    handles += [Line2D([], [], color="#555", lw=1.4, ls=(0, (5, 3)), label="interception reach")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9, ncols=2)

    ax.set_xlim(0, th.W); ax.set_ylim(0, th.H)
    ax.set_aspect("equal"); ax.set_xlabel("km"); ax.set_ylabel("km")
    fig.subplots_adjust(bottom=0.10)
    out = f"assets/gen39_theatre_{name}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.28, facecolor="white")
    plt.close(fig)
    print(f"[written] {out}  ({base.H} sites, {base.R} paths, scale x{sc:.2f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default=",".join(MAPS))
    for name in ap.parse_args().maps.split(","):
        draw(name)


if __name__ == "__main__":
    main()
