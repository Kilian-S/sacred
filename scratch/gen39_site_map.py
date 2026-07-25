#!/usr/bin/env python3
"""gen39: one PNG per theatre showing EVERY candidate emplacement, coloured by the ground it
stands on (Kilian 2026-07-25).

This is the audit picture for the quota sampler: candidate class shares are supposed to match the
theatre's terrain composition, and every point is supposed to stand INSIDE the terrain whose
weapon characteristics it carries. The caption prints both numbers so the two can be checked
against each other by eye.

    PYTHONPATH=. python scratch/gen39_site_map.py
    PYTHONPATH=. python scratch/gen39_site_map.py --maps kgd_gvardeysk --n-sites 200

Writes assets/gen39_sites_<name>.png.
"""
from __future__ import annotations

import argparse
import collections
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from src.envs.aerial_conceal import ConcealBase
from src.envs.aerial_theatre_vec import (_class_parts, lateral_width, load_vec_theatre, terrain_v2)

MAPS = ["kgd_gvardeysk", "ukraine", "narva", "fulda"]
PATH = "data/maps/theatre_%s_vec.json"

FILL = {"water": "#a8c8e0", "sea": "#a8c8e0", "urban": "#c9c3bd",
        "forest": "#a9c79a", "field": "#ece5c4", "alpine": "#d6cec4"}
SITE = {"open": "#c0392b", "field": "#e08a1e", "forest": "#1e7a3c", "urban": "#5b3fa0"}
LABEL = {"open": "open ground", "field": "farmland", "forest": "forest", "urban": "urban"}
OPEN_BG = "#f7f4ea"
EMPL = ("open", "field", "forest", "urban")


def true_shares(th):
    a = {k: float(sum(g.area for g in _class_parts(th, k))) for k in EMPL if k != "open"}
    a["open"] = max(th.W * th.H - sum(float(sum(g.area for g in _class_parts(th, k)))
                                      for k in th.polys), 0.0)
    tot = sum(a.values()) or 1.0
    return {k: 100.0 * v / tot for k, v in a.items()}


def draw(name, n_sites):
    d = json.load(open(PATH % name))
    th = load_vec_theatre(PATH % name)
    ref = lateral_width(load_vec_theatre(PATH % "kgd_gvardeysk"))
    sc = lateral_width(th) / ref
    terr = terrain_v2(hidden_leth=1.0, conceal_reach=0.85)
    base = ConcealBase(PATH % name, terrain=terr, range_scale=sc, spacing_km=2.0 * sc,
                       standoff_km=4.0 * sc, n_sites=n_sites)

    asp = th.H / th.W
    fig_w, fig_h = ((15.0, max(5.0, 15.0 * asp * 0.92)) if asp <= 1.0
                    else (max(6.0, 15.0 / asp * 1.15), 15.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor(OPEN_BG)
    min_area = 2e-4 * th.W * th.H
    for cls in ("field", "forest", "urban", "water", "sea", "alpine"):
        rings = [np.asarray(g.exterior.coords) for g in _class_parts(th, cls)
                 if g.area >= min_area * (0.02 if cls in ("urban", "forest") else 1.0)
                 and g.exterior is not None]
        if rings:
            ax.add_collection(PolyCollection(rings, facecolors=FILL[cls], edgecolors="none",
                                             zorder=1, rasterized=True))

    for i, r_ in enumerate(base.menu):
        ax.plot(r_[:, 0], r_[:, 1], lw=0.5, alpha=0.30, zorder=2,
                color="#2b2b2b" if i in base.lane_idx else "#7b2d8e")

    cnt = collections.Counter(base.cls)
    for cls in EMPL:                                   # candidates, coloured by their own ground
        idx = [i for i, c in enumerate(base.cls) if c == cls]
        if idx:
            ax.scatter(base.coords[idx, 0], base.coords[idx, 1], s=26, c=SITE[cls],
                       edgecolors="white", linewidths=0.5, zorder=5)

    for cls, frac in (("open", 0.32), ("field", 0.48), ("forest", 0.64), ("urban", 0.80)):
        idx = [i for i, c in enumerate(base.cls) if c == cls]
        if not idx:
            continue
        anchor = th.base + frac * (th.target - th.base)
        j = idx[int(np.argmin(np.linalg.norm(base.coords[idx] - anchor, axis=1)))]
        r = float(base.rr[j])
        ax.add_patch(Circle(base.coords[j], r, fill=False, ec=SITE[cls], lw=1.5, ls=(0, (5, 3)),
                            zorder=6))
        ax.annotate(f"{r:.1f} km", base.coords[j] + np.array([0, r]), color=SITE[cls], fontsize=8,
                    weight="bold", ha="center", va="bottom", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))

    for pt, lab, mk in ((th.base, d["base"]["label"], "^"), (th.target, d["target"]["label"], "s")):
        ax.plot(*pt, mk, ms=12, mfc="#0b6fa4", mec="white", mew=1.5, zorder=8)
        ax.annotate(lab, pt + np.array([0, -0.03 * th.H]), fontsize=9.5, weight="bold",
                    ha="center", va="top", zorder=8,
                    bbox=dict(fc="white", ec="#0b6fa4", alpha=0.9, pad=1.4))

    sh = true_shares(th)
    rows = [f"{LABEL[c]}: {cnt.get(c, 0)} pts = {100 * cnt.get(c, 0) / base.H:.1f}% "
            f"(ground {sh.get(c, 0):.1f}%), reach {terr[c]['r_km'] * sc:.1f} km, "
            f"lethality {terr[c]['p_max']:.2f}, "
            f"{'HIDDEN' if not terr[c]['reveal'] else 'reveals'}" for c in EMPL]
    ax.set_title(f"{name}   candidate emplacements, coloured by the ground they stand on   "
                 f"({base.H} points, {th.W:.0f} x {th.H:.0f} km, range scale x{sc:.2f})",
                 fontsize=11.5, weight="bold", pad=9)
    fig.text(0.5, 0.005, "     ".join(rows), fontsize=8.4, va="bottom", ha="center", color="#222")
    handles = [Line2D([], [], marker="o", ls="", mfc=SITE[c], mec="white",
                      label=f"{LABEL[c]}  ({cnt.get(c, 0)})") for c in EMPL]
    handles += [Line2D([], [], color="#555", lw=1.3, ls=(0, (5, 3)), label="interception reach")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.92, ncols=2)
    ax.set_xlim(0, th.W); ax.set_ylim(0, th.H)
    ax.set_aspect("equal"); ax.set_xlabel("km"); ax.set_ylabel("km")
    fig.subplots_adjust(bottom=0.10)
    out = f"assets/gen39_sites_{name}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.28, facecolor="white")
    plt.close(fig)
    print(f"[written] {out}  " + "  ".join(f"{c}:{cnt.get(c, 0)}({sh.get(c, 0):.0f}% ground)"
                                           for c in EMPL), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default=",".join(MAPS))
    ap.add_argument("--n-sites", type=int, default=200)
    a = ap.parse_args()
    for name in a.maps.split(","):
        draw(name, a.n_sites)


if __name__ == "__main__":
    main()
