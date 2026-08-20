#!/usr/bin/env python3
"""Computes the reference statistics for the four aerial theatres and writes the theatre atlas.

It loads the committed vector maps, measures them, and counts candidate emplacement sites under
terrain tables v1 and v2, with no training and no model calls. Area shares come from dense point
sampling through the same priority order the game's `classify` uses, water before sea before urban
before alpine before forest before field, else open, with an STRtree lookup so the big maps stay
affordable, and the sampler is verified against `hazard_sites` before anything is reported.

    PYTHONPATH=. python analysis/theatre_atlas.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from shapely.geometry import Point

from src.envs.aerial_theatre_vec import (PRIORITY, TERRAIN, hazard_sites, lateral_width,
                                         load_vec_theatre, terrain_v2)

MAPS = ["kgd_gvardeysk", "ukraine", "narva", "fulda"]
PATH = "data/maps/theatre_%s_vec.json"
SPACING_KM, STANDOFF_KM = 2.0, 4.0
TARGET_SAMPLES = 60_000

CHARACTER = {
    "kgd_gvardeysk": "Kaliningrad -> Gvardeysk along the Pregolya. The reference theatre: a short, "
                     "narrow corridor with the city as a line-of-sight wall at the mouth and open "
                     "farmland beyond. The reference theatre for the aerial acts.",
    "ukraine": "Dnipro -> Zaporizhzhia along the river. A long north-south corridor, farmland "
               "dominated, heavily built up at both ends: the longest run of the four.",
    "narva": "Kohtla-Jarve -> Kingisepp across the Narva river border. Forest dominated with a "
             "natural river pinch, and the Gulf of Finland closing the north flank.",
    "fulda": "Point Alpha -> Frankfurt, the Fulda Gap. The Cold War invasion axis, flanked by the "
             "forested Vogelsberg and Rhoen uplands. Much the largest and the most detailed.",
}


def n_verts(g) -> int:
    """Ring vertices of a Polygon or MultiPolygon (buffer(0) repairs can yield either)."""
    if hasattr(g, "geoms"):
        return sum(n_verts(p) for p in g.geoms)
    return len(g.exterior.coords) + sum(len(r.coords) for r in g.interiors)


def classify_fast(th, xy, polys, trees):
    p = Point(float(xy[0]), float(xy[1]))
    for cls in PRIORITY:
        tree = trees.get(cls)
        if tree is None:
            continue
        for i in tree.query(p):
            if polys[cls][int(i)].contains(p):
                return cls
    return "open"


def area_shares(th, polys, trees):
    step = float(np.sqrt(th.W * th.H / TARGET_SAMPLES))
    xs = np.arange(step / 2, th.W, step)
    ys = np.arange(step / 2, th.H, step)
    cnt: dict = {}
    for x in xs:
        for y in ys:
            k = classify_fast(th, (x, y), polys, trees)
            cnt[k] = cnt.get(k, 0) + 1
    tot = sum(cnt.values())
    return {k: 100.0 * v / tot for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}, step, tot


def site_counts(th, terrain, polys, trees):
    """Mirrors the `hazard_sites` loop with the fast classifier."""
    xs = np.arange(1.0, th.W, SPACING_KM)
    ys = np.arange(1.0, th.H, SPACING_KM)
    cnt: dict = {}
    for x in xs:
        for y in ys:
            xy = np.array([x, y])
            if (np.linalg.norm(xy - th.base) < STANDOFF_KM
                    or np.linalg.norm(xy - th.target) < STANDOFF_KM):
                continue
            k = classify_fast(th, xy, polys, trees)
            if not terrain[k]["emplace"]:
                continue
            cnt[k] = cnt.get(k, 0) + 1
    return cnt


def main():
    t2 = terrain_v2()
    rows, ref_lat = [], None
    for name in MAPS:
        t0 = time.time()
        d = json.load(open(PATH % name))
        th = load_vec_theatre(PATH % name)
        load_s = time.time() - t0
        polys, trees = th.polys, th._tree

        if name == MAPS[0]:                       # sampler verification against the real loop
            got = site_counts(th, TERRAIN, polys, trees)
            _, _, _, cls = hazard_sites(th, spacing_km=SPACING_KM, standoff_km=STANDOFF_KM)
            ref = {}
            for c in cls:
                ref[c] = ref.get(c, 0) + 1
            assert got == ref, f"atlas sampler disagrees with hazard_sites: {got} vs {ref}"

        lat = lateral_width(th)
        ref_lat = lat if ref_lat is None else ref_lat
        shares, step, nsamp = area_shares(th, polys, trees)
        rows.append(dict(
            name=name, d=d, th=th, load_s=load_s, lat=lat, step=step, nsamp=nsamp,
            shares=shares,
            counts={k: len(v) for k, v in sorted(polys.items(), key=lambda kv: -len(kv[1]))},
            verts={k: int(sum(n_verts(p) for p in v)) for k, v in polys.items()},
            corridor=float(np.linalg.norm(th.target - th.base)),
            v1=site_counts(th, TERRAIN, polys, trees),
            v2=site_counts(th, t2, polys, trees),
            scale=lat / ref_lat,
        ))
        print(f"{name}: load {load_s:.1f}s, {nsamp} samples at {step:.2f} km")

    out = ["# Theatre atlas: the four scored aerial theatres",
           "",
           "- **status: REFERENCE.** The aerial acts use these four theatres. Regenerate with "
           "`PYTHONPATH=. python analysis/theatre_atlas.py`; the maps themselves live under "
           "`data/maps/` and are gitignored, so this file is the committed record of what they "
           "contain. Oracle-only: no training, no model calls.",
           "- **Companion:** `experiments/gen39_concealment.md` (terrain table v2 and the "
           "concealment mechanic these statistics are being reviewed for).",
           "",
           "## At a glance",
           "",
           "| theatre | box (km) | corridor | lateral width | range scale | sites (v2) | concealed |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        th, d = r["th"], r["d"]
        hid = r["v2"].get("forest", 0) + r["v2"].get("urban", 0)
        tot = sum(r["v2"].values())
        r["hidden_share"] = 100.0 * hid / max(tot, 1)
        out.append("| %s | %.0f x %.0f | %s -> %s, %.1f km | %.1f km | %.2f | %s | **%.0f%%** |" % (
            r["name"], th.W, th.H,
            d["base"]["label"], d["target"]["label"], r["corridor"], r["lat"], r["scale"],
            f"{tot:,}", r["hidden_share"]))
    out += ["",
            "*Range scale* is the weapon-range multiplier the game applies so that the fraction of "
            "the corridor a team covers is comparable across maps, taken relative to Kaliningrad "
            "(lateral width %.1f km). A 2.5 km system on Kaliningrad and a %.1f km system on Fulda "
            "contest the same share of the width." % (rows[0]["lat"], 2.5 * rows[-1]["scale"]),
            ""]

    for r in rows:
        th, d = r["th"], r["d"]
        bb = d.get("bbox_lonlat")
        out += [f"## {r['name']}", "",
                CHARACTER[r["name"]], "",
                "| property | value |", "|---|---|",
                "| base | %s at (%.1f, %.1f) km |" % (d["base"]["label"], *d["base"]["xy_km"]),
                "| target | %s at (%.1f, %.1f) km |" % (d["target"]["label"], *d["target"]["xy_km"]),
                "| box | %.3f x %.3f km (%.0f km2) |" % (th.W, th.H, th.W * th.H),
                "| bounding box (lon/lat) | W %.3f, S %.3f, E %.3f, N %.3f |" % tuple(bb) if bb
                else "| bounding box | not recorded |",
                "| projection | %s |" % d.get("epsg", "not recorded (pre-2026-07-22 format)"),
                "| corridor length | %.1f km |" % r["corridor"],
                "| lateral width | %.1f km |" % r["lat"],
                "| range scale | %.2f (Kaliningrad = 1.00) |" % r["scale"],
                "| load time | %.1f s |" % r["load_s"],
                "",
                "**Terrain by area** (%d samples at %.2f km, priority order as the game classifies):"
                % (r["nsamp"], r["step"]), "",
                "| class | share of area | polygons | ring vertices |", "|---|---|---|---|"]
        for k, v in r["shares"].items():
            out.append("| %s | %.1f%% | %s | %s |" % (
                k, v, f"{r['counts'].get(k, 0):,}" if k != "open" else "-",
                f"{r['verts'].get(k, 0):,}" if k != "open" else "-"))
        v1, v2 = r["v1"], r["v2"]
        out += ["",
                "**Candidate emplacement sites** (%.1f km grid, %.1f km terminal standoff):"
                % (SPACING_KM, STANDOFF_KM), "",
                "| table | total | by class |", "|---|---|---|",
                "| v1 | %d | %s |" % (sum(v1.values()),
                                               ", ".join(f"{k} {n}" for k, n in sorted(v1.items()))),
                "| v2 (gen39) | %d | %s |" % (sum(v2.values()),
                                              ", ".join(f"{k} {n}" for k, n in sorted(v2.items()))),
                ""]

    grad = ", ".join("%s %.0f%%" % (r["name"].split("_")[0], r["hidden_share"])
                     for r in sorted(rows, key=lambda r: r["hidden_share"]))
    out += ["## Notes for the terrain review", "",
            "- **The four maps form a gradient in exactly the variable table v2 introduces.** The "
            "share of candidate sites that sit on CONCEALED ground (forest or urban, the classes "
            "that do not give themselves away when they engage) runs %s. That is not four "
            "repetitions of one theatre: it is a designed axis running from a corridor where "
            "hiding is barely available to one where most of the ground conceals, and it is the "
            "natural held-out structure for the concealment act." % grad,
            "- **Fulda's range scale is the outlier and needs a decision.** Scaling by lateral "
            "width to keep the coverage fraction comparable turns a 2.5 km system into a 14.5 km "
            "one there, which is a different weapon class in everything but name. Either accept "
            "the abstraction and say so, or cap the scale and accept that Fulda is a "
            "lower-coverage theatre. This must be settled before any cross-map claim.",
            "- **The exact interdiction matrix does not survive multi-team games on the big maps.** "
            "It is routes x C(sites, K). At K=1 that is fine everywhere (Fulda: 3,431 columns), "
            "but K=3 on Fulda is C(3431,3), about 6.7 billion columns. The gen33 scoring "
            "semantics, where a force induces a soft site prior rather than a hard K-subset, "
            "scales fine; the exact-LP path does not. Any multi-team screen on Narva or Fulda uses "
            "the soft-prior semantics or a coarser site grid, and says which.",
            "- All four maps are covered by the terrain table: every polygon class present is one "
            "the game models, so the loader drops nothing. (The four maritime theatres that were "
            "dropped from the line carry `land`, `island` and `coast` layers that the table does "
            "not model, which is why they were unusable for anything scored.)",
            "- Narva is the only one of the four with a `sea` layer, and it is a single polygon: "
            "the Gulf of Finland closing the north flank. It is non-emplaceable and does not block "
            "line of sight.",
            "- Alpine appears in the Narva and Fulda fetches as an empty or near-empty layer, so "
            "high-terrain walls are not in play on any of the four.",
            "- Under table v2 the urban class becomes emplaceable, which is where the extra sites "
            "in the v2 row come from; the forest and urban sites are the concealed ones that do "
            "not reveal themselves when they engage.",
            "- Fulda is much the largest and most detailed map and is correspondingly the most "
            "expensive to load and to build games on; budget for that before using it in a sweep.",
            ""]

    open("experiments/theatre_atlas.md", "w").write("\n".join(out) + "\n")
    print("wrote experiments/theatre_atlas.md")


if __name__ == "__main__":
    main()
