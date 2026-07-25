#!/usr/bin/env python3
"""gen39: does the concentration smear leak across terrain classes, and does the hide-vs-open
result survive removing the leak? (Kilian's catch, 2026-07-25.)

The engagement concentration was spread over EVERY nearby candidate site regardless of ground, so
a team nominally in woodland drew most of its reach and lethality from neighbouring open ground
while keeping woodland's invisibility (reveal is decided by the team's OWN site). That dilutes the
price of concealment. `same_class=True` masks the concentration to the team's own ground.

    PYTHONPATH=. python scratch/gen39_leak_probe.py
"""
from __future__ import annotations

import collections
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

PATH = "data/maps/theatre_%s_vec.json"
FIELDS = (5100, 5101, 5102)
DOC = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3)


def leak(base, pp, sc):
    """Share of each class's effect delivered from its OWN ground, leaked and masked."""
    out = {}
    thr = base.threat_rank(pp)
    for c in ("open", "field", "forest", "urban"):
        pool = np.array([i for i, x in enumerate(base.cls) if x == c])
        if not len(pool):
            continue
        L = pool[np.argsort(-thr[pool])[:1]]
        row = {}
        for samec in (False, True):
            g = ConcealDyn(base, pp, L, w=2, same_class=samec, **DOC)
            by = collections.defaultdict(float)
            for i, x in enumerate(base.cls):
                by[x] += g.prior_j[0][i]
            row[samec] = by[c]
        out[c] = row
    return out


def cell(base, seed, hl, K, samec):
    pp = base.lethality(resample_field(base.coords, seed), hidden_leth=hl)
    thr = base.threat_rank(pp)
    res = {}
    for kind, pool in (("open", np.where(~base.concealed)[0]),
                       ("hidden", np.where(base.concealed)[0])):
        L = pool[np.argsort(-thr[pool])[:K]]
        g = ConcealDyn(base, pp, L, w=2, same_class=samec, **DOC)
        d = g.blind_supports()["uniform_lanes"]
        blind = min(g.episodic(rule=lambda i, m, p, M=np.asarray(g._anti(dd), float): M, T=40)
                    for dd in g.blind_supports().values())
        obs = min([blind] + [g.episodic_rule(dd, anti_repeat=a, softness=s, topm=t, T=40)
                             for dd in g.blind_supports().values() for a in (False, True)
                             for s, t in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3),
                                          (0.0, 5))])
        res[kind] = (g.episodic(T=40), blind, obs)
    return res


def main():
    ref = lateral_width(load_vec_theatre(PATH % "kgd_gvardeysk"))
    for m in ("kgd_gvardeysk", "ukraine"):
        sc = lateral_width(load_vec_theatre(PATH % m)) / ref
        base = ConcealBase(PATH % m, terrain=terrain_v2(hidden_leth=0.4, conceal_reach=0.85),
                           range_scale=sc, spacing_km=2.0 * sc, standoff_km=4.0 * sc)
        pp0 = base.lethality(resample_field(base.coords, FIELDS[0]), hidden_leth=0.4)
        print(f"\n=== {m} : where a team's effect comes from ===")
        for c, row in leak(base, pp0, sc).items():
            print(f"  {c:7s} own-ground share: leaked {100 * row[False]:3.0f}%  ->  "
                  f"masked {100 * row[True]:3.0f}%")
        print(f"--- hide-vs-open at the pinned point (K=3, hl 0.4, reach 0.85), {len(FIELDS)} fields ---")
        print(f'  {"":10s} {"omniscient":>26s} {"observing":>26s}')
        print(f'  {"":10s} {"open":>8s} {"hidden":>8s} {"ratio":>8s} {"open":>8s} {"hidden":>8s} {"ratio":>8s}')
        for samec in (False, True):
            got = [cell(base, s, 0.4, 3, samec) for s in FIELDS]
            oo = np.median([g["open"][0] for g in got]); ho = np.median([g["hidden"][0] for g in got])
            ob = np.median([g["open"][2] for g in got]); hb = np.median([g["hidden"][2] for g in got])
            tag = "masked" if samec else "leaked"
            print(f'  {tag:10s} {oo:8.4f} {ho:8.4f} {ho / max(oo, 1e-9):8.2f} '
                  f'{ob:8.4f} {hb:8.4f} {hb / max(ob, 1e-9):8.2f}')


if __name__ == "__main__":
    main()
