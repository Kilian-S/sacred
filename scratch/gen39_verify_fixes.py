#!/usr/bin/env python3
"""gen39: the hide-vs-open question with ALL THREE repairs in place, before any screen re-run.

  1. no terrain leak      the engagement concentration stays on the team's own ground
  2. fair sampling        candidate class shares match the theatre's composition (Kilian's quota
                          scheme), evenly spaced skeleton, snapped inside the polygons
  3. a real force picker  best COMBINATION of K positions, never worse than the old K-best-points

    PYTHONPATH=. python scratch/gen39_verify_fixes.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

PATH = "data/maps/theatre_%s_vec.json"
DOC = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3)
FIELDS = (5100, 5101, 5102)


def score(base, pp, L):
    g = ConcealDyn(base, pp, L, w=2, **DOC)
    sup = g.blind_supports()
    blind = min(g.episodic(rule=lambda i, m, p, M=np.asarray(g._anti(d), float): M, T=40)
                for d in sup.values())
    obs = min([blind] + [g.episodic_rule(d, anti_repeat=a, softness=s, topm=t, T=40)
                         for d in sup.values() for a in (False, True)
                         for s, t in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3), (0.0, 5))])
    return g.episodic(T=40), blind, obs


def force(base, pp, K, pool):
    """The best force we can find: the old K-best-points picker AND the combination search."""
    thr = base.threat_rank(pp)
    cands = [pool[np.argsort(-thr[pool])[:K]]] + base.best_laydown(pp, K, pool=pool, n_out=3)
    got = [score(base, pp, L) for L in cands]
    return max(got, key=lambda t: t[0])


def main():
    ref = lateral_width(load_vec_theatre(PATH % "kgd_gvardeysk"))
    for m in ("kgd_gvardeysk", "ukraine"):
        sc = lateral_width(load_vec_theatre(PATH % m)) / ref
        base = ConcealBase(PATH % m, terrain=terrain_v2(hidden_leth=1.0, conceal_reach=0.85),
                           range_scale=sc, spacing_km=2.0 * sc, standoff_km=4.0 * sc, n_sites=200)
        print(f"\n=== {m}  (H={base.H} candidates, class shares match the map) ===")
        print(f'{"K":>2s} {"force":8s} | {"vs perfect play":>15s} {"blind rule":>10s} '
              f'{"observing rule":>14s} | {"sight worth":>11s}')
        for K in (3, 6):
            rows = {}
            for kind, pool in (("open", np.where(~base.concealed)[0]),
                               ("hidden", np.where(base.concealed)[0]),
                               ("mixed", np.arange(base.H))):
                got = [force(base, base.lethality(resample_field(base.coords, s), hidden_leth=1.0),
                             K, pool) for s in FIELDS]
                o, b, ob = (float(np.median([g[i] for g in got])) for i in range(3))
                rows[kind] = (o, b, ob)
                print(f'{K:2d} {kind:8s} | {o:15.4f} {b:10.4f} {ob:14.4f} | {b / max(ob, 1e-9):10.2f}x')
            for kind in ("hidden", "mixed"):
                print(f'   {kind} vs open: {rows[kind][0] / max(rows["open"][0], 1e-9):.2f} against '
                      f'perfect play, {rows[kind][2] / max(rows["open"][2], 1e-9):.2f} against an '
                      f'observing defender')


if __name__ == "__main__":
    main()
