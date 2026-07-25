#!/usr/bin/env python3
"""gen39: WHAT IS CONCEALMENT ACTUALLY COSTING, and what would make it worth buying?

The verification showed a concealed force reaches only 0.14-0.43 of an open force against a
defender that must observe, and that a free force search never chooses cover. That is a verdict,
not a diagnosis. This decomposes the verdict into the three things cover charges you:

  REACH        forest engages at `conceal_reach` x the open radius
  LETHALITY    forest kills at p_max 0.55 against open's 0.90 (before the hidden_leth knob)
  OPTIONS      cover is a small share of the ground, so a concealed force chooses from ~34
               positions on kgd where an open force chooses from ~81 (same points per km2, which
               is fair, but not the same amount of choice)

Part 1 pays each charge back one at a time and reports what it buys. Part 2 sweeps reach and
lethality to find the BREAK-EVEN contour: the weapons a concealed team would need for hiding to be
worth it. Both are scored against a defender that must observe (the yardstick that can see the
information channel at all), with the exact optimum reported alongside.

    PYTHONPATH=. python scratch/gen39_conceal_cost.py
    PYTHONPATH=. python scratch/gen39_conceal_cost.py --maps kgd_gvardeysk --teams 3
"""
from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

PATH = "data/maps/theatre_%s_vec.json"
DOC = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3)
FIELDS = (5100, 5101, 5102)
N_SITES = 200
BASE_REACH, BASE_LETH = 0.85, 0.55          # the concealed team's table values, as fractions/levels
OPEN_REACH, OPEN_LETH = 1.00, 0.90


def build(name, reach, sc):
    """One theatre per (map, reach), built at the PINNED lethality: the lethality axis is applied
    at SCORE time via the `hidden_leth` knob, so the route menu stays frozen across it (the
    screen's same-game convention; rebuilding the table per lethality also moved the menu, a
    confound repaired 2026-07-25). Terminal standoff scales with the map, matching every other
    gen39 artefact (the earlier unscaled 4 km made the ukraine columns a different game)."""
    t = terrain_v2(hidden_leth=1.0, conceal_reach=reach)
    return ConcealBase(PATH % name, terrain=t, range_scale=sc, standoff_km=4.0 * sc,
                       n_sites=N_SITES)


def score(base, pp, L):
    g = ConcealDyn(base, pp, L, w=2, **DOC)
    sup = g.blind_supports()
    blind = min(g.episodic(rule=lambda i, m, p, M=np.asarray(g._anti(d), float): M, T=40)
                for d in sup.values())
    obs = min([blind] + [g.episodic_rule(d, anti_repeat=a, softness=s, topm=t, T=40)
                         for d in sup.values() for a in (False, True)
                         for s, t in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3), (0.0, 5))])
    return g.episodic(T=40), obs


def best_force(base, pp, K, pool):
    """Selected by the OBSERVING-defender score (t[1]): the matchup every reported share is
    measured in. The earlier selection by perfect-play damage picked forces for the wrong
    defender (repaired 2026-07-25; both columns still come from the ONE chosen force)."""
    thr = base.threat_rank(pp)
    if len(pool) < K:
        return None
    cands = [pool[np.argsort(-thr[pool])[:K]]] + base.best_laydown(pp, K, pool=pool, n_out=3)
    return max((score(base, pp, L) for L in cands), key=lambda t: t[1])


def run(base, K, kind, leth=BASE_LETH, sub=0, seed_pool=0):
    """Median over fields. `leth` = concealed lethality, applied via the score-time knob (menu
    frozen). `sub` restricts the pool to a random subset (the OPTIONS ablation)."""
    got = []
    for s in FIELDS:
        pp = base.lethality(resample_field(base.coords, s), hidden_leth=leth / BASE_LETH)
        pool = np.where(base.concealed if kind == "hidden" else ~base.concealed)[0]
        if sub and len(pool) > sub:
            pool = np.random.default_rng(seed_pool + s).choice(pool, size=sub, replace=False)
        r = best_force(base, pp, K, pool)
        if r:
            got.append(r)
    if not got:
        return None
    return float(np.median([g[0] for g in got])), float(np.median([g[1] for g in got]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="kgd_gvardeysk,ukraine")
    ap.add_argument("--teams", default="3,6")
    a = ap.parse_args()
    ref = lateral_width(load_vec_theatre(PATH % "kgd_gvardeysk"))

    for name in a.maps.split(","):
        sc = lateral_width(load_vec_theatre(PATH % name)) / ref
        bases = {}

        def get(reach):
            if reach not in bases:
                bases[reach] = build(name, reach, sc)
            return bases[reach]

        b0 = get(BASE_REACH)
        n_hid = int(b0.concealed.sum())
        n_open = int((~b0.concealed).sum())
        print(f"\n{'=' * 92}\n{name}: {n_open} open/farmland positions, {n_hid} concealed "
              f"(same points per km2)\n{'=' * 92}")
        for K in [int(x) for x in a.teams.split(",")]:
            ref_open = run(b0, K, "open")
            print(f"\n--- {K} teams --- open force = {ref_open[0]:.4f} vs perfect play, "
                  f"{ref_open[1]:.4f} vs an observing defender")
            print(f'{"concealed force pays back":42s} {"vs perfect":>11s} {"vs observing":>13s} '
                  f'{"% of open":>10s}')
            rows = [("nothing (the table as pinned)", BASE_REACH, BASE_LETH),
                    ("+ open REACH", OPEN_REACH, BASE_LETH),
                    ("+ open LETHALITY", BASE_REACH, OPEN_LETH),
                    ("+ open reach AND lethality", OPEN_REACH, OPEN_LETH)]
            for lab, rch, lth in rows:
                r = run(get(rch), K, "hidden", leth=lth)
                if r:
                    print(f'{lab:42s} {r[0]:11.4f} {r[1]:13.4f} {100 * r[1] / ref_open[1]:9.0f}%')
            r = run(b0, K, "open", sub=n_hid)          # the OPTIONS charge, isolated
            print(f'{"[control] OPEN force, only " + str(n_hid) + " positions":42s} '
                  f'{r[0]:11.4f} {r[1]:13.4f} {100 * r[1] / ref_open[1]:9.0f}%')

            print(f'\n  break-even sweep (% of the open force, vs an observing defender):')
            reaches = (0.85, 1.0, 1.2, 1.5)
            leths = (0.55, 0.70, 0.90)
            print("   reach\\lethality " + "".join(f"{x:>10.2f}" for x in leths))
            for rch in reaches:
                cells = []
                for lth in leths:
                    r = run(get(rch), K, "hidden", leth=lth)
                    cells.append(f"{100 * r[1] / ref_open[1]:9.0f}%" if r else "        -")
                print(f"   {rch:15.2f}" + "".join(cells))


if __name__ == "__main__":
    main()
