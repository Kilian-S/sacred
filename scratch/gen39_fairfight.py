#!/usr/bin/env python3
"""gen39: THE MATCHED-EFFORT LADDER (Kilian, 2026-07-27).

The step-3 control (gen32 doctrine + exact placement search) carries two generations of tuning
plus an optimiser; the LLM gets one zero-shot prompt and a rule placer. This measures what each
of those two advantages is actually worth, by crossing them:

    doctrine  x  placement
    TUNED     = gen32 q=(0.6 rep, 0.2 flee, 0.3 anti-rep), tau 0.10  [2 generations of hunts]
    UNTUNED   = the three first guesses a person writes with no measurement at all:
                  naive_repeat  q_rep=1.0            ("shoot where they just flew")
                  uniform       q=(1/3,1/3,1/3)      ("hedge everything")
                  hold          q_hold=1.0           ("sit on the best ground")
    ORACLE    = choose_force, the exact combination search
    RULE      = the step-2 placer (best-threat site of the stated class in the stated region)

Both controls STAY in the ladder: the matched-effort cell (untuned + rule) answers "does the LLM
beat a human's first guess", the tuned cell answers "does it beat the best we can build". Neither
replaces the other.

    PYTHONPATH=. python scratch/gen39_fairfight.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from src.envs.aerial_conceal import ConcealBase, ConcealDyn, choose_force, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

MAP, CR, RM, K, W, TAU = "narva", 0.85, 0.7, 3, 2, 0.10
PATH = "data/maps/theatre_%s_vec.json"
FIELDS = tuple(range(1000, 1008))
OUT = Path("models/runs/gen39_fairfight.json")

TUNED = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3, tau=TAU, w=W)
UNTUNED = {
    "naive_repeat": dict(q_rep=1.0, q_flee=0.0, q_ar=0.0, q_hold=0.0, tau=TAU, w=W),
    "uniform":      dict(q_rep=1 / 3, q_flee=1 / 3, q_ar=1 / 3, q_hold=0.0, tau=TAU, w=W),
    "hold":         dict(q_rep=0.0, q_flee=0.0, q_ar=0.0, q_hold=1.0, tau=TAU, w=W),
}
POSTURES = ("open", "hidden", "mixed")


def base_of():
    sc = lateral_width(load_vec_theatre(PATH % MAP)) / lateral_width(
        load_vec_theatre(PATH % "kgd_gvardeysk"))
    return ConcealBase(PATH % MAP, terrain=terrain_v2(hidden_leth=1.0, conceal_reach=CR),
                       range_scale=sc * RM, spacing_km=2.0 * sc, standoff_km=4.0 * sc,
                       n_sites=200)


def score(base, pp, sites, doctrines):
    g = ConcealDyn(base, pp, np.asarray(sites, int), w=W, tau=TAU, doctrines=doctrines)
    sup = g.blind_supports()
    blind = min(g.episodic(rule=lambda i, m, p, M=np.asarray(g._anti(d), float): M, T=40)
                for d in sup.values())
    obs = min([blind] + [g.episodic_rule(d, anti_repeat=a, softness=s, topm=t, T=40)
                         for d in sup.values() for a in (False, True)
                         for s, t in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3), (0.0, 5))])
    return float(g.episodic(T=40)), float(obs)


def rule_sites(base, pp, posture):
    """The step-2 placer on a first-guess posture spread across the three corridor thirds."""
    from scratch.gen39_compose import place
    terr = {"open": "open", "hidden": "forest", "mixed": "open"}[posture]
    ag = [{"emplacement_zone": {"terrain": (terr if not (posture == "mixed" and i) else "forest"),
                                "region": r},
           "doctrine": {"punish_pattern": 1.0, "anticipate_flight": 0.0, "hold_static": 0.0}}
          for i, r in enumerate(("near_base", "mid_corridor", "near_target_standoff"))]
    return place({"agents": ag}, base, pp)


def main():
    base = base_of()
    rows: dict[str, list] = {}
    for f in FIELDS:
        pp = base.lethality(resample_field(base.coords, f), hidden_leth=1.0)
        rng = np.random.default_rng(f)
        for posture in POSTURES:
            L_or, _, _ = choose_force(base, pp, posture, K, rng, w=W, tau=TAU,
                                      doctrine={k: v for k, v in TUNED.items()
                                                if k in ("q_rep", "q_flee", "q_ar")})
            L_rl = rule_sites(base, pp, posture)
            rows.setdefault("TUNED + oracle", []).append(score(base, pp, L_or, [dict(TUNED)] * K))
            rows.setdefault("TUNED + rule", []).append(score(base, pp, L_rl, [dict(TUNED)] * K))
            for name, doc in UNTUNED.items():
                rows.setdefault(f"UNTUNED:{name} + oracle", []).append(
                    score(base, pp, L_or, [dict(doc)] * K))
                rows.setdefault(f"UNTUNED:{name} + rule", []).append(
                    score(base, pp, L_rl, [dict(doc)] * K))
        print(f"  field {f} done", flush=True)

    print(f'\n{"force family":34s} {"n":>3s} {"IRREDUCIBLE (vs knowing)":>25s} {"vs searching rule":>18s}')
    summ = {}
    for k in sorted(rows):
        a = np.array(rows[k])
        summ[k] = dict(irr=float(np.median(a[:, 0])), obs=float(np.median(a[:, 1])), n=len(a))
        print(f'{k:34s} {len(a):3d} {np.median(a[:, 0]):25.5f} {np.median(a[:, 1]):18.4f}')
    best_unt = min((v["irr"] for k, v in summ.items() if k.startswith("UNTUNED") and "rule" in k),
                   default=0)
    best_unt_k = max((k for k in summ if k.startswith("UNTUNED") and "rule" in k),
                     key=lambda k: summ[k]["irr"])
    print(f'\nLLM reference rows (same scorer, same fields class): zero-shot 0.00115, '
          f'iterated 0.00173, curated 0.00427, best single force 0.0091')
    print(f'matched-effort cell = untuned doctrine + rule placement: '
          f'best of the three first guesses is {best_unt_k.split(":")[1]} at '
          f'{summ[best_unt_k]["irr"]:.5f} irreducible')
    print(f'tuning is worth {summ["TUNED + rule"]["irr"] / max(summ[best_unt_k]["irr"], 1e-9):.2f}x '
          f'(doctrine only) and {summ["TUNED + oracle"]["irr"] / max(summ[best_unt_k]["irr"], 1e-9):.2f}x '
          f'(doctrine + placement search)')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summ, "raw": rows}, indent=1))
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
