#!/usr/bin/env python3
"""gen39 Phase 1a: HOW MUCH OF THE HEURISTIC ARM'S MARGIN IS PLACEMENT, NOT COMPOSITION?

The step-3 arms did not face equally SITED enemies: the heuristic arm's laydowns came from
`choose_force` (the exact combination optimiser), the llm and random arms' from the step-2 rule
placer. This scores, on ONE common yardstick, the enemies each arm actually trained against, and
adds the two counterfactuals that isolate the two variables:

  heur_oracle    gen32 doctrine + ORACLE placement      (what the heuristic arm faced)
  heur_placer    gen32 doctrine + RULE placement        (composition held, placement removed)
  llm_placer     LLM doctrine   + RULE placement        (what the llm arm faced)
  llm_oracle     LLM doctrine   + ORACLE placement      (posture-restricted; the Phase-2 design)
  rnd_placer     random doctrine + RULE placement       (what the random arm faced)

Reported per force: damage against the best OBSERVING rule (the deployable defender) and against
the omniscient optimum, median over the training fields. A stronger opponent = a harder
curriculum. Oracle-only, no training, free under the standing rule.

    PYTHONPATH=. python scratch/gen39_phase1_confound.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from src.envs.aerial_conceal import ConcealBase, ConcealDyn, choose_force, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

MAP, CR, RM, K, W, TAU = "narva", 0.85, 0.7, 3, 2, 0.10
PATH = "data/maps/theatre_%s_vec.json"
DOC32 = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3)
FIELDS = tuple(range(1000, 1008))            # a subset of the training fields (median over 8)
FORCES = "models/runs/gen39_compose/forces_llm.json"
OUT = Path("models/runs/gen39_phase1_confound.json")


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


def posture_pool(base, force):
    """Sites consistent with the force's stated terrain postures (the Phase-2 oracle placer:
    the model still chooses the ground, the optimiser only chooses where within it)."""
    want = {a["emplacement_zone"]["terrain"] for a in force["agents"]}
    keep = {c for c in want}
    if "open" in want:
        keep |= {"field"}
    idx = [i for i, c in enumerate(base.cls) if c in keep]
    return np.array(idx if len(idx) >= K else range(base.H), dtype=int)


def main():
    from scratch.gen39_compose import doctrines_of, place, random_force
    base = base_of()
    llm = [r["force"] for r in json.load(open(FORCES))
           if r.get("force") and not r.get("errors") and len(r["force"]["agents"]) == K]
    print(f"[phase1a] {MAP} K={K}; {len(llm)} llm forces; {len(FIELDS)} fields\n")
    rows: dict[str, list] = {k: [] for k in
                             ("heur_oracle", "heur_placer", "llm_placer", "llm_oracle",
                              "rnd_placer")}
    for f in FIELDS:
        pp = base.lethality(resample_field(base.coords, f), hidden_leth=1.0)
        rng = np.random.default_rng(f)
        # heuristic arm, as trained against: oracle placement, gen32 doctrine
        for kind in ("open", "hidden", "mixed"):
            L, g, _ = choose_force(base, pp, kind, K, rng, w=W, tau=TAU, doctrine=DOC32)
            rows["heur_oracle"].append(score(base, pp, L, None))
        # same doctrine, RULE placement (the placement variable, isolated)
        for kind in ("open", "hidden", "mixed"):
            fake = {"agents": [{"emplacement_zone": {
                "terrain": {"open": "open", "hidden": "forest", "mixed": "open"}[kind],
                "region": r}, "doctrine": {"punish_pattern": 0.6, "anticipate_flight": 0.2,
                                           "hold_static": 0.0}}
                for r in ("near_base", "mid_corridor", "near_target_standoff")]}
            rows["heur_placer"].append(score(base, pp, place(fake, base, pp), [dict(DOC32)] * K))
        # llm arm, as trained against: rule placement, llm doctrine
        for fo in llm[:6]:
            rows["llm_placer"].append(score(base, pp, place(fo, base, pp), doctrines_of(fo)))
        # llm doctrine + ORACLE placement inside its stated posture (the Phase-2 design)
        for fo in llm[:6]:
            L, _, _ = choose_force(base, pp, "open", K, rng, w=W, tau=TAU, doctrine=DOC32)
            pool = posture_pool(base, fo)
            best, bv = None, -1.0
            for cand in [base.best_laydown(pp, K, pool=pool)]:
                v = score(base, pp, cand, doctrines_of(fo))
                if v[0] > bv:
                    best, bv = v, v[0]
            rows["llm_oracle"].append(best)
        # random arm
        for j in range(3):
            fo = random_force(np.random.default_rng(7000 + 10 * f + j))
            rows["rnd_placer"].append(score(base, pp, place(fo, base, pp), doctrines_of(fo)))
        print(f"  field {f} done", flush=True)

    print(f'\n{"opponent family":14s} {"n":>3s} {"vs perfect play":>15s} {"vs observing rule":>17s}')
    summ = {}
    for k, v in rows.items():
        a = np.array(v)
        summ[k] = dict(n=len(v), opt=float(np.median(a[:, 0])), obs=float(np.median(a[:, 1])))
        print(f'{k:14s} {len(v):3d} {np.median(a[:, 0]):15.4f} {np.median(a[:, 1]):17.4f}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summ, "raw": {k: v for k, v in rows.items()}}, indent=1))
    ho, hp = summ["heur_oracle"]["obs"], summ["heur_placer"]["obs"]
    lp, lo = summ["llm_placer"]["obs"], summ["llm_oracle"]["obs"]
    print(f"\nPLACEMENT effect, doctrine held fixed (gen32): oracle {ho:.4f} vs placer {hp:.4f} "
          f"= {ho / max(hp, 1e-9):.2f}x stronger opponent")
    print(f"COMPOSITION effect, placement held fixed (rule): heuristic {hp:.4f} vs llm {lp:.4f} "
          f"= {lp / max(hp, 1e-9):.2f}x")
    print(f"LLM under oracle placement (the Phase-2 curriculum): {lo:.4f} "
          f"({lo / max(ho, 1e-9):.2f}x the heuristic arm's opponents)")
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
