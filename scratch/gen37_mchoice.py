#!/usr/bin/env python3
"""gen37 M-sweep (oracle-only, no training, no LLM): does ANY shortlist size M isolate
reasoning quality?

The M=50 mechanism rows showed random spans the joint space so well that its restricted-game
ceiling (LP-over-shortlist) already reaches ~1.07x eq on held-out - so at M=50 curation
QUALITY barely matters and the comparative clause is near-dead at the ceiling before training.
This probe sweeps M downward for the two arms whose orderings are meaningful (random draw order;
heuristic rank) to find M* where random's ceiling DEGRADES - the regime where an LLM call would
actually be tested. If random's ceiling stays low for all M, reasoning-curation is falsified at
the ceiling (cheap negative, no training). If it degrades sharply at small M, that M* is where
the real gen37 experiment lives.

Run (gen29 worktree): OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  /Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python scratch/gen37_mchoice.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from scratch.gen37_shortlist import edge_id_map, heuristic_shortlist, lp_over_shortlist
from scripts.train_multiod_generalist import Inst
from src.baselines.multiconvoy_oracle import _row_minimiser

MS = [6, 10, 15, 20, 30, 50]


def random_ranked(env, seed, cap):
    rng = np.random.default_rng(seed)
    R = [len(rs) for rs in env.route_sets]
    allt = list(itertools.product(range(R[0]), range(R[1]), range(R[2])))
    idx = rng.permutation(len(allt))[:cap]
    return [list(allt[i]) for i in idx]


def heuristic_ranked(env, edges, cap):
    return heuristic_shortlist.__wrapped__(env, edges, cap) if hasattr(
        heuristic_shortlist, "__wrapped__") else _heur(env, edges, cap)


def _heur(env, edges, cap):
    R = [len(rs) for rs in env.route_sets]
    wv = [env.worst_vuln[f] for f in range(env.F)]
    scored = []
    for a, b, c in itertools.product(range(R[0]), range(R[1]), range(R[2])):
        ea, eb, ec = set(edges[0][a]), set(edges[1][b]), set(edges[2][c])
        shared = max(len(ea & eb), len(ea & ec), len(eb & ec))
        scored.append((shared, wv[0][a] + wv[1][b] + wv[2][c], [a, b, c]))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [t for _, _, t in scored[:cap]]


def main():
    sc = json.load(open("models/runs/gen29_screen.json"))
    held = [Inst(c) for c in sc["held_out"]]
    per_M = {M: {"random": [], "heuristic": []} for M in MS}
    for it in held:
        env = it.env
        v_eq, _ = _row_minimiser(env.obj_matrix)
        edges = edge_id_map(env)
        rand_full = random_ranked(env, 9000, 50)
        heur_full = _heur(env, edges, 50)
        for M in MS:
            per_M[M]["random"].append(lp_over_shortlist(env, rand_full[:M]) / v_eq)
            per_M[M]["heuristic"].append(lp_over_shortlist(env, heur_full[:M]) / v_eq)
    print("=== held-out pooled LP-over-shortlist / eq  (lower = shortlist retains more value) ===")
    print(f"{'M':>4} {'random':>10} {'heuristic':>10}")
    for M in MS:
        r = float(np.mean(per_M[M]["random"]))
        h = float(np.mean(per_M[M]["heuristic"]))
        print(f"{M:>4} {r:>10.3f} {h:>10.3f}")
    print("\nreference: llm@50 held ceiling was 1.213; llm containment>0 on 2/6 cells.")
    json.dump({str(M): {k: [float(x) for x in v] for k, v in per_M[M].items()} for M in MS},
              open("models/runs/gen37_reasoning_curation/mchoice.json", "w"), indent=1)
    print("wrote models/runs/gen37_reasoning_curation/mchoice.json")


if __name__ == "__main__":
    main()
