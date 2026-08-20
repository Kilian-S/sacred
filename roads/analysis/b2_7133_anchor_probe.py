#!/usr/bin/env python3
"""B2 71-33 cell: anchor reproduction probe (ORACLE/EVAL-ONLY, free; no model calls).

Builds the 71-33 game exactly as analysis/b2_llm_benchmark.py does (same env call, same
stacked scoring path) and reproduces the banked one-shot (v*, stack) and dynamic
(opt / rotation / iid_eq at w=3, tau=0.15) anchors, before any live call fires.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     analysis/b2_7133_anchor_probe.py
Writes models/runs/b2_llm/b2_7133_anchors.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from analysis.critique_followup_probes import disjoint_subset
from analysis.gen40_dyn_sensitivity import cell
from scripts.train_b1lite1 import stacked_L
from scripts.train_generalist import CITY_PATHS
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(2)

N, K, KX, BAND = 3, 1, 8, (0.15, 0.95)
W, TAU = 3, 0.15

# banked expectations, 4 dp where banked so
EXPECT = {
    "R": 11, "m": 6,
    "v_eq": 0.1276,                     # exact LP
    "stack_uniform_disjoint": 0.1666,
    "stack_invvuln_disjoint": 0.1276,   # = v* (worst-edge and budget-max coincide at K=1)
    "stack_uniform_full": 0.2252,
    "stack_invvuln_full": 0.2502,
    "dyn_opt": 0.0313,                  # Karp, exact
    "dyn_rotation": 0.0387,
    "dyn_iid_eq": 0.0967,
}


def main():
    nodes_path, edges_path = CITY_PATHS["kaliningrad"]
    env = make_multiconvoy_env(od=("71", "33"), N=N, K=K, k_extra_routes=KX,
                               menu_select=True, edge_vuln_band=BAND,
                               interception_loss=10.0, seed=0,
                               nodes_path=nodes_path, edges_path=edges_path)
    game = env.game
    L = stacked_L(game, N)
    R = L.shape[0]
    dis = disjoint_subset([set(e) for e in game.route_edges])

    sol = solve_multiconvoy(game, N, "mission")

    def stack_expl(dist):
        dd = np.zeros(len(env.occupancies))
        for i in range(R):
            if dist[i] > 0:
                dd[env._occ_index[tuple(N if j == i else 0 for j in range(R))]] = dist[i]
        return float(env.exploitability_of_occupancy_dist(dd))

    uni_dis = np.zeros(R)
    uni_dis[np.asarray(dis)] = 1.0 / len(dis)
    inv_dis = np.zeros(R)
    q = np.array([L[r].max() for r in dis])
    inv_dis[np.asarray(dis)] = (1.0 / q) / (1.0 / q).sum()
    uni_full = np.ones(R) / R
    qf = np.array([L[r].max() for r in range(R)])
    inv_full = (1.0 / qf) / (1.0 / qf).sum()

    got = {
        "R": R, "m": len(dis), "disjoint_core": [int(r) for r in dis],
        "v_eq": float(sol.loss_mixed), "loss_det": float(sol.loss_det),
        "stack_uniform_disjoint": stack_expl(uni_dis),
        "stack_invvuln_disjoint": stack_expl(inv_dis),
        "stack_uniform_full": stack_expl(uni_full),
        "stack_invvuln_full": stack_expl(inv_full),
    }

    dyn = cell(env, K, KX, W, "71-33")
    got.update({"dyn_opt": dyn["opt"], "dyn_rotation": dyn["rotation"],
                "dyn_anti_core": dyn["anti_core"], "dyn_anti_full": dyn["anti_full"],
                "dyn_iid_eq": dyn["iid_eq"], "dyn_static_det": dyn["static_det"],
                "dyn_best_naive": dyn["best_naive"],
                "dyn_best_naive_name": dyn["best_naive_name"]})

    verdicts = {}
    for k, exp in EXPECT.items():
        v = got[k]
        ok = (v == exp) if isinstance(exp, int) else abs(v - exp) < 5e-4
        verdicts[k] = "PASS" if ok else f"FAIL (got {v:.4f} vs banked {exp})"
    out = {"got": got, "expect": EXPECT, "verdicts": verdicts,
           "all_pass": all(v == "PASS" for v in verdicts.values())}
    print(json.dumps(out, indent=2))
    path = pathlib.Path("models/runs/b2_llm/b2_7133_anchors.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"[written] {path}")


if __name__ == "__main__":
    main()
