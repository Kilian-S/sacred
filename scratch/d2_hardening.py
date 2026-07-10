"""D2: defender-side hardening = the TACTICAL tier of the holistic supply-chain stack (ORACLE-ONLY).

Before the operational game, the defender spends a budget B to reduce EDGE vulnerabilities (escorts /
route clearance): each unit spent on edge e multiplies its interception prob by (1 - eta). Optimise
the allocation against the resulting equilibrium mission-failure (greedy on the submodular-ish
marginal, with a full-resolve check), then read how hardening changes the equilibrium AND where
randomisation pays. Completes the three tiers on one game: harden (strategic/tactical investment,
D2) -> place bases + size the fleet (tactical, D1) -> randomised routing (operational, SACRED).

Run: PYTHONPATH=. .venv/bin/python scratch/d2_hardening.py
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route, length_band_vulnerability,
    survival_intercept_fn)
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS
from src.utils.graph_utils import load_osm_graph_and_demands

OD, N, K, BAND, KX = ("35", "159"), 3, 1, (0.15, 0.95), 8
ETA = 0.5          # each hardening unit halves an edge's interception prob
BUDGET = 4         # number of hardening units to allocate

nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
G = nx.Graph()
for u, v, d in edges:
    G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
ALL = list(G.edges())

routes = build_route_set(G, OD[0], OD[1], KX, "w")
cand = sorted(set().union(*(edges_of_route(r) for r in routes)), key=repr)
base_vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=ALL)


def solve_with(vuln):
    game = build_interdiction_game(G, OD[0], OD[1], K, k_extra=KX, weight="w",
                                   intercept_fn=survival_intercept_fn(vuln))
    sol = solve_multiconvoy(game, N, "mission")
    # equilibrium defender mass on each ROUTE (via stacked occupancies) -> "where randomisation pays"
    R = game.n_routes
    lead = np.zeros(R)
    for i, occ in enumerate(sol.occupancies):
        nz = [r for r, c in enumerate(occ) if c > 0]
        if len(nz) == 1 and occ[nz[0]] == N:
            lead[nz[0]] += sol.defender_strategy[i]
    return sol.loss_mixed, sol.loss_det, lead, game


base_eq, base_det, base_lead, base_game = solve_with(base_vuln)
print(f"=== D2 hardening ({OD[0]}-{OD[1]}, N={N}, K={K}, eta={ETA}, budget={BUDGET}) ===")
print(f"  UNHARDENED: equilibrium {base_eq:.3f}  loss_det {base_det:.3f}")

# greedy hardening: each unit -> the edge whose hardening most reduces the equilibrium
vuln = dict(base_vuln)
alloc = {}
history = [base_eq]
for step in range(BUDGET):
    best_e, best_eq = None, base_eq if step == 0 else history[-1]
    for e in cand:
        trial = dict(vuln); trial[e] = trial[e] * (1.0 - ETA)
        eq, _, _, _ = solve_with(trial)
        if eq < best_eq - 1e-9:
            best_eq, best_e = eq, e
    if best_e is None:
        break
    vuln[best_e] = vuln[best_e] * (1.0 - ETA)
    alloc[repr(tuple(best_e))] = alloc.get(repr(tuple(best_e)), 0) + 1
    history.append(best_eq)
    print(f"  +unit {step+1}: harden {tuple(best_e)} -> equilibrium {best_eq:.3f}")

hard_eq, hard_det, hard_lead, _ = solve_with(vuln)
# random-allocation baseline (same budget, uniformly at random over candidate edges)
rng = np.random.default_rng(0)
rand_eqs = []
for _ in range(20):
    tv = dict(base_vuln)
    for e in rng.choice(len(cand), BUDGET, replace=True):
        tv[cand[e]] = tv[cand[e]] * (1.0 - ETA)
    rand_eqs.append(solve_with(tv)[0])

print(f"\n  HARDENED (greedy): equilibrium {base_eq:.3f} -> {hard_eq:.3f} "
      f"({100*(base_eq-hard_eq)/base_eq:.0f}% reduction); loss_det {base_det:.3f} -> {hard_det:.3f}")
print(f"  random-allocation baseline (same budget): equilibrium {np.mean(rand_eqs):.3f} "
      f"+/- {np.std(rand_eqs):.3f}  -> greedy hardening beats random by "
      f"{np.mean(rand_eqs)-hard_eq:+.3f}")
# does hardening move WHERE randomisation pays? (leader-route mass shift)
shift = float(np.abs(hard_lead - base_lead).sum())
print(f"  equilibrium leader-route mass shift (L1) after hardening: {shift:.2f} "
      f"(>0 => hardening RELOCATES where randomisation pays: the tier interaction)")
json.dump({"od": f"{OD[0]}-{OD[1]}", "eta": ETA, "budget": BUDGET,
           "base_eq": base_eq, "hardened_eq": hard_eq, "greedy_history": history,
           "random_eq_mean": float(np.mean(rand_eqs)), "random_eq_std": float(np.std(rand_eqs)),
           "leader_mass_shift_L1": shift, "alloc": alloc},
          open("models/runs/d2_hardening.json", "w"), indent=2)
print("  [written] models/runs/d2_hardening.json")
