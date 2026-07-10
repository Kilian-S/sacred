"""B4: correlated interception (ORACLE-ONLY, no training). Sweeps the within-route interception
correlation rho (0 = independent, the disclosed conservative model everything else uses; 1 = one
ambush team catches a stacked column all-or-nothing) and reports the headline ladder as a function
of rho on the multi-convoy headline instances. Converts the independence CAVEAT
(CRITIQUE_INTERDICTION §3.3) into an Obj-5 robustness curve.

Run: PYTHONPATH=. .venv/bin/python scratch/correlated_interception_probe.py
"""
from __future__ import annotations

import itertools
import json

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import objective_value, occupancies
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS
from src.utils.graph_utils import load_osm_graph_and_demands

N, K, BAND, KX = 3, 1, (0.15, 0.95), 8
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
ODS = [("35", "159"), ("62", "97")]

nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
G = nx.Graph()
for u, v, d in edges:
    G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
ALL = list(G.edges())


def matrix(game, rho):
    occs = occupancies(game.n_routes, N)
    M = np.zeros((len(occs), game.payoff.shape[1]))
    for oi, occ in enumerate(occs):
        for j in range(game.payoff.shape[1]):
            M[oi, j] = objective_value(np.asarray(occ), game.payoff[:, j], N, "mission", rho=rho)
    return occs, M


def row_min(M):
    n, k = M.shape
    c = np.zeros(n + 1); c[-1] = 1.0
    A = np.hstack([M.T, -np.ones((k, 1))])
    res = linprog(c, A_ub=A, b_ub=np.zeros(k),
                  A_eq=np.hstack([np.ones((1, n)), np.zeros((1, 1))]), b_eq=[1.0],
                  bounds=[(0, 1)] * n + [(None, None)], method="highs")
    return float(res.x[-1])


out = {}
for od in ODS:
    routes = build_route_set(G, od[0], od[1], KX, "w")
    cand = set().union(*(edges_of_route(r) for r in routes))
    vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=ALL)
    game = build_interdiction_game(G, od[0], od[1], K, k_extra=KX, weight="w",
                                   intercept_fn=survival_intercept_fn(vuln))
    print(f"\n=== {od[0]}-{od[1]} (N={N}, K={K}, mission) ===")
    print(f"  {'rho':>5} {'loss_det (ALNS)':>16} {'loss_mixed (eq)':>16} {'gap':>7}")
    rows = []
    for rho in RHOS:
        occs, M = matrix(game, rho)
        eq = row_min(M)
        det = float(min(M[i].max() for i in range(M.shape[0])))
        rows.append({"rho": rho, "loss_det": det, "loss_mixed": eq, "gap": det - eq})
        print(f"  {rho:>5.2f} {det:>16.3f} {eq:>16.3f} {det - eq:>7.3f}")
    out[f"{od[0]}-{od[1]}"] = rows

print("\nReading: E[fraction lost] is rho-invariant; under the MISSION objective the equilibrium "
      "(loss_mixed) FALLS as rho rises (a stacked column shares one shock) -> independence (rho=0) "
      "is the CONSERVATIVE assumption for the SACRED randomised stack; the gap over ALNS holds.")
json.dump(out, open("models/runs/correlated_interception.json", "w"), indent=2)
print("[written] models/runs/correlated_interception.json")
