"""Oracle probe pinning the I3 heterogeneous-vulnerability instances (NO training, seconds of CPU).

For each candidate OD pair and length-band, computes on the edge-disjoint route set (k_extra=0,
the I2 instance family): the per-route max vulnerability p_i*, the (non-uniform) equilibrium
defender strategy and value, and the exploitability ladder
    shortest-route deterministic  >  uniform mixing  >  equilibrium (loss_mixed),
plus loss_det (the best deterministic route's worst case). The gen08 ledger pins the band/OD
choices from THIS table (pinned by probes, never by outcomes: the A4 rule). The interesting
instances are those where uniform mixing is measurably suboptimal (uniform/equilibrium ratio
well above 1), because that is exactly the vanilla-vs-SACRED separation I3 needs.

Run: PYTHONPATH=. .venv/bin/python scratch/vuln_band_probe.py
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    best_response_attacker, build_interdiction_game, length_band_vulnerability,
    solve, survival_intercept_fn)
from src.utils.graph_utils import load_osm_graph_and_demands
from src.envs.interdiction import _DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS

ODS = [("33", "71"), ("110", "135")]
# ascending bands: vulnerability correlates with travel cost (the wave-1 instances).
# DESCENDING bands: inverse correlation (short edges = watched chokepoints), the B' candidates:
# cost and security conflict, so cost-driven mixing is miscalibrated by construction.
BANDS = [(0.2, 0.9), (0.15, 0.95), (0.3, 0.8), (0.95, 0.15), (0.9, 0.2)]
KS = [1, 2]


def kaliningrad() -> nx.Graph:
    nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    return G


def main() -> None:
    G = kaliningrad()
    for s, t in ODS:
        hard = build_interdiction_game(G, s, t, K=1, k_extra=0)
        cand = sorted(set().union(*hard.route_edges), key=repr)
        print(f"\n=== OD {s}->{t}: {hard.n_routes} edge-disjoint routes, {len(cand)} candidate edges ===")
        for band in BANDS:
            vuln = length_band_vulnerability(G, cand, band=band)
            fn = survival_intercept_fn(vuln)
            for K in KS:
                game = build_interdiction_game(G, s, t, K=K, k_extra=0, intercept_fn=fn)
                sol = solve(game)
                p_star = [max(vuln[e] for e in re) for re in game.route_edges]
                sp = int(np.argmin(game.travel_cost))
                det_sp = np.zeros(game.n_routes); det_sp[sp] = 1.0
                _, expl_sp = best_response_attacker(game, det_sp)
                uni = np.ones(game.n_routes) / game.n_routes
                _, expl_uni = best_response_attacker(game, uni)
                closed = 1.0 / sum(1.0 / p for p in p_star) if K == 1 else float("nan")
                d = sol.defender_strategy
                print(f"  band={band} K={K}:  p*={np.round(p_star, 3).tolist()}")
                print(f"    expl: shortest={expl_sp:.3f}  uniform={expl_uni:.3f}  "
                      f"equilibrium={sol.value:.3f} (closed-form {closed:.3f})  loss_det={sol.loss_det:.3f}")
                print(f"    uniform/equilibrium = {expl_uni / sol.value:.2f}x   "
                      f"defender d = {np.round(d, 3).tolist()}")


if __name__ == "__main__":
    main()
