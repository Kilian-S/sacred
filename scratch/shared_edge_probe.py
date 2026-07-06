"""Oracle probe pinning the class-(b) SHARED-EDGE instances (I3/B2; NO training, seconds of CPU).

Class (b) = candidate route sets containing k-shortest paths, which on a real road network are
near-duplicates funnelling through the same cheap chokepoint edges, plus the edge-disjoint
alternatives. Hard interception. The asymmetry is OVERLAP structure: probability mass on routes
sharing an edge STACKS on that edge, so the equilibrium spreads across structurally independent
routes, while any COST-driven mixture piles onto the cheap shared cluster.

For each (OD, k_extra, K) this prints:
  * the route set: costs, first-hop groups (first-hop collision => the trie/walk build is needed),
    overlap of each route with the cheapest route;
  * loss_det, equilibrium value (loss_mixed) + defender strategy;
  * exploitability anchors: cheapest-route deterministic, uniform, and a COST-SOFTMAX family
    softmax(-cost/T) over routes for a T grid: the best incidental mixture any cost-driven
    defender could realise. The pin for a B2 instance is min_T expl(softmax_T) >> value:
    then NO cost-calibrated vanilla can sit near the equilibrium, unlike wave-1 instance B.

Run: PYTHONPATH=. .venv/bin/python scratch/shared_edge_probe.py
"""

from __future__ import annotations

import numpy as np

from src.baselines.interdiction_oracle import (
    best_response_attacker, build_interdiction_game, solve)
from scratch.vuln_band_probe import kaliningrad

ODS = [("33", "71"), ("110", "135")]
K_EXTRAS = [4, 8, 12]
KS = [1, 2]


def cost_softmax(costs: np.ndarray, T: float) -> np.ndarray:
    z = -(costs - costs.min()) / max(T, 1e-9)
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    G = kaliningrad()
    for s, t in ODS:
        for k_extra in K_EXTRAS:
            game = build_interdiction_game(G, s, t, K=1, k_extra=k_extra)
            costs = game.travel_cost
            fh_groups: dict = {}
            for i, r in enumerate(game.routes):
                fh_groups.setdefault(r[1], []).append(i)
            cheapest = int(np.argmin(costs))
            overlap = [len(game.route_edges[i] & game.route_edges[cheapest]) for i in range(game.n_routes)]
            print(f"\n=== OD {s}->{t} k_extra={k_extra}: {game.n_routes} routes, "
                  f"{len(fh_groups)} first hops (max group {max(len(v) for v in fh_groups.values())}) ===")
            print(f"  costs={np.round(costs, 2).tolist()}")
            print(f"  edges shared with cheapest route: {overlap}")
            for K in KS:
                g = game if K == 1 else build_interdiction_game(G, s, t, K=K, k_extra=k_extra)
                sol = solve(g)
                det = np.zeros(g.n_routes); det[cheapest] = 1.0
                _, expl_det = best_response_attacker(g, det)
                uni = np.ones(g.n_routes) / g.n_routes
                _, expl_uni = best_response_attacker(g, uni)
                # cost-softmax family: T scaled to the cost spread (0 -> greedy, inf -> uniform).
                spread = max(float(costs.max() - costs.min()), 1e-6)
                expl_sm = {}
                for frac in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
                    _, e = best_response_attacker(g, cost_softmax(costs, frac * spread))
                    expl_sm[frac] = e
                best_frac = min(expl_sm, key=expl_sm.get)
                print(f"  K={K}: loss_det={sol.loss_det:.3f} value={sol.value:.3f}  "
                      f"cheapest-det={expl_det:.3f}  uniform={expl_uni:.3f} ({expl_uni/sol.value:.2f}x)")
                print(f"       cost-softmax expl over T/spread: "
                      f"{ {k: round(v, 3) for k, v in expl_sm.items()} }")
                print(f"       BEST cost mixture = {expl_sm[best_frac]:.3f} "
                      f"({expl_sm[best_frac]/sol.value:.2f}x value)  "
                      f"defender eq d(top5)={np.round(np.sort(sol.defender_strategy)[::-1][:5], 3).tolist()}")


if __name__ == "__main__":
    main()
