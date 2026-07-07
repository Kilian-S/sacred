"""gen08 F1 (waves A + C) pre-launch probes: anchors + smooth-FP tau re-pin. NO training, seconds.
Reproducibility record for the F1 launch (experiments/gen08_interdiction.md, F1 launch record).
Run: PYTHONPATH=. .venv/bin/python scratch/gen08_f1_probe.py
"""
from __future__ import annotations

import numpy as np

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, solve, survival_intercept_fn, best_response_attacker)
from src.envs.interdiction import make_interdiction_env
from scratch.vuln_band_probe import kaliningrad

CELLS = [  # (label, od, K, band)
    ("A K=1", ("33", "71"), 1, None),
    ("A K=2", ("33", "71"), 2, None),
    ("A K=3", ("33", "71"), 3, None),
    ("C K=1 band", ("110", "135"), 1, (0.15, 0.95)),
]
TAUS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]


def softmax(x, tau):
    e = np.exp((x - x.max()) / tau)
    return e / e.sum()


def anchors():
    print("=== Anchors (loss_det / equilibrium / uniform / shortest) ===")
    for label, od, K, band in CELLS:
        env = make_interdiction_env(od=od, K=K, k_extra_routes=0, edge_vuln_band=band)
        sol = solve(env.game); n = env.game.n_routes
        _, eu = best_response_attacker(env.game, np.ones(n) / n)
        det = np.zeros(n); det[env.shortest_route_index()] = 1.0
        _, esp = best_response_attacker(env.game, det)
        print(f"  {label:14s}: routes={n} loss_det={sol.loss_det:.3f} eq={sol.value:.3f} "
              f"uniform={eu:.3f} shortest={esp:.3f}")


def tau_pin():
    print("\n=== tau re-pin (SMOOTH: H(vs eq)>=1.5 ; RESPONSIVE: punish(parked)>=0.6) ===")
    for label, od, K, band in CELLS:
        G = kaliningrad(); s, t = od
        ifn = None
        if band is not None:
            routes = build_route_set(G, s, t, 0, "w")
            cand = set().union(*(edges_of_route(r) for r in routes))
            ifn = survival_intercept_fn(length_band_vulnerability(G, cand, band=band, weight="w"))
        game = build_interdiction_game(G, s, t, K=K, k_extra=0, intercept_fn=ifn)
        eq = solve(game).defender_strategy
        cheapest = int(np.argmin(game.travel_cost))
        parked = np.full(game.n_routes, 0.2 / (game.n_routes - 1)); parked[cheapest] = 0.8
        covers = game.payoff[cheapest, :] > 0.0
        print(f"  {label} ({game.payoff.shape[1]} isets):")
        for tau in TAUS:
            a = softmax(eq @ game.payoff, tau)
            h = float(-(a[a > 0] * np.log(a[a > 0])).sum())
            punish = float(softmax(parked @ game.payoff, tau)[covers].sum())
            ok = "OK" if (h >= 1.5 and punish >= 0.6) else ""
            print(f"    tau={tau:4.2f}  H(vs eq)={h:5.2f}  punish(parked)={punish:.2f}  {ok}")


if __name__ == "__main__":
    anchors()
    tau_pin()
