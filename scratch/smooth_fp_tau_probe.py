"""Pin the smooth-fictitious-play attacker temperature tau for B2-P3 (NO training, seconds).

B2-P (latest pure BR, held 50 sorties) OVER-disciplines the defender -> last-iterate cycling.
B2-P2 (uniform mixture over ALL past BRs) UNDER-disciplines -> stale attacker -> the travel-cost
gradient parks the policy on one route unpunished. Smooth FP is the canonical middle: each
switch, compute per-iset expected interception e_j against the defender's TRAILING-window play,
and sample each sortie's interdiction from softmax(e_j / tau).

tau must satisfy BOTH pinned criteria on the B2-P instance (33->71, k_extra=8, K=1):
  (a) SMOOTH: vs the equilibrium-like defender (uniform over the 6 disjoint routes), the
      attacker's entropy >= ~1.5 nats (no pure-BR camping -> no cycling);
  (b) RESPONSIVE: vs a parked defender (0.8 on the cheapest route), the interdiction mass on
      isets that cover the cheapest route >= ~0.6 (drift is punished hard and fast).

Run: PYTHONPATH=. .venv/bin/python scratch/smooth_fp_tau_probe.py
"""

from __future__ import annotations

import numpy as np

from src.baselines.interdiction_oracle import build_interdiction_game, solve
from scratch.vuln_band_probe import kaliningrad

TAUS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    G = kaliningrad()
    game = build_interdiction_game(G, "33", "71", K=1, k_extra=8)
    sol = solve(game)
    eq_support = sol.defender_strategy > 1e-6      # the 6 disjoint routes
    uniform6 = np.where(eq_support, 1.0 / eq_support.sum(), 0.0)
    cheapest = int(np.argmin(game.travel_cost))
    parked = np.full(game.n_routes, 0.2 / (game.n_routes - 1)); parked[cheapest] = 0.8
    covers_cheapest = game.payoff[cheapest, :] > 0.0
    print(f"{game.n_routes} routes; {game.payoff.shape[1]} isets; cheapest route {cheapest} "
          f"(cost {game.travel_cost[cheapest]:.1f}), covered by {covers_cheapest.sum()} isets\n")
    for tau in TAUS:
        a_smooth = softmax(uniform6 @ game.payoff, tau)
        h = float(-(a_smooth[a_smooth > 0] * np.log(a_smooth[a_smooth > 0])).sum())
        a_resp = softmax(parked @ game.payoff, tau)
        punish = float(a_resp[covers_cheapest].sum())
        # what the defender then feels: expected interception per route under each attacker.
        drift_cost = float(game.payoff[cheapest] @ a_resp)   # parked route's interception rate
        print(f"tau={tau:５.2f}:  H(vs uniform6)={h:5.2f} nats   punish(parked)={punish:.2f}   "
              f"parked-route interception={drift_cost:.2f}")


if __name__ == "__main__":
    main()
