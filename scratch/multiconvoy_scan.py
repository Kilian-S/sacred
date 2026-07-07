"""Probe A: does the multi-convoy + mission-failure SACRED win GENERALISE across Kaliningrad OD pairs?
(soft band 0.15-0.95, disjoint routes, K=1). Reports the loss_det/loss_mixed gap and whether the
deterministic optimum coordinates (spreads convoys = a real metaheuristic job). NO training."""
from __future__ import annotations

import random
import statistics as st

import numpy as np

from scratch.multiconvoy_probe import occupancies, row_minimiser
from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from scratch.vuln_band_probe import kaliningrad


def mission_linear(payoff: np.ndarray, N: int):
    R, nj = payoff.shape
    occs = list(occupancies(R, N))
    Mm = np.zeros((len(occs), nj)); Ml = np.zeros((len(occs), nj))
    for oi, o in enumerate(occs):
        for j in range(nj):
            p = payoff[:, j]
            Mm[oi, j] = 1.0 - float(np.prod((1.0 - p) ** o))   # P(>=1 lost) = mission-failure
            Ml[oi, j] = float(o @ p) / N                        # E[fraction lost]
    return occs, Mm, Ml


def gaps(M):
    ld = float(min(M[i].max() for i in range(M.shape[0])))
    lm, _ = row_minimiser(M)
    di = int(np.argmin(M.max(axis=1)))
    return ld, lm, di


def analyse(G, s, t, band=(0.15, 0.95)):
    routes = build_route_set(G, str(s), str(t), 0, "w")
    if len(routes) < 3:
        return None
    cand = set().union(*(edges_of_route(r) for r in routes))
    ifn = survival_intercept_fn(length_band_vulnerability(G, cand, band=band, weight="w"))
    game = build_interdiction_game(G, str(s), str(t), K=1, k_extra=0, intercept_fn=ifn)
    out = {"routes": game.n_routes}
    for N in (2, 3):
        occs, Mm, Ml = mission_linear(game.payoff, N)
        ld, lm, di = gaps(Mm)
        out[f"m{N}"] = (ld, lm, ld - lm, int((occs[di] > 0).sum()))
        ld2, lm2, _ = gaps(Ml)
        out[f"l{N}"] = ld2 - lm2
    return out


def main():
    G = kaliningrad(); nodes = list(G.nodes()); rng = random.Random(0)
    tried = set(); results = []
    while len(tried) < 400 and len(results) < 20:
        u, v = rng.sample(nodes, 2)
        if (u, v) in tried:
            continue
        tried.add((u, v))
        try:
            r = analyse(G, u, v)
        except Exception:
            r = None
        if r:
            results.append(((u, v), r))
    print(f"Analysed {len(results)} OD pairs (soft band 0.15-0.95, K=1, >=3 disjoint routes).\n")
    print(f"{'OD':>11} {'rts':>3} | N2 mission det/mix/gap spr | N3 gap | N2 lin | N3 lin")
    g2 = []; g3 = []; spr = []; l2 = []; l3 = []
    for (u, v), r in results:
        ld, lm, gp, sp = r["m2"]; g2.append(gp); spr.append(sp)
        g3.append(r["m3"][2]); l2.append(r["l2"]); l3.append(r["l3"])
        print(f"{u+'->'+v:>11} {r['routes']:>3} | {ld:.2f}/{lm:.2f}/{gp:.2f}  spread {sp} | "
              f"{r['m3'][2]:.2f}  | {r['l2']:.2f}  | {r['l3']:.2f}")
    print(f"\nSUMMARY mission N=2: median gap {st.median(g2):.2f}, range [{min(g2):.2f},{max(g2):.2f}], "
          f"frac gap>0.3={sum(x>0.3 for x in g2)/len(g2):.0%}, frac det-SPREAD={sum(s>1 for s in spr)/len(spr):.0%}")
    print(f"SUMMARY mission N=3: median gap {st.median(g3):.2f}, range [{min(g3):.2f},{max(g3):.2f}]")
    print(f"CONTRAST linear   N=2: median gap {st.median(l2):.2f} ; N=3: median gap {st.median(l3):.2f} "
          f"(smaller = risk-neutral dilutes SACRED)")


if __name__ == "__main__":
    main()
