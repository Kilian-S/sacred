"""Probe B: the risk-aversion SPECTRUM x fleet size x interdictor budget. Maps how loss-averse the
objective must be for SACRED to keep winning as the fleet grows: linear (risk-neutral) -> P(>=1 lost)
(mission) -> P(>=2) ... -> P(all lost). loss_det (deterministic/ALNS) vs loss_mixed (SACRED). NO training."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from scratch.multiconvoy_probe import occupancies, row_minimiser
from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from scratch.vuln_band_probe import kaliningrad


def caught_pmf(o, p):
    pmf = np.array([1.0])
    for r in range(len(o)):
        if o[r] > 0:
            pmf = np.convolve(pmf, binom.pmf(np.arange(o[r] + 1), o[r], p[r]))
    return pmf


def obj_gaps(payoff, N, kind, m=1):
    R, nj = payoff.shape
    occs = list(occupancies(R, N))
    M = np.zeros((len(occs), nj))
    for oi, o in enumerate(occs):
        for j in range(nj):
            pmf = caught_pmf(o, payoff[:, j])
            if kind == "linear":
                M[oi, j] = float(np.arange(len(pmf)) @ pmf) / N
            else:
                M[oi, j] = float(pmf[m:].sum()) if m < len(pmf) else 0.0
    ld = float(min(M[i].max() for i in range(M.shape[0])))
    lm, _ = row_minimiser(M)
    return ld, lm


def run(label, s, t, k_extra, band, Ns, Ks):
    G = kaliningrad()
    routes = build_route_set(G, str(s), str(t), k_extra, "w")
    ifn = None
    if band:
        cand = set().union(*(edges_of_route(r) for r in routes))
        ifn = survival_intercept_fn(length_band_vulnerability(G, cand, band=band, weight="w"))
    print(f"\n#### {label} ####")
    for K in Ks:
        game = build_interdiction_game(G, str(s), str(t), K=K, k_extra=k_extra, intercept_fn=ifn)
        print(f"  K={K} routes={game.n_routes}  (each cell = gap = loss_det - loss_mixed)")
        print(f"    {'N':>2} {'linear':>8} {'P(>=1)':>8} {'P(>=2)':>8} {'P(all)':>8}")
        for N in Ns:
            cells = []
            for kind, m in [("linear", 0), ("geq", 1), ("geq", 2), ("geq", N)]:
                if m > N:
                    cells.append("  -  "); continue
                ld, lm = obj_gaps(game.payoff, N, kind, m)
                cells.append(f"{ld-lm:6.2f}")
            print(f"    {N:>2} {cells[0]:>8} {cells[1]:>8} {cells[2]:>8} {cells[3]:>8}")


if __name__ == "__main__":
    print("SPECTRUM: gap by objective (risk-neutral -> loss-averse), fleet size N, budget K.")
    run("BAND 110->135 (soft, 3 routes)", "110", "135", 0, (0.15, 0.95), Ns=(1, 2, 3, 4, 5), Ks=(1, 2, 3))
    run("BAND 33->71 disjoint (soft, 6 routes)", "33", "71", 0, (0.15, 0.95), Ns=(1, 2, 3, 4), Ks=(1, 2))
