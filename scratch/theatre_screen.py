"""gen28 v3-theatre oracle screen (free): the naive-vs-eq gap on the REAL corridor at the N=3
fleet mission register, over a small aiming grid (K, hazard stride, standoff, LOS on/off)."""
import itertools
from math import factorial
import numpy as np
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_theatre import build_theatre_game, load_theatre, hazard_sites, CLASS_NAME
N = 3

def occ(R):
    os_ = list(itertools.combinations_with_replacement(range(R), N))
    V = np.zeros((len(os_), R))
    for i, c in enumerate(os_):
        for r in c: V[i, r] += 1
    return os_, V, {tuple(int(x) for x in v): i for i, v in enumerate(V)}

def stack(d, V, idx):
    o = np.zeros(len(V))
    for r, p in enumerate(d):
        if p > 1e-12:
            v = np.zeros(len(d)); v[r] = N; o[idx[tuple(int(x) for x in v)]] += p
    return o

def indep(d, os_, V):
    o = np.zeros(len(V))
    for i, c in enumerate(os_):
        cnt = {}
        for r in c: cnt[r] = cnt.get(r, 0) + 1
        coef, pr = factorial(N), 1.0
        for r, k in cnt.items(): coef //= factorial(k); pr *= d[r] ** k
        o[i] = coef * pr
    return o

def run(K, stride, standoff, los=True, menu=24):
    th = load_theatre("data/maps/theatre_kgd_gvardeysk.json")
    game, mr, coords, rr, pp, S = build_theatre_game(th, K=K, menu_size=menu, site_stride=stride,
                                                     los=los, standoff_km=standoff)
    sol = solve_multiconvoy(game, N, "mission")
    os_, V, idx = occ(game.n_routes)
    _, M = objective_matrix(game, N, "mission", 1)
    exp = 1.0 - S.min(axis=1)
    d_uni = np.full(game.n_routes, 1.0/game.n_routes)
    d_inv = 1.0/np.clip(exp,1e-9,None); d_inv/=d_inv.sum()
    rows = {"uni-stack": float((stack(d_uni,V,idx)@M).max()),
            "inv-stack": float((stack(d_inv,V,idx)@M).max()),
            "uni-indep": float((indep(d_uni,os_,V)@M).max())}
    bn = min(rows.values())
    print(f"K={K} stride={stride} standoff={standoff}km los={los}: R={game.n_routes} H={len(coords)} "
          f"| eq={sol.loss_mixed:.3f} det={sol.loss_det:.3f} best_naive={bn:.3f} "
          f"naive/eq={bn/max(sol.loss_mixed,1e-9):.2f}", flush=True)
    return sol.loss_mixed, bn

if __name__ == "__main__":
    print("=== aiming grid (real Kaliningrad-Gvardeysk corridor, N=3 mission) ===")
    for standoff in (4.0, 7.0, 10.0):
        for stride in (3, 4):
            for K in (1, 2):
                run(K, stride, standoff)
    print("--- LOS ablation at a mid cell ---")
    run(1, 3, 7.0, los=False)
