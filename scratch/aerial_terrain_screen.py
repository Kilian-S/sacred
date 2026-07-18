"""gen28 v5 theatre screen (oracle-only, free): does the naive-vs-optimal fleet gap survive on
procedurally-generated heterogeneous theatres, and where? N=3 mission. Reports per theatre:
fleet-stack equilibrium (the deployable-class yardstick), deterministic optimum, and the
COMPLETE naive family INCLUDING the terrain-aware CORRIDOR PIPELINE (inverse-exposure route
selection = segment terrain -> prefer cover -> spread). Timing + tractability printed (the
gen28 dogma). No training."""
import itertools, time
import numpy as np
from src.baselines.interdiction_oracle import best_response_attacker, _row_minimiser
from src.baselines.multiconvoy_oracle import objective_matrix
from src.envs.aerial_terrain import generate_theatre, build_theatre_game, CLASS_NAME
from math import factorial

N = 3

def occ_index(R):
    occs = list(itertools.combinations_with_replacement(range(R), N))
    vecs = np.zeros((len(occs), R))
    for i, c in enumerate(occs):
        for r in c: vecs[i, r] += 1
    return occs, vecs, {tuple(int(x) for x in v): i for i, v in enumerate(vecs)}

def stack_dist(d, vecs, idx):
    out = np.zeros(len(vecs))
    for r, p in enumerate(d):
        if p > 1e-12:
            v = np.zeros(len(d)); v[r] = N
            out[idx[tuple(int(x) for x in v)]] += p
    return out

def indep_dist(d, occs, vecs):
    out = np.zeros(len(vecs))
    for i, c in enumerate(occs):
        cnt = {}
        for r in c: cnt[r] = cnt.get(r, 0)+1
        coef, p = factorial(N), 1.0
        for r, k in cnt.items(): coef //= factorial(k); p *= d[r]**k
        out[i] = coef*p
    return out

def screen(seed, ny=17, nx=40, step=0.75, R=36):
    t0 = time.time()
    th = generate_theatre(ny=ny, nx=nx, seed=seed)
    try:
        game, S, menu, centres, pmax, rad, exp = build_theatre_game(th, K=1, step=step, R=R, seed=0)
    except ValueError as e:
        print(f"seed {seed}: SKIP ({e})"); return None
    t_build = time.time()-t0
    occs, vecs, idx = occ_index(game.n_routes)
    _, M = objective_matrix(game, N, "mission", 1)
    def val(od): return float((od @ M).max())
    # fleet-stack equilibrium (deployable class): LP over the R stacked rows
    stack_rows = np.stack([M[idx[tuple(int(x) for x in (np.eye(game.n_routes)[r]*N))]]
                           for r in range(game.n_routes)])
    eq, _ = _row_minimiser(stack_rows)
    det = float(M.max(axis=1).min())
    # complete naive family
    exposure = 1.0 - S.min(axis=1)
    rows = {}
    # corridor pipeline: inverse-exposure over the k safest routes, stacked
    for k in (3, 5, 8):
        safe = np.argsort(exposure)[:k]
        w = 1.0/np.clip(exposure[safe], 1e-9, None); w/=w.sum()
        d = np.zeros(game.n_routes); d[safe] = w
        rows[f"corridor_stack_k{k}"] = val(stack_dist(d, vecs, idx))
        rows[f"corridor_indep_k{k}"] = val(indep_dist(d, occs, vecs))
    # uniform / inv-exposure over the FULL menu (stack + indep)
    uni = np.full(game.n_routes, 1.0/game.n_routes)
    invx = 1.0/np.clip(exposure, 1e-9, None); invx/=invx.sum()
    for nm, d in (("uniform", uni), ("invrisk", invx)):
        rows[f"{nm}_full_stack"] = val(stack_dist(d, vecs, idx))
        rows[f"{nm}_full_indep"] = val(indep_dist(d, occs, vecs))
    # single safest route (pure corridor, deterministic) + shortest
    d0 = np.zeros(game.n_routes); d0[int(np.argmin(exposure))] = 1.0
    rows["safest_single_stack"] = val(stack_dist(d0, vecs, idx))
    bn = min(rows.values()); bnk = min(rows, key=rows.get)
    terr = {CLASS_NAME[c]: int((th.grid==c).sum()) for c in range(8) if (th.grid==c).any()}
    print(f"seed {seed}: eq={eq:.3f} det={det:.3f} best_naive={bn:.3f} ({bnk}) "
          f"naive/eq={bn/eq:.2f} det/eq={det/eq:.2f} | R={game.n_routes} H={len(centres)} "
          f"nocc={len(occs)} | {t_build:.1f}s | {terr}", flush=True)
    return eq, bn, bn/eq

if __name__ == "__main__":
    print("=== v5 theatre screen: N=3 mission, procedural terrain ===")
    ratios = []
    for seed in range(12):
        r = screen(seed)
        if r: ratios.append(r[2])
    if ratios:
        ratios = np.array(ratios)
        print(f"\nnaive/eq across {len(ratios)} theatres: median={np.median(ratios):.2f} "
              f"min={ratios.min():.2f} max={ratios.max():.2f} "
              f">=1.3: {(ratios>=1.3).mean()*100:.0f}% >=1.5: {(ratios>=1.5).mean()*100:.0f}%")
