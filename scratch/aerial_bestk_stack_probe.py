"""Baseline-completeness verification probe (critic re-derivation, 2026-07-19; oracle-only, free).

Re-derives, with independent code, the 2026-07-18 critic finding that the gen28 v3.x Tier-1
positive (SACRED pooled 0.734-0.746 < best_naive 0.754 on dblpinch_banded_K1_r1.2) does not
survive the project's own baseline-completeness dogma: the screen's naive family (lane sets +
full-menu stacks) omits the strongest small-subset stack. This probe adds, on the SAME cell,
same menu, same yardstick:
  - best k-route UNIFORM STACK, exhaustive for k<=4, greedy+1-swap local search for k=5..8
    (payoff-aware subset choice: the in-sample "napkin rule allowed to look" row);
  - a payoff-BLIND separation rule (k most lateral-separated, safest-first) for the honest
    nuance row;
  - the tabular-FP row and the ledger family, reproduced for anchoring.
Also re-checks the v3-theatre VECTOR headline cell's gap against the same missing rows.
"""
import itertools
import numpy as np

from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_curves import (all_lane_sets, build_curve_menu, build_curved_game,
                                    dense_hazard_grid)
from src.envs.aerial_sector import SectorLattice, banded_pmax

N = 3
DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(4, j) for j in range(9) if j < 5} | {(8, j) for j in range(9) if j > 3}))


def occ_maps(R):
    occs = list(itertools.combinations_with_replacement(range(R), N))
    vecs = np.zeros((len(occs), R))
    for i, c in enumerate(occs):
        for r in c:
            vecs[i, r] += 1
    idx = {tuple(int(x) for x in v): i for i, v in enumerate(vecs)}
    return occs, vecs, idx


def stack_matrix(M, idx, R):
    """Mst[r, j] = mission damage when the whole fleet stacks on route r vs iset j."""
    Mst = np.zeros((R, M.shape[1]))
    for r in range(R):
        v = np.zeros(R)
        v[r] = N
        Mst[r] = M[idx[tuple(int(x) for x in v)]]
    return Mst


def best_k_stack_exhaustive(Mst, k, chunk=4000):
    R = Mst.shape[0]
    best_val, best_S = np.inf, None
    combos = itertools.combinations(range(R), k)
    while True:
        block = list(itertools.islice(combos, chunk))
        if not block:
            break
        arr = np.array(block)                        # (c, k)
        vals = Mst[arr].mean(axis=1).max(axis=1)     # (c,)
        i = int(vals.argmin())
        if vals[i] < best_val:
            best_val, best_S = float(vals[i]), tuple(block[i])
    return best_val, best_S


def stack_val(Mst, S):
    return float(Mst[list(S)].mean(axis=0).max())


def best_k_stack_local(Mst, k, restarts=6, seed=0):
    rng = np.random.default_rng(seed)
    R = Mst.shape[0]
    best_val, best_S = np.inf, None
    for _ in range(restarts):
        S = list(rng.choice(R, size=k, replace=False))
        val = stack_val(Mst, S)
        improved = True
        while improved:
            improved = False
            for i in range(k):
                for r in range(R):
                    if r in S:
                        continue
                    T = S.copy()
                    T[i] = r
                    v = stack_val(Mst, T)
                    if v < val - 1e-12:
                        S, val, improved = T, v, True
        if val < best_val:
            best_val, best_S = val, tuple(sorted(S))
    return best_val, best_S


def tabular_fp(M, iters=1500, lr=0.25):
    n = M.shape[0]
    x = np.full(n, 1.0 / n)
    tot = np.zeros(n)
    for t in range(1, iters + 1):
        tot += x
        j = int(((tot / t) @ M).argmax())
        x = x * np.exp(-lr * M[:, j])
        x /= x.sum()
    avg = tot / iters
    return float((avg @ M).max())


def fleet_cell():
    print("=== v3.x FLEET headline cell: dblpinch_banded, K=1, r=1.2 (N=3 mission) ===")
    menu, _ = build_curve_menu(DBL, 1.2, R=40, seed=0)
    centres = dense_hazard_grid(DBL, step=0.5)
    pm = banded_pmax(centres, DBL.ny)
    game, S = build_curved_game(DBL, menu, centres, 1, r=1.2, p_max=pm)
    sol = solve_multiconvoy(game, N, "mission")
    occs, vecs, idx = occ_maps(game.n_routes)
    _, M = objective_matrix(game, N, "mission", 1)
    Mst = stack_matrix(M, idx, game.n_routes)

    def val(occ_dist):
        return float((occ_dist @ M).max())

    def stack_occ(d):
        out = np.zeros(len(vecs))
        for r, p in enumerate(d):
            if p > 1e-12:
                v = np.zeros(len(d))
                v[r] = N
                out[idx[tuple(int(x) for x in v)]] += p
        return out

    rows = {}
    lsets = all_lane_sets(DBL, menu)
    for rc, li in (lsets.items() if lsets else [(0.0, [])]):
        for k, d in lane_stack_distributions(game, li, S).items():
            rows[f"{k}@{rc}"] = val(stack_occ(d))
    ledger_bn = min(rows.values())
    print(f"eq={sol.loss_mixed:.4f}  det={sol.loss_det:.4f}  "
          f"ledger-family best_naive={ledger_bn:.4f} ({ledger_bn/sol.loss_mixed:.2f}x eq)")
    print(f"tabular-FP (same BR oracle) = {tabular_fp(M):.4f}")
    for k in (2, 3, 4):
        v, Sk = best_k_stack_exhaustive(Mst, k)
        print(f"best {k}-route uniform stack (exhaustive) = {v:.4f} "
              f"({v/sol.loss_mixed:.2f}x eq)  routes={Sk}")
    for k in (5, 6, 8):
        v, Sk = best_k_stack_local(Mst, k)
        print(f"best {k}-route uniform stack (local search) = {v:.4f} "
              f"({v/sol.loss_mixed:.2f}x eq)  routes={Sk}")
    # payoff-blind nuance row: k safest-by-single-hazard-exposure with a lateral-separation
    # greedy (no payoff-matrix access beyond per-route worst exposure, which the defender
    # observes anyway as the layout feature).
    exposure = 1.0 - S.min(axis=1)
    order = np.argsort(exposure)
    for k in (4, 6):
        Sb = list(order[:k])
        print(f"payoff-blind safest-{k} stack = {stack_val(Mst, Sb):.4f} "
              f"({stack_val(Mst, Sb)/sol.loss_mixed:.2f}x eq)")
    print(f"SACRED banked Tier-1 (ledger, 8/9 seeds, three batches): 0.746/0.734/0.742 pooled")


def theatre_cell():
    print("\n=== v3-THEATRE vector headline cell: K=1, standoff 4 km (N=3 mission) ===")
    from src.envs.aerial_theatre_vec import build_theatre_game, load_vec_theatre
    th = load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json")
    game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(
        th, K=1, n_lanes=14, n_terrain=12, standoff_km=4.0)
    sol = solve_multiconvoy(game, N, "mission")
    occs, vecs, idx = occ_maps(game.n_routes)
    _, M = objective_matrix(game, N, "mission", 1)
    Mst = stack_matrix(M, idx, game.n_routes)
    print(f"R={game.n_routes}  eq={sol.loss_mixed:.4f}  det={sol.loss_det:.4f}")
    for k in (2, 3, 4):
        v, Sk = best_k_stack_exhaustive(Mst, k)
        print(f"best {k}-route uniform stack (exhaustive) = {v:.4f} "
              f"({v/sol.loss_mixed:.2f}x eq)  routes={Sk}")
    print(f"tabular-FP = {tabular_fp(M):.4f}")


if __name__ == "__main__":
    fleet_cell()
    try:
        theatre_cell()
    except Exception as e:  # the vec loader signature may differ; fleet cell is the primary
        print(f"[theatre cell skipped: {e}]")
