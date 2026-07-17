"""gen28 DYNAMIC screen (oracle-only, free): the gen19/gen27 register on the aerial fleet
game. Enemy = softmax-BR (tau) to the trailing w-window of realised fleet routes. Per layout:
iid_eq (the static equilibrium mixture's stationary value vs the adaptive enemy = the cap on
LP-style static play), history_opt (RVI over the window MDP; the dynamic optimum), and the
COMPLETE naive-DYNAMIC family pre-registered up front (the gen27 second-amendment lesson):
rotation and anti-repeat over every lane spacing and the full menu. Non-degeneracy gate:
history_opt materially below iid_eq AND the naive-dynamic rules materially above history_opt."""
import itertools
import numpy as np
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_curves import all_lane_sets, build_curve_menu, build_curved_game, dense_hazard_grid
from src.envs.aerial_sector import SectorLattice
from scripts.train_aerial_generalist import random_field

DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(4,j) for j in range(9) if j < 5} | {(8,j) for j in range(9) if j > 3}))
N = 3

def build(seed, w, tau):
    menu, _ = build_curve_menu(DBL, 1.2, R=40, seed=0)
    centres = dense_hazard_grid(DBL, step=0.5)
    pm = random_field(centres, seed)
    game, S = build_curved_game(DBL, menu, centres, 1, r=1.2, p_max=pm)
    R = game.n_routes
    # per-route mission damage if the enemy sits at position h and the fleet stacks route r:
    # 1 - (1-p_rh)^N with p_rh = 1 - S[r,h]
    dmg = 1.0 - S ** N                                        # [R, H]
    def br_probs(window):                                     # softmax-BR to the window routes
        v = dmg[list(window)].mean(axis=0)                    # expected damage vs recent play
        z = np.exp((v - v.max()) / tau)
        return z / z.sum()
    def step_damage(window, r):                               # expected damage of playing r now
        return float(br_probs(window) @ dmg[r])
    # stationary value of a STATIONARY route rule pi(window)->dist, by exact chain rollout on
    # the analytic per-step damage (rule randomness only; 30k steps, burn 2k)
    def stationary(rule, T=30000, burn=2000, rng=np.random.default_rng(0)):
        win = tuple(rng.integers(R, size=w)); acc = 0.0; n = 0
        for t in range(T):
            d = rule(win)
            r = int(rng.choice(R, p=d))
            if t >= burn:
                acc += step_damage(win, r); n += 1
            win = tuple(list(win[1:]) + [r])
        return acc / n
    sol = solve_multiconvoy(game, N, "mission")
    # the stacked route mixture of the fleet equilibrium (mass on stacked occupancies by route)
    occs = list(itertools.combinations_with_replacement(range(R), N))
    d_eq = np.zeros(R)
    for i, o in enumerate(occs):
        if len(set(o)) == 1:
            d_eq[o[0]] += sol.defender_strategy[i]
    d_eq = d_eq / d_eq.sum() if d_eq.sum() > 0 else np.full(R, 1/R)
    iid_eq = stationary(lambda wdw: d_eq)
    rows = {"iid_eq": iid_eq}
    lsets = all_lane_sets(DBL, menu)
    fam = {}
    for rc, li in (lsets.items() if lsets else [(0.0, [])]):
        for k, dd in lane_stack_distributions(game, li, S).items():
            fam[f"{k}@{rc}"] = dd
    for name, dd in fam.items():
        sup = np.where(dd > 1e-9)[0]
        def anti(wdw, dd=dd, sup=sup):
            m = dd.copy()
            m[list(wdw)] = 0.0
            return m / m.sum() if m.sum() > 1e-12 else dd
        rows[f"anti_{name}"] = stationary(anti)
        if len(sup) > w:
            def rot(wdw, sup=sup):
                out = np.zeros(R)
                cand = [r for r in sup if r not in wdw]
                out[cand[0] if cand else sup[0]] = 1.0
                return out
            rows[f"rot_{name}"] = stationary(rot)
    # history_opt by RVI over the window MDP (exact; w<=2 tractable at R=40)
    hist_opt = None
    if w <= 2:
        states = list(itertools.product(range(R), repeat=w))
        sidx = {s: i for i, s in enumerate(states)}
        Dmat = np.array([[step_damage(s, r) for r in range(R)] for s in states])
        V = np.zeros(len(states)); g = 0.0
        for it in range(400):
            Q = Dmat + np.array([[V[sidx[tuple(list(s[1:]) + [r])]] for r in range(R)]
                                 for s in states])
            Vn = Q.min(axis=1); g = Vn.mean(); Vd = Vn - g
            if np.abs(Vd - V).max() < 1e-9: V = Vd; break
            V = Vd
        hist_opt = float(g)
    best_naive_dyn = min(v for k, v in rows.items() if k != "iid_eq")
    best_name = min((k for k in rows if k != "iid_eq"), key=lambda k: rows[k])
    print(f"layout{seed} w={w} tau={tau}: iid_eq={iid_eq:.3f} "
          f"best_naive_dyn={best_naive_dyn:.3f} ({best_name}) "
          + (f"history_opt={hist_opt:.3f} " if hist_opt is not None else "")
          + f"| eq_static={sol.loss_mixed:.3f}", flush=True)
    return iid_eq, best_naive_dyn, hist_opt

if __name__ == "__main__":
    for w, tau in ((2, 0.15), (2, 0.10), (3, 0.15)):
        print(f"--- operating point w={w} tau={tau} ---")
        for s in (2100, 2101, 2102):
            build(s, w, tau)
