"""Multi-convoy fleet-routing baselines (gen08 Phase M / Obj-5): the CLASSICAL opponents SACRED must
beat.

  * shortest_path_fleet   -- the naive, interdiction-UNAWARE planner: every convoy takes the cheapest
                             route (they stack, and the interdictor ambushes the obvious road);
  * alns_fleet_planner    -- an Adaptive Large Neighbourhood Search coordinating planner (destroy /
                             repair over the joint route assignment, adaptive operator weights,
                             simulated-annealing acceptance) that minimises WORST-CASE mission-failure.
                             It knows the vulnerability map (the fairest, strongest classical
                             opponent) and converges to the oracle's ``loss_det`` = the best any
                             DETERMINISTIC fleet plan can achieve. SACRED (``loss_mixed``) still beats
                             it because a deterministic plan cannot randomise.

`classical_baselines` reports each planner's exploitability (worst-case objective under the
best-response interdictor) beside the oracle references, for the Obj-5 comparison in M3.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy


def _occ(assignment: list[int], R: int) -> tuple[int, ...]:
    o = [0] * R
    for r in assignment:
        o[r] += 1
    return tuple(o)


def shortest_path_fleet(game: InterdictionGame, N: int) -> list[int]:
    """Naive planner: all N convoys on the single cheapest route (interdiction-unaware)."""
    return [int(np.argmin(game.travel_cost))] * N


@dataclass(frozen=True)
class FleetPlan:
    assignment: tuple[int, ...]
    occupancy: tuple[int, ...]
    exploitability: float          # worst-case objective (best-response interdictor)


def _wpick(w: list[float], rng: random.Random) -> int:
    t = sum(w); r = rng.random() * t; c = 0.0
    for i, x in enumerate(w):
        c += x
        if r <= c:
            return i
    return len(w) - 1


def alns_fleet_planner(game: InterdictionGame, N: int, objective: str = "mission", m: int = 1,
                       iters: int = 500, seed: int = 0, restarts: int = 4) -> FleetPlan:
    """ALNS over convoy->route assignments minimising worst-case objective. Converges to loss_det."""
    occs, M = objective_matrix(game, N, objective, m)
    idx = {tuple(int(x) for x in o): i for i, o in enumerate(occs)}
    R = game.n_routes
    rng = random.Random(seed)

    def score(a):
        return float(M[idx[_occ(a, R)]].max())

    def destroy_random(a):
        return rng.sample(range(N), rng.randint(1, max(1, N // 2)))

    def destroy_worst(a):
        j = int(M[idx[_occ(a, R)]].argmax())               # the interdictor's best response
        per = [game.payoff[a[i], j] for i in range(N)]      # each convoy's interception prob under it
        return [int(np.argmax(per))]

    def repair_greedy(a, rem):
        for i in rem:
            best_r, best_s = a[i], math.inf
            for r in range(R):
                a[i] = r; s = score(a)
                if s < best_s - 1e-12:
                    best_s, best_r = s, r
            a[i] = best_r

    def repair_random(a, rem):
        for i in rem:
            a[i] = rng.randrange(R)

    DES, REP = [destroy_random, destroy_worst], [repair_greedy, repair_random]
    wd, wr = [1.0, 1.0], [1.0, 1.0]
    best, best_s = None, math.inf
    for _ in range(restarts):
        cur = [rng.randrange(R) for _ in range(N)]; cur_s = score(cur); T = 0.3
        for _it in range(iters):
            di, ri = _wpick(wd, rng), _wpick(wr, rng)
            cand = cur[:]; repair = REP[ri]; repair(cand, DES[di](cand)); cs = score(cand)
            accept, reward = False, 0
            if cs < best_s - 1e-12:
                best, best_s, reward, accept = cand[:], cs, 3, True
            elif cs < cur_s - 1e-12:
                reward, accept = 2, True
            elif rng.random() < math.exp(-(cs - cur_s) / max(T, 1e-9)):
                reward, accept = 1, True
            if accept:
                cur, cur_s = cand, cs
            wd[di] = 0.85 * wd[di] + 0.15 * reward
            wr[ri] = 0.85 * wr[ri] + 0.15 * reward
            T *= 0.995
        if cur_s < best_s:
            best, best_s = cur[:], cur_s
    return FleetPlan(tuple(best), _occ(best, R), best_s)


def classical_baselines(game: InterdictionGame, N: int, objective: str = "mission", m: int = 1,
                        seed: int = 0) -> dict:
    """Exploitability of each classical planner vs the best-response interdictor, with oracle refs.
    Ordering (mission objective): shortest_path >= alns == optimal_deterministic >= equilibrium."""
    occs, M = objective_matrix(game, N, objective, m)
    idx = {tuple(int(x) for x in o): i for i, o in enumerate(occs)}
    sp = shortest_path_fleet(game, N)
    sp_expl = float(M[idx[_occ(sp, game.n_routes)]].max())
    plan = alns_fleet_planner(game, N, objective, m, seed=seed)
    sol = solve_multiconvoy(game, N, objective, m)
    return {"shortest_path": sp_expl, "alns": plan.exploitability,
            "optimal_deterministic": sol.loss_det, "equilibrium": sol.loss_mixed,
            "alns_plan": plan}
