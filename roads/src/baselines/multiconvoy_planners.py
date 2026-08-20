"""Classical multi-convoy fleet-routing baselines.

``shortest_path_fleet`` is the interdiction-unaware planner: every convoy takes the cheapest
route, so they stack and the interdictor ambushes the obvious road. ``alns_fleet_planner`` is an
Adaptive Large Neighbourhood Search planner (destroy and repair over the joint route assignment,
adaptive operator weights, simulated-annealing acceptance) that knows the vulnerability map and
minimises worst-case mission failure, converging to the oracle's ``loss_det``, the best any
deterministic fleet plan can achieve. ``classical_baselines`` reports each planner's
exploitability under the best-response interdictor beside the oracle references.
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
    """Exploitability of each classical planner against the best-response interdictor.

    Reported beside the oracle references. Under the mission objective the ordering is
    shortest_path >= alns == optimal_deterministic >= equilibrium.
    """
    occs, M = objective_matrix(game, N, objective, m)
    idx = {tuple(int(x) for x in o): i for i, o in enumerate(occs)}
    sp = shortest_path_fleet(game, N)
    sp_expl = float(M[idx[_occ(sp, game.n_routes)]].max())
    plan = alns_fleet_planner(game, N, objective, m, seed=seed)
    sol = solve_multiconvoy(game, N, objective, m)
    # Best deterministic STACKED plan (all N convoys on one route), worst-cased over interdiction
    # sets. ALNS is free to stack, since the occupancy set includes every stack, so its loss_det
    # is at most this value.
    forced_stack = min(float(M[idx[_occ([r] * N, game.n_routes)]].max()) for r in range(game.n_routes))
    return {"shortest_path": sp_expl, "alns": plan.exploitability,
            "alns_forced_stack": forced_stack,
            "optimal_deterministic": sol.loss_det, "equilibrium": sol.loss_mixed,
            "alns_plan": plan}
