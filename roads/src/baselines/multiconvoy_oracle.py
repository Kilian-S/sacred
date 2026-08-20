"""Multi-convoy interdiction oracle (gen08 Phase M / Obj-1 + Obj-5 ground truth).

N convoys route base -> FOB against a K-asset committing interdictor. The defender chooses a JOINT
routing, represented (convoys are interchangeable) as an OCCUPANCY of convoys over the candidate
routes; the interdictor commits K edges (hidden). Generalising `interdiction_oracle` from 1 to N
convoys, this module computes:
  * ``loss_det``   = the best DETERMINISTIC joint plan, worst-cased over the attacker's response
                     (what a coordinating classical metaheuristic / ALNS produces);
  * ``loss_mixed`` = the minimax value of the best RANDOMISED joint routing (what SACRED targets);
  * the equilibrium defender (a distribution over occupancies) and attacker (over interdiction sets);
  * the exploitability of ANY (e.g. learned) defender occupancy distribution.

Two objective families (see `scratch/multiconvoy_*.py` for the finding that the OBJECTIVE decides
whether SACRED wins): LINEAR (expected fraction of convoys lost, risk-neutral) and THRESHOLD(m)
(P(>= m convoys lost)); MISSION-failure is THRESHOLD(1) = P(>= 1 convoy lost), the loss-averse and
operationally-realistic objective under which the multi-convoy gap holds and grows with N.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom

from src.baselines.interdiction_oracle import InterdictionGame


def occupancies(n_routes: int, N: int) -> list[np.ndarray]:
    """Every way to place N interchangeable convoys over ``n_routes`` routes (occupancy vectors
    summing to N)."""
    out: list[np.ndarray] = []
    for combo in itertools.combinations_with_replacement(range(n_routes), N):
        v = np.zeros(n_routes, dtype=int)
        for r in combo:
            v[r] += 1
        out.append(v)
    return out


def caught_pmf(occ: np.ndarray, p: np.ndarray) -> np.ndarray:
    """PMF of the number of convoys intercepted, given occupancy ``occ`` and per-route interception
    probabilities ``p`` under a fixed interdiction set (each convoy on route r is caught
    independently with prob p[r]; = a sum of Binomial(occ[r], p[r]) = a Poisson-binomial)."""
    pmf = np.array([1.0])
    for r in range(len(occ)):
        if occ[r] > 0:
            pmf = np.convolve(pmf, binom.pmf(np.arange(occ[r] + 1), occ[r], p[r]))
    return pmf


def objective_value(occ: np.ndarray, p: np.ndarray, N: int, objective: str = "mission", m: int = 1,
                    rho: float = 0.0) -> float:
    """Loss for occupancy ``occ`` vs interception probs ``p``: ``linear`` = E[fraction lost];
    ``mission`` = P(>=1 lost); ``threshold`` = P(>= m lost).

    ``rho`` (B4, correlated interception): a within-route common-shock mix between INDEPENDENT
    draws (rho=0, the default and the model everything else uses) and COMONOTONE draws (rho=1: the
    convoys stacked on a route are caught all-or-nothing by ONE ambush team). The expectation
    E[fraction lost] is INVARIANT to rho (linearity), so only the loss-averse objectives feel it.
    Under the mission objective, rho > 0 makes STACKING less mission-exploitable (a stacked column
    shares one shock instead of drawing one per convoy), i.e. independence is the CONSERVATIVE
    assumption for the SACRED stack (the disclosed caveat, now a tunable curve)."""
    if objective == "linear":
        return float(occ @ p) / N                      # linearity: rho-invariant
    thr = 1 if objective == "mission" else int(m)
    if rho <= 0.0:
        pmf = caught_pmf(occ, p)
    else:
        # per-route caught-count is a rho-mix of Binomial(occ_r, p_r) (indep) and a two-point
        # comonotone law (all occ_r caught w.p. p_r, else 0); routes remain independent of each
        # other (distinct edges/teams). Convolve the per-route mixed pmfs.
        pmf = np.array([1.0])
        for r in range(len(occ)):
            c = int(occ[r])
            if c == 0:
                continue
            indep = binom.pmf(np.arange(c + 1), c, p[r])
            como = np.zeros(c + 1); como[0] = 1.0 - p[r]; como[c] = p[r]
            pmf = np.convolve(pmf, (1.0 - rho) * indep + rho * como)
    return float(pmf[thr:].sum()) if thr < len(pmf) else 0.0


def objective_matrix(game: InterdictionGame, N: int, objective: str = "mission", m: int = 1):
    """(occupancies, loss matrix) with M[occ, iset] = the chosen objective's loss.

    ``mission`` and ``linear`` have closed forms and are computed as single matrix products
    (mission: M = 1 - exp(O @ log(1 - payoff)), the survival product; linear: M = O @ payoff / N),
    which is what makes the K >= 3 sweep instances buildable (the generic per-entry
    Poisson-binomial convolution at 28.8M entries costs ~half an hour; the matmul is sub-second).
    ``threshold`` with m > 1 keeps the exact convolution path. Equivalence is regression-tested
    against the loop implementation (tests/test_multiconvoy_oracle_vectorised.py)."""
    occs = occupancies(game.n_routes, N)
    O = np.asarray(occs, dtype=float)                    # [n_occ, R]
    if objective == "linear":
        return occs, (O @ game.payoff) / N
    if objective == "mission" or (objective == "threshold" and int(m) <= 1):
        # P(>=1 caught) = 1 - prod_r (1 - p[r, j])^occ[r]; p == 1 -> log(0) clamped so the
        # survival product is exactly 0 (interception certain), matching the loop semantics.
        log_surv = np.log(np.clip(1.0 - game.payoff, 1e-300, 1.0))   # [R, n_isets]
        return occs, 1.0 - np.exp(O @ log_surv)
    n_isets = game.payoff.shape[1]
    M = np.zeros((len(occs), n_isets))
    for oi, occ in enumerate(occs):
        for j in range(n_isets):
            M[oi, j] = objective_value(occ, game.payoff[:, j], N, objective, m)
    return occs, M


def _row_minimiser(M: np.ndarray) -> tuple[float, np.ndarray]:
    """Zero-sum matrix game: ROW minimises expected loss, COL maximises. Returns (value, row mixed
    strategy) via LP (same construction as interdiction_oracle)."""
    n, k = M.shape
    c = np.zeros(n + 1); c[-1] = 1.0
    A_ub = np.hstack([M.T, -np.ones((k, 1))]); b_ub = np.zeros(k)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0; b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"multi-convoy LP failed: {res.message}")
    x = np.clip(res.x[:n], 0.0, None); x = x / x.sum()
    return float(res.x[-1]), x


@dataclass(frozen=True)
class MultiConvoySolution:
    N: int
    objective: str
    loss_det: float                 # best deterministic joint plan (ALNS), worst-cased
    loss_mixed: float               # minimax randomised joint routing (SACRED target)
    defender_strategy: np.ndarray   # equilibrium distribution over occupancies
    attacker_strategy: np.ndarray   # equilibrium distribution over interdiction sets
    occupancies: tuple              # occupancy vectors, indexing defender_strategy

    @property
    def gap(self) -> float:
        return self.loss_det - self.loss_mixed


def solve_multiconvoy(game: InterdictionGame, N: int, objective: str = "mission", m: int = 1) -> MultiConvoySolution:
    occs, M = objective_matrix(game, N, objective, m)
    loss_mixed, defender = _row_minimiser(M)
    _, attacker = _row_minimiser(-M.T)              # attacker = column optimum
    loss_det = float(min(M[i].max() for i in range(M.shape[0])))
    return MultiConvoySolution(N, objective, loss_det, loss_mixed, defender, attacker,
                               tuple(tuple(int(x) for x in o) for o in occs))


def greedy_br_attacker(route_edges, edge_vuln: dict, occ_support, N: int, K: int,
                       objective: str = "mission", m: int = 1, rho: float = 0.0):
    """MATRIX-FREE best-response interdictor (A4): pick K edges greedily to maximise the defender's
    expected loss, WITHOUT enumerating the C(E, K) interdiction sets or the [occ x iset] matrix -
    the regime where the exact oracle is infeasible (K >= 4).

    For the MISSION objective, expected mission-failure of a defender occupancy distribution is a
    monotone SUBMODULAR function of the interdicted edge set (it is a weighted "at-least-one"
    coverage over the convoys' edge-crossing events), so the greedy K-edge choice carries the
    classic (1 - 1/e) approximation guarantee (verified against the exact BR at K <= 2 in
    tests/test_greedy_br.py). Cost O(E * K * |support| * R).

    Args: ``route_edges`` = tuple of per-route frozenset edge sets (game.route_edges);
    ``edge_vuln`` = {frozenset_edge: p_e}; ``occ_support`` = list of (occ_tuple, weight) with
    weights summing to 1 (the defender's play, e.g. a trailing-window occupancy histogram);
    ``rho`` = within-route interception correlation (B4). Returns (chosen_edge_frozensets, value)."""
    cand = sorted(set().union(*route_edges), key=repr)
    vuln = np.array([edge_vuln.get(e, 1.0) for e in cand])   # hard interception -> p_e = 1
    R = len(route_edges)
    # route x candidate-edge incidence (route r crosses edge c?)
    inc = np.array([[1.0 if cand[c] in route_edges[r] else 0.0 for c in range(len(cand))]
                    for r in range(R)])
    occ = np.array([o for o, _ in occ_support], dtype=float)          # [S, R]
    w = np.array([wt for _, wt in occ_support], dtype=float)          # [S]

    def value(chosen: list[int]) -> float:
        if not chosen:
            p_r = np.zeros(R)
        else:
            surv = np.prod([(1.0 - vuln[c]) ** inc[:, c] for c in chosen], axis=0)  # per route
            p_r = 1.0 - surv
        return float(sum(wt * objective_value(o, p_r, N, objective, m, rho=rho)
                         for o, wt in zip(occ, w)))

    chosen: list[int] = []
    remaining = set(range(len(cand)))
    for _ in range(min(K, len(cand))):
        best_c, best_v = None, -1.0
        for c in remaining:
            v = value(chosen + [c])
            if v > best_v + 1e-12:
                best_v, best_c = v, c
        if best_c is None:
            break
        chosen.append(best_c); remaining.discard(best_c)
    return tuple(cand[c] for c in chosen), value(chosen)


def best_response_attacker_multi(obj_matrix: np.ndarray, occupancy_dist: np.ndarray) -> tuple[int, float]:
    """The committing interdictor's best interdiction set against a defender OCCUPANCY distribution,
    and the loss it achieves = that defender's EXPLOITABILITY."""
    per_iset = np.asarray(occupancy_dist, dtype=float) @ obj_matrix
    j = int(per_iset.argmax())
    return j, float(per_iset[j])


def objective_of(obj_matrix: np.ndarray, occupancy_dist, attacker_dist) -> float:
    """Expected loss of a defender occupancy distribution vs an attacker distribution."""
    return float(np.asarray(occupancy_dist, float) @ obj_matrix @ np.asarray(attacker_dist, float))
