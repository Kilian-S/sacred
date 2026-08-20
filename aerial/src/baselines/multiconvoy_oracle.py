"""Multi-convoy interdiction oracle, generalising `interdiction_oracle` from one to N convoys.

N convoys route base -> FOB against a K-asset committing interdictor. The defender chooses a joint
routing, represented (convoys being interchangeable) as an occupancy of convoys over the candidate
routes, while the interdictor commits K hidden edges. The module computes ``loss_det``, the best
deterministic joint plan worst-cased over the attacker's response, ``loss_mixed``, the minimax
value of the best randomised joint routing, the equilibrium defender and attacker strategies, and
the exploitability of any given defender occupancy distribution. Two objective families are
supported, linear (expected fraction of convoys lost, risk-neutral) and threshold(m) (P(>= m
convoys lost)); mission failure is threshold(1), the loss-averse objective under which the
multi-convoy gap holds and grows with N.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom

from src.baselines.interdiction_oracle import InterdictionGame


def occupancies(n_routes: int, N: int) -> list[np.ndarray]:
    """Every way to place N interchangeable convoys over ``n_routes`` routes.

    Each entry is an occupancy vector summing to N.
    """
    out: list[np.ndarray] = []
    for combo in itertools.combinations_with_replacement(range(n_routes), N):
        v = np.zeros(n_routes, dtype=int)
        for r in combo:
            v[r] += 1
        out.append(v)
    return out


def caught_pmf(occ: np.ndarray, p: np.ndarray) -> np.ndarray:
    """PMF of the number of convoys intercepted under a fixed interdiction set.

    Each convoy on route r is caught independently with probability p[r], so the count is a sum of
    Binomial(occ[r], p[r]) terms, that is a Poisson binomial.
    """
    pmf = np.array([1.0])
    for r in range(len(occ)):
        if occ[r] > 0:
            pmf = np.convolve(pmf, binom.pmf(np.arange(occ[r] + 1), occ[r], p[r]))
    return pmf


def objective_value(occ: np.ndarray, p: np.ndarray, N: int, objective: str = "mission", m: int = 1,
                    rho: float = 0.0) -> float:
    """Loss for occupancy ``occ`` against per-route interception probabilities ``p``.

    ``linear`` is E[fraction lost], ``mission`` is P(>= 1 lost) and ``threshold`` is P(>= m lost).
    ``rho`` is a within-route common-shock mix between independent draws (rho=0, the default) and
    comonotone draws (rho=1, where the convoys stacked on a route are caught all-or-nothing by one
    ambush team). E[fraction lost] is invariant to rho by linearity, so only the loss-averse
    objectives feel it, and rho > 0 makes stacking less mission-exploitable, which makes
    independence the conservative assumption.
    """
    if objective == "linear":
        return float(occ @ p) / N                      # linearity: rho-invariant
    thr = 1 if objective == "mission" else int(m)
    if rho <= 0.0:
        pmf = caught_pmf(occ, p)
    else:
        # The per-route caught count is a rho-mix of Binomial(occ_r, p_r) and a two-point
        # comonotone law (all occ_r caught with probability p_r, else none). Routes stay
        # independent of each other, so the per-route mixed pmfs convolve.
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
    """The occupancies and the loss matrix, with M[occ, iset] the chosen objective's loss.

    ``mission`` and ``linear`` have closed forms computed as single matrix products, mission as
    M = 1 - exp(O @ log(1 - payoff)) and linear as M = O @ payoff / N, which is what makes the
    larger K instances buildable at all; the generic per-entry Poisson-binomial convolution costs
    minutes where the matmul costs under a second. ``threshold`` with m > 1 keeps the exact
    convolution path.
    """
    occs = occupancies(game.n_routes, N)
    O = np.asarray(occs, dtype=float)                    # [n_occ, R]
    if objective == "linear":
        return occs, (O @ game.payoff) / N
    if objective == "mission" or (objective == "threshold" and int(m) <= 1):
        # P(>= 1 caught) = 1 - prod_r (1 - p[r, j])^occ[r]. At p == 1 the log is clamped so the
        # survival product is exactly 0, matching the loop semantics.
        log_surv = np.log(np.clip(1.0 - game.payoff, 1e-300, 1.0))   # [R, n_isets]
        return occs, 1.0 - np.exp(O @ log_surv)
    n_isets = game.payoff.shape[1]
    M = np.zeros((len(occs), n_isets))
    for oi, occ in enumerate(occs):
        for j in range(n_isets):
            M[oi, j] = objective_value(occ, game.payoff[:, j], N, objective, m)
    return occs, M


def _row_minimiser(M: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve the zero-sum matrix game where the row player minimises expected loss.

    Returns the value and the row player's mixed strategy from the LP.
    """
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
    loss_det: float                 # best deterministic joint plan, worst-cased
    loss_mixed: float               # minimax value of the best randomised joint routing
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
    """Matrix-free best-response interdictor: greedily pick K edges maximising the defender's loss.

    Nothing here enumerates the C(E, K) interdiction sets or the [occ x iset] matrix, which is what
    keeps the regime where the exact oracle is infeasible (K >= 4) reachable. Under the mission
    objective the expected failure of a defender occupancy distribution is a monotone submodular
    function of the interdicted edge set, being a weighted at-least-one coverage over the convoys'
    edge-crossing events, so the greedy choice carries the classic (1 - 1/e) guarantee. Costs
    O(E * K * |support| * R).

    Args:
        route_edges: per-route frozenset edge sets.
        edge_vuln: {frozenset_edge: p_e}.
        occ_support: (occ_tuple, weight) pairs with weights summing to 1, the defender's play.
        rho: within-route interception correlation.

    Returns:
        The chosen edge frozensets and the value they achieve.
    """
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
    """The committing interdictor's best set against a defender occupancy distribution.

    The loss it achieves is that defender's exploitability.
    """
    per_iset = np.asarray(occupancy_dist, dtype=float) @ obj_matrix
    j = int(per_iset.argmax())
    return j, float(per_iset[j])


def objective_of(obj_matrix: np.ndarray, occupancy_dist, attacker_dist) -> float:
    """Expected loss of a defender occupancy distribution vs an attacker distribution."""
    return float(np.asarray(occupancy_dist, float) @ obj_matrix @ np.asarray(attacker_dist, float))
