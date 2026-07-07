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


def objective_value(occ: np.ndarray, p: np.ndarray, N: int, objective: str = "mission", m: int = 1) -> float:
    """Loss for occupancy ``occ`` vs interception probs ``p``: ``linear`` = E[fraction lost];
    ``mission`` = P(>=1 lost); ``threshold`` = P(>= m lost)."""
    if objective == "linear":
        return float(occ @ p) / N                      # linearity: expectation needs no pmf
    thr = 1 if objective == "mission" else int(m)
    pmf = caught_pmf(occ, p)
    return float(pmf[thr:].sum()) if thr < len(pmf) else 0.0


def objective_matrix(game: InterdictionGame, N: int, objective: str = "mission", m: int = 1):
    """(occupancies, loss matrix) with M[occ, iset] = the chosen objective's loss."""
    occs = occupancies(game.n_routes, N)
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


def best_response_attacker_multi(obj_matrix: np.ndarray, occupancy_dist: np.ndarray) -> tuple[int, float]:
    """The committing interdictor's best interdiction set against a defender OCCUPANCY distribution,
    and the loss it achieves = that defender's EXPLOITABILITY."""
    per_iset = np.asarray(occupancy_dist, dtype=float) @ obj_matrix
    j = int(per_iset.argmax())
    return j, float(per_iset[j])


def objective_of(obj_matrix: np.ndarray, occupancy_dist, attacker_dist) -> float:
    """Expected loss of a defender occupancy distribution vs an attacker distribution."""
    return float(np.asarray(occupancy_dist, float) @ obj_matrix @ np.asarray(attacker_dist, float))
