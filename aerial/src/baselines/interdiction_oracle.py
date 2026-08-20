"""Network-interdiction security-game oracle.

Convoy routing against a committing interdictor as a zero-sum matrix game. The defender's pure
strategies are the candidate routes from base to FOB, each an edge set; the attacker's are the
K-subsets of the edges appearing on some route, which it commits to and hides before the convoy
moves; and payoff[i, j] is the interception probability of route i under interdiction set j, hard
by default and soft through a pluggable ``intercept_fn``. The defender minimises expected
interception and the attacker maximises it, and the module returns the best deterministic
defender's worst case (``loss_det``), the minimax value of the best mixed defender (``value``) and
the equilibrium mixed strategies of both sides. ``best_response_attacker`` and
``interception_of_distribution`` extend the same scoring to any frozen defender route-distribution.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import networkx as nx
import numpy as np
from scipy.optimize import linprog

NodeId = Any
Route = tuple[NodeId, ...]


# ---------------------------------------------------------------------------
# Game construction


def edges_of_route(route: Route) -> frozenset:
    """Undirected edge set of a node path (interception is direction-agnostic)."""
    return frozenset(frozenset((route[i], route[i + 1])) for i in range(len(route) - 1))


def k_shortest_routes(G: nx.Graph, s: NodeId, t: NodeId, k: int, weight: str = "w") -> list[Route]:
    from networkx.algorithms.simple_paths import shortest_simple_paths
    out: list[Route] = []
    for i, p in enumerate(shortest_simple_paths(G, s, t, weight=weight)):
        if i >= k:
            break
        out.append(tuple(p))
    return out


def build_route_set(G: nx.Graph, s: NodeId, t: NodeId, k_extra: int = 8, weight: str = "w") -> list[Route]:
    """Candidate defender routes, the edge-disjoint paths plus the k-shortest paths.

    The k-shortest paths alone are near-duplicates sharing the cheap chokepoint and understate the
    mixing room. The disjoint paths are what matter for interdiction, since by Menger's theorem
    their number equals the min-cut.
    """
    routes: list[Route] = [tuple(p) for p in nx.edge_disjoint_paths(G, s, t)]
    seen = set(routes)
    for p in k_shortest_routes(G, s, t, k_extra, weight):
        if p not in seen:
            routes.append(p)
            seen.add(p)
    return routes


@dataclass(frozen=True)
class InterdictionGame:
    routes: tuple[Route, ...]
    route_edges: tuple[frozenset, ...]
    interdiction_sets: tuple[tuple[frozenset, ...], ...]  # each = K undirected edges
    payoff: np.ndarray            # [n_routes, n_isets] interception in [0, 1]
    travel_cost: np.ndarray       # [n_routes] route lengths (for the defender's cost term)
    K: int

    @property
    def n_routes(self) -> int:
        return len(self.routes)


def build_interdiction_game(
    G: nx.Graph, s: NodeId, t: NodeId, K: int, *, k_extra: int = 8, weight: str = "w",
    intercept_fn: Callable[[frozenset, tuple[frozenset, ...]], float] | None = None,
) -> InterdictionGame:
    """Build the K-asset interdiction game for the OD pair (s, t).

    ``intercept_fn(route_edges, iset) -> [0, 1]`` defaults to hard interception, returning 1 if the
    route crosses any interdicted edge.
    """
    routes = build_route_set(G, s, t, k_extra, weight)
    route_edges = [edges_of_route(r) for r in routes]
    cand_edges = sorted(set().union(*route_edges), key=repr)
    if K <= len(cand_edges):
        isets = [tuple(c) for c in itertools.combinations(cand_edges, K)]
    else:
        isets = [tuple(cand_edges)]
    if intercept_fn is None:
        def intercept_fn(re: frozenset, iset: tuple[frozenset, ...]) -> float:
            return 1.0 if re & set(iset) else 0.0
    payoff = np.array([[intercept_fn(re, iset) for iset in isets] for re in route_edges], dtype=float)
    travel = np.array([sum(G[u][v].get(weight, 1.0) for u, v in zip(r, r[1:])) for r in routes])
    return InterdictionGame(tuple(routes), tuple(route_edges), tuple(isets), payoff, travel, K)


# ---------------------------------------------------------------------------
# Solving


@dataclass(frozen=True)
class InterdictionSolution:
    value: float                    # loss_mixed: minimax interception probability
    loss_det: float                 # best deterministic defender's worst-case interception
    defender_strategy: np.ndarray   # equilibrium distribution over routes
    attacker_strategy: np.ndarray   # equilibrium distribution over interdiction sets

    @property
    def gap(self) -> float:
        return self.loss_det - self.value


def _row_minimiser(payoff: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve the zero-sum matrix game where the row player minimises expected payoff.

    Returns the value and the row player's mixed strategy from the LP
    min v s.t. sum_i x_i payoff[i, j] <= v for all j, sum_i x_i = 1, x >= 0.
    """
    n, m = payoff.shape
    c = np.zeros(n + 1); c[-1] = 1.0
    A_ub = np.hstack([payoff.T, -np.ones((m, 1))]); b_ub = np.zeros(m)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0; b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"interdiction LP failed: {res.message}")
    x = np.clip(res.x[:n], 0.0, None); x = x / x.sum()
    return float(res.x[-1]), x


def solve(game: InterdictionGame) -> InterdictionSolution:
    value, defender = _row_minimiser(game.payoff)
    # attacker equilibrium = the column player's optimum = row-minimiser of the negated transpose.
    neg_value, attacker = _row_minimiser(-game.payoff.T)
    loss_det = float(min(game.payoff[i].max() for i in range(game.n_routes)))
    return InterdictionSolution(value=value, loss_det=loss_det,
                                defender_strategy=defender, attacker_strategy=attacker)


# ---------------------------------------------------------------------------
# Evaluating arbitrary (learned) defender strategies -> exploitability


def best_response_attacker(game: InterdictionGame, defender_strategy: np.ndarray) -> tuple[int, float]:
    """The committing attacker's best interdiction set against a distribution over ``game.routes``.

    The interception it achieves is that defender's exploitability. The attacker responds to the
    strategy, not to the realised route, which is the hidden-commit structure.
    """
    d = np.asarray(defender_strategy, dtype=float)
    per_iset = d @ game.payoff              # expected interception for each interdiction set
    j = int(per_iset.argmax())
    return j, float(per_iset[j])


def interception_of_distribution(game: InterdictionGame, defender_strategy: np.ndarray,
                                 attacker_strategy: np.ndarray) -> float:
    """Expected interception of a defender route-distribution vs a given attacker distribution."""
    d = np.asarray(defender_strategy, float); a = np.asarray(attacker_strategy, float)
    return float(d @ game.payoff @ a)


# ---------------------------------------------------------------------------
# Heterogeneous edge vulnerability (soft interception).
# On edge-disjoint routes with hard interception the equilibrium is uniquely uniform for every K,
# because the best response takes the top-K defender masses and only a uniform mix minimises those.
# Per-edge interception probabilities break that symmetry, and for disjoint routes with per-route
# maximum vulnerability p_i* the equilibrium is d_i proportional to 1/p_i* with value
# 1/sum_i(1/p_i*), a non-uniform target.


def length_band_vulnerability(G: nx.Graph, edges: Iterable[frozenset], *,
                              band: tuple[float, float] = (0.2, 0.9),
                              weight: str = "w",
                              norm_edges: Iterable | None = None) -> dict[frozenset, float]:
    """Per-edge interception probability from edge length, mapped affinely into ``band``.

    The shortest edge maps to band[0] and the longest to band[1], with all-equal lengths mapping to
    the band midpoint. An ascending band models exposure scaling with transit time, so that
    vulnerability correlates with travel cost; a descending band (band[0] > band[1]) inverts the
    correlation, making short edges the watched chokepoints, so cost and security conflict and
    cost-driven mixing is miscalibrated by construction. The map is derived from the graph rather
    than hand-tuned.

    Args:
        norm_edges: reference edge set for the length-to-band map. None normalises over ``edges``
            themselves, which is per-instance, so the same physical road can take a different p_e
            under a different route set. Passing the whole graph's edges makes the map absolute and
            therefore comparable across OD instances; candidate lengths are then clamped into the
            reference range so p stays inside the band.
    """
    es = sorted(edges, key=repr)
    if not es:
        raise ValueError("no candidate edges to assign vulnerability to")
    lo, hi = band
    if not (0.0 < min(lo, hi) and max(lo, hi) <= 1.0):
        raise ValueError(f"band values must lie in (0, 1], got {band}")
    lens = {e: float(G[u][v].get(weight, 1.0)) for e in es for u, v in [tuple(e)]}
    ref = ([float(G[u][v].get(weight, 1.0)) for e in norm_edges for u, v in [tuple(e)]]
           if norm_edges is not None else list(lens.values()))
    lmin, lmax = min(ref), max(ref)
    if lmax <= lmin:
        return {e: (lo + hi) / 2.0 for e in es}
    return {e: lo + (hi - lo) * (min(max(lens[e], lmin), lmax) - lmin) / (lmax - lmin) for e in es}


def survival_intercept_fn(vulnerability: dict[frozenset, float]) -> Callable[[frozenset, tuple[frozenset, ...]], float]:
    """Build a soft-survival ``intercept_fn`` for ``build_interdiction_game``.

    Each interdicted edge the route crosses is survived independently with probability 1 - p_e, so
    interception is 1 - prod(1 - p_e) over the crossed interdicted edges, which reduces to p_e at
    K=1. Raises KeyError on an edge with no assigned vulnerability.
    """
    def fn(route_edges: frozenset, iset: tuple[frozenset, ...]) -> float:
        survival = 1.0
        for e in iset:
            if e in route_edges:
                survival *= 1.0 - vulnerability[e]
        return 1.0 - survival
    return fn


def cost_constrained_value(game: InterdictionGame, budget: float) -> tuple[float, np.ndarray]:
    """One point of the cost-security frontier.

    The minimax interception achievable by a mixed route strategy whose expected travel cost is at
    most ``budget``. Sweep the budget for the whole curve; a budget at or above the maximum useful
    cost reproduces the unconstrained ``solve`` value, and budgets below the cheapest route are
    infeasible.
    """
    c_min = float(game.travel_cost.min())
    if budget < c_min - 1e-12:
        raise ValueError(f"budget {budget} infeasible: the cheapest route costs {c_min}")
    n, m = game.payoff.shape
    c = np.zeros(n + 1); c[-1] = 1.0
    A_ub = np.vstack([np.hstack([game.payoff.T, -np.ones((m, 1))]),
                      np.hstack([game.travel_cost[None, :], np.zeros((1, 1))])])
    b_ub = np.concatenate([np.zeros(m), [budget]])
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0; b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"frontier LP failed: {res.message}")
    x = np.clip(res.x[:n], 0.0, None); x = x / x.sum()
    return float(res.x[-1]), x


def route_distribution_from_first_hops(game: InterdictionGame, s: NodeId,
                                       first_hop_probs: dict[NodeId, float]) -> np.ndarray:
    """Map a distribution over the first hop out of s to a distribution over routes.

    Each route is credited to its first edge, which is exact for edge-disjoint routes because the
    first hop identifies the route. This is how a next-hop policy's branch-at-source probabilities
    become a route mixture comparable with the equilibrium ``defender_strategy``.
    """
    dist = np.zeros(game.n_routes)
    groups: dict[NodeId, list[int]] = {}
    for i, r in enumerate(game.routes):
        groups.setdefault(r[1], []).append(i)
    for nxt, idxs in groups.items():
        p = first_hop_probs.get(nxt, 0.0) / len(idxs)
        for i in idxs:
            dist[i] += p
    tot = dist.sum()
    return dist / tot if tot > 0 else dist
