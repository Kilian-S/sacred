"""Network-interdiction security-game oracle: convoy routing against a committing interdictor,
modelled as a zero-sum matrix game. Defender pure strategies are candidate routes (base -> FOB);
attacker pure strategies are K-edge interdiction sets committed before the convoy moves; the
payoff is the route's interception probability under that set (hard interception, or a pluggable
soft survival model via ``intercept_fn``). The defender minimises expected interception, the
attacker maximises it. This module computes the best deterministic defender's worst-case
interception (``loss_det``), the mixed-strategy minimax value (``value`` / ``loss_mixed``) and
equilibrium strategies for both sides, and, via ``best_response_attacker`` and
``interception_of_distribution``, the exploitability of an arbitrary (e.g. learned) defender
route distribution.
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
    """Candidate defender routes: the edge-disjoint paths (Menger: their count equals the min-cut)
    plus k-shortest paths for realism. k-shortest alone tend to be near-duplicates sharing the
    cheap chokepoint, understating the mixing room."""
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
    """Build the K-asset interdiction game for OD (s, t). ``intercept_fn(route_edges, iset) -> [0,1]``
    defaults to hard interception (1 if the route crosses any interdicted edge)."""
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
    defender_strategy: np.ndarray   # equilibrium distribution over routes (the SACRED target)
    attacker_strategy: np.ndarray   # equilibrium distribution over interdiction sets (oracle attacker)

    @property
    def gap(self) -> float:
        return self.loss_det - self.value


def _row_minimiser(payoff: np.ndarray) -> tuple[float, np.ndarray]:
    """Zero-sum matrix game: ROW minimises expected payoff, COL maximises. Returns (value, row mixed
    strategy) via LP: min v s.t. sum_i x_i payoff[i,j] <= v for all j; sum x_i = 1; x >= 0."""
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
    """Best-responding interdiction set against a defender route distribution, and the
    interception it achieves (the defender's exploitability)."""
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
# On edge-disjoint routes with hard interception the equilibrium is uniquely uniform for every K
# (best response picks the top-K defender masses, minimised only by uniform mixing). Per-edge
# interception probabilities break that symmetry: for disjoint routes with per-route max
# vulnerability p_i* the equilibrium is d_i proportional to 1/p_i*, with value 1/sum_i(1/p_i*)
# (closed form; verified in tests).


def length_band_vulnerability(G: nx.Graph, edges: Iterable[frozenset], *,
                              band: tuple[float, float] = (0.2, 0.9),
                              weight: str = "w",
                              norm_edges: Iterable | None = None) -> dict[frozenset, float]:
    """Per-edge interception probability from edge length, mapped affinely into ``band`` (shortest
    edge -> band[0], longest -> band[1]; all-equal lengths -> the band midpoint). An ascending
    band models exposure scaling with transit time (longer edges more dangerous); a descending
    band (band[0] > band[1]) puts vulnerability on short edges instead, so cost and security
    conflict by construction.

    ``norm_edges`` sets the reference edge set for the length -> band affine map. None (default)
    normalises over ``edges`` themselves (per-instance: the same physical road can get a different
    p_e under a different route set). Passing the whole graph's edges makes the map absolute (a
    road's vulnerability is stable across OD instances); candidate lengths are clamped into the
    reference range so p stays inside the band."""
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
    """``intercept_fn`` for ``build_interdiction_game``: each interdicted edge the route crosses is
    survived independently with probability 1 - p_e, so interception = 1 - prod(1 - p_e) over the
    crossed interdicted edges (reduces to p_e for K=1). Raises KeyError on an edge without an
    assigned vulnerability (the factory guarantees coverage of all candidate edges)."""
    def fn(route_edges: frozenset, iset: tuple[frozenset, ...]) -> float:
        survival = 1.0
        for e in iset:
            if e in route_edges:
                survival *= 1.0 - vulnerability[e]
        return 1.0 - survival
    return fn


def cost_constrained_value(game: InterdictionGame, budget: float) -> tuple[float, np.ndarray]:
    """One point on the cost-security frontier: the minimax interception achievable by a mixed
    route strategy whose expected travel cost is at most ``budget`` (sweeping budget traces the
    curve; budget above the max useful cost reproduces the unconstrained ``solve`` value). Raises
    if ``budget`` is below the cheapest route's cost."""
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
    """Map a distribution over the first hop out of ``s`` to a distribution over routes (each
    route credited to its first edge). Exact for edge-disjoint routes, since the first hop
    identifies the route; used to turn a next-hop policy's branch probabilities into a route
    mixture comparable against the equilibrium ``defender_strategy``."""
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
