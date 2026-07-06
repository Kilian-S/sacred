"""Network-interdiction security-game oracle (gen08 / Obj 1 + Obj 5 ground truth).

Convoy routing vs a committing interdictor as a zero-sum matrix game:
  * defender pure strategies  = candidate routes (base -> FOB), each an edge set;
  * attacker pure strategies  = K-subsets of the edges appearing on some route (the K interdiction
    assets it commits, HIDDEN, before the convoy moves);
  * payoff[i, j] = interception probability of route i under interdiction set j (1 if the route
    crosses an interdicted edge; a soft survival model is pluggable via ``intercept_fn``).

The defender MINIMISES expected interception, the attacker MAXIMISES it. This module computes:
  * ``loss_det``            = the best DETERMINISTIC defender's worst-case interception
                             (min over routes of max over interdiction sets): what shortest-path /
                             greedy / a collapsed vanilla-SAC policy is bounded by (fully exploitable);
  * ``value`` (loss_mixed)  = the minimax value: the best MIXED-strategy defender vs the
                             best-responding interdictor (what SACRED, learning the equilibrium via
                             SAC entropy + ATLA, should approach);
  * the equilibrium mixed strategies for BOTH sides (the defender's is the target SACRED is
    validated against; the attacker's is the strong "oracle interdictor" for exploitability).

It also evaluates ARBITRARY (e.g. learned) defender route-distributions: ``best_response_attacker``
and ``interception_of_distribution`` give the exploitability of any frozen policy against the
committing interdictor. This is the ground truth the gen08 experiment scores against; the LP is
tractable for the single-convoy instances and provides the yardstick where deep RL then scales past
it. First prototyped in ``scratch/interdiction_game_probe.py``.
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
    """Candidate defender routes: the edge-DISJOINT paths (the min-cut worth of genuinely
    alternative routes: the strategic diversity a convoy planner uses) plus k-shortest paths for
    realism. k-shortest alone are near-duplicates sharing the cheap chokepoint, understating the
    mixing room; the disjoint paths are what matter for interdiction (Menger: #disjoint = min-cut)."""
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
    """Given a defender distribution over ``game.routes``, the committing attacker's best interdiction
    set and the interception it achieves = the defender's EXPLOITABILITY. (The attacker commits to
    the strategy, not the realised route: the hidden-commit structure.)"""
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
# Heterogeneous edge vulnerability (soft interception): the I3 asymmetric instances.
# On edge-disjoint routes with HARD interception the equilibrium is uniquely UNIFORM for every K
# (best response = the top-K defender masses, minimised only by uniform), so vanilla's incidental
# max-entropy mixing is near-optimal and the SACRED-vs-vanilla gap is thin (the I2 caveat).
# Per-edge interception probabilities break that symmetry: for disjoint routes with per-route max
# vulnerability p_i* the equilibrium is d_i proportional to 1/p_i* with value 1/sum_i(1/p_i*),
# a NON-uniform target vanilla does not track (closed form; verified in tests).


def length_band_vulnerability(G: nx.Graph, edges: Iterable[frozenset], *,
                              band: tuple[float, float] = (0.2, 0.9),
                              weight: str = "w") -> dict[frozenset, float]:
    """Per-edge interception probability from edge length: exposure scales with transit time, so
    each candidate edge's length is mapped affinely into ``band`` (shortest edge -> band[0],
    longest -> band[1]; all-equal lengths -> the band midpoint). Objective and graph-derived (no
    hand-tuned threat map); the band itself is pinned in the ledger by an oracle probe
    (`scratch/vuln_band_probe.py`) BEFORE any training."""
    es = sorted(edges, key=repr)
    if not es:
        raise ValueError("no candidate edges to assign vulnerability to")
    lo, hi = band
    if not (0.0 < lo <= hi <= 1.0):
        raise ValueError(f"band must satisfy 0 < lo <= hi <= 1, got {band}")
    lens = {e: float(G[u][v].get(weight, 1.0)) for e in es for u, v in [tuple(e)]}
    lmin, lmax = min(lens.values()), max(lens.values())
    if lmax <= lmin:
        return {e: (lo + hi) / 2.0 for e in es}
    return {e: lo + (hi - lo) * (lens[e] - lmin) / (lmax - lmin) for e in es}


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


def route_distribution_from_first_hops(game: InterdictionGame, s: NodeId,
                                       first_hop_probs: dict[NodeId, float]) -> np.ndarray:
    """Map a distribution over the FIRST hop out of s to a distribution over routes (each route is
    credited to its first edge). For edge-disjoint routes this is exact (first hop identifies the
    route); it is how a next-hop policy's branch-at-source probabilities become a route mixture for
    comparison against the equilibrium ``defender_strategy``."""
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
