"""Aerial free-flight interdiction sector (gen28).

A UAV crosses a 2D sector discretised as a waypoint lattice with forward-progress arcs only, so
the route set is finite and enumerable. The interdictor commits K hidden hazard centres from a
candidate grid; arcs within a hazard's effective radius r carry a proximity-graded interception
probability, survived independently, so route interception is a union-of-events coverage
objective. The game is built as an ``InterdictionGame`` (interdiction sets are K-tuples of
hazard-centre indices), so the road solver machinery applies unchanged. Observable weather cells
are pure detour cost and never touch the interception payoff.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame

Node = tuple[int, int]          # (col i = depth 0..nx-1, row j = lateral 0..ny-1)
Path = tuple[Node, ...]

_SQRT2 = float(np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Lattice


@dataclass(frozen=True)
class SectorLattice:
    ny: int                     # lateral rows (W = ny - 1)
    nx: int                     # depth columns (D = nx - 1)
    blocked: frozenset = field(default_factory=frozenset)   # blocked waypoints (pinch geometry)

    @property
    def W(self) -> float:
        return float(self.ny - 1)

    @property
    def base(self) -> Node:
        return (0, (self.ny - 1) // 2)

    @property
    def target(self) -> Node:
        return (self.nx - 1, (self.ny - 1) // 2)

    def nodes(self) -> list[Node]:
        return [(i, j) for i in range(self.nx) for j in range(self.ny)
                if (i, j) not in self.blocked]

    def graph(self) -> nx.DiGraph:
        """Forward-progress DAG with arc-length weights 'w' (1 forward, sqrt(2) diagonal)."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes())
        for i in range(self.nx - 1):
            for j in range(self.ny):
                if (i, j) in self.blocked:
                    continue
                for dj in (-1, 0, 1):
                    jj = j + dj
                    if 0 <= jj < self.ny and (i + 1, jj) not in self.blocked:
                        G.add_edge((i, j), (i + 1, jj), w=1.0 if dj == 0 else _SQRT2)
        return G


def path_length(path: Path) -> float:
    return float(sum(1.0 if b[1] == a[1] else _SQRT2 for a, b in zip(path, path[1:])))


def arc_midpoints(path: Path) -> np.ndarray:
    """[n_arcs, 2] midpoints in (x=depth, y=lateral) coordinates."""
    pts = np.asarray(path, dtype=float)
    return (pts[:-1] + pts[1:]) / 2.0


def lane_path(lat: SectorLattice, row: int) -> Path | None:
    """The canonical lane path via row ``row``: diagonal out as early as possible, straight along
    the lane, diagonal back as late as possible. None if it would cross a blocked waypoint or the
    depth cannot accommodate the excursion."""
    mid = lat.base[1]
    off = abs(row - mid)
    if 2 * off > lat.nx - 1:
        return None
    nodes: list[Node] = []
    for i in range(lat.nx):
        j = mid + int(np.sign(row - mid)) * min(off, i, lat.nx - 1 - i)
        nodes.append((i, j))
    if any(n in lat.blocked for n in nodes):
        return None
    return tuple(nodes)


def build_aerial_menu(lat: SectorLattice, R: int = 40) -> list[Path]:
    """Candidate route menu: the lane paths first (one per reachable row), then
    k-shortest-by-length padding up to ``R`` routes."""
    menu: list[Path] = []
    seen: set[Path] = set()
    for row in range(lat.ny):
        p = lane_path(lat, row)
        if p is not None and p not in seen:
            menu.append(p)
            seen.add(p)
    if len(menu) < R:
        from networkx.algorithms.simple_paths import shortest_simple_paths
        for p in shortest_simple_paths(lat.graph(), lat.base, lat.target, weight="w"):
            tp = tuple(p)
            if tp not in seen:
                menu.append(tp)
                seen.add(tp)
            if len(menu) >= R:
                break
    return menu


# ---------------------------------------------------------------------------
# Hazards and proximity-graded risk


def hazard_grid(lat: SectorLattice, cols: tuple[int, ...] | None = None,
                rows: tuple[int, ...] | None = None) -> np.ndarray:
    """Candidate hazard-centre positions [H, 2] in (x, y). Defaults: interior columns
    {2, 4, ..., nx-3} x all rows. Centres on blocked waypoints are excluded."""
    if cols is None:
        cols = tuple(range(2, lat.nx - 2, 2))
    if rows is None:
        rows = tuple(range(lat.ny))
    pts = [(float(i), float(j)) for i in cols for j in rows if (i, j) not in lat.blocked]
    return np.asarray(pts, dtype=float)


def arc_hazard_prob(mids: np.ndarray, centres: np.ndarray, r: float,
                    p_max: float | np.ndarray, taper: str = "linear") -> np.ndarray:
    """[n_arcs, H] per-(arc, hazard) interception probability.

    ``linear``: p = p_max * max(0, 1 - d/r). ``gauss``: p = p_max * exp(-d^2 / (2 sigma^2)) with
    sigma = r/2, truncated to zero beyond r. ``p_max`` may be scalar or a per-position array [H]
    for heterogeneous hazard effectiveness.
    """
    d = np.linalg.norm(mids[:, None, :] - centres[None, :, :], axis=2)
    pm = np.asarray(p_max, dtype=float)                     # scalar or [H]; broadcasts over arcs
    if taper == "linear":
        return pm * np.clip(1.0 - d / r, 0.0, None)
    if taper == "gauss":
        sigma = r / 2.0
        return np.where(d <= r, pm * np.exp(-d ** 2 / (2.0 * sigma ** 2)), 0.0)
    raise ValueError(f"unknown taper {taper!r}")


def banded_pmax(centres: np.ndarray, ny: int, band: tuple[float, float] = (0.5, 0.95)
                ) -> np.ndarray:
    """Per-position hazard effectiveness: an affine map of the centre's lateral position into
    ``band`` (row 0 -> band[1], row ny-1 -> band[0])."""
    lo, hi = band
    y = centres[:, 1]
    return hi - (hi - lo) * y / float(ny - 1)


def route_survival_matrix(menu: list[Path], centres: np.ndarray, r: float,
                          p_max: float | np.ndarray, taper: str = "linear") -> np.ndarray:
    """S[i, h] = P(route i survives hazard h alone) = prod over arcs of (1 - p(arc, h))."""
    S = np.empty((len(menu), len(centres)))
    for i, path in enumerate(menu):
        p = arc_hazard_prob(arc_midpoints(path), centres, r, p_max, taper)
        S[i] = np.prod(1.0 - p, axis=0)
    return S


def weather_cost_penalty(menu: list[Path], cells: list[tuple[tuple[float, float], float, float]],
                         taper: str = "linear") -> np.ndarray:
    """Observable-weather detour cost per route: sum over arcs and cells of
    severity * proximity taper (pure cost, never interception). ``cells`` =
    [((x, y), radius, severity), ...]."""
    if not cells:
        return np.zeros(len(menu))
    centres = np.asarray([c for c, _, _ in cells], dtype=float)
    out = np.zeros(len(menu))
    for i, path in enumerate(menu):
        mids = arc_midpoints(path)
        for k, (_, rad, sev) in enumerate(cells):
            p = arc_hazard_prob(mids, centres[k:k + 1], rad, 1.0, taper)
            out[i] += sev * float(p.sum())
    return out


# ---------------------------------------------------------------------------
# Game construction (an InterdictionGame over hazard-centre K-tuples)

_MAX_PAYOFF_ENTRIES = 60_000_000  # exact-matrix guard; past this, use the greedy yardstick


def build_aerial_game(lat: SectorLattice, menu: list[Path], centres: np.ndarray, K: int, *,
                      r: float, p_max: float | np.ndarray = 0.9, taper: str = "linear",
                      weather: list[tuple[tuple[float, float], float, float]] | None = None,
                      ) -> InterdictionGame:
    """The K-hazard aerial interdiction game. ``interdiction_sets`` holds K-tuples of centre
    INDICES into ``centres`` (the LP/BR machinery never inspects their contents). Payoff:
    intercept(route, iset) = 1 - prod_{h in iset} S[route, h]."""
    S = route_survival_matrix(menu, centres, r, p_max, taper)
    logS = np.log(np.clip(S, 1e-300, 1.0))
    H = len(centres)
    isets = list(itertools.combinations(range(H), K)) if K <= H else [tuple(range(H))]
    if len(menu) * len(isets) > _MAX_PAYOFF_ENTRIES:
        raise MemoryError(
            f"exact payoff matrix {len(menu)} x {len(isets)} exceeds the guard; "
            f"use greedy_br_hazards (the certified yardstick) at this K")
    idx = np.asarray(isets, dtype=int)                       # [n_isets, K]
    payoff = 1.0 - np.exp(logS[:, idx].sum(axis=2))          # [R, n_isets]
    route_edges = tuple(
        frozenset(frozenset((a, b)) for a, b in zip(p, p[1:])) for p in menu)
    travel = np.array([path_length(p) for p in menu])
    travel = travel + weather_cost_penalty(menu, weather or [], taper)
    return InterdictionGame(tuple(menu), route_edges, tuple(tuple(t) for t in isets),
                            payoff, travel, K)


def coverage_fraction(K: int, r: float, W: float) -> float:
    """phi = 2 K r / W: the continuous coverage boundary parameter."""
    return 2.0 * K * r / W


def greedy_br_hazards(S: np.ndarray, defender: np.ndarray, K: int) -> tuple[tuple[int, ...], float]:
    """Matrix-free greedy best-response interdictor over hazard centres.

    Expected interception of the defender's route mixture is monotone submodular in the hazard
    set (independent-survival coverage), so greedy carries the (1 - 1/e) guarantee. Cost
    O(K * H * R).
    """
    d = np.asarray(defender, dtype=float)
    logS = np.log(np.clip(S, 1e-300, 1.0))
    cur = np.zeros(S.shape[0])                    # summed log-survival of chosen set, per route
    chosen: list[int] = []
    for _ in range(min(K, S.shape[1])):
        vals = 1.0 - d @ np.exp(cur[:, None] + logS)          # value of adding each candidate
        if chosen:
            vals[np.asarray(chosen)] = -np.inf
        c = int(vals.argmax())
        chosen.append(c)
        cur = cur + logS[:, c]
    return tuple(chosen), float(1.0 - d @ np.exp(cur))


def solve_cost_weighted(game: InterdictionGame, lam: float):
    """Defender minimises worst-case interception + lam * expected detour premium, attacker
    best-responds to interception.

    The premium is dimensionless: E[cost]/c_min - 1.

    Returns:
        (worst_interception, expected_cost, strategy).
    """
    from scipy.optimize import linprog
    n, m = game.payoff.shape
    c_min = float(game.travel_cost.min())
    prem = game.travel_cost / c_min - 1.0
    c = np.concatenate([lam * prem, [1.0]])
    A_ub = np.hstack([game.payoff.T, -np.ones((m, 1))])
    b_ub = np.zeros(m)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([1.0]),
                  bounds=[(0.0, 1.0)] * n + [(None, None)], method="highs")
    if not res.success:
        raise RuntimeError(f"cost-weighted aerial LP failed: {res.message}")
    x = np.clip(res.x[:n], 0.0, None); x = x / x.sum()
    worst = float((x @ game.payoff).max())
    return worst, float(x @ game.travel_cost), x
