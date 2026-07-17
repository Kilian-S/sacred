"""gen28 game v2: smooth curvature-bounded flight routes + hazard-rate line-integral exposure
(the 'middle design', Kilian 2026-07-17: aeronautical realism WITHOUT giving up the finite
matrix game the yardstick machinery needs).

Routes are Catmull-Rom splines through lateral-offset control points at fixed depth stations
(what a waypoint autopilot actually flies), curvature-bounded (bank limit), densely sampled.
Exposure is a survival line integral: a hazard at centre c with radius r and effectiveness
p_max contributes hazard RATE lambda(s) = kappa * max(0, 1 - d(s)/r) along the flight path,
with kappa = -ln(1 - p_max) / r, so a STRAIGHT TRANSIT THROUGH THE CENTRE is intercepted with
probability EXACTLY p_max (the calibration that keeps p_max's meaning from game v1; verified
in tests). Total interception = 1 - exp(-sum over hazards of their integrals). This removes
the arc-discretisation dependence of v1's per-arc Bernoulli and makes grazing exposure
genuinely continuous.

The menu is still a FINITE screened family (lane curves at CONTINUOUS lateral offsets first =
the strengthened naive-heuristic support, then curvature-feasible diverse curves), so
`InterdictionGame`, the LP, the greedy BR and the menu-select head all apply verbatim. Each
curve carries a lattice node sequence (nearest unblocked waypoint per column) for the GNN
menu head. Obstacles are axis-aligned blocked cells today; the rejection test is a generic
point-in-rectangle check, so real terrain/building polygons are a drop-in extension (recorded
in the ledger as the staged v3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame
from src.envs.aerial_sector import SectorLattice

N_SAMPLES = 72          # spline sample points (quadrature resolution; calibration test <=1%)
KAPPA_MAX = 1.5         # max curvature (1/turn-radius) a curve may demand: the bank limit


@dataclass(frozen=True)
class CurveRoute:
    pts: np.ndarray          # [S, 2] sampled points (x=depth, y=lateral)
    length: float
    node_seq: tuple          # nearest unblocked lattice node per column (the GNN projection)
    offsets: tuple           # the control offsets that generated it (provenance/dedup)


def _catmull_rom(ctrl: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Centripetal-flavoured Catmull-Rom through ctrl [M, 2], endpoint-clamped, n samples."""
    P = np.vstack([ctrl[0], ctrl, ctrl[-1]])
    segs = len(ctrl) - 1
    ts = np.linspace(0.0, segs, n)
    out = np.empty((n, 2))
    for k, t in enumerate(ts):
        i = min(int(t), segs - 1)
        u = t - i
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out[k] = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
    return out


def _max_curvature(pts: np.ndarray) -> float:
    """Discrete curvature |d theta / d s| over the sampled polyline."""
    d = np.diff(pts, axis=0)
    ds = np.linalg.norm(d, axis=1)
    theta = np.arctan2(d[:, 1], d[:, 0])
    dtheta = np.abs(np.diff(theta))
    seg = (ds[:-1] + ds[1:]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(seg > 1e-9, dtheta / seg, 0.0)
    return float(k.max()) if len(k) else 0.0


def _blocked_rects(lat: SectorLattice) -> np.ndarray:
    """Blocked waypoints as axis-aligned unit cells [(x0, y0, x1, y1)]. The generic obstacle
    representation: terrain/building polygons later reduce to the same hit test."""
    return np.array([[i - 0.5, j - 0.5, i + 0.5, j + 0.5] for i, j in sorted(lat.blocked)]
                    ) if lat.blocked else np.empty((0, 4))


def _hits_obstacle(pts: np.ndarray, rects: np.ndarray) -> bool:
    if not len(rects):
        return False
    for x0, y0, x1, y1 in rects:
        if np.any((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)):
            return True
    return False


def make_curve(lat: SectorLattice, offsets, rects: np.ndarray | None = None,
               kappa_max: float = KAPPA_MAX) -> CurveRoute | None:
    """A route from interior-station lateral offsets (stations every 2 depth units, endpoints
    pinned to base/target). None if it leaves the sector, exceeds the bank limit, or crosses
    an obstacle."""
    xs = np.arange(0, lat.nx, 2, dtype=float)              # 0, 2, ..., 12
    ys = np.concatenate([[lat.base[1]], np.asarray(offsets, float), [lat.target[1]]])
    if len(ys) != len(xs):
        raise ValueError(f"need {len(xs) - 2} interior offsets, got {len(offsets)}")
    if np.any(ys < 0) or np.any(ys > lat.ny - 1):
        return None
    pts = _catmull_rom(np.stack([xs, ys], axis=1))
    if np.any(pts[:, 1] < -0.25) or np.any(pts[:, 1] > lat.ny - 0.75):
        return None
    if _max_curvature(pts) > kappa_max:
        return None
    if _hits_obstacle(pts, rects if rects is not None else _blocked_rects(lat)):
        return None
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    node_seq = []
    for i in range(lat.nx):
        y = float(np.interp(i, pts[:, 0], pts[:, 1]))
        rows = sorted((abs(j - y), j) for j in range(lat.ny) if (i, j) not in lat.blocked)
        node_seq.append((i, rows[0][1]))
    return CurveRoute(pts=pts, length=float(seglen.sum()), node_seq=tuple(node_seq),
                      offsets=tuple(round(float(o), 3) for o in offsets))


def lane_offsets(lat: SectorLattice, r: float) -> list[float]:
    """CONTINUOUS lane offsets: n = floor(W / 2r) + 1 evenly spaced lateral positions (the
    strengthened naive rule: the continuum lets lanes always space optimally)."""
    n = min(int(lat.W / (2.0 * r) + 1e-9) + 1, 9)
    return [float(v) for v in np.linspace(0.0, lat.W, n)]


def lane_curve(lat: SectorLattice, offset: float, rects: np.ndarray | None = None
               ) -> CurveRoute | None:
    """The canonical lane at a continuous lateral offset: transition out over two stations,
    hold the lane, transition back (the smooth analogue of v1's lane path)."""
    mid = float(lat.base[1])
    o = [mid + (offset - mid) * 0.65] + [offset] * 3 + [mid + (offset - mid) * 0.65]
    return make_curve(lat, o, rects)


def build_curve_menu(lat: SectorLattice, r: float, R: int = 40, seed: int = 0
                     ) -> tuple[list[CurveRoute], list[int]]:
    """The finite route family: lane curves FIRST (this instance's naive-rule support), the
    straight centre line, then curvature-feasible diverse curves (seeded, deterministic;
    greedy max-min selection in offset space). Returns (menu, lane_indices)."""
    rects = _blocked_rects(lat)
    menu: list[CurveRoute] = []
    seen: set[tuple] = set()

    def add(c: CurveRoute | None) -> bool:
        if c is None or c.offsets in seen:
            return False
        menu.append(c)
        seen.add(c.offsets)
        return True

    lane_idx: list[int] = []
    for off in lane_offsets(lat, r):
        if add(lane_curve(lat, off, rects)):
            lane_idx.append(len(menu) - 1)
    add(make_curve(lat, [float(lat.base[1])] * 5, rects))       # straight centre line
    rng = np.random.default_rng(seed)
    cands: list[CurveRoute] = []
    tries = 0
    while len(cands) < 6 * R and tries < 60 * R:
        tries += 1
        c = make_curve(lat, rng.uniform(0.0, lat.W, size=5), rects)
        if c is not None and c.offsets not in seen:
            cands.append(c)
    while len(menu) < R and cands:                              # greedy max-min diversity
        chosen_off = np.array([c.offsets for c in menu])
        d = [float(np.min(np.linalg.norm(chosen_off - np.array(c.offsets), axis=1)))
             for c in cands]
        best = int(np.argmax(d))
        add(cands.pop(best))
    return menu, lane_idx


# ---------------------------------------------------------------------------
# Line-integral exposure and the game


def curve_survival_matrix(menu: list[CurveRoute], centres: np.ndarray, r: float,
                          p_max) -> np.ndarray:
    """S[i, h] = exp(-integral of hazard rate along curve i for hazard h alone), with
    kappa_h = -ln(1 - p_max_h) / r (dead-centre straight transit intercepted w.p. p_max_h)."""
    pm = np.broadcast_to(np.asarray(p_max, float), (len(centres),))
    kappa = -np.log(np.clip(1.0 - pm, 1e-12, 1.0)) / r
    S = np.empty((len(menu), len(centres)))
    for i, c in enumerate(menu):
        mids = (c.pts[:-1] + c.pts[1:]) / 2.0
        ds = np.linalg.norm(np.diff(c.pts, axis=0), axis=1)
        d = np.linalg.norm(mids[:, None, :] - centres[None, :, :], axis=2)   # [S-1, H]
        taper = np.clip(1.0 - d / r, 0.0, None)
        S[i] = np.exp(-(kappa[None, :] * taper * ds[:, None]).sum(axis=0))
    return S


def dense_hazard_grid(lat: SectorLattice, step: float = 0.5) -> np.ndarray:
    """Candidate hazard centres on a dense grid over the interior (x in [2, nx-3]), excluding
    points inside obstacles. step=0.5 is the v2 default; the convergence row sweeps step."""
    xs = np.arange(2.0, lat.nx - 2 + 1e-9, step)
    ys = np.arange(0.0, lat.ny - 1 + 1e-9, step)
    rects = _blocked_rects(lat)
    pts = np.array([[x, y] for x in xs for y in ys])
    if len(rects):
        keep = [not _hits_obstacle(p[None, :], rects) for p in pts]
        pts = pts[keep]
    return pts


def build_curved_game(lat: SectorLattice, menu: list[CurveRoute], centres: np.ndarray,
                      K: int, *, r: float, p_max=0.9) -> tuple[InterdictionGame, np.ndarray]:
    """The K-hazard game over the curved menu; returns (game, S). Exact only while the iset
    enumeration fits (K <= 2 on dense grids); past that use greedy_br_hazards on S."""
    import itertools
    S = curve_survival_matrix(menu, centres, r, p_max)
    logS = np.log(np.clip(S, 1e-300, 1.0))
    H = len(centres)
    isets = list(itertools.combinations(range(H), K)) if K <= H else [tuple(range(H))]
    if len(menu) * len(isets) > 60_000_000:
        raise MemoryError(f"exact matrix {len(menu)} x {len(isets)}: use the greedy yardstick")
    idx = np.asarray(isets, dtype=int)
    payoff = 1.0 - np.exp(logS[:, idx].sum(axis=2))
    route_edges = tuple(frozenset(frozenset((a, b)) for a, b in zip(c.node_seq, c.node_seq[1:]))
                        for c in menu)
    travel = np.array([c.length for c in menu])
    return InterdictionGame(tuple(c.node_seq for c in menu), route_edges,
                            tuple(tuple(t) for t in isets), payoff, travel, K), S
