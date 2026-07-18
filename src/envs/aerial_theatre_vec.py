"""gen28 v3-theatre CONTINUOUS env (Kilian 2026-07-18: no rasterising).

Terrain stays as the real OSM POLYGONS (km coords, `scratch/fetch_theatre_vector.py`). The game
is fully continuous: smooth curvature-free Catmull-Rom flight LANES base->target across the
corridor width; hazard sites sampled in continuous space and KEPT only where terrain permits
emplacement (point-in-polygon), with terrain-set radius/effectiveness; exposure = the continuous
line integral of hazard rate along a lane (dead-centre calibration exact), with dense URBAN
polygons MASKING line of sight (segment hazard->arc crosses an urban polygon => no engagement).
Map detail is decoupled from training: the policy reads per-route features + a route-vertex
graph, never pixels, so the terrain can be arbitrarily detailed at zero training cost.

Built as an `InterdictionGame` so the LP / greedy BR / fleet mission oracle apply verbatim.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from shapely.ops import unary_union

from src.baselines.interdiction_oracle import InterdictionGame

# terrain -> (emplaceable, radius_km, p_max, blocks_LOS)
TERRAIN = {
    "open":   dict(emplace=True,  r_km=2.5, p_max=0.90, los=False),
    "field":  dict(emplace=True,  r_km=2.5, p_max=0.90, los=False),
    "forest": dict(emplace=True,  r_km=1.2, p_max=0.92, los=True),
    "urban":  dict(emplace=False, r_km=0.0, p_max=0.00, los=True),
    "water":  dict(emplace=False, r_km=0.0, p_max=0.00, los=False),
}
PRIORITY = ["water", "urban", "forest", "field"]


@dataclass
class VecTheatre:
    name: str
    W: float
    H: float
    base: np.ndarray            # (x_km, y_km)
    target: np.ndarray
    polys: dict                 # class -> list[Polygon]
    _union: dict                # class -> unary_union
    _tree: dict                 # class -> STRtree
    _urban_union: object

    def classify(self, xy) -> str:
        p = Point(float(xy[0]), float(xy[1]))
        for cls in PRIORITY:
            u = self._union.get(cls)
            if u is not None and not u.is_empty and u.contains(p):
                return cls
        return "open"


def load_vec_theatre(path: str) -> VecTheatre:
    d = json.load(open(path))
    polys, union, tree = {}, {}, {}
    for cls, rings in d["classes"].items():
        ps = [Polygon(r) for r in rings if len(r) >= 4]
        ps = [p if p.is_valid else p.buffer(0) for p in ps]
        polys[cls] = ps
        union[cls] = unary_union(ps) if ps else None
        tree[cls] = STRtree(ps) if ps else None
    return VecTheatre(d["name"], d["W_km"], d["H_km"],
                      np.array(d["base"]["xy_km"]), np.array(d["target"]["xy_km"]),
                      polys, union, tree, union.get("urban"))


# ---------------------------------------------------------------------------
# continuous lanes


def _axis(th: VecTheatre):
    v = th.target - th.base
    u = v / (np.linalg.norm(v) + 1e-9)
    return u, np.array([-u[1], u[0]])


def _catmull(ctrl: np.ndarray, per_km: float = 2.0) -> np.ndarray:
    """Catmull-Rom through control points, sampled ~per_km points per km of chord."""
    P = np.vstack([ctrl[0], ctrl, ctrl[-1]])
    segs = len(ctrl) - 1
    n = max(int(per_km * np.linalg.norm(ctrl[-1] - ctrl[0])), 40)
    out = []
    for k in range(n):
        t = k / (n - 1) * segs
        i = min(int(t), segs - 1)
        u = t - i
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    return np.array(out)


def lane(th: VecTheatre, offset_frac: float, stations: int = 6) -> np.ndarray:
    """A smooth lane base->target holding a lateral offset that ramps 0->offset*halfwidth->0
    (a hat profile), so lanes fan across the corridor and reconverge at the terminals."""
    u, nrm = _axis(th)
    span = float((th.target - th.base) @ u)
    half = 0.5 * th.H
    ctrl = []
    for s in np.linspace(0.0, 1.0, stations):
        along = th.base + s * span * u
        lat = offset_frac * half * np.sin(np.pi * s)
        pt = along + lat * nrm
        pt[0] = np.clip(pt[0], 0.4, th.W - 0.4)          # keep the lane inside the theatre
        pt[1] = np.clip(pt[1], 0.4, th.H - 0.4)
        ctrl.append(pt)
    ctrl[0] = th.base.copy()
    ctrl[-1] = th.target.copy()
    return _catmull(np.array(ctrl))


def build_menu(th: VecTheatre, R: int = 24) -> list[np.ndarray]:
    return [lane(th, float(o)) for o in np.linspace(-1.05, 1.05, R)]


# ---------------------------------------------------------------------------
# continuous hazards + exposure


def hazard_sites(th: VecTheatre, spacing_km: float = 2.0, standoff_km: float = 4.0):
    """Continuous candidate sites on emplaceable terrain, outside terminal standoff. Returns
    (coords[H,2], r_km[H], p_max[H], cls[H])."""
    xs = np.arange(1.0, th.W, spacing_km)
    ys = np.arange(1.0, th.H, spacing_km)
    coords, rr, pp, cls = [], [], [], []
    for x in xs:
        for y in ys:
            xy = np.array([x, y])
            if (np.linalg.norm(xy - th.base) < standoff_km
                    or np.linalg.norm(xy - th.target) < standoff_km):
                continue
            k = th.classify(xy)
            spec = TERRAIN[k]
            if not spec["emplace"]:
                continue
            coords.append(xy); rr.append(spec["r_km"]); pp.append(spec["p_max"]); cls.append(k)
    return np.array(coords), np.array(rr), np.array(pp), cls


def route_survival(th: VecTheatre, route: np.ndarray, coords, rr, pp, *, los: bool) -> np.ndarray:
    """S[h] = survival vs hazard h alone: exp(-integral of rate along the lane), rate =
    kappa_h * max(0, 1 - d/r_h), kappa_h = -ln(1-p_h)/r_h (dead-centre leg -> p_h). LOS-masked
    by urban polygons (a hazard cannot engage an arc if the segment crosses urban)."""
    mids = (route[:-1] + route[1:]) / 2.0
    ds = np.linalg.norm(np.diff(route, axis=0), axis=1)
    kappa = -np.log(np.clip(1.0 - pp, 1e-12, 1.0)) / np.clip(rr, 1e-9, None)
    S = np.ones(len(coords))
    urb = th._urban_union if los else None
    for h in range(len(coords)):
        d = np.linalg.norm(mids - coords[h], axis=1)
        taper = np.clip(1.0 - d / rr[h], 0.0, None)
        if urb is not None and not urb.is_empty:
            for a in np.where(taper > 0)[0]:
                if LineString([tuple(coords[h]), tuple(mids[a])]).intersects(urb):
                    taper[a] = 0.0
        S[h] = np.exp(-(kappa[h] * taper * ds).sum())
    return S


def build_theatre_game(th: VecTheatre, K: int = 1, menu_size: int = 24, spacing_km: float = 2.0,
                       standoff_km: float = 4.0, los: bool = True):
    """(InterdictionGame, menu, coords, r_km, p_max, S[R,H]) on the continuous polygon terrain.
    Route 'edges' for the InterdictionGame are consecutive rounded-waypoint pairs (the graph the
    menu head would pool); exact only while C(H,K) fits, else use greedy BR on S."""
    menu = build_menu(th, R=menu_size)
    coords, rr, pp, cls = hazard_sites(th, spacing_km=spacing_km, standoff_km=standoff_km)
    S = np.stack([route_survival(th, r_, coords, rr, pp, los=los) for r_ in menu])
    H = len(coords)
    isets = list(itertools.combinations(range(H), K)) if K <= H else [tuple(range(H))]
    logS = np.log(np.clip(S, 1e-300, 1.0))
    idx = np.asarray(isets, dtype=int)
    if len(menu) * len(isets) > 60_000_000:
        raise MemoryError("exact matrix too large; use greedy BR on S")
    payoff = 1.0 - np.exp(logS[:, idx].sum(axis=2))
    travel = np.array([np.linalg.norm(np.diff(r_, axis=0), axis=1).sum() for r_ in menu])
    # coarse waypoint tokens (0.5 km) as the route-edge graph, resolution-independent of terrain
    def toks(r_):
        return [(round(p[0] * 2) / 2, round(p[1] * 2) / 2) for p in r_[::4]]
    route_edges = tuple(frozenset(frozenset((a, b)) for a, b in zip(t, t[1:]))
                        for t in (toks(r_) for r_ in menu))
    game = InterdictionGame(tuple(tuple(map(tuple, r_)) for r_ in menu), route_edges,
                            tuple(tuple(t) for t in isets), payoff, travel, K)
    return game, menu, coords, rr, pp, S
