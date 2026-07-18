"""gen28 v3-theatre: the REAL-terrain aerial interdiction game (Kilian 2026-07-18).

A real ~45 km corridor (OSM land cover, `scratch/fetch_theatre.py`) becomes the game:
  * lattice = the terrain grid cells (metric UTM coords); the UAV flies at altitude so EVERY
    cell is navigable (terrain never blocks the drone, only shapes cost/threat);
  * base and target are two REAL settlements at ARBITRARY cells (not centred); "forward
    progress" = increasing projection onto the base->target axis, so the corridor runs in its
    true direction and the route set stays a finite DAG;
  * TERRAIN drives the interdictor, not the drone: hazards emplace only on emplaceable ground
    (NOT water, NOT dense urban), with terrain-set radius + effectiveness (open/field = long
    range high p; forest = short range high p, concealed); and dense URBAN cells MASK
    line-of-sight (a hazard cannot engage an arc if the segment between them crosses an urban
    cell) - the "urban corridors block contesting" mechanic;
  * the game is an `InterdictionGame` over a screened diverse route menu x K-hazard sets, so the
    LP / greedy BR / fleet mission oracle (solve_multiconvoy) all apply verbatim.

Reuses the line-integral exposure calibration (kappa = -ln(1-p)/r; straight dead-centre transit
intercepted w.p. p). Ledger: experiments/gen28_aerial.md (v3-theatre section).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame

CLASS_NAME = {0: "open", 1: "field", 2: "forest", 3: "urban", 4: "water"}
# terrain -> (emplaceable, radius_km, p_max, blocks_LOS): the pinned mechanic table
TERRAIN = {
    "open":   dict(emplace=True,  r_km=2.5, p_max=0.90, los_block=False),
    "field":  dict(emplace=True,  r_km=2.5, p_max=0.90, los_block=False),
    "forest": dict(emplace=True,  r_km=1.2, p_max=0.92, los_block=True),   # concealed short-range
    "urban":  dict(emplace=False, r_km=0.0, p_max=0.00, los_block=True),   # blocked + shields
    "water":  dict(emplace=False, r_km=0.0, p_max=0.00, los_block=False),
}


@dataclass(frozen=True)
class Theatre:
    name: str
    grid: np.ndarray            # [nrow, ncol] class ids
    base: tuple                 # (row, col)
    target: tuple
    cell_m: float
    nrow: int
    ncol: int

    def cls(self, rc) -> str:
        return CLASS_NAME[int(self.grid[rc[0], rc[1]])]

    def xy(self, rc) -> np.ndarray:
        return np.array([rc[1] * self.cell_m, rc[0] * self.cell_m])   # (easting, northing)


def load_theatre(path: str) -> Theatre:
    d = json.load(open(path))
    return Theatre(d["name"], np.array(d["grid"], dtype=int),
                   tuple(d["base"]["cell"]), tuple(d["target"]["cell"]),
                   float(d["cell_m"]), int(d["nrow"]), int(d["ncol"]))


def _axis(th: Theatre):
    b, t = th.xy(th.base), th.xy(th.target)
    v = t - b
    return b, v / (np.linalg.norm(v) + 1e-9)


def forward_dag(th: Theatre):
    """8-neighbour arcs from each cell to neighbours with strictly greater axis projection
    (a DAG base->target; the generalisation of 'forward in x' to an arbitrary axis)."""
    b, u = _axis(th)
    proj = {}
    for r in range(th.nrow):
        for c in range(th.ncol):
            proj[(r, c)] = float((th.xy((r, c)) - b) @ u)
    succ = {}
    for r in range(th.nrow):
        for c in range(th.ncol):
            nb = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < th.nrow and 0 <= cc < th.ncol and proj[(rr, cc)] > proj[(r, c)] + 1e-6:
                        nb.append((rr, cc))
            succ[(r, c)] = sorted(nb, key=lambda x: -proj[x])
    return succ, proj


def lane_route(th: Theatre, succ, proj, offset_frac: float,
               mask_pref: float = 0.0) -> tuple | None:
    """A corridor-spanning LANE: forward walk holding a target LATERAL offset that ramps 0 at
    base -> offset_frac*half-width at mid-corridor -> 0 at target (a hat profile), so lanes
    fan out across the width and reconverge (the small-game lane, on the diagonal axis).
    ``mask_pref`` >0 nudges toward LOS-masking terrain (forest/urban) among near-tied steps:
    terrain-aware routes that hug cover (the terrain-following option)."""
    b, u = _axis(th)
    nrm = np.array([-u[1], u[0]])
    half = 0.5 * th.nrow * th.cell_m
    span = float((th.xy(th.target) - b) @ u)
    node, path, seen = th.base, [th.base], {th.base}
    for _ in range(6 * (th.nrow + th.ncol)):
        if node == th.target:
            return tuple(path)
        nb = [m for m in succ[node] if m not in seen]
        if not nb:
            return None
        s_here = float((th.xy(node) - b) @ u) / (span + 1e-9)     # 0..1 along axis
        ramp = np.sin(np.pi * np.clip(s_here, 0, 1))              # hat: 0 ends, 1 middle
        want_lat = offset_frac * half * ramp
        tgt = th.xy(th.target)
        best, bs = None, -1e18
        for m in nb:
            lat = float((th.xy(m) - b) @ nrm)
            # DAG already guarantees forward progress, so LATERAL tracking dominates (per cell);
            # end-game: pull to the actual target cell as s->1; tiny mask tiebreak.
            score = -abs(lat - want_lat) / th.cell_m
            score += -(s_here ** 3) * np.linalg.norm(th.xy(m) - tgt) / th.cell_m
            score += 0.02 * float((th.xy(m) - th.xy(node)) @ u) / th.cell_m
            score += mask_pref * (1.0 if TERRAIN[th.cls(m)]["los_block"] else 0.0)
            if score > bs:
                bs, best = score, m
        node = best; path.append(node); seen.add(node)
    return None


def sample_route(th: Theatre, succ, rng, lateral_bias: float) -> tuple | None:
    _, proj = None, {}
    return lane_route(th, succ, {}, lateral_bias)


def build_route_menu(th: Theatre, R: int = 24, seed: int = 0) -> list[tuple]:
    """The corridor-spanning menu: LANES at lateral offsets across the width (both
    terrain-blind and terrain-hugging variants), deduped. Fans out so no single hazard
    covers the whole mixture (the funnel fix)."""
    succ, proj = forward_dag(th)
    cands, seen = [], set()
    for off in np.linspace(-1.6, 1.6, 17):
        for mp in (0.0, 0.6):                        # blind + terrain-hugging
            p = lane_route(th, succ, proj, float(off), mask_pref=mp)
            if p and p not in seen:
                seen.add(p); cands.append(p)
    if not cands:
        raise RuntimeError("no forward routes found")
    b, u = _axis(th); nrm = np.array([-u[1], u[0]])

    def sig(p):
        xs = np.array([(th.xy(c) - b) @ u for c in p])
        ls = np.array([(th.xy(c) - b) @ nrm for c in p])
        st = np.linspace(xs.min(), xs.max(), 10)
        return np.interp(st, xs, ls) / th.cell_m
    menu, sigs = [cands[0]], [sig(cands[0])]
    while len(menu) < R and len(menu) < len(cands):
        best, bd = None, -1
        for p in cands:
            if p in menu:
                continue
            d = min(np.linalg.norm(sig(p) - t) for t in sigs)
            if d > bd:
                bd, best = d, p
        if best is None:
            break
        menu.append(best); sigs.append(sig(best))
    return menu


def hazard_sites(th: Theatre, stride: int = 2, standoff_km: float = 4.0):
    """Emplaceable cells (terrain emplace=True), subsampled by stride, EXCLUDING sites within
    ``standoff_km`` of base or target (friendly-controlled terminal airspace: the funnel fix,
    the small-game standoff lesson applied at theatre scale). Returns (coords[H,2], r_m[H],
    p_max[H], cells[H])."""
    b, t = th.xy(th.base), th.xy(th.target)
    so = standoff_km * 1000.0
    coords, rr, pp, cells = [], [], [], []
    for r in range(0, th.nrow, stride):
        for c in range(0, th.ncol, stride):
            spec = TERRAIN[th.cls((r, c))]
            if not spec["emplace"]:
                continue
            xy = th.xy((r, c))
            if np.linalg.norm(xy - b) < so or np.linalg.norm(xy - t) < so:
                continue
            coords.append(xy); rr.append(spec["r_km"] * 1000.0)
            pp.append(spec["p_max"]); cells.append((r, c))
    return np.array(coords), np.array(rr), np.array(pp), cells


def _los_blocked(th: Theatre, p_hazard: np.ndarray, p_arc: np.ndarray) -> bool:
    """True if the segment hazard->arc-midpoint crosses an LOS-blocking cell (urban/forest):
    dense terrain shields the corridor from that hazard (the urban-canyon mechanic)."""
    steps = max(2, int(np.linalg.norm(p_arc - p_hazard) / th.cell_m) + 1)
    for t in np.linspace(0.15, 0.85, steps):          # skip the endpoints themselves
        q = p_hazard + t * (p_arc - p_hazard)
        c = int(round(q[0] / th.cell_m)); r = int(round(q[1] / th.cell_m))
        if 0 <= r < th.nrow and 0 <= c < th.ncol and TERRAIN[th.cls((r, c))]["los_block"]:
            return True
    return False


def route_survival(th: Theatre, route: tuple, coords, rr, pp, *, los: bool) -> np.ndarray:
    """S[h] = survival of the route against hazard h alone (line-integral exposure, LOS-masked;
    kappa_h = -ln(1-p_h)/r_h so a dead-centre straight leg is intercepted w.p. p_h)."""
    pts = np.array([th.xy(c) for c in route], dtype=float)
    mids = (pts[:-1] + pts[1:]) / 2.0
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    kappa = -np.log(np.clip(1.0 - pp, 1e-12, 1.0)) / np.clip(rr, 1e-9, None)
    S = np.ones(len(coords))
    for h in range(len(coords)):
        d = np.linalg.norm(mids - coords[h], axis=1)
        taper = np.clip(1.0 - d / rr[h], 0.0, None)
        if los:
            for a in np.where(taper > 0)[0]:
                if _los_blocked(th, coords[h], mids[a]):
                    taper[a] = 0.0
        S[h] = np.exp(-(kappa[h] * taper * ds).sum())
    return S


def build_theatre_game(th: Theatre, K: int = 1, menu_size: int = 24, site_stride: int = 2,
                       seed: int = 0, los: bool = True, standoff_km: float = 4.0):
    """Returns (InterdictionGame, menu routes, hazard coords, r_m, p_max, S[R,H]). Exact only
    while C(H,K) fits; past that use the greedy BR on S (the gen26 pattern)."""
    menu = build_route_menu(th, R=menu_size, seed=seed)
    coords, rr, pp, cells = hazard_sites(th, stride=site_stride, standoff_km=standoff_km)
    S = np.stack([route_survival(th, r_, coords, rr, pp, los=los) for r_ in menu])   # [R,H]
    H = len(coords)
    isets = list(itertools.combinations(range(H), K)) if K <= H else [tuple(range(H))]
    logS = np.log(np.clip(S, 1e-300, 1.0))
    idx = np.asarray(isets, dtype=int)
    if len(menu) * len(isets) > 60_000_000:
        raise MemoryError(f"exact matrix {len(menu)}x{len(isets)} too large; use greedy BR")
    payoff = 1.0 - np.exp(logS[:, idx].sum(axis=2))
    travel = np.array([sum(np.linalg.norm(th.xy(b) - th.xy(a))
                           for a, b in zip(r_, r_[1:])) for r_ in menu])
    route_edges = tuple(frozenset(frozenset((a, b)) for a, b in zip(r_, r_[1:])) for r_ in menu)
    game = InterdictionGame(tuple(menu), route_edges, tuple(tuple(t) for t in isets),
                            payoff, travel, K)
    return game, menu, coords, rr, pp, S
