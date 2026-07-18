"""gen28 v5 THEATRE: a procedurally-generated heterogeneous 50-km-scale aerial sector
(Kilian's mega-map vision, built at reviewable theatre scale on the existing machinery).

Terrain is a per-cell class grid; each class maps to the two things the game machinery already
consumes: (a) FLYABILITY (obstacle cells block the flight DAG) and (b) EMPLACEMENT + threat
character (whether the interdictor may commit a hazard there, and with what effectiveness p_max
and radius r). The scientific point is heterogeneity PER DECISION: the drone trades a direct
exposed field crossing against a longer cover route (urban / river / valley), and the strongest
naive rule stops being two lines and becomes a corridor-selection PIPELINE - the regime where
the gen28 "rules always catch up" pattern is most likely to break.

  class     fly?   emplace?  p_max  r      role
  FIELD     yes    yes       0.90   2.0    open kill-zone (direct but naked)
  FOREST    yes    yes       0.90   0.8    concealed short-range ambush (deadly up close)
  SUBURB    yes    yes       0.60   1.2    intermediate
  ROAD      yes    yes       0.90   1.4    mobile-SAM danger line (fast repositioning)
  URBAN     yes    NO        -      -      cover corridor (interceptor cannot emplace)
  WATER     yes    NO        -      -      river highway (safe corridor)
  SWAMP     yes    NO        -      -      safe corridor (vehicles cannot emplace)
  MOUNTAIN  NO     NO        -      -      obstacle (organic pinches / valleys)

Terrain is drawn INDEPENDENTLY of base/target and of the lattice geometry (seeded), so a
theatre generalist evaluated zero-shot on unseen theatres is the genuine map-conditioning test.
Everything reuses aerial_curves (curved routes, line-integral exposure) and the InterdictionGame
oracle; per-position p_max/r arrays are already supported. Ledger experiments/gen28_aerial.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.envs.aerial_curves import build_curved_game, curve_survival_matrix, make_curve
from src.envs.aerial_sector import SectorLattice

FIELD, FOREST, SUBURB, ROAD, URBAN, WATER, SWAMP, MOUNTAIN = range(8)
CLASS_NAME = {FIELD: "field", FOREST: "forest", SUBURB: "suburb", ROAD: "road",
              URBAN: "urban", WATER: "water", SWAMP: "swamp", MOUNTAIN: "mountain"}
FLYABLE = {FIELD, FOREST, SUBURB, ROAD, URBAN, WATER, SWAMP}
EMPLACEABLE = {FIELD, FOREST, SUBURB, ROAD}
# (p_max, r) per emplaceable class
THREAT = {FIELD: (0.90, 2.0), FOREST: (0.90, 0.8), SUBURB: (0.60, 1.2), ROAD: (0.90, 1.4)}
FLIGHT_COST = {FIELD: 1.0, FOREST: 1.15, SUBURB: 1.05, ROAD: 1.0,
               URBAN: 1.25, WATER: 1.0, SWAMP: 1.4, MOUNTAIN: 99.0}


@dataclass
class Theatre:
    grid: np.ndarray            # [ny, nx] int class labels (grid[j, i] = class at col i, row j)
    ny: int
    nx: int
    seed: int

    def cls(self, i: int, j: int) -> int:
        return int(self.grid[j, i])

    def lattice(self) -> SectorLattice:
        blocked = frozenset((i, j) for i in range(self.nx) for j in range(self.ny)
                            if self.grid[j, i] == MOUNTAIN)
        return SectorLattice(ny=self.ny, nx=self.nx, blocked=blocked)

    def hazard_field(self, step: float = 0.5, safe_r: float = 3.0):
        """Emplaceable candidate centres + per-centre (p_max, r), read from the terrain class at
        each centre; excludes non-emplaceable terrain and the base/target standoff zones."""
        lat = self.lattice()
        base = np.asarray(lat.base, float)
        target = np.asarray(lat.target, float)
        xs = np.arange(1.0, self.nx - 1 + 1e-9, step)
        ys = np.arange(0.0, self.ny - 1 + 1e-9, step)
        centres, pmax, rad = [], [], []
        for x in xs:
            for y in ys:
                c = self.cls(int(round(x)), int(round(y)))
                if c not in EMPLACEABLE:
                    continue
                if np.hypot(x - base[0], y - base[1]) < safe_r:
                    continue
                if np.hypot(x - target[0], y - target[1]) < safe_r:
                    continue
                p, rr = THREAT[c]
                centres.append([x, y]); pmax.append(p); rad.append(rr)
        return (np.asarray(centres, float), np.asarray(pmax, float), np.asarray(rad, float))

    def flight_cost_field(self):
        """Per-cell flight-cost multiplier (urban/swamp cost more to traverse: the cover premium
        that makes the corridor trade a genuine cost-vs-security decision)."""
        return np.vectorize(FLIGHT_COST.get)(self.grid)


def _blob(grid, rng, cls, cx, cy, rx, ry):
    ny, nx = grid.shape
    for j in range(ny):
        for i in range(nx):
            if ((i - cx) / rx) ** 2 + ((j - cy) / ry) ** 2 <= 1.0:
                grid[j, i] = cls


def generate_theatre(ny: int = 17, nx: int = 40, seed: int = 0) -> Theatre:
    """Procedural theatre: field base, 1-2 mountain ridges with guaranteed valley gaps (organic
    pinches), a river highway (safe corridor), 1-2 urban cores with suburb rings (cover), forest
    patches (short-range ambush), roads along a couple of rows (danger lines). Deterministic in
    ``seed``; base/target rows kept flyable and reachable."""
    rng = np.random.default_rng(seed)
    grid = np.full((ny, nx), FIELD, dtype=int)
    mid = (ny - 1) // 2

    # 1-2 mountain ridges with a gap (the valley the drone must thread)
    n_ridge = rng.integers(1, 3)
    ridge_cols = sorted(rng.choice(range(nx // 4, 3 * nx // 4), size=n_ridge, replace=False))
    for c in ridge_cols:
        gap_c = int(rng.integers(1, ny - 3))                      # gap start row
        gap_w = int(rng.integers(3, 5))                           # gap width
        width = int(rng.integers(1, 3))                           # ridge thickness (cols)
        for dc in range(width):
            cc = min(c + dc, nx - 2)
            for j in range(ny):
                if not (gap_c <= j < gap_c + gap_w):
                    grid[j, cc] = MOUNTAIN

    # river: a thin water highway meandering across (safe corridor)
    ry = int(rng.integers(2, ny - 2))
    for i in range(nx):
        ry = int(np.clip(ry + rng.integers(-1, 2), 1, ny - 2))
        if grid[ry, i] != MOUNTAIN:
            grid[ry, i] = WATER

    # 1-2 urban cores with suburb rings (cover the interceptor cannot use)
    for _ in range(int(rng.integers(1, 3))):
        cx = int(rng.integers(nx // 5, 4 * nx // 5)); cy = int(rng.integers(2, ny - 2))
        _blob(grid, rng, SUBURB, cx, cy, 3.2, 2.4)
        _blob(grid, rng, URBAN, cx, cy, 1.8, 1.4)

    # forest patches (concealed short-range ambush)
    for _ in range(int(rng.integers(2, 5))):
        cx = int(rng.integers(2, nx - 2)); cy = int(rng.integers(0, ny))
        if grid[cy, cx] in (FIELD, SUBURB):
            _blob(grid, rng, FOREST, cx, cy, rng.uniform(1.5, 3.0), rng.uniform(1.5, 3.0))

    # a road danger-line along 1-2 rows (mobile SAM repositioning corridor)
    for _ in range(int(rng.integers(1, 3))):
        rr = int(rng.integers(1, ny - 1))
        for i in range(nx):
            if grid[rr, i] in (FIELD, SUBURB):
                grid[rr, i] = ROAD

    # keep base and target cells flyable (carve if a ridge/urban landed on them)
    for (i, j) in ((0, mid), (nx - 1, mid)):
        if grid[j, i] not in FLYABLE:
            grid[j, i] = FIELD
    # guarantee at least the mid rows near the terminals are flyable
    for i in (0, 1, nx - 2, nx - 1):
        if grid[mid, i] == MOUNTAIN:
            grid[mid, i] = FIELD
    return Theatre(grid=grid, ny=ny, nx=nx, seed=seed)


# ---------------------------------------------------------------------------
# Terrain-aware curved menu: include COVER routes (low-exposure), so the naive corridor
# pipeline has strong material (baseline completeness) and the policy shares the same menu.


def _curve_exposure(curve, centres, pmax, rad) -> float:
    if len(centres) == 0:
        return 0.0
    S = curve_survival_matrix([curve], centres, r=rad, p_max=pmax)[0]
    return float((1.0 - S).sum())          # total single-hazard exposure summed over sites


def build_theatre_menu(theatre: Theatre, centres, pmax, rad, R: int = 40, seed: int = 0):
    """Menu = the k lowest-EXPOSURE legal curves (cover/corridor routes: what a terrain-aware
    planner would fly) + lateral-diversity fill, all curvature-bounded and obstacle-legal on the
    theatre lattice. Returns (menu, exposures). Deterministic in ``seed``."""
    lat = theatre.lattice()
    from src.envs.aerial_curves import _blocked_rects
    rects = _blocked_rects(lat)
    rng = np.random.default_rng(seed)
    cands, seen = [], set()
    # straight + a spread of control-offset curves
    tries = 0
    while len(cands) < 12 * R and tries < 400 * R:
        tries += 1
        offs = tuple(np.round(rng.uniform(0, lat.ny - 1, size=6), 2))
        c = make_curve(lat, offs, rects)
        if c is not None and c.offsets not in seen:
            seen.add(c.offsets); cands.append(c)
    if not cands:
        return [], np.array([])
    exp = np.array([_curve_exposure(c, centres, pmax, rad) for c in cands])
    order = np.argsort(exp)
    menu = [cands[i] for i in order[:R // 2]]                       # the cover half (safest)
    chosen = {id(m) for m in menu}
    # diversity fill on lateral profile
    def rows(c): return np.array([n[1] for n in c.node_seq], float)
    while len(menu) < R and len(menu) < len(cands):
        best, bd = None, -1.0
        base = np.stack([rows(m) for m in menu])
        for c in cands:
            if id(c) in chosen:
                continue
            d = float(np.min(np.linalg.norm(base - rows(c), axis=1)))
            if d > bd:
                bd, best = d, c
        if best is None:
            break
        menu.append(best); chosen.add(id(best))
    menu_exp = np.array([_curve_exposure(c, centres, pmax, rad) for c in menu])
    return menu, menu_exp


def build_theatre_game(theatre: Theatre, K: int = 1, step: float = 0.5, R: int = 40,
                       seed: int = 0):
    """Full theatre game: (game, S, menu, centres, pmax, rad, menu_exp). Exact for K<=1 fleet;
    greedy yardstick beyond (as gen26)."""
    centres, pmax, rad = theatre.hazard_field(step=step)
    menu, menu_exp = build_theatre_menu(theatre, centres, pmax, rad, R=R, seed=seed)
    if not menu:
        raise ValueError(f"theatre seed {theatre.seed}: no legal routes (over-blocked)")
    game, S = build_curved_game(theatre.lattice(), menu, centres, K, r=rad, p_max=pmax)
    return game, S, menu, centres, pmax, rad, menu_exp
