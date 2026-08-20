"""Continuous aerial theatre game on vector terrain, with no rasterising anywhere.

Terrain stays as land-cover polygons in kilometre coordinates. Flight paths are smooth Catmull-Rom
lanes from base to target, hazard sites are sampled in continuous space and kept only where terrain
permits emplacement, and exposure is the line integral of the hazard rate along a lane, masked
wherever the sight line crosses blocking terrain. Map detail is decoupled from training because the
policy reads per-route features and a route-vertex graph rather than pixels. The result is an
`InterdictionGame`, so the LP, greedy best response and fleet mission oracle all apply unchanged.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from shapely.ops import nearest_points, unary_union

from src.baselines.interdiction_oracle import InterdictionGame

# terrain class -> emplaceable, radius (km), p_max, whether it blocks line of sight. The radii and
# lethalities are reference-scale values; ranges scale per theatre through range_scale so that the
# coverage fraction phi = 2Kr/W_lat stays comparable across maps of different size, with the ratios
# between classes fixed. Firepower is set by terrain, never by the adversary.
TERRAIN = {
    "open":   dict(emplace=True,  r_km=2.5, p_max=0.90, los=False),
    "field":  dict(emplace=True,  r_km=2.5, p_max=0.90, los=False),
    "forest": dict(emplace=True,  r_km=1.2, p_max=0.92, los=True),
    "urban":  dict(emplace=False, r_km=0.0, p_max=0.00, los=True),
    "water":  dict(emplace=False, r_km=0.0, p_max=0.00, los=False),
    "sea":    dict(emplace=False, r_km=0.0, p_max=0.00, los=False),
    # high ground is a passive wall, non-emplaceable but blocking sight lines
    "alpine": dict(emplace=False, r_km=0.0, p_max=0.00, los=True),
}
PRIORITY = ["water", "sea", "urban", "alpine", "forest", "field"]

# Terrain table v2, in which concealment buys persistence. The table above stays the module default;
# v2 adds a reach-versus-cover trade, so open ground reaches furthest and kills hardest while cover
# shoots short and weak, and a `reveal` flag, so a site on revealing ground that engages the flight
# becomes visible to the defender from the next serial while concealed ground never gives itself
# away. Urban ground is emplaceable here, which relies on the self-polygon sight-line exemption in
# `route_survival`, without which an urban site masks itself and can never engage.
TERRAIN_V2 = {
    "open":   dict(emplace=True,  r_km=3.5, p_max=0.90, los=False, reveal=True),
    "field":  dict(emplace=True,  r_km=2.5, p_max=0.85, los=False, reveal=True),
    # forest hides without blinding, since canopy conceals a ground team from an aircraft looking
    # down but does not stop that team engaging an aircraft above the treeline, hence reveal=False
    # with los=False. The symmetric variant that blocks both is reachable via
    # terrain_v2(forest_los=True) and is a different game.
    "forest": dict(emplace=True,  r_km=1.5, p_max=0.55, los=False, reveal=False),
    "urban":  dict(emplace=True,  r_km=1.0, p_max=0.45, los=True,  reveal=False),
    "water":  dict(emplace=False, r_km=0.0, p_max=0.00, los=False, reveal=True),
    "sea":    dict(emplace=False, r_km=0.0, p_max=0.00, los=False, reveal=True),
    "alpine": dict(emplace=False, r_km=0.0, p_max=0.00, los=True,  reveal=True),
}


def terrain_v2(hidden_leth: float = 1.0, forest_los: bool = False,
               conceal_reach: float | None = None) -> dict:
    """Return ``TERRAIN_V2`` with the concealment knobs applied.

    Args:
        hidden_leth: multiplier on the concealed classes' lethality, that is how much of the visible
            classes' killing power cover retains.
        conceal_reach: forest reach as a fraction of open reach, with urban keeping 70% of forest's.
        forest_los: True restores the symmetric rule where woodland also masks sight lines, which is
            a different game whose numbers may not be mixed with the default's.
    """
    t = {k: dict(v) for k, v in TERRAIN_V2.items()}
    for k in ("forest", "urban"):
        t[k]["p_max"] = float(np.clip(t[k]["p_max"] * hidden_leth, 0.0, 0.999))
    if conceal_reach is not None:
        t["forest"]["r_km"] = float(conceal_reach * t["open"]["r_km"])
        t["urban"]["r_km"] = float(0.7 * t["forest"]["r_km"])
    t["forest"]["los"] = bool(forest_los)
    return t


def blocker_union(th: "VecTheatre", terrain: dict | None = None):
    """Union of every terrain class that blocks line of sight, cached on the theatre."""
    terrain = TERRAIN if terrain is None else terrain
    keys = tuple(sorted(k for k, v in terrain.items() if v.get("los")))
    cache = getattr(th, "_los_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(th, "_los_cache", cache)
    if keys not in cache:
        parts = [u for k in keys if (u := th._union.get(k)) is not None and not u.is_empty]
        cache[keys] = unary_union(parts) if parts else None
    return cache[keys]


def containing_blockers(th: "VecTheatre", coords, terrain: dict | None = None) -> list:
    """Per site, the sight-line-blocking polygon it stands in, or None.

    A site is never masked by its own polygon, since otherwise an urban or forest site starts its
    own sight line inside a blocker and can never engage anything.
    """
    # terrain=None masks on urban only; passing a table honours that table's blocking classes
    keys = ["urban"] if terrain is None else [k for k, v in terrain.items() if v.get("los")]
    out = []
    for xy in np.asarray(coords, dtype=float):
        p = Point(float(xy[0]), float(xy[1]))
        own = None
        for k in keys:
            for poly in th.polys.get(k, []):
                if poly.contains(p):
                    own = poly
                    break
            if own is not None:
                break
        out.append(own)
    return out


def reveal_flags(cls, terrain: dict | None = None) -> np.ndarray:
    """Per site, whether engaging the flight gives that site away to the defender."""
    terrain = TERRAIN if terrain is None else terrain
    return np.array([bool(terrain[c].get("reveal", True)) for c in cls], dtype=bool)


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
    """Load a vector theatre, accepting both the ``classes`` and the ``poly``/``line`` layouts.

    Classes the terrain table does not model are skipped, so only known terrain drives the game.
    """
    d = json.load(open(path))
    raw = d.get("classes")
    if raw is None:
        raw = dict(d.get("poly", {}))
        raw.pop("land_holes", None)
    polys, union, tree = {}, {}, {}
    for cls, rings in raw.items():
        if cls not in TERRAIN:
            continue
        ps = [Polygon(r) for r in rings if len(r) >= 4]
        ps = [p if p.is_valid else p.buffer(0) for p in ps]
        polys[cls] = ps
        union[cls] = unary_union(ps) if ps else None
        tree[cls] = STRtree(ps) if ps else None
    return VecTheatre(d["name"], d["W_km"], d["H_km"],
                      np.array(d["base"]["xy_km"]), np.array(d["target"]["xy_km"]),
                      polys, union, tree, union.get("urban"))


def lateral_width(th: VecTheatre) -> float:
    """Extent of the theatre box perpendicular to the base-to-target axis.

    Weapon ranges scale by the ratio of this width to the reference theatre's, so that a short-range
    system on a narrow corridor and a scaled one on a wide corridor contest the same fraction of the
    width.
    """
    _, nrm = _axis(th)
    corners = np.array([[0.0, 0.0], [th.W, 0.0], [0.0, th.H], [th.W, th.H]])
    lat = corners @ nrm
    return float(lat.max() - lat.min())


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
    """A smooth base-to-target lane holding a hat-shaped lateral offset.

    The offset ramps from zero to ``offset_frac`` of the half-width and back, so lanes fan across
    the corridor and reconverge at the terminals.
    """
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


def _class_parts(th: VecTheatre, cls: str) -> list:
    out = []
    for p in th.polys.get(cls, []):
        out += list(p.geoms) if hasattr(p, "geoms") else [p]
    return [g for g in out if g.area > 0 and g.is_valid]


def _snap_into(g, xy, eps_km: float = 0.05):
    """Nearest point inside polygon ``g``, nudged off the boundary so classification agrees."""
    p = nearest_points(g, Point(float(xy[0]), float(xy[1])))[0]
    c = g.representative_point()
    v = np.array([c.x - p.x, c.y - p.y])
    n = float(np.linalg.norm(v))
    q = np.array([p.x, p.y]) + (v / n) * min(eps_km, n / 2.0) if n > 1e-12 else np.array([c.x, c.y])
    return q if g.covers(Point(*q)) else np.array([c.x, c.y])


def quota_sites(th: VecTheatre, n_sites: int = 200, spacing_km: float = 0.0,
                standoff_km: float = 4.0, range_scale: float = 1.0,
                terrain: dict | None = None, snap_cells: float = 1.0, min_sep_frac: float = 0.25,
                anchor_mult: float = 3.0):
    """Candidate emplacements whose class shares match the theatre's terrain composition.

    A plain raster labels each node by the polygon it lands on, so cover in patches smaller than the
    grid is never offered at all, while snapping every node to nearby cover over-represents it and
    sampling purely by area clumps the points lengthwise. This combines the fixes: quotas come from
    the whole-map emplaceable area shares, an even grid supplies the spatial skeleton, each class's
    quota is filled by the grid nodes nearest that class and snapped inside the polygon they are
    assigned to, and snaps are capped at ``snap_cells`` grid cells and kept ``min_sep_frac`` of a
    cell apart so that one small patch cannot absorb a whole quota. Nodes never reassigned keep the
    ground they already stand on, and only emplaceable classes take part, so the shares are
    renormalised over ground a team can actually stand on.
    """
    terrain = TERRAIN if terrain is None else terrain
    # the anchor grid sizes itself from the budget, roughly anchor_mult x n_sites nodes, so that
    # every class quota can be filled and the assignment has room to choose. A fixed spacing caps
    # the anchors at whatever the map size gives, which can starve a class of points entirely.
    cell = float(spacing_km) if spacing_km else float(
        np.sqrt(th.W * th.H / max(anchor_mult * n_sites, 1.0)))
    anchors = [np.array([x, y])
               for x in np.arange(1.0, th.W, cell) for y in np.arange(1.0, th.H, cell)
               if not (np.linalg.norm(np.array([x, y]) - th.base) < standoff_km
                       or np.linalg.norm(np.array([x, y]) - th.target) < standoff_km)]
    empl = [k for k, v in terrain.items() if v["emplace"]]
    area = {k: float(sum(g.area for g in _class_parts(th, k))) for k in empl}
    area["open"] = max(th.W * th.H - sum(float(sum(g.area for g in _class_parts(th, k)))
                                         for k in th.polys), 0.0)          # open = the residual
    tot = sum(area.get(k, 0.0) for k in empl) or 1.0
    quota = {k: int(round(n_sites * area.get(k, 0.0) / tot)) for k in empl}

    A = np.asarray(anchors, float)
    taken = np.zeros(len(anchors), bool)
    placed: dict[str, list] = {k: [] for k in empl}

    def spread(pool_idx, n):
        """Farthest-point selection of ``n`` anchors from ``pool_idx``.

        Every anchor standing inside a patch ties at distance zero, so without this the quota fills
        in raster scan order and the candidates clump at one end of the theatre.
        """
        pool_idx = np.asarray(pool_idx, int)
        if len(pool_idx) <= n:
            return list(pool_idx)
        seed_i = int(pool_idx[np.argmin(np.linalg.norm(A[pool_idx] - A[pool_idx].mean(0), axis=1))])
        sel = [seed_i]
        d = np.linalg.norm(A[pool_idx] - A[seed_i], axis=1)
        while len(sel) < n:
            j = int(np.argmax(d))
            sel.append(int(pool_idx[j]))
            d = np.minimum(d, np.linalg.norm(A[pool_idx] - A[pool_idx[j]], axis=1))
        return sel

    # polygon classes first and rarest quota first, since those are the constrained ones, while
    # `open` is the residual and can be satisfied anywhere
    for k in sorted([k for k in empl if _class_parts(th, k)], key=lambda k: quota[k]):
        parts = _class_parts(th, k)
        tree = STRtree(parts)
        near = {}
        for i, xy in enumerate(anchors):
            p = Point(float(xy[0]), float(xy[1]))
            j = int(tree.nearest(p))
            if parts[j].distance(p) <= snap_cells * cell:
                near[i] = j
        order = spread([i for i in near if not taken[i]], quota[k])
        rest = [i for i in near if not taken[i] and i not in set(order)]
        for i in list(order) + rest:                     # top up if a snap is rejected
            if len(placed[k]) >= quota[k]:
                break
            if taken[i]:
                continue
            q = _snap_into(parts[near[i]], anchors[i])
            if (np.linalg.norm(q - th.base) < standoff_km
                    or np.linalg.norm(q - th.target) < standoff_km
                    or th.classify(q) != k):
                continue
            if any(np.linalg.norm(q - o) < min_sep_frac * cell for o in placed[k]):
                continue                                 # one patch cannot absorb a whole quota
            taken[i] = True
            placed[k].append(q)
    for k in empl:                                       # the residual classes, spread likewise
        want = quota.get(k, 0) - len(placed.get(k, []))
        if want <= 0:
            continue
        pool_idx = [i for i, xy in enumerate(anchors)
                    if not taken[i] and th.classify(xy) == k and terrain[k]["emplace"]]
        for i in spread(pool_idx, want):
            taken[i] = True
            placed[k].append(np.asarray(anchors[i], float))

    coords, rr, pp, cls = [], [], [], []
    for k, pts in placed.items():
        for q in pts:
            coords.append(q); rr.append(terrain[k]["r_km"] * range_scale)
            pp.append(terrain[k]["p_max"]); cls.append(k)
    return np.array(coords), np.array(rr), np.array(pp), cls


def hazard_sites(th: VecTheatre, spacing_km: float = 2.0, standoff_km: float = 4.0,
                 range_scale: float = 1.0, terrain: dict | None = None,
                 stratified: int = 0, seed: int = 0):
    """Continuous candidate sites on emplaceable terrain, outside the terminal standoff.

    Args:
        range_scale: multiplies every weapon range, giving coverage-fraction comparability across
            map sizes.
        stratified: adds this many extra candidates drawn inside the emplaceable polygons, allocated
            across classes in proportion to area and uniform by area within a class. A plain raster
            systematically misses cover, which comes in patches smaller than the grid while open
            ground comes in blocks, and that biases concealment results downwards. Zero keeps the
            pure raster.

    Returns:
        ``(coords[H, 2], r_km[H], p_max[H], cls[H])``.
    """
    terrain = TERRAIN if terrain is None else terrain
    xs = np.arange(1.0, th.W, spacing_km)
    ys = np.arange(1.0, th.H, spacing_km)
    coords, rr, pp, cls = [], [], [], []

    def offer(xy):
        if (np.linalg.norm(xy - th.base) < standoff_km
                or np.linalg.norm(xy - th.target) < standoff_km):
            return
        k = th.classify(xy)
        spec = terrain[k]
        if not spec["emplace"]:
            return
        coords.append(np.asarray(xy, float)); rr.append(spec["r_km"] * range_scale)
        pp.append(spec["p_max"]); cls.append(k)

    for x in xs:
        for y in ys:
            offer(np.array([x, y]))

    if stratified > 0:
        rng = np.random.default_rng(seed)
        pool = {k: _class_parts(th, k) for k, v in terrain.items()
                if v["emplace"] and _class_parts(th, k)}
        areas = {k: sum(g.area for g in v) for k, v in pool.items()}
        tot = sum(areas.values()) or 1.0
        for k, parts in pool.items():
            n = int(round(stratified * areas[k] / tot))
            if n <= 0:
                continue
            w = np.array([g.area for g in parts], float)
            w /= w.sum()
            for j in rng.choice(len(parts), size=n, p=w):        # patch by area, point uniform
                g = parts[int(j)]
                x0, y0, x1, y1 = g.bounds
                for _ in range(40):                              # rejection-sample inside it
                    xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                    if g.covers(Point(*xy)):
                        offer(xy)
                        break
    return np.array(coords), np.array(rr), np.array(pp), cls


def route_survival(th: VecTheatre, route: np.ndarray, coords, rr, pp, *, los: bool,
                   terrain: dict | None = None, own_polys: list | None = None,
                   return_exposed: bool = False):
    """Survival of the lane against each hazard alone.

    The rate along the lane is ``kappa_h * max(0, 1 - d/r_h)`` with ``kappa_h = -ln(1 - p_h) / r_h``,
    so a leg flown dead centre is intercepted with probability ``p_h``. A hazard cannot engage an arc
    whose sight line crosses a blocker other than the polygon the hazard itself stands in.

    Args:
        return_exposed: also return a per-site flag for whether the flight passed inside that site's
            ring with line of sight, that is whether the site engaged this serial.
    """
    mids = (route[:-1] + route[1:]) / 2.0
    ds = np.linalg.norm(np.diff(route, axis=0), axis=1)
    kappa = -np.log(np.clip(1.0 - pp, 1e-12, 1.0)) / np.clip(rr, 1e-9, None)
    S = np.ones(len(coords))
    exposed = np.zeros(len(coords), dtype=bool)
    # terrain=None masks on urban only, matching containing_blockers above
    blk = (th._urban_union if terrain is None else blocker_union(th, terrain)) if los else None
    for h in range(len(coords)):
        d = np.linalg.norm(mids - coords[h], axis=1)
        taper = np.clip(1.0 - d / rr[h], 0.0, None)
        if blk is not None and not blk.is_empty:
            own = own_polys[h] if own_polys is not None else None
            for a in np.where(taper > 0)[0]:
                seg = LineString([tuple(coords[h]), tuple(mids[a])])
                if own is not None:
                    seg = seg.difference(own)          # never masked by your own polygon
                    if seg.is_empty:
                        continue
                if seg.intersects(blk):
                    taper[a] = 0.0
        S[h] = np.exp(-(kappa[h] * taper * ds).sum())
        exposed[h] = bool((taper > 0).any())
    return (S, exposed) if return_exposed else S


def engagement_footprint(th: VecTheatre, center, r_km: float, n_rays: int = 96,
                         terrain: dict | None = None) -> list:
    """The hazard's engagement silhouette, as polygon vertices ``[[x, y], ...]`` in km.

    The range circle is ray-cast against the sight-line blockers so each ray stops at the first one
    it meets, giving a star-shaped viewshed with shadow zones behind built-up ground.
    """
    c = np.asarray(center, float)
    urb = th._urban_union if terrain is None else blocker_union(th, terrain)
    verts = []
    for k in range(n_rays):
        a = 2 * np.pi * k / n_rays
        d = np.array([np.cos(a), np.sin(a)])
        reach = r_km
        if urb is not None and not urb.is_empty:
            inter = LineString([tuple(c), tuple(c + r_km * d)]).intersection(urb.boundary)
            if not inter.is_empty:
                pts = []
                if inter.geom_type == "Point":
                    pts = [inter]
                elif inter.geom_type == "MultiPoint":
                    pts = list(inter.geoms)
                else:
                    for g in getattr(inter, "geoms", [inter]):
                        pts += [Point(xy) for xy in g.coords]
                if pts:
                    reach = min(reach, min(np.hypot(q.x - c[0], q.y - c[1]) for q in pts))
        verts.append([round(float(c[0] + reach * d[0]), 3), round(float(c[1] + reach * d[1]), 3)])
    return verts


def _threat_field(th: VecTheatre, coords, rr, pp, step=1.0, terrain: dict | None = None):
    """Peak engageable interception intensity at each coarse node, the map a planner routes against.

    The maximum is taken over candidate sites both in range and with line of sight. This grid is a
    routing helper only; the game itself stays continuous.
    """
    xs = np.arange(0.0, th.W + 1e-9, step)
    ys = np.arange(0.0, th.H + 1e-9, step)
    urb = th._urban_union if terrain is None else blocker_union(th, terrain)
    P = np.asarray(coords)
    T = {}
    for x in xs:
        for y in ys:
            d = np.hypot(P[:, 0] - x, P[:, 1] - y)
            inten = np.where(d < rr, pp * (1.0 - d / np.clip(rr, 1e-9, None)), 0.0)
            best = 0.0
            for h in np.argsort(-inten):
                if inten[h] <= best:
                    break
                if urb is None or urb.is_empty or not LineString(
                        [(float(P[h, 0]), float(P[h, 1])), (float(x), float(y))]).intersects(urb):
                    best = float(inten[h]); break
            T[(round(float(x), 3), round(float(y), 3))] = best
    return xs, ys, T


def build_terrain_menu(th: VecTheatre, coords, rr, pp, R: int = 24, step: float = 1.0,
                       terrain: dict | None = None) -> list:
    """Terrain-aware routes that bend through sight-line shadow and around threat coverage.

    Shortest paths are found on a coarse 8-connected grid whose edge cost is length times
    ``1 + lam * peak threat along the edge``, swept over ``lam`` from direct and exposed to long and
    covered, seeded with lateral mid-waypoints for diversity, then smoothed into flight paths.
    Falls back to the geometric lanes if routing fails.
    """
    import networkx as nx
    xs, ys, T = _threat_field(th, coords, rr, pp, step=step, terrain=terrain)
    _, nrm = _axis(th)

    def snap(xy):
        return (round(float(xs[np.argmin(np.abs(xs - xy[0]))]), 3),
                round(float(ys[np.argmin(np.abs(ys - xy[1]))]), 3))

    b, t = snap(th.base), snap(th.target)

    def build_graph(lam, seed_lat):
        G = nx.DiGraph()
        for x in xs:
            for y in ys:
                xk = (round(float(x), 3), round(float(y), 3))
                for dx in (-step, 0, step):
                    for dy in (-step, 0, step):
                        if dx == 0 and dy == 0:
                            continue
                        nx_, ny_ = round(float(x + dx), 3), round(float(y + dy), 3)
                        if 0 <= nx_ <= th.W and 0 <= ny_ <= th.H:
                            seg = np.hypot(dx, dy)
                            thr = 0.5 * (T[xk] + T.get((nx_, ny_), T[xk]))
                            lat = abs(float((np.array([x, y]) - th.base) @ nrm) - seed_lat) * 0.02
                            G.add_edge(xk, (nx_, ny_), w=seg * (1 + lam * thr) + lat * seg)
        return G

    cands, seen = [], set()
    for lam in (0.0, 3.0, 8.0, 18.0):
        for seed in np.linspace(-0.4 * th.H, 0.4 * th.H, 7):
            try:
                path = nx.shortest_path(build_graph(lam, float(seed)), b, t, weight="w")
            except Exception:
                continue
            key = tuple(round(v, 0) for pt in path[::3] for v in pt)
            if key in seen:
                continue
            seen.add(key)
            arr = np.array([th.base] + [np.array(p) for p in path[1:-1]] + [th.target])
            idx = np.unique(np.linspace(0, len(arr) - 1,
                                        max(4, int(len(arr) * step / 4))).astype(int))
            cands.append(_catmull(arr[idx]))
    if not cands:
        return build_menu(th, R=R)
    u2, nrm2 = _axis(th)

    def sig(rte):
        al = np.array([(p - th.base) @ u2 for p in rte])
        la = np.array([(p - th.base) @ nrm2 for p in rte])
        return np.interp(np.linspace(al.min(), al.max(), 10), al, la)

    menu, sigs = [cands[0]], [sig(cands[0])]
    while len(menu) < R and len(menu) < len(cands):
        best, bd = None, -1
        for i, rte in enumerate(cands):
            if any(rte is m for m in menu):
                continue
            dd = min(np.linalg.norm(sig(rte) - x) for x in sigs)
            if dd > bd:
                bd, best = dd, rte
        if best is None:
            break
        menu.append(best); sigs.append(sig(best))
    return menu


def build_theatre_game(th: VecTheatre, K: int = 1, n_lanes: int = 14, n_terrain: int = 12,
                       spacing_km: float = 2.0, standoff_km: float = 4.0, los: bool = True,
                       range_scale: float = 1.0, terrain: dict | None = None,
                       return_cls: bool = False, stratified: int = 0, site_seed: int = 0,
                       n_sites: int = 0):
    """Assemble the theatre game on the continuous polygon terrain.

    The menu carries both the geometric lanes, which are the naive rule's direct and exposed
    support, and the terrain-aware cover-seeking routes, so the game can measure terrain-smart
    mixing against naive lane play.

    Returns:
        ``(InterdictionGame, menu, coords, r_km, p_max, S[R, H], lane_idx)``, where ``lane_idx``
        gives the menu indices of the geometric lanes.
    """
    if n_sites:                       # fixed site budget, allocated by terrain composition
        coords, rr, pp, cls = quota_sites(th, n_sites=n_sites, standoff_km=standoff_km,
                                          range_scale=range_scale, terrain=terrain)
    else:
        coords, rr, pp, cls = hazard_sites(th, spacing_km=spacing_km, standoff_km=standoff_km,
                                           range_scale=range_scale, terrain=terrain,
                                           stratified=stratified, seed=site_seed)
    own = containing_blockers(th, coords, terrain) if terrain is not None else None
    lanes = build_menu(th, R=n_lanes)
    lane_idx = list(range(len(lanes)))
    menu = list(lanes)
    u2, nrm2 = _axis(th)

    def sig(rte):
        al = np.array([(np.asarray(p) - th.base) @ u2 for p in rte])
        la = np.array([(np.asarray(p) - th.base) @ nrm2 for p in rte])
        return np.interp(np.linspace(al.min(), al.max(), 10), al, la)
    lane_sigs = [sig(l) for l in lanes]
    for r_ in build_terrain_menu(th, coords, rr, pp, R=n_terrain, terrain=terrain):
        if min(np.linalg.norm(sig(r_) - s) for s in lane_sigs) > 1.0:   # distinct from a lane
            menu.append(r_)
    S = np.stack([route_survival(th, r_, coords, rr, pp, los=los, terrain=terrain, own_polys=own)
                  for r_ in menu])
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
    if return_cls:                                   # callers needing the per-site terrain class
        return game, menu, coords, rr, pp, S, lane_idx, cls
    return game, menu, coords, rr, pp, S, lane_idx
