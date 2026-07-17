"""SAC-trainable aerial interdiction env (gen28): the thin adapter that presents an aerial
sector game through the SAME observation/menu contract the road multiconvoy env exposes, so
`route_one`, `featurize_state`, `node_index_map`, the menu-select head and the whole
ProtagonistSAC update path work UNCHANGED on the lattice.

Contract replicated (see MultiConvoyInterdictionEnv + featurize_state):
  * observe() -> {nodes: {id: {x, y, demand, has_depot}}, edges: {(u,v): {distance,
    congestion_level}}, trucks: {0: {...}}, edge_vulnerability (col 4 = the layout's
    per-arc threat projection), menu_route_node_idx / menu_route_feats (per-instance, riding
    the transition), active_truck, taken_node_frac};
  * commit(j) / current_convoy() / defender_action_mask() / route_convoy_by_index() /
    defender_occupancy() / occupancies / _occ_index / obj_matrix / game / config.N.

Node ids are zero-padded "ii,jj" strings so featurize_state's sorted() row order is stable and
`menu_route_node_idx` (built with the same sort) can never repeat the 2026-07-09 ordering bug.
The observable threat projection (edge col 4) is max_h p(arc, h) under the instance's layout:
"how dangerous is this arc at the enemy's best nearby position" (recorded design decision;
the full field also reaches the head via the per-route exposure feature).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.baselines.multiconvoy_oracle import objective_matrix
from src.envs.aerial_sector import (Path, SectorLattice, arc_hazard_prob, arc_midpoints,
                                    build_aerial_game, route_survival_matrix)


def _nid(n) -> str:
    return f"{n[0]:02d},{n[1]:02d}"


@dataclass(frozen=True)
class AerialConfig:
    N: int = 1                       # single UAV (Kilian's pin); fleet = the A4 extension
    K: int = 1
    menu_select: bool = True         # always: the aerial policy is a route-menu policy
    objective: str = "mission"       # at N=1 mission == interception probability
    threshold_m: int = 1


class AerialInterdictionEnv:
    def __init__(self, lat: SectorLattice, menu: list[Path], centres: np.ndarray, *,
                 K: int = 1, r: float, p_max=0.9, taper: str = "linear", weather=None,
                 N: int = 1):
        self.lat = lat
        self.menu = menu
        self.centres = centres
        self.r, self.p_max, self.taper = r, p_max, taper
        self.config = AerialConfig(N=N, K=K)
        self.game = build_aerial_game(lat, menu, centres, K, r=r, p_max=p_max, taper=taper,
                                      weather=weather or [])
        self.S = route_survival_matrix(menu, centres, r=r, p_max=p_max, taper=taper)
        self.occupancies, self.obj_matrix = objective_matrix(self.game, N, "mission", 1)
        self._occ_index = {tuple(int(x) for x in o): i for i, o in enumerate(self.occupancies)}
        self._committed_iset: int | None = None
        self._routes: list[int | None] = [None] * N
        self._cur = 0
        self._obs_cache = self._build_obs()

    # -- observation (built once; per-sortie fields patched in observe) --------
    def _build_obs(self) -> dict:
        lat = self.lat
        nodes = {_nid(n): {"x": float(n[0]), "y": float(n[1]), "demand": 0.0,
                           "has_depot": n == lat.base}
                 for n in lat.nodes()}
        G = lat.graph()
        edges = {(_nid(u), _nid(v)): {"distance": float(d["w"]), "congestion_level": 0.0}
                 for u, v, d in G.edges(data=True)}
        # the layout's observable projection onto arcs: worst single-hazard probability
        vuln: dict[tuple[str, str], float] = {}
        for u, v in G.edges():
            mid = (np.asarray(u, float) + np.asarray(v, float))[None, :] / 2.0
            p = arc_hazard_prob(mid, self.centres, self.r, self.p_max, self.taper)
            vuln[(_nid(u), _nid(v))] = float(p.max()) if p.size else 0.0
        # menu route node indices in featurize_state's sorted row order
        pos = {nid: i for i, nid in enumerate(sorted(nodes.keys()))}
        menu_idx = [torch.tensor([pos[_nid(n)] for n in route], dtype=torch.long)
                    for route in self.menu]
        cost = np.asarray(self.game.travel_cost, float)
        exposure = 1.0 - self.S.min(axis=1)

        def _mm(x):
            rng_ = x.max() - x.min()
            return (x - x.min()) / rng_ if rng_ > 0 else np.zeros_like(x)

        feats = torch.tensor(np.stack([_mm(cost), _mm(exposure)], axis=1), dtype=torch.float32)
        return {
            "nodes": nodes, "edges": edges,
            "trucks": {0: {"current_node": _nid(lat.base), "destination": None, "load": 0.0,
                           "capacity": 1.0, "assigned_target": _nid(lat.target)}},
            "edge_vulnerability": vuln,
            "menu_route_node_idx": menu_idx, "menu_route_feats": feats,
        }

    # -- episode ---------------------------------------------------------------
    def reset(self) -> dict:
        self._committed_iset = None
        self._routes = [None] * self.config.N
        self._cur = 0
        return self.observe()

    def observe(self) -> dict:
        obs = dict(self._obs_cache)
        obs["active_truck"] = self._cur
        obs["taken_node_frac"] = {}
        if self.config.N > 1:
            taken: dict = {}
            for ri in self._routes[:self._cur]:
                if ri is not None:
                    for n in self.menu[ri]:
                        taken[_nid(n)] = taken.get(_nid(n), 0.0) + 1.0 / self.config.N
            obs["taken_node_frac"] = taken
        return obs

    # -- attacker ----------------------------------------------------------------
    def commit(self, iset_index: int) -> None:
        if not 0 <= iset_index < len(self.game.interdiction_sets):
            raise IndexError("iset_index out of range")
        self._committed_iset = int(iset_index)

    # -- defender ----------------------------------------------------------------
    def current_convoy(self) -> int | None:
        return self._cur if self._cur < self.config.N else None

    def defender_action_mask(self) -> dict:
        return {self._cur: list(range(self.game.n_routes))}

    def route_convoy_by_index(self, ri: int) -> int:
        if self.current_convoy() is None:
            raise RuntimeError("all UAVs already routed this sortie")
        self._routes[self._cur] = int(ri)
        self._cur += 1
        return int(ri)

    def defender_occupancy(self) -> tuple[int, ...]:
        occ = [0] * self.game.n_routes
        for ri in self._routes:
            if ri is not None:
                occ[ri] += 1
        return tuple(occ)
