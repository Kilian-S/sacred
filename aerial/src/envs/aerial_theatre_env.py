"""SAC-trainable env adapter for the REAL vec-theatre (gen32). Presents a pre-built theatre
InterdictionGame (routes = continuous flight polylines over real OSM terrain; a resampled hidden
effectiveness field per instance) through the SAME observation/menu contract the lattice aerial
env exposes, so `featurize_state`, `node_index_map`, the menu-select head and the whole
ProtagonistSAC update path work UNCHANGED (the aerial_interdiction_env pattern, terrain-agnostic).

The route "nodes" are coarse 0.5 km waypoint tokens (the same tokens build_theatre_game uses for
its route-edge graph), zero-padded so featurize_state's sorted() row order is stable and
`menu_route_node_idx` can never repeat the 2026-07-09 ordering bug. Per-route head features are
set EXTERNALLY per window by the dynamic trainer (exposure + recency + doctrine); the env supplies
the token graph, the static exposure default, and a per-edge threat projection for the GNN.
"""
from __future__ import annotations

import numpy as np
import torch

from src.baselines.multiconvoy_oracle import objective_matrix
from src.envs.aerial_interdiction_env import AerialConfig


def _tokens(route: np.ndarray) -> list[tuple[float, float]]:
    return [(round(float(p[0]) * 2) / 2, round(float(p[1]) * 2) / 2) for p in route[::4]]


class TheatreEnv:
    """Built from a theatre game + survival matrix + route polylines (all field-specific)."""

    def __init__(self, routes: list[np.ndarray], game, S: np.ndarray, N: int = 3):
        self.routes = routes
        self.game = game
        self.S = S
        self.config = AerialConfig(N=N, K=1)
        self.occupancies, self.obj_matrix = objective_matrix(game, N, "mission", 1)
        self._occ_index = {tuple(int(x) for x in o): i for i, o in enumerate(self.occupancies)}
        self._committed_iset: int | None = None
        self._routes: list[int | None] = [None] * N
        self._cur = 0
        self._obs_cache = self._build_obs()

    def _build_obs(self) -> dict:
        rtoks = [_tokens(r) for r in self.routes]
        alltok = sorted({t for rt in rtoks for t in rt})
        nid = {t: f"{i:04d}" for i, t in enumerate(alltok)}
        nodes = {nid[t]: {"x": float(t[0]), "y": float(t[1]), "demand": 0.0,
                          "has_depot": False} for t in alltok}
        base_t, target_t = rtoks[0][0], rtoks[0][-1]
        nodes[nid[base_t]]["has_depot"] = True
        exposure = 1.0 - self.S.min(axis=1)               # per-route worst exposure
        edges: dict[tuple[str, str], dict] = {}
        edge_exp: dict[tuple[str, str], float] = {}
        for ri, rt in enumerate(rtoks):
            for a, b in zip(rt, rt[1:]):
                if a == b:
                    continue
                key = (nid[a], nid[b])
                d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
                edges.setdefault(key, {"distance": max(d, 0.1), "congestion_level": 0.0})
                edge_exp[key] = max(edge_exp.get(key, 0.0), float(exposure[ri]))
        pos = {n: i for i, n in enumerate(sorted(nodes.keys()))}
        menu_idx = [torch.tensor([pos[nid[t]] for t in rt], dtype=torch.long) for rt in rtoks]

        def _mm(x):
            r = x.max() - x.min()
            return (x - x.min()) / r if r > 0 else np.zeros_like(x)

        feats = torch.tensor(_mm(exposure)[:, None], dtype=torch.float32)
        return {
            "nodes": nodes, "edges": edges,
            "trucks": {i: {"current_node": nid[base_t], "destination": None, "load": 0.0,
                           "capacity": 1.0, "assigned_target": nid[target_t]}
                       for i in range(self.config.N)},
            "edge_vulnerability": edge_exp,
            "menu_route_node_idx": menu_idx, "menu_route_feats": feats,
        }

    def reset(self) -> dict:
        self._committed_iset = None
        self._routes = [None] * self.config.N
        self._cur = 0
        return self.observe()

    def observe(self) -> dict:
        obs = dict(self._obs_cache)
        obs["active_truck"] = self._cur
        obs["taken_node_frac"] = {}
        return obs

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
