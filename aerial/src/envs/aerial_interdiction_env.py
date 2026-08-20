"""SAC-trainable aerial interdiction env (gen28): a thin adapter presenting an aerial sector game
through the same observation and menu contract the road multiconvoy env exposes, so routing,
feature extraction, the menu-select head and the protagonist update path work unchanged on the
lattice. Node ids are zero-padded "ii,jj" strings so ``featurize_state``'s sorted row order is
stable; the observable per-arc threat is max_h p(arc, h) under the instance's layout, while the
full field reaches the head via the per-route exposure feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.baselines.multiconvoy_oracle import objective_matrix
from src.envs.aerial_curves import CurveRoute, build_curved_game
from src.envs.aerial_sector import SectorLattice, arc_hazard_prob


def _nid(n) -> str:
    return f"{n[0]:02d},{n[1]:02d}"


@dataclass(frozen=True)
class AerialConfig:
    N: int = 1                       # number of UAVs
    K: int = 1
    menu_select: bool = True         # the aerial policy is always a route-menu policy
    objective: str = "mission"       # at N=1 mission == interception probability
    threshold_m: int = 1


class AerialInterdictionEnv:
    def __init__(self, lat: SectorLattice, menu: list[CurveRoute], centres: np.ndarray, *,
                 K: int = 1, r: float, p_max=0.9, N: int = 1,
                 head_feats: tuple = ("cost", "exposure")):
        """Build the game from a curvature-bounded route menu over the sector lattice.

        Args:
            head_feats: which per-route columns reach the menu head. Cost is reward-irrelevant
                under the mission objective and measurably railroads the policy when included.
        """
        self._head_feats = tuple(head_feats)
        self.lat = lat
        self.menu = menu
        self.centres = centres
        self.r, self.p_max = r, p_max
        self.config = AerialConfig(N=N, K=K)
        self.game, self.S = build_curved_game(lat, menu, centres, K, r=r, p_max=p_max)
        self.occupancies, self.obj_matrix = objective_matrix(self.game, N, "mission", 1)
        self._occ_index = {tuple(int(x) for x in o): i for i, o in enumerate(self.occupancies)}
        self._committed_iset: int | None = None
        self._routes: list[int | None] = [None] * N
        self._cur = 0
        self._obs_cache = self._build_obs()

    # -- observation (built once; per-sortie fields patched in observe) -----------
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
            p = arc_hazard_prob(mid, self.centres, self.r, self.p_max)
            vuln[(_nid(u), _nid(v))] = float(p.max()) if p.size else 0.0
        # menu route node indices in featurize_state's sorted row order
        pos = {nid: i for i, nid in enumerate(sorted(nodes.keys()))}
        menu_idx = [torch.tensor([pos[_nid(n)] for n in c.node_seq], dtype=torch.long)
                    for c in self.menu]
        cost = np.asarray(self.game.travel_cost, float)
        exposure = 1.0 - self.S.min(axis=1)

        def _mm(x):
            rng_ = x.max() - x.min()
            return (x - x.min()) / rng_ if rng_ > 0 else np.zeros_like(x)

        cols = {"cost": _mm(cost), "exposure": _mm(exposure)}
        feats = torch.tensor(np.stack([cols[c] for c in self._head_feats], axis=1),
                             dtype=torch.float32)
        return {
            "nodes": nodes, "edges": edges,
            "trucks": {i: {"current_node": _nid(lat.base), "destination": None, "load": 0.0,
                           "capacity": 1.0, "assigned_target": _nid(lat.target)}
                       for i in range(self.config.N)},
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
                    for n in self.menu[ri].node_seq:
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
