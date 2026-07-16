"""Multi-convoy interdiction environment (gen08 Phase M / Obj-2): N convoys route base -> FOB against
a hidden K-asset interdictor, under SOFT (probabilistic) interception and a LOSS-AVERSE
(mission-failure) objective.

Built additively on the single-convoy `interdiction.py` machinery. One sortie:
  1. the interdictor COMMITS K interdiction assets (hidden);
  2. the defender routes the N convoys SEQUENTIALLY (convoy 0, 1, ..., N-1); each `observe()` exposes
     the routes already chosen by earlier convoys, so a policy MAY condition later convoys on earlier
     ones (the mission-optimal joint strategy is CORRELATED, see `multiconvoy_oracle`); an
     independent policy simply ignores those columns;
  3. `resolve()` samples each convoy's interception (independent seeded Bernoulli under the committed
     set) and returns the zero-sum reward (defender = -objective - travel cost).
The env is validated against `src/baselines/multiconvoy_oracle.py` (it reproduces loss_det/loss_mixed
by Monte-Carlo: the G-M1 gate), which is also the exploitability yardstick for a learned defender.
Disjoint-route (first-hop) instances first, matching the oracle findings; the walk trie (shared-edge)
is a later extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import best_response_attacker_multi, objective_matrix
from src.env.graph_env import GraphEnv
from src.utils.graph_utils import load_osm_graph_and_demands

NodeId = Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_NODES = str(PROJECT_ROOT / "data/maps/kaliningrad_simplified_30m/kaliningrad_nodes.geojson")
_DEFAULT_EDGES = str(PROJECT_ROOT / "data/maps/kaliningrad_simplified_30m/kaliningrad_edges.geojson")
_DEFAULT_TASKS = str(PROJECT_ROOT / "data/maps/koenigsberg1.json")


@dataclass
class MultiConvoyConfig:
    od: tuple[NodeId, NodeId]
    N: int = 2                              # number of convoys (fleet size)
    K: int = 1                             # interdiction assets committed per sortie
    interception_loss: float = 1.0         # reward magnitude of the objective (zero-sum)
    travel_cost_weight: float = 0.0        # small defender-only per-distance cost (0 = pure game)
    k_extra_routes: int = 0                # 0 = clean edge-disjoint routes only (first-hop routing)
    weight: str = "w"
    edge_vuln_band: tuple[float, float] | None = (0.15, 0.95)  # soft interception band; None = hard
    absolute_vuln_norm: bool = True        # True = map arc length->prob over ALL graph arcs (intrinsic,
                                           # cross-instance comparable); False = per-route-set normalise
    menu_select: bool = False              # True = convoy picks a ROUTE INDEX (shared-edge menu);
                                           # False = first-hop (disjoint). Menu is scalable, no walk trie.
    objective: str = "mission"             # mission (P>=1) | linear (E[frac]) | threshold (P>=m)
    threshold_m: int = 1
    seed: int = 0                          # interception-sampling RNG seed
    greedy_br: bool = False                # gen26: MATRIX-FREE mode for K past the exact wall (K>=4).
                                           # The game object is built at K=1 (routes/costs/per-edge
                                           # vulnerabilities only); obj_matrix is NOT built; the
                                           # attacker/eval use the verified submodular greedy BR
                                           # (A4-core, (1-1/e) guarantee). Default False = the exact
                                           # path, byte-identical.


@dataclass
class MultiConvoyOutcome:
    routes: tuple[int, ...]                # per-convoy route index
    iset_index: int
    caught: tuple[bool, ...]              # per-convoy interception
    n_caught: int
    objective_value: float
    defender_reward: float
    attacker_reward: float
    travel_cost: float


class MultiConvoyInterdictionEnv:
    def __init__(self, graph: nx.Graph, config: MultiConvoyConfig, graph_env: GraphEnv | None = None):
        self.graph = graph
        self.config = config
        self.graph_env = graph_env
        s, t = config.od
        if s not in graph or t not in graph:
            raise ValueError(f"OD nodes {s!r},{t!r} not in graph")
        self.base, self.fob = s, t
        intercept_fn = None
        self.edge_vulnerability: dict = {}   # (u, v) key -> p_e; the OBSERVABLE threat map (A1)
        if config.edge_vuln_band is not None:
            routes = build_route_set(graph, s, t, config.k_extra_routes, config.weight)
            cand = set().union(*(edges_of_route(r) for r in routes))
            norm = graph.edges() if config.absolute_vuln_norm else None
            vuln = length_band_vulnerability(graph, cand, band=tuple(config.edge_vuln_band),
                                             weight=config.weight, norm_edges=norm)
            intercept_fn = survival_intercept_fn(vuln)
            # The FULL-GRAPH intrinsic map for the observation (featurise edge col 4): under the
            # absolute norm a road's p_e is graph-intrinsic, so the policy sees the whole threat
            # map, not just the candidate edges - the map-conditioning signal ZST step 1 requires.
            full = length_band_vulnerability(
                graph, (frozenset(e) for e in graph.edges() if e[0] != e[1]),  # skip self-loops
                band=tuple(config.edge_vuln_band), weight=config.weight, norm_edges=norm)
            self.edge_vulnerability = {tuple(sorted(e, key=repr)): p for e, p in full.items()}
        # gen26 greedy-BR mode: build the game at K=1 (routes, costs, per-edge vulnerabilities);
        # the TRUE K lives in config.K and is honoured by the greedy attacker/eval paths below.
        game_K = 1 if config.greedy_br else config.K
        self.game = build_interdiction_game(graph, s, t, game_K, k_extra=config.k_extra_routes,
                                            weight=config.weight, intercept_fn=intercept_fn)
        if self.game.n_routes < 2:
            raise ValueError("interdiction game needs >= 2 candidate routes (higher-connectivity OD)")
        self._rng = np.random.default_rng(config.seed)
        # first-hop route lookup (disjoint routes: first hop identifies the route)
        self.routes_by_first_hop: dict[NodeId, list[int]] = {}
        for i, r in enumerate(self.game.routes):
            self.routes_by_first_hop.setdefault(r[1], []).append(i)
        self.first_hops: list[NodeId] = sorted(self.routes_by_first_hop, key=repr)
        self._cand_edges = sorted(set().union(*self.game.route_edges), key=repr)
        # oracle objective matrix (occupancies x interdiction sets): the exploitability yardstick.
        # greedy_br mode (K >= 4, past the exact wall): NO matrix; the yardstick is the verified
        # submodular greedy BR (A4-core) via exploitability_of_occupancy_dist below.
        if config.greedy_br:
            from src.baselines.multiconvoy_oracle import occupancies as _occupancies
            self.occupancies = _occupancies(self.game.n_routes, config.N)
            self.obj_matrix = None
            # per-edge vulnerability from the K=1 game itself (payoff[:, j] = p_e on routes
            # crossing single-edge iset j, else 0), so hard AND soft interception both work and
            # the greedy attacker sees exactly the exact game's edge model.
            self.vuln_by_edge = {iset[0]: float(self.game.payoff[:, j].max())
                                 for j, iset in enumerate(self.game.interdiction_sets)}
        else:
            self.occupancies, self.obj_matrix = objective_matrix(
                self.game, config.N, config.objective, config.threshold_m)
            self.vuln_by_edge = {}
        self._occ_index = {tuple(int(x) for x in o): i for i, o in enumerate(self.occupancies)}
        self._committed_iset: int | None = None
        self._committed_edges: tuple | None = None   # greedy_br mode: explicit K-edge commitment
        self._convoy_routes: list[int | None] = [None] * config.N
        self._cur = 0
        # Menu-mode per-instance conditioning, attached to every observation so it travels WITH the
        # transition into the replay buffer (the A1 generalist samples instances per sortie, and a
        # replayed instance-i transition must be scored under instance i's menu/features, never the
        # net's current attributes). Cached once; observations share the references (cheap).
        self._menu_idx_cache: list | None = None
        self._menu_feats_cache = None
        if config.menu_select and graph_env is not None:
            import torch as _torch
            self._menu_idx_cache = [_torch.tensor(r, dtype=_torch.long)
                                    for r in self.menu_route_node_idx()]
            cost = np.asarray(self.game.travel_cost, dtype=float)
            worst = self.game.payoff.max(axis=1)

            def _mm(x):
                rng_ = x.max() - x.min()
                return (x - x.min()) / rng_ if rng_ > 0 else np.zeros_like(x)

            self._menu_feats_cache = _torch.tensor(
                np.stack([_mm(cost), _mm(worst)], axis=1), dtype=_torch.float32)

    # -- episode ---------------------------------------------------------------
    def reset(self) -> dict | None:
        self._committed_iset = None
        self._committed_edges = None
        self._convoy_routes = [None] * self.config.N
        self._cur = 0
        if self.graph_env is not None:
            self.graph_env.reset()
            for truck in self.graph_env.trucks.values():
                truck.assigned_target = self.fob
        return self.observe() if self.graph_env is not None else None

    # -- attacker (interdictor) ------------------------------------------------
    def attacker_action_mask(self) -> dict:
        return {"can_wait": False, "levels_by_edge": {self._edge_key(e): [1.0] for e in self._cand_edges}}

    def commit(self, iset_index: int) -> None:
        if not 0 <= iset_index < len(self.game.interdiction_sets):
            raise IndexError("iset_index out of range")
        self._committed_iset = int(iset_index)

    def commit_set(self, edges) -> None:
        """greedy_br mode: commit an explicit set of K edges (frozensets), no iset enumeration."""
        self._committed_edges = tuple(frozenset(e) for e in edges)

    def route_interception(self, edges) -> "np.ndarray":
        """Per-route interception probability under an explicit committed edge set (greedy_br
        mode's analytic-reward companion to game.payoff[:, iset]): p_r = 1 - prod(1 - p_e) over
        the committed edges the route crosses."""
        p = np.zeros(self.game.n_routes)
        edges = [frozenset(e) for e in edges]
        for r, re_ in enumerate(self.game.route_edges):
            surv = 1.0
            for e in edges:
                if e in re_:
                    surv *= 1.0 - self.vuln_by_edge.get(e, 1.0)
            p[r] = 1.0 - surv
        return p

    def commit_edge(self, edge) -> None:
        """K=1: commit by edge (frozenset or (u,v) key)."""
        target = frozenset(edge)
        for j, iset in enumerate(self.game.interdiction_sets):
            if len(iset) == 1 and iset[0] == target:
                self._committed_iset = j
                return
        raise ValueError(f"edge {tuple(edge)} is not a candidate interdiction edge")

    # -- defender (sequential per-convoy routing) ------------------------------
    def current_convoy(self) -> int | None:
        return self._cur if self._cur < self.config.N else None

    def defender_action_mask(self) -> dict:
        """The current convoy's action set: route ids (menu-select, shared-edge) or first hops."""
        if self.config.menu_select:
            return {self._cur: list(range(self.game.n_routes))}
        return {self._cur: list(self.first_hops)}

    def route_convoy_by_index(self, ri: int) -> int:
        """Route the current convoy on route index ``ri`` directly (menu-select), then advance."""
        if self.current_convoy() is None:
            raise RuntimeError("all convoys already routed this sortie")
        self._convoy_routes[self._cur] = int(ri)
        self._cur += 1
        return int(ri)

    def menu_route_node_idx(self) -> list[list[int]]:
        """Per-route node indices in featurize_state's SORTED row order (for the route menu-select
        head: each route is scored by the mean-pooled embedding of these nodes). Sorted, NOT dict
        insertion order: featurize_state sorts node ids (the 2026-07-09 node-ordering fix)."""
        pos = {str(n): i for i, n in enumerate(sorted(self.graph_env.observe()["nodes"].keys()))}
        return [[pos[str(n)] for n in route if str(n) in pos] for route in self.game.routes]

    def observe(self) -> dict:
        if self.graph_env is None:
            raise RuntimeError("no graph_env attached; use make_multiconvoy_env()")
        obs = dict(self.graph_env.observe())
        obs["active_truck"] = self._cur
        if self.edge_vulnerability:
            obs["edge_vulnerability"] = self.edge_vulnerability
        if self._menu_idx_cache is not None:
            obs["menu_route_node_idx"] = self._menu_idx_cache      # per-instance, rides the transition
            obs["menu_route_feats"] = self._menu_feats_cache       # [R, 2] = (cost, worst-vuln), [0,1]
        earlier = [r for r in self._convoy_routes[:self._cur] if r is not None]
        obs["routed_convoys"] = list(earlier)
        # route-correlation signal for the followers' menu head: per-node fraction of EARLIER convoys
        # whose route passes through that node. A candidate route overlapping the leader's route then
        # scores high, and the leader's EXACT route scores highest -> the followers can "follow".
        taken: dict = {}
        for r in earlier:
            for n in self.game.routes[r]:
                taken[str(n)] = taken.get(str(n), 0.0) + 1.0 / self.config.N
        obs["taken_node_frac"] = taken
        return obs

    def route_of_first_hop(self, first_hop: NodeId) -> int:
        idxs = self.routes_by_first_hop.get(first_hop)
        if not idxs:
            raise ValueError(f"no candidate route starts with hop {first_hop!r}")
        return min(idxs, key=lambda i: self.game.travel_cost[i])

    def route_convoy_first_hop(self, first_hop: NodeId) -> int:
        """Route the current convoy via ``first_hop`` and advance to the next convoy."""
        if self.current_convoy() is None:
            raise RuntimeError("all convoys already routed this sortie")
        ri = self.route_of_first_hop(first_hop)
        self._convoy_routes[self._cur] = ri
        if self.graph_env is not None:
            self.graph_env.trucks[self._cur].current_node = first_hop  # for the next observe()
        self._cur += 1
        return ri

    def set_convoy_routes(self, route_indices) -> None:
        """Explicitly set all N convoy routes (evaluation / the G-M1 gate)."""
        route_indices = list(route_indices)
        if len(route_indices) != self.config.N:
            raise ValueError(f"expected {self.config.N} routes, got {len(route_indices)}")
        self._convoy_routes = [int(r) for r in route_indices]
        self._cur = self.config.N

    def defender_occupancy(self) -> tuple[int, ...]:
        occ = [0] * self.game.n_routes
        for r in self._convoy_routes:
            if r is not None:
                occ[r] += 1
        return tuple(occ)

    # -- resolution ------------------------------------------------------------
    def resolve(self) -> MultiConvoyOutcome:
        if self._committed_iset is None and self._committed_edges is None:
            raise RuntimeError("attacker has not committed this sortie")
        if any(r is None for r in self._convoy_routes):
            raise RuntimeError("not all convoys have been routed")
        if self._committed_edges is not None:            # greedy_br mode: explicit K-edge set
            j = -1
            p = self.route_interception(self._committed_edges)
            self._committed_edges = None
        else:
            j = self._committed_iset
            p = self.game.payoff[:, j]                   # per-route interception probability
        caught = tuple(bool(self._rng.random() < float(p[r])) for r in self._convoy_routes)
        n_caught = int(sum(caught))
        obj = self._objective_value(n_caught)
        travel = float(sum(self.game.travel_cost[r] for r in self._convoy_routes))
        defender_reward = -self.config.interception_loss * obj - self.config.travel_cost_weight * travel
        attacker_reward = self.config.interception_loss * obj
        out = MultiConvoyOutcome(tuple(int(r) for r in self._convoy_routes), j, caught, n_caught,
                                 obj, defender_reward, attacker_reward, travel)
        self._committed_iset = None
        return out

    def _objective_value(self, n_caught: int) -> float:
        if self.config.objective == "linear":
            return n_caught / self.config.N
        thr = 1 if self.config.objective == "mission" else self.config.threshold_m
        return 1.0 if n_caught >= thr else 0.0

    # -- exploitability (env-side yardstick) -----------------------------------
    def occupancy_dist_of(self, route_samples) -> np.ndarray:
        """Empirical occupancy distribution from a list of per-convoy route tuples (a policy's
        Monte-Carlo joint routing)."""
        dist = np.zeros(len(self.occupancies))
        for routes in route_samples:
            occ = [0] * self.game.n_routes
            for r in routes:
                occ[r] += 1
            dist[self._occ_index[tuple(occ)]] += 1.0
        return dist / dist.sum() if dist.sum() > 0 else dist

    def exploitability_of_occupancy_dist(self, occupancy_dist: np.ndarray) -> float:
        """Loss of a defender occupancy distribution under the best-response interdictor.
        Exact (obj_matrix) below the wall; the verified submodular greedy BR (A4-core,
        (1 - 1/e) guarantee, disclosed wherever cited) in greedy_br mode."""
        if self.obj_matrix is None:
            from src.baselines.multiconvoy_oracle import greedy_br_attacker
            support = [(tuple(int(x) for x in o), float(w))
                       for o, w in zip(self.occupancies, occupancy_dist) if w > 1e-12]
            total = sum(w for _, w in support)
            support = [(o, w / total) for o, w in support]
            _, loss = greedy_br_attacker(self.game.route_edges, self.vuln_by_edge, support,
                                         self.config.N, self.config.K,
                                         self.config.objective, self.config.threshold_m)
            return float(loss)
        _, loss = best_response_attacker_multi(self.obj_matrix, occupancy_dist)
        return loss

    # -- helpers ---------------------------------------------------------------
    def occupancy_to_routes(self, occ) -> list[int]:
        """A route list realising occupancy vector ``occ`` (e.g. (0,2,0) -> [1,1])."""
        routes: list[int] = []
        for r, c in enumerate(occ):
            routes.extend([r] * int(c))
        return routes

    def _edge_key(self, edge) -> tuple:
        u, v = tuple(edge)
        return self.graph_env._edge_key(u, v) if self.graph_env is not None else tuple(sorted((u, v), key=repr))


def make_multiconvoy_env(
    od: tuple[NodeId, NodeId] = ("110", "135"),
    *,
    N: int = 2,
    K: int = 1,
    interception_loss: float = 1.0,
    travel_cost_weight: float = 0.0,
    k_extra_routes: int = 0,
    edge_vuln_band: tuple[float, float] | None = (0.15, 0.95),
    absolute_vuln_norm: bool = True,
    menu_select: bool = False,
    objective: str = "mission",
    threshold_m: int = 1,
    seed: int = 0,
    greedy_br: bool = False,
    nodes_path: str = _DEFAULT_NODES,
    edges_path: str = _DEFAULT_EDGES,
    tasks_path: str = _DEFAULT_TASKS,
) -> MultiConvoyInterdictionEnv:
    """Build the multi-convoy interdiction env on the Kaliningrad graph. Default OD 110->135 (the
    soft-band connectivity instance from the oracle findings); soft interception + mission objective."""
    s, t = od
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)
    if s not in nodes or t not in nodes:
        raise ValueError(f"OD nodes {s!r},{t!r} not in graph")
    for node_id in nodes:
        nodes[node_id]["demand"] = 0.0
    nodes[t]["demand"] = 1.0
    nodes[s]["has_depot"] = True
    graph_env = GraphEnv(nodes=nodes, edges=edges, num_trucks=N, truck_capacity=1.0,
                         truck_starting_nodes=[s] * N, max_time=400)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    cfg = MultiConvoyConfig(od=(str(s), str(t)), N=N, K=K, interception_loss=interception_loss,
                            travel_cost_weight=travel_cost_weight, k_extra_routes=k_extra_routes,
                            weight="w", edge_vuln_band=edge_vuln_band,
                            absolute_vuln_norm=absolute_vuln_norm, menu_select=menu_select,
                            objective=objective, threshold_m=threshold_m, seed=seed,
                            greedy_br=greedy_br)
    return MultiConvoyInterdictionEnv(G, cfg, graph_env=graph_env)
