"""Multi-convoy interdiction environment, N convoys routing base to FOB against a hidden interdictor.

Each sortie the interdictor commits K assets, the defender then routes the convoys one at a time,
with every observation exposing the routes earlier convoys already took so that a policy may
correlate them, and ``resolve()`` samples the interceptions and returns the zero-sum reward.
Interception is soft and the objective is loss-averse, counting mission failure rather than expected
losses. `src/baselines/multiconvoy_oracle.py` supplies the exploitability yardstick.
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
    absolute_vuln_norm: bool = True        # True = map arc length->prob over ALL graph arcs, which is
                                           # graph-intrinsic and cross-instance comparable;
                                           # False = normalise within the route set
    menu_select: bool = False              # True = convoy picks a route index from a shared-edge
                                           # menu; False = first-hop routing on disjoint routes
    objective: str = "mission"             # mission (P>=1) | linear (E[frac]) | threshold (P>=m)
    threshold_m: int = 1
    seed: int = 0                          # interception-sampling RNG seed
    greedy_br: bool = False                # matrix-free mode for K past the exact-enumeration wall.
                                           # The game is built at K=1 and no objective matrix is
                                           # formed; the attacker and the evaluation instead use the
                                           # submodular greedy best response, which carries a
                                           # (1 - 1/e) guarantee. False = the exact path.


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
        self.edge_vulnerability: dict = {}   # (u, v) key -> p_e, the observable threat map
        if config.edge_vuln_band is not None:
            routes = build_route_set(graph, s, t, config.k_extra_routes, config.weight)
            cand = set().union(*(edges_of_route(r) for r in routes))
            norm = graph.edges() if config.absolute_vuln_norm else None
            vuln = length_band_vulnerability(graph, cand, band=tuple(config.edge_vuln_band),
                                             weight=config.weight, norm_edges=norm)
            intercept_fn = survival_intercept_fn(vuln)
            # the full-graph map the observation carries. Under the absolute norm a road's p_e is
            # graph-intrinsic, so the policy sees the whole threat map, not only candidate edges.
            full = length_band_vulnerability(
                graph, (frozenset(e) for e in graph.edges() if e[0] != e[1]),  # skip self-loops
                band=tuple(config.edge_vuln_band), weight=config.weight, norm_edges=norm)
            self.edge_vulnerability = {tuple(sorted(e, key=repr)): p for e, p in full.items()}
        # in greedy-BR mode the game itself is built at K=1; the true K lives in config.K and is
        # honoured by the greedy attacker and evaluation paths below
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
        # objective matrix over occupancies x interdiction sets, the exploitability yardstick. In
        # greedy-BR mode no matrix is built and the greedy best response is the yardstick instead.
        if config.greedy_br:
            from src.baselines.multiconvoy_oracle import occupancies as _occupancies
            self.occupancies = _occupancies(self.game.n_routes, config.N)
            self.obj_matrix = None
            # per-edge vulnerability read off the K=1 game, where payoff[:, j] is p_e on the routes
            # crossing single-edge set j and zero elsewhere, so the greedy attacker sees the same
            # edge model as the exact path under both hard and soft interception
            self.vuln_by_edge = {iset[0]: float(self.game.payoff[:, j].max())
                                 for j, iset in enumerate(self.game.interdiction_sets)}
        else:
            self.occupancies, self.obj_matrix = objective_matrix(
                self.game, config.N, config.objective, config.threshold_m)
            self.vuln_by_edge = {}
        self._occ_index = {tuple(int(x) for x in o): i for i, o in enumerate(self.occupancies)}
        self._committed_iset: int | None = None
        self._committed_edges: tuple | None = None   # explicit K-edge commitment, greedy-BR mode
        self._convoy_routes: list[int | None] = [None] * config.N
        self._cur = 0
        # Menu-mode conditioning is attached to every observation so that it travels with the
        # transition into the replay buffer: a generalist samples a fresh instance per sortie, and a
        # replayed transition must be scored under its own instance's menu and features rather than
        # whichever instance is current. Cached once, and observations share the references.
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
        """Commit an explicit set of K edges, bypassing interdiction-set enumeration."""
        self._committed_edges = tuple(frozenset(e) for e in edges)

    def route_interception(self, edges) -> "np.ndarray":
        """Per-route interception probability under an explicit committed edge set.

        ``p_r = 1 - prod(1 - p_e)`` over the committed edges that route r crosses, the greedy-BR
        counterpart of a column of ``game.payoff``.
        """
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
        """Commit a single interdiction edge, given as a frozenset or a ``(u, v)`` key."""
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
        """The current convoy's action set, either menu route ids or first hops."""
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
        """Per-route node indices, in the sorted row order ``featurize_state`` produces.

        The menu-select head scores each route by the mean-pooled embedding of these nodes, so the
        indices must follow sorted node id order rather than dict insertion order.
        """
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
            obs["menu_route_feats"] = self._menu_feats_cache       # [R, 2] cost and worst-case p, in [0, 1]
        earlier = [r for r in self._convoy_routes[:self._cur] if r is not None]
        obs["routed_convoys"] = list(earlier)
        # route-correlation signal for the followers' menu head, the per-node fraction of earlier
        # convoys whose route passes through that node. A candidate route overlapping an earlier one
        # then scores high, so followers are able to trail the leader.
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
        """Explicitly set all N convoy routes, used by evaluation paths."""
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
        if self._committed_edges is not None:            # explicit K-edge set, greedy-BR mode
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
        """Empirical occupancy distribution over a list of per-convoy route tuples."""
        dist = np.zeros(len(self.occupancies))
        for routes in route_samples:
            occ = [0] * self.game.n_routes
            for r in routes:
                occ[r] += 1
            dist[self._occ_index[tuple(occ)]] += 1.0
        return dist / dist.sum() if dist.sum() > 0 else dist

    def exploitability_of_occupancy_dist(self, occupancy_dist: np.ndarray) -> float:
        """Loss of a defender occupancy distribution under the best-response interdictor.

        Exact from the objective matrix below the enumeration wall, and from the submodular greedy
        best response with its (1 - 1/e) guarantee in greedy-BR mode.
        """
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
    """Build the multi-convoy interdiction env on the Kaliningrad graph.

    The defaults give the well-connected 110 -> 135 OD pair under soft interception and the mission
    objective.
    """
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
