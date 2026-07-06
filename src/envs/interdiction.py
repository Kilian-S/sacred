"""Single-convoy interdiction environment (gen08 / Obj 2): convoy routing as a security game.

A convoy must travel base -> FOB across a contested network. An interdictor COMMITS K interdiction
assets to edges each sortie, HIDDEN from the convoy; the convoy routes; crossing an interdicted edge
is an INTERCEPTION (terminal, high loss). Reactivity is useless (the ambush is set before the move);
the only defence is an unpredictable, mixed-strategy route. See `REDESIGN_INTERDICTION.md`.

The game CORE is at route granularity, the level the equilibrium oracle
(`src/baselines/interdiction_oracle.py`) solves, so the env reproduces `loss_det`/`loss_mixed`
exactly (the G1 env-fidelity gate). Agent interfaces reuse the existing SAC:
  * defender action = the FIRST HOP out of the base (for edge-disjoint routes the first hop
    identifies the route), a NODE selection like the next-hop protagonist;
  * attacker action (K=1 first) = one EDGE to interdict, like the antagonist.
`observe()` returns a GraphEnv observation (base convoy, FOB target) that the existing
`featurize_state` consumes, so `ProtagonistSAC`/`AntagonistSAC` act on it unchanged. The multi-branch
next-hop physics is a later extension; for the disjoint-route single-convoy headline the first-hop
decision IS the route decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from src.baselines.interdiction_oracle import InterdictionGame, build_interdiction_game
from src.env.graph_env import GraphEnv
from src.utils.graph_utils import load_osm_graph_and_demands

NodeId = Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_NODES = str(PROJECT_ROOT / "data/maps/kaliningrad_simplified_30m/kaliningrad_nodes.geojson")
_DEFAULT_EDGES = str(PROJECT_ROOT / "data/maps/kaliningrad_simplified_30m/kaliningrad_edges.geojson")
_DEFAULT_TASKS = str(PROJECT_ROOT / "data/maps/koenigsberg1.json")


@dataclass
class InterdictionConfig:
    od: tuple[NodeId, NodeId]              # (base, FOB)
    K: int = 1                             # interdiction assets committed per sortie
    interception_loss: float = 1.0         # reward magnitude of an interception (zero-sum)
    travel_cost_weight: float = 0.0        # small defender-only per-distance cost (0 = pure game)
    k_extra_routes: int = 8
    weight: str = "w"


@dataclass
class InterdictionOutcome:
    route_index: int
    iset_index: int
    intercepted: bool
    defender_reward: float
    attacker_reward: float
    travel_cost: float


class InterdictionEnv:
    """The game core + a GraphEnv-backed observation. One sortie = attacker commits an interdiction
    set (hidden) -> defender picks a route (via first hop) -> interception + reward. Repeated across
    sorties (the RL / co-evolution). Both agents act only on the graph (neither observes the other's
    realised action): the hidden-commit Stackelberg structure."""

    def __init__(self, graph: nx.Graph, config: InterdictionConfig, graph_env: GraphEnv | None = None):
        self.graph = graph
        self.config = config
        self.graph_env = graph_env
        s, t = config.od
        if s not in graph or t not in graph:
            raise ValueError(f"OD nodes {s!r},{t!r} not in graph")
        self.base, self.fob = s, t
        self.game: InterdictionGame = build_interdiction_game(
            graph, s, t, config.K, k_extra=config.k_extra_routes, weight=config.weight)
        if self.game.n_routes < 2:
            raise ValueError("interdiction game needs >= 2 candidate routes (pick a higher-connectivity OD)")
        self.routes_by_first_hop: dict[NodeId, list[int]] = {}
        for i, r in enumerate(self.game.routes):
            self.routes_by_first_hop.setdefault(r[1], []).append(i)
        self.first_hops: list[NodeId] = sorted(self.routes_by_first_hop, key=repr)
        # candidate interdiction edges (undirected frozensets), as (u,v) keys for the attacker mask.
        self._cand_edges = sorted(set().union(*self.game.route_edges), key=repr)
        self._committed_iset: int | None = None

    # -- episode ---------------------------------------------------------------
    def reset(self) -> dict:
        """Start a sortie: reset the convoy to base (target FOB), clear the interdiction. Returns the
        defender observation."""
        self._committed_iset = None
        if self.graph_env is not None:
            self.graph_env.reset()
            self.graph_env.trucks[0].assigned_target = self.fob
        return self.observe()

    def observe(self) -> dict:
        """GraphEnv observation with the convoy (truck 0) at base and target = FOB, plus active_truck
        so `featurize_state` populates truck features. Requires a graph_env (built by the factory)."""
        if self.graph_env is None:
            raise RuntimeError("no graph_env attached; use make_interdiction_env()")
        obs = dict(self.graph_env.observe())
        obs["active_truck"] = 0
        return obs

    # -- attacker (interdictor) -------------------------------------------------
    @property
    def interdiction_sets(self):
        return self.game.interdiction_sets

    def attacker_action_mask(self) -> dict:
        """Antagonist-format mask over candidate interdiction edges (level 1.0 = interdict)."""
        levels = {self._edge_key(e): [1.0] for e in self._cand_edges}
        return {"can_wait": False, "levels_by_edge": levels}

    def commit(self, iset_index: int) -> None:
        if not 0 <= iset_index < len(self.game.interdiction_sets):
            raise IndexError("iset_index out of range")
        self._committed_iset = iset_index

    def commit_edge(self, edge) -> None:
        """K=1: commit by edge (accepts a frozenset or a (u,v) key; matches the antagonist action)."""
        target = frozenset(edge)
        for j, iset in enumerate(self.game.interdiction_sets):
            if len(iset) == 1 and iset[0] == target:
                self._committed_iset = j
                return
        raise ValueError(f"edge {tuple(edge)} is not a candidate interdiction edge")

    # -- defender (convoy) ------------------------------------------------------
    def defender_action_mask(self) -> dict:
        """Protagonist-format mask: truck 0 chooses its first hop out of base (identifies the route)."""
        return {0: list(self.first_hops)}

    def route_of_first_hop(self, first_hop: NodeId) -> int:
        idxs = self.routes_by_first_hop.get(first_hop)
        if not idxs:
            raise ValueError(f"no candidate route starts with hop {first_hop!r}")
        return min(idxs, key=lambda i: self.game.travel_cost[i])

    def resolve(self, route_index: int) -> InterdictionOutcome:
        """Defender commits to a route; compute interception + rewards vs the committed interdiction."""
        if self._committed_iset is None:
            raise RuntimeError("attacker has not committed this sortie")
        j = self._committed_iset
        intercepted = bool(self.game.payoff[route_index, j] > 0.0)
        travel = float(self.game.travel_cost[route_index])
        loss = self.config.interception_loss if intercepted else 0.0
        defender_reward = -loss - self.config.travel_cost_weight * travel
        attacker_reward = loss
        self._committed_iset = None
        return InterdictionOutcome(route_index, j, intercepted, defender_reward, attacker_reward, travel)

    def resolve_first_hop(self, first_hop: NodeId) -> InterdictionOutcome:
        return self.resolve(self.route_of_first_hop(first_hop))

    # -- helpers ---------------------------------------------------------------
    def shortest_route_index(self) -> int:
        return int(min(range(self.game.n_routes), key=lambda i: self.game.travel_cost[i]))

    def _edge_key(self, edge) -> tuple:
        u, v = tuple(edge)
        return self.graph_env._edge_key(u, v) if self.graph_env is not None else tuple(sorted((u, v), key=repr))


def make_interdiction_env(
    od: tuple[NodeId, NodeId] = ("33", "71"),
    *,
    K: int = 1,
    interception_loss: float = 1.0,
    travel_cost_weight: float = 0.0,
    k_extra_routes: int = 8,
    nodes_path: str = _DEFAULT_NODES,
    edges_path: str = _DEFAULT_EDGES,
    tasks_path: str = _DEFAULT_TASKS,
) -> InterdictionEnv:
    """Build the single-convoy interdiction env on the Kaliningrad graph: base (depot, truck 0) ->
    FOB (demand), plus the NetworkX graph for the oracle/game core. Default OD 33->71 (edge-conn 6)."""
    s, t = od
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)
    if s not in nodes or t not in nodes:
        raise ValueError(f"OD nodes {s!r},{t!r} not in graph")
    for node_id in nodes:
        nodes[node_id]["demand"] = 0.0
    nodes[t]["demand"] = 1.0          # the FOB resupply demand (the convoy's target)
    nodes[s]["has_depot"] = True      # base
    graph_env = GraphEnv(nodes=nodes, edges=edges, num_trucks=1, truck_capacity=1.0,
                         truck_starting_nodes=[s], max_time=400)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    cfg = InterdictionConfig(od=(str(s), str(t)), K=K, interception_loss=interception_loss,
                             travel_cost_weight=travel_cost_weight, k_extra_routes=k_extra_routes, weight="w")
    return InterdictionEnv(G, cfg, graph_env=graph_env)
