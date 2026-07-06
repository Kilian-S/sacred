"""Single-convoy interdiction environment (gen08 / Obj 2): convoy routing as a security game.

A convoy must travel base -> FOB across a contested network. An interdictor COMMITS K interdiction
assets to edges each sortie, HIDDEN from the convoy; the convoy routes; crossing an interdicted edge
is an INTERCEPTION (terminal, high loss). Reactivity is useless (the ambush is set before the move);
the only defence is an unpredictable, mixed-strategy route. See `REDESIGN_INTERDICTION.md`.

This module is the game CORE at route granularity, the level the equilibrium oracle
(`src/baselines/interdiction_oracle.py`) solves, so the env reproduces `loss_det`/`loss_mixed`
exactly (the G1 env-fidelity gate). Agent interfaces are chosen to reuse the existing SAC:
  * defender action = the FIRST HOP out of the base (for edge-disjoint routes the first hop
    identifies the route), a NODE selection like the next-hop protagonist;
  * attacker action (K=1 first) = one EDGE to interdict, like the antagonist.
The full next-hop routing physics (multi-branch paths) is the I2 extension; for the disjoint-route
single-convoy headline the first-hop decision IS the route decision, and this core is what SACRED
trains against and is validated against the oracle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from src.baselines.interdiction_oracle import InterdictionGame, build_interdiction_game

NodeId = Any


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
    """The game core. One sortie = attacker commits an interdiction set (hidden) -> defender picks a
    route -> interception + reward. Repeated across sorties (the RL / co-evolution). Both agents act
    only on the graph (neither observes the other's realised action), which is the hidden-commit
    Stackelberg structure."""

    def __init__(self, graph: nx.Graph, config: InterdictionConfig):
        self.graph = graph
        self.config = config
        s, t = config.od
        if s not in graph or t not in graph:
            raise ValueError(f"OD nodes {s!r},{t!r} not in graph")
        self.game: InterdictionGame = build_interdiction_game(
            graph, s, t, config.K, k_extra=config.k_extra_routes, weight=config.weight)
        if self.game.n_routes < 2:
            raise ValueError("interdiction game needs >= 2 candidate routes (pick a higher-connectivity OD)")
        # first-hop node -> route indices starting with it (edge-disjoint routes: 1:1).
        self.routes_by_first_hop: dict[NodeId, list[int]] = {}
        for i, r in enumerate(self.game.routes):
            self.routes_by_first_hop.setdefault(r[1], []).append(i)
        self.first_hops: list[NodeId] = sorted(self.routes_by_first_hop, key=repr)
        self._committed_iset: int | None = None

    # -- attacker (interdictor) -------------------------------------------------
    @property
    def interdiction_sets(self):
        return self.game.interdiction_sets

    def commit(self, iset_index: int) -> None:
        """Attacker commits an interdiction set for this sortie (HIDDEN from the defender)."""
        if not 0 <= iset_index < len(self.game.interdiction_sets):
            raise IndexError("iset_index out of range")
        self._committed_iset = iset_index

    def commit_edge(self, edge: frozenset) -> None:
        """K=1 convenience: commit by edge (matches the antagonist's edge-selection action)."""
        target = frozenset(edge)
        for j, iset in enumerate(self.game.interdiction_sets):
            if len(iset) == 1 and iset[0] == target:
                self._committed_iset = j
                return
        raise ValueError(f"edge {edge} is not a candidate interdiction edge")

    # -- defender (convoy) ------------------------------------------------------
    def route_of_first_hop(self, first_hop: NodeId) -> int:
        """The candidate route taken when the convoy's first hop out of the base is ``first_hop``
        (shortest such route; for disjoint routes it is unique)."""
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
        attacker_reward = loss                        # zero-sum on interception
        self._committed_iset = None
        return InterdictionOutcome(route_index, j, intercepted, defender_reward, attacker_reward, travel)

    # -- oracle-linked references (for baselines + G1) --------------------------
    def shortest_route_index(self) -> int:
        return int(min(range(self.game.n_routes), key=lambda i: self.game.travel_cost[i]))
