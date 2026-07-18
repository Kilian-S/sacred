"""gen29: the multi-OD (three-stream) coordination interdiction game (GEN29_MULTIOD_HANDOFF.md).

One base s supplies THREE destinations t1,t2,t3 whose candidate route sets share corridor edges.
Each sortie: one convoy per stream; a K=1 interdictor commits one edge from the UNION candidate
list (hidden). Objective = loss-averse mission P(>=1 of the 3 lost) (the additive objective is
provably correlation-gap-free, so the coupling is load-bearing: B3 law extended). The defender is
ONE policy routing the streams SEQUENTIALLY (stream 0, then 1 observing 0's committed route, then 2
observing both): coordination lives inside one policy's sequential joint action (the pattern that
trains; avoids the gen18 independent-learner boundary).

The env presents the standard SAC observation/menu contract (featurize_state + menu-select head):
for the active stream, `menu_route_node_idx` = that stream's routes' node indices in
featurize_state's SORTED row order (the node-ordering contract), `menu_route_feats` =
[worst-vulnerability, OVERLAP-WITH-COMMITTED] (the second column is the coordination signal,
delivered undiluted at the head; NO cost channel, per the railroading lessons), and
`taken_node_frac` = per-node fraction of earlier streams routed through it. The exact joint
distribution is recovered by conditional enumeration (no Monte Carlo). Additive/new file.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (build_route_set, edges_of_route,
                                               length_band_vulnerability)
from src.env.graph_env import GraphEnv
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS
from src.utils.graph_utils import load_osm_graph_and_demands

NodeId = str


@dataclass
class MultiODConfig:
    s: NodeId
    targets: tuple                      # (t1, t2, t3)
    K: int = 1
    k_extra_routes: int = 8
    band: tuple = (0.15, 0.95)
    interception_loss: float = 10.0
    objective: str = "mission"


class MultiODInterdictionEnv:
    def __init__(self, graph: nx.Graph, config: MultiODConfig, graph_env: GraphEnv):
        self.graph = graph
        self.config = config
        self.graph_env = graph_env
        s, targets = config.s, config.targets
        self.F = len(targets)
        # per-stream candidate routes; shared union candidate edge list
        self.route_sets = [build_route_set(graph, s, t, config.k_extra_routes, "w")
                           for t in targets]
        self.route_edges = [[edges_of_route(r) for r in rs] for rs in self.route_sets]
        cand = sorted(set().union(*(es for flow in self.route_edges for es in flow)),
                      key=lambda e: tuple(sorted(map(str, e))))
        self.cand_edges = cand
        vuln = length_band_vulnerability(graph, cand, band=config.band, weight="w",
                                         norm_edges=list(graph.edges()))
        self.edge_vuln = {tuple(sorted(e, key=str)): float(vuln[e]) for e in cand}
        # per-stream single-edge SURVIVAL matrices S_f [R_f, E]
        self.S = []
        for flow in self.route_edges:
            P = np.zeros((len(flow), len(cand)))
            for k, e in enumerate(cand):
                for i, es in enumerate(flow):
                    P[i, k] = vuln[e] if e in es else 0.0
            self.S.append(1.0 - P)
        self.worst_vuln = [1.0 - S.min(axis=1) for S in self.S]         # per-route peak exposure
        # interdiction sets over the union candidate edges
        E = len(cand)
        self.isets = ([(e,) for e in range(E)] if config.K == 1
                      else list(itertools.combinations(range(E), config.K)))
        # full observable threat map (whole graph) for the featuriser edge column
        full = length_band_vulnerability(graph, (frozenset(e) for e in graph.edges()
                                                 if e[0] != e[1]), band=config.band, weight="w",
                                         norm_edges=list(graph.edges()))
        self.edge_vulnerability = {tuple(sorted(e, key=str)): p for e, p in full.items()}
        self.blind = False           # causal control: zero the coordination channel
        self._committed: list[int | None] = [None] * self.F
        self._cur = 0
        self._iset: int | None = None
        # menu node-index caches per stream (sorted featurize row order)
        self._menu_idx = self._build_menu_idx()
        self._obj_matrix = None      # built lazily (large); the eval yardstick

    # -- joint payoff (eval yardstick) -----------------------------------------
    @property
    def obj_matrix(self) -> np.ndarray:
        """M[joint_route_tuple, iset] = P(>=1 lost) under that iset. Row order = itertools.product
        over streams (stream 0 outermost)."""
        if self._obj_matrix is None:
            per_flow = []
            idx = np.asarray(self.isets)
            for S in self.S:
                logS = np.log(np.clip(S, 1e-300, 1.0))
                per_flow.append(np.exp(logS[:, idx].sum(axis=2)))       # [R_f, n_isets]
            surv = per_flow[0]
            for nxt in per_flow[1:]:
                surv = (surv[:, None, :] * nxt[None, :, :]).reshape(-1, surv.shape[-1])
            self._obj_matrix = 1.0 - surv
        return self._obj_matrix

    def exploitability_of_joint_dist(self, dist: np.ndarray) -> float:
        return float((np.asarray(dist) @ self.obj_matrix).max())

    @property
    def n_joint(self) -> int:
        return int(np.prod([len(rs) for rs in self.route_sets]))

    # -- SAC observation / menu contract ---------------------------------------
    def _build_menu_idx(self):
        import torch
        pos = {str(n): i for i, n in enumerate(sorted(self.graph_env.observe()["nodes"].keys()))}
        out = []
        for rs in self.route_sets:
            out.append([torch.tensor([pos[str(n)] for n in route if str(n) in pos],
                                     dtype=torch.long) for route in rs])
        return out

    def reset(self) -> dict:
        self._committed = [None] * self.F
        self._cur = 0
        self._iset = None
        self.graph_env.reset()
        for f, tr in enumerate(self.graph_env.trucks.values()):
            tr.assigned_target = self.config.targets[f]
        return self.observe()

    def commit(self, iset_index: int) -> None:
        self._iset = int(iset_index)

    def current_stream(self) -> int | None:
        return self._cur if self._cur < self.F else None

    def defender_action_mask(self) -> dict:
        return {self._cur: list(range(len(self.route_sets[self._cur])))}

    def route_stream_by_index(self, ri: int) -> int:
        self._committed[self._cur] = int(ri)
        self._cur += 1
        return int(ri)

    def set_committed(self, routes) -> None:
        """Explicit committed-route prefix (for exact conditional enumeration)."""
        self._committed = [None] * self.F
        for f, r in enumerate(routes):
            self._committed[f] = int(r)
        self._cur = len(routes)

    def _overlap_with_committed(self, stream: int) -> np.ndarray:
        """Per-candidate-route edge-share fraction with the union of earlier committed routes'
        edges (the coordination signal for the active stream)."""
        committed_edges = set()
        for f in range(stream):
            if self._committed[f] is not None:
                committed_edges |= self.route_edges[f][self._committed[f]]
        out = np.zeros(len(self.route_sets[stream]))
        if committed_edges:
            for i, es in enumerate(self.route_edges[stream]):
                out[i] = len(es & committed_edges) / max(1, len(es))
        return out

    def observe(self) -> dict:
        import torch
        obs = dict(self.graph_env.observe())
        cur = self._cur
        obs["active_truck"] = cur
        obs["edge_vulnerability"] = self.edge_vulnerability
        if cur < self.F:
            obs["menu_route_node_idx"] = self._menu_idx[cur]
            cost = np.array([1.0] * len(self.route_sets[cur]))         # placeholder (no cost head)
            wv = self.worst_vuln[cur]

            def _mm(x):
                rng = x.max() - x.min()
                return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)
            overlap = np.zeros(len(wv)) if self.blind else self._overlap_with_committed(cur)
            feats = np.stack([_mm(wv), overlap], axis=1)
            obs["menu_route_feats"] = torch.tensor(feats, dtype=torch.float32)
        # taken_node_frac: fraction of earlier streams routed through each node (zeroed when blind)
        taken: dict = {}
        if not self.blind:
            for f in range(cur):
                if self._committed[f] is not None:
                    for n in self.route_sets[f][self._committed[f]]:
                        taken[str(n)] = taken.get(str(n), 0.0) + 1.0 / self.F
        obs["taken_node_frac"] = taken
        return obs

    # -- resolution ------------------------------------------------------------
    def committed_routes(self) -> tuple:
        return tuple(self._committed)

    def joint_index(self, routes) -> int:
        idx = 0
        for f, r in enumerate(routes):
            idx = idx * len(self.route_sets[f]) + int(r)
        return idx

    def mission_failure(self, routes, iset_index: int) -> float:
        surv = 1.0
        for f, r in enumerate(routes):
            for e in self.isets[iset_index]:
                surv *= self.S[f][r, e]
        return 1.0 - surv


def make_multiod_env(s: NodeId, targets, *, K: int = 1, k_extra_routes: int = 8,
                     band=(0.15, 0.95), interception_loss: float = 10.0,
                     nodes_path: str = _DEFAULT_NODES, edges_path: str = _DEFAULT_EDGES,
                     tasks_path: str = _DEFAULT_TASKS) -> MultiODInterdictionEnv:
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)
    for nid in nodes:
        nodes[nid]["demand"] = 0.0
    for t in targets:
        if t not in nodes:
            raise ValueError(f"target {t} not in graph")
        nodes[t]["demand"] = 1.0
    nodes[s]["has_depot"] = True
    F = len(targets)
    graph_env = GraphEnv(nodes=nodes, edges=edges, num_trucks=F, truck_capacity=1.0,
                         truck_starting_nodes=[s] * F, max_time=400)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    cfg = MultiODConfig(s=str(s), targets=tuple(str(t) for t in targets), K=K,
                        k_extra_routes=k_extra_routes, band=band,
                        interception_loss=interception_loss)
    return MultiODInterdictionEnv(G, cfg, graph_env)
