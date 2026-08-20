"""The walker game (gen28): the policy is a next-waypoint walker on the forward DAG, giving
position-conditioned decisions per sortie rather than one menu pick. Every strategy, the
walker's flights and every reference row alike, is scored as a waypoint-leg polyline under the
hazard-rate line integral (kappa = -ln(1-p_max)/r). Legs make survival arc-separable, which
gives the walker an exact exploitability by dynamic programming over the lattice, with no Monte
Carlo. The menu class is a strict subset of the walker class, so the menu-restricted equilibrium
is a reference anchor rather than the walker's optimum.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame
from src.envs.aerial_sector import SectorLattice, arc_hazard_prob

Node = tuple[int, int]


def _nid(n) -> str:
    return f"{n[0]:02d},{n[1]:02d}"


def build_arc_survival(lat: SectorLattice, centres: np.ndarray, r, p_max,
                       n_quad: int = 9) -> tuple[list, dict, np.ndarray]:
    """Per-arc single-hazard survival by n_quad-point midpoint quadrature.

    A[a, h] = exp(-integral of hazard h's rate along leg a), over the DAG's directed arcs.

    Returns:
        (arcs, arc_index, A).
    """
    pm = np.broadcast_to(np.asarray(p_max, float), (len(centres),))
    rr = np.broadcast_to(np.asarray(r, float), (len(centres),))
    kappa = -np.log(np.clip(1.0 - pm, 1e-12, 1.0)) / rr
    G = lat.graph()
    arcs = sorted(G.edges())
    A = np.empty((len(arcs), len(centres)))
    ts = (np.arange(n_quad) + 0.5) / n_quad
    for a, (u, v) in enumerate(arcs):
        p0, p1 = np.asarray(u, float), np.asarray(v, float)
        pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        ds = np.linalg.norm(p1 - p0) / n_quad
        d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2)
        taper = np.clip(1.0 - d / rr[None, :], 0.0, None)
        A[a] = np.exp(-(kappa[None, :] * taper * ds).sum(axis=0))
    return arcs, {arc: i for i, arc in enumerate(arcs)}, A


def path_survival(path: tuple, arc_index: dict, A: np.ndarray) -> np.ndarray:
    """s[h] = survival of a node path against each single hazard (product over its legs)."""
    idx = [arc_index[(path[i], path[i + 1])] for i in range(len(path) - 1)]
    return A[idx].prod(axis=0)


def build_polyline_game(lat: SectorLattice, node_paths: list[tuple], centres: np.ndarray,
                        K: int, arc_index: dict, A: np.ndarray) -> InterdictionGame:
    """The menu-restricted reference game: routes are node paths and
    payoff[i, iset] = 1 - prod over the path's legs of the iset's joint survival."""
    S1 = np.stack([path_survival(p, arc_index, A) for p in node_paths])   # [R, H]
    H = len(centres)
    isets = list(itertools.combinations(range(H), K)) if K <= H else [tuple(range(H))]
    logS = np.log(np.clip(S1, 1e-300, 1.0))
    idx = np.asarray(isets, dtype=int)
    if len(node_paths) * len(isets) > 60_000_000:
        raise MemoryError("polyline reference game too large; use the greedy yardstick")
    payoff = 1.0 - np.exp(logS[:, idx].sum(axis=2))
    lengths = np.array([sum(np.hypot(b[0] - a[0], b[1] - a[1])
                            for a, b in zip(p, p[1:])) for p in node_paths])
    route_edges = tuple(frozenset(frozenset((a, b)) for a, b in zip(p, p[1:]))
                        for p in node_paths)
    return InterdictionGame(tuple(node_paths), route_edges, tuple(tuple(t) for t in isets),
                            payoff, lengths, K)


class AerialWalkerEnv:
    """Next-waypoint walker env presenting the road observation contract in node mode.

    There are no menu keys anywhere: ``select_action`` scores candidate nodes. One UAV walks
    column by column from base to target, and interception is resolved analytically by the
    trainer from the realised path.
    """

    def __init__(self, lat: SectorLattice, centres: np.ndarray, *, K: int = 1, r, p_max=0.9):
        self.lat = lat
        self.centres = centres
        self.K = K
        self.r, self.p_max = r, p_max
        self.G = lat.graph()
        self.arcs, self.arc_index, self.A = build_arc_survival(lat, centres, r, p_max)
        # only nodes that can still reach the target are legal, which prunes dead ends behind
        # obstacles and funnels the final columns home
        canreach = nx.ancestors(self.G, lat.target) | {lat.target}
        self._succ = {n: sorted(t for t in self.G.successors(n) if t in canreach)
                      for n in canreach if n != lat.target}
        # goal-distance field (feature column 12): shortest leg-length to target on the DAG
        rev = self.G.reverse()
        dist = nx.single_source_dijkstra_path_length(rev, lat.target, weight="w")
        self._goal = {_nid(n): float(dist.get(n, 99.0)) for n in self.G.nodes}
        self._obs_static = self._build_static_obs()
        self.pos: Node = lat.base
        self.path: list[Node] = [lat.base]

    def _build_static_obs(self) -> dict:
        lat = self.lat
        nodes = {_nid(n): {"x": float(n[0]), "y": float(n[1]), "demand": 0.0,
                           "has_depot": n == lat.base}
                 for n in lat.nodes()}
        edges = {(_nid(u), _nid(v)): {"distance": float(d["w"]), "congestion_level": 0.0}
                 for u, v, d in self.G.edges(data=True)}
        vuln: dict = {}
        for u, v in self.G.edges():
            mid = (np.asarray(u, float) + np.asarray(v, float))[None, :] / 2.0
            p = arc_hazard_prob(mid, self.centres, self.r, self.p_max)
            vuln[(_nid(u), _nid(v))] = float(p.max()) if p.size else 0.0
        return {"nodes": nodes, "edges": edges, "edge_vulnerability": vuln}

    # -- episode -----------------------------------------------------------------
    def reset(self) -> dict:
        self.pos = self.lat.base
        self.path = [self.lat.base]
        return self.observe()

    def observe(self) -> dict:
        obs = dict(self._obs_static)
        obs["active_truck"] = 0
        obs["trucks"] = {0: {"current_node": _nid(self.pos), "destination": None, "load": 0.0,
                             "capacity": 1.0, "assigned_target": _nid(self.lat.target)}}
        obs["goal_dists"] = {0: self._goal}
        return obs

    def action_mask(self) -> dict:
        return {0: [_nid(n) for n in self._succ.get(self.pos, [])]}

    def step(self, chosen_nid: str) -> bool:
        nxt = next(n for n in self._succ[self.pos] if _nid(n) == chosen_nid)
        self.pos = nxt
        self.path.append(nxt)
        return self.pos[0] == self.lat.nx - 1        # done at the target column

    def realised_survival(self) -> np.ndarray:
        return path_survival(tuple(self.path), self.arc_index, self.A)


def build_dag_menu(env: AerialWalkerEnv, R: int = 40, seed: int = 0
                   ) -> tuple[list[tuple], dict[float, list[int]]]:
    """The reference menu as native DAG paths.

    Canonical lane paths per spacing come first (rounded to lattice rows, and absent where
    blocked), then seeded diverse legal walks by greedy max-min on row vectors, up to ``R``.

    Returns:
        (paths, lane_sets keyed by spacing).
    """
    from src.envs.aerial_curves import lane_offsets
    from src.envs.aerial_sector import lane_path
    lat = env.lat
    paths: list[tuple] = []
    seen: dict[tuple, int] = {}
    lane_sets: dict[float, list[int]] = {}
    for rc in (0.8, 1.2, 1.6, 2.0):
        idxs = []
        for off in lane_offsets(lat, rc):
            lp = lane_path(lat, int(round(off)))
            if lp is None or lp[-1] != lat.target:
                continue
            if any((lp[i], lp[i + 1]) not in env.arc_index for i in range(len(lp) - 1)):
                continue
            if lp not in seen:
                seen[lp] = len(paths)
                paths.append(lp)
            idxs.append(seen[lp])
        if idxs:
            lane_sets[rc] = sorted(set(idxs))
    rng = np.random.default_rng(seed)
    cands: list[tuple] = []
    tries = 0
    while len(cands) < 8 * R and tries < 200 * R:
        tries += 1
        node, acc = lat.base, [lat.base]
        while node != lat.target:
            succ = env._succ[node]
            node = succ[int(rng.integers(len(succ)))]
            acc.append(node)
        t = tuple(acc)
        if t not in seen and t not in cands:
            cands.append(t)
    def rows(p):
        return np.array([n[1] for n in p], dtype=float)
    while len(paths) < R and cands:
        if paths:
            chosen = np.stack([rows(p) for p in paths])
            d = [float(np.min(np.linalg.norm(chosen - rows(c)[None, :], axis=1)))
                 for c in cands]
            best = int(np.argmax(d))
        else:
            best = 0
        c = cands.pop(best)
        seen[c] = len(paths)
        paths.append(c)
    return paths, lane_sets


def walker_policy_probs(prot, env: AerialWalkerEnv):
    """pi(next | node) for every non-terminal node from one encoder pass.

    The head is applied per node on shared embeddings, so the result is exact, not sampled.
    """
    import torch
    from src.agents.networks import featurize_state, node_index_map
    from src.agents.sac import _clip_ea, _clip_x
    obs = env.reset()
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    prot.actor.eval()
    with torch.no_grad():
        h = prot.actor.encoder(pyg.x, pyg.edge_index, pyg.edge_attr)
        out = {}
        for node, succ in env._succ.items():
            if not succ:
                continue
            idxs = [n2i[_nid(s)] for s in succ]
            probs, _ = prot.actor.head(h, n2i[_nid(node)], idxs, None)
            out[node] = (succ, probs.cpu().numpy())
    prot.actor.train()
    return out


def walker_exploitability(prot, env: AerialWalkerEnv, isets: list[tuple] | None = None,
                          chunk: int = 4000) -> tuple[float, float]:
    """Exact worst-case interception of the walker policy.

    Expected joint survival per interdiction set is computed by backward dynamic programming
    over the DAG (legs are arc-separable) and maximised over isets. No Monte Carlo.

    Returns:
        (worst-case interception, expected path length).
    """
    pi = walker_policy_probs(prot, env)
    H = len(env.centres)
    if isets is None:
        isets = ([(h,) for h in range(H)] if env.K == 1
                 else list(itertools.combinations(range(H), env.K)))
    lat = env.lat
    # expected length (iset-independent)
    elen = {lat.target: 0.0}
    for col in range(lat.nx - 2, -1, -1):
        for node in [n for n in pi if n[0] == col]:
            succ, p = pi[node]
            elen[node] = float(sum(pj * (np.hypot(s[0] - node[0], s[1] - node[1]) + elen[s])
                                   for s, pj in zip(succ, p)))
    worst = 0.0
    A = env.A
    ai = env.arc_index
    for c0 in range(0, len(isets), chunk):
        blk = isets[c0:c0 + chunk]
        idx = np.asarray(blk, dtype=int)                     # [B, K]
        # per-arc joint survival for the block: [n_arcs, B]
        SA = A[:, idx].prod(axis=2) if idx.shape[1] > 1 else A[:, idx[:, 0]]
        V = {lat.target: np.ones(len(blk))}
        for col in range(lat.nx - 2, -1, -1):
            for node in [n for n in pi if n[0] == col]:
                succ, p = pi[node]
                acc = np.zeros(len(blk))
                for s, pj in zip(succ, p):
                    acc += pj * SA[ai[(node, s)]] * V[s]
                V[node] = acc
        worst = max(worst, float((1.0 - V[lat.base]).max()))
    return worst, elen[lat.base]
