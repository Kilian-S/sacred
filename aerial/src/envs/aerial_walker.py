"""gen28 v2.3: the WALKER game (the pre-registered structural re-aim after the v2.2 fail).

The policy is a NEXT-WAYPOINT walker on the forward DAG (the proven road single-vehicle
pattern, B2-P3): 12 position-conditioned decisions per sortie instead of one menu pick, so
replay-state diversity is structural (the v2.2 failure was the pre-flagged saturating-bandit
cell: N=1 x menu-select = one state per instance).

GEOMETRY CLASS (v2.3, disclosed): every strategy - the walker's flights AND every reference
row (equilibrium, lane rules, menu stacks, tabular FP) - is scored as a WAYPOINT-LEG POLYLINE
under the same hazard-rate line integral (kappa = -ln(1-p_max)/r; the dead-centre calibration
is exact on a straight leg). Legs make survival ARC-SEPARABLE, which is what gives the walker
an EXACT exploitability via dynamic programming over the lattice (no Monte Carlo anywhere).
Spline rendering remains cosmetic in the viz; the payoff geometry is the legs. The menu class
is a strict subset of the walker class (disclosed: the walker may in principle beat the
menu-restricted equilibrium; it is a reference anchor, not the walker's optimum).
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
    """Per-arc single-hazard survival: arcs = the DAG's directed arcs; A[a, h] =
    exp(-integral of hazard h's rate along leg a) via n_quad-point midpoint quadrature.
    Returns (arcs, arc_index, A)."""
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
    """The menu-restricted reference game in the v2.3 geometry class: routes = node paths,
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
    """Next-waypoint walker env presenting the road observation contract in NODE mode
    (no menu keys anywhere: select_action scores candidate NODES). One UAV; the sortie is a
    column-by-column walk base -> target; interception is resolved analytically by the trainer
    from the realised path (the env carries the survival machinery)."""

    def __init__(self, lat: SectorLattice, centres: np.ndarray, *, K: int = 1, r, p_max=0.9):
        self.lat = lat
        self.centres = centres
        self.K = K
        self.r, self.p_max = r, p_max
        self.G = lat.graph()
        self.arcs, self.arc_index, self.A = build_arc_survival(lat, centres, r, p_max)
        # crash-proof topology (the repo dogma): only nodes that can still REACH the target are
        # legal, which both prunes dead ends behind obstacles and funnels the final columns home.
        canreach = nx.ancestors(self.G, lat.target) | {lat.target}
        self._succ = {n: sorted(t for t in self.G.successors(n) if t in canreach)
                      for n in canreach if n != lat.target}
        # goal-distance field (featurise col 12): shortest leg-length to target on the DAG
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
    """The v2.3 reference menu as NATIVE DAG paths: canonical lane paths per spacing (rounded
    to lattice rows; absent where blocked - on the double pinch no lane exists, correctly),
    then seeded diverse legal walks (greedy max-min on row vectors) up to R. Returns
    (paths, lane_sets by spacing)."""
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
    """pi(next | node) for every non-terminal node from ONE encoder pass (the head is applied
    per node on shared embeddings; exact, no sampling)."""
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
    """EXACT worst-case interception of the walker policy: expected joint survival per
    interdiction set by backward DP over the DAG (arc-separable legs), maximised over isets.
    Also returns the expected path length (the fleet-cost column). No Monte Carlo."""
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
