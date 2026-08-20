"""Checks the matrix-free greedy best-response interdictor against the exact best response taken
from the full [occ x iset] objective matrix on Kaliningrad instances: equality at K=1 and the
submodular (1 - 1/e) bound at K=2."""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, greedy_br_attacker, objective_matrix, occupancies)
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS
from src.utils.graph_utils import load_osm_graph_and_demands

_nodes, _edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
_G = nx.Graph()
for _u, _v, _d in _edges:
    _G.add_edge(str(_u), str(_v), w=float(_d.get("distance", 1.0)))


def _game_and_vuln(od, k_extra=8, band=(0.15, 0.95)):
    routes = build_route_set(_G, od[0], od[1], k_extra, "w")
    cand = set().union(*(edges_of_route(r) for r in routes))
    vuln = length_band_vulnerability(_G, cand, band=band, weight="w", norm_edges=list(_G.edges()))
    game = build_interdiction_game(_G, od[0], od[1], 1, k_extra=k_extra, weight="w",
                                   intercept_fn=survival_intercept_fn(vuln))
    return game, {e: vuln[e] for e in cand}


def _random_occ_support(game, N, seed):
    rng = np.random.default_rng(seed)
    occs = occupancies(game.n_routes, N)
    idx = rng.choice(len(occs), size=min(6, len(occs)), replace=False)
    w = rng.random(len(idx)); w /= w.sum()
    return [(tuple(occs[i]), float(w[k])) for k, i in enumerate(idx)]


def _exact_value(game, N, K, occ_support):
    _, M = objective_matrix(game, N, "mission")  # only valid to enumerate at small K
    occs = occupancies(game.n_routes, N)
    idx = {tuple(int(x) for x in o): i for i, o in enumerate(occs)}
    d = np.zeros(len(occs))
    for o, wt in occ_support:
        d[idx[tuple(int(x) for x in o)]] += wt
    gK = build_interdiction_game(_G, *_od, K, k_extra=8, weight="w",
                                 intercept_fn=survival_intercept_fn(_vuln))
    _, MK = objective_matrix(gK, N, "mission")
    return float(best_response_attacker_multi(MK, d)[1])


_od = ("62", "97")
_gm, _vuln = _game_and_vuln(_od)


def test_greedy_equals_exact_K1():
    for seed in range(4):
        supp = _random_occ_support(_gm, 3, seed)
        _, gv = greedy_br_attacker(_gm.route_edges, _vuln, supp, 3, 1, "mission")
        ev = _exact_value(_gm, 3, 1, supp)
        assert abs(gv - ev) < 1e-9, (gv, ev)


def test_greedy_within_guarantee_K2():
    # submodularity guarantees greedy >= (1 - 1/e) * exact optimum
    for seed in range(4):
        supp = _random_occ_support(_gm, 3, seed)
        _, gv = greedy_br_attacker(_gm.route_edges, _vuln, supp, 2, 2, "mission")
        ev = _exact_value(_gm, 3, 2, supp)
        assert gv >= (1.0 - 1.0 / np.e) * ev - 1e-9
        assert gv <= ev + 1e-9   # a BR value cannot exceed the exact best response
