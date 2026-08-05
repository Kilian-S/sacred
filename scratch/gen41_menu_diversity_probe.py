#!/usr/bin/env python3
"""gen41 screen 2b (ORACLE-ONLY): does DIVERSE padding (penalised shortest paths, less
corridor overlap, same R) help or hurt the act's structure? Pre-registered in
gen41_deepwindow_zst.md. Same corridor core, same padded count, identical per-edge p_e
(absolute vulnerability norm), so the menus differ only in the padded routes' shapes.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     scratch/gen41_menu_diversity_probe.py
Writes models/runs/gen41_menu_diversity.json
"""
from __future__ import annotations

import itertools
import json
import math
import time

import networkx as nx
import numpy as np
import torch

from scratch.critique_followup_probes import disjoint_subset, rotation_value
from scratch.dyn_exact import karp_mmc
from scratch.gen40_dyn_sensitivity import (
    BAND, N, TAU, best_rotation, enum_windows, inv_vuln_dist, static_stationary,
    uniform_dist, window_losses, antirepeat_stationary)
from scratch.gen41_pool_screen import ext_rotation_value, iid_eq_exact
from scripts.train_b1lite1 import stacked_L
from scripts.train_generalist import CITY_PATHS
from src.baselines.interdiction_oracle import (
    InterdictionGame, build_route_set, edges_of_route, length_band_vulnerability,
    survival_intercept_fn)
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.utils.graph_utils import load_osm_graph_and_demands

torch.set_num_threads(1)
K, KX, PEN = 2, 12, 5.0
INSTANCES = [("kaliningrad", ("23", "242")), ("east_london", ("182", "155")),
             ("gdansk", ("194", "173"))]


def load_G(city):
    np_, ep = CITY_PATHS[city]
    nodes, edges = load_osm_graph_and_demands(np_, ep, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def diverse_menu(G, s, t, n_padded):
    """Corridor core + penalised-shortest-path padding (edges of every accepted route
    reweighted x PEN), same padded count as the standing menu."""
    corridors = build_route_set(G, s, t, 0, "w")
    H = G.copy()
    routes = list(corridors)
    seen = {tuple(r) for r in routes}
    for r in routes:
        for u, v in zip(r, r[1:]):
            H[u][v]["pen"] = H[u][v].get("pen", H[u][v]["w"]) * PEN
    for u, v in H.edges():
        H[u][v].setdefault("pen", H[u][v]["w"])
    tries = 0
    while len(routes) < 3 + n_padded and tries < 60:
        tries += 1
        p = tuple(nx.shortest_path(H, s, t, weight="pen"))
        if p not in seen:
            seen.add(p)
            routes.append(list(p))
        for u, v in zip(p[:-1], p[1:]):
            H[u][v]["pen"] *= PEN
    return routes


def build_game(G, routes, K):
    route_edges = [edges_of_route(r) for r in routes]
    cand = sorted(set().union(*route_edges), key=repr)
    vuln = length_band_vulnerability(G, cand, band=BAND, weight="w",
                                     norm_edges=G.edges())
    fn = survival_intercept_fn(vuln)
    isets = [tuple(c) for c in itertools.combinations(cand, K)]
    payoff = np.array([[fn(re, i) for i in isets] for re in route_edges])
    travel = np.array([sum(G[u][v]["w"] for u, v in zip(r, r[1:])) for r in routes])
    return InterdictionGame(tuple(tuple(r) for r in routes), tuple(route_edges),
                            tuple(isets), payoff, travel, K)


def menu_rows(game, label):
    t0 = time.time()
    L = stacked_L(game, N)
    R = game.n_routes
    re_ = [set(e) for e in game.route_edges]
    dis = disjoint_subset(re_)
    core_union = set().union(*[re_[i] for i in dis])
    own = [1 - len(re_[i] & core_union) / len(re_[i])
           for i in range(R) if i not in dis]
    v_eq, eq = _row_minimiser(L)
    rng = np.random.default_rng(0)
    full_orders = lambda w: min(rotation_value(o, L, TAU, w) for o in
                                [list(range(R))] + [list(rng.permutation(R))
                                                    for _ in range(20)])
    row = dict(label=label, R=R, m=len(dis), n_edges=int(game.payoff.shape[1] ** 0.5)
               if False else len(set().union(*re_)),
               median_own=float(np.median(own)), v_eq=v_eq)
    for w in (3, 6):
        dec_c, counts_c = enum_windows(dis, w, R)
        lw_c = window_losses(counts_c, L, TAU)
        opt_core = karp_mmc(lw_c[:, dis], 3 ** w, 3, 3 ** (w - 1))
        d = dict(opt_core=opt_core,
                 rot=best_rotation(dis, L, TAU, w),
                 anti=antirepeat_stationary(lw_c, dec_c, dis, w),
                 full_rot=full_orders(w),
                 st_uni_core=static_stationary(uniform_dist(dis, R), L, TAU, w),
                 st_inv_core=static_stationary(inv_vuln_dist(dis, L, R), L, TAU, w),
                 st_uni_full=iid_eq_exact(uniform_dist(list(range(R)), R), L, w=w),
                 iid_eq=iid_eq_exact(eq, L, w=w))
        if w == 3 and R ** 3 <= 8000:
            _, counts_f = enum_windows(list(range(R)), 3, R)
            lw_f = window_losses(counts_f, L, TAU)
            d["opt_full"] = karp_mmc(lw_f, R ** 3, R, R ** 2)
            d["padding_value_pct"] = 100 * (opt_core - d["opt_full"]) / max(opt_core, 1e-12)
        if w == 6:
            d["ext_rot"] = ext_rotation_value(game, L, dis)[0]
        row[f"w{w}"] = d
    row["secs"] = round(time.time() - t0, 1)
    return row


def main():
    t0 = time.time()
    out = []
    for city, od in INSTANCES:
        G = load_G(city)
        np_, ep = CITY_PATHS[city]
        env = make_multiconvoy_env(od, N=N, K=K, k_extra_routes=KX, menu_select=True,
                                   edge_vuln_band=BAND, nodes_path=np_, edges_path=ep)
        std = menu_rows(env.game, f"{city}-{od[0]}-{od[1]}-STANDARD")
        n_padded = env.game.n_routes - 3
        routes_d = diverse_menu(G, od[0], od[1], n_padded)
        gd = build_game(G, routes_d, K)
        div = menu_rows(gd, f"{city}-{od[0]}-{od[1]}-DIVERSE")
        out.extend([std, div])
        for r in (std, div):
            w3, w6 = r["w3"], r["w6"]
            pv = w3.get("padding_value_pct")
            print(f"{r['label']}: R={r['R']} |E|={r['n_edges']} own-med "
                  f"{100*r['median_own']:.0f}% veq {r['v_eq']:.3f}", flush=True)
            print(f"   w3: opt_full {w3.get('opt_full', float('nan')):.4f} "
                  f"opt_core {w3['opt_core']:.4f} (padding {pv:.0f}%) | "
                  f"best-rule {min(w3['rot'], w3['anti'], w3['full_rot']):.4f} "
                  f"min-static {min(x for x in (w3['st_uni_core'], w3['st_inv_core'], w3['st_uni_full'], w3['iid_eq']) if x is not None):.4f}",
                  flush=True)
            print(f"   w6: opt_core {w6['opt_core']:.4f} | rot {w6['rot']:.4f} "
                  f"anti {w6['anti']:.4f} ext {w6['ext_rot']:.4f} fullrot "
                  f"{w6['full_rot']:.4f} | statics uniC {w6['st_uni_core']:.4f} "
                  f"invC {w6['st_inv_core']:.4f} uniFULL "
                  f"{w6['st_uni_full'] if w6['st_uni_full'] else float('nan'):.4f} "
                  f"iid {w6['iid_eq'] if w6['iid_eq'] else float('nan'):.4f}", flush=True)
    with open("models/runs/gen41_menu_diversity.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote models/runs/gen41_menu_diversity.json "
          f"({round(time.time() - t0, 1)}s)", flush=True)


if __name__ == "__main__":
    main()
