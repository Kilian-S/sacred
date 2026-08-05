#!/usr/bin/env python3
"""gen41 pool screen (ORACLE-ONLY): select six m=3 ODs per city at the deep-window operating
point (w=6, K=2, kx=12), measure the full two-line rule family per instance (including the
extended rotation), and render per-city PNG contact sheets for Kilian's review.

Pre-registered in experiments/gen41_deepwindow_zst.md (SCREEN section). No training.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     scratch/gen41_pool_screen.py
Writes models/runs/gen41_pool_screen.json + assets/gen41_pool/<city>.png
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.collections import LineCollection

from scratch.critique_followup_probes import disjoint_subset, rotation_value
from scratch.dyn_exact import karp_mmc
from scratch.gen40_dyn_sensitivity import (
    BAND, N, TAU, best_rotation, enum_windows, inv_vuln_dist, static_stationary,
    uniform_dist, window_losses)
from scratch.gen40_dyn_sensitivity import antirepeat_stationary
from scripts.train_b1lite1 import stacked_L
from scripts.train_generalist import CITY_PATHS
from src.baselines.interdiction_oracle import build_route_set
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.utils.graph_utils import load_osm_graph_and_demands

torch.set_num_threads(1)

W, K, KX = 6, 2, 12
R_RANGE = (13, 15)
CITIES = ["kaliningrad", "east_london", "istanbul", "gdansk"]
N_SELECT, N_CAND = 6, 250
BAR_RULE, BAR_STATIC = 1.35, 1.5
OUT_JSON = "models/runs/gen41_pool_screen.json"
PNG_DIR = "assets/gen41_pool"


def iid_eq_exact(eq, L, w=W, max_classes=150_000):
    """Exact stationary value of the equilibrium mixture played iid, by count-class
    enumeration with multinomial weights (the response depends only on window counts)."""
    sup = np.where(eq > 1e-12)[0]
    n_classes = math.comb(len(sup) + w - 1, w)
    if n_classes > max_classes:
        return None
    R = L.shape[0]
    fw = math.factorial(w)
    counts_list, wts = [], []
    for multi in itertools.combinations_with_replacement(range(len(sup)), w):
        counts = np.zeros(R)
        prob = 1.0
        for j in multi:
            counts[sup[j]] += 1
            prob *= eq[sup[j]]
        denom = 1
        for j in set(multi):
            denom *= math.factorial(multi.count(j))
        counts_list.append(counts)
        wts.append((fw // denom) * prob)
    lw = window_losses(np.stack(counts_list), L, TAU)
    return float(np.asarray(wts) @ (lw @ eq))


def diverse_subset(route_edges, dis, cost, L_size):
    """Greedy edge-diversity pick: seed with the corridors, add the route sharing fewest
    edges with the chosen union (ties by lower travel cost)."""
    chosen = list(dis)
    union = set().union(*[route_edges[i] for i in chosen])
    while len(chosen) < L_size:
        best, best_key = None, None
        for i in range(len(route_edges)):
            if i in chosen:
                continue
            key = (len(route_edges[i] & union), cost[i])
            if best_key is None or key < best_key:
                best, best_key = i, key
        if best is None:
            break
        chosen.append(best)
        union |= route_edges[best]
    return chosen


def ext_rotation_value(game, L, dis, w=W):
    """The extended-rotation family: greedy-diverse subsets of length 7 and 8, natural order
    plus 10 seeded shuffles, exact cycle values; returns (best value, best descriptor)."""
    route_edges = [set(e) for e in game.route_edges]
    cost = np.asarray(game.travel_cost, float)
    best_v, best_d = np.inf, None
    for L_size in (7, 8):
        if L_size > len(route_edges):
            continue
        sub = diverse_subset(route_edges, dis, cost, L_size)
        rng = np.random.default_rng(0)
        orders = [list(sub)] + [list(rng.permutation(sub)) for _ in range(10)]
        for o in orders:
            v = rotation_value(o, L, TAU, w)
            if v < best_v:
                best_v, best_d = v, f"L={L_size}"
    return float(best_v), best_d


def screen_city(city):
    nodes_path, edges_path = CITY_PATHS[city]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    deg3 = sorted(n for n, d in G.degree() if d >= 3)
    rng = random.Random(0)
    pairs = set()
    rows = []
    t0 = time.time()
    while len(pairs) < N_CAND and len(pairs) < math.comb(len(deg3), 2):
        s, t = rng.sample(deg3, 2)
        key = (s, t) if s < t else (t, s)
        if key in pairs:
            continue
        pairs.add(key)
        try:
            base = build_route_set(G, s, t, 0, "w")
        except Exception:
            continue
        if len(base) != 3:
            continue
        try:
            env = make_multiconvoy_env((s, t), N=N, K=K, k_extra_routes=KX,
                                       menu_select=True, edge_vuln_band=BAND,
                                       nodes_path=nodes_path, edges_path=edges_path)
        except Exception:
            continue
        game = env.game
        R = game.n_routes
        if not R_RANGE[0] <= R <= R_RANGE[1]:
            continue
        dis = disjoint_subset([set(e) for e in game.route_edges])
        if len(dis) != 3:
            continue
        L = stacked_L(game, N)
        v_eq, eq = _row_minimiser(L)
        if v_eq < 0.05:
            continue
        # the (w=6, K=2) rows, all exact
        dec_c, counts_c = enum_windows(dis, W, R)
        lw_c = window_losses(counts_c, L, TAU)
        opt_core = karp_mmc(lw_c[:, dis], 3 ** W, 3, 3 ** (W - 1))
        rot = best_rotation(dis, L, TAU, W)
        anti = antirepeat_stationary(lw_c, dec_c, dis, W)
        ext, ext_d = ext_rotation_value(game, L, dis)
        st_uni = static_stationary(uniform_dist(dis, R), L, TAU, W)
        st_inv = static_stationary(inv_vuln_dist(dis, L, R), L, TAU, W)
        iid = iid_eq_exact(eq, L)
        statics = [x for x in (st_uni, st_inv, iid) if x is not None]
        best_rule = min(rot, anti, ext)
        row = dict(city=city, od=[s, t], R=R, v_eq=v_eq, opt_core=opt_core,
                   rotation=rot, anti_core=anti, ext_rotation=ext, ext_desc=ext_d,
                   static_uni=st_uni, static_inv=st_inv, iid_eq=iid,
                   best_rule=best_rule, rule_over_opt=best_rule / max(opt_core, 1e-12),
                   static_over_opt=min(statics) / max(opt_core, 1e-12),
                   ext_over_opt=ext / max(opt_core, 1e-12),
                   passes=bool(best_rule / max(opt_core, 1e-12) >= BAR_RULE
                               and min(statics) / max(opt_core, 1e-12) >= BAR_STATIC))
        rows.append(row)
        print(f"{city} {s}-{t} R={R} veq={v_eq:.3f} | opt {opt_core:.4f} rot {rot:.4f} "
              f"anti {anti:.4f} EXT {ext:.4f}({ext_d}) statics {min(statics):.4f} | "
              f"rule/opt {row['rule_over_opt']:.2f} stat/opt {row['static_over_opt']:.2f} "
              f"{'PASS' if row['passes'] else 'fail'}", flush=True)
        if sum(1 for r in rows if r["passes"]) >= 2 * N_SELECT and len(pairs) > 120:
            break
    passers = sorted([r for r in rows if r["passes"]],
                     key=lambda r: -r["rule_over_opt"])[:N_SELECT]
    print(f"{city}: {len(rows)} candidates screened, {sum(r['passes'] for r in rows)} "
          f"passed, {len(passers)} selected ({round(time.time() - t0, 1)}s)", flush=True)
    return rows, passers, G, nodes


def render_city(city, passers, G, nodes, game_lookup):
    os.makedirs(PNG_DIR, exist_ok=True)
    npanels = max(1, len(passers))
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    axes = axes.ravel()
    segs = [[(nodes[u]["x"], nodes[u]["y"]), (nodes[v]["x"], nodes[v]["y"])]
            for u, v in G.edges() if u in nodes and v in nodes]
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i >= len(passers):
            continue
        r = passers[i]
        game = game_lookup[tuple(r["od"])]
        ax.add_collection(LineCollection(segs, colors="#d9d9d9", linewidths=0.5, zorder=1))
        dis = disjoint_subset([set(e) for e in game.route_edges])
        for j, route in enumerate(game.routes):
            pts = [(nodes[n]["x"], nodes[n]["y"]) for n in route if n in nodes]
            if j in dis:
                ci = dis.index(j)
                ax.plot(*zip(*pts), color=["#c62828", "#2e7d32", "#e65100"][ci],
                        lw=2.4, zorder=3)
            else:
                ax.plot(*zip(*pts), color="#5b8def", lw=0.9, alpha=0.75, zorder=2)
        s, t = r["od"]
        ax.scatter([nodes[s]["x"]], [nodes[s]["y"]], s=90, c="black", marker="^", zorder=4)
        ax.scatter([nodes[t]["x"]], [nodes[t]["y"]], s=90, c="black", marker="*", zorder=4)
        ax.set_title(f"{s} -> {t}   R={r['R']}  rule/opt {r['rule_over_opt']:.2f}  "
                     f"stat/opt {r['static_over_opt']:.2f}  ext-rot {r['ext_over_opt']:.2f}",
                     fontsize=10)
        xs = [p[0] for route in game.routes for p in
              [(nodes[n]["x"], nodes[n]["y"]) for n in route if n in nodes]]
        ys = [p[1] for route in game.routes for p in
              [(nodes[n]["x"], nodes[n]["y"]) for n in route if n in nodes]]
        if xs:
            mx = (max(xs) - min(xs)) * 0.35 + 1e-5
            my = (max(ys) - min(ys)) * 0.35 + 1e-5
            ax.set_xlim(min(xs) - mx, max(xs) + mx)
            ax.set_ylim(min(ys) - my, max(ys) + my)
    fig.suptitle(f"gen41 pool candidates: {city} (corridors bold red/green/orange, "
                 f"padded routes blue; ^ origin, * destination)", fontsize=13)
    fig.tight_layout()
    path = f"{PNG_DIR}/{city}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main():
    t0 = time.time()
    out = dict(config=dict(N=N, K=K, kx=KX, w=W, tau=TAU, band=BAND, r_range=R_RANGE,
                           bars=dict(rule=BAR_RULE, static=BAR_STATIC)),
               cities={})
    for city in CITIES:
        rows, passers, G, nodes = screen_city(city)
        nodes_path, edges_path = CITY_PATHS[city]
        game_lookup = {}
        for r in passers:
            env = make_multiconvoy_env(tuple(r["od"]), N=N, K=K, k_extra_routes=KX,
                                       menu_select=True, edge_vuln_band=BAND,
                                       nodes_path=nodes_path, edges_path=edges_path)
            game_lookup[tuple(r["od"])] = env.game
        render_city(city, passers, G, nodes, game_lookup)
        out["cities"][city] = dict(candidates=rows,
                                   selected=[r["od"] for r in passers])
    out["total_secs"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT_JSON} ({out['total_secs']}s)", flush=True)


if __name__ == "__main__":
    main()
