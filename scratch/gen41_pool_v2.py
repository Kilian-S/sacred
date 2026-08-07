#!/usr/bin/env python3
"""gen41 ACT 3 pool expansion (ORACLE-ONLY, rules pre-registered in the ledger): 12
instances per training city across four cities; existing cities take their original
screen's ranked passers re-validated at (w=3, K=2); Kyiv is freshly screened. Gdansk test
pool unchanged. Renders one contact sheet per city for Kilian's asynchronous review.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     scratch/gen41_pool_v2.py
Writes models/runs/gen41_pool_v2.json + assets/gen41_pool/<city>_v2.png
"""
from __future__ import annotations

import json
import math
import random
import time

import networkx as nx
import numpy as np
import torch

from scratch.critique_followup_probes import disjoint_subset
from scratch.dyn_exact import karp_mmc
from scratch.gen40_dyn_sensitivity import (
    BAND, N, TAU, enum_windows, inv_vuln_dist, static_stationary, uniform_dist,
    window_losses)
from scratch.gen41_pool_screen import iid_eq_exact
from scratch.gen41_render import render_city
from scratch.gen41_repr_gate import composed_exact_w3
from scripts.train_b1lite1 import stacked_L
from scripts.train_generalist import CITY_PATHS
from src.baselines.interdiction_oracle import build_route_set
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.utils.graph_utils import load_osm_graph_and_demands

torch.set_num_threads(1)
CITY_PATHS = dict(CITY_PATHS)
CITY_PATHS["kyiv"] = ("data/maps/kyiv/nodes.geojson", "data/maps/kyiv/edges.geojson")
W, K, KX = 3, 2, 12
BAR_RULE, BAR_STATIC = 1.35, 1.8
N_PER_CITY = 12


def w3_rows(game, L):
    """The (w=3, K=2) validation rows for one instance, all exact."""
    R = game.n_routes
    dis = disjoint_subset([set(e) for e in game.route_edges])
    _, counts = enum_windows(list(range(R)), W, R)
    lw = window_losses(counts, L, TAU)
    opt_full = karp_mmc(lw, R ** W, R, R ** (W - 1))
    v_eq, eq = _row_minimiser(L)
    iid = iid_eq_exact(eq, L, w=W)
    st_uni = static_stationary(uniform_dist(dis, R), L, TAU, W)
    st_inv = static_stationary(inv_vuln_dist(dis, L, R), L, TAU, W)
    comp = min(composed_exact_w3(L, game.route_edges, wp) for wp in (1, 2, 3, 4))
    min_static = min(x for x in (st_uni, st_inv, iid) if x is not None)
    return dict(opt_full=opt_full, iid_eq=float(iid), composed_best=comp,
                min_static=min_static, v_eq=v_eq,
                rule_over_opt=comp / max(opt_full, 1e-12),
                static_over_opt=min_static / max(opt_full, 1e-12),
                passes=bool(comp / max(opt_full, 1e-12) >= BAR_RULE
                            and min_static / max(opt_full, 1e-12) >= BAR_STATIC))


def validate(city, od):
    np_, ep = CITY_PATHS[city]
    env = make_multiconvoy_env(tuple(od), N=N, K=K, k_extra_routes=KX, menu_select=True,
                               edge_vuln_band=BAND, nodes_path=np_, edges_path=ep)
    game = env.game
    if not 13 <= game.n_routes <= 15:
        return None
    if len(disjoint_subset([set(e) for e in game.route_edges])) != 3:
        return None
    L = stacked_L(game, N)
    row = w3_rows(game, L)
    row.update(city=city, od=list(od), R=game.n_routes)
    return row


def main():
    t0 = time.time()
    scr = json.load(open("models/runs/gen41_pool_screen.json"))
    out_rows, selected = [], {}
    # existing cities: original selected first, then ranked passers, re-validated at w=3
    for city in ("kaliningrad", "east_london", "istanbul"):
        blob = scr["cities"][city]
        chosen = [tuple(od) for od in blob["selected"]]
        ranked = [tuple(r["od"]) for r in
                  sorted([r for r in blob["candidates"] if r["passes"]],
                         key=lambda r: -r["rule_over_opt"])]
        pool = []
        for od in chosen + [od for od in ranked if od not in chosen]:
            row = validate(city, od)
            if row is None or not row["passes"]:
                if row:
                    out_rows.append(row)
                continue
            out_rows.append(row)
            pool.append(list(od))
            print(f"{city} {od[0]}-{od[1]}: rule/opt {row['rule_over_opt']:.2f} "
                  f"stat/opt {row['static_over_opt']:.2f} SELECT ({len(pool)}/12)",
                  flush=True)
            if len(pool) == N_PER_CITY:
                break
        selected[city] = pool
    # kyiv: fresh screen
    np_, ep = CITY_PATHS["kyiv"]
    nodes, edges = load_osm_graph_and_demands(np_, ep, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    deg3 = sorted(n for n, d in G.degree() if d >= 3)
    rng = random.Random(0)
    pool, seen = [], set()
    while len(pool) < N_PER_CITY and len(seen) < 400:
        s, t = rng.sample(deg3, 2)
        key = (s, t) if s < t else (t, s)
        if key in seen:
            continue
        seen.add(key)
        try:
            if len(build_route_set(G, s, t, 0, "w")) != 3:
                continue
        except Exception:
            continue
        row = validate("kyiv", (s, t))
        if row is None:
            continue
        out_rows.append(row)
        if row["passes"]:
            pool.append([s, t])
            print(f"kyiv {s}-{t}: rule/opt {row['rule_over_opt']:.2f} "
                  f"stat/opt {row['static_over_opt']:.2f} SELECT ({len(pool)}/12)",
                  flush=True)
    selected["kyiv"] = pool
    pool_v2 = {"train": [[c, od[0], od[1]] for c in
                         ("kaliningrad", "east_london", "istanbul", "kyiv")
                         for od in selected[c]],
               "test": json.load(open("models/runs/gen41_pool.json"))["test"]}
    json.dump(pool_v2, open("models/runs/gen41_pool_v2.json", "w"), indent=1)
    json.dump(dict(rows=out_rows, selected=selected,
                   secs=round(time.time() - t0, 1)),
              open("models/runs/gen41_pool_v2_screen.json", "w"), indent=1)
    print(f"pool v2: {len(pool_v2['train'])} train + {len(pool_v2['test'])} test "
          f"({round(time.time() - t0, 1)}s)", flush=True)
    # contact sheets (two pages of 6 per city, v2 suffix)
    import scratch.gen41_render as gr
    gr.CITY_PATHS = dict(CITY_PATHS)
    for city, ods in selected.items():
        for page in range(2):
            gr.CITY_PATHS[f"{city}_v2p{page + 1}"] = CITY_PATHS[city]
    for city, ods in selected.items():
        rows_by_od = {tuple(r["od"]): dict(r, ext_over_opt=float("nan"))
                      for r in out_rows if r["city"] == city}
        for page in range(2):
            chunk = [tuple(od) for od in ods[page * 6:(page + 1) * 6]]
            if not chunk:
                continue
            try:
                render_city(f"{city}_v2p{page + 1}", chunk, rows_by_od)
            except Exception as e:  # noqa: BLE001 - sheets are review aids, never gating
                print(f"render {city} p{page + 1} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
