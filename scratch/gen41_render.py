#!/usr/bin/env python3
"""gen41 contact-sheet renderer v2 (post-review): every padded route individually visible.

Changes vs the screen's first render, per Kilian's 2026-08-05 review: (i) the gdansk
70-297 -> 303-15 swap applied (recorded in the ledger); (ii) padded routes each get a
distinct colour, with the edges NOT shared with the corridor union drawn thicker so each
route's own detour is discernible despite 59-100% edge overlap with the corridors;
(iii) per-panel annotation counts corridors + padded and the median own-edge share.

Run: OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen41_render.py
Rewrites assets/gen41_pool/<city>.png (previous versions preserved in git history).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

from scratch.critique_followup_probes import disjoint_subset
from scripts.train_generalist import CITY_PATHS
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.utils.graph_utils import load_osm_graph_and_demands

BAND, KX = (0.15, 0.95), 12
PNG_DIR = "assets/gen41_pool"
SWAPS = {"gdansk": {("70", "297"): ("303", "15")}}
CORRIDOR_COLOURS = ["#c62828", "#2e7d32", "#e65100"]


def edge_key(u, v):
    return frozenset((u, v))


def render_city(city, selected, rows_by_od):
    nodes_path, edges_path = CITY_PATHS[city]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v))
    segs = [[(nodes[u]["x"], nodes[u]["y"]), (nodes[v]["x"], nodes[v]["y"])]
            for u, v in G.edges() if u in nodes and v in nodes]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    axes = axes.ravel()
    pad_cmap = plt.get_cmap("tab20")
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i >= len(selected):
            continue
        od = tuple(selected[i])
        r = rows_by_od.get(od, {})
        env = make_multiconvoy_env(od, N=3, K=2, k_extra_routes=KX, menu_select=True,
                                   edge_vuln_band=BAND, nodes_path=nodes_path,
                                   edges_path=edges_path)
        game = env.game
        re_ = [set(e) for e in game.route_edges]
        dis = disjoint_subset(re_)
        core_union = set().union(*[re_[j] for j in dis])
        ax.add_collection(LineCollection(segs, colors="#e0e0e0", linewidths=0.5, zorder=1))
        own_shares = []
        pad_i = 0
        for j, route in enumerate(game.routes):
            pts = [(nodes[n]["x"], nodes[n]["y"]) for n in route if n in nodes]
            if j in dis:
                continue
            col = pad_cmap(pad_i % 20)
            pad_i += 1
            ax.plot(*zip(*pts), color=col, lw=0.9, alpha=0.65, zorder=2)
            own = [(a, b) for a, b in zip(route[:-1], route[1:])
                   if edge_key(a, b) not in core_union]
            own_shares.append(len(own) / max(1, len(route) - 1))
            for a, b in own:
                if a in nodes and b in nodes:
                    ax.plot([nodes[a]["x"], nodes[b]["x"]], [nodes[a]["y"], nodes[b]["y"]],
                            color=col, lw=2.2, alpha=0.95, zorder=3)
        for ci, j in enumerate(dis):
            pts = [(nodes[n]["x"], nodes[n]["y"]) for n in game.routes[j] if n in nodes]
            ax.plot(*zip(*pts), color=CORRIDOR_COLOURS[ci % 3], lw=2.6, zorder=4)
        s, t = od
        ax.scatter([nodes[s]["x"]], [nodes[s]["y"]], s=100, c="black", marker="^", zorder=5)
        ax.scatter([nodes[t]["x"]], [nodes[t]["y"]], s=110, c="black", marker="*", zorder=5)
        med_own = 100 * float(np.median(own_shares)) if own_shares else 0.0
        ann = (f"{s} -> {t}   R={game.n_routes} = 3 corridors + "
               f"{game.n_routes - 3} padded (median own-edge {med_own:.0f}%)")
        if r:
            ann += (f"\nrule/opt {r['rule_over_opt']:.2f}   stat/opt "
                    f"{r['static_over_opt']:.2f}   ext-rot {r['ext_over_opt']:.2f}")
        ax.set_title(ann, fontsize=9.5)
        xs = [p[0] for route in game.routes for p in
              [(nodes[n]["x"], nodes[n]["y"]) for n in route if n in nodes]]
        ys = [p[1] for route in game.routes for p in
              [(nodes[n]["x"], nodes[n]["y"]) for n in route if n in nodes]]
        mx = (max(xs) - min(xs)) * 0.3 + 1e-5
        my = (max(ys) - min(ys)) * 0.3 + 1e-5
        ax.set_xlim(min(xs) - mx, max(xs) + mx)
        ax.set_ylim(min(ys) - my, max(ys) + my)
    fig.suptitle(f"gen41 pool: {city} | corridors bold red/green/orange; each padded route "
                 f"its own light colour, with its NON-corridor detour edges drawn thick; "
                 f"^ origin, * destination", fontsize=12)
    fig.tight_layout()
    path = f"{PNG_DIR}/{city}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main():
    d = json.load(open("models/runs/gen41_pool_screen.json"))
    for city, blob in d["cities"].items():
        selected = [tuple(od) for od in blob["selected"]]
        for old, new in SWAPS.get(city, {}).items():
            if old in selected:
                selected[selected.index(old)] = new
        blob["selected"] = [list(od) for od in selected]
        blob["selection_note"] = ("gdansk 70-297 swapped for 303-15 on Kilian's review "
                                  "2026-08-05" if city == "gdansk" else
                                  blob.get("selection_note", ""))
        rows_by_od = {tuple(r["od"]): r for r in blob["candidates"]}
        render_city(city, selected, rows_by_od)
    with open("models/runs/gen41_pool_screen.json", "w") as f:
        json.dump(d, f, indent=1)
    print("updated models/runs/gen41_pool_screen.json (swap recorded)", flush=True)


if __name__ == "__main__":
    main()
