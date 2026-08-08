#!/usr/bin/env python3
"""Render the Act-2 consolidation instance (Koenigsberg 71 -> 33) as a PNG map.

ORACLE/EVAL-ONLY, no training. The instance is the one the proposed unified static and
dynamic K-ladder runs on: 71-33 on the Kaliningrad road graph, N=3 convoys, k_extra=8,
edge-vulnerability band (0.15, 0.95), menu-select. Its structure is K-independent, so one
map serves every cell of the ladder.

Three panels, matching the act's argument:
  A. the six edge-disjoint corridors, the structure the two-line max-flow rule plays;
  B. the five padded routes, whose own detour edges are drawn thick, the shared-edge menu
     where the equilibrium's remaining mass lives;
  C. the vulnerability field over the 43 interdictable menu edges, the interdictor's
     target set.

Run: OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen43_od_map.py
Writes assets/gen43_od_71_33.png.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from scratch.critique_followup_probes import disjoint_subset
from scripts.train_generalist import CITY_PATHS
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.utils.graph_utils import load_osm_graph_and_demands

OD = ("71", "33")
N, KX, BAND = 3, 8, (0.15, 0.95)
OUT = "assets/gen43_od_71_33.png"

# Okabe-Ito qualitative palette (colour-vision-safe), yellow dropped for contrast on white.
CORRIDOR_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
GRAPH_GREY = "#dcdcdc"


def route_points(route, nodes):
    return [(nodes[n]["x"], nodes[n]["y"]) for n in route if n in nodes]


def draw_backdrop(ax, segs):
    ax.add_collection(LineCollection(segs, colors=GRAPH_GREY, linewidths=0.5, zorder=1))


def mark_endpoints(ax, nodes, s, t, label=True):
    ax.scatter([nodes[s]["x"]], [nodes[s]["y"]], s=150, c="black", marker="^",
               zorder=8, edgecolors="white", linewidths=1.0)
    ax.scatter([nodes[t]["x"]], [nodes[t]["y"]], s=190, c="black", marker="*",
               zorder=8, edgecolors="white", linewidths=1.0)
    if label:
        ax.annotate(f"origin {s}", (nodes[s]["x"], nodes[s]["y"]),
                    textcoords="offset points", xytext=(9, -12), fontsize=9,
                    fontweight="bold", zorder=9)
        ax.annotate(f"depot {t}", (nodes[t]["x"], nodes[t]["y"]),
                    textcoords="offset points", xytext=(9, 8), fontsize=9,
                    fontweight="bold", zorder=9)


def main():
    nodes_path, edges_path = CITY_PATHS["kaliningrad"]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, _d in edges:
        G.add_edge(str(u), str(v))
    segs = [[(nodes[u]["x"], nodes[u]["y"]), (nodes[v]["x"], nodes[v]["y"])]
            for u, v in G.edges() if u in nodes and v in nodes]

    env = make_multiconvoy_env(OD, N=N, K=1, k_extra_routes=KX, edge_vuln_band=BAND,
                               absolute_vuln_norm=True, menu_select=True,
                               objective="mission")
    game = env.game
    route_edge_sets = [set(e) for e in game.route_edges]
    corridors = disjoint_subset(route_edge_sets)
    padded = [j for j in range(game.n_routes) if j not in corridors]
    core_union = set().union(*[route_edge_sets[j] for j in corridors])
    menu_union = set().union(*route_edge_sets)
    vuln = {frozenset(k): v for k, v in env.edge_vulnerability.items()}
    s, t = OD

    # Zoom window: the menu's own bounding box plus a margin.
    pts = [p for j in range(game.n_routes) for p in route_points(game.routes[j], nodes)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = (max(xs) - min(xs)) * 0.16 + 1e-6, (max(ys) - min(ys)) * 0.16 + 1e-6
    xlim, ylim = (min(xs) - mx, max(xs) + mx), (min(ys) - my, max(ys) + my)

    # Coordinates are raw longitude and latitude, so a degree of longitude is shorter than
    # a degree of latitude by cos(lat). Without this the map is stretched sideways.
    lat_aspect = 1.0 / np.cos(np.radians(float(np.mean(ys))))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.9))

    # ---- Panel A: the edge-disjoint corridors -------------------------------------
    ax = axes[0]
    draw_backdrop(ax, segs)
    for ci, j in enumerate(corridors):
        p = route_points(game.routes[j], nodes)
        ax.plot(*zip(*p), color=CORRIDOR_COLOURS[ci % len(CORRIDOR_COLOURS)], lw=3.0,
                zorder=4, solid_capstyle="round",
                label=f"corridor {ci + 1} (cost {game.travel_cost[j]:.1f})")
    mark_endpoints(ax, nodes, s, t)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax.set_title(f"A. The {len(corridors)} edge-disjoint corridors\n"
                 f"the structure a two-line max-flow rule spreads over", fontsize=11)

    # City-scale inset showing where the instance sits.
    axin = ax.inset_axes([0.71, 0.015, 0.285, 0.285])
    axin.add_collection(LineCollection(segs, colors="#c8c8c8", linewidths=0.25, zorder=1))
    axin.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                             fill=False, edgecolor="#c62828", linewidth=1.4, zorder=3))
    # Include the zoom rectangle in the inset's limits: its margin can reach past the
    # graph's own bounding box, and a half-drawn rectangle reads as a rendering fault.
    allx = [nodes[n]["x"] for n in nodes] + list(xlim)
    ally = [nodes[n]["y"] for n in nodes] + list(ylim)
    padx = (max(allx) - min(allx)) * 0.03
    pady = (max(ally) - min(ally)) * 0.03
    axin.set_xlim(min(allx) - padx, max(allx) + padx)
    axin.set_ylim(min(ally) - pady, max(ally) + pady)
    axin.set_aspect(lat_aspect, adjustable="box")
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_facecolor("white")
    for spine in axin.spines.values():
        spine.set_edgecolor("#999999")
    axin.set_title("the whole graph", fontsize=7.5, pad=2)

    # ---- Panel B: the padded shared-edge routes ------------------------------------
    ax = axes[1]
    draw_backdrop(ax, segs)
    for j in corridors:
        p = route_points(game.routes[j], nodes)
        ax.plot(*zip(*p), color="#b0b0b0", lw=1.8, zorder=3)
    own_shares = []
    pad_cmap = plt.get_cmap("tab10")
    for pi, j in enumerate(padded):
        route = game.routes[j]
        p = route_points(route, nodes)
        col = pad_cmap(pi % 10)
        ax.plot(*zip(*p), color=col, lw=1.5, zorder=4)
        own = [(a, b) for a, b in zip(route[:-1], route[1:])
               if frozenset((a, b)) not in core_union]
        own_shares.append(len(own) / max(1, len(route) - 1))
        for a, b in own:
            if a in nodes and b in nodes:
                ax.plot([nodes[a]["x"], nodes[b]["x"]], [nodes[a]["y"], nodes[b]["y"]],
                        color=col, lw=3.4, alpha=0.95, zorder=5)
    mark_endpoints(ax, nodes, s, t, label=False)
    med_own = 100 * float(np.median(own_shares)) if own_shares else 0.0
    ax.legend(handles=[Line2D([], [], color="#444444", lw=3.4,
                              label="edges a padded route does not share with any corridor"),
                       Line2D([], [], color="#444444", lw=1.5,
                              label="the rest of each padded route"),
                       Line2D([], [], color="#b0b0b0", lw=1.8,
                              label="the corridors of panel A, for context")],
              loc="lower left", fontsize=8.5, framealpha=0.92)
    ax.set_title(f"B. The {len(padded)} padded routes\n"
                 f"heavy overlap with the corridors, median own-edge share "
                 f"{med_own:.0f} per cent", fontsize=11)

    # ---- Panel C: the vulnerability field ------------------------------------------
    ax = axes[2]
    draw_backdrop(ax, segs)
    menu_segs, menu_vals = [], []
    for e in sorted(menu_union, key=lambda fs: sorted(fs)):
        a, b = sorted(e)
        if a in nodes and b in nodes:
            menu_segs.append([(nodes[a]["x"], nodes[a]["y"]),
                              (nodes[b]["x"], nodes[b]["y"])])
            menu_vals.append(float(vuln.get(e, BAND[0])))
    # Normalise over the menu's own range, not the whole band: most edges sit near the
    # band's floor, so band-normalisation would render almost every edge the same pale hue.
    norm = Normalize(vmin=min(menu_vals), vmax=max(menu_vals))
    base = plt.get_cmap("YlOrRd")
    cmap = base.resampled(256).from_list(
        "menu_threat", base(np.linspace(0.18, 1.0, 256)))
    ax.add_collection(LineCollection(menu_segs, colors=[cmap(norm(v)) for v in menu_vals],
                                     linewidths=3.4, zorder=4, capstyle="round"))
    mark_endpoints(ax, nodes, s, t, label=False)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.040,
                      pad=0.02)
    cb.set_label("interception probability if an ambush sits on the edge", fontsize=9)
    ax.set_title(f"C. The {len(menu_segs)} interdictable menu edges\n"
                 f"the interdictor chooses K of these, vulnerability "
                 f"{min(menu_vals):.2f} to {max(menu_vals):.2f}", fontsize=11)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(lat_aspect, adjustable="box")
        ax.set_axis_off()

    fig.suptitle(
        f"The consolidated Act-2 instance: Königsberg {s} to {t}, "
        f"{game.n_routes} routes ({len(corridors)} edge-disjoint corridors and "
        f"{len(padded)} padded), {len(menu_segs)} interdictable edges, "
        f"{N} convoys.  Triangle marks the origin, star the depot.", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}", flush=True)
    print(f"  corridors {corridors} | padded {padded} | menu edges {len(menu_segs)} | "
          f"median own-edge share {med_own:.1f}%", flush=True)


if __name__ == "__main__":
    main()
