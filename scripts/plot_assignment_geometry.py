#!/usr/bin/env python3
"""Plot the 3b assignment-probe geometry on the OSM map for visual review.

Shows the full Kaliningrad graph (faint), the two depots (green squares), and the demand
nodes (orange), with each depot's shortest route to every demand node drawn so you can see
the corridors and how 'contested' the demand is. Saves a PNG.

    PYTHONPATH=. python scripts/plot_assignment_geometry.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.envs.assignment_factory import make_assignment_env

OUT = "scratch/assignment_geometry.png"
DEPOT_COLORS = ["#1b7a3d", "#7a1b6b"]  # A, B
PATH_COLORS = ["#2ca02c", "#9467bd"]


def main() -> None:
    env = make_assignment_env()
    pos = {n: (d["x"], d["y"]) for n, d in env.graph.nodes(data=True)}
    depots = list(env.assignment_depots)
    demand = list(env.assignment_demand)

    fig, ax = plt.subplots(figsize=(15, 10))
    # faint full graph
    for u, v in env.graph.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="0.88", lw=0.6, zorder=1)
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
               s=6, color="0.62", zorder=2)

    # each depot's shortest route to every demand node (shows corridors + reach overlap)
    for di, dep in enumerate(depots):
        for t in demand:
            try:
                path = nx.dijkstra_path(env.graph, dep, t, weight="distance")
            except nx.NetworkXNoPath:
                continue
            ax.plot([pos[p][0] for p in path], [pos[p][1] for p in path],
                    color=PATH_COLORS[di % len(PATH_COLORS)], lw=1.6, alpha=0.45, zorder=3)

    # demand nodes
    for t in demand:
        ax.scatter(*pos[t], s=140, color="#ff7f0e", edgecolor="k", linewidth=1.2, zorder=5)
        ax.annotate(t, pos[t], textcoords="offset points", xytext=(6, 5), fontsize=10, weight="bold")

    # depots
    for di, dep in enumerate(depots):
        ax.scatter(*pos[dep], s=420, marker="s", color=DEPOT_COLORS[di % len(DEPOT_COLORS)],
                   edgecolor="k", linewidth=1.5, zorder=6)
        ax.annotate(f"Depot {'AB'[di]} ({dep})", pos[dep], textcoords="offset points",
                    xytext=(10, 8), fontsize=13, weight="bold", color=DEPOT_COLORS[di % len(DEPOT_COLORS)])

    ax.set_title(f"3b assignment geometry — depots {depots} (green=A route, purple=B route), "
                 f"{len(demand)} demand nodes (orange)", fontsize=13)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved {OUT}")

    # contested-ness table
    print("\ndemand node | dist from A | dist from B | |A-B|")
    for t in demand:
        da = nx.dijkstra_path_length(env.graph, depots[0], t, weight="distance")
        db = nx.dijkstra_path_length(env.graph, depots[1], t, weight="distance")
        print(f"   {t:>5}     |   {da:6.1f}    |   {db:6.1f}    | {abs(da-db):5.1f}")


if __name__ == "__main__":
    main()
