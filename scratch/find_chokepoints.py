"""Find + rank chokepoints in the Kaliningrad OSM graph via graph theory (no eyeballing).

Two complementary notions:
  * edge betweenness centrality (weighted, all-pairs AND depot->demand subset) = how many routes
    funnel through an edge;
  * bridges (cut edges) = edges with NO alternative path (removing them disconnects the graph);
  * detour ratio = (shortest path between an edge's endpoints with it removed) / (the edge length)
    -> how costly is routing AROUND it. inf = a true bridge.

Interpretation for the hybrid rung:
  * BRIDGE / inf-detour edges -> *assignment* matters (no route-around; pick the truck on the right
    side). Blocking them is devastating but not a routing decision.
  * high-betweenness, high-but-finite-detour edges -> *routing* matters (take the costly alternative).
    These are the ones where next-hop route-around + anticipation can pay.

Saves scratch/chokepoints.png and prints ranked tables.
    PYTHONPATH=. python scratch/find_chokepoints.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from src.envs.assignment_factory import make_assignment_env


def ek(u, v):
    return (u, v) if repr(u) <= repr(v) else (v, u)


def main():
    env = make_assignment_env()
    G = env.graph
    depots = list(env.assignment_depots)
    demand = list(env.assignment_demand)
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    # 1. betweenness: all-pairs (global), normalized
    bc_all = {ek(*e): v for e, v in nx.edge_betweenness_centrality(G, weight="distance", normalized=True).items()}
    # depot->demand usage: directly count how many of the 16 (depot,demand) shortest paths use each
    # edge (raw integer 0..16 — interpretable, unlike the diluted normalized subset).
    bc_sub = {ek(*e): 0 for e in G.edges()}
    for dep in depots:
        for tgt in demand:
            try:
                p = nx.shortest_path(G, dep, tgt, weight="distance")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for i in range(len(p) - 1):
                bc_sub[ek(p[i], p[i + 1])] += 1

    # 2. bridges (cut edges)
    bridges = {ek(*e) for e in nx.bridges(G)}

    # 3. detour ratio per edge (cost of routing around it)
    detour = {}
    for u, v, data in list(G.edges(data=True)):
        w = data["distance"]
        attrs = dict(data)
        G.remove_edge(u, v)
        try:
            d = nx.shortest_path_length(G, u, v, weight="distance")
            detour[ek(u, v)] = d / w
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            detour[ek(u, v)] = float("inf")
        G.add_edge(u, v, **attrs)

    edges = list(bc_sub.keys())
    cap = lambda d: min(d, 10.0)  # cap detour so an inf/huge value doesn't dominate the score

    # ---- rankings ----
    # On OUR current geometry (depots 110/135 -> contested-8): chokepoints our trucks actually cross.
    on_route = [e for e in edges if bc_sub[e] > 0]
    route_assign = sorted([e for e in on_route if e in bridges], key=lambda e: bc_sub[e], reverse=True)
    route_routing = sorted([e for e in on_route if e not in bridges and detour[e] >= 1.25],
                           key=lambda e: bc_sub[e] * cap(detour[e]), reverse=True)
    # Globally strongest ROUTING chokepoints (for (re)designing geometry): busy AND costly to bypass.
    routing_ck = sorted([e for e in edges if e not in bridges and 1.5 <= detour[e] < float("inf")],
                        key=lambda e: bc_all[e] * cap(detour[e]), reverse=True)
    # Critical bridges = cut edges that carry real all-pairs traffic (true crossings, not dead-end spurs).
    crit_bridges = sorted(bridges, key=lambda e: bc_all[e], reverse=True)

    def show(title, lst, n=10):
        print(f"\n=== {title} (top {n}) ===")
        print(f"{'edge':>14} | {'route_use':>9} | {'all_betw':>8} | {'detour x':>8} | bridge")
        for e in lst[:n]:
            dr = detour[e]
            drs = "inf" if dr == float("inf") else f"{dr:.2f}"
            print(f"{str(e):>14} | {bc_sub[e]:9d} | {bc_all[e]:8.3f} | {drs:>8} | {'YES' if e in bridges else ''}")

    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(bridges)} bridges; depots={depots}, {len(demand)} demand; max bridge all_betw={max(bc_all[e] for e in bridges):.3f}")
    show("CURRENT geometry — routing chokepoints our trucks cross (route_use = #of 16 depot->demand paths)", route_routing)
    show("CURRENT geometry — bridges our trucks cross (route_use>0)", route_assign)
    show("GLOBAL strongest routing chokepoints (busy x costly detour) — for redesigning geometry", routing_ck)
    show("Most CRITICAL bridges (cut edges carrying real traffic)", crit_bridges)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(16, 11))
    maxbc = max(bc_all.values()) or 1.0
    for u, v in G.edges():
        e = ek(u, v)
        b = bc_all[e] / maxbc
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=plt.cm.YlOrRd(0.2 + 0.8 * b), lw=0.5 + 4 * b, zorder=1 + b)
    # bridges in blue dashed
    for e in bridges:
        u, v = e
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="#1f77b4", lw=2.0, ls=(0, (3, 2)), alpha=0.7, zorder=3)
    # highlight + label top chokepoints
    for rank, e in enumerate(routing_ck[:6], 1):
        u, v = e
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#9467bd", lw=5, alpha=0.6, zorder=4)
        ax.annotate(f"R{rank}", (mx, my), fontsize=12, weight="bold", color="#5b2d8f", zorder=7)
    # mark node 0 (the dominant chokepoint hub)
    if "0" in pos:
        ax.scatter(*pos["0"], s=260, marker="*", color="red", edgecolor="k", lw=1, zorder=8)
        ax.annotate("node 0 (hub)", pos["0"], textcoords="offset points", xytext=(8, -14),
                    fontsize=11, weight="bold", color="red", zorder=8)
    # depots + demand
    for di, dep in enumerate(depots):
        ax.scatter(*pos[dep], s=420, marker="s", color=["#1b7a3d", "#7a1b6b"][di], edgecolor="k", lw=1.5, zorder=6)
        ax.annotate(f"Depot {'AB'[di]}", pos[dep], textcoords="offset points", xytext=(8, 6), fontsize=12, weight="bold")
    for nname in demand:
        ax.scatter(*pos[nname], s=90, color="#ff7f0e", edgecolor="k", lw=0.8, zorder=5)

    legend = [
        Line2D([0], [0], color=plt.cm.YlOrRd(0.95), lw=4, label="high edge betweenness (busy road)"),
        Line2D([0], [0], color="#1f77b4", lw=2, ls="--", label="bridge / cut edge (no route-around)"),
        Line2D([0], [0], color="#9467bd", lw=5, alpha=0.6, label="top ROUTING chokepoint (R#)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=11)
    ax.set_title("Kaliningrad chokepoints — edge betweenness (heat) · bridges (blue dashed) · "
                 "top routing chokepoints (R#, purple) · assignment chokepoints (A#)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.savefig("scratch/chokepoints.png", dpi=130, bbox_inches="tight")
    print("\nsaved scratch/chokepoints.png")


if __name__ == "__main__":
    main()
