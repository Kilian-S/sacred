"""Joint search over DEPOT PAIRS + demand for the hybrid rung. Scores each depot pair by the
hybrid quality of the best demand set it can support (routing chokepoints x contested x the two
depots crossing DIFFERENT gateways), and compares against the current depots 110/135.

    PYTHONPATH=. python scratch/search_hybrid_depots.py
"""

from __future__ import annotations

from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from src.envs.assignment_factory import make_assignment_env

N_DEMAND = 8
MIN_SEP = 0.0006
CONTEST_SCALE = 6.0
DEPOT_MIN_DIST = 20.0   # depots must be at least this far apart (distance units)
DEPOT_MAX_DIST = 55.0


def ek(u, v):
    return (u, v) if repr(u) <= repr(v) else (v, u)


def main():
    env = make_assignment_env()
    G = env.graph
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    bc = {ek(*e): v for e, v in nx.edge_betweenness_centrality(G, weight="distance", normalized=True).items()}
    detour = {}
    for u, v, data in list(G.edges(data=True)):
        w = data["distance"]; a = dict(data); G.remove_edge(u, v)
        try:
            detour[ek(u, v)] = nx.shortest_path_length(G, u, v, weight="distance") / w
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            detour[ek(u, v)] = float("inf")
        G.add_edge(u, v, **a)

    def strength(e):
        d = detour[e]
        return 0.0 if d == float("inf") else bc[e] * min(d, 8.0)

    # precompute single-source dist + gateway (best chokepoint on the path) for junction candidates
    cand = [n for n in G if G.degree(n) >= 3]
    SS = {}
    for dp in cand:
        dist, paths = nx.single_source_dijkstra(G, dp, weight="distance")
        gw = {}
        for node, p in paths.items():
            be, bs = None, 0.0
            for i in range(len(p) - 1):
                e = ek(p[i], p[i + 1]); s = strength(e)
                if s > bs:
                    bs, be = s, e
            gw[node] = (be, bs, p)
        SS[dp] = (dist, gw)
    print(f"{len(cand)} junction candidates (deg>=3); scoring depot pairs...")

    demand_all = list(G.nodes())

    def best_demand(A, B, want_set=False):
        distA, gwA = SS[A]; distB, gwB = SS[B]
        scored = []
        for d in demand_all:
            if d in (A, B) or d not in distA or d not in distB:
                continue
            (eA, sA, pA), (eB, sB, pB) = gwA[d], gwB[d]
            contested = 1.0 / (1.0 + abs(distA[d] - distB[d]) / CONTEST_SCALE)
            diff = 1.4 if (eA and eB and eA != eB) else 1.0
            sc = max(sA, sB) * contested * diff
            scored.append((sc, d, eA, eB, sA, sB, distA[d], distB[d], pA, pB))
        scored.sort(key=lambda r: r[0], reverse=True)
        if not want_set:
            return sum(r[0] for r in scored[:N_DEMAND]) / N_DEMAND if scored else 0.0
        chosen, used = [], []
        for r in scored:
            x, y = pos[r[1]]
            if all((x - ux) ** 2 + (y - uy) ** 2 > MIN_SEP ** 2 for ux, uy in used):
                chosen.append(r); used.append((x, y))
            if len(chosen) >= N_DEMAND:
                break
        return chosen

    pairs = []
    for A, B in combinations(cand, 2):
        dab = SS[A][0].get(B, 1e9)
        if dab < DEPOT_MIN_DIST or dab > DEPOT_MAX_DIST:
            continue
        pairs.append((best_demand(A, B), A, B))
    pairs.sort(key=lambda r: r[0], reverse=True)

    cur = best_demand("110", "135") if "110" in SS and "135" in SS else float("nan")
    print(f"\ncurrent depots (110,135) score = {cur:.2f}\n")
    print(f"{'rank':>4} | {'depots':>12} | {'score':>6} | separation")
    for i, (sc, A, B) in enumerate(pairs[:10], 1):
        print(f"{i:>4} | {('('+A+','+B+')'):>12} | {sc:6.2f} | {SS[A][0][B]:.1f}")

    # plot the best pair's geometry
    _, A, B = pairs[0]
    chosen = best_demand(A, B, want_set=True)
    print(f"\nBEST pair ({A},{B}) demand = {tuple(r[1] for r in chosen)}")
    gateways = set()
    for r in chosen:
        gateways.add(r[2]); gateways.add(r[3])

    fig, ax = plt.subplots(figsize=(16, 11))
    mb = max(bc.values()) or 1.0
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="0.85", lw=0.4 + 2.5 * bc[ek(u, v)] / mb, zorder=1)
    for r in chosen:
        for p, col in ((r[8], "#1b7a3d"), (r[9], "#7a1b6b")):
            ax.plot([pos[n][0] for n in p], [pos[n][1] for n in p], color=col, lw=1.3, alpha=0.5, zorder=3)
    for e in gateways:
        if e:
            u, v = e
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="red", lw=5, alpha=0.7, zorder=4)
    for r in chosen:
        ax.scatter(*pos[r[1]], s=160, color="#ff7f0e", edgecolor="k", lw=1.2, zorder=6)
        ax.annotate(r[1], pos[r[1]], textcoords="offset points", xytext=(5, 4), fontsize=9, weight="bold")
    for di, dp in enumerate((A, B)):
        ax.scatter(*pos[dp], s=440, marker="s", color=["#1b7a3d", "#7a1b6b"][di], edgecolor="k", lw=1.5, zorder=7)
        ax.annotate(f"Depot {'AB'[di]} ({dp})", pos[dp], textcoords="offset points", xytext=(8, 6), fontsize=13, weight="bold")
    ax.legend(handles=[
        Line2D([0], [0], color="#1b7a3d", lw=2, label=f"Depot A ({A}) routes"),
        Line2D([0], [0], color="#7a1b6b", lw=2, label=f"Depot B ({B}) routes"),
        Line2D([0], [0], color="red", lw=5, label="gateway chokepoint"),
    ], loc="upper right", fontsize=11)
    ax.set_title(f"Best searched depot pair ({A},{B}) — score {pairs[0][0]:.2f} vs current (110,135) {cur:.2f}")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.savefig("scratch/hybrid_geometry_searched_depots.png", dpi=130, bbox_inches="tight")
    print("saved scratch/hybrid_geometry_searched_depots.png")


if __name__ == "__main__":
    main()
