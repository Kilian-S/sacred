"""Search the OSM graph for a hybrid-rung geometry: demand placement (depots fixed at 110/135)
that maximises BOTH levers —
  * ROUTING: the depot->demand route crosses a strong, *blockable* chokepoint (high betweenness +
    costly-but-finite detour) -> next-hop route-around matters;
  * ASSIGNMENT: the node is *contested* (both depots similar distance) and the two depots cross
    DIFFERENT chokepoints -> blocking one shifts the advantage to the other depot.

Score(d) = max(gatewayStrength_A, gatewayStrength_B) * contested_factor * diff_gateway_bonus.
Picks the top spread-out demand set, prints the chosen nodes + their gateways, saves a PNG showing
the routes (green=DepotA, purple=DepotB) and the gateway edges.

    PYTHONPATH=. python scratch/design_hybrid_geometry.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.envs.assignment_factory import make_assignment_env

DEPOTS = ("110", "135")
N_DEMAND = 8
MIN_SEP = 0.0006     # min lon/lat separation between chosen demand nodes (avoid one cluster)
CONTEST_SCALE = 6.0  # distance units; |dA-dB| < this ~ contested


def ek(u, v):
    return (u, v) if repr(u) <= repr(v) else (v, u)


def main():
    env = make_assignment_env()
    G = env.graph
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    bc = {ek(*e): v for e, v in nx.edge_betweenness_centrality(G, weight="distance", normalized=True).items()}
    detour = {}
    for u, v, data in list(G.edges(data=True)):
        w = data["distance"]; attrs = dict(data); G.remove_edge(u, v)
        try:
            detour[ek(u, v)] = nx.shortest_path_length(G, u, v, weight="distance") / w
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            detour[ek(u, v)] = float("inf")
        G.add_edge(u, v, **attrs)

    def strength(e):
        d = detour[e]
        return 0.0 if d == float("inf") else bc[e] * min(d, 8.0)  # bridge => no route-around => not routing chokepoint

    dist = {dp: nx.shortest_path_length(G, dp, weight="distance") for dp in DEPOTS}

    def gateway(dp, d):
        """The strongest blockable chokepoint on the dp->d shortest route."""
        try:
            p = nx.shortest_path(G, dp, d, weight="distance")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, 0.0, p if False else []
        be, bs = None, 0.0
        for i in range(len(p) - 1):
            e = ek(p[i], p[i + 1]); s = strength(e)
            if s > bs:
                bs, be = s, e
        return be, bs, p

    rows = []
    for d in G.nodes():
        if d in DEPOTS or any(dist[dp].get(d) is None for dp in DEPOTS):
            continue
        (eA, sA, pA), (eB, sB, pB) = gateway("110", d), gateway("135", d)
        dA, dB = dist["110"][d], dist["135"][d]
        contested = 1.0 / (1.0 + abs(dA - dB) / CONTEST_SCALE)
        diff_bonus = 1.4 if (eA and eB and eA != eB) else 1.0
        score = max(sA, sB) * contested * diff_bonus
        rows.append(dict(d=d, score=score, sA=sA, sB=sB, eA=eA, eB=eB, dA=dA, dB=dB,
                         pA=pA, pB=pB, contested=contested, diff=eA != eB))
    rows.sort(key=lambda r: r["score"], reverse=True)

    chosen, used = [], []
    for r in rows:
        x, y = pos[r["d"]]
        if all((x - ux) ** 2 + (y - uy) ** 2 > MIN_SEP ** 2 for ux, uy in used):
            chosen.append(r); used.append((x, y))
        if len(chosen) >= N_DEMAND:
            break

    print(f"depots {DEPOTS}; scored {len(rows)} candidate demand nodes. Top {len(chosen)} (spread):")
    print(f"{'node':>5} | {'score':>6} | {'distA':>6} {'distB':>6} | {'gatewayA':>12} {'sA':>5} | {'gatewayB':>12} {'sB':>5} | diff")
    gateways = set()
    for r in chosen:
        gateways.add(r["eA"]); gateways.add(r["eB"])
        print(f"{r['d']:>5} | {r['score']:6.2f} | {r['dA']:6.1f} {r['dB']:6.1f} | "
              f"{str(r['eA']):>12} {r['sA']:5.2f} | {str(r['eB']):>12} {r['sB']:5.2f} | {'Y' if r['diff'] else ''}")
    print(f"\nrecommended demand tuple: {tuple(r['d'] for r in chosen)}")
    print(f"gateway edges the antagonist would target: {sorted(g for g in gateways if g)}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(16, 11))
    maxbc = max(bc.values()) or 1.0
    for u, v in G.edges():
        b = bc[ek(u, v)] / maxbc
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="0.85", lw=0.4 + 2.5 * b, zorder=1)
    # routes
    for r in chosen:
        for p, col in ((r["pA"], "#1b7a3d"), (r["pB"], "#7a1b6b")):
            ax.plot([pos[n][0] for n in p], [pos[n][1] for n in p], color=col, lw=1.3, alpha=0.5, zorder=3)
    # gateway edges (chokepoints these routes depend on)
    for e in gateways:
        if e:
            u, v = e
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="red", lw=5, alpha=0.7, zorder=4)
    # demand + depots
    for r in chosen:
        ax.scatter(*pos[r["d"]], s=160, color="#ff7f0e", edgecolor="k", lw=1.2, zorder=6)
        ax.annotate(r["d"], pos[r["d"]], textcoords="offset points", xytext=(5, 4), fontsize=9, weight="bold")
    for di, dp in enumerate(DEPOTS):
        ax.scatter(*pos[dp], s=440, marker="s", color=["#1b7a3d", "#7a1b6b"][di], edgecolor="k", lw=1.5, zorder=7)
        ax.annotate(f"Depot {'AB'[di]} ({dp})", pos[dp], textcoords="offset points", xytext=(8, 6), fontsize=13, weight="bold")
    if "0" in pos:
        ax.scatter(*pos["0"], s=240, marker="*", color="blue", edgecolor="k", lw=1, zorder=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#1b7a3d", lw=2, label="Depot A routes"),
        Line2D([0], [0], color="#7a1b6b", lw=2, label="Depot B routes"),
        Line2D([0], [0], color="red", lw=5, label="gateway chokepoint (antagonist target)"),
    ], loc="upper right", fontsize=11)
    ax.set_title("Hybrid-rung geometry search — demand placed to force routes through blockable "
                 "chokepoints (red); routes by depot (green/purple)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.savefig("scratch/hybrid_geometry.png", dpi=130, bbox_inches="tight")
    print("\nsaved scratch/hybrid_geometry.png")


if __name__ == "__main__":
    main()
