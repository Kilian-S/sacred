#!/usr/bin/env python3
"""Visualise the Stage-1.5 dynamic assignment environment (to SEE it before training).

Drives one episode with a tick-level greedy dispatcher + the heuristic congest-near-trucks
adversary, and renders:
  * scratch/dynassign_geometry.png -- static layout (graph, 2 depots, hotspot band)
  * scratch/dynassign_demo.gif     -- animation: 2 trucks serving Poisson demand, requests
                                      appearing and AGEING light->dark red the longer they wait,
                                      red edges = antagonist congestion.

    PYTHONPATH=. python scripts/animate_dynassign.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src.envs.assignment_factory import make_dynamic_assign_env
from src.baselines.greedy_dispatch import _congestion_aware_distance, _id_key

SEED = 3
RATE = 0.06
T = 400
FRAME_EVERY = 3
AGE_MAX = 120.0  # wait (ticks) mapped to the darkest red
TRUCK_COLORS = ["#1f77b4", "#d62728"]
DEPOT_COLORS = ["#1b7a3d", "#7a1b6b"]
# antagonist (matches the gate's congest_near_trucks heuristic + dynassign config)
ANTAG_INTERVAL, CONG_DUR, MAX_LEVEL, BUDGET, CONG_COST = 20, 30, 1.0, 4000.0, 0.1


def run_and_record():
    env = make_dynamic_assign_env(arrival_rate=RATE, demand_seed=SEED, max_time=T)
    env.reset(demand_seed=SEED)
    pos = {n: (d["x"], d["y"]) for n, d in env.graph.nodes(data=True)}
    depots = list(env.assignment_depots)
    hotspots = list(env.dynamic_hotspots)

    def truck_xy(t):
        if t.current_node is not None:
            return pos[t.current_node]
        if t.edge is not None:
            u, v = t.edge
            dist = env.graph.edges[u, v]["distance"]
            f = t.edge_progress / dist if dist > 0 else 0.0
            return (pos[u][0] + f * (pos[v][0] - pos[u][0]), pos[u][1] + f * (pos[v][1] - pos[u][1]))
        return pos[t.home_depot]

    cong_expiry: dict = {}
    spent = 0.0
    frames = []
    for _ in range(T):
        # expire congestion
        for e, exp in list(cong_expiry.items()):
            if exp <= env.time:
                if env.graph.has_edge(*e):
                    env.set_congestion(e, 0.0)
                del cong_expiry[e]
        # antagonist: congest a few edges near trucks at max level (budget-limited)
        if env.time % ANTAG_INTERVAL == 0:
            near = set()
            for t in env.trucks.values():
                node = t.current_node if t.current_node is not None else (t.edge[0] if t.edge else None)
                if node is not None:
                    near |= env._k_hop_edges.get(node, set())
            for e in sorted((e for e in near if e not in cong_expiry), key=repr)[:4]:
                cost = MAX_LEVEL * CONG_DUR * CONG_COST
                if spent + cost <= BUDGET and env.graph.has_edge(*e):
                    env.set_congestion(e, MAX_LEVEL)
                    cong_expiry[e] = env.time + CONG_DUR
                    spent += cost
        # tick-level greedy dispatch (sequential claiming) for idle trucks
        dispatch, claimed = {}, set()
        for tid in sorted(env.trucks):
            t = env.trucks[tid]
            if t.current_node is None:
                continue  # moving
            if t.load <= 0:
                if t.current_node != t.home_depot:
                    dispatch[tid] = t.home_depot
                continue
            avail = [n for n in hotspots if env.graph.nodes[n]["demand"] > 0 and n not in claimed]
            if avail:
                best = min(avail, key=lambda d: (_congestion_aware_distance(env, t.current_node, d), _id_key(d)))
                dispatch[tid] = best
                claimed.add(best)
            elif t.current_node != t.home_depot:
                dispatch[tid] = t.home_depot
        env.step(dispatch_actions=dispatch or None)

        if env.time % FRAME_EVERY == 0:
            demands = {}
            for n in hotspots:
                c = env.graph.nodes[n]["demand"]
                if c > 0:
                    dq = env._pending_arrivals.get(n)
                    demands[n] = (c, (env.time - dq[0]) if dq else 0)
            cong = {e: env.graph.edges[e]["congestion_level"] for e in list(cong_expiry) if env.graph.has_edge(*e)}
            frames.append(dict(
                tick=env.time,
                trucks=[truck_xy(env.trucks[i]) for i in sorted(env.trucks)],
                demands=demands, cong=cong,
                delivered=len(env._delivered_latencies), queue=int(env.remaining_demand)))
    return env, pos, depots, hotspots, frames


def draw_static(pos, depots, hotspots):
    fig, ax = plt.subplots(figsize=(13, 9))
    for u, v in _edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="0.9", lw=0.5, zorder=1)
    for n in hotspots:
        ax.scatter(*pos[n], s=120, color="#ff7f0e", edgecolor="k", lw=1, zorder=5)
        ax.annotate(n, pos[n], textcoords="offset points", xytext=(5, 4), fontsize=9, weight="bold")
    for di, dep in enumerate(depots):
        ax.scatter(*pos[dep], s=400, marker="s", color=DEPOT_COLORS[di], edgecolor="k", lw=1.5, zorder=6)
        ax.annotate(f"Depot {'AB'[di]} ({dep})", pos[dep], textcoords="offset points",
                    xytext=(8, 6), fontsize=12, weight="bold", color=DEPOT_COLORS[di])
    ax.set_title(f"Stage 1.5 dynamic assignment — 2 depots, {len(hotspots)} hotspot nodes "
                 f"(demand arrives Poisson over time)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.savefig("scratch/dynassign_geometry.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved scratch/dynassign_geometry.png")


def animate(pos, depots, hotspots, frames):
    cmap = plt.get_cmap("YlOrRd")
    norm = Normalize(0, AGE_MAX)
    fig, ax = plt.subplots(figsize=(13, 9))
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]

    def render(fr):
        ax.clear()
        for u, v in _edges:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="0.9", lw=0.5, zorder=1)
        # congested edges (red, thicker = higher level)
        for (u, v), lvl in fr["cong"].items():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="red", lw=1 + 3 * lvl, alpha=0.8, zorder=2)
        # depots
        for di, dep in enumerate(depots):
            ax.scatter(*pos[dep], s=360, marker="s", color=DEPOT_COLORS[di], edgecolor="k", lw=1.5, zorder=6)
        # pending demand: colour by age (light->dark red), size by count
        for n, (count, age) in fr["demands"].items():
            ax.scatter(*pos[n], s=90 + 60 * count, color=cmap(norm(age)),
                       edgecolor="k", lw=1, zorder=5)
        # trucks
        for ti, (x, y) in enumerate(fr["trucks"]):
            ax.scatter(x, y, s=240, marker="^", color=TRUCK_COLORS[ti], edgecolor="k", lw=1.5, zorder=8)
        ax.set_xlim(min(xs) - 0.005, max(xs) + 0.005)
        ax.set_ylim(min(ys) - 0.005, max(ys) + 0.005)
        ax.set_title(f"t={fr['tick']:3d} | delivered={fr['delivered']:2d}  queue={fr['queue']:2d}  "
                     f"| ^=truck  circle=request (darker=older)  red=congestion")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="request wait (ticks)", shrink=0.7)
    anim = FuncAnimation(fig, render, frames=frames, interval=120)
    anim.save("scratch/dynassign_demo.gif", writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"saved scratch/dynassign_demo.gif ({len(frames)} frames)")


def main():
    global _edges
    env, pos, depots, hotspots, frames = run_and_record()
    _edges = list(env.graph.edges())
    draw_static(pos, depots, hotspots)
    animate(pos, depots, hotspots, frames)
    print(f"\nEpisode summary: delivered={frames[-1]['delivered']}, final queue={frames[-1]['queue']}, "
          f"frames={len(frames)} (T={T}, lambda={RATE}, seed={SEED})")


if __name__ == "__main__":
    main()
