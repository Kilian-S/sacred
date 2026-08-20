#!/usr/bin/env python3
"""Oracle-only probe of the disjoint-route naive baseline.

The route menu's leading entries are an edge-disjoint max-flow decomposition, whereas the
"uniform" ladder anchors mix over the padded k-shortest menu, whose near-duplicates share edges
and so stack probability mass under a single interdicted edge. This probe measures the naive row
a planner would actually use, uniform over the edge-disjoint routes, plus its
inverse-vulnerability refinement, on the single-convoy headline instance, both multi-convoy
headline instances and the held-out-city transfer pools. Exact oracle arithmetic, no training.
"""
from __future__ import annotations

import numpy as np

from src.baselines.interdiction_oracle import (
    best_response_attacker, build_interdiction_game, solve)
from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, objective_matrix, solve_multiconvoy)
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

N, K, BAND, KX = 3, 1, (0.15, 0.95), 8


def disjoint_subset(route_edges) -> list[int]:
    """Greedy maximal edge-disjoint subset of routes, taken in menu order.

    The menu's prefix is the max-flow decomposition, so this recovers the disjoint base routes.
    """
    kept: list[int] = []
    used: set = set()
    for i, re in enumerate(route_edges):
        if not (re & used):
            kept.append(i)
            used |= re
    return kept


def stack_dist_over(occs, routes_idx, weights=None) -> np.ndarray:
    """Occupancy distribution: all N convoys stacked on one route drawn from routes_idx."""
    if weights is None:
        weights = np.ones(len(routes_idx))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    dist = np.zeros(len(occs))
    for w, r in zip(weights, routes_idx):
        for j, o in enumerate(occs):
            if int(o[r]) == N and int(o.sum()) == N:
                dist[j] = w
                break
    assert abs(dist.sum() - 1.0) < 1e-9
    return dist


def mc_rows(env, label: str) -> dict:
    game = env.game
    occs, M = objective_matrix(game, N)
    sol = solve_multiconvoy(game, N)
    dis = disjoint_subset(game.route_edges)
    # vulnerability of a stacked fleet on route r: q_r = 1 - (1 - p*_r)^N, p*_r = worst edge vuln
    vuln = {}
    for r in dis:
        # per-route worst-case interception under the best single-edge interdiction of that route
        col = [game.payoff[r, j] for j in range(game.payoff.shape[1])]
        p_star = max(col)
        vuln[r] = 1.0 - (1.0 - p_star) ** N
    uni = stack_dist_over(occs, dis)
    inv = stack_dist_over(occs, dis, [1.0 / max(vuln[r], 1e-9) for r in dis])
    _, v_uni = best_response_attacker_multi(M, uni)
    _, v_inv = best_response_attacker_multi(M, inv)
    return {"label": label, "R": game.n_routes, "n_disjoint": len(dis),
            "loss_det": sol.loss_det, "eq": sol.loss_mixed,
            "uni_disjoint_stack": float(v_uni), "inv_vuln_disjoint_stack": float(v_inv)}


def main() -> None:
    print("=== Multi-convoy headline instances (mission exploitability, lower better) ===")
    for od in [("35", "159"), ("62", "97")]:
        env = make_multiconvoy_env(od, N=N, K=K, k_extra_routes=KX, edge_vuln_band=BAND,
                                   absolute_vuln_norm=True, menu_select=True, objective="mission")
        r = mc_rows(env, f"{od[0]}->{od[1]}")
        print(f"{r['label']} (R={r['R']}, disjoint={r['n_disjoint']}): "
              f"det={r['loss_det']:.3f} eq={r['eq']:.3f} | "
              f"UNIFORM-DISJOINT-STACK={r['uni_disjoint_stack']:.3f} "
              f"(ratio {r['uni_disjoint_stack']/r['eq']:.2f}x) | "
              f"INV-VULN-DISJOINT-STACK={r['inv_vuln_disjoint_stack']:.3f} "
              f"(ratio {r['inv_vuln_disjoint_stack']/r['eq']:.2f}x)")
    print("  [SACRED refs: 35-159 = 0.256 (n=10 CI [0.246,0.266]); 62-97 pre-fix exact = 0.295]")

    print("\n=== Single-convoy headline 33-71 k8 HARD K=1 ===")
    import networkx as nx
    from src.utils.graph_utils import load_osm_graph_and_demands
    from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS
    from scripts.train_generalist import CITY_PATHS
    nodes_path, edges_path = CITY_PATHS["kaliningrad"]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    game = build_interdiction_game(G, "33", "71", 1, k_extra=8)
    sol = solve(game)
    dis = disjoint_subset(game.route_edges)
    strat = np.zeros(game.n_routes)
    for r in dis:
        strat[r] = 1.0 / len(dis)
    _, expl = best_response_attacker(game, strat)
    print(f"R={game.n_routes}, disjoint={len(dis)}: eq={sol.value:.3f} | "
          f"UNIFORM-DISJOINT={expl:.3f}")
    print("  [refs: uniform-over-menu 0.455; SACRED TAP 0.362 (B2-P3) / 0.276 (gen10-SC) / "
          "0.310 (gen14 n=10)]")

    print("\n=== Held-out-city ZST pools (zero training, zero labels, zero graph exposure) ===")
    from scripts.train_generalist import sample_instances
    for city, ref in [("gdansk", "generalist 1.733 (sel-on-train) / distill 1.555 / retrieval 1.676"),
                      ("istanbul", "generalist 1.880")]:
        insts = sample_instances(6, N, K, BAND, KX, 0, city=city)
        ratios_u, ratios_i, beats = [], [], 0
        for it in insts:
            r = mc_rows(it.env, f"{city}:{it.od}")
            ratios_u.append(r["uni_disjoint_stack"] / r["eq"])
            ratios_i.append(r["inv_vuln_disjoint_stack"] / r["eq"])
            if r["uni_disjoint_stack"] < r["loss_det"]:
                beats += 1
        print(f"{city}: UNIFORM-DISJOINT-STACK mean ratio {np.mean(ratios_u):.3f} "
              f"(per-OD {['%.2f' % x for x in ratios_u]}), beats loss_det {beats}/6 | "
              f"INV-VULN mean {np.mean(ratios_i):.3f}")
        print(f"  [ref: {ref}; random-init ~1.99]")


def sweep_cells() -> None:
    """Print the K/N sweep cells on held-out 35-159, heuristic row beside SACRED.

    The crossover where trained calibration first beats the max-flow heuristic is K = m-1, with m
    the number of disjoint routes: the interdiction budget must approach the min-cut before
    shared-edge calibration carries value beyond naive disjointness.
    """
    for (n, k, sacred) in [(3, 1, 0.261), (3, 2, 0.500), (3, 3, 0.661), (2, 1, 0.232), (5, 1, 0.389)]:
        env = make_multiconvoy_env(("35", "159"), N=n, K=k, k_extra_routes=KX, edge_vuln_band=BAND,
                                   absolute_vuln_norm=True, menu_select=True, objective="mission")
        game = env.game
        occs, M = objective_matrix(game, n)
        sol = solve_multiconvoy(game, n)
        dis = disjoint_subset(game.route_edges)
        dist = np.zeros(len(occs))
        for r in dis:
            for j, o in enumerate(occs):
                if int(o[r]) == n and int(o.sum()) == n:
                    dist[j] = 1.0 / len(dis)
                    break
        _, v = best_response_attacker_multi(M, dist)
        print(f"N={n} K={k}: eq={sol.loss_mixed:.3f} det={sol.loss_det:.3f} | "
              f"uniform-disjoint-stack={v:.3f} | SACRED={sacred}")


if __name__ == "__main__":
    main()
    print("\n=== gen12 sweep cells (held-out 35-159), heuristic vs SACRED ===")
    sweep_cells()
