#!/usr/bin/env python3
"""Block R0 (ORACLE-ONLY, free): repair + aim.

R0a: disjoint-heuristic rows WITH fleet-cost columns for the headline ladders; the
     population prevalence of the heuristic's suboptimality (the honest A8 companion row).
R0c: the K-scan (heuristic/eq vs K) on the headline instances + the gen26 step-3 instance
     shortlist (m = 5-6 Kaliningrad ODs, heuristic saturation at K = m-1/m under the greedy
     yardstick).

Everything is exact LP / greedy-BR arithmetic; no training. Ledger: Block R in
NEXT_STEPS_MASTER.md; results fold into gen13/gen14/gen12/gen16/gen22/a6_a7_a8/b2 + gen26.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "analysis")
from disjoint_baseline_probe import disjoint_subset, stack_dist_over

from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, greedy_br_attacker, objective_matrix, solve_multiconvoy)
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

N, K1, BAND, KX = 3, 1, (0.15, 0.95), 8
OUT = {"r0a_headline_costs": [], "r0a_population": [], "r0c_kscan": [], "r0c_shortlist": []}


def heuristic_dists(game, occs, n=N):
    dis = disjoint_subset(game.route_edges)
    q = {}
    for r in dis:
        p_star = float(game.payoff[r].max())
        q[r] = 1.0 - (1.0 - p_star) ** n
    uni_w = [1.0 for _ in dis]
    inv_w = [1.0 / max(q[r], 1e-9) for r in dis]

    def stack(weights):
        w = np.asarray(weights, float); w = w / w.sum()
        d = np.zeros(len(occs))
        for wt, r in zip(w, dis):
            for j, o in enumerate(occs):
                if int(o[r]) == n and int(o.sum()) == n:
                    d[j] = wt
                    break
        return d
    return dis, stack(uni_w), stack(inv_w)


def mixture_fleet_cost(game, occs, dist, n=N):
    cost = 0.0
    for w, o in zip(dist, occs):
        if w > 0:
            cost += w * float(np.dot(o, game.travel_cost))
    return cost


def part1_headline_costs():
    print("=== R0a: headline rows with fleet-cost columns ===")
    for od in (("35", "159"), ("62", "97")):
        env = make_multiconvoy_env(od, N=N, K=K1, k_extra_routes=KX, edge_vuln_band=BAND,
                                   absolute_vuln_norm=True, menu_select=True, objective="mission")
        game = env.game
        occs, M = objective_matrix(game, N)
        sol = solve_multiconvoy(game, N)
        dis, uni, inv = heuristic_dists(game, occs)
        det_row = int(np.argmin(M.max(axis=1)))
        det_dist = np.zeros(len(occs)); det_dist[det_row] = 1.0
        rows = {}
        for name, d in (("uniform_disjoint_stack", uni), ("inv_vuln_disjoint_stack", inv),
                        ("equilibrium_mixture", np.asarray(sol.defender_strategy)),
                        ("det_plan(ALNS)", det_dist)):
            _, expl = best_response_attacker_multi(M, d)
            rows[name] = {"expl": round(float(expl), 4),
                          "fleet_cost": round(mixture_fleet_cost(game, occs, d), 1)}
        entry = {"od": f"{od[0]}-{od[1]}", "n_disjoint": len(dis), "R": game.n_routes,
                 "eq": round(sol.loss_mixed, 4), "det": round(sol.loss_det, 4), "rows": rows}
        OUT["r0a_headline_costs"].append(entry)
        print(json.dumps(entry, indent=1))


def part2_population():
    print("\n=== R0a: population prevalence of the heuristic's suboptimality (A8 companion) ===")
    ratios, all_rows = [], []
    for city in ("kaliningrad", "east_london", "istanbul", "gdansk"):
        insts = sample_instances(40, N, K1, BAND, KX, seed=3, city=city)
        for it in insts:
            game = it.env.game
            occs, M = objective_matrix(game, N)
            dis, uni, _ = heuristic_dists(game, occs)
            _, expl = best_response_attacker_multi(M, uni)
            r = float(expl) / it.eq
            ratios.append(r)
            all_rows.append({"city": city, "od": f"{it.od[0]}-{it.od[1]}", "m": len(dis),
                             "disjoint_stack_eq": round(r, 3),
                             "det_eq": round(it.loss_det / it.eq, 3)})
        print(f"  {city}: {len(insts)} ODs done", flush=True)
    arr = np.array(ratios)
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    summary = {"quantiles_10_25_50_75_90": [round(x, 3) for x in q],
               "frac_ge_1.5": round(float((arr >= 1.5).mean()), 3),
               "frac_ge_1.2": round(float((arr >= 1.2).mean()), 3)}
    OUT["r0a_population"] = {"rows": all_rows, "summary": summary}
    print(f"  disjoint-stack/eq quantiles (10/25/50/75/90): {summary['quantiles_10_25_50_75_90']}")
    print(f"  fraction >= 1.5: {summary['frac_ge_1.5']}   fraction >= 1.2: {summary['frac_ge_1.2']}")


def part3_kscan_and_shortlist():
    print("\n=== R0c: K-scan on the headlines (exact K<=3; greedy yardstick K>=4) ===")
    for od in (("35", "159"), ("62", "97")):
        env = make_multiconvoy_env(od, N=N, K=K1, k_extra_routes=KX, edge_vuln_band=BAND,
                                   absolute_vuln_norm=True, menu_select=True, objective="mission")
        vuln_fs = {frozenset(k): v for k, v in env.edge_vulnerability.items()}
        scan = {"od": f"{od[0]}-{od[1]}"}
        for k in (1, 2, 3):
            e2 = make_multiconvoy_env(od, N=N, K=k, k_extra_routes=KX, edge_vuln_band=BAND,
                                      absolute_vuln_norm=True, menu_select=True,
                                      objective="mission")
            game = e2.game
            occs, M = objective_matrix(game, N)
            sol = solve_multiconvoy(game, N)
            dis, uni, _ = heuristic_dists(game, occs)
            _, h = best_response_attacker_multi(M, uni)
            scan[f"K{k}"] = {"eq": round(sol.loss_mixed, 3), "det": round(sol.loss_det, 3),
                             "heuristic": round(float(h), 3),
                             "heuristic_eq_ratio": round(float(h) / sol.loss_mixed, 2)}
        # K = 4, 5: greedy yardstick (no exact eq exists = the point)
        game = env.game
        dis = disjoint_subset(game.route_edges)
        uni_support = [(tuple(N if i == r else 0 for i in range(game.n_routes)), 1.0 / len(dis))
                       for r in dis]
        occs1, M1 = objective_matrix(game, N)
        det_row = int(np.argmin(M1.max(axis=1)))
        det_support = [(tuple(int(x) for x in occs1[det_row]), 1.0)]
        for k in (4, 5):
            _, h = greedy_br_attacker(game.route_edges, vuln_fs, uni_support, N, k)
            _, d = greedy_br_attacker(game.route_edges, vuln_fs, det_support, N, k)
            scan[f"K{k}_greedy"] = {"heuristic": round(float(h), 3), "det_plan": round(float(d), 3)}
        OUT["r0c_kscan"].append(scan)
        print(json.dumps(scan, indent=1))

    print("\n=== R0c: gen26 step-3 shortlist (Kaliningrad, m in {5, 6}) ===")
    insts = sample_instances(40, N, K1, BAND, KX, seed=5, city="kaliningrad")
    for it in insts:
        game = it.env.game
        dis = disjoint_subset(game.route_edges)
        m = len(dis)
        if m < 5:
            continue
        vuln_fs = {frozenset(k): v for k, v in it.env.edge_vulnerability.items()}
        occs, M = objective_matrix(game, N)
        _, uni, _ = heuristic_dists(game, occs)
        uni_support = [(tuple(int(x) for x in o), float(w)) for o, w in zip(occs, uni) if w > 0]
        det_row = int(np.argmin(M.max(axis=1)))
        det_support = [(tuple(int(x) for x in occs[det_row]), 1.0)]
        row = {"od": f"{it.od[0]}-{it.od[1]}", "m": m, "R": game.n_routes,
               "eq_K1": round(it.eq, 3), "det_eq_K1": round(it.loss_det / it.eq, 2)}
        for k in (m - 1, m):
            _, h = greedy_br_attacker(game.route_edges, vuln_fs, uni_support, N, k)
            _, d = greedy_br_attacker(game.route_edges, vuln_fs, det_support, N, k)
            row[f"K{k}_greedy"] = {"heuristic": round(float(h), 3), "det": round(float(d), 3),
                                   "saturation": round(float(h) / max(float(d), 1e-9), 2)}
        OUT["r0c_shortlist"].append(row)
        print(json.dumps(row), flush=True)


def main():
    part1_headline_costs()
    part2_population()
    part3_kscan_and_shortlist()
    json.dump(OUT, open("models/runs/r0_screen.json", "w"), indent=2)
    print("\n[written] models/runs/r0_screen.json")


if __name__ == "__main__":
    main()
