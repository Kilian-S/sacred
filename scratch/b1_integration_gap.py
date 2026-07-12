#!/usr/bin/env python3
"""B1 (EVAL/ORACLE-ONLY): the holistic-SBO integration gap on the held-out city.

Joint SBO over (placement x fleet x hardening menu) vs the classical tier-by-tier decomposition
at matched evaluation budget, priced by frozen zero-shot generalists (gen16 seeds 0 and 1) on
Gdansk. Pre-registration: experiments/b1_integration_gap.md (binding).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import networkx as nx
import numpy as np
import torch
from scipy.stats import spearmanr

from scripts.train_generalist import CITY_PATHS, exact_ratio
from src.agents.sac import ProtagonistSAC
from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, objective_matrix, solve_multiconvoy)
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.sbo.surrogate import train_surrogate
from src.utils.graph_utils import load_osm_graph_and_demands

K, BAND, KX, NS = 1, (0.15, 0.95), 8, (2, 3, 4)
ETA, HBUDGET = 0.5, 4
N0, BUDGET, REPEATS, KAPPA, ENSEMBLE = 15, 60, 12, 1.0, 5
ACTORS = {0: "models/runs/gen16_multicity/seed0_ckpts/actor_ep1000.pt",
          1: "models/runs/gen16_multicity/seed1_ckpts/actor_ep500.pt"}


def _mkprot(path):
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu", role_alpha=True)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2))
    prot.actor.route_feats = None
    prot.actor.load_state_dict(torch.load(path, map_location="cpu"))
    return prot


def harden(vuln: dict, routes_edges, option: int) -> dict:
    """The 4-option doctrine menu; budget 4 units, each multiplies an edge's p by (1 - ETA)."""
    v = dict(vuln)
    cand = sorted(v, key=lambda e: tuple(sorted(map(str, e))))
    if option == 0:
        return v
    if option == 1:                      # top-4 most vulnerable edges
        picks = sorted(cand, key=lambda e: -v[e])[:HBUDGET]
    elif option == 2:                    # 4 most route-shared edges (chokepoints)
        share = {e: sum(e in es for es in routes_edges) for e in cand}
        picks = sorted(cand, key=lambda e: (-share[e], -v[e]))[:HBUDGET]
    else:                                # worst edge of each of the 4 highest-pmax routes
        pmax = [(max(v[e] for e in es), es) for es in routes_edges]
        picks = []
        for _, es in sorted(pmax, key=lambda x: -x[0]):
            worst = max(es, key=lambda e: v[e])
            if worst not in picks:
                picks.append(worst)
            if len(picks) == HBUDGET:
                break
    for e in picks:
        v[e] = v[e] * (1.0 - ETA)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json-out", default="models/runs/b1_integration_gap.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    nodes_path, edges_path = CITY_PATHS["gdansk"]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    import random
    deg3 = [n for n, d in G.degree() if d >= 3]
    rng0 = random.Random(5)
    pairs, seen = [], set()
    while len(pairs) < 45 and len(seen) < 4000:
        s, t = rng0.sample(deg3, 2)
        key = tuple(sorted((s, t)))
        if key in seen:
            continue
        seen.add(key)
        try:
            if 3 <= len(build_route_set(G, s, t, 0, "w")) <= 6:
                pairs.append((s, t))
        except Exception:
            pass

    prots = {s: _mkprot(p) for s, p in ACTORS.items()}
    t0 = time.time()
    designs = []          # dicts: features X, per-actor policy expl, cost, ids
    for (s, t) in pairs:
        try:
            routes = build_route_set(G, s, t, KX, "w")
            if not 10 <= len(routes) <= 14:
                continue
            redges = [edges_of_route(r) for r in routes]
            cand = sorted(set().union(*redges), key=lambda e: tuple(sorted(map(str, e))))
            vuln0 = length_band_vulnerability(G, cand, band=BAND, weight="w",
                                              norm_edges=list(G.edges()))
            costs = np.array([sum(G[u][v]["w"] for u, v in zip(r, r[1:])) for r in routes])
            jac = [len(x & y) / len(x | y) for x, y in itertools.combinations(redges, 2)] or [0.0]
            for h in range(4):
                vh = harden(vuln0, redges, h)
                game = build_interdiction_game(G, s, t, K, k_extra=KX, weight="w",
                                               intercept_fn=survival_intercept_fn(vh))
                for N in NS:
                    occs, M = objective_matrix(game, N, "mission")
                    occ_index = {tuple(int(x) for x in o): i for i, o in enumerate(occs)}
                    sol_v, sol_x = None, None
                    vs = np.array([vh[e] for e in cand])
                    pmax = np.array([max(vh[e] for e in es) for es in redges])
                    harm = float(1.0 / np.sum(1.0 / np.clip(pmax, 1e-6, None)))
                    fx = [nx.shortest_path_length(G, s, t, weight="w"), len(routes), len(cand),
                          costs.min(), costs.mean(), costs.std(), float(vs.mean()),
                          float(vs.max()), float(np.mean(jac)), float(N), harm,
                          float(pmax.min()), float(pmax.mean()),
                          1.0 * (h == 1), 1.0 * (h == 2), 1.0 * (h == 3)]
                    row = {"s": s, "t": t, "N": N, "h": h, "X": fx}
                    # frozen-policy exploitability per actor (exact stacked dist under this
                    # hardened game's BR) + fleet cost
                    env = make_multiconvoy_env(od=(s, t), N=N, K=K, k_extra_routes=KX,
                                               menu_select=True, edge_vuln_band=BAND,
                                               interception_loss=10.0, seed=0,
                                               nodes_path=nodes_path, edges_path=edges_path)
                    it = type("I", (), {})()
                    it.env = env; it.eq = 1.0; it.pol_hist = []; it.od = (s, t)
                    # override the OBSERVED map + features to the hardened reality
                    emap = {tuple(sorted(tuple(e), key=repr)): p for e, p in vh.items()}
                    env.edge_vulnerability.update(emap)
                    wv = pmax
                    cost_n = (costs - costs.min()) / max(costs.max() - costs.min(), 1e-9)
                    wv_n = (wv - wv.min()) / max(wv.max() - wv.min(), 1e-9)
                    env._menu_feats_cache = torch.tensor(
                        np.stack([cost_n, wv_n], axis=1), dtype=torch.float32)
                    for sd, prot in prots.items():
                        _, d = exact_ratio(prot, it)
                        # score the stacked dist under the HARDENED game's exact matrix
                        R = len(routes)
                        dd = np.zeros(len(occs))
                        for r in range(R):
                            dd[occ_index[tuple(N if i == r else 0 for i in range(R))]] = \
                                d[env._occ_index[tuple(N if i == r else 0 for i in range(R))]]
                        _, expl = best_response_attacker_multi(M, dd)
                        row[f"expl_{sd}"] = float(expl)
                        if sd == 0:
                            lead = np.array([d[env._occ_index[tuple(N if i == r else 0
                                                                    for i in range(R))]]
                                             for r in range(R)])
                            row["fleet_cost"] = float(N * (lead @ costs))
                    designs.append(row)
        except Exception:
            continue
    print(f"[B1] {len(designs)} joint designs enumerated in {time.time()-t0:.0f}s", flush=True)

    X = np.array([d["X"] for d in designs], np.float32)
    out = {"n_designs": len(designs)}
    for sd in prots:
        Y = np.array([d[f"expl_{sd}"] for d in designs], np.float32)
        OPT = float(Y.min())

        def fit(idx, seed):
            r = np.random.default_rng(seed)
            mu, sdv = X[idx].mean(0), X[idx].std(0) + 1e-9
            ms = []
            for m in range(ENSEMBLE):
                b = r.choice(idx, len(idx), replace=True)
                torch.manual_seed(seed * 100 + m)
                mm, _ = train_surrogate((X[b] - mu) / sdv, Y[b], epochs=150, lr=5e-3,
                                        batch_size=8, hidden_dim=32)
                mm.eval(); ms.append(mm)
            with torch.no_grad():
                P = np.stack([mm(torch.tensor((X - mu) / sdv)).squeeze(-1).numpy() for mm in ms])
            return P.mean(0), P.std(0)

        h0_idx = [i for i, d in enumerate(designs) if d["h"] == 0]

        def sbo(seed, pool):
            r = np.random.default_rng(seed)
            ev = list(r.choice(pool, N0, replace=False))
            while len(ev) < BUDGET - (0 if pool is not h0_idx else 4):
                mu, sdv = fit(np.array(ev), seed)
                lcb = mu - KAPPA * sdv
                mask = np.full(len(Y), np.inf)
                mask[pool] = 0
                lcb = lcb + mask * 0  # restrict to pool
                lcb[[i for i in range(len(Y)) if i not in pool]] = np.inf
                lcb[np.array(ev)] = np.inf
                ev.append(int(np.argmin(lcb)))
            return ev

        seq_vals, joint_vals = [], []
        for rep in range(REPEATS):
            # ARM A: tier-by-tier. Tier 1 on h0 pool (budget 56), then all 4 h options (4)
            ev = sbo(rep, h0_idx)
            best_h0 = min(ev, key=lambda i: Y[i])
            d0 = designs[best_h0]
            fam = [i for i, d in enumerate(designs)
                   if d["s"] == d0["s"] and d["t"] == d0["t"] and d["N"] == d0["N"]]
            seq_vals.append(float(min(Y[i] for i in fam)))
            # ARM B: joint over everything (budget 60)
            ev = sbo(rep, list(range(len(Y))))
            joint_vals.append(float(min(Y[i] for i in ev)))
        gap = np.median(np.array(seq_vals) - np.array(joint_vals))
        rel = gap / np.median(joint_vals)
        # mechanism row: does hardening reorder placements?
        fam_ids = sorted({(d["s"], d["t"], d["N"]) for d in designs})
        y_h0, y_best = [], []
        for f in fam_ids:
            ys = {d["h"]: Y[i] for i, d in enumerate(designs)
                  if (d["s"], d["t"], d["N"]) == f}
            if 0 in ys and len(ys) == 4:
                y_h0.append(ys[0]); y_best.append(min(ys.values()))
        rho, _ = spearmanr(y_h0, y_best)
        out[f"actor{sd}"] = {
            "opt": OPT, "seq_median": float(np.median(seq_vals)),
            "joint_median": float(np.median(joint_vals)),
            "gap_median": float(gap), "gap_rel": float(rel),
            "seq_regret": float(np.median(seq_vals) - OPT),
            "joint_regret": float(np.median(joint_vals) - OPT),
            "placement_rank_corr_h0_vs_besth": float(rho)}
        print(f"[B1] actor{sd}: seq median {np.median(seq_vals):.4f} vs joint "
              f"{np.median(joint_vals):.4f} | gap {gap:+.4f} ({100*rel:+.1f}%) | true opt {OPT:.4f} "
              f"| placement rank corr h0-vs-best-h {rho:.3f}", flush=True)

    json.dump(out, open(a.json_out, "w"), indent=2)
    print(f"[written] {a.json_out}")


if __name__ == "__main__":
    main()
