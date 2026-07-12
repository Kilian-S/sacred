#!/usr/bin/env python3
"""A6 (EVAL-ONLY): the retrieval baseline for the ZST act.

Nearest-training-instance equilibrium lookup: for each held-out Gdansk OD, find the most similar
of the 18 gen16 training instances in scale-free feature space and play ITS equilibrium stacked
mixture mapped onto the test menu by cost rank (secondary: worst-vulnerability rank). Scored
exactly under the test OD's oracle BR, ratio to its equilibrium.
Pre-registration: experiments/a6_a7_a8_completions.md §A6.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import torch

from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import best_response_attacker_multi, solve_multiconvoy

N, K, KX, BAND = 3, 1, 8, (0.15, 0.95)
TRAIN_CITIES = ["kaliningrad", "east_london", "istanbul"]


def inst_feats(it) -> np.ndarray:
    g = it.env.game
    G = it.env.graph
    costs = np.array([sum(G[u][v]["w"] for u, v in zip(r, r[1:])) for r in g.routes])
    emap = it.env.edge_vulnerability

    def ev(e):
        u, v = tuple(e)
        return emap.get((u, v), emap.get((v, u), emap.get(tuple(sorted((u, v), key=repr)), 0.0)))
    cand = sorted(set().union(*g.route_edges), key=lambda e: tuple(sorted(map(str, e))))
    vs = np.array([ev(e) for e in cand])
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(g.route_edges, 2)] or [0.0]
    pmax = np.array([max(ev(e) for e in es) for es in g.route_edges])
    harm = float(1.0 / np.sum(1.0 / np.clip(pmax, 1e-6, None)))
    return np.array([g.n_routes, len(cand), costs.min() / costs.mean(),
                     costs.std() / costs.mean(), float(vs.mean()), float(vs.max()),
                     float(np.mean(jac)), harm, float(pmax.min()), float(pmax.mean())])


def stacked_eq_route_mixture(it) -> np.ndarray:
    """The instance's equilibrium mass per ROUTE (stacked support; = the full equilibrium on this
    pool, verified in gen24's ceilings)."""
    sol = solve_multiconvoy(it.env.game, N, "mission")
    d = np.asarray(sol.defender_strategy, float)
    R = it.env.game.n_routes
    mix = np.zeros(R)
    for oi, occ in enumerate(sol.occupancies):
        occ = tuple(int(x) for x in occ)
        if max(occ) == N:                      # stacked occupancy -> its route
            mix[occ.index(N)] += d[oi]
    s = mix.sum()
    return mix / s if s > 0 else np.full(R, 1.0 / R)


def map_by_rank(src_mix: np.ndarray, src_key: np.ndarray, dst_key: np.ndarray) -> np.ndarray:
    """Assign the k-th (by src_key) source route's mass to the k-th (by dst_key) dest route;
    surplus source ranks renormalised away; surplus dest routes get zero."""
    src_order = np.argsort(src_key)
    dst_order = np.argsort(dst_key)
    dst = np.zeros(len(dst_key))
    for k in range(min(len(src_order), len(dst_order))):
        dst[dst_order[k]] = src_mix[src_order[k]]
    s = dst.sum()
    return dst / s if s > 0 else np.full(len(dst_key), 1.0 / len(dst_key))


def score_stacked(it, route_mix: np.ndarray) -> float:
    env = it.env
    R = env.game.n_routes
    d = np.zeros(len(env.occupancies))
    for r in range(R):
        d[env._occ_index[tuple(N if i == r else 0 for i in range(R))]] = route_mix[r]
    _, expl = best_response_attacker_multi(env.obj_matrix, d)
    return float(expl) / it.eq


def main():
    torch.set_num_threads(4)
    train = []
    for c in TRAIN_CITIES:
        train += sample_instances(6, N, K, BAND, KX, 0, city=c)
    test = sample_instances(6, N, K, BAND, KX, 0, city="gdansk")

    F = np.stack([inst_feats(it) for it in train])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fz = (F - mu) / sd
    mixes = [stacked_eq_route_mixture(it) for it in train]

    def keys(it):
        g = it.env.game
        G = it.env.graph
        costs = np.array([sum(G[u][v]["w"] for u, v in zip(r, r[1:])) for r in g.routes])
        emap = it.env.edge_vulnerability

        def ev(e):
            u, v = tuple(e)
            return emap.get((u, v), emap.get((v, u), emap.get(tuple(sorted((u, v), key=repr)), 0.0)))
        pmax = np.array([max(ev(e) for e in es) for es in g.route_edges])
        return costs, pmax

    out = {"per_od": []}
    for variant in ("cost_rank", "vuln_rank"):
        rs = []
        for it in test:
            fz = (inst_feats(it) - mu) / sd
            nn = int(np.argmin(((Fz - fz) ** 2).sum(1)))
            src_c, src_p = keys(train[nn])
            dst_c, dst_p = keys(it)
            if variant == "cost_rank":
                mix = map_by_rank(mixes[nn], src_c, dst_c)
            else:
                mix = map_by_rank(mixes[nn], src_p, dst_p)
            r = score_stacked(it, mix)
            rs.append(r)
            if variant == "cost_rank":
                out["per_od"].append({"od": f"{it.od[0]}-{it.od[1]}",
                                      "nn": f"{train[nn].city}:{train[nn].od[0]}-{train[nn].od[1]}",
                                      "ratio": round(r, 3)})
        out[variant] = {"mean": float(np.mean(rs)), "per_od": [round(x, 2) for x in rs]}
        print(f"A6 retrieval ({variant}): mean ratio {np.mean(rs):.3f} | {[round(x,2) for x in rs]}")
    print("anchors: gen16 adversarial 1.733 (select-on-train) / 1.677; random-init ~1.99; "
          "uniform-stack per OD would be the naive row")
    json.dump(out, open("models/runs/a6_retrieval.json", "w"), indent=2)
    print("[written] models/runs/a6_retrieval.json")


if __name__ == "__main__":
    main()
