#!/usr/bin/env python3
"""A5 (EVAL-ONLY): reliability check for the d3_gdansk 0.109 policy-vs-oracle correlation.

The design-target evaluator is EXACT (one forward pass + one oracle BR), so within-seed
test-retest is 1 by construction; the meaningful reliability axis is CROSS-SEED: do independently
trained gen16 actors (seeds 0/1/2, each at its selected best checkpoint) induce the SAME design
ranking on the never-trained city? Report (a) pairwise cross-seed Spearman of the design-target
vectors (the reliability), (b) per-seed policy-vs-oracle Spearman (is 0.109 seed-stable?).
Gate (pre-committed in NEXT_STEPS_MASTER A5): the poster claim ("on an unseen theatre, design
against the deployed policy, not the equilibrium") is EARNED iff cross-seed reliability is high
(mean pairwise rho >= 0.5) while policy-vs-oracle stays low (~0.1-0.3) for every seed; otherwise
the exhibit moves to in-distribution only.
"""
from __future__ import annotations

import json
import random

import networkx as nx
import numpy as np
import torch
from scipy.stats import spearmanr

from scripts.train_generalist import CITY_PATHS, exact_ratio
from src.agents.sac import ProtagonistSAC
from src.baselines.interdiction_oracle import build_route_set
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.utils.graph_utils import load_osm_graph_and_demands

K, BAND, KX, NS = 1, (0.15, 0.95), 8, (2, 3, 4)
ACTORS = {
    0: "models/runs/gen16_multicity/seed0_ckpts/actor_ep1000.pt",
    1: "models/runs/gen16_multicity/seed1_ckpts/actor_ep500.pt",
    2: "models/runs/gen16_multicity/seed2_ckpts/actor_ep500.pt",
}


def _mkprot(path):
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu", role_alpha=True)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2))
    prot.actor.route_feats = None
    prot.actor.load_state_dict(torch.load(path, map_location="cpu"))
    return prot


def main():
    torch.set_num_threads(4)
    nodes_path, edges_path = CITY_PATHS["gdansk"]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # SAME design-space construction as scratch/d3_gdansk.py (rng seed 5, 70 pairs, N in {2,3,4})
    deg3 = [n for n, d in G.degree() if d >= 3]
    rng0 = random.Random(5)
    pairs, seen = [], set()
    while len(pairs) < 70 and len(seen) < 4000:
        s, t = rng0.sample(deg3, 2)
        key = tuple(sorted((s, t), key=repr))
        if key in seen:
            continue
        seen.add(key)
        try:
            if 3 <= len(build_route_set(G, s, t, 0, "w")) <= 6:
                pairs.append((s, t))
        except Exception:
            pass

    prots = {s: _mkprot(p) for s, p in ACTORS.items()}
    Y = {s: [] for s in prots}
    Yoracle = []
    kept = 0
    for s_, t_ in pairs:
        for N in NS:
            try:
                env = make_multiconvoy_env(od=(s_, t_), N=N, K=K, k_extra_routes=KX,
                                           menu_select=True, edge_vuln_band=BAND,
                                           interception_loss=10.0, seed=0,
                                           nodes_path=nodes_path, edges_path=edges_path)
                if not 10 <= env.game.n_routes <= 14:
                    continue
                sol = solve_multiconvoy(env.game, N, "mission")
                if sol.loss_mixed < 0.05:
                    continue
                it = type("I", (), {})()
                it.env = env; it.eq = float(sol.loss_mixed); it.pol_hist = []; it.od = (s_, t_)
                for s in prots:
                    ratio, _ = exact_ratio(prots[s], it)
                    Y[s].append(ratio * sol.loss_mixed)
                Yoracle.append(float(sol.loss_mixed))
                kept += 1
            except Exception:
                continue
    print(f"[A5] {kept} designs evaluated under 3 independently-trained actors")

    out = {"n_designs": kept, "pairwise": {}, "policy_vs_oracle": {}}
    for a in prots:
        for b in prots:
            if a < b:
                rho, _ = spearmanr(Y[a], Y[b])
                out["pairwise"][f"{a}-{b}"] = float(rho)
                print(f"  cross-seed design-ranking Spearman seed{a} vs seed{b}: {rho:.3f}")
    for s in prots:
        rho, _ = spearmanr(Y[s], Yoracle)
        out["policy_vs_oracle"][str(s)] = float(rho)
        print(f"  policy-vs-oracle target Spearman (seed {s}): {rho:.3f}")
    mean_rel = float(np.mean(list(out["pairwise"].values())))
    out["mean_cross_seed_reliability"] = mean_rel
    print(f"  MEAN cross-seed reliability {mean_rel:.3f} "
          f"(gate: >= 0.5 with policy-vs-oracle low for every seed)")
    json.dump(out, open("models/runs/d3_gdansk_reliability.json", "w"), indent=2)
    print("[written] models/runs/d3_gdansk_reliability.json")


if __name__ == "__main__":
    main()
