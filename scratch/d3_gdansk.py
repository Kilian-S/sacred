"""D3-on-Gdansk (expansion item 5; EVAL-ONLY): the composite exhibit on a NEVER-TRAINED city.

The D1 SBO acquisition loop over (placement x fleet) designs ON THE HELD-OUT CITY (Gdansk), where
each design's objective is the frozen MULTI-CITY generalist's operational exploitability (one forward
pass + one oracle BR per design). Strategic base/fleet design in a theatre the policy never trained
on, priced by zero-shot transfer, in a loop no LP can enter (it re-solves per design AND cannot
score a policy). The poster centrepiece + the empirical backbone of the ZST-vs-LP argument.

Run: PYTHONPATH=. .venv/bin/python scratch/d3_gdansk.py <generalist_actor.pt>
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time

import networkx as nx
import numpy as np
import torch
from scipy.stats import spearmanr

from scripts.train_generalist import CITY_PATHS, Instance, exact_ratio
from src.agents.sac import ProtagonistSAC
from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.sbo.surrogate import train_surrogate
from src.utils.graph_utils import load_osm_graph_and_demands
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env

K, BAND, KX, NS = 1, (0.15, 0.95), 8, (2, 3, 4)
N0, BUDGET, REPEATS, KAPPA, ENSEMBLE = 12, 45, 12, 1.0, 5
CITY = "gdansk"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("actor")
    ap.add_argument("--json-out", default="models/runs/d3_gdansk.json"); args = ap.parse_args()
    torch.set_num_threads(4)
    nodes_path, edges_path = CITY_PATHS[CITY]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu", role_alpha=True)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2)); prot.actor.route_feats = None
    prot.actor.load_state_dict(torch.load(args.actor, map_location="cpu"))

    # design space: high-connectivity Gdansk ODs x fleet size N
    deg3 = [n for n, d in G.degree() if d >= 3]; rng0 = random.Random(5)
    pairs, seen = [], set()
    while len(pairs) < 70 and len(seen) < 4000:
        s, t = rng0.sample(deg3, 2); key = tuple(sorted((s, t), key=repr))
        if key in seen:
            continue
        seen.add(key)
        try:
            if 3 <= len(build_route_set(G, s, t, 0, "w")) <= 6:
                pairs.append((s, t))
        except Exception:
            pass

    def feats(s, t, N):
        routes = build_route_set(G, s, t, KX, "w")
        cand = sorted(set().union(*(edges_of_route(r) for r in routes)), key=repr)
        vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=list(G.edges()))
        game = build_interdiction_game(G, s, t, K, k_extra=KX, weight="w",
                                       intercept_fn=survival_intercept_fn(vuln))
        costs = np.array([sum(G[u][v]["w"] for u, v in zip(r, r[1:])) for r in game.routes])
        vs = np.array([vuln[e] for e in cand]); esets = [edges_of_route(r) for r in game.routes]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(esets, 2)] or [0.0]
        pmax = np.array([max(vuln[e] for e in es) for es in esets])
        harm = float(1.0 / np.sum(1.0 / np.clip(pmax, 1e-6, None)))
        return [nx.shortest_path_length(G, s, t, weight="w"), len(game.routes), len(cand),
                costs.min(), costs.mean(), costs.std(), float(vs.mean()), float(vs.max()),
                float(np.mean(jac)), float(N), harm, float(pmax.min()), float(pmax.mean())], game

    t0 = time.time(); X, Ypol, Yoracle = [], [], []
    for s, t in pairs:
        for N in NS:
            try:
                env = make_multiconvoy_env(od=(s, t), N=N, K=K, k_extra_routes=KX, menu_select=True,
                                           edge_vuln_band=BAND, interception_loss=10.0, seed=0,
                                           nodes_path=nodes_path, edges_path=edges_path)
                if not 10 <= env.game.n_routes <= 14:
                    continue
                sol = solve_multiconvoy(env.game, N, "mission")
                if sol.loss_mixed < 0.05:
                    continue
                it = type("I", (), {})(); it.env = env; it.eq = float(sol.loss_mixed); it.pol_hist = []
                it.od = (s, t)
                ratio, _ = exact_ratio(prot, it)
                fx, _ = feats(s, t, N)
                X.append(fx); Ypol.append(ratio * sol.loss_mixed); Yoracle.append(float(sol.loss_mixed))
            except Exception:
                continue
    X = np.array(X, np.float32); Ypol = np.array(Ypol, np.float32); Yoracle = np.array(Yoracle, np.float32)
    OPT = float(Ypol.min()); rho_t, _ = spearmanr(Ypol, Yoracle)
    print(f"[D3-Gdansk] {len(Ypol)} designs on the NEVER-TRAINED city in {time.time()-t0:.0f}s; "
          f"policy-expl optimum {OPT:.3f}; policy-vs-oracle target corr {rho_t:.3f}")

    def fit(idx, seed):
        rng = np.random.default_rng(seed); mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-9; ms = []
        for m in range(ENSEMBLE):
            b = rng.choice(idx, len(idx), replace=True); torch.manual_seed(seed * 100 + m)
            mm, _ = train_surrogate((X[b] - mu) / sd, Ypol[b], epochs=150, lr=5e-3, batch_size=8, hidden_dim=32)
            mm.eval(); ms.append(mm)
        with torch.no_grad():
            P = np.stack([mm(torch.tensor((X - mu) / sd)).squeeze(-1).numpy() for mm in ms])
        return P.mean(0), P.std(0)

    def sbo(seed):
        rng = np.random.default_rng(seed); ev = list(rng.choice(len(Ypol), N0, replace=False))
        best = [float(Ypol[ev].min())]
        while len(ev) < BUDGET:
            mu, sd = fit(np.array(ev), seed); lcb = mu - KAPPA * sd; lcb[np.array(ev)] = np.inf
            ev.append(int(lcb.argmin())); best.append(float(Ypol[ev].min()))
        return best

    def rand(seed):
        rng = np.random.default_rng(9 + seed); o = rng.permutation(len(Ypol))[:BUDGET]
        return [float(Ypol[o[:b]].min()) for b in range(N0, BUDGET + 1)]

    def etb(c, bar=0.01, start=N0):
        for i, v in enumerate(c):
            if v - OPT <= bar:
                return start + i
        return float("inf")

    # held-out surrogate quality
    idx = np.arange(len(Ypol)); r = np.random.default_rng(0); r.shuffle(idx); cut = int(0.75 * len(idx))
    mu, sd = X[idx[:cut]].mean(0), X[idx[:cut]].std(0) + 1e-9; torch.manual_seed(0)
    m0, _ = train_surrogate((X[idx[:cut]] - mu) / sd, Ypol[idx[:cut]], epochs=300, lr=5e-3, batch_size=16)
    m0.eval()
    with torch.no_grad():
        pr = m0(torch.tensor((X[idx[cut:]] - mu) / sd)).squeeze(-1).numpy()
    rho_s, _ = spearmanr(pr, Ypol[idx[cut:]])

    sc = [sbo(i) for i in range(REPEATS)]; rc = [rand(i) for i in range(REPEATS)]
    med = lambda v: float(np.median(v))
    se = [etb(c) for c in sc]; re_ = [etb(c) for c in rc]
    print(f"\n=== D3-on-Gdansk RESULT (never-trained city) ===")
    print(f"  surrogate over TRAINED generalist exploitability (held-out): Spearman {rho_s:.3f}")
    print(f"  SBO median evals-to-regret<=0.01: {med(se)} vs random {med(re_)} (PASS: {med(se) <= 0.5*med(re_)})")
    print(f"  SBO median final regret {med([c[-1]-OPT for c in sc]):.4f} vs random {med([c[-1]-OPT for c in rc]):.4f}")
    print(f"  policy-vs-oracle design-target corr {rho_t:.3f}")
    print(f"  THE COMPOSITE ON AN UNSEEN THEATRE: strategic design priced by a zero-shot policy, "
          f"a loop no LP can enter.")
    json.dump({"city": CITY, "n": int(len(Ypol)), "opt": OPT, "surrogate_spearman": float(rho_s),
               "sbo_median_etb": med(se), "random_median_etb": med(re_),
               "policy_vs_oracle_corr": float(rho_t),
               "sbo_final_regret": med([c[-1]-OPT for c in sc])}, open(args.json_out, "w"), indent=2)
    print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
