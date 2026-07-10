"""D1: the SBO acquisition loop (pre-registered in experiments/d1_sbo_loop.md; ORACLE-ONLY).

Surrogate-guided design optimisation over (placement x fleet size): fit a small MLP ensemble on
evaluated designs, propose by LOWER CONFIDENCE BOUND, evaluate with the exact oracle, iterate.
Compared against random search and the one-shot surrogate argmin at matched budgets; regret is
measured against the true optimum of the fully-enumerated space (the enumeration exists only to
score regret; no optimiser sees it).

Run: PYTHONPATH=. .venv/bin/python scratch/sbo_acquisition_loop.py
"""
from __future__ import annotations

import itertools
import json
import random
import time

import networkx as nx
import numpy as np
import torch

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS
from src.sbo.surrogate import train_surrogate
from src.utils.graph_utils import load_osm_graph_and_demands

K, BAND, KX = 1, (0.15, 0.95), 8
NS = (2, 3, 4)
N_PAIRS = 300
N0, BUDGET = 15, 60
REPEATS = 20
KAPPA = 1.0
ENSEMBLE = 5
REGRET_BAR = 0.01

nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
G = nx.Graph()
for u, v, d in edges:
    G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
ALL_EDGES = list(G.edges())

# --- enumerate the design space once (regret reference only) ---
deg3 = [n for n, d in G.degree() if d >= 3]
rng0 = random.Random(0)
pairs, seen = [], set()
while len(pairs) < N_PAIRS and len(seen) < 8000:
    s, t = rng0.sample(deg3, 2)
    key = tuple(sorted((s, t), key=repr))
    if key in seen:
        continue
    seen.add(key)
    try:
        if 3 <= len(build_route_set(G, s, t, 0, "w")) <= 6:
            pairs.append((s, t))
    except Exception:
        continue

def features_and_value(s, t, N):
    routes = build_route_set(G, s, t, KX, "w")
    cand = sorted(set().union(*(edges_of_route(r) for r in routes)), key=repr)
    vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=ALL_EDGES)
    game = build_interdiction_game(G, s, t, K, k_extra=KX, weight="w",
                                   intercept_fn=survival_intercept_fn(vuln))
    costs = np.array([sum(G[u][v]["w"] for u, v in zip(r, r[1:])) for r in game.routes])
    vs = np.array([vuln[e] for e in cand])
    esets = [edges_of_route(r) for r in game.routes]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(esets, 2)] or [0.0]
    sp = nx.shortest_path_length(G, s, t, weight="w")
    route_pmax = np.array([max(vuln[e] for e in es) for es in esets])
    harm = float(1.0 / np.sum(1.0 / np.clip(route_pmax, 1e-6, None)))
    x = [sp, len(game.routes), len(cand), costs.min(), costs.mean(), costs.std(),
         float(vs.mean()), float(vs.max()), float(np.mean(jac)), float(N),
         harm, float(route_pmax.min()), float(route_pmax.mean())]
    y = float(solve_multiconvoy(game, N, "mission").loss_mixed)
    return x, y

t0 = time.time()
X, Y, meta = [], [], []
for s, t in pairs:
    try:
        for N in NS:
            x, y = features_and_value(s, t, N)
            X.append(x); Y.append(y); meta.append((f"{s}-{t}", N))
    except Exception:
        continue
X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
TRUE_OPT = float(Y.min())
print(f"[D1] space: {len(Y)} designs in {time.time()-t0:.0f}s; true optimum {TRUE_OPT:.3f} "
      f"({meta[int(Y.argmin())]})")


def fit_ensemble(idx, seed):
    models = []
    rng = np.random.default_rng(seed)
    mu, sd = X[idx].mean(axis=0), X[idx].std(axis=0) + 1e-9
    for m in range(ENSEMBLE):
        boot = rng.choice(idx, size=len(idx), replace=True)
        torch.manual_seed(seed * 100 + m)
        model, _ = train_surrogate((X[boot] - mu) / sd, Y[boot], epochs=150, lr=5e-3,
                                   batch_size=8, hidden_dim=32)
        model.eval()
        models.append(model)
    with torch.no_grad():
        preds = np.stack([mm(torch.tensor((X - mu) / sd)).squeeze(-1).numpy() for mm in models])
    return preds.mean(axis=0), preds.std(axis=0)


def run_sbo(seed):
    rng = np.random.default_rng(seed)
    evald = list(rng.choice(len(Y), size=N0, replace=False))
    best = [float(Y[evald].min())]
    while len(evald) < BUDGET:
        mu_p, sd_p = fit_ensemble(np.array(evald), seed)
        lcb = mu_p - KAPPA * sd_p
        lcb[np.array(evald)] = np.inf
        evald.append(int(lcb.argmin()))
        best.append(float(Y[evald].min()))
    return best


def run_random(seed):
    rng = np.random.default_rng(10_000 + seed)
    order = rng.permutation(len(Y))[:BUDGET]
    vals = Y[order]
    return [float(vals[:b].min()) for b in range(N0, BUDGET + 1)]


def run_oneshot(seed):
    rng = np.random.default_rng(seed)
    evald = list(rng.choice(len(Y), size=N0, replace=False))
    mu_p, _ = fit_ensemble(np.array(evald), seed)
    mu_p[np.array(evald)] = np.inf
    pick = int(mu_p.argmin())
    return float(min(Y[evald].min(), Y[pick]))


def evals_to_bar(curve, start=N0):
    for i, v in enumerate(curve):
        if v - TRUE_OPT <= REGRET_BAR:
            return start + i
    return float("inf")


sbo_curves, rnd_curves, oneshot_vals = [], [], []
for r in range(REPEATS):
    sbo_curves.append(run_sbo(r))
    rnd_curves.append(run_random(r))
    oneshot_vals.append(run_oneshot(r))
    print(f"  repeat {r}: SBO final regret {sbo_curves[-1][-1]-TRUE_OPT:.4f} "
          f"(evals-to-{REGRET_BAR}: {evals_to_bar(sbo_curves[-1])}) | "
          f"random final {rnd_curves[-1][-1]-TRUE_OPT:.4f} "
          f"({evals_to_bar(rnd_curves[-1])}) | one-shot {oneshot_vals[-1]-TRUE_OPT:.4f}", flush=True)

sbo_e = [evals_to_bar(c) for c in sbo_curves]
rnd_e = [evals_to_bar(c) for c in rnd_curves]
med = lambda v: float(np.median(v))
print(f"\n=== D1 RESULT ===")
print(f"median evals-to-regret<= {REGRET_BAR}: SBO {med(sbo_e)} vs random {med(rnd_e)} "
      f"(PASS bar: SBO <= half of random: {med(sbo_e) <= 0.5 * med(rnd_e)})")
print(f"median final regret @ B={BUDGET}: SBO {med([c[-1]-TRUE_OPT for c in sbo_curves]):.4f} "
      f"(STRONG bar 0: {med([c[-1]-TRUE_OPT for c in sbo_curves]) == 0.0}) | "
      f"random {med([c[-1]-TRUE_OPT for c in rnd_curves]):.4f} | "
      f"one-shot {med([v-TRUE_OPT for v in oneshot_vals]):.4f}")
json.dump({"true_opt": TRUE_OPT, "n_designs": int(len(Y)), "n0": N0, "budget": BUDGET,
           "kappa": KAPPA, "repeats": REPEATS,
           "sbo_curves": sbo_curves, "random_curves": rnd_curves, "oneshot": oneshot_vals,
           "median_evals_to_bar": {"sbo": med(sbo_e), "random": med(rnd_e)}},
          open("models/runs/d1_sbo_loop.json", "w"), indent=2)
print("[written] models/runs/d1_sbo_loop.json")
