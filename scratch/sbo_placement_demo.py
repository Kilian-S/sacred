"""F3: the Obj-4 SBO demonstrator (interdiction-aware base/FOB placement + fleet sizing,
reduced form; EVAL-ONLY, no training of policies).

Design space = (OD pair, fleet size N) on the Kaliningrad graph: candidate base/FOB placements
(high-connectivity OD pairs, shared-edge k8 route menus, soft band 0.15-0.95, K=1, absolute
vulnerability norm) x N in {2, 3, 4}. For each design the ORACLE gives the exact objective:
loss_mixed = the equilibrium mission-failure exploitability of the best randomised fleet routing
(how DEFENSIBLE the placement is under optimal play; lower = better siting). A neural metamodel
(the repo's SurrogateMLP) is fitted on CHEAP pre-solve features (no LP required at query time) and
validated the SBO way: held-out accuracy (RMSE, Spearman rank corr) + argmin regret (does the
surrogate's chosen placement match the true best?). Split is BY OD PAIR (a pair's designs never
straddle train/test), so the validation is generalisation to unseen placements, not interpolation
across N of a seen pair.

This is the reduced form of lit-review Obj-4 ("neural network metamodel to approximate facility
location and fleet composition"): the full SBO loop (acquisition, refinement, coupling to trained
policies) is future work. Run: PYTHONPATH=. .venv/bin/python scratch/sbo_placement_demo.py
"""
from __future__ import annotations

import itertools
import json
import math
import random
import time

import networkx as nx
import numpy as np
import torch
from scipy.stats import spearmanr

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS
from src.sbo.surrogate import SurrogateMLP, train_surrogate
from src.utils.graph_utils import load_osm_graph_and_demands

K = 1
BAND = (0.15, 0.95)
KX = 8
NS = (2, 3, 4)
N_PAIRS = 150
TEST_FRac = 0.25
SEED = 0

nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
G = nx.Graph()
for u, v, d in edges:
    G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
ALL_EDGES = list(G.edges())

# --- candidate placements (high-connectivity OD pairs, 3-6 disjoint routes) ---
deg3 = [n for n, d in G.degree() if d >= 3]
rng = random.Random(SEED)
seen, pairs = set(), []
while len(pairs) < N_PAIRS and len(seen) < 4000:
    s, t = rng.sample(deg3, 2)
    key = tuple(sorted((s, t), key=repr))
    if key in seen:
        continue
    seen.add(key)
    try:
        base_routes = build_route_set(G, s, t, 0, "w")
    except Exception:
        continue
    if 3 <= len(base_routes) <= 6:
        pairs.append((s, t))
print(f"[F3] {len(pairs)} candidate OD placements x N in {NS} = {len(pairs) * len(NS)} designs")


def design_features(s: str, t: str, routes, cand_edges, vuln: dict, N: int) -> list[float]:
    """CHEAP pre-solve features (no LP): placement geometry + route-set structure + threat stats,
    plus a theory-guided aggregate: on DISJOINT routes the single-convoy equilibrium value has the
    closed form 1 / sum_r(1 / p*_r) (p*_r = the route's worst edge vulnerability), so the harmonic
    aggregate of per-route worst vulnerabilities is the natural first-order predictor of
    loss_mixed; shared edges and the mission objective bend it, which is what the MLP learns."""
    costs = np.array([sum(G[u][v]["w"] for u, v in zip(r, r[1:])) for r in routes])
    vs = np.array([vuln[e] for e in cand_edges])
    esets = [edges_of_route(r) for r in routes]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(esets, 2)] or [0.0]
    sp = nx.shortest_path_length(G, s, t, weight="w")
    route_pmax = np.array([max(vuln[e] for e in es) for es in esets])
    harm = float(1.0 / np.sum(1.0 / np.clip(route_pmax, 1e-6, None)))
    return [sp, len(routes), len(cand_edges), costs.min(), costs.mean(), costs.std(),
            float(vs.mean()), float(vs.max()), float(np.mean(jac)), float(N),
            harm, float(route_pmax.min()), float(route_pmax.mean())]


rows, t0 = [], time.time()
for s, t in pairs:
    try:
        routes = build_route_set(G, s, t, KX, "w")
        cand = sorted(set().union(*(edges_of_route(r) for r in routes)), key=repr)
        vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=ALL_EDGES)
        game = build_interdiction_game(G, s, t, K, k_extra=KX, weight="w",
                                       intercept_fn=survival_intercept_fn(vuln))
    except Exception as e:
        print(f"  skip {s}-{t}: {e}")
        continue
    for N in NS:
        try:
            sol = solve_multiconvoy(game, N, "mission")
        except Exception as e:
            print(f"  skip {s}-{t} N={N}: {e}")
            continue
        rows.append({"od": f"{s}-{t}", "N": N,
                     "x": design_features(s, t, game.routes, cand, vuln, N),
                     "y": float(sol.loss_mixed), "loss_det": float(sol.loss_det)})
print(f"[F3] oracle dataset: {len(rows)} designs in {time.time() - t0:.0f}s")

# --- split BY OD PAIR ---
ods = sorted({r["od"] for r in rows})
rng.shuffle(ods)
n_test = max(3, int(len(ods) * TEST_FRac))
test_ods = set(ods[:n_test])
train = [r for r in rows if r["od"] not in test_ods]
test = [r for r in rows if r["od"] in test_ods]
Xtr = np.array([r["x"] for r in train], dtype=np.float32)
ytr = np.array([r["y"] for r in train], dtype=np.float32)
Xte = np.array([r["x"] for r in test], dtype=np.float32)
yte = np.array([r["y"] for r in test], dtype=np.float32)
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9   # normalise features on TRAIN stats only

torch.manual_seed(SEED)
model, losses = train_surrogate((Xtr - mu) / sd, ytr, epochs=300, lr=5e-3, batch_size=16, hidden_dim=32)
model.eval()
with torch.no_grad():
    pred = model(torch.tensor((Xte - mu) / sd)).squeeze(-1).numpy()

rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
rho, pval = spearmanr(pred, yte)
# argmin validation: the surrogate's chosen placement (lowest predicted equilibrium exploitability)
best_pred_i = int(np.argmin(pred))
best_true_i = int(np.argmin(yte))
regret = float(yte[best_pred_i] - yte[best_true_i])
top5_pred = set(np.argsort(pred)[:5].tolist())
top5_true = set(np.argsort(yte)[:5].tolist())

print(f"\n=== F3 SBO DEMONSTRATOR (Obj-4 reduced form) ===")
print(f"designs: {len(rows)} ({len(train)} train / {len(test)} test; split by OD pair, "
      f"{len(test_ods)} held-out placements)")
print(f"target: oracle equilibrium mission-failure (loss_mixed); range "
      f"[{min(r['y'] for r in rows):.3f}, {max(r['y'] for r in rows):.3f}]")
print(f"surrogate held-out: RMSE {rmse:.4f} | Spearman rho {rho:.3f} (p={pval:.1e})")
print(f"argmin validation: surrogate pick {test[best_pred_i]['od']} N={test[best_pred_i]['N']} "
      f"(true {yte[best_pred_i]:.3f}) vs true best {test[best_true_i]['od']} "
      f"N={test[best_true_i]['N']} ({yte[best_true_i]:.3f}) -> regret {regret:.4f}")
print(f"top-5 overlap (pred vs true): {len(top5_pred & top5_true)}/5")
out = {"config": {"K": K, "band": BAND, "k_extra": KX, "Ns": NS, "n_pairs": len(pairs), "seed": SEED},
       "n_designs": len(rows), "n_train": len(train), "n_test": len(test),
       "rmse": rmse, "spearman_rho": float(rho), "spearman_p": float(pval),
       "argmin_regret": regret, "top5_overlap": len(top5_pred & top5_true),
       "test_rows": [{"od": r["od"], "N": r["N"], "true": float(y), "pred": float(p)}
                     for r, y, p in zip(test, yte, pred)],
       "rows": rows}
with open("models/runs/sbo_placement_demo.json", "w") as f:
    json.dump(out, f, indent=2)
print("[written] models/runs/sbo_placement_demo.json")
