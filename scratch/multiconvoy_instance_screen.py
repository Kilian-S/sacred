"""Fork-A instance screen (gen08 Phase M, NO training): find a DISJOINT multi-convoy instance whose
single-convoy LEADER game is (a) ASYMMETRIC -> non-uniform leader equilibrium (entropy comfortably
below ln R, so fictitious play has a gradient, unlike the flat 33->71) and (b) has MARGIN -> a wide
ALNS/equilibrium ratio (aim >= 3x, so the leader's realistic ~2.2x-equilibrium landing still clears
ALNS). Reports (equilibrium, ALNS=loss_det, ratio, leader-entropy, disjoint?, #routes) before any
training. Absolute length->prob normalisation (cross-instance comparable), N=3, K=1, mission.
"""
from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.multiconvoy_interdiction import _DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS
from src.utils.graph_utils import load_osm_graph_and_demands

N, K = 3, 1
BANDS = [(0.15, 0.95), (0.05, 0.98), (0.30, 0.99)]

nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
G = nx.Graph()
for u, v, d in edges:
    G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
ALL_EDGES = list(G.edges())


def leader_stats(sol, R):
    d = np.asarray(sol.defender_strategy)
    lead = np.zeros(R); stacked = 0.0
    for i, occ in enumerate(sol.occupancies):
        nz = [r for r, c in enumerate(occ) if c > 0]
        if len(nz) == 1 and occ[nz[0]] == N:
            lead[nz[0]] = d[i]; stacked += d[i]
    if lead.sum() < 1e-9:
        return float("nan"), stacked
    p = lead / lead.sum()
    return float(-(p[p > 0] * np.log(p[p > 0])).sum()), stacked


def evaluate(s, t, band, k_extra):
    routes = build_route_set(G, s, t, k_extra, "w")
    cand = set().union(*(edges_of_route(r) for r in routes))
    vuln = length_band_vulnerability(G, cand, band=band, weight="w", norm_edges=ALL_EDGES)
    game = build_interdiction_game(G, s, t, K, k_extra=k_extra, weight="w",
                                   intercept_fn=survival_intercept_fn(vuln))
    sol = solve_multiconvoy(game, N, "mission")
    R = game.n_routes
    H_lead, stacked = leader_stats(sol, R)
    ratio = sol.loss_det / sol.loss_mixed if sol.loss_mixed > 1e-6 else float("inf")
    return dict(od=f"{s}-{t}", R=R, kx=k_extra, band=band, eq=sol.loss_mixed, alns=sol.loss_det,
                ratio=ratio, H_lead=H_lead, lnR=math.log(R), Hrel=H_lead / math.log(R),
                stacked=stacked)


# candidate disjoint OD pairs (3..6 routes)
deg3 = [n for n, d in G.degree() if d >= 3]
rng = random.Random(0)
seen, cands = set(), []
while len(cands) < 70 and len(seen) < 3000:
    s, t = rng.sample(deg3, 2)
    key = tuple(sorted((s, t), key=repr))
    if key in seen:
        continue
    seen.add(key)
    try:
        routes = build_route_set(G, s, t, 0, "w")
    except Exception:
        continue
    if 3 <= len(routes) <= 6:
        cands.append((s, t, routes))
# always include the two reference instances
for s, t in [("33", "71"), ("110", "135")]:
    if (s, t) not in [(a, b) for a, b, _ in cands]:
        try:
            cands.append((s, t, build_route_set(G, s, t, 0, "w")))
        except Exception:
            pass

rows = []
SHARED = 8  # k-shortest extra routes -> shared edges -> asymmetric (non-uniform) leader equilibrium
for s, t, _routes in cands:
    for band in BANDS:
        for kx in (0, SHARED):
            try:
                rows.append(evaluate(s, t, band, kx))
            except Exception:
                pass

hdr = (f"{'OD':>9} {'R':>2} {'kx':>3} {'band':>12} {'eq':>6} {'ALNS':>6} {'ratio':>6} "
       f"{'H_lead':>7} {'lnR':>5} {'H/lnR':>6} {'stack':>6}")


def fmt(r):
    return (f"{r['od']:>9} {r['R']:>2} {r['kx']:>3} {str(r['band']):>12} {r['eq']:>6.3f} "
            f"{r['alns']:>6.3f} {r['ratio']:>6.2f} {r['H_lead']:>7.3f} {r['lnR']:>5.2f} "
            f"{r['Hrel']:>6.2f} {r['stacked']:>6.2f}")


print(f"screened {len(cands)} OD pairs x {len(BANDS)} bands x [disjoint, +{SHARED} shared] "
      f"= {len(rows)} instances (N={N}, K={K}, mission, absolute norm)\n")

short = [r for r in rows if r["ratio"] >= 3.0 and r["Hrel"] <= 0.85 and r["eq"] > 0.02]
short.sort(key=lambda r: (-r["ratio"], r["Hrel"]))
print("=== SHORTLIST: ratio>=3x AND leader H<=0.85*lnR (asymmetry + margin) ===")
print(hdr)
for r in short[:20]:
    print(fmt(r))
if not short:
    print("  (none)")

print("\n=== best BY COMBINED SCORE ratio * (1 - Hrel) among eq>0.03 (asymmetry x margin), top 15 ===")
print(hdr)
for r in sorted([r for r in rows if r["eq"] > 0.03], key=lambda r: -(r["ratio"] * (1 - r["Hrel"])))[:15]:
    print(fmt(r))

print("\n=== most non-uniform leader (lowest H/lnR) among ratio>=3.0, top 12 ===")
print(hdr)
for r in sorted([r for r in rows if r["ratio"] >= 3.0], key=lambda r: r["Hrel"])[:12]:
    print(fmt(r))

print("\n=== reference rows (current instances, disjoint + shared) ===")
print(hdr)
for r in sorted([x for x in rows if x["od"] in ("33-71", "110-135")], key=lambda r: (r["od"], r["kx"], str(r["band"]))):
    print(fmt(r))
