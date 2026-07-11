"""B1-lite-1 ORACLE SCREEN (free, NO training): the within-episode PATTERN-OF-LIFE game.

Restores the D (within-episode dynamism) to the interdiction headline. An episode = S sorties; at
each sortie the fleet stacks on ONE route; the interdictor commits K assets by SOFTMAX-best-responding
to the defender's REALISED routes over a trailing WINDOW of w recent sorties (pattern-of-life: the
enemy positions against your recent operations, not your long-run distribution). Soft interception,
mission objective, latency-free (isolates the pure dynamism effect).

Because the softmax-BR adversary is a DETERMINISTIC-transition function of the window, the defender's
optimal HISTORY-DEPENDENT policy is an average-cost MDP over the window state, solved EXACTLY by
relative value iteration - the computable yardstick is preserved. This screen reports, per (w, tau):

  * V_eq          : the single-shot STACKED equilibrium value (the current static headline anchor,
                    = a static-mixed defender vs a NON-adaptive equilibrium adversary);
  * static_det    : a deterministic defender (same route every sortie) vs the adaptive adversary;
  * iid_eq        : the STATIC-mixed defender (plays V_eq's mixture i.i.d.) vs the ADAPTIVE adversary;
  * history_opt   : the best HISTORY-DEPENDENT defender vs the adaptive adversary (the D payoff).

The claim the screen tests: static_det >> iid_eq > history_opt, i.e. (i) an adaptive adversary makes
a static-mixed defender suffer above its equilibrium value, and (ii) a history-aware (dynamic)
defender recovers below it - dynamism pays on both sides, with a computable optimum.

Run: PYTHONPATH=. .venv/bin/python scratch/within_episode_screen.py
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import _row_minimiser, objective_value, solve_multiconvoy
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS
from src.utils.graph_utils import load_osm_graph_and_demands

N, K, BAND, KX = 3, 1, (0.15, 0.95), 8
ODS = [("35", "159"), ("62", "97")]
WS = [1, 2, 3]
TAUS = [0.05, 0.15]

nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
G = nx.Graph()
for u, v, d in edges:
    G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
ALL = list(G.edges())


def stacked_L(game):
    R, nj = game.n_routes, game.payoff.shape[1]
    L = np.zeros((R, nj))
    for r in range(R):
        occ = np.zeros(R, dtype=int); occ[r] = N
        for j in range(nj):
            L[r, j] = objective_value(occ, game.payoff[:, j], N, "mission")
    return L


def softmax_br(counts, L, tau):
    tot = counts.sum()
    d = counts / tot if tot > 0 else np.ones(len(counts)) / len(counts)
    e = d @ L
    z = np.exp((e - e.max()) / tau)
    return z / z.sum()


def iid_eq_cost(L, eq, w, tau):
    R = len(eq)
    total = 0.0
    eqL = eq @ L  # expected loss vector over isets for a current draw ~ eq
    for window in itertools.product(range(R), repeat=w):
        pw = 1.0
        for i in window:
            pw *= eq[i]
        if pw == 0.0:
            continue
        counts = np.bincount(window, minlength=R).astype(float)
        total += pw * float(eqL @ softmax_br(counts, L, tau))
    return total


def static_det_cost(L, w, tau):
    R = L.shape[0]
    best = np.inf
    for r in range(R):
        counts = np.zeros(R); counts[r] = w
        best = min(best, float(L[r] @ softmax_br(counts, L, tau)))
    return best


def history_opt_cost(L, w, tau):
    R = L.shape[0]
    states = list(itertools.product(range(R), repeat=w))
    idx = {s: i for i, s in enumerate(states)}
    cost = np.zeros((len(states), R)); nxt = np.zeros((len(states), R), dtype=int)
    for si, s in enumerate(states):
        counts = np.bincount(s, minlength=R).astype(float)
        p = softmax_br(counts, L, tau)
        cost[si] = L @ p
        for r in range(R):
            nxt[si, r] = idx[s[1:] + (r,)]
    V = np.zeros(len(states)); ref = 0
    for _ in range(5000):
        Q = cost + V[nxt]
        Vn = Q.min(axis=1)
        g = Vn[ref] - V[ref]
        Vn = Vn - Vn[ref]
        if np.max(np.abs(Vn - V)) < 1e-10:
            V = Vn; break
        V = Vn
    Q = cost + V[nxt]
    return float((Q.min(axis=1) - V)[ref])


for od in ODS:
    routes = build_route_set(G, od[0], od[1], KX, "w")
    cand = set().union(*(edges_of_route(r) for r in routes))
    vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=ALL)
    game = build_interdiction_game(G, od[0], od[1], K, k_extra=KX, weight="w",
                                   intercept_fn=survival_intercept_fn(vuln))
    L = stacked_L(game)
    v_eq, eq = _row_minimiser(L)   # single-shot stacked equilibrium (static headline anchor)
    print(f"\n=== {od[0]}-{od[1]} (R={game.n_routes}, N={N}, K={K}) | single-shot stacked eq V_eq={v_eq:.3f} ===")
    print(f"  {'w':>2} {'tau':>5} | {'static_det':>10} {'iid_eq':>8} {'history_opt':>11} | "
          f"{'iid/V_eq':>8} {'hist/iid':>8}")
    for w in WS:
        for tau in TAUS:
            sd = static_det_cost(L, w, tau)
            ie = iid_eq_cost(L, eq, w, tau)
            ho = history_opt_cost(L, w, tau)
            print(f"  {w:>2} {tau:>5.2f} | {sd:>10.3f} {ie:>8.3f} {ho:>11.3f} | "
                  f"{ie / v_eq:>8.2f} {ho / ie:>8.2f}")
print("\nRead: dynamism pays if static_det >> iid_eq AND history_opt < iid_eq (a history-aware "
      "defender beats the static-mixed one against the pattern-of-life adversary), with a "
      "meaningful gap -> the SACRED experiment (env exposes the window; does RL reach history_opt?).")
