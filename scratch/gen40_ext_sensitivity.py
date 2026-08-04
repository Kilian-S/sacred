#!/usr/bin/env python3
"""gen40 EXTENSION sweep (ORACLE-ONLY, no training): w to 10, m towards 10 (screened), exact
K to 5 where the wall permits, menus to ~20 routes, and the greedy-adversary budget ladder
(a DIFFERENT game, reported separately per binding rule 5).

Pre-registered as the EXTENSION section of experiments/gen40_dyn_sensitivity.md. Every
guard-skipped cell is recorded.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     scratch/gen40_ext_sensitivity.py
Writes models/runs/gen40_ext_sensitivity.json
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

from scratch.critique_followup_probes import disjoint_subset, rotation_value
from scratch.dyn_exact import damped_rvi, karp_mmc
from scratch.gen40_dyn_sensitivity import (
    BAND, IID_SUPPORT_CAP, N, TAU, best_rotation, enum_windows, inv_vuln_dist,
    static_stationary, uniform_dist, window_losses)
from scripts.train_b1lite1 import softmax_br, stacked_L
from scripts.train_generalist import CITY_PATHS
from src.baselines.interdiction_oracle import build_route_set
from src.baselines.multiconvoy_oracle import _row_minimiser, objective_value
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.utils.graph_utils import load_osm_graph_and_demands

torch.set_num_threads(1)

CITY_PATHS = dict(CITY_PATHS)
CITY_PATHS["kyiv"] = ("data/maps/kyiv/nodes.geojson", "data/maps/kyiv/edges.geojson")

KARP_CAP = 8_000          # Karp memory wall; damped RVI above it
FULL_CAP = 8_000          # full-menu optimum (Karp) where R^w fits
STATE_CAP = 2_000_000     # hard state-count guard
WORK_CAP = 6e9            # states x columns guard
ISET_CAP = 1_000_000      # raised exact-column cap (extension C)
OCC_CAP = 1.5e8           # occupancy-matrix memory guard (n_occ x n_isets)
OUT_PATH = "models/runs/gen40_ext_sensitivity.json"


# ------------------------------------------------------------------ shared exact machinery
def opt_exact(lw_core, b, w):
    """Karp below the memory wall, damped RVI above it; returns (value, method, converged)."""
    n = b ** w
    cost = lw_core
    if n <= KARP_CAP:
        return karp_mmc(cost, n, b, b ** (w - 1)), "karp", True
    iters = 6_000 if n > 200_000 else 30_000
    g, conv = damped_rvi(cost, n, b, b ** (w - 1), iters=iters, tol=1e-9)
    return g, "rvi", conv


def antirepeat_fast(lw, dec, sub, w, damp=0.5, tol=1e-11):
    """Composed anti-repeat stationary value; bincount-based (fast at large n)."""
    b = len(sub)
    n = b ** w
    inwin = np.zeros((n, b), dtype=bool)
    for j in range(b):
        inwin[:, j] = (dec == j).any(axis=1)
    allowed = ~inwin
    allowed[~allowed.any(axis=1)] = True
    na = allowed.sum(axis=1)
    lsub = lw[:, sub] if lw.shape[1] != b else lw
    c = np.where(allowed, lsub, 0.0).sum(axis=1) / na
    w_a = allowed / na[:, None]
    heads = (np.arange(n) % (b ** (w - 1))) * b
    pi = np.ones(n) / n
    iters = 4_000 if n > 100_000 else 40_000
    for _ in range(iters):
        nxt = np.zeros(n)
        for a in range(b):
            nxt += np.bincount(heads + a, weights=pi * w_a[:, a], minlength=n)
        nxt = damp * nxt + (1 - damp) * pi
        if np.max(np.abs(nxt - pi)) < tol:
            pi = nxt
            break
        pi = nxt
    return float(pi @ c)


def softmax_cell(env, K, kx, w, label):
    """One exact softmax-adversary cell (extension conventions; guards recorded)."""
    game = env.game
    L = stacked_L(game, N)
    R = L.shape[0]
    C = L.shape[1]
    dis = disjoint_subset([set(e) for e in game.route_edges])
    m = len(dis)
    n_core = m ** w
    if n_core > STATE_CAP or n_core * C > WORK_CAP:
        return dict(od=label, m=m, K=K, k_extra=kx, R=R, w=w,
                    skipped=f"guard n={n_core} C={C}")
    t0 = time.time()
    dec_c, counts_c = enum_windows(dis, w, R)
    lw_c = window_losses(counts_c, L, TAU)
    opt_core, method, conv = opt_exact(lw_c[:, dis], m, w)
    anti_core = antirepeat_fast(lw_c, dec_c, dis, w)
    rot = best_rotation(dis, L, TAU, w)
    opt_full = None
    if R ** w <= FULL_CAP and (R ** w) * C <= 2.5e9:
        full = list(range(R))
        _, counts_f = enum_windows(full, w, R)
        lw_f = window_losses(counts_f, L, TAU)
        opt_full = karp_mmc(lw_f, R ** w, R, R ** (w - 1))
    v_eq, eq = _row_minimiser(L)
    sup = int((eq > 1e-12).sum())
    iid_eq = static_stationary(eq, L, TAU, w) \
        if sup ** w <= 250_000 and sup ** w * C <= 2.5e9 else None
    st_uni = static_stationary(uniform_dist(dis, R), L, TAU, w) \
        if n_core * C <= 2.5e9 else None
    naive = min(rot, anti_core)
    opt = opt_full if opt_full is not None else opt_core
    return dict(od=label, m=m, K=K, k_extra=kx, R=R, w=w, n_isets=int(C),
                opt=opt, opt_core=opt_core, opt_full=opt_full,
                opt_method=method, opt_converged=bool(conv),
                rotation=rot, anti_core=anti_core, iid_eq=iid_eq,
                static_uni_core=st_uni, v_eq_oneshot=v_eq,
                best_naive=naive,
                best_naive_name="rotation" if rot <= anti_core else "anti_core",
                naive_over_opt=naive / max(opt, 1e-12),
                iid_over_opt=(iid_eq / max(opt, 1e-12)) if iid_eq else None,
                secs=round(time.time() - t0, 1))


# ------------------------------------------------------------------ greedy-adversary tier
def edge_vuln_from_k1(env_k1):
    """{edge_key: p_e} recovered from the K=1 game (columns are single edges; the route-level
    payoff of a crossing route IS p_e). The line-127 convention of the env, rebuilt here."""
    game = env_k1.game
    return {iset[0]: float(game.payoff[:, j].max())
            for j, iset in enumerate(game.interdiction_sets)}


def _pool_machinery(env_k1):
    """Shared pieces for the pool-softmax adversary: incidence, vulns, greedy set builder,
    and the per-set stacked mission-loss vector."""
    game = env_k1.game
    route_edges = [frozenset(e) for e in game.route_edges]
    R = len(route_edges)
    vuln = edge_vuln_from_k1(env_k1)
    cand = sorted(set().union(*route_edges), key=repr)
    ve = np.array([vuln.get(e, 1.0) for e in cand])
    inc = np.array([[1.0 if cand[c] in route_edges[r] else 0.0
                     for c in range(len(cand))] for r in range(R)])
    log_surv_edge = np.log1p(-np.clip(ve, 0.0, 1.0 - 1e-12))

    def loss_of_set(chosen):
        ls = inc[:, chosen] @ log_surv_edge[chosen] if chosen else np.zeros(R)
        p_r = 1.0 - np.exp(ls)
        return 1.0 - (1.0 - p_r) ** N                                  # stacked mission per route

    def greedy_set(d, K):
        chosen: list[int] = []
        remaining = list(range(len(cand)))
        cur_ls = np.zeros(R)
        for _ in range(min(K, len(cand))):
            best_c, best_v = None, -1.0
            for c in remaining:
                p_r = 1.0 - np.exp(cur_ls + inc[:, c] * log_surv_edge[c])
                v = float(d @ (1.0 - (1.0 - p_r) ** N))
                if v > best_v + 1e-12:
                    best_v, best_c = v, c
            chosen.append(best_c)
            remaining.remove(best_c)
            cur_ls = cur_ls + inc[:, best_c] * log_surv_edge[best_c]
        return tuple(sorted(chosen))

    return R, cand, ve, inc, loss_of_set, greedy_set


def greedy_costs(env_k1, K, w):
    """cost[s, a] for the POOL-SOFTMAX adversary (the E-REVISED design): softmax(tau) over a
    deterministic per-count-signature candidate pool of K-sets. Memoised on counts."""
    R, cand, ve, inc, loss_of_set, greedy_set = _pool_machinery(env_k1)
    game = env_k1.game
    dis = disjoint_subset([set(e) for e in game.route_edges])
    L1 = stacked_L(game, N)
    _, eq = _row_minimiser(L1)
    uni_core = uniform_dist(dis, R)
    static_pool = [greedy_set(uni_core, K), greedy_set(eq, K),
                   tuple(sorted(np.argsort(-ve)[:K].tolist()))]        # top-K raw vulnerability

    def response(counts):
        d = counts / counts.sum()
        pool = {greedy_set(d, K)}
        for r in np.where(counts > 0)[0]:
            er = np.zeros(R)
            er[r] = 1.0
            pool.add(greedy_set(er, K))
        pool.update(static_pool)
        sets = sorted(pool)
        losses = np.stack([loss_of_set(list(s)) for s in sets])        # [P, R]
        damages = losses @ d
        z = np.exp((damages - damages.max()) / TAU)
        q = z / z.sum()
        return q @ losses                                              # expected loss per action

    n = R ** w
    dec, counts = enum_windows(list(range(R)), w, R)
    cost = np.empty((n, R))
    memo: dict[tuple, np.ndarray] = {}
    for s in range(n):
        key = tuple(counts[s].astype(int))
        if key not in memo:
            memo[key] = response(counts[s])
        cost[s] = memo[key]
    return cost, dec, n, R


def greedy_cell(env_k1, K, w, label):
    t0 = time.time()
    game = env_k1.game
    L1 = stacked_L(game, N)
    R = L1.shape[0]
    dis = disjoint_subset([set(e) for e in game.route_edges])
    m = len(dis)
    cost, dec, n, _ = greedy_costs(env_k1, K, w)
    opt = karp_mmc(cost, n, R, R ** (w - 1))
    # rules under the SAME greedy enemy (cost rows already per action)
    rot_orders = [tuple(dis)]
    rng = np.random.default_rng(0)
    while len(rot_orders) < min(20, math.factorial(m)):
        p = tuple(rng.permutation(dis).tolist())
        if p not in rot_orders:
            rot_orders.append(p)

    def rot_val(seq):
        mm = len(seq)
        v = 0.0
        for t in range(mm):
            win = [seq[(t - 1 - i) % mm] for i in range(w)]
            sid = 0
            for x in win:
                sid = sid * R + x
            v += cost[sid, seq[t]]
        return v / mm

    rot = min(rot_val(list(o)) for o in rot_orders)
    anti = _anti_on_costs(cost, dec, dis, w, R)
    uni = uniform_dist(dis, R)
    st_uni = _static_on_costs(cost, uni, w, R)
    naive = min(rot, anti)
    return dict(od=label, m=m, K=K, k_extra=8, R=R, w=w, adversary="greedy",
                opt=opt, rotation=rot, anti_core=anti, static_uni_core=st_uni,
                best_naive=naive,
                best_naive_name="rotation" if rot <= anti else "anti_core",
                naive_over_opt=naive / max(opt, 1e-12),
                static_over_opt=st_uni / max(opt, 1e-12),
                secs=round(time.time() - t0, 1))


def _anti_on_costs(cost, dec, dis, w, R, damp=0.5, iters=40_000, tol=1e-12):
    """Composed anti-repeat under an arbitrary cost table on FULL-menu window ids."""
    b = len(dis)
    nb = b ** w
    sub_dec, _ = enum_windows(dis, w, R)
    full_id = np.zeros(nb, dtype=np.int64)
    for i in range(w):
        full_id = full_id * R + np.array(dis)[sub_dec[:, i]]
    inwin = np.zeros((nb, b), dtype=bool)
    for j in range(b):
        inwin[:, j] = (sub_dec == j).any(axis=1)
    allowed = ~inwin
    allowed[~allowed.any(axis=1)] = True
    na = allowed.sum(axis=1)
    lsub = cost[full_id][:, dis]
    c = np.where(allowed, lsub, 0.0).sum(axis=1) / na
    w_a = allowed / na[:, None]
    heads = (np.arange(nb) % (b ** (w - 1))) * b
    pi = np.ones(nb) / nb
    for _ in range(iters):
        nxt = np.zeros(nb)
        for a in range(b):
            nxt += np.bincount(heads + a, weights=pi * w_a[:, a], minlength=nb)
        nxt = damp * nxt + (1 - damp) * pi
        if np.max(np.abs(nxt - pi)) < tol:
            pi = nxt
            break
        pi = nxt
    return float(pi @ c)


def _static_on_costs(cost, dist, w, R):
    sup = np.where(dist > 1e-12)[0]
    dec, _ = enum_windows(list(sup), w, R)
    full_id = np.zeros(len(dec), dtype=np.int64)
    for i in range(w):
        full_id = full_id * R + sup[dec[:, i]]
    p = dist[sup]
    wts = p[dec].prod(axis=1)
    return float((wts * (cost[full_id] @ dist)).sum())


# ------------------------------------------------------------------ the m >= 7 screen
def screen_high_m(cities=("istanbul", "kyiv"), targets=(7, 8, 9, 10), tries=1_200):
    found, record = {}, []
    for city in cities:
        nodes_path, edges_path = CITY_PATHS[city]
        nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
        G = nx.Graph()
        for u, v, d in edges:
            G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        degs = dict(G.degree())
        max_deg = max(degs.values())
        record.append(dict(city=city, max_degree=int(max_deg)))
        for tm in targets:
            if tm in found:
                continue
            elig = sorted(nd for nd, dg in degs.items() if dg >= tm)
            if len(elig) < 2:
                record.append(dict(city=city, target_m=tm, result="no deg>=m nodes"))
                continue
            rng = random.Random(0)
            pairs = [(a, b) for i, a in enumerate(elig) for b in elig[i + 1:]]
            rng.shuffle(pairs)
            pairs = pairs[:tries]
            seen = set()
            hit = None
            for s, t in pairs:
                seen.add((s, t))
                try:
                    base = build_route_set(G, s, t, 0, "w")
                except Exception:
                    continue
                if len(base) != tm:
                    continue
                # validate the built instance per the pre-registration (core == m at the
                # k8 menu; one-shot value >= 0.05)
                try:
                    nodes_path2, edges_path2 = CITY_PATHS[city]
                    env = make_multiconvoy_env((s, t), N=N, K=1, k_extra_routes=8,
                                               menu_select=True, edge_vuln_band=BAND,
                                               nodes_path=nodes_path2,
                                               edges_path=edges_path2)
                except Exception:
                    continue
                core = disjoint_subset([set(e) for e in env.game.route_edges])
                if len(core) != tm:
                    continue
                v_eq, _ = _row_minimiser(stacked_L(env.game, N))
                if v_eq < 0.05:
                    continue
                hit = (s, t)
                break
            record.append(dict(city=city, target_m=tm, tried=len(seen),
                               result=list(hit) if hit else "dry"))
            if hit:
                found[tm] = (city, hit)
    return found, record


# ------------------------------------------------------------------ main
def main():
    import sys
    tiers = sys.argv[1] if len(sys.argv) > 1 else "ABCDE"
    t_start = time.time()
    rows, greedy_rows = [], []
    screen_rec = []

    def dump():
        out = dict(config=dict(N=N, band=BAND, tau=TAU, karp_cap=KARP_CAP,
                               state_cap=STATE_CAP, work_cap=WORK_CAP,
                               iset_cap=ISET_CAP, occ_cap=OCC_CAP, tiers=tiers),
                   screen=screen_rec, rows=rows, greedy_rows=greedy_rows,
                   total_secs=round(time.time() - t_start, 1))
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=1)
    base_ods = [("klg-23-242", ("23", "242"), "kaliningrad"),
                ("35-159", ("35", "159"), "kaliningrad"),
                ("klg-29-80", ("29", "80"), "kaliningrad"),
                ("71-33", ("71", "33"), "kaliningrad")]

    def build(od, city, K, kx):
        nodes_path, edges_path = CITY_PATHS[city]
        return make_multiconvoy_env(od, N=N, K=K, k_extra_routes=kx, menu_select=True,
                                    edge_vuln_band=BAND, nodes_path=nodes_path,
                                    edges_path=edges_path)

    def run_block(label, od, city, kxs, ks, ws):
        for kx in kxs:
            env1 = build(od, city, 1, kx)
            n_E = env1.game.payoff.shape[1]
            R = env1.game.n_routes
            n_occ = math.comb(R + 2, 3)
            for K in ks:
                nC = math.comb(n_E, K)
                if nC > ISET_CAP or n_occ * nC > OCC_CAP:
                    rows.append(dict(od=label, K=K, k_extra=kx,
                                     skipped=f"wall C({n_E},{K})={nC}"))
                    print(f"{label} kx={kx} K={K}: past the wall "
                          f"(C={nC:,}; occ x C={n_occ * nC:.1e})", flush=True)
                    continue
                env = env1 if K == 1 else build(od, city, K, kx)
                for w in ws:
                    row = softmax_cell(env, K, kx, w, label)
                    rows.append(row)
                    if "skipped" in row:
                        print(f"{label} kx={kx} K={K} w={w} SKIP {row['skipped']}",
                              flush=True)
                        continue
                    ii = "-" if row["iid_eq"] is None else f"{row['iid_eq']:.4f}"
                    print(f"{label} m={row['m']} kx={kx} K={K} R={row['R']} w={w} | "
                          f"opt {row['opt']:.4f} ({row['opt_method']}"
                          f"{'' if row['opt_converged'] else ' UNCONV'}) "
                          f"rot {row['rotation']:.4f} antiC {row['anti_core']:.4f} "
                          f"iid {ii} | rule/opt {row['naive_over_opt']:.2f} "
                          f"({row['secs']}s)", flush=True)

    if "A" in tiers:
        print("=== A: window to 10 (kx=8, K 1-3) ===", flush=True)
        for label, od, city in base_ods:
            run_block(label, od, city, kxs=[8], ks=[1, 2, 3], ws=[6, 7, 8, 9, 10])
        dump()

    if "B" in tiers:
        print("=== B: the m>=7 screen (Istanbul, Kyiv) ===", flush=True)
        found, screen_rec[:] = screen_high_m()
        for tm, (city, od) in sorted(found.items()):
            label = f"{city}-{od[0]}-{od[1]}-m{tm}"
            print(f"  m={tm}: {city} {od}", flush=True)
            run_block(label, od, city, kxs=[0, 8], ks=[1, 2, 3], ws=[1, 2, 3, 4, 5])
        if not found:
            print("  screen DRY at every target m (recorded)", flush=True)
        dump()

    if "C" in tiers:
        print("=== C: exact budget to 5 ===", flush=True)
        kplan = [("klg-23-242", ("23", "242"), "kaliningrad", [(0, [4])]),
                 ("35-159", ("35", "159"), "kaliningrad", [(0, [4])]),
                 ("71-33", ("71", "33"), "kaliningrad", [(0, [4, 5]), (4, [4]), (8, [4])])]
        for label, od, city, plan in kplan:
            for kx, ks in plan:
                run_block(label, od, city, kxs=[kx], ks=ks, ws=[1, 2, 3, 4, 5])
        dump()

    if "D" in tiers:
        print("=== D: menu to ~20 (w=3) ===", flush=True)
        for label, od, city in base_ods:
            run_block(label, od, city, kxs=[12, 16], ks=[1, 2, 3], ws=[3])
        dump()

    if "E" not in tiers:
        dump()
        print(f"done ({round(time.time() - t_start, 1)}s total)", flush=True)
        return
    print("=== E: greedy-adversary ladder (different game; w=3, kx=8) ===", flush=True)
    for label, od, city in base_ods:
        env1 = build(od, city, 1, 8)
        # sanity: the greedy builder at K=1 against a pure stack on route r must recover
        # exactly the worst single edge of that route (value equality against the exact
        # K=1 stacked loss matrix; validates the vulnerability recovery + machinery)
        R1, _, _, _, loss_of_set, greedy_set = _pool_machinery(env1)
        L1 = stacked_L(env1.game, N)
        for r in range(R1):
            er = np.zeros(R1)
            er[r] = 1.0
            s1 = greedy_set(er, 1)
            assert abs(loss_of_set(list(s1))[r] - L1[r].max()) < 1e-9, \
                f"pool-adversary K=1 sanity failed on route {r}"
        for K in [3, 4, 5, 6, 8, 10]:
            row = greedy_cell(env1, K, 3, label)
            greedy_rows.append(row)
            print(f"{label} GREEDY K={K} | opt {row['opt']:.4f} rot {row['rotation']:.4f} "
                  f"antiC {row['anti_core']:.4f} statU {row['static_uni_core']:.4f} | "
                  f"rule/opt {row['naive_over_opt']:.2f} stat/opt "
                  f"{row['static_over_opt']:.2f} ({row['secs']}s)", flush=True)

    dump()
    print(f"wrote {OUT_PATH} ({round(time.time() - t_start, 1)}s total)", flush=True)


if __name__ == "__main__":
    main()
