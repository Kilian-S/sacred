#!/usr/bin/env python3
"""Oracle-only sensitivity sweep of the dynamic register (experiment gen40).

Maps the register's structure along window length w, corridor count m, adversary budget K and
menu size R. Every quantity is exact: dynamic optima come from Karp's minimum mean cycle,
restricted to the disjoint core wherever R^w exceeds FULL_STATE_CAP and cross-checked against the
full-menu optimum wherever both are computable. No training.
"""
from __future__ import annotations

import json
import math
import random
import time

import networkx as nx
import numpy as np
import torch

from analysis.critique_followup_probes import disjoint_subset, rotation_value
from analysis.dyn_exact import karp_mmc
from scripts.train_b1lite1 import softmax_br, stacked_L
from scripts.train_generalist import CITY_PATHS
from src.baselines.interdiction_oracle import build_route_set
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS, make_multiconvoy_env
from src.utils.graph_utils import load_osm_graph_and_demands

torch.set_num_threads(1)

TAU, N, BAND = 0.15, 3, (0.15, 0.95)
WS = [1, 2, 3, 4, 5]
KS = [1, 2, 3]
KXS = [0, 4, 8]
FULL_STATE_CAP = 3_200      # full-menu exact objects only when R^w <= this
IID_SUPPORT_CAP = 70_000    # iid_eq exact enumeration cap (support^w)
MAX_ISETS = 250_000         # skip cells past this column count


# ---------------------------------------------------------------- exact helpers (vectorised)
def enum_windows(sub, w, R):
    """Enumerate all windows over the route subset ``sub``.

    Base-|sub| encoding with s_0 oldest, the same ordering convention as analysis/dyn_exact.py.

    Returns:
        The decoded windows [n, w] in subset indices, and counts [n, R] over full route ids.
    """
    b = len(sub)
    n = b ** w
    dec = np.empty((n, w), dtype=np.int64)
    x = np.arange(n)
    for i in range(w):
        dec[:, w - 1 - i] = x % b
        x = x // b
    counts = np.zeros((n, R))
    for j in range(b):
        counts[:, sub[j]] = (dec == j).sum(axis=1)
    return dec, counts


def window_losses(counts, L, tau, chunk=256):
    """Per-route expected loss L @ q(s) for every window state, batched into [n, R].

    Follows softmax_br semantics exactly, with counts normalised by their sum.
    """
    n = counts.shape[0]
    out = np.empty((n, L.shape[0]))
    for i in range(0, n, chunk):
        cm = counts[i:i + chunk]
        D = cm / cm.sum(axis=1, keepdims=True)
        E = D @ L
        E = (E - E.max(axis=1, keepdims=True)) / tau
        Q = np.exp(E)
        Q /= Q.sum(axis=1, keepdims=True)
        out[i:i + chunk] = Q @ L.T
    return out


def restricted_opt(lw, sub, w):
    """Exact dynamic optimum of the policy class restricted to the routes ``sub``.

    Karp's algorithm on the base-|sub| window graph; ``lw`` is window_losses for
    enum_windows(sub, w).
    """
    b = len(sub)
    cost = lw[:, sub]
    return karp_mmc(cost, b ** w, b, b ** (w - 1))


def antirepeat_stationary(lw, dec, sub, w, damp=0.5, iters=60_000, tol=1e-13):
    """Exact stationary loss of playing uniformly over the ``sub`` routes absent from the window.

    Falls back to uniform over ``sub`` when every route is punished. Damped power iteration on
    the window chain.
    """
    b = len(sub)
    n = b ** w
    inwin = np.zeros((n, b), dtype=bool)
    for j in range(b):
        inwin[:, j] = (dec == j).any(axis=1)
    allowed = ~inwin
    allowed[~allowed.any(axis=1)] = True
    na = allowed.sum(axis=1)
    lsub = lw[:, sub]
    c = np.where(allowed, lsub, 0.0).sum(axis=1) / na
    w_a = allowed / na[:, None]
    heads = (np.arange(n) % (b ** (w - 1))) * b
    pi = np.ones(n) / n
    for _ in range(iters):
        nxt = np.zeros(n)
        for a in range(b):
            np.add.at(nxt, heads + a, pi * w_a[:, a])
        nxt = damp * nxt + (1 - damp) * pi
        if np.max(np.abs(nxt - pi)) < tol:
            pi = nxt
            break
        pi = nxt
    return float(pi @ c)


def static_stationary(dist, L, tau, w, cap=IID_SUPPORT_CAP):
    """Exact stationary loss of a fixed mixture `dist` played iid every sortie."""
    sup = np.where(dist > 1e-12)[0]
    if len(sup) ** w > cap:
        return None
    dec, counts = enum_windows(list(sup), w, L.shape[0])
    lw = window_losses(counts, L, tau)
    p = dist[sup]
    wts = p[dec].prod(axis=1)
    return float((wts * (lw @ dist)).sum())


def best_rotation(dis, L, tau, w):
    orders = [tuple(dis)]
    rng = np.random.default_rng(0)
    n_perm = min(20, math.factorial(len(dis)))
    seen = set(orders)
    while len(orders) < n_perm:
        p = tuple(rng.permutation(dis).tolist())
        if p not in seen:
            seen.add(p)
            orders.append(p)
    return min(rotation_value(list(o), L, tau, w) for o in orders)


def uniform_dist(sub, R):
    d = np.zeros(R)
    d[np.asarray(sub)] = 1.0 / len(sub)
    return d


def inv_vuln_dist(sub, L, R):
    q = np.array([L[r].max() for r in sub])
    w_ = 1.0 / np.maximum(q, 1e-9)
    d = np.zeros(R)
    d[np.asarray(sub)] = w_ / w_.sum()
    return d


# ---------------------------------------------------------------- instance screen
def screen_od(city, target_m, kx=8, seed=0, tries=4000):
    """Find the first degree-3 OD whose disjoint count and menu core both equal ``target_m``.

    The menu-size and one-shot-value screens are applied as well.
    """
    nodes_path, edges_path = CITY_PATHS[city]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    deg3 = sorted(n for n, d in G.degree() if d >= 3)
    rng = random.Random(seed)
    seen = set()
    while len(seen) < tries:
        s, t = rng.sample(deg3, 2)
        key = (s, t) if s < t else (t, s)
        if key in seen:
            continue
        seen.add(key)
        try:
            base = build_route_set(G, s, t, 0, "w")
        except Exception:
            continue
        if len(base) != target_m:
            continue
        try:
            env = make_multiconvoy_env((s, t), N=N, K=1, k_extra_routes=kx,
                                       menu_select=True, edge_vuln_band=BAND,
                                       nodes_path=nodes_path, edges_path=edges_path)
        except Exception:
            continue
        game = env.game
        if len(disjoint_subset([set(e) for e in game.route_edges])) != target_m:
            continue
        L = stacked_L(game, N)
        if not 10 <= L.shape[0] <= 14:
            continue
        v_eq, _ = _row_minimiser(L)
        if v_eq < 0.05:
            continue
        return (s, t)
    return None


# ---------------------------------------------------------------- the sweep
def cell(env, K, kx, w, label):
    game = env.game
    L = stacked_L(game, N)
    R = L.shape[0]
    dis = disjoint_subset([set(e) for e in game.route_edges])
    m = len(dis)
    t0 = time.time()

    dec_c, counts_c = enum_windows(dis, w, R)
    lw_c = window_losses(counts_c, L, TAU)
    opt_core = restricted_opt(lw_c, dis, w)
    anti_core = antirepeat_stationary(lw_c, dec_c, dis, w)
    rot = best_rotation(dis, L, TAU, w)

    opt_full = anti_full = None
    if R ** w <= FULL_STATE_CAP:
        full = list(range(R))
        dec_f, counts_f = enum_windows(full, w, R)
        lw_f = window_losses(counts_f, L, TAU)
        opt_full = restricted_opt(lw_f, full, w)
        anti_full = antirepeat_stationary(lw_f, dec_f, full, w)

    v_eq, eq = _row_minimiser(L)
    iid_eq = static_stationary(eq, L, TAU, w)
    st_uni = static_stationary(uniform_dist(dis, R), L, TAU, w)
    st_inv = static_stationary(inv_vuln_dist(dis, L, R), L, TAU, w)
    st_full = static_stationary(uniform_dist(list(range(R)), R), L, TAU, w) \
        if R ** w <= IID_SUPPORT_CAP else None
    sd = min(float(L[r] @ softmax_br(np.eye(R)[r] * w, L, TAU)) for r in range(R))

    opt = opt_full if opt_full is not None else opt_core
    naive = {"rotation": rot, "anti_core": anti_core}
    if anti_full is not None:
        naive["anti_full"] = anti_full
    best_name = min(naive, key=naive.get)
    row = dict(od=label, m=m, K=K, k_extra=kx, R=R, w=w,
               n_isets=int(game.payoff.shape[1]),
               opt=opt, opt_core=opt_core, opt_full=opt_full,
               rotation=rot, anti_core=anti_core, anti_full=anti_full,
               iid_eq=iid_eq, static_uni_core=st_uni, static_inv_core=st_inv,
               static_uni_full=st_full, static_det=sd, v_eq_oneshot=v_eq,
               best_naive=naive[best_name], best_naive_name=best_name,
               naive_over_opt=naive[best_name] / max(opt, 1e-12),
               rot_over_opt=rot / max(opt, 1e-12),
               iid_over_opt=(iid_eq / max(opt, 1e-12)) if iid_eq is not None else None,
               core_vs_full_gap=(None if opt_full is None
                                 else opt_core - opt_full),
               secs=round(time.time() - t0, 1))
    return row


def main():
    t_start = time.time()
    print("screening Kaliningrad for m=3 and m=5 ODs ...", flush=True)
    ods = []
    od3 = screen_od("kaliningrad", 3)
    if od3 is None:
        od3 = screen_od("gdansk", 3)
        ods.append(("gdansk-" + "-".join(od3), od3, "gdansk"))
        print(f"  m=3: kaliningrad screen dry, gdansk fallback {od3}", flush=True)
    else:
        ods.append(("klg-" + "-".join(od3), od3, "kaliningrad"))
        print(f"  m=3: {od3}", flush=True)
    ods.append(("35-159", ("35", "159"), "kaliningrad"))
    od5 = screen_od("kaliningrad", 5)
    if od5 is not None:
        ods.append(("klg-" + "-".join(od5), od5, "kaliningrad"))
        print(f"  m=5: {od5}", flush=True)
    else:
        print("  m=5: screen dry, axis reported without an m=5 point", flush=True)
    ods.append(("71-33", ("71", "33"), "kaliningrad"))

    rows = []
    for label, od, city in ods:
        nodes_path, edges_path = CITY_PATHS[city]
        for kx in KXS:
            for K in KS:
                try:
                    env = make_multiconvoy_env(od, N=N, K=K, k_extra_routes=kx,
                                               menu_select=True, edge_vuln_band=BAND,
                                               nodes_path=nodes_path, edges_path=edges_path)
                except Exception as e:  # noqa: BLE001 - report and continue the grid
                    rows.append(dict(od=label, K=K, k_extra=kx, error=str(e)))
                    print(f"{label} kx={kx} K={K} BUILD ERROR {e}", flush=True)
                    continue
                if env.game.payoff.shape[1] > MAX_ISETS:
                    rows.append(dict(od=label, K=K, k_extra=kx,
                                     skipped=f"n_isets {env.game.payoff.shape[1]}"))
                    continue
                for w in WS:
                    row = cell(env, K, kx, w, label)
                    rows.append(row)
                    of = "-" if row["opt_full"] is None else f"{row['opt_full']:.4f}"
                    ii = "-" if row["iid_eq"] is None else f"{row['iid_eq']:.4f}"
                    print(f"{label} m={row['m']} kx={kx} K={K} R={row['R']} w={w} | "
                          f"opt {row['opt']:.4f} (core {row['opt_core']:.4f} full {of}) "
                          f"rot {row['rotation']:.4f} antiC {row['anti_core']:.4f} "
                          f"iid {ii} | rule/opt {row['naive_over_opt']:.2f} "
                          f"({row['secs']}s)", flush=True)

    out = dict(config=dict(N=N, band=BAND, tau=TAU, ws=WS, ks=KS, kxs=KXS,
                           full_cap=FULL_STATE_CAP, iid_cap=IID_SUPPORT_CAP,
                           ods=[(l, list(o), c) for l, o, c in ods]),
               rows=rows, total_secs=round(time.time() - t_start, 1))
    with open("models/runs/gen40_dyn_sensitivity.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote models/runs/gen40_dyn_sensitivity.json "
          f"({round(time.time() - t_start, 1)}s total)", flush=True)


if __name__ == "__main__":
    main()
