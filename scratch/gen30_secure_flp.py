#!/usr/bin/env python3
"""gen30_secure_flp sweep (ORACLE-ONLY; pre-registered in experiments/gen30_secure_flp.md
BEFORE this file existed). Components A (single-depot cost/security frontier) and B
(two-depot redundancy priced by the correlation gap), per city.

Game = the committed b4/gen29 mechanics verbatim: F=3 streams (one convoy each), K=1 hidden
edge interdictor over the union candidate list, band (0.15, 0.95) soft interception, k8 menus,
mission objective P(>=1 of 3 lost). Security value of a design = exact joint equilibrium
v_joint, with the complete hostile family beside it on every row (v_det, best independent
product, deconflict-uniform, the in-sample m<=4-pairing cap).

Usage:
  PYTHONPATH=. .venv/bin/python scratch/gen30_secure_flp.py --city kaliningrad
  PYTHONPATH=. .venv/bin/python scratch/gen30_secure_flp.py --city kaliningrad --targets random
  PYTHONPATH=. .venv/bin/python scratch/gen30_secure_flp.py --city gdansk
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time

import networkx as nx
import numpy as np

from scratch.b4_joint_napkin_probe import best_m_pairings
from scratch.b4_multiod_probe import build_graph as build_kaliningrad
from scratch.b4_widen_probe import (
    best_product_general, deconflict_uniform_general, joint_payoff, survival_mats)
from src.baselines.interdiction_oracle import build_route_set, edges_of_route
from src.baselines.multiconvoy_oracle import _row_minimiser

KX = 8
PRIMARY_TARGETS = ("212", "188", "195")   # gen29's screened headline targets (disclosed)
ANCHOR_SOURCE = "147"                     # the committed gen29 headline source
MAX_ROWS = 40_000                         # joint-play cap (size guard, recorded)
MAX_ENTRIES = 25_000_000


def build_city(city: str) -> nx.Graph:
    if city == "kaliningrad":
        return build_kaliningrad()
    # mirrors src/utils/graph_utils.load_osm_graph_and_demands edge conventions exactly
    # (w = max(1, round(length_m/100, 1))); reimplemented because the shared loader needs a
    # demand-tasks file this act has no use for (disclosed in the ledger).
    paths = {"gdansk": ("data/maps/gdansk/nodes.geojson", "data/maps/gdansk/edges.geojson")}
    nodes_path, edges_path = paths[city]
    ej = json.load(open(edges_path))
    G = nx.Graph()
    for f in ej["features"]:
        p = f["properties"]
        val = p.get("length")
        w = max(1.0, round((float(val) if val is not None else 100.0) / 100.0, 1))
        G.add_edge(str(p["u"]), str(p["v"]), w=w)
    nj = json.load(open(nodes_path))
    for f in nj["features"]:
        nid = str(f["properties"]["osmid"])
        if nid in G:
            G.nodes[nid]["lon"], G.nodes[nid]["lat"] = f["geometry"]["coordinates"]
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def route_len(G: nx.Graph, route) -> float:
    return float(sum(G[route[i]][route[i + 1]]["w"] for i in range(len(route) - 1)))


def family_rows(G, rsets):
    """The complete hostile family for one design (route sets per stream). Returns dict + the
    equilibrium joint distribution (for the operating-cost column)."""
    S_list, cand = survival_mats(G, rsets)
    E = len(cand)
    isets = [(e,) for e in range(E)]
    M, shape = joint_payoff(S_list, isets)
    if M.shape[0] > MAX_ROWS or M.shape[0] * M.shape[1] > MAX_ENTRIES:
        return None, None, None
    v_joint, x = _row_minimiser(M)
    v_det = float(M.max(axis=1).min())
    v_ind = best_product_general(M, shape, restarts=3, iters=40)
    v_dec, _, _ = deconflict_uniform_general(M, rsets)
    caps = best_m_pairings(M)
    v_cap = float(min(caps.values()))
    rows = {"E": E, "R": [len(r) for r in rsets], "v_joint": float(v_joint),
            "v_det": v_det, "v_indep": float(v_ind), "v_deconflict": float(v_dec),
            "v_cap": v_cap, "caps": {str(k): float(v) for k, v in caps.items()},
            "gap_vs_cap": float((v_cap - v_joint) / max(v_joint, 1e-9))}
    return rows, x, shape


def eq_expected_len(G, rsets, x, shape):
    """Expected total flown route length under the equilibrium joint mixture (per-stream
    marginals of the joint distribution)."""
    xt = np.asarray(x).reshape(shape)
    total = 0.0
    for f, rs in enumerate(rsets):
        marg = xt.sum(axis=tuple(a for a in range(len(shape)) if a != f))
        total += float(sum(m * route_len(G, r) for m, r in zip(marg, rs)))
    return total


def valid_route_sets(G, s, targets):
    if s in targets:
        return None
    try:
        rsets = [build_route_set(G, s, t, KX, "w") for t in targets]
    except Exception:
        return None
    if any(not (4 <= len(r) <= 14) for r in rsets):
        return None
    return rsets


def stream_features(G, s, targets, rsets):
    """Cheap pre-solve features for the Component C surrogate (no LP)."""
    costs = [nx.shortest_path_length(G, s, t, weight="w") for t in targets]
    ncut = [len(list(nx.edge_disjoint_paths(G, s, t))) for t in targets]
    e_sets = [set().union(*(edges_of_route(r) for r in rs)) for rs in rsets]
    union_e = set().union(*e_sets)
    from src.baselines.interdiction_oracle import length_band_vulnerability
    vuln = length_band_vulnerability(G, union_e, band=(0.15, 0.95), weight="w",
                                     norm_edges=list(G.edges()))
    vv = list(vuln.values())
    return {"cost_total": float(sum(costs)), "cost_max": float(max(costs)),
            "cost_min": float(min(costs)), "mincut_min": int(min(ncut)),
            "mincut_sum": int(sum(ncut)), "routes_sum": int(sum(len(r) for r in rsets)),
            "E_union": len(union_e), "vuln_mean": float(np.mean(vv)),
            "vuln_max": float(np.max(vv))}


def component_a(G, targets, site_cap, seed, tag):
    deg3 = sorted(n for n, d in G.degree() if d >= 3)
    rng = random.Random(seed)
    rng.shuffle(deg3)
    rows, t0 = [], time.time()
    for s in deg3:
        rsets = valid_route_sets(G, s, targets)
        if rsets is None:
            continue
        fam, x, shape = family_rows(G, rsets)
        if fam is None:
            continue
        cost = float(sum(nx.shortest_path_length(G, s, t, weight="w") for t in targets))
        fam.update({"site": s, "cost": cost,
                    "feat": stream_features(G, s, targets, rsets)})
        rows.append(fam)
        if len(rows) % 20 == 0:
            print(f"  [{tag}] {len(rows)} sites, {time.time()-t0:.0f}s", flush=True)
        if len(rows) >= site_cap:
            break
    print(f"  [{tag}] DONE: {len(rows)} sites in {time.time()-t0:.0f}s", flush=True)
    return rows


def pareto_front(rows, xk="cost", yk="v_joint"):
    pts = sorted(((r[xk], r[yk], r["site"]) for r in rows))
    front, best_y = [], np.inf
    for x, y, s in pts:
        if y < best_y - 1e-12:
            front.append(s)
            best_y = y
    return front


def component_b(G, targets, sites, n_pairs, seed, tag):
    """Stratified-by-distance, payoff-blind depot-pair sampling; classical vs redundant."""
    rng = random.Random(seed + 1)
    cand_pairs = []
    pool = list(sites)
    for _ in range(min(600, len(pool) * (len(pool) - 1) // 2)):
        d1, d2 = rng.sample(pool, 2)
        try:
            dist = nx.shortest_path_length(G, d1, d2, weight="w")
        except Exception:
            continue
        cand_pairs.append((d1, d2, float(dist)))
    cand_pairs = sorted(set(cand_pairs), key=lambda p: p[2])
    n = len(cand_pairs)
    bounds = [round(i * n / 5) for i in range(6)]
    picked = []
    per_bucket = max(1, n_pairs // 5)
    for i in range(5):
        q = cand_pairs[bounds[i]:bounds[i + 1]]
        rng.shuffle(q)
        picked.extend(q[:per_bucket])
    rows, t0 = [], time.time()
    for d1, d2, dist in picked:
        rs1 = valid_route_sets(G, d1, targets)
        rs2 = valid_route_sets(G, d2, targets)
        if rs1 is None or rs2 is None:
            continue
        c1 = [nx.shortest_path_length(G, d1, t, weight="w") for t in targets]
        c2 = [nx.shortest_path_length(G, d2, t, weight="w") for t in targets]
        assign = [0 if a <= b else 1 for a, b in zip(c1, c2)]
        cls_sets = [rs1[f] if assign[f] == 0 else rs2[f] for f in range(len(targets))]
        red_sets = [rs1[f] + rs2[f] for f in range(len(targets))]
        cls_cost = float(sum(min(a, b) for a, b in zip(c1, c2)))
        fam_c, x_c, sh_c = family_rows(G, cls_sets)
        fam_r, x_r, sh_r = family_rows(G, red_sets)
        if fam_c is None or fam_r is None:
            continue
        e1 = set().union(*(edges_of_route(r) for rs in rs1 for r in rs))
        e2 = set().union(*(edges_of_route(r) for rs in rs2 for r in rs))
        jac = len(e1 & e2) / max(1, len(e1 | e2))
        row = {"d1": d1, "d2": d2, "dist": dist, "jaccard": float(jac),
               "assign": assign, "cls_cost": cls_cost,
               "cls": fam_c, "red": fam_r,
               "cls_op_len": eq_expected_len(G, cls_sets, x_c, sh_c),
               "red_op_len": eq_expected_len(G, red_sets, x_r, sh_r),
               "cls_min_len": float(sum(min(route_len(G, r) for r in rs)
                                        for rs in cls_sets)),
               "gain_rel": float((fam_c["v_joint"] - fam_r["v_joint"])
                                 / max(fam_c["v_joint"], 1e-9)),
               "gain_vs_cap_rel": float((fam_c["v_joint"] - fam_r["v_cap"])
                                        / max(fam_c["v_joint"], 1e-9)),
               "feat": stream_features(G, d1, targets, red_sets)}
        row["feat"]["dist_pair"] = dist
        row["feat"]["jaccard_pair"] = float(jac)
        rows.append(row)
        print(f"  [{tag}] pair {d1}-{d2} dist {dist:.0f} jac {jac:.2f}: "
              f"cls {fam_c['v_joint']:.3f} red {fam_r['v_joint']:.3f} "
              f"(gain {100*row['gain_rel']:.0f}%) red-cap {fam_r['v_cap']:.3f} "
              f"(vs-cls-eq {100*row['gain_vs_cap_rel']:.0f}%)", flush=True)
    print(f"  [{tag}] DONE: {len(rows)} pairs in {time.time()-t0:.0f}s", flush=True)
    return rows


def pick_random_targets(G, seed, min_sites=30, tries=25):
    """Unscreened seeded target draw: first triple of deg>=3 nodes admitting enough valid
    sites (validity only, never payoff: the payoff-blind rule)."""
    deg3 = sorted(n for n, d in G.degree() if d >= 3)
    rng = random.Random(seed)
    for _ in range(tries):
        targets = tuple(rng.sample(deg3, 3))
        n_ok = 0
        for s in rng.sample(deg3, min(120, len(deg3))):
            if valid_route_sets(G, s, targets) is not None:
                n_ok += 1
            if n_ok >= min_sites:
                return targets
    raise RuntimeError("no valid unscreened target triple found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="kaliningrad")
    ap.add_argument("--targets", default="primary", choices=["primary", "random"])
    ap.add_argument("--site-cap", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=30)
    args = ap.parse_args()

    G = build_city(args.city)
    print(f"[gen30] {args.city}: {G.number_of_nodes()} nodes {G.number_of_edges()} edges",
          flush=True)
    if args.city == "kaliningrad" and args.targets == "primary":
        targets, tag = PRIMARY_TARGETS, "kal_primary"
        site_cap = args.site_cap or 150
        n_pairs = args.n_pairs or 40
    elif args.city == "kaliningrad":
        targets, tag = pick_random_targets(G, args.seed), f"kal_random_s{args.seed}"
        site_cap = args.site_cap or 80
        n_pairs = args.n_pairs or 0          # robustness row = frontier only by default
    else:
        targets, tag = pick_random_targets(G, args.seed), f"{args.city}_s{args.seed}"
        site_cap = args.site_cap or 60
        n_pairs = args.n_pairs or 30
    print(f"[gen30] targets = {targets} ({args.targets}); site cap {site_cap}, "
          f"pairs {n_pairs}", flush=True)

    a_rows = component_a(G, targets, site_cap, args.seed, tag)
    front = pareto_front(a_rows)
    sites = [r["site"] for r in a_rows]
    b_rows = component_b(G, targets, sites, n_pairs, args.seed, tag) if n_pairs else []

    out = {"city": args.city, "targets": list(targets), "target_mode": args.targets,
           "seed": args.seed, "anchor_source": ANCHOR_SOURCE,
           "component_a": a_rows, "pareto_sites": front, "component_b": b_rows}
    path = f"models/runs/gen30_secure_flp_{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"[written] {path}", flush=True)


if __name__ == "__main__":
    main()
