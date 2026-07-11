#!/usr/bin/env python3
"""A1: the GENERALIST interdiction policy (ZST step 1; DIRECTION_EXPANSION keystone).

Trains ONE fleet-route policy across SAMPLED instances (OD pairs with their own route menus and
threat maps) on the Kaliningrad graph, conditioned on the instance through (a) the
edge-vulnerability observation column (featurise edge col 4) and (b) per-route transferable
features [cost, worst-vulnerability] delivered undiluted at both heads with a SHARED learned
weight vector (`route_feat_w`, dedicated lr; the gen11b mechanism). NO per-route identity
capacity (`route_bias` is deliberately absent: identity does not transfer). Menus and features
ride ON each transition (state keys `menu_route_node_idx`/`menu_route_feats`), so replayed
instance-i transitions are always scored under instance i's menu.

Adversary: per-instance SMOOTH fictitious play (each instance keeps its own trailing occupancy
window; softmax best response recomputed and sampled fresh EVERY sortie - affordable at K=1).
NOT a fixed equilibrium mixture (see DIRECTION_EXPANSION addendum §3b self-correction: fixed
mixtures are exploitable-by-indifference).

Evaluation: exact fleet occupancy distribution per instance (one forward pass), exploitability
under each instance's own oracle BR, reported as the RATIO to that instance's equilibrium; the
pre-registered primary is the held-out (zero-shot) mean best-checkpoint TAP ratio.

Run: PYTHONPATH=. python scripts/train_generalist.py --sorties 12000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from scripts.train_multiconvoy import _transition, route_one
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.fp_dynamics import sample_smooth_iset, smooth_fp_probs
from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, objective_value, solve_multiconvoy)
from src.baselines.multiconvoy_planners import alns_fleet_planner
from src.envs.multiconvoy_interdiction import MultiConvoyInterdictionEnv, make_multiconvoy_env

TAP_K = 3

# City registry (gen16 multi-city): name -> (nodes_path, edges_path). 'kaliningrad' = the training
# graph every prior generation used; the others were extracted by scripts/extract_city.py (same
# arterial-filter + 30m-consolidation pipeline) and length-repaired by scratch/repair_map_lengths.py.
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES  # noqa: E402

CITY_PATHS = {
    "kaliningrad": (_DEFAULT_NODES, _DEFAULT_EDGES),
    "gdansk": ("data/maps/gdansk/nodes.geojson", "data/maps/gdansk/edges.geojson"),
    "east_london": ("data/maps/east_london/nodes.geojson", "data/maps/east_london/edges.geojson"),
    "istanbul": ("data/maps/istanbul/nodes.geojson", "data/maps/istanbul/edges.geojson"),
}


class Instance:
    def __init__(self, od: tuple[str, str], N: int, K: int, band, k_extra: int, seed: int,
                 city: str = "kaliningrad"):
        self.od = od
        self.city = city
        nodes_path, edges_path = CITY_PATHS[city]
        self.env: MultiConvoyInterdictionEnv = make_multiconvoy_env(
            od=od, N=N, K=K, k_extra_routes=k_extra, menu_select=True,
            edge_vuln_band=band, interception_loss=10.0, seed=seed,
            nodes_path=nodes_path, edges_path=edges_path)
        sol = solve_multiconvoy(self.env.game, N, "mission")
        self.eq = float(sol.loss_mixed)
        self.loss_det = float(sol.loss_det)
        self.occ_seq: list[int] = []      # this instance's own trailing play window (smooth FP)
        self.pol_hist: list[np.ndarray] = []

    @property
    def R(self):
        return self.env.game.n_routes


def sample_instances(n_total: int, N: int, K: int, band, k_extra: int, seed: int,
                     r_range=(10, 14), city: str = "kaliningrad") -> list[Instance]:
    """High-connectivity OD pool (the F3/screen recipe), filtered to comparable menu sizes;
    sampled within the city's largest connected component."""
    from src.utils.graph_utils import load_osm_graph_and_demands
    from src.envs.multiconvoy_interdiction import _DEFAULT_TASKS
    from src.baselines.interdiction_oracle import build_route_set
    nodes_path, edges_path = CITY_PATHS[city]
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    deg3 = [n for n, d in G.degree() if d >= 3]
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n_total and len(seen) < 6000:
        s, t = rng.sample(deg3, 2)
        key = tuple(sorted((s, t), key=repr))
        if key in seen:
            continue
        seen.add(key)
        try:
            base = build_route_set(G, s, t, 0, "w")
            if not 3 <= len(base) <= 6:
                continue
            inst = Instance((s, t), N, K, band, k_extra, seed, city=city)
            if not r_range[0] <= inst.R <= r_range[1] or inst.eq < 0.05:
                continue
            out.append(inst)
        except Exception:
            continue
    return out


def exact_ratio(prot: ProtagonistSAC, inst: Instance) -> tuple[float, np.ndarray]:
    """Exact fleet occupancy distribution of the CURRENT policy on this instance -> (ratio, dist)."""
    env = inst.env
    env.reset()
    obs = env.observe()
    # point the nets at THIS instance (eval path bypasses select_action's per-state assignment)
    for net in (prot.actor,):
        net.menu_routes = obs["menu_route_node_idx"]
        if hasattr(net, "route_feat_w"):
            net.route_feats = obs["menu_route_feats"]
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    R = inst.env.game.n_routes   # robust to ad-hoc instance objects (A2/D3) that lack .R
    prot.actor.eval()
    with torch.no_grad():
        lead, _ = prot.actor(pyg, n2i[obs["trucks"][0]["current_node"]],
                             list(range(R)), torch.zeros(R))
    prot.actor.train()
    lead = lead.numpy()
    d = np.zeros(len(env.occupancies))
    for r in range(R):
        d[env._occ_index[tuple(env.config.N if i == r else 0 for i in range(R))]] = lead[r]
    _, expl = best_response_attacker_multi(env.obj_matrix, d)
    return float(expl) / inst.eq, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=16)
    p.add_argument("--n-test", type=int, default=6)
    p.add_argument("--cities", default="",
                   help="gen16 multi-city mode: comma list of TRAIN cities (from CITY_PATHS); "
                        "with --holdout-city, train instances are sampled per train city and ALL "
                        "test instances come from the held-out city. Empty = single-city (gen15).")
    p.add_argument("--holdout-city", default="",
                   help="the city held out ENTIRELY for zero-shot evaluation (gen16)")
    p.add_argument("--n-per-city", type=int, default=6,
                   help="train instances sampled per train city (gen16)")
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--k-extra", type=int, default=8)
    p.add_argument("--band", default="0.15,0.95")
    p.add_argument("--sorties", type=int, default=12000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-seed", type=int, default=0,
                   help="instance-pool RNG (FIXED across seeds so every seed sees the same split)")
    p.add_argument("--fp-tau", type=float, default=0.05)
    p.add_argument("--smooth-window", type=int, default=250)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--leader-ent-frac", type=float, default=0.5)
    p.add_argument("--follower-ent-frac", type=float, default=0.05)
    p.add_argument("--leader-alpha-floor", type=float, default=0.20)
    p.add_argument("--follower-warmup", type=int, default=250)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--stack-dup", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json-out", default="")
    p.add_argument("--ckpt-dir", default="")
    p.add_argument("--vanilla", action="store_true",
                   help="Obj-5 transfer control: travel objective, NO adversary (reward = -normalised "
                        "route cost); still map-conditioned, still evaluated zero-shot under the oracle BR")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    band = tuple(float(x) for x in args.band.split(","))

    if args.cities and args.holdout_city:
        train_cities = [c.strip() for c in args.cities.split(",") if c.strip()]
        assert args.holdout_city not in train_cities, "held-out city must not be trained on"
        print(f"[A1-multicity] TRAIN cities {train_cities} x {args.n_per_city} instances; "
              f"HOLD-OUT city {args.holdout_city} x {args.n_test} (pool-seed {args.pool_seed})",
              flush=True)
        train = []
        for c in train_cities:
            got = sample_instances(args.n_per_city, args.N, args.K, band, args.k_extra,
                                   args.pool_seed, city=c)
            if len(got) < args.n_per_city:
                raise RuntimeError(f"{c}: only {len(got)} instances")
            train += got
        test = sample_instances(args.n_test, args.N, args.K, band, args.k_extra,
                                args.pool_seed, city=args.holdout_city)
        if len(test) < args.n_test:
            raise RuntimeError(f"{args.holdout_city}: only {len(test)} test instances")
    else:
        print(f"[A1] sampling {args.n_train}+{args.n_test} instances (pool-seed {args.pool_seed})...", flush=True)
        pool = sample_instances(args.n_train + args.n_test, args.N, args.K, band, args.k_extra,
                                args.pool_seed)
        if len(pool) < args.n_train + args.n_test:
            raise RuntimeError(f"only {len(pool)} instances sampled")
        test, train = pool[:args.n_test], pool[args.n_test:]
    print(f"[A1] TRAIN: {[(i.city, i.od) for i in train]}")
    print(f"[A1] TEST (zero-shot): {[(i.city, i.od, round(i.eq, 3)) for i in test]}", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3,
                          alpha_floor=args.leader_alpha_floor)
    # transferable head terms ONLY (no route_bias); registration order matched across q/target nets
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(2))
        net.route_feats = None  # assigned per state/sample
    prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w]})
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w],
                                          "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.follow_w, prot.q2.follow_w]})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.route_feat_w, prot.q2.route_feat_w],
                                           "lr": args.head_term_lr})

    hist = []
    t0 = time.time()
    for k in range(args.sorties):
        inst = train[int(rng.integers(len(train)))]
        env = inst.env
        if args.vanilla:
            # Obj-5 transfer control: NO adversary; reward = -normalised fleet travel cost.
            env.reset()
            steps, occ, _ = route_one(prot, env, fleet_route=True)
            mean_cost = float(env.game.travel_cost.mean())
            fleet_cost = float(sum(occ[r] * env.game.travel_cost[r] for r in range(env.game.n_routes)))
            reward = -args.interception_loss * (fleet_cost / (env.config.N * mean_cost))
        else:
            # per-instance smooth FP: softmax BR to THIS instance's trailing play, sampled fresh
            probs = smooth_fp_probs(inst.occ_seq, len(env.occupancies), env.obj_matrix,
                                    args.fp_tau, args.smooth_window)
            j = sample_smooth_iset(probs, rng)
            env.reset()
            env.commit(j)
            steps, occ, _ = route_one(prot, env, fleet_route=True)
            inst.occ_seq.append(env._occ_index[tuple(occ)])
            pay = env.game.payoff[:, j]
            reward = -args.interception_loss * objective_value(
                np.asarray(occ), pay, env.config.N, env.config.objective, env.config.threshold_m)
        leader_te = args.leader_ent_frac * math.log(inst.R)
        follower_te = args.follower_ent_frac * math.log(inst.R)
        for obs_j, ci_j, _, _ in steps:
            is_follower = (ci_j != 0) and k >= args.follower_warmup
            obs_j["target_entropy"] = follower_te if is_follower else leader_te
            obs_j["alpha_group"] = 1 if is_follower else 0
        is_stack = sum(1 for c in occ if c > 0) == 1
        n_push = args.stack_dup if is_stack else 1
        N = env.config.N
        for i, (obs, ci, hop, mask) in enumerate(steps):
            last = i == N - 1
            nobs, nci, nmask = ((steps[i + 1][0], steps[i + 1][1], steps[i + 1][3])
                                if not last else (None, None, None))
            t = _transition(obs, ci, hop, mask, reward if last else 0.0, nobs, nci, nmask, last)
            for _ in range(n_push):
                prot.replay_buffer.push(t)
        prot.update(args.batch_size)

        if (k + 1) % args.eval_every == 0:
            for inst_set in (train, test):
                for it in inst_set:
                    ratio, d = exact_ratio(prot, it)
                    it.pol_hist.append(d)
            def mean_tap_ratio(insts):
                vals = []
                for it in insts:
                    tap = np.mean(it.pol_hist[-TAP_K:], axis=0)
                    _, expl = best_response_attacker_multi(it.env.obj_matrix, tap)
                    vals.append(float(expl) / it.eq)
                return float(np.mean(vals)), vals
            tr_m, _ = mean_tap_ratio(train)
            te_m, te_v = mean_tap_ratio(test)
            fw = tuple(float(x) for x in prot.actor.route_feat_w.detach())
            hist.append((k + 1, tr_m, te_m, te_v, fw,
                         float(prot.alpha), float(prot.alpha_foll)))
            if args.ckpt_dir:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(),
                           str(Path(args.ckpt_dir) / f"actor_ep{k + 1}.pt"))
            print(f"  sortie {k+1:6d}: TRAIN mean TAP ratio {tr_m:.2f} | TEST (zero-shot) "
                  f"{te_m:.2f} {['%.2f' % v for v in te_v]} | rw[{fw[0]:.2f},{fw[1]:.2f}] "
                  f"alpha L{prot.alpha:.2f}/F{prot.alpha_foll:.2f} | {time.time()-t0:5.0f}s",
                  flush=True)

    best_test = min((h[2] for h in hist), default=float("nan"))
    best_at = next((h[0] for h in hist if h[2] == best_test), None)
    alns_rows = {f"{it.od[0]}-{it.od[1]}": {"eq": it.eq, "loss_det": it.loss_det}
                 for it in test}
    print(f"\n=== A1 GENERALIST (seed {args.seed}) ===")
    print(f"  BEST held-out mean TAP ratio {best_test:.3f} @ sortie {best_at} "
          f"(final {hist[-1][2]:.3f}); train final {hist[-1][1]:.3f}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "train_ods": [(i.city, i.od) for i in train],
             "test_ods": [(i.city, i.od) for i in test],
             "holdout_city": args.holdout_city or None, "test_refs": alns_rows,
             "history": hist, "best_test_ratio": best_test, "best_at": best_at}, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
