"""A2: graph-geometry transfer (EVAL-ONLY). Evaluate the frozen A1 generalist zero-shot on a
STRUCTURALLY DIFFERENT graph (default: the unsimplified kaliningrad_original export; any geojson
pair via --nodes/--edges, e.g. a real second city once network exists). Scored against each
held-out OD's own oracle equilibrium, vs a random-init reference.

Run: PYTHONPATH=. .venv/bin/python scratch/a2_graph_transfer.py <generalist_actor.pt> \
       [--nodes <nodes.geojson> --edges <edges.geojson> --tag original]
"""
from __future__ import annotations

import argparse
import json
import random

import networkx as nx
import numpy as np
import torch

from scripts.train_generalist import Instance, exact_ratio
from src.agents.sac import ProtagonistSAC
from src.baselines.interdiction_oracle import build_route_set
from src.envs.multiconvoy_interdiction import (
    _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS, make_multiconvoy_env)
from src.utils.graph_utils import load_osm_graph_and_demands

PROotloop = None


def make_prot(actor=None):
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu", role_alpha=True)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2))
    prot.actor.route_feats = None
    if actor:
        prot.actor.load_state_dict(torch.load(actor, map_location="cpu"))
    return prot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("actor")
    p.add_argument("--nodes", default="data/maps/kaliningrad_original/kaliningrad_nodes.geojson")
    p.add_argument("--edges", default="data/maps/kaliningrad_original/kaliningrad_edges.geojson")
    p.add_argument("--tag", default="original")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--pool-seed", type=int, default=3)
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    torch.set_num_threads(4)

    nodes, edges = load_osm_graph_and_demands(args.nodes, args.edges, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    deg3 = [n for n, d in G.degree() if d >= 3]
    rng = random.Random(args.pool_seed)
    insts, seen = [], set()
    while len(insts) < args.n and len(seen) < 6000:
        s, t = rng.sample(deg3, 2)
        key = tuple(sorted((s, t), key=repr))
        if key in seen:
            continue
        seen.add(key)
        try:
            if not 3 <= len(build_route_set(G, s, t, 0, "w")) <= 6:
                continue
            inst = Instance((s, t), 3, 1, (0.15, 0.95), 8, 0)  # uses default (30m) graph? -> override
        except Exception:
            continue
        # rebuild Instance on THIS graph via make_multiconvoy_env with the explicit paths
        try:
            env = make_multiconvoy_env(od=(s, t), N=3, K=1, k_extra_routes=8, menu_select=True,
                                       edge_vuln_band=(0.15, 0.95), interception_loss=10.0, seed=0,
                                       nodes_path=args.nodes, edges_path=args.edges)
        except Exception:
            continue
        if not 10 <= env.game.n_routes <= 14:
            continue
        from src.baselines.multiconvoy_oracle import solve_multiconvoy
        sol = solve_multiconvoy(env.game, 3, "mission")
        if sol.loss_mixed < 0.05:
            continue
        obj = type("I", (), {})()
        obj.env = env; obj.eq = float(sol.loss_mixed); obj.loss_det = float(sol.loss_det)
        obj.od = (s, t); obj.pol_hist = []
        insts.append(obj)

    gen, rnd = make_prot(args.actor), make_prot()
    print(f"=== A2 graph transfer -> {args.tag} ({len(insts)} held-out ODs) ===")
    rows = []
    for it in insts:
        gr, _ = exact_ratio(gen, it)
        rr, _ = exact_ratio(rnd, it)
        beats = gr * it.eq < it.loss_det
        rows.append({"od": f"{it.od[0]}-{it.od[1]}", "eq": it.eq, "loss_det": it.loss_det,
                     "gen_ratio": gr, "rand_ratio": rr, "beats_loss_det": bool(beats)})
        print(f"  {it.od[0]}-{it.od[1]}: gen {gr:.2f}x eq | rand {rr:.2f}x | "
              f"gen expl {gr*it.eq:.3f} vs loss_det {it.loss_det:.3f} "
              f"({'beats' if beats else 'MISSES'})")
    gm, rm = np.mean([r['gen_ratio'] for r in rows]), np.mean([r['rand_ratio'] for r in rows])
    allbeat = all(r['beats_loss_det'] for r in rows)
    print(f"\n  generalist mean {gm:.2f}x eq vs random-init {rm:.2f}x; beats loss_det on ALL: {allbeat}")
    print(f"  TRANSFER: {'YES' if (gm < rm and allbeat) else 'PARTIAL/NO'} "
          f"(strong if mean <= 2.0: {gm <= 2.0})")
    out = args.json_out or f"models/runs/a2_graph_transfer_{args.tag}.json"
    json.dump({"tag": args.tag, "rows": rows, "gen_mean": float(gm), "rand_mean": float(rm),
               "beats_all": allbeat}, open(out, "w"), indent=2)
    print(f"  [written] {out}")


if __name__ == "__main__":
    main()
