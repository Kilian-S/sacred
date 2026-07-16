#!/usr/bin/env python3
"""R0b (EVAL-ONLY): the structure-discovery row.

How much probability mass do the banked policies place on the DISJOINT CORE (the max-flow
routes) versus the padded k-shortest duplicates? This grounds the surviving positive sentence:
self-play discovers the independent-route structure without being told it (no max-flow call, no
labels, no solver) — the heuristic cannot cheapen this claim because it is TOLD the structure.

Also computes (R4a rider) the gen14 headline policy's fleet cost on 35-159, completing the
cost ladder beside the R0a heuristic rows.

Rows: gen14 MC (35-159, n=3 ckpt seeds) at the best checkpoint; gen16 generalist (seed 0,
best-ckpt window) zero-shot on the 6 held-out Gdansk ODs. The single-convoy row is already
banked (gen08 B2-P3: policy mass on shared duplicates 0.01/0.00/0.19 = mass 0.81-1.00 on the
disjoint six), cited, not recomputed.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import torch

from scripts.train_generalist import exact_ratio, sample_instances
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

TAP_K = 5
OUT = {}


def disjoint_subset(route_edges):
    kept, used = [], set()
    for i, re_ in enumerate(route_edges):
        if not (re_ & used):
            kept.append(i)
            used |= re_
    return kept


def gen14_rows():
    env = make_multiconvoy_env(("35", "159"), N=3, K=1, k_extra_routes=8, menu_select=True,
                               edge_vuln_band=(0.15, 0.95), interception_loss=10.0, seed=0)
    R, N = env.game.n_routes, env.config.N
    cost = np.asarray(env.game.travel_cost)
    dis = disjoint_subset(env.game.route_edges)
    env.reset(); obs = env.observe()
    menu = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]

    def lead_of(state):
        prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=4, hidden_dim=64, num_layers=2,
                              heads=4, device="cpu")
        prot.actor.menu_routes = menu
        if any(k == "follow_w" for k in state):
            prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        prot.actor.load_state_dict(state)
        pyg = featurize_state(obs, 0).to(prot.device)
        pyg.x = _clip_x(pyg.x, prot.node_in_dim)
        pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
        active = node_index_map(obs)[obs["trucks"][0]["current_node"]]
        prot.actor.eval()
        with torch.no_grad():
            p, _ = prot.actor(pyg, active, list(range(R)), torch.zeros(R))
        return p.numpy()

    rows = []
    for seed_dir in sorted(glob.glob("models/runs/gen14_evidence/mc_seed*_ckpts")):
        cks = sorted(glob.glob(f"{seed_dir}/actor_ep*.pt"),
                     key=lambda p: int(re.search(r"ep(\d+)", p).group(1)))
        if not cks:
            continue
        pol, best = [], None
        for cp in cks:
            lead = lead_of(torch.load(cp, map_location="cpu"))
            occ = np.zeros(len(env.occupancies))
            for r in range(R):
                occ[env._occ_index[tuple(N if i == r else 0 for i in range(R))]] = lead[r]
            pol.append((occ, lead))
            tap_occ = np.mean([o for o, _ in pol[-TAP_K:]], axis=0)
            tap_lead = np.mean([l for _, l in pol[-TAP_K:]], axis=0)
            expl = env.exploitability_of_occupancy_dist(tap_occ)
            if best is None or expl < best[0]:
                best = (expl, tap_lead)
        dmass = float(sum(best[1][r] for r in dis))
        fcost = float(N * (best[1] @ cost))
        rows.append({"seed_dir": seed_dir, "best_tap": round(float(best[0]), 3),
                     "disjoint_mass": round(dmass, 3), "fleet_cost": round(fcost, 1)})
        print(f"  gen14 {Path(seed_dir).name}: best TAP {best[0]:.3f} | "
              f"disjoint-core mass {dmass:.3f} | fleet cost {fcost:.1f}")
    OUT["gen14_35_159"] = rows


def gen16_rows():
    ck_dir = "models/runs/gen16_multicity/seed0_ckpts"
    cks = {int(re.search(r"ep(\d+)", c).group(1)): c
           for c in glob.glob(f"{ck_dir}/actor_ep*.pt")}
    eps = sorted(cks)
    best_at = 1000  # gen16 seed-0 selected best (ledger); window = best-1, best, best+1
    ci = eps.index(best_at) if best_at in eps else len(eps) - 1
    window = eps[max(0, ci - 1):ci + 2]
    states = [torch.load(cks[e], map_location="cpu") for e in window]

    def mkprot(state):
        prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2,
                              heads=4, device="cpu", role_alpha=True)
        prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2))
        prot.actor.route_feats = None
        prot.actor.load_state_dict(state)
        return prot

    rows = []
    for it in sample_instances(6, 3, 1, (0.15, 0.95), 8, 0, city="gdansk"):
        dis = disjoint_subset(it.env.game.route_edges)
        stacked_idx = {r: it.env._occ_index[tuple(3 if i == r else 0
                                                  for i in range(it.env.game.n_routes))]
                       for r in range(it.env.game.n_routes)}
        ds = [exact_ratio(mkprot(st), it)[1] for st in states]
        tap = np.mean(ds, axis=0)
        dmass = float(sum(tap[stacked_idx[r]] for r in dis))
        rows.append({"od": f"{it.od[0]}-{it.od[1]}", "disjoint_mass": round(dmass, 3),
                     "m": len(dis), "R": it.env.game.n_routes})
        print(f"  gen16 zero-shot gdansk {it.od}: disjoint-core mass {dmass:.3f} "
              f"(m={len(dis)}/{it.env.game.n_routes} routes)")
    OUT["gen16_gdansk_zeroshot"] = rows


if __name__ == "__main__":
    torch.set_num_threads(2)
    print("=== R0b: policy mass on the disjoint core (structure discovered, not told) ===")
    gen14_rows()
    gen16_rows()
    json.dump(OUT, open("models/runs/r0b_structure_discovery.json", "w"), indent=2)
    print("[written] models/runs/r0b_structure_discovery.json")
