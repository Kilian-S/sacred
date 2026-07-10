"""A3: the amortisation benchmark (the honest successor to the retired wall-clock claim; EVAL-ONLY).

Streams M fresh sampled instances and, per instance, measures the cost of getting a deployable
mixed routing from (a) the exact oracle LP re-solve vs (b) one forward pass of the A1 generalist
(loaded from a saved actor). Reports the QUALITY-ADJUSTED frontier: cumulative wall-clock AND
solution quality (exploitability / equilibrium ratio), plus the generalist's amortised training
cost, so the honest trade is visible (LP = slow + exact; policy = fast + ~ratio). The wall-clock
scaling claim is dead (gen09 ledger); THIS is the defensible deployment claim.

Run: PYTHONPATH=. .venv/bin/python scratch/amortisation_benchmark.py <generalist_actor.pt> [--train-cost-s S]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from scripts.train_generalist import Instance, exact_ratio, sample_instances
from src.agents.sac import ProtagonistSAC
from src.baselines.multiconvoy_oracle import solve_multiconvoy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("actor")
    p.add_argument("--m", type=int, default=40, help="stream length (fresh held-out instances)")
    p.add_argument("--pool-seed", type=int, default=7, help="DIFFERENT from training pool-seed 0")
    p.add_argument("--train-cost-s", type=float, default=None,
                   help="measured generalist training wall (s); read from the run log if omitted")
    p.add_argument("--json-out", default="models/runs/a3_amortisation.json")
    args = p.parse_args()
    torch.set_num_threads(4)

    insts = sample_instances(args.m, N=3, K=1, band=(0.15, 0.95), k_extra=8, seed=args.pool_seed)
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu", role_alpha=True)
    for net in (prot.actor,):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(2))
        net.route_feats = None
    prot.actor.load_state_dict(torch.load(args.actor, map_location="cpu"))

    lp_cum, pol_cum, ratios, lp_t, pol_t = 0.0, 0.0, [], [], []
    for it in insts:
        t = time.time(); _ = solve_multiconvoy(it.env.game, 3, "mission"); lp_t.append(time.time() - t)
        t = time.time(); ratio, _ = exact_ratio(prot, it); pol_t.append(time.time() - t)
        lp_cum += lp_t[-1]; pol_cum += pol_t[-1]; ratios.append(ratio)
    tc = args.train_cost_s
    print(f"=== A3 AMORTISATION ({args.m} fresh held-out instances, pool-seed {args.pool_seed}) ===")
    print(f"  per-instance: LP re-solve {np.mean(lp_t)*1000:.1f} ms | policy forward "
          f"{np.mean(pol_t)*1000:.1f} ms  ({np.mean(lp_t)/max(np.mean(pol_t),1e-9):.0f}x)")
    print(f"  cumulative over {args.m}: LP {lp_cum:.2f} s (exact, ratio 1.00) | "
          f"policy {pol_cum:.2f} s (ratio {np.mean(ratios):.2f} +/- {np.std(ratios):.2f})")
    if tc:
        cross = tc / max(np.mean(lp_t) - np.mean(pol_t), 1e-9)
        print(f"  generalist amortised training {tc:.0f} s -> crossover at M ~ {cross:.0f} instances "
              f"(beyond which the trained policy's total compute < re-solving each; QUALITY-ADJUSTED: "
              f"the policy is {np.mean(ratios):.2f}x eq, the LP is exact)")
    print(f"  HONEST FRAME: the LP is faster AND exact per instance here; the policy's case is NOT "
          f"speed but (i) it never re-solves (ZST) and (ii) it can price a TRAINED policy inside a "
          f"design loop where the LP cannot participate (D3).")
    json.dump({"m": args.m, "lp_ms": np.mean(lp_t)*1000, "pol_ms": np.mean(pol_t)*1000,
               "ratio_mean": float(np.mean(ratios)), "ratio_std": float(np.std(ratios)),
               "lp_cum_s": lp_cum, "pol_cum_s": pol_cum, "train_cost_s": tc},
              open(args.json_out, "w"), indent=2)
    print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
