#!/usr/bin/env python3
"""gen24 (A1): the LP-distillation control for the ZST act.

Supervised amortisation of the solver: the gen16 generalist ARCHITECTURE (menu-select actor,
transferable route_feat_w head terms, edge-vulnerability observation), trained on the gen16
multi-city instance pool with a full-batch KL loss to each training instance's STACKED-MINIMAX
route mixture (the optimum of the fleet-route policy class, `_row_minimiser` over the stacked
occupancy rows). NO adversary, NO reward, NO replay buffer. Evaluated zero-shot on the held-out
city under the identical exact ratio-to-equilibrium metric as gen16.

Pre-registration: experiments/gen24_distill.md (binding).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_generalist import Instance, exact_ratio, sample_instances, TAP_K
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.multiconvoy_oracle import _row_minimiser, best_response_attacker_multi


def stacked_minimax(inst: Instance) -> tuple[np.ndarray, float]:
    """Minimax route mixture over STACKED occupancies (all N on one route) -> (mixture[R], value)."""
    env = inst.env
    R = env.game.n_routes
    N = env.config.N
    rows = [env._occ_index[tuple(N if i == r else 0 for i in range(R))] for r in range(R)]
    M = env.obj_matrix[np.asarray(rows), :]
    v, x = _row_minimiser(M)
    return x, v


def prep(inst: Instance, prot: ProtagonistSAC):
    """Precompute the static per-instance forward inputs (the observation never changes in
    fleet-route mode: the leader decides at reset state)."""
    env = inst.env
    env.reset()
    obs = env.observe()
    pyg = featurize_state(obs, 0)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    return {
        "pyg": pyg,
        "active": n2i[obs["trucks"][0]["current_node"]],
        "menu": obs["menu_route_node_idx"],
        "feats": obs["menu_route_feats"],
        "R": env.game.n_routes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", default="kaliningrad,east_london,istanbul")
    p.add_argument("--holdout-city", default="gdansk")
    p.add_argument("--n-per-city", type=int, default=6)
    p.add_argument("--n-test", type=int, default=6)
    p.add_argument("--pool-seed", type=int, default=0)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--k-extra", type=int, default=8)
    p.add_argument("--band", default="0.15,0.95")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json-out", default="")
    p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    band = tuple(float(x) for x in args.band.split(","))

    train_cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    assert args.holdout_city not in train_cities
    print(f"[gen24] TRAIN cities {train_cities} x {args.n_per_city}; HOLD-OUT {args.holdout_city} "
          f"x {args.n_test} (pool-seed {args.pool_seed})", flush=True)
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
    print(f"[gen24] TRAIN: {[(i.city, i.od) for i in train]}")
    print(f"[gen24] TEST (zero-shot): {[(i.city, i.od, round(i.eq, 3)) for i in test]}", flush=True)

    # actor architecture verbatim from gen16 (critics constructed for parity but unused)
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3, alpha_floor=0.20)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(2))
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w]})
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w],
                                          "lr": args.head_term_lr})

    # distillation targets (stacked minimax) + the disclosed ceiling vs the full equilibrium
    targets, ceilings = {}, []
    for it in train:
        x, v = stacked_minimax(it)
        targets[id(it)] = torch.tensor(x, dtype=torch.float32)
        ceilings.append((f"{it.city}:{it.od[0]}-{it.od[1]}", round(v, 4), round(it.eq, 4),
                         round(v / it.eq, 3)))
    print(f"[gen24] stacked-optimum/equilibrium ceilings per train instance: {ceilings}", flush=True)

    cache = {id(it): prep(it, prot) for it in train}
    hist = []
    t0 = time.time()
    for step in range(args.steps):
        prot.actor.train()
        losses = []
        for it in train:
            c = cache[id(it)]
            prot.actor.menu_routes = c["menu"]
            prot.actor.route_feats = c["feats"]
            probs, _ = prot.actor(c["pyg"], c["active"], list(range(c["R"])),
                                  torch.zeros(c["R"]))
            tgt = targets[id(it)]
            losses.append(torch.sum(tgt * (torch.log(tgt + 1e-12) - torch.log(probs + 1e-12))))
        loss = torch.stack(losses).mean()
        prot.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prot.actor.parameters(), 10.0)
        prot.actor_optimizer.step()

        if (step + 1) % args.eval_every == 0:
            for inst_set in (train, test):
                for it in inst_set:
                    _, d = exact_ratio(prot, it)
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
            hist.append((step + 1, tr_m, te_m, te_v, fw, float(loss.detach())))
            if args.ckpt_dir:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(),
                           str(Path(args.ckpt_dir) / f"actor_ep{step + 1}.pt"))
            print(f"  step {step+1:5d}: KL {float(loss):.4f} | TRAIN mean TAP ratio {tr_m:.2f} | "
                  f"TEST {te_m:.2f} {['%.2f' % v for v in te_v]} | rw[{fw[0]:.2f},{fw[1]:.2f}] | "
                  f"{time.time()-t0:5.0f}s", flush=True)

    # SELECT-ON-TRAIN (standing default): the checkpoint with the lowest TRAIN mean ratio;
    # held-out reported there. Select-on-test dual-reported as the optimistic bound.
    sel_train = min(hist, key=lambda h: h[1])
    sel_test = min(hist, key=lambda h: h[2])
    print(f"\n=== gen24 DISTILL (seed {args.seed}) ===")
    print(f"  SELECT-ON-TRAIN: held-out mean TAP ratio {sel_train[2]:.3f} @ step {sel_train[0]} "
          f"(train {sel_train[1]:.3f})")
    print(f"  select-on-test (optimistic bound): {sel_test[2]:.3f} @ step {sel_test[0]}; "
          f"final iterate {hist[-1][2]:.3f}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "train_ods": [(i.city, i.od) for i in train],
             "test_ods": [(i.city, i.od) for i in test],
             "test_refs": {f"{it.od[0]}-{it.od[1]}": {"eq": it.eq, "loss_det": it.loss_det}
                           for it in test},
             "ceilings": ceilings, "history": hist,
             "select_on_train": {"step": sel_train[0], "test_ratio": sel_train[2],
                                 "train_ratio": sel_train[1]},
             "select_on_test": {"step": sel_test[0], "test_ratio": sel_test[2]}}, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
