#!/usr/bin/env python3
"""gen36 Step A: DISTILLATION control for the three-stream coordination act
(experiments/gen36_multiod_rescue.md; pre-registered 2026-07-23).

Trains the SAME policy class as the failed gen29 self-play half (stream-sequential actor with
the overlap head) to imitate each train instance's EXACT coordinated joint mixture
(dstar = _row_minimiser(env.obj_matrix)[1]; verified scratch/gen36_label_probe.py: anchors
reproduce exactly, supports are 2-11 atoms). Supervision replaces self-play entirely: the
question is CAPACITY (can the class express and transfer the correlated optimum?), separated
from self-play dynamics. gen24 discipline throughout: validation cells + early stopping,
select-on-val; zero-shot eval on the held-out cells via the exact exploitability.

Loss per instance: cross-entropy of the policy's factorised joint under the sparse label:
  L = - sum_{(r1,r2,r3) in supp(dstar)} dstar * [log d1(r1) + log d2(r2|r1) + log d3(r3|r1,r2)]
computed with differentiable prefix-conditioned forwards (few per instance: the support is
sparse).

Run (gen29 worktree): OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  /Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python scripts/distill_multiod.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_multiod_generalist import Inst, joint_dist, stream_probs  # noqa: F401
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.multiconvoy_oracle import _row_minimiser


def stream_logprobs_grad(prot, env, prefix):
    """Differentiable log-probs of the active stream's routes given a committed prefix."""
    env.set_committed(list(prefix))
    obs = env.observe()
    cur = env.current_stream()
    prot.actor.menu_routes = obs["menu_route_node_idx"]
    prot.actor.route_feats = obs["menu_route_feats"]
    pyg = featurize_state(obs, cur).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    active = n2i[obs["trucks"][cur]["current_node"]]
    R = len(env.route_sets[cur])
    probs, _ = prot.actor(pyg, active, list(range(R)), None)
    return torch.log(probs.clamp_min(1e-12))


def instance_loss(prot, inst):
    """Cross-entropy of the factorised policy joint under the sparse label support."""
    R = [len(rs) for rs in inst.env.route_sets]
    supp = np.nonzero(inst.label > 1e-9)[0]
    triples = [((j // R[2]) // R[1], (j // R[2]) % R[1], j % R[2]) for j in supp]
    lp1 = stream_logprobs_grad(prot, inst.env, [])
    loss = 0.0
    lp2_cache, lp3_cache = {}, {}
    for (r1, r2, r3), j in zip(triples, supp):
        if r1 not in lp2_cache:
            lp2_cache[r1] = stream_logprobs_grad(prot, inst.env, [r1])
        if (r1, r2) not in lp3_cache:
            lp3_cache[(r1, r2)] = stream_logprobs_grad(prot, inst.env, [r1, r2])
        lp = lp1[r1] + lp2_cache[r1][r2] + lp3_cache[(r1, r2)][r3]
        loss = loss - float(inst.label[j]) * lp
    return loss


def eval_pool(prot, insts):
    rows = {}
    for it in insts:
        d = joint_dist(prot, it.env)
        v = it.env.exploitability_of_joint_dist(d)
        rows[it.name] = dict(value=float(v), ratio_eq=float(v / it.eq),
                             ratio_cap=float(v / it.cap))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--screen", default="models/runs/gen29_screen.json")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10, help="val checks without improvement")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sc = json.load(open(args.screen))
    print("[gen36A] building pool + labels...", flush=True)
    t0 = time.time()
    train = [Inst(c) for c in ([sc["headline"]] + sc["pool"])]
    val = [Inst(c) for c in sc["validation"]]
    held = [Inst(c) for c in sc["held_out"]]
    for it in train:
        v, dstar = _row_minimiser(it.env.obj_matrix)
        assert abs(v - it.eq) / it.eq < 2e-3, (it.name, v, it.eq)
        it.label = dstar
    print(f"[gen36A] {len(train)} train (labelled) / {len(val)} val / {len(held)} held-out "
          f"in {time.time()-t0:.0f}s", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=args.lr, autotune_alpha=True,
                          alpha_init=1.0, device="cpu")
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.route_feat_w = torch.nn.Parameter(torch.zeros(2))
        net.route_feats = None
    opt = torch.optim.Adam([
        {"params": [pp for pp in prot.actor.parameters() if pp is not prot.actor.route_feat_w],
         "lr": args.lr},
        {"params": [prot.actor.route_feat_w], "lr": args.head_term_lr}])

    best = dict(val=float("inf"), epoch=-1, held=None, state=None)
    stale = 0
    hist = []
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        ep_loss = 0.0
        for i in order:
            opt.zero_grad()
            loss = instance_loss(prot, train[int(i)])
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
        if ep % args.val_every == 0:
            vr = eval_pool(prot, val)
            vmean = float(np.mean([r["ratio_eq"] for r in vr.values()]))
            hist.append(dict(epoch=ep, loss=ep_loss, val_ratio_eq=vmean))
            marker = ""
            if vmean < best["val"] - 1e-4:
                hr = eval_pool(prot, held)
                best = dict(val=vmean, epoch=ep, held=hr,
                            state={k: v.clone() for k, v in prot.actor.state_dict().items()})
                stale = 0
                marker = "  <-- new best (held-out snapshotted)"
            else:
                stale += 1
            print(f"  epoch {ep:4d}: CE {ep_loss:7.3f} | VAL ratio-to-eq {vmean:.3f}{marker}",
                  flush=True)
            if stale >= args.patience:
                print(f"[gen36A] early stop at epoch {ep} (patience {args.patience})", flush=True)
                break

    hr = best["held"]
    ratios_eq = [r["ratio_eq"] for r in hr.values()]
    ratios_cap = [r["ratio_cap"] for r in hr.values()]
    beats_cap = int(sum(1 for x in ratios_cap if x < 1.0))
    pooled_eq = float(np.mean(ratios_eq))
    print(f"\n=== gen36 STEP A (distillation, seed {args.seed}; select-on-val epoch "
          f"{best['epoch']}) ===")
    print(f"  HELD-OUT pooled ratio-to-eq {pooled_eq:.3f} (A-TIER-1 bar: < 1.44) | "
          f"beats-cap {beats_cap}/{len(hr)} (A-TIER-2 bar: >= 4/6)")
    for nm, r in hr.items():
        print(f"    {nm}: ratio-eq {r['ratio_eq']:.3f} ratio-cap {r['ratio_cap']:.3f}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            dict(seed=args.seed, select_on_val_epoch=best["epoch"], val_ratio=best["val"],
                 held_out=hr, pooled_ratio_eq=pooled_eq, beats_cap=beats_cap,
                 history=hist), indent=2))
        print(f"[written] {args.json_out}")


if __name__ == "__main__":
    main()
