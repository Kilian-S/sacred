#!/usr/bin/env python3
"""gen34: train ONE history-aware fleet policy against a HIDDEN adversary-TYPE family
(experiments/gen34_hidden_adversary.md; pre-registered 2026-07-23).

The gen27 recipe (multi-city pool, fleet-route, pattern-of-life window head) with two changes:
  1. The enemy is drawn per EPISODE, hidden, uniformly from the five-member doctrine family
     (`scratch/gen34_family_probe.py:member_fns` - the normative definitions).
  2. The defender's route features gain two INTEL columns computed from the realised
     interdiction placements the game reveals after each sortie: col 3 = minmax of
     L[r, j_last]; col 4 = minmax of an EWMA (decay 0.8, per-episode reset) of L[r, j_s].
     `--no-intel` zeroes both (the causal control; the window col 2 stays).

Reward stays ANALYTIC (expected loss vs the member's current response - low variance, the
gen19/27 estimator); the intel columns use SAMPLED placements (the realistic observation).
Scored against the exact type-blind cap per instance (models/runs/gen34_hidden_adversary/
family_refs.json, produced by scratch/gen34_refs.py; Karp/damped-RVI, never the defective
undamped RVI).

Run: PYTHONPATH=. .venv/bin/python scripts/train_family_generalist.py --sorties 12000 --seed 0
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from scratch.gen34_family_probe import member_fns
from scripts.train_b1lite1 import softmax_br, stacked_L  # noqa: F401 (softmax_br via members)
from scripts.train_generalist import sample_instances
from src.agents.sac import ProtagonistSAC
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.env.smdp_wrapper import SMDPTransition

MEMBERS = ("reactive", "sharp", "anticipatory", "doctrine", "scattergun")


def prep_instance(it, refs):
    it.L = stacked_L(it.env.game, it.env.config.N)
    _, it.eq_mix = _row_minimiser(it.L)
    it.fns = member_fns(it.L, it.eq_mix)
    it.menu_idx = [torch.tensor(r, dtype=torch.long) for r in it.env.menu_route_node_idx()]
    it.nR = it.env.game.n_routes
    key = f"{it.city}:{it.od[0]}-{it.od[1]}"
    it.refs = refs[key]
    mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    it.base_cost = mm(np.asarray(it.env.game.travel_cost, float))
    it.base_worst = mm(it.env.game.payoff.max(axis=1))
    return it


def feats5(it, counts, w, j_last, ewma, no_intel=False):
    mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    freq = counts / w if counts.sum() > 0 else np.zeros_like(counts)
    intel1 = mm(it.L[:, j_last]) if j_last is not None else np.zeros(it.nR)
    intel2 = mm(ewma) if ewma is not None and ewma.max() > 0 else np.zeros(it.nR)
    f = np.stack([it.base_cost, it.base_worst, freq, intel1, intel2], axis=1)
    if no_intel:
        f[:, 3:] = 0.0
    return torch.tensor(f, dtype=torch.float32)


def build_obs(it, feats):
    it.env.reset()
    obs = it.env.observe()
    obs["menu_route_node_idx"] = it.menu_idx
    obs["menu_route_feats"] = feats
    return obs


def pick_route(prot, obs, R, deterministic=False):
    return int(prot.select_action(obs, {0: list(range(R))}, deterministic=deterministic)[0])


def run_episode(prot, it, member, w, ep_len, rng, no_intel, collect=None, il=10.0):
    """One 40-sortie episode vs a fixed hidden member. Returns mean per-sortie expected loss.
    collect: list to append SMDP steps to (training); None = eval only."""
    window = deque(maxlen=w)
    j_last, ewma = None, np.zeros(it.nR)
    fn = it.fns[member]
    tot = 0.0
    steps = []
    for _ in range(ep_len):
        counts = np.bincount(list(window), minlength=it.nR).astype(float)
        obs = build_obs(it, feats5(it, counts, w, j_last, ewma, no_intel))
        r = pick_route(prot, obs, it.nR)
        q = fn(counts)
        exp_loss = float(it.L[r] @ q)
        tot += exp_loss
        if collect is not None:
            steps.append((obs, r, -il * exp_loss))
        j_last = int(rng.choice(len(q), p=q))
        ewma = 0.8 * ewma + 0.2 * it.L[:, j_last]
        window.append(r)
    if collect is not None:
        collect.extend(steps)
    return tot / ep_len


def eval_mixture(prot, it, w, ep_len, rng, no_intel, eps_per_member):
    per = {}
    for m in MEMBERS:
        per[m] = float(np.mean([run_episode(prot, it, m, w, ep_len, rng, no_intel)
                                for _ in range(eps_per_member)]))
    val = float(np.mean(list(per.values())))
    return val, val / it.refs["blind_cap"], per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", default="kaliningrad,east_london,istanbul")
    p.add_argument("--holdout-city", default="gdansk")
    p.add_argument("--n-per-city", type=int, default=6); p.add_argument("--n-test", type=int, default=6)
    p.add_argument("--pool-seed", type=int, default=0)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--episode-len", type=int, default=40); p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--sorties", type=int, default=12000); p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-eps", type=int, default=10, help="held-out eval episodes per member")
    p.add_argument("--eval-eps-train", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--head-term-lr", type=float, default=3e-2); p.add_argument("--threads", type=int, default=3)
    p.add_argument("--no-intel", action="store_true", help="causal control: zero intel cols 3-4")
    p.add_argument("--refs", default="models/runs/gen34_hidden_adversary/family_refs.json")
    p.add_argument("--json-out", default=""); p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    N, K, BAND, KX = 3, 1, (0.15, 0.95), 8
    w = args.window
    refs = json.load(open(args.refs))

    train = []
    for c in args.cities.split(","):
        train += sample_instances(args.n_per_city, N, K, BAND, KX, args.pool_seed, city=c)
    test = sample_instances(args.n_test, N, K, BAND, KX, args.pool_seed, city=args.holdout_city)
    for it in train + test:
        prep_instance(it, refs)
    print(f"[gen34{'/no-intel' if args.no_intel else ''}] {len(train)} train + {len(test)} "
          f"held-out; family {MEMBERS}; per-instance blind caps:", flush=True)
    for it in test:
        print(f"    HELD-OUT {it.city} {it.od}: blind {it.refs['blind_cap']:.4f} "
              f"omni {it.refs['omni_cap']:.4f} gap {it.refs['blind_cap']/it.refs['omni_cap']:.2f}x",
              flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, gamma=args.gamma, autotune_alpha=True,
                          alpha_init=1.0, device="cpu")
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.route_feat_w = torch.nn.Parameter(torch.zeros(5))
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w], "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group(
        {"params": [prot.q1.route_feat_w, prot.q2.route_feat_w], "lr": args.head_term_lr})

    hist = []
    k = 0
    while k < args.sorties:
        it = train[int(rng.integers(len(train)))]
        member = MEMBERS[int(rng.integers(len(MEMBERS)))]
        steps = []
        run_episode(prot, it, member, w, args.episode_len, rng, args.no_intel,
                    collect=steps, il=args.interception_loss)
        k += args.episode_len
        for i, (obs, r, reward) in enumerate(steps):
            last = i == len(steps) - 1
            nstate = {}
            if not last:
                nstate = dict(steps[i + 1][0]); nstate["active_truck"] = 0
                nstate["allowed_destinations"] = {"protagonist": {0: list(range(it.nR))}}
            prot.replay_buffer.push(SMDPTransition(
                agent="protagonist", state=obs, action={0: r}, reward=reward, next_state=nstate,
                done=last, elapsed_ticks=1, action_mask={"protagonist": {0: list(range(it.nR))}},
                info={}))
        for _ in range(args.episode_len):
            prot.update(args.batch_size)
        if k % args.eval_every < args.episode_len:
            erng = np.random.default_rng(10_000 + k)   # fixed eval stream, decoupled from training
            tr = [eval_mixture(prot, it2, w, args.episode_len, erng, args.no_intel,
                               args.eval_eps_train) for it2 in train]
            te = [eval_mixture(prot, it2, w, args.episode_len, erng, args.no_intel,
                               args.eval_eps) for it2 in test]
            train_ratio = float(np.mean([x[1] for x in tr]))
            test_ratio = float(np.mean([x[1] for x in te]))
            beats = int(sum(1 for x in te if x[1] < 1.0))
            hist.append({"sortie": k, "train_ratio": train_ratio, "test_ratio": test_ratio,
                         "test_beats_blind": beats,
                         "test_values": [round(x[0], 4) for x in te],
                         "test_ratios": [round(x[1], 3) for x in te],
                         "test_per_member": [x[2] for x in te]})
            if args.ckpt_dir:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(Path(args.ckpt_dir) / f"actor_ep{k}.pt"))
            rw = prot.actor.route_feat_w.detach().numpy()
            print(f"  sortie {k:6d}: TRAIN ratio-to-blind {train_ratio:.3f} | HELD-OUT "
                  f"{test_ratio:.3f} (beats blind cap {beats}/{len(test)}) | "
                  f"rw[{rw[0]:.2f},{rw[1]:.2f},{rw[2]:.2f},{rw[3]:.2f},{rw[4]:.2f}] "
                  f"alpha {prot.alpha:.2f}", flush=True)

    sel = min(hist, key=lambda h: h["train_ratio"]) if hist else None
    opt = min(hist, key=lambda h: h["test_ratio"]) if hist else None
    print(f"\n=== gen34 HIDDEN-ADVERSARY GENERALIST (seed {args.seed}"
          f"{', NO-INTEL control' if args.no_intel else ''}) ===")
    if sel:
        print(f"  select-on-train @ sortie {sel['sortie']}: HELD-OUT ratio-to-blind-cap "
              f"{sel['test_ratio']:.3f}, beats the blind cap on {sel['test_beats_blind']}/"
              f"{len(test)} ODs")
        print(f"  select-on-test (optimistic bound): {opt['test_ratio']:.3f} @ sortie "
              f"{opt['sortie']} | final iterate: {hist[-1]['test_ratio']:.3f}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "no_intel": args.no_intel, "w": w, "members": MEMBERS,
             "test_refs": [{"city": it.city, "od": it.od, **it.refs} for it in test],
             "train_refs": [{"city": it.city, "od": it.od, "blind_cap": it.refs["blind_cap"],
                             "omni_cap": it.refs["omni_cap"]} for it in train],
             "history": hist, "select_on_train": sel, "select_on_test": opt}, indent=2))
        print(f"[written] {args.json_out}")


if __name__ == "__main__":
    main()
