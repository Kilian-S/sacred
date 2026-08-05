#!/usr/bin/env python3
"""gen27: the DYNAMIC GENERALIST (gen19 x gen16; Block R2).

ONE history-aware fleet-route policy trained across a multi-city instance pool against each
instance's OWN within-episode pattern-of-life interdictor (softmax-BR, temperature tau, to the
defender's realised routes over a trailing w-window), evaluated ZERO-SHOT on a held-out city
against each held-out instance's computable dynamic yardsticks (iid_eq, history_opt by RVI).

Why this is the rescued ZST claim: every STATIC object — the disjoint max-flow heuristic, the
LP equilibrium mixture, any distilled/retrieved policy — is mathematically capped at iid_eq
against this adversary; only history-conditioned play can go below it (gen19 measured 0.050 vs
the 0.147 cap, single-instance). Beating iid_eq zero-shot on a never-seen city is therefore a
claim no static method can match, by construction.

Recipe: gen19 mechanism VERBATIM (w=3, tau=0.15 operating point, S=40 sortie episodes chained
with gamma=0.95, analytic expected-mission-failure reward, [cost, worst-vuln, window-freq] route
features at head-term lr) on the gen16 pools VERBATIM (train cities x 6 ODs, held-out city x 6,
pool-seed 0). Selection = select-on-train (standing default); select-on-test dual-reported.

Run: PYTHONPATH=. python scripts/train_dyn_generalist.py --sorties 12000 --seed 0
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from scripts.train_b1lite1 import oracle_refs, route_feats, softmax_br, stacked_L
from scripts.train_generalist import Instance, sample_instances
from src.agents.sac import ProtagonistSAC
from src.env.smdp_wrapper import SMDPTransition


def _karp_mmc(cost, n, R, pw):
    """Exact minimum mean cycle (scratch/dyn_exact.py's solver, inlined so the trainer stays
    self-contained; the corrected dynamic-optimum method per binding rule 8)."""
    v = np.arange(n)
    heads = v // R
    U = (np.arange(R)[:, None] * pw) + heads[None, :]
    A = v % R
    Cin = cost[U, A[None, :]]
    d = np.full((n + 1, n), np.inf)
    d[0] = 0.0
    for k in range(1, n + 1):
        d[k] = (d[k - 1][U] + Cin).min(axis=0)
    ks = np.arange(n)[:, None]
    with np.errstate(invalid="ignore"):
        ratios = (d[n][None, :] - d[:n]) / (n - ks)
    ratios = np.where(np.isfinite(ratios), ratios, -np.inf)
    per_v = ratios.max(axis=0)
    per_v = np.where(np.isfinite(d[n]), per_v, np.inf)
    return float(per_v.min())


def fast_refs(L, tau, w, route_edges):
    """gen41 references for deep windows where oracle_refs' R^w enumeration is infeasible:
    exact iid_eq by count-class enumeration with multinomial weights, static_det, and the
    corridor-restricted exact optimum (Karp over the m^w core window graph)."""
    import itertools as _it
    import math as _math
    from src.baselines.multiconvoy_oracle import _row_minimiser as _rm
    R = L.shape[0]
    v_eq, eq = _rm(L)
    sup = np.where(eq > 1e-12)[0]
    fw = _math.factorial(w)
    iid = 0.0
    for multi in _it.combinations_with_replacement(range(len(sup)), w):
        counts = np.zeros(R)
        prob = 1.0
        for j in multi:
            counts[sup[j]] += 1
            prob *= eq[sup[j]]
        denom = 1
        for j in set(multi):
            denom *= _math.factorial(multi.count(j))
        iid += (fw // denom) * prob * float((L @ softmax_br(counts, L, tau)) @ eq)
    sd = min(float(L[r] @ softmax_br(np.eye(R)[r] * w, L, tau)) for r in range(R))
    # corridor core (greedy disjoint subset, the standing convention)
    kept, used = [], set()
    for i, re_ in enumerate(route_edges):
        if not (set(re_) & used):
            kept.append(i)
            used |= set(re_)
    m = len(kept)
    n = m ** w
    dec = np.empty((n, w), dtype=np.int64)
    x = np.arange(n)
    for i in range(w):
        dec[:, w - 1 - i] = x % m
        x = x // m
    cost = np.empty((n, m))
    for s in range(n):
        counts = np.zeros(R)
        for j in range(m):
            counts[kept[j]] = (dec[s] == j).sum()
        q = softmax_br(counts, L, tau)
        cost[s] = (L @ q)[kept]
    opt_core = _karp_mmc(cost, n, m, m ** (w - 1))
    return dict(v_eq=v_eq, iid_eq=iid, static_det=sd, opt_core=opt_core)


def prep_instance(it, tau, w, fast=False):
    """Attach the dynamic-game apparatus to a pool instance: stacked payoff L, oracle refs
    (iid_eq / history_opt / static_det via RVI; or the gen41 fast refs for deep windows),
    cached menus."""
    it.L = stacked_L(it.env.game, it.env.config.N)
    it.refs = fast_refs(it.L, tau, w, it.env.game.route_edges) if fast \
        else oracle_refs(it.L, tau, w)
    it.menu_idx = [torch.tensor(r, dtype=torch.long) for r in it.env.menu_route_node_idx()]
    it.nR = it.env.game.n_routes
    return it


def load_pool_file(path, N, K, band, kx, seed):
    """gen41 pool injection: {'train': [[city, s, t], ...], 'test': [[city, s, t], ...]}."""
    spec = json.loads(Path(path).read_text())
    def build(rows):
        return [Instance((s, t), N, K, band, kx, seed, city=c) for c, s, t in rows]
    return build(spec["train"]), build(spec["test"])


def build_obs(it, counts, w, no_window=False):
    it.env.reset()
    obs = it.env.observe()
    f = route_feats(it.env, counts, w)
    if no_window:
        f[:, 2] = 0.0
    obs["menu_route_node_idx"] = it.menu_idx
    obs["menu_route_feats"] = f
    return obs


def pick_route(prot, obs, R, deterministic=False):
    return int(prot.select_action(obs, {0: list(range(R))}, deterministic=deterministic)[0])


def eval_instance(prot, it, tau, w, n=1000, no_window=False):
    """Stationary per-sortie expected mission failure vs the instance's pattern-of-life
    adversary (the gen19 estimator), plus the ratio to the instance's own iid_eq cap."""
    window = deque(maxlen=w)
    tot = 0.0
    for _ in range(n):
        counts = np.bincount(list(window), minlength=it.nR).astype(float)
        obs = build_obs(it, counts, w, no_window=no_window)
        r = pick_route(prot, obs, it.nR)
        tot += float(it.L[r] @ softmax_br(counts, it.L, tau))
        window.append(r)
    loss = tot / n
    return loss, loss / it.refs["iid_eq"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", default="kaliningrad,east_london,istanbul")
    p.add_argument("--holdout-city", default="gdansk")
    p.add_argument("--n-per-city", type=int, default=6); p.add_argument("--n-test", type=int, default=6)
    p.add_argument("--pool-seed", type=int, default=0)
    p.add_argument("--K", type=int, default=1); p.add_argument("--k-extra", type=int, default=8)
    p.add_argument("--pool-file", default="", help="gen41: explicit OD pools (json)")
    p.add_argument("--fast-refs", action="store_true",
                   help="gen41 deep-window refs (count-class iid_eq + core Karp); required "
                        "where oracle_refs' R^w enumeration is infeasible")
    p.add_argument("--window", type=int, default=3); p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--episode-len", type=int, default=40); p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--sorties", type=int, default=12000); p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=1000, help="held-out eval sorties per instance")
    p.add_argument("--eval-n-train", type=int, default=400, help="train-set eval sorties per instance")
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--head-term-lr", type=float, default=3e-2); p.add_argument("--threads", type=int, default=3)
    p.add_argument("--no-window", action="store_true", help="causal control: window feature zeroed")
    p.add_argument("--json-out", default=""); p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    N, K, BAND, KX = 3, args.K, (0.15, 0.95), args.k_extra
    w, tau = args.window, args.tau

    cities = args.cities.split(",")
    if args.pool_file:
        train, test = load_pool_file(args.pool_file, N, K, BAND, KX, args.pool_seed)
    else:
        train = []
        for c in cities:
            train += sample_instances(args.n_per_city, N, K, BAND, KX, args.pool_seed, city=c)
        test = sample_instances(args.n_test, N, K, BAND, KX, args.pool_seed, city=args.holdout_city)
    for it in train + test:
        prep_instance(it, tau, w, fast=args.fast_refs)
    print(f"[gen27] {len(train)} train + {len(test)} held-out; w={w} tau={tau} K={K} "
          f"kx={KX}; per-instance yardsticks:", flush=True)
    for tag, pool in (("train", train), ("HELD-OUT", test)):
        for it in pool:
            deep = it.refs.get("history_opt", it.refs.get("opt_core"))
            deep_name = "history_opt" if "history_opt" in it.refs else "opt_core"
            print(f"    {tag} {it.city} {it.od}: static_det {it.refs['static_det']:.3f} > "
                  f"iid_eq {it.refs['iid_eq']:.3f} > {deep_name} {deep:.3f}", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, gamma=args.gamma, autotune_alpha=True,
                          alpha_init=1.0, device="cpu")
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w], "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group(
        {"params": [prot.q1.route_feat_w, prot.q2.route_feat_w], "lr": args.head_term_lr})

    hist = []
    k = 0
    while k < args.sorties:
        it = train[int(rng.integers(len(train)))]   # one EPISODE per sampled training instance
        window = deque(maxlen=w)
        steps = []
        for _ in range(args.episode_len):
            counts = np.bincount(list(window), minlength=it.nR).astype(float)
            obs = build_obs(it, counts, w, no_window=args.no_window)
            r = pick_route(prot, obs, it.nR)
            reward = -args.interception_loss * float(it.L[r] @ softmax_br(counts, it.L, tau))
            steps.append((obs, r, reward)); window.append(r); k += 1
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
            tr = [eval_instance(prot, it2, tau, w, n=args.eval_n_train,
                                no_window=args.no_window) for it2 in train]
            te = [eval_instance(prot, it2, tau, w, n=args.eval_n,
                                no_window=args.no_window) for it2 in test]
            train_ratio = float(np.mean([x[1] for x in tr]))
            test_ratio = float(np.mean([x[1] for x in te]))
            beats = int(sum(1 for x in te if x[1] < 1.0))
            hist.append({"sortie": k, "train_ratio": train_ratio, "test_ratio": test_ratio,
                         "test_beats_iid_eq": beats,
                         "test_losses": [round(x[0], 4) for x in te],
                         "test_ratios": [round(x[1], 3) for x in te]})
            if args.ckpt_dir:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(Path(args.ckpt_dir) / f"actor_ep{k}.pt"))
            rw = prot.actor.route_feat_w.detach().numpy()
            print(f"  sortie {k:6d}: TRAIN ratio-to-iid_eq {train_ratio:.3f} | HELD-OUT "
                  f"{test_ratio:.3f} (beats cap {beats}/{len(test)}) | "
                  f"rw[{rw[0]:.2f},{rw[1]:.2f},{rw[2]:.2f}] alpha {prot.alpha:.2f}", flush=True)

    # selection: select-on-TRAIN (standing default); select-on-test dual-reported
    sel = min(hist, key=lambda h: h["train_ratio"]) if hist else None
    opt = min(hist, key=lambda h: h["test_ratio"]) if hist else None
    print(f"\n=== gen27 DYNAMIC GENERALIST (seed {args.seed}"
          f"{', NO-WINDOW control' if args.no_window else ''}) ===")
    if sel:
        print(f"  select-on-train @ sortie {sel['sortie']}: HELD-OUT ratio-to-iid_eq "
              f"{sel['test_ratio']:.3f}, beats the static cap on {sel['test_beats_iid_eq']}/{len(test)} ODs")
        print(f"  select-on-test (optimistic bound): {opt['test_ratio']:.3f} @ sortie {opt['sortie']} | "
              f"final iterate: {hist[-1]['test_ratio']:.3f}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "no_window": args.no_window, "w": w, "tau": tau,
             "K": args.K, "k_extra": args.k_extra, "pool_file": args.pool_file,
             "train_refs": [{"city": it.city, "od": it.od,
                             **{kk: float(it.refs[kk]) for kk in
                                ("static_det", "iid_eq", "history_opt", "opt_core", "v_eq")
                                if kk in it.refs}} for it in train],
             "test_refs": [{"city": it.city, "od": it.od,
                            **{kk: float(it.refs[kk]) for kk in
                               ("static_det", "iid_eq", "history_opt", "opt_core", "v_eq")
                               if kk in it.refs}} for it in test],
             "history": hist,
             "select_on_train": sel, "select_on_test": opt}, indent=2))
        print(f"[written] {args.json_out}")


if __name__ == "__main__":
    main()
