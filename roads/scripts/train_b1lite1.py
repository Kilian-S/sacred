#!/usr/bin/env python3
"""Trains a history-aware fleet-route policy against a within-episode pattern-of-life
interdictor. An episode is S sorties; each sortie the fleet stacks on one route, and the
adversary softmax-best-responds (temperature tau) to the defender's realised routes over a
trailing window of w sorties. The defender observes the window (per-route recent frequency as a
route-feature column) and must learn to vary its routing to exploit the adversary's
predictability.

Reward is the analytic expected mission failure; sorties are chained with bootstrapping since
the window is state. Scored against the exact oracle references: static_det > iid_eq
(history-blind) > SACRED >= history_opt.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.multiconvoy_oracle import _row_minimiser, objective_value
from src.env.smdp_wrapper import SMDPTransition
from src.envs.multiconvoy_interdiction import make_multiconvoy_env


def stacked_L(game, N):
    R, nj = game.n_routes, game.payoff.shape[1]
    L = np.zeros((R, nj))
    for r in range(R):
        occ = np.zeros(R, dtype=int); occ[r] = N
        for j in range(nj):
            L[r, j] = objective_value(occ, game.payoff[:, j], N, "mission")
    return L


def softmax_br(counts, L, tau):
    tot = counts.sum()
    d = counts / tot if tot > 0 else np.ones(len(counts)) / len(counts)
    e = d @ L
    z = np.exp((e - e.max()) / tau)
    return z / z.sum()


def oracle_refs(L, tau, w):
    R = L.shape[0]
    v_eq, eq = _row_minimiser(L)
    # iid_eq
    eqL = eq @ L; ie = 0.0
    for win in itertools.product(range(R), repeat=w):
        pw = 1.0
        for i in win:
            pw *= eq[i]
        if pw:
            ie += pw * float(eqL @ softmax_br(np.bincount(win, minlength=R).astype(float), L, tau))
    # static_det
    sd = min(float(L[r] @ softmax_br(np.eye(R)[r] * w, L, tau)) for r in range(R))
    # history_opt (relative value iteration)
    states = list(itertools.product(range(R), repeat=w)); idx = {s: i for i, s in enumerate(states)}
    cost = np.zeros((len(states), R)); nxt = np.zeros((len(states), R), dtype=int)
    for si, s in enumerate(states):
        p = softmax_br(np.bincount(s, minlength=R).astype(float), L, tau); cost[si] = L @ p
        for r in range(R):
            nxt[si, r] = idx[s[1:] + (r,)]
    V = np.zeros(len(states))
    for _ in range(5000):
        Vn = (cost + V[nxt]).min(axis=1); Vn = Vn - Vn[0]
        if np.max(np.abs(Vn - V)) < 1e-10:
            V = Vn; break
        V = Vn
    ho = float(((cost + V[nxt]).min(axis=1) - V)[0])
    return dict(v_eq=v_eq, eq=eq, iid_eq=ie, static_det=sd, history_opt=ho)


def route_feats(env, window_counts, w):
    """[R,3]: per-instance normalised [cost, worst-vuln] + window recent-frequency (the history)."""
    cost = np.asarray(env.game.travel_cost, float); worst = env.game.payoff.max(axis=1)
    mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    freq = window_counts / w if window_counts.sum() > 0 else np.zeros_like(window_counts)
    return torch.tensor(np.stack([mm(cost), mm(worst), freq], axis=1), dtype=torch.float32)


def build_obs(env, menu_idx, feats):
    env.reset(); obs = env.observe()
    obs["menu_route_node_idx"] = menu_idx
    obs["menu_route_feats"] = feats
    return obs


def pick_route(prot, obs, R, deterministic=False):
    mask = {0: list(range(R))}
    return int(prot.select_action(obs, mask, deterministic=deterministic)[0])


def eval_policy(prot, env, menu_idx, L, tau, w, n=2000, deterministic=False):
    """Stationary per-sortie expected mission-failure of the policy vs the pattern-of-life adversary."""
    window = deque(maxlen=w); tot = 0.0; R = env.game.n_routes
    for t in range(n):
        counts = np.bincount(list(window), minlength=R).astype(float)
        obs = build_obs(env, menu_idx, route_feats(env, counts, w))
        r = pick_route(prot, obs, R, deterministic=deterministic)
        tot += float(L[r] @ softmax_br(counts, L, tau))
        window.append(r)
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--od", default="35-159"); p.add_argument("--N", type=int, default=3)
    p.add_argument("--K", type=int, default=1); p.add_argument("--k-extra", type=int, default=8)
    p.add_argument("--band", default="0.15,0.95")
    p.add_argument("--window", type=int, default=3); p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--episode-len", type=int, default=40); p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--sorties", type=int, default=8000); p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--head-term-lr", type=float, default=3e-2); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-window", action="store_true", help="causal control: zero the window feature")
    p.add_argument("--json-out", default=""); p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    s, t = args.od.split("-"); band = tuple(float(x) for x in args.band.split(","))
    env = make_multiconvoy_env(od=(s, t), N=args.N, K=args.K, k_extra_routes=args.k_extra,
                               menu_select=True, edge_vuln_band=band,
                               interception_loss=args.interception_loss, seed=args.seed)
    R = env.game.n_routes; w = args.window
    L = stacked_L(env.game, args.N); refs = oracle_refs(L, args.tau, w)
    print(f"[b1lite1] {args.od} R={R} w={w} tau={args.tau} | oracle: static_det {refs['static_det']:.3f} "
          f"> iid_eq {refs['iid_eq']:.3f} > history_opt {refs['history_opt']:.3f} (V_eq {refs['v_eq']:.3f})",
          flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, gamma=args.gamma, autotune_alpha=True,
                          alpha_init=1.0, device="cpu")
    menu_idx = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.menu_routes = menu_idx
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w], "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group(
        {"params": [prot.q1.route_feat_w, prot.q2.route_feat_w], "lr": args.head_term_lr})

    def feats_for(counts):
        f = route_feats(env, counts, w)
        if args.no_window:
            f[:, 2] = 0.0
        return f

    hist = []; k = 0; window = deque(maxlen=w)
    while k < args.sorties:
        window.clear()
        steps = []
        for _ in range(args.episode_len):
            counts = np.bincount(list(window), minlength=R).astype(float)
            obs = build_obs(env, menu_idx, feats_for(counts))
            r = pick_route(prot, obs, R)
            reward = -args.interception_loss * float(L[r] @ softmax_br(counts, L, args.tau))
            steps.append((obs, r, reward)); window.append(r); k += 1
        # push chained transitions (window is state; bootstrap through the episode)
        for i, (obs, r, reward) in enumerate(steps):
            last = i == len(steps) - 1
            nobs = steps[i + 1][0] if not last else {}
            nstate = {}
            if not last:
                nstate = dict(nobs); nstate["active_truck"] = 0
                nstate["allowed_destinations"] = {"protagonist": {0: list(range(R))}}
            prot.replay_buffer.push(SMDPTransition(
                agent="protagonist", state=obs, action={0: r}, reward=reward, next_state=nstate,
                done=last, elapsed_ticks=1, action_mask={"protagonist": {0: list(range(R))}}, info={}))
        for _ in range(args.episode_len):
            prot.update(args.batch_size)
        if k % args.eval_every < args.episode_len:
            ev = eval_policy(prot, env, menu_idx, L, args.tau, w)
            hist.append((k, ev, 0.0))
            rw = prot.actor.route_feat_w.detach().numpy()
            if args.ckpt_dir:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(Path(args.ckpt_dir) / f"actor_ep{k}.pt"))
            print(f"  sortie {k:6d}: SACRED per-sortie loss {ev:.3f} "
                  f"(vs iid_eq {refs['iid_eq']:.3f}, history_opt {refs['history_opt']:.3f}) | "
                  f"rw[{rw[0]:.2f},{rw[1]:.2f},{rw[2]:.2f}] alpha {prot.alpha:.2f}", flush=True)

    evs = [h[1] for h in hist]
    best = min(evs) if evs else float("nan")
    print(f"\n=== B1-LITE-1 ({args.od}, w={w}, tau={args.tau}, seed={args.seed}"
          f"{', NO-WINDOW control' if args.no_window else ''}) ===")
    print(f"  static_det {refs['static_det']:.3f} > iid_eq {refs['iid_eq']:.3f} > "
          f"SACRED best {best:.3f} >= history_opt {refs['history_opt']:.3f}")
    print(f"  PRIMARY (SACRED < iid_eq {refs['iid_eq']:.3f}): {'PASS' if best < refs['iid_eq'] else 'FAIL'} | "
          f"STRONG (within 0.03 of history_opt): {'YES' if best <= refs['history_opt'] + 0.03 else 'no'}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"od": args.od, "w": w, "tau": args.tau, "seed": args.seed, "no_window": args.no_window,
             "refs": {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
                      for kk, vv in refs.items()},
             "history": hist, "best": best}, indent=2))


if __name__ == "__main__":
    main()
