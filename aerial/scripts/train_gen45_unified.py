#!/usr/bin/env python3
"""gen45: the gen32 dynamic-register trainer on the UNIFIED real-corridor game
(ledger experiments/gen45_unified_corridor.md; Phase 0 pinned w=2, tau=0.10, DOC32
q=(0.6 repeat, 0.2 flee, 0.3 anti-repeat-anticipation) on 2026-08-09, gates G1 min 3.71,
G2 12/12). Same SAC machinery as gen32 verbatim; the only changes are the substrate
(ConcealBase: terrain v2 with conceal_reach 0.85, range_scale 0.7, 200 quota sites, the gen39
MULTIPLIER field band 0.55-1.0) and the pinned w=2. The enemy is DynTheatre unchanged, i.e.
the gen39 machinery's flat full-map-relocation limit (selfcheck 3.9e-12 in the hunt).

Refs (exact, per instance): iid_eq, local static optimum, the payoff-blind dynamic family,
the fitted doctrine rules (disclosed caps), history_opt. PRIMARY bar object = the static CAP
min(iid_eq, static_opt). Policy eval = EXACT stationary damage of the policy-induced window
chain. Field ranges per the ledger: train 45300-45317, val 45400-45403, dev-test 45101-45102
(diagnostics only), GATED pristine 45200-45205 (behind --eval-gated, confirmation only).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path as _P

import numpy as np
import torch

from scripts.train_multiconvoy import route_one  # noqa: F401  (kept for parity)
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.agents.transition_builder import SMDPTransition
from analysis.gen32_theatre_hunt import DynTheatre, rule_family
from analysis.gen45_corridor_hunt import make_base, lethality_for

W, TAU, S_EP = 2, 0.10, 40
Q_REP, Q_FLEE, Q_AR = 0.6, 0.2, 0.3


class Inst:
    """Wraps a DynTheatre on the unified substrate (oracle refs + doctrine + env)."""

    def __init__(self, base, name: str, seed: int):
        self.name = name
        self.g = DynTheatre(base, lethality_for(base, seed), W, TAU,
                            Q_REP, Q_FLEE, Q_AR, build_env=True)
        self.env = self.g.env
        self.R = self.g.R
        rows = rule_family(self.g)
        self.iid_eq = rows["iid_eq"]
        self.static_opt = rows["static_localopt*fit"]
        self.cap = min(self.iid_eq, self.static_opt)
        blind = [k for k in rows if k.startswith(("anti_", "rot_"))]
        self.best_blind = min(rows[k] for k in blind)
        self.fitted = min(rows[k] for k in ("myopic_dodge", "softdodge*fit", "composed*fit"))
        self.hist_opt = self.g.history_opt()
        self.wins = self.g.states

    def feats(self, window):
        return self.g.feats(tuple(int(x) for x in window))

    def chain_value(self, dists: np.ndarray) -> float:
        """Exact stationary damage of a window-conditioned route distribution (dists[state]->R)."""
        succ, stepdmg = self.g.succ, self.g.stepdmg
        Sn = len(self.wins)
        pi = np.full(Sn, 1.0 / Sn)
        for _ in range(600):
            flow = pi[:, None] * dists
            nxt = np.zeros(Sn)
            np.add.at(nxt, succ.ravel(), flow.ravel())
            nxt = 0.5 * nxt + 0.5 * pi
            if np.abs(nxt - pi).max() < 1e-13:
                pi = nxt
                break
            pi = nxt
        return float((pi[:, None] * dists * stepdmg).sum())


def policy_dists(prot, inst: Inst) -> np.ndarray:
    """The policy's route distribution for every window (encoder once, head per window)."""
    env = inst.env
    env.reset()
    obs = env.observe()
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    active = n2i[obs["trucks"][0]["current_node"]]
    prot.actor.menu_routes = obs["menu_route_node_idx"]
    prot.actor.eval()
    R = inst.R
    dists = np.zeros((len(inst.wins), R))
    with torch.no_grad():
        h = prot.actor.encoder(pyg.x, pyg.edge_index, pyg.edge_attr)
        for i, w in enumerate(inst.wins):
            prot.actor.route_feats = inst.feats(w)
            probs, _ = prot.actor.head(h, active, list(range(R)), torch.zeros(R))
            dists[i] = probs.cpu().numpy()
    prot.actor.train()
    return dists


def policy_value(prot, inst: Inst) -> float:
    return inst.chain_value(policy_dists(prot, inst))


def make_pool(eval_gated=False):
    base = make_base()
    train = [Inst(base, f"tr{45300 + s}", 45300 + s) for s in range(18)]
    val = [Inst(base, f"val{45400 + s}", 45400 + s) for s in range(4)]
    if eval_gated:
        test = [Inst(base, f"gated{45200 + s}", 45200 + s) for s in range(6)]
    else:
        test = [Inst(base, f"dev{s}", s) for s in (45101, 45102)]  # burned, diagnostics only
    return base, train, val, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sorties", type=int, default=16000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--ent-frac", type=float, default=0.5)
    p.add_argument("--alpha-floor", type=float, default=0.20)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-gated", action="store_true")
    p.add_argument("--blind", action="store_true")
    p.add_argument("--json-out", default="")
    p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("[gen45] building the unified-corridor pool (exact refs per field)...", flush=True)
    t0 = time.time()
    base, train, val, test = make_pool(eval_gated=args.eval_gated)
    if args.blind:                       # causal control: kill recency + doctrine columns
        for inst in train + val + test:
            inst._blind = True
    print(f"[gen45] pool in {time.time() - t0:.0f}s ({len(train)} train, {len(val)} val, "
          f"{len(test)} test)", flush=True)
    for it in val + test:
        print(f"    {it.name}: iid_eq={it.iid_eq:.3f} static_opt={it.static_opt:.3f} "
              f"CAP={it.cap:.3f} blind={it.best_blind:.3f} fit={it.fitted:.3f} "
              f"hist_opt={it.hist_opt:.3f}", flush=True)

    def feats_of(inst, w):
        f = inst.feats(w)
        if getattr(inst, "_blind", False):
            f = f.clone(); f[:, 1] = 0.0; f[:, 2] = 0.0   # zero recency + doctrine
        return f

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3,
                          alpha_floor=args.alpha_floor)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))   # [exposure, recency, doctrine]
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w]})
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w],
                                          "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.follow_w, prot.q2.follow_w]})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.route_feat_w, prot.q2.route_feat_w],
                                           "lr": args.head_term_lr})

    if args.sorties == 0:
        rows = {it.name: policy_value(prot, it) for it in val + test}
        beats = sum(1 for i in test if rows[i.name] < i.cap)
        print(f"[gen45] UNTRAINED: test beats-CAP {beats}/{len(test)}; ratios-to-iid "
              + " ".join(f"{rows[i.name]/i.iid_eq:.2f}" for i in test))
        return

    hist = []
    sortie = 0
    t0 = time.time()
    while sortie < args.sorties:
        inst = train[int(rng.integers(len(train)))]
        env = inst.env
        window = tuple(int(x) for x in rng.integers(inst.R, size=W))
        ep = []
        for _ in range(S_EP):
            env.reset()
            f0 = feats_of(inst, window)
            steps = []
            leader_act = None
            for _ in range(env.config.N):
                ci = env.current_convoy()
                obs = env.observe()
                obs["menu_route_feats"] = f0
                mask = env.defender_action_mask()
                act = (prot.select_action(obs, mask)[ci] if ci == 0 else leader_act)
                if ci == 0:
                    leader_act = act
                env.route_convoy_by_index(int(act))
                steps.append((obs, ci, act, mask))
            r_lead = int(leader_act)
            dmgv = float(inst.g.stepdmg[inst.g.widx(window), r_lead])
            ep.append((steps, -args.interception_loss * dmgv))
            window = tuple(list(window[1:]) + [r_lead])
            sortie += 1
        flat = []
        for si, (steps, rew) in enumerate(ep):
            for j, (obs, ci, act, mask) in enumerate(steps):
                last = j == len(steps) - 1
                flat.append((obs, ci, act, mask, rew if last else 0.0, 1 if last else 0,
                             si == len(ep) - 1 and last))
        for idx, (obs, ci, act, mask, rew, dt, done) in enumerate(flat):
            obs["target_entropy"] = (0.05 if ci != 0 and sortie > 250 else args.ent_frac) \
                * math.log(inst.R)
            obs["alpha_group"] = 1 if (ci != 0 and sortie > 250) else 0
            if done or idx == len(flat) - 1:
                nstate = {}
            else:
                nobs, nci, nmask = flat[idx + 1][0], flat[idx + 1][1], flat[idx + 1][3]
                nstate = dict(nobs)
                nstate["active_truck"] = nci
                nstate["allowed_destinations"] = {"protagonist": {nci: list(nmask[nci])}}
            prot.replay_buffer.push(SMDPTransition(
                agent="protagonist", state=obs, action={ci: act}, reward=rew,
                next_state=nstate, done=bool(done), elapsed_ticks=dt,
                action_mask={"protagonist": mask}, info={}))
        for _ in range(S_EP):
            prot.update(args.batch_size)

        if sortie % args.eval_every < S_EP:
            rows = {}
            groups = (val + test) if args.eval_gated else (train[:3] + val + test)
            for it in groups:
                rows[it.name] = policy_value(prot, it)
            va = float(np.mean([rows[i.name] / i.iid_eq for i in val]))
            beats = sum(1 for i in test if rows[i.name] < i.cap)
            bblind = sum(1 for i in test if rows[i.name] < i.best_blind)
            te = float(np.mean([rows[i.name] / i.iid_eq for i in test]))
            fw = tuple(float(x) for x in prot.actor.route_feat_w.detach())
            hist.append((sortie, va, te, beats, bblind, rows, fw, float(prot.alpha)))
            if args.ckpt_dir:
                _P(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(),
                           str(_P(args.ckpt_dir) / f"actor_ep{sortie}.pt"))
            print(f"  sortie {sortie:6d}: VAL {va:.2f} | TEST beats-CAP {beats}/{len(test)} "
                  f"beats-BLIND {bblind}/{len(test)} ratio {te:.2f} | "
                  f"rw[{fw[0]:.2f},{fw[1]:.2f},{fw[2]:.2f}] a{prot.alpha:.2f} | "
                  f"{time.time()-t0:5.0f}s", flush=True)

    if args.json_out:
        refs = {it.name: {"iid_eq": it.iid_eq, "static_opt": it.static_opt,
                          "cap": it.cap, "best_blind": it.best_blind, "fitted": it.fitted,
                          "hist_opt": it.hist_opt} for it in train + val + test}
        _P(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _P(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "blind": args.blind, "refs": refs, "history": hist}, indent=2))
        print(f"  [written] {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
