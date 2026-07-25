#!/usr/bin/env python3
"""gen33 METRIC 2 (curriculum transfer): train the gen32 defender on kgd against one of three
enemy curricula, then (separately, scripts/eval_aerial_dyn33.py) evaluate zero-shot on the 6
held-out cells at the validation-selected checkpoint.

Arms (equal budget, identical SAC settings = the gen32 trainer's):
  llm     enemy per train instance = a force from the banked LLM population (kgd, both phases,
          fixed force-to-instance assignment across seeds), under the pinned scorer semantics.
  random  enemy per train instance = a random force (fixed per instance across arms/seeds),
          K alternating 1/3 to match the population's phase mix.
  single  enemy = the screened gen32 operating point (0.7, 0.3, 0) tau 0.10 w2, flat prior,
          every instance (the hand-tuned single-doctrine control).

Discipline: evaluation during training is VALIDATION ONLY (4 kgd fields, the arm's OWN
curriculum); held-out cells are never touched until the post-hoc eval at the val-selected
checkpoint. Checkpoints saved every eval. Thread pools capped.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path as _P

import numpy as np
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.agents.transition_builder import SMDPTransition
from src.redforce_score import (GEN32_DOCTRINE, ScoreBase, force_aim, random_force)

S_EP = 40
SIGMA0 = 8.0
KGD = "data/maps/theatre_kgd_gvardeysk_vec.json"


def _mm(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


class ForceInst:
    """One (field, enemy-force) training/val instance on the gen32 window machinery."""

    def __init__(self, base: ScoreBase, name, seed, sites, doctrine, sigma_km):
        self.name = name
        fc = base.field(seed)
        A, ctx = force_aim(fc, sites, doctrine, sigma_km)
        self.stepdmg = A @ fc.dmg.T
        self.states, self.succ = ctx.states, ctx.succ
        self.w = self.states.shape[1]
        self.R = base.R
        self.pows = self.R ** np.arange(self.w - 1, -1, -1)
        self.exposure = _mm(1.0 - fc.S.min(axis=1))
        from src.envs.aerial_theatre_env import TheatreEnv
        self.env = TheatreEnv(base.menu, fc.game, fc.S, N=3)

    def widx(self, window) -> int:
        return int(np.asarray(window) @ self.pows)

    def feats(self, window):
        wf = np.zeros(self.R)
        for r in window:
            wf[r] += 1.0 / self.w
        doc = _mm(self.stepdmg[self.widx(window)])
        return torch.tensor(np.stack([self.exposure, wf, doc], axis=1), dtype=torch.float32)

    def chain_value(self, dists: np.ndarray) -> float:
        Sn = len(self.states)
        pi = np.full(Sn, 1.0 / Sn)
        for _ in range(600):
            flow = pi[:, None] * dists
            nxt = np.zeros(Sn)
            np.add.at(nxt, self.succ.ravel(), flow.ravel())
            nxt = 0.5 * nxt + 0.5 * pi
            if np.abs(nxt - pi).max() < 1e-13:
                pi = nxt
                break
            pi = nxt
        return float((pi[:, None] * dists * self.stepdmg).sum())


def policy_dists(prot, inst: ForceInst) -> np.ndarray:
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
    dists = np.zeros((len(inst.states), R))
    with torch.no_grad():
        h = prot.actor.encoder(pyg.x, pyg.edge_index, pyg.edge_attr)
        for i, w in enumerate(inst.states):
            prot.actor.route_feats = inst.feats(tuple(int(x) for x in w))
            probs, _ = prot.actor.head(h, active, list(range(R)), torch.zeros(R))
            dists[i] = probs.cpu().numpy()
    prot.actor.train()
    return dists


def policy_value(prot, inst: ForceInst) -> float:
    return inst.chain_value(policy_dists(prot, inst))


def load_population(model):
    forces = []
    for phase in ("single", "coordinated"):
        art = json.load(open(f"models/runs/gen33_forces/force_{model}_kgd_{phase}.json"))
        for rec in art["forces"]:
            forces.append((rec["resolved"]["sites"],
                           [tuple(d) for d in rec["resolved"]["doctrine"]]))
    return forces


def arm_enemy(arm, i, base, llm_forces):
    """The enemy spec for instance index i (fixed across seeds and arms-of-same-kind)."""
    if arm == "llm":
        return llm_forces[i % len(llm_forces)]
    if arm == "random":
        rng = np.random.default_rng(9000 + i)
        return random_force(base, 1 if i % 2 == 0 else 3, rng)
    return [0], [GEN32_DOCTRINE]                    # single: flat prior via sigma=None


def make_pool(arm, llm_model):
    base = ScoreBase(KGD)
    sigma = None if arm == "single" else SIGMA0 * base.scale
    llm_forces = load_population(llm_model) if arm == "llm" else None
    train = [ForceInst(base, f"tr{1000 + i}", 1000 + i,
                       *arm_enemy(arm, i, base, llm_forces), sigma)
             for i in range(18)]
    val = [ForceInst(base, f"val{3000 + j}", 3000 + j,
                     *arm_enemy(arm, 18 + j, base, llm_forces), sigma)
           for j in range(4)]
    return base, train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["llm", "random", "single"], required=True)
    p.add_argument("--llm-model", default="llama-3.3-70b")
    p.add_argument("--sorties", type=int, default=8000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--ent-frac", type=float, default=0.5)
    p.add_argument("--alpha-floor", type=float, default=0.20)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=1)
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

    t0 = time.time()
    base, train, val = make_pool(args.arm, args.llm_model)
    print(f"[gen33 {args.arm} s{args.seed}] pool in {time.time()-t0:.0f}s "
          f"({len(train)} train, {len(val)} val)", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3,
                          alpha_floor=args.alpha_floor)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w]})
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w],
                                          "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.follow_w, prot.q2.follow_w]})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.route_feat_w, prot.q2.route_feat_w],
                                           "lr": args.head_term_lr})

    hist = []
    sortie = 0
    t0 = time.time()
    while sortie < args.sorties:
        inst = train[int(rng.integers(len(train)))]
        env = inst.env
        window = tuple(int(x) for x in rng.integers(inst.R, size=inst.w))
        ep = []
        for _ in range(S_EP):
            env.reset()
            f0 = inst.feats(window)
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
            dmgv = float(inst.stepdmg[inst.widx(window), r_lead])
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
            va = float(np.mean([policy_value(prot, it) for it in val]))
            hist.append((sortie, va, float(prot.alpha)))
            if args.ckpt_dir:
                _P(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(),
                           str(_P(args.ckpt_dir) / f"actor_ep{sortie}.pt"))
            print(f"  [{args.arm} s{args.seed}] sortie {sortie:6d}: VAL {va:.4f} "
                  f"a{prot.alpha:.2f} | {time.time()-t0:5.0f}s", flush=True)

    if args.json_out:
        _P(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _P(args.json_out).write_text(json.dumps(
            {"arm": args.arm, "seed": args.seed, "llm_model": args.llm_model,
             "history": hist}, indent=2))
        print(f"  [written] {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
