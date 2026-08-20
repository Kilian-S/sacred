#!/usr/bin/env python3
"""gen38 Step V2: trains a type-conditioned SACRED policy whose menu-route features include a
per-route TYPE-THREAT column (the believed enemy type's expected loss next sortie), training the
policy with the true type told each episode. Evaluates on held-out Gdansk cells both when told
the true enemy type and when told the LLM's predicted type (from V1), against the gen34 blind
wall.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from analysis.gen34_family_probe import member_fns
from analysis.gen38_enemy_id import instance_apparatus
from analysis.gen38_narratives import MEMBERS
from scripts.train_b1lite1 import softmax_br, stacked_L
from scripts.train_generalist import sample_instances
from src.agents.sac import ProtagonistSAC
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.env.smdp_wrapper import SMDPTransition

N, K, BAND, KX, W, TAU = 3, 1, (0.15, 0.95), 8, 3, 0.15
IL, EP_LEN = 10.0, 40


def prep(it, refs):
    it.L = stacked_L(it.env.game, N)
    _, eq = _row_minimiser(it.L)
    it.fns = member_fns(it.L, eq)
    it.menu_idx = [torch.tensor(r, dtype=torch.long) for r in it.env.menu_route_node_idx()]
    it.nR = it.env.game.n_routes
    mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    it.base_cost = mm(np.asarray(it.env.game.travel_cost, float))
    it.base_worst = mm(it.env.game.payoff.max(axis=1))
    it.key = f"{it.city}:{it.od[0]}-{it.od[1]}"
    it.refs = refs[it.key]
    return it


def feats_cond(it, counts, w, j_last, ewma, member_idx):
    """Builds the 6 menu-route feature columns: 5 base columns (gen34) plus a per-route
    type-conditioning column, the believed type's expected loss next sortie
    (threat_m[r] = (L @ q_m(counts))[r], with q_m the conditioned member's response to the
    current window). Unlike a broadcast one-hot, this is per-route and type-dependent, so it
    actually discriminates between routes."""
    mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    freq = counts / w if counts.sum() > 0 else np.zeros_like(counts)
    intel1 = mm(it.L[:, j_last]) if j_last is not None else np.zeros(it.nR)
    intel2 = mm(ewma) if ewma is not None and ewma.max() > 0 else np.zeros(it.nR)
    q_m = it.fns[MEMBERS[member_idx]](counts)         # believed type's response to the window
    threat = mm(it.L @ q_m)                            # per-route expected loss under that type
    f = np.stack([it.base_cost, it.base_worst, freq, intel1, intel2, threat], axis=1)
    return torch.tensor(f, dtype=torch.float32)


def obs_for(it, feats):
    it.env.reset()
    obs = it.env.observe()
    obs["menu_route_node_idx"] = it.menu_idx
    obs["menu_route_feats"] = feats
    return obs


def episode(prot, it, member, cond_idx, rng, collect=None):
    window = deque(maxlen=W)
    j_last, ewma = None, np.zeros(it.nR)
    fn = it.fns[member]
    tot = 0.0
    steps = []
    for _ in range(EP_LEN):
        counts = np.bincount(list(window), minlength=it.nR).astype(float)
        obs = obs_for(it, feats_cond(it, counts, W, j_last, ewma, cond_idx))
        r = int(prot.select_action(obs, {0: list(range(it.nR))})[0])
        q = fn(counts)
        loss = float(it.L[r] @ q)
        tot += loss
        if collect is not None:
            steps.append((obs, r, -IL * loss))
        j_last = int(rng.choice(len(q), p=q))
        ewma = 0.8 * ewma + 0.2 * it.L[:, j_last]
        window.append(r)
    if collect is not None:
        collect.extend(steps)
    return tot / EP_LEN


def eval_told(prot, it, cond_of, rng, eps=8):
    """Mean over members of the episodic value when conditioned on cond_of(true_member)."""
    vals = []
    for m in MEMBERS:
        cond = cond_of(m)
        vals.append(np.mean([episode(prot, it, m, cond, rng) for _ in range(eps)]))
    return float(np.mean(vals))


def load_llm_preds():
    """Per-member majority LLM prediction from the V1 transcripts (deployment conditioning)."""
    tdir = Path("models/runs/gen38_llm_enemy_id/transcripts")
    by_type = {m: [] for m in MEMBERS}
    for f in tdir.glob("*.json"):
        d = json.loads(f.read_text())
        preds = [x.get("pred") for x in d["draws"] if x.get("pred") in MEMBERS]
        if preds:
            by_type[d["true"]].append(max(set(preds), key=preds.count))
    # map true member -> the LLM's majority predicted member (across its narratives)
    return {m: (max(set(v), key=v.count) if v else m) for m, v in by_type.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sorties", type=int, default=12000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--head-term-lr", type=float, default=3e-2)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--ckpt-dir", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    refs = json.load(open("models/runs/gen34_hidden_adversary/family_refs.json"))
    train = []
    for c in ("kaliningrad", "east_london", "istanbul"):
        train += [prep(it, refs) for it in sample_instances(6, N, K, BAND, KX, 0, city=c)]
    test = [prep(it, refs) for it in sample_instances(6, N, K, BAND, KX, 0, city="gdansk")]
    llm_map = load_llm_preds()
    idx_of = {m: i for i, m in enumerate(MEMBERS)}
    print(f"[v2] {len(train)} train + {len(test)} held-out; LLM deploy map {llm_map}", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, gamma=0.95, autotune_alpha=True,
                          alpha_init=1.0, device="cpu")
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.route_feat_w = torch.nn.Parameter(torch.zeros(6))
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w], "lr": a.head_term_lr})
    prot.critic_optimizer.add_param_group(
        {"params": [prot.q1.route_feat_w, prot.q2.route_feat_w], "lr": a.head_term_lr})

    blind = float(np.mean([apps["blind_cap"] for apps in
                           ({"blind_cap": it.refs["blind_cap"]} for it in test)]))
    omni = float(np.mean([it.refs["omni_cap"] for it in test]))
    hist = []
    k = 0
    while k < a.sorties:
        it = train[int(rng.integers(len(train)))]
        m = MEMBERS[int(rng.integers(len(MEMBERS)))]
        steps = []
        episode(prot, it, m, idx_of[m], rng, collect=steps)
        k += EP_LEN
        for i, (obs, r, rew) in enumerate(steps):
            last = i == len(steps) - 1
            ns = {}
            if not last:
                ns = dict(steps[i + 1][0]); ns["active_truck"] = 0
                ns["allowed_destinations"] = {"protagonist": {0: list(range(it.nR))}}
            prot.replay_buffer.push(SMDPTransition(
                agent="protagonist", state=obs, action={0: r}, reward=rew, next_state=ns,
                done=last, elapsed_ticks=1, action_mask={"protagonist": {0: list(range(it.nR))}},
                info={}))
        for _ in range(EP_LEN):
            prot.update(32)
        if k % a.eval_every < EP_LEN:
            erng = np.random.default_rng(7000 + k)
            told_true = float(np.mean([eval_told(prot, it, lambda m: idx_of[m], erng) / it.refs["blind_cap"]
                                       for it in test]))
            told_llm = float(np.mean([eval_told(prot, it, lambda m: idx_of[llm_map[m]], erng) / it.refs["blind_cap"]
                                      for it in test]))
            hist.append(dict(sortie=k, told_true_ratio_blind=told_true, told_llm_ratio_blind=told_llm))
            if a.ckpt_dir:
                Path(a.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(Path(a.ckpt_dir) / f"actor_ep{k}.pt"))
            print(f"  sortie {k:6d}: told-TRUE {told_true:.3f}x blind | told-LLM {told_llm:.3f}x "
                  f"blind (blind {blind:.4f} omni {omni:.4f}; omni/blind {omni/blind:.3f}) "
                  f"a{prot.alpha:.2f}", flush=True)
    sel = min(hist, key=lambda h: h["told_true_ratio_blind"]) if hist else None
    print(f"\n=== gen38 V2 (seed {a.seed}) ===")
    if sel:
        print(f"  best told-TRUE {sel['told_true_ratio_blind']:.3f}x blind @ {sel['sortie']} | "
              f"told-LLM at that ckpt {sel['told_llm_ratio_blind']:.3f}x blind "
              f"(< 1.0 crosses the wall; gen34 blind generalist was 1.373x)")
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(
            dict(seed=a.seed, blind=blind, omni=omni, llm_map=llm_map, history=hist,
                 select_on_told_true=sel), indent=2))
        print(f"[written] {a.json_out}")


if __name__ == "__main__":
    main()
