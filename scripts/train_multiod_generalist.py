#!/usr/bin/env python3
"""gen29: the multi-OD three-stream coordination GENERALIST (GEN29_MULTIOD_HANDOFF.md §3).

The gen16/v3.1 recipe on the sequential three-stream game: one policy routes streams 0,1,2 in
turn, conditioning on earlier committed routes via the OVERLAP head feature (undiluted, dedicated
lr) + taken_node_frac; per-instance smooth FP over joint plays; EXACT joint distribution by
conditional enumeration (no Monte Carlo); validation-set checkpoint selection. A --blind arm
zeroes the coordination channel (the causal control: must land ~ the best independent product).

Refs-only probe (anchors + untrained-context row): --sorties 0
Run (per seed): PYTHONPATH=. python scripts/train_multiod_generalist.py --sorties 14000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path as _P

import numpy as np
import torch

from scratch.b4_multiod_probe import build_graph
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.agents.transition_builder import SMDPTransition
from src.baselines.fp_dynamics import sample_smooth_iset, smooth_fp_probs
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiod_interdiction import MultiODInterdictionEnv, MultiODConfig
from src.env.graph_env import GraphEnv
from src.utils.graph_utils import load_osm_graph_and_demands
def make_env(s, targets, K=1):
    from src.envs.multiod_interdiction import make_multiod_env
    return make_multiod_env(s, targets, K=K)


class Inst:
    def __init__(self, spec, blind=False):
        self.name = f"{spec['s']}->{','.join(spec['targets'])}"
        self.env = make_env(spec["s"], tuple(spec["targets"]), K=spec.get("K", 1))
        self.env.blind = blind
        self.eq = float(spec["eq"]); self.cap = float(spec["cap"])
        self.indep = float(spec["indep"]); self.det = float(spec["det"])
        self.occ_seq: list[int] = []
        self.pol_hist: list[np.ndarray] = []


def stream_probs(prot, env, prefix):
    """Policy's conditional route distribution for the active stream given a committed prefix
    (the prefix sets the overlap feature + taken_node_frac)."""
    env.set_committed(list(prefix))
    obs = env.observe()
    cur = env.current_stream()
    prot.actor.menu_routes = obs["menu_route_node_idx"]
    prot.actor.route_feats = obs["menu_route_feats"]
    pyg = featurize_state(obs, cur).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    active = n2i[obs["trucks"][cur]["current_node"]]
    R = len(env.route_sets[cur])
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, active, list(range(R)), None)
    prot.actor.train()
    p = probs.cpu().numpy()
    sl = getattr(env, "shortlist", None)               # gen37: renormalise over allowed routes
    if sl is not None:
        pref = tuple(int(x) for x in prefix)
        allowed = sorted({t[len(pref)] for t in sl if tuple(t[:len(pref)]) == pref})
        if allowed:
            mask = np.zeros_like(p); mask[allowed] = 1.0
            p = p * mask
            p = p / p.sum() if p.sum() > 1e-12 else mask / mask.sum()
    return p


def joint_dist(prot, env) -> np.ndarray:
    """EXACT joint route-tuple distribution by conditional enumeration (~1 + R1 + R1R2 forwards).
    gen37: when env.shortlist is set, stream_probs renormalises over the shortlist-allowed
    routes per prefix, so this is the exact distribution of the MASKED (deployed) policy."""
    R = [len(rs) for rs in env.route_sets]
    d1 = stream_probs(prot, env, [])
    dist = np.zeros(env.n_joint)
    for r1 in range(R[0]):
        if d1[r1] < 1e-6:
            continue
        d2 = stream_probs(prot, env, [r1])
        for r2 in range(R[1]):
            p12 = d1[r1] * d2[r2]
            if p12 < 1e-6:
                continue
            d3 = stream_probs(prot, env, [r1, r2])
            base = (r1 * R[1] + r2) * R[2]
            dist[base:base + R[2]] += p12 * d3
    return dist


def exploit_ratio(prot, inst):
    d = joint_dist(prot, inst.env)
    return inst.env.exploitability_of_joint_dist(d) / inst.eq, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sorties", type=int, default=14000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp-tau", type=float, default=0.05)
    p.add_argument("--smooth-window", type=int, default=250)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--ent-frac", type=float, default=0.5)
    p.add_argument("--alpha-floor", type=float, default=0.20)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--blind", action="store_true", help="causal control: zero coordination channel")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--screen", default="models/runs/gen29_screen.json")
    p.add_argument("--shortlist", default="", help="gen37: JSON {inst.name: [[r0,r1,r2],...]} "
                   "restricting each instance's joint action space to the shortlist")
    p.add_argument("--json-out", default="")
    p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed); np.random.seed(args.seed); rng = np.random.default_rng(args.seed)

    sc = json.load(open(args.screen))
    print(f"[gen29{'/blind' if args.blind else ''}] building pool...", flush=True)
    t0 = time.time()
    train = [Inst(c, blind=args.blind) for c in ([sc["headline"]] + sc["pool"])]
    held = [Inst(c, blind=args.blind) for c in sc["held_out"]]
    val = [Inst(c, blind=args.blind) for c in sc["validation"]]
    if args.shortlist:                                   # gen37: attach per-instance shortlist S
        sl = json.load(open(args.shortlist))
        miss = 0
        for it in train + held + val:
            S = sl.get(it.name)
            if S is None:
                miss += 1
            else:
                it.env.shortlist = [tuple(int(x) for x in t) for t in S]
        print(f"[gen37] shortlist attached ({args.shortlist}); {miss} instances unmatched "
              f"(full-space)", flush=True)
    print(f"[gen29] pool {len(train)} train / {len(held)} held-out / {len(val)} val "
          f"in {time.time()-t0:.0f}s", flush=True)
    for it in [train[0]] + held:
        print(f"    {it.name}: eq={it.eq:.3f} cap={it.cap:.3f} indep={it.indep:.3f}", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3, alpha_floor=args.alpha_floor)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.route_feat_w = torch.nn.Parameter(torch.zeros(2))   # [worst-vuln, OVERLAP]
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w], "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.route_feat_w, prot.q2.route_feat_w],
                                           "lr": args.head_term_lr})

    def eval_rows(insts):
        return {it.name: exploit_ratio(prot, it) for it in insts}

    if args.sorties == 0:
        tr = {it.name: exploit_ratio(prot, it)[0] for it in train}
        te = {it.name: exploit_ratio(prot, it)[0] for it in held}
        beats = sum(1 for it in held if exploit_ratio(prot, it)[0] * it.eq < it.cap)
        print(f"[gen29] UNTRAINED: train ratio {np.mean(list(tr.values())):.2f} | held-out "
              f"{np.mean(list(te.values())):.2f} beats-cap {beats}/{len(held)}", flush=True)
        return

    hist = []
    t0 = time.time()
    for k in range(args.sorties):
        inst = train[int(rng.integers(len(train)))]
        env = inst.env
        probs = smooth_fp_probs(inst.occ_seq, env.n_joint, env.obj_matrix, args.fp_tau, args.smooth_window)
        j = sample_smooth_iset(probs, rng)
        env.reset(); env.commit(j)
        steps, routes = [], []
        for f in range(env.F):
            obs = env.observe(); cur = env.current_stream(); mask = env.defender_action_mask()
            act = prot.select_action(obs, mask)[cur]
            env.route_stream_by_index(int(act)); steps.append((obs, cur, int(act), mask)); routes.append(int(act))
        inst.occ_seq.append(env.joint_index(routes))
        reward = -args.interception_loss * env.mission_failure(routes, j)
        te = args.ent_frac * math.log(max(2, len(env.route_sets[0])))
        for i, (obs, cur, act, mask) in enumerate(steps):
            last = i == env.F - 1
            obs["target_entropy"] = te; obs["alpha_group"] = 0     # all streams mix equally
            if last:
                nstate = {}
            else:
                nobs, ncur, nmask = steps[i + 1][0], steps[i + 1][1], steps[i + 1][3]
                nstate = dict(nobs); nstate["active_truck"] = ncur
                nstate["allowed_destinations"] = {"protagonist": {ncur: list(nmask[ncur])}}
            prot.replay_buffer.push(SMDPTransition(
                agent="protagonist", state=obs, action={cur: act}, reward=reward if last else 0.0,
                next_state=nstate, done=bool(last), elapsed_ticks=1,
                action_mask={"protagonist": mask}, info={}))
        prot.update(args.batch_size)

        if (k + 1) % args.eval_every == 0:
            for grp in (train, held, val):
                for it in grp:
                    it.pol_hist.append(exploit_ratio(prot, it)[1])
            def tap(insts):
                out = {}
                for it in insts:
                    d = np.mean(it.pol_hist[-3:], axis=0)
                    out[it.name] = it.env.exploitability_of_joint_dist(d)
                return out
            trv = tap(train); hev = tap(held); vav = tap(val)
            tr_m = float(np.mean([trv[it.name] / it.eq for it in train]))
            va_m = float(np.mean([vav[it.name] / it.eq for it in val]))
            he_m = float(np.mean([hev[it.name] / it.eq for it in held]))
            beats = sum(1 for it in held if hev[it.name] < it.cap)
            fw = tuple(float(x) for x in prot.actor.route_feat_w.detach())
            hist.append((k + 1, tr_m, va_m, he_m, beats, trv, hev, vav, fw, float(prot.alpha)))
            if args.ckpt_dir:
                _P(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(_P(args.ckpt_dir) / f"actor_ep{k+1}.pt"))
            print(f"  sortie {k+1:6d}: TRAIN {tr_m:.2f} VAL {va_m:.2f} | HELD-OUT {he_m:.2f} "
                  f"beats-cap {beats}/{len(held)} | rw[{fw[0]:.2f},{fw[1]:.2f}] a{prot.alpha:.2f} | "
                  f"{time.time()-t0:5.0f}s", flush=True)

    if args.json_out:
        refs = {it.name: {"eq": it.eq, "cap": it.cap, "indep": it.indep, "det": it.det}
                for it in train + held + val}
        _P(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _P(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "blind": args.blind, "refs": refs, "history": hist}, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
