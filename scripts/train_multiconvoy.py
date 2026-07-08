#!/usr/bin/env python3
"""M3: train SACRED on the MULTI-CONVOY interdiction game (gen08 Phase M).

Each sortie is an N-step episode (route convoy 0 -> 1 -> ... -> N-1, terminal reward = -mission
failure), so the SAC defender's credit propagates across the fleet's joint decision and it can learn
the correlated optimum (the env exposes earlier convoys' routes via truck positions). Interdictor =
the oracle best-response to the defender's empirical OCCUPANCY play (fictitious play). Arms: vanilla
(no adversary, nominal travel objective) and sacred. Evaluated by EXPLOITABILITY = mission-failure of
the policy's occupancy distribution under the best-response interdictor, vs the M2 classical ladder
(shortest-path, ALNS) and the oracle (loss_det, loss_mixed).

Run: PYTHONPATH=. python scripts/train_multiconvoy.py --sorties 3000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.agents.networks import featurize_state
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.multiconvoy_oracle import best_response_attacker_multi, solve_multiconvoy
from src.baselines.multiconvoy_planners import classical_baselines
from src.env.smdp_wrapper import SMDPTransition
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

TAP_K = 5


def hop_probs(prot, obs, ci, allowed):
    pyg = featurize_state(obs, ci).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    node_ids = list(obs["nodes"].keys()); n2i = {n: i for i, n in enumerate(node_ids)}
    active = n2i[obs["trucks"][ci]["current_node"]]; mask_idx = [n2i[n] for n in allowed]
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, active, mask_idx)
    prot.actor.train()
    return {allowed[i]: float(probs[i]) for i in range(len(allowed))}


def route_one(prot, env, deterministic=False):
    """Route all N convoys sequentially; return (steps, occupancy, route_indices)."""
    env_routes = []
    steps = []
    for _ in range(env.config.N):
        ci = env.current_convoy(); obs = env.observe(); mask = env.defender_action_mask()
        act = prot.select_action(obs, mask, deterministic=deterministic)
        hop = act[ci]; ri = env.route_of_first_hop(hop)
        env.route_convoy_first_hop(hop)
        steps.append((obs, ci, hop, mask)); env_routes.append(ri)
    return steps, env.defender_occupancy(), env_routes


def _transition(obs, ci, hop, mask, reward, nobs, nci, nmask, done):
    nstate = {}
    if not done:
        nstate = dict(nobs); nstate["active_truck"] = nci
        nstate["allowed_destinations"] = {"protagonist": {nci: list(nmask[nci])}}
    return SMDPTransition(agent="protagonist", state=obs, action={ci: hop}, reward=reward,
                          next_state=nstate, done=done, elapsed_ticks=1,
                          action_mask={"protagonist": mask}, info={})


def policy_occ_dist(prot, env, samples=400):
    routes = []
    for _ in range(samples):
        env.reset()
        _, _, rs = route_one(prot, env, deterministic=False)
        routes.append(rs)
    return env.occ_dist(routes) if hasattr(env, "occ_dist") else env.occupancy_dist_of(routes)


def train_defender(env, *, sorties, seed, adversarial, switch_every, batch_size, eval_every,
                   attacker_mode, sol, baselines, interception_loss, mean_cost, reward_scale, verbose):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    prot = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=reward_scale, lr_actor=3e-4, autotune_alpha=True,
                          alpha_init=1.0, device="cpu")
    n_occ = len(env.occupancies)
    played = np.zeros(n_occ)
    committed = None
    pol_hist = []
    hist = []
    t_chunk = time.time()
    for k in range(sorties):
        if adversarial and (committed is None or k % switch_every == 0):
            occ_dist = played / played.sum() if played.sum() > 0 else np.ones(n_occ) / n_occ
            if attacker_mode == "smooth":
                e = occ_dist @ env.obj_matrix
                z = np.exp((e - e.max()) / 0.05); committed = int(rng.choice(len(z), p=z / z.sum()))
            else:
                committed, _ = best_response_attacker_multi(env.obj_matrix, occ_dist)
        env.reset()
        if adversarial:
            env.commit(committed)
        steps, occ, _ = route_one(prot, env)
        played[env._occ_index[tuple(occ)]] += 1.0
        if adversarial:
            reward = env.resolve().defender_reward                     # -interception_loss * mission
        else:
            travel = float(sum(env.game.travel_cost[r] for r in _route_of(steps, env)))
            reward = -interception_loss * (travel / (env.config.N * mean_cost))   # nominal travel
        N = env.config.N
        for i, (obs, ci, hop, mask) in enumerate(steps):
            last = i == N - 1
            nobs, nci, nmask = (steps[i + 1][0], steps[i + 1][1], steps[i + 1][3]) if not last else (None, None, None)
            prot.replay_buffer.push(_transition(obs, ci, hop, mask, reward if last else 0.0, nobs, nci, nmask, last))
        prot.update(batch_size)
        if eval_every and (k + 1) % eval_every == 0:
            t_train = time.time() - t_chunk
            t_ev = time.time()
            d = policy_occ_dist(prot, env, samples=400)
            pol_hist.append(d)
            _, expl = best_response_attacker_multi(env.obj_matrix, d)
            _, expl_tap = best_response_attacker_multi(env.obj_matrix, np.mean(pol_hist[-TAP_K:], axis=0))
            nz = d[d > 0]; h = float(-(nz * np.log(nz)).sum())
            t_eval = time.time() - t_ev
            hist.append((k + 1, expl, expl_tap, float(prot.alpha), h, t_train, t_eval))
            if verbose:
                print(f"    sortie {k+1:5d}: expl {expl:.3f} | TAP {expl_tap:.3f} | alpha {prot.alpha:.2f} "
                      f"H(occ) {h:.2f} | train {t_train:5.1f}s ({t_train/eval_every*1000:.0f}ms/sortie) "
                      f"eval {t_eval:4.1f}s   (loss_mixed={sol.loss_mixed:.3f}, ALNS={baselines['alns']:.3f})",
                      flush=True)
            t_chunk = time.time()
    d = policy_occ_dist(prot, env, samples=800)
    pol_hist.append(d)
    _, expl_tap = best_response_attacker_multi(env.obj_matrix, np.mean(pol_hist[-TAP_K:], axis=0))
    _, expl = best_response_attacker_multi(env.obj_matrix, d)
    return {"expl": expl, "expl_tap": expl_tap, "history": hist,
            "occ_dist": d.tolist()}


def _route_of(steps, env):
    return [env.route_of_first_hop(hop) for (_, _, hop, _) in steps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--od", default="110-135"); p.add_argument("--N", type=int, default=3)
    p.add_argument("--K", type=int, default=1); p.add_argument("--sorties", type=int, default=3000)
    p.add_argument("--switch-every", type=int, default=50); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=250); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--attacker-mode", default="latest", choices=("latest", "smooth"))
    p.add_argument("--band", default="0.15,0.95"); p.add_argument("--json-out", default="")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (use 3 for 3-parallel)")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    s, t = args.od.split("-"); band = tuple(float(x) for x in args.band.split(","))
    env = make_multiconvoy_env(od=(s, t), N=args.N, K=args.K, edge_vuln_band=band,
                               interception_loss=args.interception_loss, objective="mission", seed=args.seed)
    sol = solve_multiconvoy(env.game, args.N, "mission")
    baselines = classical_baselines(env.game, args.N, "mission")
    mean_cost = float(env.game.travel_cost.mean())
    print(f"Multi-convoy {s}->{t} N={args.N} K={args.K}: {env.game.n_routes} routes, "
          f"{len(env.occupancies)} occupancies. Ladder: shortest_path {baselines['shortest_path']:.3f} > "
          f"ALNS {baselines['alns']:.3f} (= loss_det {sol.loss_det:.3f}) >> loss_mixed {sol.loss_mixed:.3f}\n")

    common = dict(sorties=args.sorties, seed=args.seed, switch_every=args.switch_every,
                  batch_size=args.batch_size, eval_every=args.eval_every, sol=sol, baselines=baselines,
                  interception_loss=args.interception_loss, mean_cost=mean_cost,
                  reward_scale=args.reward_scale, verbose=True)
    print("[vanilla] training (nominal travel objective, no adversary)...")
    v = train_defender(env, adversarial=False, attacker_mode=args.attacker_mode, **common)
    print(f"[vanilla] expl {v['expl']:.3f} | TAP {v['expl_tap']:.3f}\n")
    print(f"[sacred] training vs the oracle best-response interdictor ({args.attacker_mode} FP)...")
    sac = train_defender(env, adversarial=True, attacker_mode=args.attacker_mode, **common)
    print(f"\n=== RESULT ({s}->{t}, N={args.N}, K={args.K}, attacker={args.attacker_mode}, seed={args.seed}) ===")
    print(f"  shortest_path   {baselines['shortest_path']:.3f}")
    print(f"  ALNS            {baselines['alns']:.3f}   (= optimal deterministic {sol.loss_det:.3f})")
    print(f"  vanilla         {v['expl_tap']:.3f} (TAP) / {v['expl']:.3f} (policy)")
    print(f"  sacred          {sac['expl_tap']:.3f} (TAP) / {sac['expl']:.3f} (policy)")
    print(f"  equilibrium     {sol.loss_mixed:.3f}   -> sacred TAP distance {abs(sac['expl_tap']-sol.loss_mixed):.3f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"od": args.od, "N": args.N, "K": args.K, "seed": args.seed, "attacker_mode": args.attacker_mode,
             "loss_det": sol.loss_det, "loss_mixed": sol.loss_mixed, "baselines": {k: baselines[k] for k in ("shortest_path", "alns")},
             "vanilla": v, "sacred": sac}, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
