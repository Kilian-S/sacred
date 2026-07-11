#!/usr/bin/env python3
"""C1 (gen23): ERB bootstrapping via a POPULATION-BASED METAHEURISTIC, done literally (Obj-3's
verbatim wording "investigate the efficacy of ERB bootstrapping via population-based metaheuristics
to accelerate training convergence"). gen01 left this inconclusive at n=1 on the campaign problem;
this closes the wording on the post-fix interdiction headline.

Seed the fleet-route defender's replay buffer with DEMONSTRATION transitions from a POPULATION of
ALNS plans (the metaheuristic run at several restart seeds -> a population of low-vulnerability route
choices; ALNS minimises worst-case mission-failure, so its favoured routes are the equilibrium's
low-vuln support). Arms {seeded, cold} x seeds; PRE-REGISTERED time-to-competence primary (sorties
to first reach a TAP bar) + final best-checkpoint parity. Otherwise the gen14 fleet-route config.

Run: PYTHONPATH=. python scripts/train_c1.py --seed 0            # cold
     PYTHONPATH=. python scripts/train_c1.py --seed 0 --erb      # seeded
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.train_multiconvoy import (_transition, exact_fleet_occ_dist, route_one)
from src.agents.sac import ProtagonistSAC
from src.baselines.fp_dynamics import sample_smooth_iset, smooth_fp_probs
from src.baselines.multiconvoy_oracle import best_response_attacker_multi, objective_value, solve_multiconvoy
from src.baselines.multiconvoy_planners import alns_fleet_planner, classical_baselines
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

TAP_K = 5


def alns_erb_routes(env, N, n_plans=8):
    """A POPULATION of ALNS plans (restart seeds) -> the set of routes they favour (demos)."""
    routes = set()
    for seed in range(n_plans):
        plan = alns_fleet_planner(env.game, N, "mission", seed=seed)
        for r in plan.assignment:
            routes.add(int(r))
    return sorted(routes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--od", default="35-159"); p.add_argument("--N", type=int, default=3)
    p.add_argument("--K", type=int, default=1); p.add_argument("--k-extra", type=int, default=8)
    p.add_argument("--band", default="0.15,0.95"); p.add_argument("--sorties", type=int, default=1200)
    p.add_argument("--eval-every", type=int, default=100); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--fp-tau", type=float, default=0.05); p.add_argument("--smooth-window", type=int, default=250)
    p.add_argument("--switch-every", type=int, default=200)
    p.add_argument("--leader-ent-frac", type=float, default=0.5); p.add_argument("--leader-alpha-floor", type=float, default=0.20)
    p.add_argument("--erb", action="store_true", help="seed the buffer with ALNS-population demos")
    p.add_argument("--erb-copies", type=int, default=200, help="demonstration transitions to seed")
    p.add_argument("--competence-bar", type=float, default=0.35, help="TAP bar for time-to-competence")
    p.add_argument("--threads", type=int, default=4); p.add_argument("--json-out", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    s, t = args.od.split("-"); band = tuple(float(x) for x in args.band.split(","))
    env = make_multiconvoy_env(od=(s, t), N=args.N, K=args.K, k_extra_routes=args.k_extra,
                               menu_select=True, edge_vuln_band=band,
                               interception_loss=args.interception_loss, seed=args.seed)
    R = env.game.n_routes; N = args.N
    sol = solve_multiconvoy(env.game, N, "mission"); base = classical_baselines(env.game, N, "mission")

    dfn = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                         reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                         device="cpu", role_alpha=True, lr_alpha=5e-3, alpha_floor=args.leader_alpha_floor)
    dmenu = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]
    for net in (dfn.actor, dfn.q1, dfn.q2, dfn.target_q1, dfn.target_q2):
        net.menu_routes = dmenu
    leader_te = args.leader_ent_frac * math.log(R)

    # --- ERB seeding: demonstrations of stacking on ALNS-population-favoured routes ---
    if args.erb:
        erb_routes = alns_erb_routes(env, N)
        eq_attacker = sol.attacker_strategy  # score demos vs the equilibrium attacker
        for _ in range(args.erb_copies):
            r = int(rng.choice(erb_routes))
            env.reset(); obs = env.observe(); mask = env.defender_action_mask()
            occ = tuple(N if i == r else 0 for i in range(R))
            mission = float(env.obj_matrix[env._occ_index[occ]] @ eq_attacker)
            reward = -args.interception_loss * mission
            obs["target_entropy"] = leader_te; obs["alpha_group"] = 0
            dfn.replay_buffer.push(_transition(obs, 0, r, mask, reward, None, None, None, True))
        print(f"[C1] seeded {args.erb_copies} demos from {len(erb_routes)} ALNS-population routes "
              f"{erb_routes}", flush=True)

    n_occ = len(env.occupancies); occ_seq = []; smooth_probs = None; committed = None; pol_hist = []
    hist = []; competence_at = None
    for k in range(args.sorties):
        if committed is None or k % args.switch_every == 0:
            smooth_probs = smooth_fp_probs(occ_seq, n_occ, env.obj_matrix, args.fp_tau, args.smooth_window)
        env.reset(); committed = sample_smooth_iset(smooth_probs, rng); env.commit(committed)
        steps, occ, _ = route_one(dfn, env, fleet_route=True)
        occ_seq.append(env._occ_index[tuple(occ)])
        p_ = env.game.payoff[:, committed]
        reward = -args.interception_loss * objective_value(np.asarray(occ), p_, N, "mission")
        for obs_j, ci_j, _, _ in steps:
            obs_j["target_entropy"] = leader_te if ci_j == 0 else 0.05 * math.log(R)
            obs_j["alpha_group"] = 0 if ci_j == 0 else 1
        for i, (obs, ci, hop, mask) in enumerate(steps):
            last = i == N - 1
            nobs, nci, nmask = ((steps[i+1][0], steps[i+1][1], steps[i+1][3]) if not last else (None, None, None))
            dfn.replay_buffer.push(_transition(obs, ci, hop, mask, reward if last else 0.0, nobs, nci, nmask, last))
        dfn.update(args.batch_size)
        if (k + 1) % args.eval_every == 0:
            d, _ = exact_fleet_occ_dist(dfn, env); pol_hist.append(d)
            _, tap = best_response_attacker_multi(env.obj_matrix, np.mean(pol_hist[-TAP_K:], axis=0))
            hist.append((k + 1, float(tap)))
            if competence_at is None and tap <= args.competence_bar:
                competence_at = k + 1
            print(f"  sortie {k+1:5d}: TAP {tap:.3f} (bar {args.competence_bar}, eq {sol.loss_mixed:.3f})",
                  flush=True)
    best = min((h[1] for h in hist), default=float("nan"))
    print(f"\n=== C1 ({args.od}, seed {args.seed}, {'SEEDED' if args.erb else 'COLD'}) ===")
    print(f"  time-to-competence (TAP <= {args.competence_bar}): "
          f"{competence_at if competence_at else 'never'} sorties | best-ckpt TAP {best:.3f}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"od": args.od, "seed": args.seed, "erb": args.erb, "competence_at": competence_at,
             "best_tap": best, "bar": args.competence_bar, "eq": sol.loss_mixed, "history": hist}, indent=2))


if __name__ == "__main__":
    main()
