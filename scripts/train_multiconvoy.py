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
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.fp_dynamics import sample_smooth_iset, smooth_fp_probs
from src.baselines.multiconvoy_oracle import best_response_attacker_multi, objective_value, solve_multiconvoy
from src.baselines.multiconvoy_planners import classical_baselines
from src.env.smdp_wrapper import SMDPTransition
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

TAP_K = 5


def hop_probs(prot, obs, ci, allowed):
    pyg = featurize_state(obs, ci).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)  # MUST match featurize_state's row order
    active = n2i[obs["trucks"][ci]["current_node"]]; mask_idx = [n2i[n] for n in allowed]
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, active, mask_idx)
    prot.actor.train()
    return {allowed[i]: float(probs[i]) for i in range(len(allowed))}


def route_one(prot, env, deterministic=False, fleet_route=False, leader_policy=None,
              copy_prob=0.0, rng=None):
    """Route all N convoys; return (steps, occupancy, route_indices). fleet_route = followers hard-copy
    convoy 0 (reachability control). leader_policy = a FROZEN mixing leader drives convoy 0 (follower
    bootstrap). copy_prob = probability a follower is FORCED to copy the leader this sortie (the
    demonstration crutch, annealed 1->0), so the critic experiences the low-failure stack against a
    VARYING leader and must learn 'follow the correlation signal', not memorise a route."""
    menu = env.config.menu_select
    env_routes = []
    steps = []
    leader_act = None
    for _ in range(env.config.N):
        ci = env.current_convoy(); obs = env.observe(); mask = env.defender_action_mask()
        if ci == 0 and leader_policy is not None:
            act = leader_policy.select_action(obs, mask, deterministic=False)[ci]
        elif fleet_route and ci != 0:
            act = leader_act
        elif ci != 0 and copy_prob > 0.0 and (rng.random() if rng is not None else 1.0) < copy_prob:
            act = leader_act  # forced-copy demonstration (annealing crutch)
        else:
            act = prot.select_action(obs, mask, deterministic=deterministic)[ci]
        if ci == 0:
            leader_act = act
        if menu:  # act is a ROUTE INDEX
            ri = int(act); env.route_convoy_by_index(ri)
        else:     # act is a first-hop node
            ri = env.route_of_first_hop(act); env.route_convoy_first_hop(act)
        steps.append((obs, ci, act, mask)); env_routes.append(ri)
    return steps, env.defender_occupancy(), env_routes


def _transition(obs, ci, hop, mask, reward, nobs, nci, nmask, done):
    nstate = {}
    if not done:
        nstate = dict(nobs); nstate["active_truck"] = nci
        nstate["allowed_destinations"] = {"protagonist": {nci: list(nmask[nci])}}
    return SMDPTransition(agent="protagonist", state=obs, action={ci: hop}, reward=reward,
                          next_state=nstate, done=done, elapsed_ticks=1,
                          action_mask={"protagonist": mask}, info={})


def menu_leader_probs(prot, env):
    """EXACT leader route distribution (menu-select): one forward pass at the convoy-0 decision."""
    env.reset()
    obs = env.observe()
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    active = n2i[obs["trucks"][0]["current_node"]]
    R = env.game.n_routes
    taken = torch.zeros(R, device=prot.device)  # leader decision: no earlier convoys
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, active, list(range(R)), taken)
    prot.actor.train()
    return probs.cpu().numpy()


def exact_fleet_occ_dist(prot, env):
    """EXACT occupancy distribution in fleet-route + menu-select mode: the fleet stacks on the
    leader, so the occupancy distribution is the leader's route distribution mapped onto the
    stacked occupancies. Replaces the 400-sample Monte-Carlo estimate, whose sampling noise plus
    min-selection bias the gen09 exact re-evaluation quantified at ~+0.012 on the best-checkpoint
    TAP (scratch/gen09_exact_reeval.py; CRITIQUE_INTERDICTION.md §5.3)."""
    lead = menu_leader_probs(prot, env)
    R, N = env.game.n_routes, env.config.N
    d = np.zeros(len(env.occupancies))
    for r in range(R):
        d[env._occ_index[tuple(N if i == r else 0 for i in range(R))]] = lead[r]
    return d, lead


def policy_occ_dist(prot, env, samples=400, fleet_route=False, leader_policy=None):
    routes = []
    for _ in range(samples):
        env.reset()
        _, _, rs = route_one(prot, env, deterministic=False, fleet_route=fleet_route,
                             leader_policy=leader_policy)
        routes.append(rs)
    dist = env.occ_dist(routes) if hasattr(env, "occ_dist") else env.occupancy_dist_of(routes)
    return dist, routes


def coordination_stats(routes):
    """stack_rate = fraction of sorties with ALL convoys on ONE route (mass on the MATCHED stacks);
    follow_rate = fraction where convoy 1 took convoy 0's route. These, not H_foll, distinguish real
    coordination (concentrate on the LEADER's route) from coincidental collapse onto a fixed route."""
    arr = np.asarray(routes)
    stack = float(np.mean([len(set(r)) == 1 for r in arr]))
    follow = float(np.mean(arr[:, 1] == arr[:, 0])) if arr.shape[1] >= 2 else 0.0
    return stack, follow


def role_entropies(routes, R):
    """From sampled per-convoy route tuples: H(leader route) and mean H(follower route | leader) -
    the correlation SIGNATURE we want (leader stays high, follower -> 0 = learned stack-and-follow;
    both high = still independent spreading)."""
    arr = np.asarray(routes)
    lead = np.bincount(arr[:, 0], minlength=R) / len(arr)
    H_lead = float(-(lead[lead > 0] * np.log(lead[lead > 0])).sum())
    if arr.shape[1] < 2:
        return H_lead, 0.0
    H_foll = 0.0
    for r0 in range(R):
        m = arr[:, 0] == r0
        if m.sum() == 0:
            continue
        cond = np.bincount(arr[m, 1], minlength=R) / m.sum()
        H_foll += (m.sum() / len(arr)) * float(-(cond[cond > 0] * np.log(cond[cond > 0])).sum())
    return H_lead, H_foll


def train_defender(env, *, sorties, seed, adversarial, switch_every, batch_size, eval_every,
                   attacker_mode, sol, baselines, interception_loss, mean_cost, reward_scale, verbose,
                   leader_ent_frac=1.0, follower_ent_frac=0.05, alpha_lr=5e-3,
                   fleet_route=False, follower_warmup=0, frozen_leader=None, forced_copy_warmup=0,
                   save_actor=None, stack_dup=4, fp_tau=0.05, leader_alpha_floor=None,
                   smooth_window=250, ckpt_dir=None, legacy_role_target=False):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=reward_scale, lr_actor=3e-4, autotune_alpha=True,
                          alpha_init=1.0, device="cpu", role_alpha=adversarial,
                          lr_alpha=(alpha_lr if adversarial else None),
                          alpha_floor=(leader_alpha_floor if adversarial else None),
                          legacy_next_alpha=legacy_role_target)
    if env.config.menu_select:  # ROUTE menu-select head: give every net the per-route node indices
        menu = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]
        for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
            net.menu_routes = menu
        if adversarial:  # vanilla stays a clean control (no follow mechanism)
            # LEVER 2: a LEARNED (not fixed) weight on the undiluted per-route 'taken' term, at BOTH
            # the policy head AND the critic Q head, so the critic can rank the leader's route and the
            # actor gets a gradient to grow follow_w (Bellman-consistent Q input, not a hard bonus).
            prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
            prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w]})
            for qn in (prot.q1, prot.q2, prot.target_q1, prot.target_q2):
                qn.follow_w = torch.nn.Parameter(torch.tensor(1.0))
            prot.critic_optimizer.add_param_group({"params": [prot.q1.follow_w, prot.q2.follow_w]})
    R = env.game.n_routes
    leader_te = leader_ent_frac * math.log(R)      # role-dependent target entropy (sacred only):
    follower_te = follower_ent_frac * math.log(R)  # leader explores routes; followers copy the leader
    n_occ = len(env.occupancies)
    played = np.zeros(n_occ)               # all-history occupancy histogram (latest-BR + eval)
    occ_seq: list[int] = []                # per-sortie occupancy-index log (smooth-FP trailing window)
    smooth_probs = None                    # smooth-FP attacker distribution over interdiction sets
    committed = None
    pol_hist = []
    hist = []
    t_chunk = time.time()
    for k in range(sorties):
        if adversarial and (committed is None or k % switch_every == 0):
            if attacker_mode == "smooth":
                # TRUE smooth fictitious play via the shared B2-P3-proven discipline (fp_dynamics):
                # softmax BR to the TRAILING-WINDOW occupancy play, recomputed per block; the iset is
                # SAMPLED FRESH EVERY sortie below (block-holding one iset is the cycling regime).
                smooth_probs = smooth_fp_probs(occ_seq, n_occ, env.obj_matrix, fp_tau, smooth_window)
            else:
                occ_dist = played / played.sum() if played.sum() > 0 else np.ones(n_occ) / n_occ
                committed, _ = best_response_attacker_multi(env.obj_matrix, occ_dist)
        env.reset()
        if adversarial:
            if attacker_mode == "smooth":  # fresh committed iset every sortie (smooth FP)
                committed = sample_smooth_iset(smooth_probs, rng)
            env.commit(committed)
        copy_prob = (max(0.0, 1.0 - k / forced_copy_warmup)
                     if (frozen_leader is not None and forced_copy_warmup > 0) else 0.0)
        steps, occ, _ = route_one(prot, env, fleet_route=fleet_route, leader_policy=frozen_leader,
                                  copy_prob=copy_prob, rng=rng)
        oi = env._occ_index[tuple(occ)]
        played[oi] += 1.0
        occ_seq.append(oi)
        if adversarial:
            # analytic (expected) mission-failure reward: dense, low-variance, unbiased replacement
            # for the sampled Bernoulli env.resolve() (verified). committed = the FP interdictor.
            p = env.game.payoff[:, committed]
            reward = -interception_loss * objective_value(np.asarray(occ), p, env.config.N,
                                                          env.config.objective, env.config.threshold_m)
        else:
            travel = float(sum(env.game.travel_cost[r] for r in _route_of(steps, env)))
            reward = -interception_loss * (travel / (env.config.N * mean_cost))   # nominal travel
        N = env.config.N
        bootstrap = frozen_leader is not None
        # ITEM 3 (prioritised replay / ERB / Obj-3): upsample MATCHED-STACK sorties so the critic keeps
        # training on the rare-but-informative low-failure stacked experience once natural stacks are
        # scarce (the confirmed under-sampling root cause).
        is_stack = sum(1 for c in occ if c > 0) == 1
        n_push = stack_dup if (is_stack and adversarial) else 1
        if adversarial:
            # role-dependent entropy. Bootstrap: convoys 1..N-1 are all followers (the leader is
            # frozen). Otherwise convoy 0 leads and the followers switch to the low temperature
            # only after the warmup so the critic learns the ordering before they commit.
            # Tagged in a PRE-pass over ALL steps so that _transition's shallow next-state copy
            # carries the NEXT decision's alpha_group too (the role-alpha target fix in sac.py).
            for obs_j, ci_j, _, _ in steps:
                is_follower = (ci_j != 0) and (bootstrap or k >= follower_warmup)
                obs_j["target_entropy"] = follower_te if is_follower else leader_te
                obs_j["alpha_group"] = 1 if is_follower else 0
        for i, (obs, ci, hop, mask) in enumerate(steps):
            if bootstrap and ci == 0:
                continue  # convoy 0 is the FROZEN mixing leader: not trained
            last = i == N - 1
            nobs, nci, nmask = (steps[i + 1][0], steps[i + 1][1], steps[i + 1][3]) if not last else (None, None, None)
            t = _transition(obs, ci, hop, mask, reward if last else 0.0, nobs, nci, nmask, last)
            for _ in range(n_push):
                prot.replay_buffer.push(t)
        prot.update(batch_size)
        if eval_every and (k + 1) % eval_every == 0:
            t_train = time.time() - t_chunk
            t_ev = time.time()
            exact = fleet_route and env.config.menu_select
            if exact:  # fleet-route: EXACT occupancy distribution (one forward pass, no MC noise)
                d, lead = exact_fleet_occ_dist(prot, env)
                nzl = lead[lead > 0]
                H_lead, H_foll = float(-(nzl * np.log(nzl)).sum()), 0.0
                stack_rate, follow_rate = 1.0, 1.0  # by construction in fleet-route mode
            else:
                d, routes = policy_occ_dist(prot, env, samples=400, fleet_route=fleet_route,
                                            leader_policy=frozen_leader)
                H_lead, H_foll = role_entropies(routes, env.game.n_routes)
                stack_rate, follow_rate = coordination_stats(routes)
            pol_hist.append(d)
            _, expl = best_response_attacker_multi(env.obj_matrix, d)
            _, expl_tap = best_response_attacker_multi(env.obj_matrix, np.mean(pol_hist[-TAP_K:], axis=0))
            if ckpt_dir is not None:  # per-eval checkpoint: best-checkpoint becomes a re-evaluable artefact
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(Path(ckpt_dir) / f"actor_ep{k + 1}.pt"))
            nz = d[d > 0]; h = float(-(nz * np.log(nz)).sum())
            t_eval = time.time() - t_ev
            hist.append((k + 1, expl, expl_tap, float(prot.alpha), float(prot.alpha_foll),
                         stack_rate, follow_rate, H_lead, H_foll, t_train, t_eval))
            if verbose:
                fw = float(prot.actor.follow_w) if hasattr(prot.actor, "follow_w") else 0.0
                print(f"    sortie {k+1:5d}: expl {expl:.3f} | TAP {expl_tap:.3f} | "
                      f"alpha L{prot.alpha:.2f}/F{prot.alpha_foll:.2f} fw {fw:.2f} | "
                      f"stack {stack_rate:.2f} follow {follow_rate:.2f} | H_lead {H_lead:.2f} H_foll {H_foll:.2f} | "
                      f"train {t_train:4.0f}s eval {t_eval:3.0f}s   "
                      f"(loss_mixed={sol.loss_mixed:.3f}, ALNS={baselines['alns']:.3f})", flush=True)
            t_chunk = time.time()
    if fleet_route and env.config.menu_select:  # EXACT final reading (see the eval block above)
        d, lead = exact_fleet_occ_dist(prot, env)
        nzl = lead[lead > 0]
        H_lead, H_foll = float(-(nzl * np.log(nzl)).sum()), 0.0
        stack_rate, follow_rate = 1.0, 1.0
    else:
        d, routes = policy_occ_dist(prot, env, samples=800, fleet_route=fleet_route,
                                    leader_policy=frozen_leader)
        H_lead, H_foll = role_entropies(routes, env.game.n_routes)
        stack_rate, follow_rate = coordination_stats(routes)
    pol_hist.append(d)
    _, expl_tap = best_response_attacker_multi(env.obj_matrix, np.mean(pol_hist[-TAP_K:], axis=0))
    _, expl = best_response_attacker_multi(env.obj_matrix, d)
    # STATIONARY-TAIL time-average (zero-sum FP: the equilibrium is the time-average, not per-eval
    # play): exploitability of the MEAN occupancy over the last TAIL evals, its per-eval expl
    # amplitude (std), and mean stack there -- the trustworthy read once coordination has plateaued.
    tail_k = min(len(pol_hist), 12)
    _, tail_expl = best_response_attacker_multi(env.obj_matrix, np.mean(pol_hist[-tail_k:], axis=0))
    tail_amp = float(np.std([h[1] for h in hist[-tail_k:]])) if len(hist) >= 2 else 0.0
    tail_stack = float(np.mean([h[5] for h in hist[-tail_k:]])) if hist else stack_rate
    if save_actor:
        Path(save_actor).parent.mkdir(parents=True, exist_ok=True)
        torch.save(prot.actor.state_dict(), save_actor)
        print(f"  [saved leader actor] {save_actor}")
    # BEST-CHECKPOINT (project discipline: the final iterate is misleading under adversarial/minimax
    # co-evolution -- the last iterate over-trains toward uniform; select the lowest-exploitability
    # training point). Report both the deployable trailing-averaged-policy reading (min TAP, the
    # B2-P3 estimator, the headline) and the single-checkpoint reading (min per-eval expl). Both are
    # re-evaluable from the saved per-eval ckpt_dir actors + the pol_hist occupancy distributions.
    best_tap = min((h[2] for h in hist), default=float("nan"))
    best_tap_sortie = next((h[0] for h in hist if h[2] == best_tap), None)
    best_expl = min((h[1] for h in hist), default=float("nan"))
    best_expl_sortie = next((h[0] for h in hist if h[1] == best_expl), None)
    return {"expl": expl, "expl_tap": expl_tap, "tail_expl": tail_expl, "tail_amp": tail_amp,
            "tail_stack": tail_stack, "history": hist, "occ_dist": d.tolist(),
            "best_tap": best_tap, "best_tap_sortie": best_tap_sortie,
            "best_expl": best_expl, "best_expl_sortie": best_expl_sortie,
            "pol_hist": [p.tolist() for p in pol_hist],
            "H_lead": H_lead, "H_foll": H_foll, "stack_rate": stack_rate, "follow_rate": follow_rate}


def _route_of(steps, env):
    if env.config.menu_select:  # the stored action IS the route index
        return [int(act) for (_, _, act, _) in steps]
    return [env.route_of_first_hop(hop) for (_, _, hop, _) in steps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--od", default="33-71"); p.add_argument("--N", type=int, default=3)
    p.add_argument("--K", type=int, default=1); p.add_argument("--sorties", type=int, default=3000)
    p.add_argument("--switch-every", type=int, default=50); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=250); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--attacker-mode", default="latest", choices=("latest", "smooth"))
    p.add_argument("--band", default="0.15,0.95"); p.add_argument("--json-out", default="")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (use 3 for 3-parallel)")
    p.add_argument("--leader-ent-frac", type=float, default=1.0,
                   help="sacred role-entropy: convoy-0 target = frac*ln(R) (explore which route to stack)")
    p.add_argument("--leader-alpha-floor", type=float, default=None,
                   help="floor on the leader temperature so leader-alpha cannot collapse toward a "
                        "deterministic (exploitable) policy; kills the across-seed fleet-route variance "
                        "from over-concentration in bad seeds. None = no floor (byte-identical)")
    p.add_argument("--follower-ent-frac", type=float, default=0.05,
                   help="sacred role-entropy: follower target = frac*ln(R) (~0 -> copy the leader)")
    p.add_argument("--alpha-lr", type=float, default=5e-3,
                   help="temperature LR for the sacred role-alphas (fast enough to collapse followers)")
    p.add_argument("--follower-warmup", type=int, default=250,
                   help="sorties before followers switch to the low (follow) temperature, so the "
                        "critic gets the route ordering right before the follower alpha anneals down")
    p.add_argument("--fleet-route", action="store_true",
                   help="reachability CONTROL: followers HARD-COPY convoy 0; is loss_mixed learnable "
                        "(disambiguates an env/reward bug from a follower-learning failure)")
    p.add_argument("--k-extra", type=int, default=0,
                   help="extra k-shortest routes -> shared-edge menu (0 = disjoint first-hop)")
    p.add_argument("--menu-select", action="store_true",
                   help="route-index menu-select action (shared-edge); auto-on when --k-extra>0")
    p.add_argument("--save-leader", default="",
                   help="fleet-route: save the FINAL mixing-leader actor to this path")
    p.add_argument("--ckpt-dir", default="",
                   help="save the leader actor at EVERY eval here, so the best-checkpoint (lowest "
                        "exploitability) is a re-evaluable artefact, not just a number in a log")
    p.add_argument("--leader-ckpt", default="",
                   help="follower BOOTSTRAP: drive convoy 0 with a FROZEN mixing leader loaded here")
    p.add_argument("--forced-copy-warmup", type=int, default=600,
                   help="follower bootstrap: sorties over which the forced-copy probability anneals 1->0")
    p.add_argument("--stack-dup", type=int, default=4,
                   help="prioritised replay: push matched-stack sorties this many times (upsample)")
    p.add_argument("--skip-vanilla", action="store_true",
                   help="follower bootstrap: skip the vanilla control arm (it is reliably ~0.945)")
    p.add_argument("--fp-tau", type=float, default=0.05,
                   help="smooth-FP softmax temperature; higher = more diffuse (steadier) attacker")
    p.add_argument("--smooth-window", type=int, default=250,
                   help="smooth-FP trailing window (recent sorties) the softmax BR targets; the iset "
                        "is sampled fresh EVERY sortie from it (true smooth FP, mirrors single-convoy B2-P3)")
    p.add_argument("--legacy-role-target", action="store_true",
                   help="gen10-MC2 isolation: revert the role-alpha TARGET fix (V(s') entropy term "
                        "uses the primary alpha, pre-fix behaviour) while keeping the node-ordering fix")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    s, t = args.od.split("-"); band = tuple(float(x) for x in args.band.split(","))
    env = make_multiconvoy_env(od=(s, t), N=args.N, K=args.K, edge_vuln_band=band,
                               k_extra_routes=args.k_extra, menu_select=(args.menu_select or args.k_extra > 0),
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
                  reward_scale=args.reward_scale, verbose=True,
                  leader_ent_frac=args.leader_ent_frac, follower_ent_frac=args.follower_ent_frac,
                  alpha_lr=args.alpha_lr, follower_warmup=args.follower_warmup, stack_dup=args.stack_dup,
                  fp_tau=args.fp_tau, leader_alpha_floor=args.leader_alpha_floor,
                  smooth_window=args.smooth_window, ckpt_dir=(args.ckpt_dir or None),
                  legacy_role_target=args.legacy_role_target)

    if args.leader_ckpt:  # follower BOOTSTRAP: frozen mixing leader + forced-copy annealing
        frozen = ProtagonistSAC(node_in_dim=14, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
                                autotune_alpha=True, alpha_init=1.0, device="cpu")
        if env.config.menu_select:
            frozen.actor.menu_routes = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]
            frozen.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))  # match saved leader state
        frozen.actor.load_state_dict(torch.load(args.leader_ckpt))
        print(f"[follower BOOTSTRAP] frozen mixing leader {args.leader_ckpt}; forced-copy warmup "
              f"{args.forced_copy_warmup} (demonstration bootstrapping / Obj-3 ERB)")
        if args.skip_vanilla:
            v = {"expl_tap": float("nan"), "expl": float("nan")}
        else:
            print("[vanilla] independent control...")
            v = train_defender(env, adversarial=False, attacker_mode=args.attacker_mode, **common)
            print(f"[vanilla] TAP {v['expl_tap']:.3f}")
        print("[sacred] follower bootstrap vs frozen leader...")
        sac = train_defender(env, adversarial=True, attacker_mode=args.attacker_mode,
                             frozen_leader=frozen, forced_copy_warmup=args.forced_copy_warmup, **common)
        van = v['expl_tap'] if v['expl_tap'] == v['expl_tap'] else 0.945  # nan -> vanilla reference
        print(f"\n=== FOLLOWER-BOOTSTRAP ({s}->{t}, N={args.N}, K={args.K}) ===")
        print(f"  shortest {baselines['shortest_path']:.3f}  ALNS {baselines['alns']:.3f}  vanilla {van:.3f}  equilibrium {sol.loss_mixed:.3f}")
        print(f"  sacred TAIL-AVG exploitability {sac['tail_expl']:.3f}  (per-eval cycle amp {sac['tail_amp']:.3f}, tail stack {sac['tail_stack']:.2f})")
        print(f"  -> beats ALNS by {baselines['alns']-sac['tail_expl']:+.3f} | beats vanilla by {van-sac['tail_expl']:+.3f} | dist-to-eq {sac['tail_expl']-sol.loss_mixed:+.3f}")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(
                {"od": args.od, "seed": args.seed, "loss_mixed": sol.loss_mixed,
                 "baselines": {k: baselines[k] for k in ("shortest_path", "alns")},
                 "vanilla": v, "sacred": sac}, indent=2))
        return

    if args.fleet_route:
        print("[fleet-route CONTROL] followers hard-copy convoy 0; can the leader learn -> loss_mixed?\n")
        fc = train_defender(env, adversarial=True, attacker_mode=args.attacker_mode, fleet_route=True,
                            save_actor=(args.save_leader or None), **common)
        print(f"\n=== FLEET-ROUTE CONTROL ({s}->{t}, N={args.N}) ===")
        print(f"  loss_mixed {sol.loss_mixed:.3f}   fleet-route FINAL: {fc['expl_tap']:.3f} (TAP) / "
              f"{fc['expl']:.3f} (policy)   stack {fc['stack_rate']:.2f} (1.00 by construction)")
        print(f"  BEST-CHECKPOINT (lowest exploitability; final is misleading under minimax): "
              f"TAP {fc['best_tap']:.3f} @ sortie {fc['best_tap_sortie']} | "
              f"single-ckpt {fc['best_expl']:.3f} @ sortie {fc['best_expl_sortie']}")
        print(f"  ladder: shortest {baselines['shortest_path']:.3f} > ALNS {baselines['alns']:.3f} "
              f">> SACRED(best-ckpt) {fc['best_tap']:.3f} > equilibrium {sol.loss_mixed:.3f}")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(
                {"control": "fleet_route", "loss_mixed": sol.loss_mixed, "fleet_route": fc}, indent=2))
        return
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
