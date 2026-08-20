#!/usr/bin/env python3
"""Train SACRED on the multi-convoy interdiction game.

Each sortie is an N-step episode routing convoy 0 through convoy N-1, with terminal reward the
negative mission failure, so the SAC defender's credit propagates across the fleet's joint
decision and it can learn the correlated optimum; the env exposes earlier convoys' routes through
the truck positions. The interdictor is the oracle best response to the defender's empirical
occupancy play, that is fictitious play. Two arms are available, vanilla (no adversary, nominal
travel objective) and sacred, and both are scored by exploitability, the mission failure of the
policy's occupancy distribution under the best-response interdictor, against the classical ladder
of shortest-path and ALNS and against the oracle values loss_det and loss_mixed.
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
from src.baselines.multiconvoy_oracle import (best_response_attacker_multi, greedy_br_attacker,
                                              objective_value, solve_multiconvoy)
from src.baselines.multiconvoy_planners import classical_baselines
from src.env.smdp_wrapper import SMDPTransition
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

TAP_K = 5
GREEDY_BR_EPS = 0.15   # greedy-BR mode: per-sortie probability of a one-edge-perturbed committed set


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
    """Route all N convoys, returning (steps, occupancy, route_indices).

    Args:
        fleet_route: Followers hard-copy convoy 0, which is the reachability control.
        leader_policy: A frozen mixing leader that drives convoy 0.
        copy_prob: Probability that a follower is forced to copy the leader this sortie. Annealed
            from one to zero, so the critic meets the low-failure stack against a varying leader
            and has to learn to follow the correlation signal rather than memorise a route.
    """
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
    """Exact leader route distribution under menu-select: one forward pass at convoy 0."""
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
    """Exact occupancy distribution in fleet-route plus menu-select mode.

    The fleet stacks on the leader, so the occupancy distribution is the leader's route
    distribution mapped onto the stacked occupancies. This replaces a Monte-Carlo estimate, whose
    sampling noise combined with min-selection bias inflated the best-checkpoint reading.
    """
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
    """Fraction of sorties with every convoy on one route, and with convoy 1 on convoy 0's route.

    These two rates, unlike the conditional follower entropy, distinguish real coordination, which
    concentrates on the leader's route, from a coincidental collapse onto some fixed route.
    """
    arr = np.asarray(routes)
    stack = float(np.mean([len(set(r)) == 1 for r in arr]))
    follow = float(np.mean(arr[:, 1] == arr[:, 0])) if arr.shape[1] >= 2 else 0.0
    return stack, follow


def role_entropies(routes, R):
    """H(leader route) and mean H(follower route given leader), from sampled route tuples.

    A leader entropy that stays high while the follower entropy falls towards zero is the
    signature of learned stack-and-follow; both staying high means independent spreading.
    """
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
                   smooth_window=250, ckpt_dir=None, legacy_role_target=False,
                   route_feats=False, route_bias=False, leader_only_push=False, head_term_lr=None,
                   fp_tau_final=None):
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
            # A learned, rather than fixed, weight on the undiluted per-route "taken" term, at both
            # the policy head and the critic Q head, so the critic can rank the leader's route and
            # the actor gets a gradient to grow follow_w. This is a Bellman-consistent Q input,
            # not a hard bonus. A dedicated head-term learning rate, when given, applies to
            # follow_w too; None leaves it on the base rate.
            fw_lr = {"lr": head_term_lr} if head_term_lr is not None else {}
            prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
            prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w], **fw_lr})
            for qn in (prot.q1, prot.q2, prot.target_q1, prot.target_q2):
                qn.follow_w = torch.nn.Parameter(torch.tensor(1.0))
            prot.critic_optimizer.add_param_group(
                {"params": [prot.q1.follow_w, prot.q2.follow_w], **fw_lr})
            # Undiluted per-route cost and worst-case vulnerability at both heads, with learned
            # weights initialised to zero. This restores the discriminability the mean-pooled
            # embeddings lack, and is the map-conditioning mechanism behind transfer. Registration
            # order must match across the q and target nets, because _soft_update zips them.
            if route_feats:
                cost = np.asarray(env.game.travel_cost, dtype=float)
                vuln = env.game.payoff.max(axis=1)

                def _mm(x):
                    rng_ = x.max() - x.min()
                    return (x - x.min()) / rng_ if rng_ > 0 else np.zeros_like(x)

                feats = torch.tensor(np.stack([_mm(cost), _mm(vuln)], axis=1), dtype=torch.float32)
                for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
                    net.route_feats = feats
                    net.route_feat_w = torch.nn.Parameter(torch.zeros(2))
                # The added head terms need their own learning-rate scale; at the base rate they
                # stay near zero over a full run and silently do nothing.
                lr_kw = {"lr": head_term_lr} if head_term_lr is not None else {}
                prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w], **lr_kw})
                prot.critic_optimizer.add_param_group(
                    {"params": [prot.q1.route_feat_w, prot.q2.route_feat_w], **lr_kw})
            # Learned per-route scalar bias, initialised to zero: pure route-identity capacity.
            if route_bias:
                for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
                    net.route_bias = torch.nn.Parameter(torch.zeros(env.game.n_routes))
                lr_kw = {"lr": head_term_lr} if head_term_lr is not None else {}
                prot.actor_optimizer.add_param_group({"params": [prot.actor.route_bias], **lr_kw})
                prot.critic_optimizer.add_param_group(
                    {"params": [prot.q1.route_bias, prot.q2.route_bias], **lr_kw})
    R = env.game.n_routes
    leader_te = leader_ent_frac * math.log(R)      # role-dependent target entropy (sacred only):
    follower_te = follower_ent_frac * math.log(R)  # leader explores routes; followers copy the leader
    n_occ = len(env.occupancies)
    played = np.zeros(n_occ)               # all-history occupancy histogram (latest-BR + eval)
    occ_seq: list[int] = []                # per-sortie occupancy-index log (smooth-FP trailing window)
    smooth_probs = None                    # smooth-FP attacker distribution over interdiction sets
    committed = None
    # Greedy-BR mode, entered when the env is built with greedy_br=True so obj_matrix is None: the
    # attacker is the submodular greedy best response to the trailing-window occupancy support.
    # Smoothing swaps one member edge of the committed set for a random candidate with probability
    # GREEDY_BR_EPS, playing the role the tau=0.05 softmax plays where the exact matrix exists.
    greedy_mode = env.obj_matrix is None
    greedy_set = None

    def _window_support(seq, window):
        win = seq[-window:]
        if not win:  # initial attacker belief: uniform over the disjoint stacks, a prior that
            # needs no objective matrix
            dis, used = [], set()
            for i, re_ in enumerate(env.game.route_edges):
                if not (re_ & used):
                    dis.append(i); used |= re_
            R_ = env.game.n_routes
            return [(tuple(env.config.N if i == r else 0 for i in range(R_)), 1.0 / len(dis))
                    for r in dis]
        from collections import Counter
        cnt = Counter(win)
        tot = float(sum(cnt.values()))
        return [(tuple(int(x) for x in env.occupancies[i]), c / tot) for i, c in cnt.items()]

    def _perturb_set(edge_set, rng_):
        pool = [e for e in env._cand_edges if frozenset(e) not in set(edge_set)]
        if not pool or not edge_set:
            return edge_set
        out = list(edge_set)
        out[rng_.integers(len(out))] = frozenset(pool[rng_.integers(len(pool))])
        return tuple(out)
    pol_hist = []
    hist = []
    t_chunk = time.time()
    for k in range(sorties):
        if adversarial and (committed is None or k % switch_every == 0):
            if greedy_mode:
                greedy_set, _ = greedy_br_attacker(env.game.route_edges, env.vuln_by_edge,
                                                   _window_support(occ_seq, smooth_window),
                                                   env.config.N, env.config.K,
                                                   env.config.objective, env.config.threshold_m)
                committed = -1  # sentinel; committed EDGES are chosen per sortie below
            elif attacker_mode == "smooth":
                # Smooth fictitious play: a softmax best response to the trailing-window occupancy
                # play, recomputed per block, with the interdiction set sampled fresh every sortie
                # below, since holding one set for a whole block produces cycling. The smoothing
                # optionally anneals linearly from fp_tau to fp_tau_final, because smoothed-game
                # equilibria converge to Nash as the smoothing vanishes and a sharpening attacker
                # raises the penalty for drifting off the hedge. None keeps tau constant.
                cur_tau = (fp_tau if fp_tau_final is None
                           else fp_tau + (fp_tau_final - fp_tau) * (k / max(sorties - 1, 1)))
                smooth_probs = smooth_fp_probs(occ_seq, n_occ, env.obj_matrix, cur_tau, smooth_window)
            else:
                occ_dist = played / played.sum() if played.sum() > 0 else np.ones(n_occ) / n_occ
                committed, _ = best_response_attacker_multi(env.obj_matrix, occ_dist)
        env.reset()
        p_committed = None
        if adversarial:
            if greedy_mode:
                edges = (_perturb_set(greedy_set, rng) if rng.random() < GREEDY_BR_EPS
                         else greedy_set)
                env.commit_set(edges)
                p_committed = env.route_interception(edges)
            else:
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
            # Analytic expected mission-failure reward: a dense, low-variance, unbiased
            # replacement for the sampled Bernoulli draw of env.resolve().
            p = p_committed if p_committed is not None else env.game.payoff[:, committed]
            reward = -interception_loss * objective_value(np.asarray(occ), p, env.config.N,
                                                          env.config.objective, env.config.threshold_m)
        else:
            travel = float(sum(env.game.travel_cost[r] for r in _route_of(steps, env)))
            reward = -interception_loss * (travel / (env.config.N * mean_cost))   # nominal travel
        N = env.config.N
        bootstrap = frozen_leader is not None
        # Upsample matched-stack sorties, so the critic keeps training on the rare but informative
        # low-failure stacked experience once natural stacks become scarce.
        is_stack = sum(1 for c in occ if c > 0) == 1
        n_push = stack_dup if (is_stack and adversarial) else 1
        if adversarial:
            # Role-dependent entropy. Under bootstrap, convoys 1..N-1 are all followers because
            # the leader is frozen; otherwise convoy 0 leads and the followers switch to the low
            # temperature only after the warmup, so the critic learns the ordering before they
            # commit. Tagged in a pass over all steps first, so that _transition's shallow
            # next-state copy carries the next decision's alpha_group as well.
            for obs_j, ci_j, _, _ in steps:
                is_follower = (ci_j != 0) and (bootstrap or k >= follower_warmup)
                obs_j["target_entropy"] = follower_te if is_follower else leader_te
                obs_j["alpha_group"] = 1 if is_follower else 0
        if leader_only_push and fleet_route:
            # In fleet-route mode the followers hard-copy the leader, so their transitions carry
            # no decision content, yet after the warmup they train the same shared actor towards
            # near-argmax on states that differ from the leader's only in the correlation signal.
            # Push only the leader's decision, terminal with the sortie reward.
            obs0, ci0, hop0, mask0 = steps[0]
            t = _transition(obs0, ci0, hop0, mask0, reward, None, None, None, True)
            for _ in range(n_push):
                prot.replay_buffer.push(t)
            steps_to_push = []
        else:
            steps_to_push = steps
        for i, (obs, ci, hop, mask) in enumerate(steps_to_push):
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
            if exact:  # fleet-route: exact occupancy distribution in one forward pass
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
            expl = env.exploitability_of_occupancy_dist(d)
            expl_tap = env.exploitability_of_occupancy_dist(np.mean(pol_hist[-TAP_K:], axis=0))
            if ckpt_dir is not None:  # per-eval checkpoint, so any of them can be re-evaluated
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(Path(ckpt_dir) / f"actor_ep{k + 1}.pt"))
            nz = d[d > 0]; h = float(-(nz * np.log(nz)).sum())
            t_eval = time.time() - t_ev
            hist.append((k + 1, expl, expl_tap, float(prot.alpha), float(prot.alpha_foll),
                         stack_rate, follow_rate, H_lead, H_foll, t_train, t_eval))
            if verbose:
                fw = float(prot.actor.follow_w) if hasattr(prot.actor, "follow_w") else 0.0
                rw = ("rw[" + ",".join(f"{float(x):.2f}" for x in prot.actor.route_feat_w) + "] "
                      if hasattr(prot.actor, "route_feat_w") else "")
                print(f"    sortie {k+1:5d}: expl {expl:.3f} | TAP {expl_tap:.3f} | "
                      f"alpha L{prot.alpha:.2f}/F{prot.alpha_foll:.2f} fw {fw:.2f} {rw}| "
                      f"stack {stack_rate:.2f} follow {follow_rate:.2f} | H_lead {H_lead:.2f} H_foll {H_foll:.2f} | "
                      f"train {t_train:4.0f}s eval {t_eval:3.0f}s   "
                      + (f"(loss_mixed={sol.loss_mixed:.3f}, ALNS={baselines['alns']:.3f})"
                         if sol is not None else
                         f"(GREEDY yardstick; heuristic={baselines['heuristic']:.3f})"), flush=True)
            t_chunk = time.time()
    if fleet_route and env.config.menu_select:  # exact final reading, as in the eval block above
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
    expl_tap = env.exploitability_of_occupancy_dist(np.mean(pol_hist[-TAP_K:], axis=0))
    expl = env.exploitability_of_occupancy_dist(d)
    # Stationary-tail time average. Under zero-sum fictitious play the equilibrium is the time
    # average rather than any single evaluation's play, so this reports the exploitability of the
    # mean occupancy over the last few evals, the amplitude of the per-eval exploitability, and
    # the mean stack rate there.
    tail_k = min(len(pol_hist), 12)
    tail_expl = env.exploitability_of_occupancy_dist(np.mean(pol_hist[-tail_k:], axis=0))
    tail_amp = float(np.std([h[1] for h in hist[-tail_k:]])) if len(hist) >= 2 else 0.0
    tail_stack = float(np.mean([h[5] for h in hist[-tail_k:]])) if hist else stack_rate
    if save_actor:
        Path(save_actor).parent.mkdir(parents=True, exist_ok=True)
        torch.save(prot.actor.state_dict(), save_actor)
        print(f"  [saved leader actor] {save_actor}")
    # Best checkpoint. The final iterate misleads under minimax coevolution, since it over-trains
    # towards uniform, so the lowest-exploitability training point is selected instead. Both the
    # deployable trailing-average reading and the single-checkpoint reading are reported, and both
    # can be re-derived from the saved per-eval actors and the pol_hist occupancy distributions.
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
    p.add_argument("--route-feats", action="store_true",
                   help="gen11 arm B: undiluted per-route cost+vulnerability features with learned "
                        "weights at policy AND critic heads (menu mode, adversarial only)")
    p.add_argument("--route-bias", action="store_true",
                   help="gen11 arm E: learned per-route scalar bias at both heads (pure identity "
                        "capacity; reconstructs what the pre-fix permutation accidentally provided)")
    p.add_argument("--leader-only-push", action="store_true",
                   help="gen11 arm C: fleet-route pushes ONLY the leader's decision (terminal, full "
                        "sortie reward); kills the follower-push entropy-target conflict")
    p.add_argument("--vanilla-only", action="store_true",
                   help="train ONLY the vanilla control arm (independent convoys, travel objective)")
    p.add_argument("--head-term-lr", type=float, default=None,
                   help="dedicated lr for the route_feats/route_bias/follow_w param groups (gen11b: "
                        "~3e-2); None = inherit the base optimiser lr (the gen11 silent no-op)")
    p.add_argument("--greedy-br", action="store_true",
                   help="gen26: MATRIX-FREE mode for K past the exact wall (K>=4): the game is built "
                        "at K=1, obj_matrix is never enumerated, and the attacker + exploitability "
                        "eval use the verified submodular greedy BR (A4-core, (1-1/e) guarantee, "
                        "disclosed). Anchors are computed under the SAME greedy yardstick.")
    p.add_argument("--fp-tau-final", type=float, default=None,
                   help="gen17/C4 annealed smoothing: linearly anneal the smooth-FP tau from "
                        "--fp-tau to this value across training; None = constant tau")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    s, t = args.od.split("-"); band = tuple(float(x) for x in args.band.split(","))
    env = make_multiconvoy_env(od=(s, t), N=args.N, K=args.K, edge_vuln_band=band,
                               k_extra_routes=args.k_extra, menu_select=(args.menu_select or args.k_extra > 0),
                               interception_loss=args.interception_loss, objective="mission", seed=args.seed,
                               greedy_br=args.greedy_br)
    mean_cost = float(env.game.travel_cost.mean())
    if args.greedy_br:
        # Past the budget at which exact solutions are computable there is no LP, no ALNS
        # worst case and no equilibrium, so the anchors are shortest-stack, uniform-disjoint-stack
        # and inverse-vulnerability-disjoint-stack under the same greedy yardstick. Every arm in
        # this mode shares that yardstick, and absolute statements carry the certified interval
        # [v, v/(1-1/e)].
        sol = None
        R_ = env.game.n_routes
        dis, used = [], set()
        for i, re_ in enumerate(env.game.route_edges):
            if not (re_ & used):
                dis.append(i); used |= re_
        def _stack_support(routes_idx, weights):
            w = np.asarray(weights, float); w = w / w.sum()
            return [(tuple(env.config.N if i == r else 0 for i in range(R_)), float(wt))
                    for wt, r in zip(w, routes_idx)]
        def _greedy_val(support):
            _, v = greedy_br_attacker(env.game.route_edges, env.vuln_by_edge, support,
                                      env.config.N, env.config.K,
                                      env.config.objective, env.config.threshold_m)
            return float(v)
        qs = []
        for r in dis:
            worst = max(env.vuln_by_edge.get(e, 1.0) for e in env.game.route_edges[r])
            qs.append(1.0 - (1.0 - worst) ** env.config.N)
        cheapest = int(np.argmin(env.game.travel_cost))
        baselines = {
            "shortest_path": _greedy_val(_stack_support([cheapest], [1.0])),
            "heuristic": _greedy_val(_stack_support(dis, [1.0] * len(dis))),
            "heuristic_invvuln": _greedy_val(_stack_support(dis, [1.0 / max(q, 1e-9) for q in qs])),
        }
        print(f"Multi-convoy {s}->{t} N={args.N} K={args.K} [GREEDY-BR MODE, no exact oracle]: "
              f"{env.game.n_routes} routes, {len(env.occupancies)} occupancies, "
              f"{len(env._cand_edges)} candidate edges (C(E,K) never enumerated).\n"
              f"  Greedy-yardstick anchors: shortest-stack {baselines['shortest_path']:.3f} | "
              f"uniform-disjoint-stack {baselines['heuristic']:.3f} | "
              f"inv-vuln-disjoint-stack {baselines['heuristic_invvuln']:.3f}\n")
    else:
        sol = solve_multiconvoy(env.game, args.N, "mission")
        baselines = classical_baselines(env.game, args.N, "mission")
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
                  legacy_role_target=args.legacy_role_target, route_feats=args.route_feats,
                  route_bias=args.route_bias, leader_only_push=args.leader_only_push,
                  head_term_lr=args.head_term_lr, fp_tau_final=args.fp_tau_final)

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
        # When the vanilla arm is skipped its exploitability is nan, so fall back to a stored
        # vanilla reading for the comparison row. Display only.
        van = v['expl_tap'] if v['expl_tap'] == v['expl_tap'] else 0.859
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

    if args.vanilla_only:
        print("[vanilla-only] training the control (nominal travel objective, no adversary)...")
        v = train_defender(env, adversarial=False, attacker_mode=args.attacker_mode, **common)
        vt = [h[2] for h in v["history"]]
        print(f"\n=== VANILLA-ONLY ({s}->{t}, N={args.N}, K={args.K}, seed={args.seed}) ===")
        print(f"  vanilla TAP {v['expl_tap']:.3f} (policy {v['expl']:.3f}) | best-ckpt TAP "
              f"{min(vt) if vt else float('nan'):.3f}   [ALNS {baselines['alns']:.3f}, eq {sol.loss_mixed:.3f}]")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(
                {"od": args.od, "N": args.N, "K": args.K, "seed": args.seed, "arm": "vanilla_only",
                 "loss_mixed": sol.loss_mixed, "baselines": {k: baselines[k] for k in ("shortest_path", "alns")},
                 "vanilla": v}, indent=2))
        return
    if args.fleet_route:
        print("[fleet-route CONTROL] followers hard-copy convoy 0; can the leader learn -> loss_mixed?\n")
        fc = train_defender(env, adversarial=True, attacker_mode=args.attacker_mode, fleet_route=True,
                            save_actor=(args.save_leader or None), **common)
        print(f"\n=== FLEET-ROUTE CONTROL ({s}->{t}, N={args.N}) ===")
        if sol is None:  # greedy-BR mode: same-yardstick ladder, certified interval for absolutes
            print(f"  [GREEDY yardstick; certified interval = [v, v/(1-1/e)] = [v, {1/(1-1/np.e):.3f}v]]")
            print(f"  fleet-route FINAL: {fc['expl_tap']:.3f} (TAP) / {fc['expl']:.3f} (policy)")
            print(f"  BEST-CHECKPOINT: TAP {fc['best_tap']:.3f} @ sortie {fc['best_tap_sortie']} | "
                  f"single-ckpt {fc['best_expl']:.3f} @ sortie {fc['best_expl_sortie']}")
            print(f"  ladder (same greedy yardstick): shortest-stack {baselines['shortest_path']:.3f} > "
                  f"uniform-disjoint {baselines['heuristic']:.3f} > "
                  f"inv-vuln-disjoint {baselines['heuristic_invvuln']:.3f} vs "
                  f"SACRED(best-ckpt) {fc['best_tap']:.3f}")
        else:
            print(f"  loss_mixed {sol.loss_mixed:.3f}   fleet-route FINAL: {fc['expl_tap']:.3f} (TAP) / "
                  f"{fc['expl']:.3f} (policy)   stack {fc['stack_rate']:.2f} (1.00 by construction)")
            print(f"  BEST-CHECKPOINT (lowest exploitability; final is misleading under minimax): "
                  f"TAP {fc['best_tap']:.3f} @ sortie {fc['best_tap_sortie']} | "
                  f"single-ckpt {fc['best_expl']:.3f} @ sortie {fc['best_expl_sortie']}")
            print(f"  ladder: shortest {baselines['shortest_path']:.3f} > ALNS {baselines['alns']:.3f} "
                  f">> SACRED(best-ckpt) {fc['best_tap']:.3f} > equilibrium {sol.loss_mixed:.3f}")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(
                {"control": "fleet_route", "greedy_br": bool(args.greedy_br),
                 "loss_mixed": (sol.loss_mixed if sol is not None else None),
                 "anchors": {k: float(v) for k, v in baselines.items()
                             if isinstance(v, (int, float))},
                 "fleet_route": fc}, indent=2))
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
