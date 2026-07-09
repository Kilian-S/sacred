#!/usr/bin/env python3
"""I2 feasibility slice: does adversarial RL learn an unexploitable mixed-strategy route policy?

Single-convoy interdiction game (Kaliningrad 33->71, K=1). Three arms, all evaluated by
EXPLOITABILITY = interception under the oracle best-response interdictor (the exploitability metric):
  * shortest_path : the deterministic operational default (no training). Oracle bound: loss_det=1.0.
  * vanilla       : SAC defender trained with the nominal (travel-cost) objective, NO adversary ->
                    converges to the shortest route -> deterministic -> exploitable.
  * sacred        : SAC defender trained against the ORACLE best-response interdictor (recomputed as
                    its policy evolves = ATLA/fictitious play with the strongest attacker). Should
                    learn a mixed strategy approaching the equilibrium loss_mixed (~0.17 here).

Positive result = Expl(sacred) << Expl(vanilla) ~ Expl(shortest_path) = 1.0, and sacred -> loss_mixed
(validated against the computable equilibrium). The learned-antagonist full co-evolution is a
follow-on; here the oracle IS the adversary (a strong, correct, computable one).

Run: PYTHONPATH=. python scripts/train_interdiction.py --sorties 3000 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.env.smdp_wrapper import SMDPTransition
from src.baselines.fp_dynamics import sample_smooth_iset, smooth_fp_probs
from src.baselines.interdiction_oracle import (
    best_response_attacker, cost_constrained_value, route_distribution_from_first_hops, solve)
from src.envs.interdiction import make_interdiction_env

TAP_K = 5   # trailing-averaged-policy window: mean of the policy distributions at the last 5 evals


def hop_probs(prot: ProtagonistSAC, obs, allowed: list) -> dict:
    """Policy probabilities over the allowed next hops at the observed convoy position."""
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)  # MUST match featurize_state's row order
    active_idx = n2i[obs["trucks"][0]["current_node"]]
    mask_idxs = [n2i[n] for n in allowed]
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, active_idx, mask_idxs)
    prot.actor.train()
    return {allowed[i]: float(probs[i]) for i in range(len(allowed))}


def defender_route_distribution(prot, env, mode: str = "first_hop") -> np.ndarray:
    """The policy's EXACT deployable route mixture: first-hop probabilities on disjoint route
    sets, or the trie branch-product on shared-edge sets (walk mode)."""
    if mode == "walk":
        return env.walk_distribution(
            lambda node, allowed: hop_probs(prot, env.observe_at(node), allowed))
    obs = env.reset()
    allowed = env.defender_action_mask()[0]
    return route_distribution_from_first_hops(env.game, env.base, hop_probs(prot, obs, allowed))


def _hist(seq, n: int) -> np.ndarray:
    """Empirical route distribution of a play sequence (uniform if empty)."""
    h = np.zeros(n)
    for i in seq:
        h[i] += 1.0
    return h / h.sum() if h.sum() > 0 else np.ones(n) / n


def final_metrics(prot, env, played_seq, window: int, mode: str, pol_hist: list) -> dict:
    """The four exploitability readings of a trained arm: TAP = the TRAILING-AVERAGED POLICY
    distribution (mean of the exact policy route distributions at the last TAP_K evals: the
    deployable late-training pattern, no fictitious-play mid-cycle bias, no exploration credit;
    the B2 primary), plus the final POLICY distribution, the TRAILING-WINDOW empirical play and
    the ALL-HISTORY empirical play (the wave-1/I2 continuity readings)."""
    d_pol = pol_hist[-1] if pol_hist else defender_route_distribution(prot, env, mode)
    d_tap = np.mean(pol_hist[-TAP_K:] if pol_hist else [d_pol], axis=0)
    _, expl_pol = best_response_attacker(env.game, d_pol)
    _, expl_tap = best_response_attacker(env.game, d_tap)
    d_win = _hist(played_seq[-window:], env.game.n_routes)
    _, expl_win = best_response_attacker(env.game, d_win)
    d_avg = _hist(played_seq, env.game.n_routes)
    _, expl_avg = best_response_attacker(env.game, d_avg)
    clean_cost = float(np.asarray(d_tap) @ env.game.travel_cost)
    return {"expl_tap": expl_tap, "expl_policy": expl_pol, "expl_window": expl_win,
            "expl_avg": expl_avg, "clean_cost_tap": clean_cost,
            "dist_tap": np.asarray(d_tap).tolist(), "dist_policy": d_pol.tolist(),
            "dist_window": d_win.tolist()}


def _prot_walk_transition(obs, hop, mask, reward, next_obs, next_mask, done) -> SMDPTransition:
    """Per-hop walk transition: intermediate hops carry zero reward and bootstrap through the
    next branch state (allowed_destinations enrichment per the SAC update contract); the terminal
    hop carries the sortie reward."""
    next_state = {}
    if not done:
        next_state = dict(next_obs)
        next_state["active_truck"] = 0
        next_state["allowed_destinations"] = {"protagonist": {0: list(next_mask[0])}}
    return SMDPTransition(agent="protagonist", state=obs, action={0: hop}, reward=reward,
                          next_state=next_state, done=done, elapsed_ticks=1,
                          action_mask={"protagonist": mask}, info={})


def _prot_transition(obs, first_hop, reward, mask) -> SMDPTransition:
    return SMDPTransition(agent="protagonist", state=obs, action={0: first_hop}, reward=reward,
                          next_state={}, done=True, elapsed_ticks=1,
                          action_mask={"protagonist": mask}, info={})


def train_defender(env, *, sorties, switch_every, batch_size, seed, adversarial, eval_every, sol,
                   reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0, window=500,
                   mode="first_hop", attacker_mode="latest", smooth_tau=0.05, smooth_window=250):
    torch.manual_seed(seed); np.random.seed(seed)
    prot = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=reward_scale, lr_actor=lr_actor,
                          autotune_alpha=autotune_alpha, alpha_init=alpha_init, device="cpu")
    committed = None
    br_history: list[int] = []   # every BR iset computed so far: the attacker's FP mixture (B2-P2)
    smooth_probs = None          # smooth-FP attacker distribution over isets (B2-P3)
    rng = np.random.default_rng(seed)
    history = []
    played = np.zeros(env.game.n_routes)   # empirical route-play histogram (fictitious-play average)
    played_seq: list[int] = []             # per-sortie route log (for the trailing-window reading)
    pol_hist: list[np.ndarray] = []        # exact policy route distributions at eval points (TAP)
    for k in range(sorties):
        if adversarial and (committed is None or k % switch_every == 0):
            # oracle best-response to the defender's empirical average play (fictitious play, which
            # CONVERGES; best-responding to the instantaneous policy makes the defender chase/oscillate).
            avg = played / played.sum() if played.sum() > 0 else np.ones(env.game.n_routes) / env.game.n_routes
            committed, _ = best_response_attacker(env.game, avg)
            br_history.append(int(committed))
            if attacker_mode == "smooth":
                # smooth fictitious play (B2-P3) via the shared fp_dynamics discipline: softmax best
                # response to the defender's TRAILING-WINDOW recent play, sampled fresh each sortie
                # (below). Mixed vs a mixed defender (no cycling pressure), sharp vs a parked one
                # (drift punished within ~smooth_window sorties). tau probe-pinned.
                smooth_probs = smooth_fp_probs(played_seq, env.game.n_routes, env.game.payoff,
                                               smooth_tau, smooth_window)
        if mode == "walk":
            # hop-by-hop route choice on the candidate-route trie (shared-edge instances).
            obs, done, ri = env.begin_walk()
            steps = []  # (obs, hop, mask, next_obs, next_mask)
            while not done:
                mask = env.walk_mask()
                act = prot.select_action(obs, mask, deterministic=False)
                nobs, done, ri = env.step_walk(act[0])
                nmask = env.walk_mask() if not done else None
                steps.append((obs, act[0], mask, nobs, nmask))
                obs = nobs
        else:
            obs = env.reset()
            mask = env.defender_action_mask()
            act = prot.select_action(obs, mask, deterministic=False)
            ri = env.route_of_first_hop(act[0])
        played[ri] += 1.0
        played_seq.append(ri)
        if adversarial:
            # latest = the current pure BR held for a block (B2-P: over-disciplines -> cycling).
            # mixture = uniform over ALL past BRs (B2-P2: goes stale -> cost-gradient parking).
            # smooth = smooth fictitious play (B2-P3): sample from softmax(e/tau) vs recent play.
            if attacker_mode == "latest":
                iset = committed
            elif attacker_mode == "mixture":
                iset = br_history[int(rng.integers(len(br_history)))]
            else:
                iset = sample_smooth_iset(smooth_probs, rng)
            env.commit(iset)
            out = env.resolve(ri)
        else:
            # no adversary: reward is the nominal travel cost only (drives to the shortest route).
            out = type("O", (), {"defender_reward": -env.config.travel_cost_weight * env.game.travel_cost[ri]})()
        if mode == "walk":
            for i, (o, h, m, no, nm) in enumerate(steps):
                last = i == len(steps) - 1
                prot.replay_buffer.push(_prot_walk_transition(
                    o, h, m, out.defender_reward if last else 0.0, no, nm, last))
        else:
            prot.replay_buffer.push(_prot_transition(obs, act[0], out.defender_reward, mask))
        prot.update(batch_size)
        if eval_every and (k + 1) % eval_every == 0:
            # four readings: TAP (trailing-averaged policy: the B2 primary), final policy,
            # trailing-window empirical play, all-history empirical play (I2/wave-1 continuity).
            d_pol = defender_route_distribution(prot, env, mode)
            pol_hist.append(d_pol)
            _, expl_pol = best_response_attacker(env.game, d_pol)
            _, expl_tap = best_response_attacker(env.game, np.mean(pol_hist[-TAP_K:], axis=0))
            _, expl_win = best_response_attacker(env.game, _hist(played_seq[-window:], env.game.n_routes))
            _, expl_avg = best_response_attacker(env.game, _hist(played_seq, env.game.n_routes))
            nz = d_pol[d_pol > 0]
            h_pol = float(-(nz * np.log(nz)).sum())   # route-mixture entropy (telemetry, B2-P2)
            history.append((k + 1, expl_pol, expl_tap, expl_win, expl_avg, float(prot.alpha), h_pol))
            print(f"    sortie {k+1:5d}: expl policy {expl_pol:.3f} | TAP {expl_tap:.3f} | "
                  f"window {expl_win:.3f} | avg {expl_avg:.3f} | alpha {prot.alpha:.2f} "
                  f"H(pol) {h_pol:.2f}   (loss_mixed={sol.value:.3f}, loss_det={sol.loss_det:.3f})",
                  flush=True)
    prot._played = played  # expose the empirical strategy for final eval
    return prot, history, played_seq, pol_hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--od", type=str, default="33-71")
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--sorties", type=int, default=3000)
    p.add_argument("--switch-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--travel-cost-weight", type=float, default=0.05)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--k-extra", type=int, default=0, help="0 = clean edge-disjoint routes only")
    p.add_argument("--edge-vuln-band", type=str, default="",
                   help="lo,hi: heterogeneous soft interception (I3 asymmetric instance); '' = hard")
    p.add_argument("--route-mode", type=str, default="first_hop", choices=("first_hop", "walk"),
                   help="walk = hop-by-hop trie routing (REQUIRED for shared-edge instances, k-extra > 0)")
    p.add_argument("--attacker-mode", type=str, default="latest", choices=("latest", "mixture", "smooth"),
                   help="latest = pure BR held per block (B2-P); mixture = uniform over all past BRs "
                        "(B2-P2); smooth = smooth fictitious play, softmax BR to recent play (B2-P3)")
    p.add_argument("--smooth-tau", type=float, default=0.05, help="smooth-FP softmax temperature")
    p.add_argument("--smooth-window", type=int, default=250, help="smooth-FP targeting window (plays)")
    p.add_argument("--window", type=int, default=500, help="trailing-window size for empirical play")
    p.add_argument("--json-out", type=str, default="", help="write the result record here (matrix aggregation)")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (use 3 for 3-parallel)")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    s, t = args.od.split("-")
    band = tuple(float(x) for x in args.edge_vuln_band.split(",")) if args.edge_vuln_band else None
    env = make_interdiction_env(od=(s, t), K=args.K, interception_loss=args.interception_loss,
                                travel_cost_weight=args.travel_cost_weight, k_extra_routes=args.k_extra,
                                edge_vuln_band=band, seed=args.seed)
    sol = solve(env.game)
    if args.route_mode == "first_hop" and max(len(v) for v in env.routes_by_first_hop.values()) > 1:
        print("WARNING: candidate routes share first hops (shared prefixes); first_hop mode cannot "
              "express every route: use --route-mode walk.", flush=True)
    print(f"Interdiction {s}->{t} K={args.K} band={band} mode={args.route_mode}: "
          f"{env.game.n_routes} routes, {len(env.first_hops)} first hops; "
          f"oracle loss_det={sol.loss_det:.3f}, loss_mixed={sol.value:.3f}, gap={sol.gap:.3f}\n")

    # deterministic shortest-path reference (the operational default) + uniform mixing reference
    # (what UNCALIBRATED randomisation buys; on asymmetric instances it is measurably suboptimal).
    det = np.zeros(env.game.n_routes); det[env.shortest_route_index()] = 1.0
    _, expl_sp = best_response_attacker(env.game, det)
    uni = np.ones(env.game.n_routes) / env.game.n_routes
    _, expl_uni = best_response_attacker(env.game, uni)
    print(f"[shortest_path] exploitability = {expl_sp:.3f} (deterministic; the operational default)")
    print(f"[uniform]       exploitability = {expl_uni:.3f} (uncalibrated mixing reference)\n")

    print("[vanilla] training defender with NO adversary (nominal travel-cost objective)...")
    vprot, vhist, vseq, vpol = train_defender(env, sorties=args.sorties, switch_every=args.switch_every,
                                              batch_size=args.batch_size, seed=args.seed, adversarial=False,
                                              eval_every=args.eval_every, sol=sol, reward_scale=args.reward_scale,
                                              lr_actor=args.lr_actor, window=args.window, mode=args.route_mode)
    vfin = final_metrics(vprot, env, vseq, args.window, args.route_mode, vpol)
    print(f"[vanilla] final expl: TAP {vfin['expl_tap']:.3f} | policy {vfin['expl_policy']:.3f} | "
          f"window {vfin['expl_window']:.3f} | avg {vfin['expl_avg']:.3f}\n")

    print("[sacred] training defender vs the ORACLE best-response interdictor (fictitious play)...")
    sprot, hist, sseq, spol = train_defender(env, sorties=args.sorties, switch_every=args.switch_every,
                                             batch_size=args.batch_size, seed=args.seed, adversarial=True,
                                             eval_every=args.eval_every, sol=sol, reward_scale=args.reward_scale,
                                             lr_actor=args.lr_actor, window=args.window, mode=args.route_mode,
                                             attacker_mode=args.attacker_mode,
                                             smooth_tau=args.smooth_tau, smooth_window=args.smooth_window)
    sfin = final_metrics(sprot, env, sseq, args.window, args.route_mode, spol)
    eq_cost = float(sol.defender_strategy @ env.game.travel_cost)
    print(f"\n=== RESULT (Kaliningrad {s}->{t}, K={args.K}, band={band}, mode={args.route_mode}, "
          f"attacker={args.attacker_mode}, seed={args.seed}) ===")
    print(f"  arm             expl_TAP  expl_policy  expl_window  expl_avg  cost(TAP)")
    print(f"  shortest_path   {expl_sp:8.3f}  {expl_sp:11.3f}  {expl_sp:11.3f}  {expl_sp:8.3f}  "
          f"{float(env.game.travel_cost.min()):9.1f}")
    print(f"  uniform         {expl_uni:8.3f}  {expl_uni:11.3f}  {expl_uni:11.3f}  {expl_uni:8.3f}  "
          f"{float(uni @ env.game.travel_cost):9.1f}")
    print(f"  vanilla         {vfin['expl_tap']:8.3f}  {vfin['expl_policy']:11.3f}  "
          f"{vfin['expl_window']:11.3f}  {vfin['expl_avg']:8.3f}  {vfin['clean_cost_tap']:9.1f}")
    print(f"  sacred          {sfin['expl_tap']:8.3f}  {sfin['expl_policy']:11.3f}  "
          f"{sfin['expl_window']:11.3f}  {sfin['expl_avg']:8.3f}  {sfin['clean_cost_tap']:9.1f}")
    print(f"  equilibrium (loss_mixed) = {sol.value:.3f} at cost {eq_cost:.1f}; loss_det = {sol.loss_det:.3f}")
    print(f"  -> sacred TAP distance to equilibrium: {abs(sfin['expl_tap'] - sol.value):.3f}")
    if args.json_out:
        budgets = sorted({round(float(b), 4) for b in env.game.travel_cost} | {round(eq_cost, 4)})
        frontier = []
        for b in budgets:
            try:
                v, _ = cost_constrained_value(env.game, b)
                frontier.append([b, v])
            except ValueError:
                continue
        record = {"od": args.od, "K": args.K, "band": band, "seed": args.seed,
                  "sorties": args.sorties, "window": args.window, "mode": args.route_mode,
                  "attacker_mode": args.attacker_mode,
                  "smooth_tau": args.smooth_tau, "smooth_window": args.smooth_window,
                  "tap_k": TAP_K, "loss_det": sol.loss_det, "loss_mixed": sol.value,
                  "equilibrium_defender": sol.defender_strategy.tolist(),
                  "equilibrium_cost": eq_cost, "route_costs": env.game.travel_cost.tolist(),
                  "frontier": frontier,
                  "arms": {"shortest_path": {"expl_tap": expl_sp,
                                             "clean_cost_tap": float(env.game.travel_cost.min())},
                           "uniform": {"expl_tap": expl_uni,
                                       "clean_cost_tap": float(uni @ env.game.travel_cost)},
                           "vanilla": {**vfin, "history": vhist},
                           "sacred": {**sfin, "history": hist}}}
        Path(args.json_out).write_text(json.dumps(record, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
