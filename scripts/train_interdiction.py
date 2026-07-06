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

from src.agents.networks import featurize_state
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.env.smdp_wrapper import SMDPTransition
from src.baselines.interdiction_oracle import (
    best_response_attacker, route_distribution_from_first_hops, solve)
from src.envs.interdiction import make_interdiction_env


def first_hop_probs(prot: ProtagonistSAC, env, obs) -> dict:
    """Policy probabilities over the defender's first hops (its route mixture at the base)."""
    allowed = env.defender_action_mask()[0]
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    node_ids = list(obs["nodes"].keys()); n2i = {n: i for i, n in enumerate(node_ids)}
    active_idx = n2i[obs["trucks"][0]["current_node"]]
    mask_idxs = [n2i[n] for n in allowed]
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, active_idx, mask_idxs)
    prot.actor.train()
    return {allowed[i]: float(probs[i]) for i in range(len(allowed))}


def defender_route_distribution(prot, env) -> np.ndarray:
    obs = env.reset()
    return route_distribution_from_first_hops(env.game, env.base, first_hop_probs(prot, env, obs))


def exploitability(prot, env) -> float:
    """Interception of the defender's route distribution under the oracle best-response interdictor."""
    d = defender_route_distribution(prot, env)
    _, expl = best_response_attacker(env.game, d)
    return expl


def _hist(seq, n: int) -> np.ndarray:
    """Empirical route distribution of a play sequence (uniform if empty)."""
    h = np.zeros(n)
    for i in seq:
        h[i] += 1.0
    return h / h.sum() if h.sum() > 0 else np.ones(n) / n


def final_metrics(prot, env, played_seq, window: int) -> dict:
    """The three exploitability readings of a trained arm: the POLICY route distribution (the
    deployable object: what sampling the frozen stochastic policy each sortie exposes), the
    TRAILING-WINDOW empirical play (late-training fictitious-play average), and the ALL-HISTORY
    empirical play (includes early exploration; the I2 metric, kept for continuity)."""
    d_pol = defender_route_distribution(prot, env)
    _, expl_pol = best_response_attacker(env.game, d_pol)
    d_win = _hist(played_seq[-window:], env.game.n_routes)
    _, expl_win = best_response_attacker(env.game, d_win)
    d_avg = _hist(played_seq, env.game.n_routes)
    _, expl_avg = best_response_attacker(env.game, d_avg)
    return {"expl_policy": expl_pol, "expl_window": expl_win, "expl_avg": expl_avg,
            "dist_policy": d_pol.tolist(), "dist_window": d_win.tolist()}


def _prot_transition(obs, first_hop, reward, mask) -> SMDPTransition:
    return SMDPTransition(agent="protagonist", state=obs, action={0: first_hop}, reward=reward,
                          next_state={}, done=True, elapsed_ticks=1,
                          action_mask={"protagonist": mask}, info={})


def train_defender(env, *, sorties, switch_every, batch_size, seed, adversarial, eval_every, sol,
                   reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0, window=500):
    torch.manual_seed(seed); np.random.seed(seed)
    prot = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=reward_scale, lr_actor=lr_actor,
                          autotune_alpha=autotune_alpha, alpha_init=alpha_init, device="cpu")
    committed = None
    history = []
    played = np.zeros(env.game.n_routes)   # empirical route-play histogram (fictitious-play average)
    played_seq: list[int] = []             # per-sortie route log (for the trailing-window reading)
    for k in range(sorties):
        if adversarial and (committed is None or k % switch_every == 0):
            # oracle best-response to the defender's empirical average play (fictitious play, which
            # CONVERGES; best-responding to the instantaneous policy makes the defender chase/oscillate).
            avg = played / played.sum() if played.sum() > 0 else np.ones(env.game.n_routes) / env.game.n_routes
            committed, _ = best_response_attacker(env.game, avg)
        obs = env.reset()
        mask = env.defender_action_mask()
        act = prot.select_action(obs, mask, deterministic=False)
        fh = act[0]
        ri = env.route_of_first_hop(fh)
        played[ri] += 1.0
        played_seq.append(ri)
        if adversarial:
            env.commit(committed)
            out = env.resolve_first_hop(fh)
        else:
            # no adversary: reward is the nominal travel cost only (drives to the shortest route).
            out = type("O", (), {"defender_reward": -env.config.travel_cost_weight * env.game.travel_cost[ri]})()
        prot.replay_buffer.push(_prot_transition(obs, fh, out.defender_reward, mask))
        prot.update(batch_size)
        if eval_every and (k + 1) % eval_every == 0:
            # the three readings: policy distribution (deployable), trailing-window empirical play
            # (late fictitious-play average), all-history empirical play (the I2 continuity metric).
            expl_pol = exploitability(prot, env)
            _, expl_win = best_response_attacker(env.game, _hist(played_seq[-window:], env.game.n_routes))
            _, expl_avg = best_response_attacker(env.game, _hist(played_seq, env.game.n_routes))
            history.append((k + 1, expl_pol, expl_win, expl_avg))
            print(f"    sortie {k+1:5d}: expl policy {expl_pol:.3f} | window {expl_win:.3f} | "
                  f"avg {expl_avg:.3f}   (loss_mixed={sol.value:.3f}, loss_det={sol.loss_det:.3f})",
                  flush=True)
    prot._played = played  # expose the empirical strategy for final eval
    return prot, history, played_seq


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
    p.add_argument("--window", type=int, default=500, help="trailing-window size for empirical play")
    p.add_argument("--json-out", type=str, default="", help="write the result record here (matrix aggregation)")
    args = p.parse_args()
    torch.set_num_threads(4)
    s, t = args.od.split("-")
    band = tuple(float(x) for x in args.edge_vuln_band.split(",")) if args.edge_vuln_band else None
    env = make_interdiction_env(od=(s, t), K=args.K, interception_loss=args.interception_loss,
                                travel_cost_weight=args.travel_cost_weight, k_extra_routes=args.k_extra,
                                edge_vuln_band=band, seed=args.seed)
    sol = solve(env.game)
    print(f"Interdiction {s}->{t} K={args.K} band={band}: {env.game.n_routes} routes, "
          f"{len(env.first_hops)} first hops; oracle loss_det={sol.loss_det:.3f}, "
          f"loss_mixed={sol.value:.3f}, gap={sol.gap:.3f}\n")

    # deterministic shortest-path reference (the operational default) + uniform mixing reference
    # (what UNCALIBRATED randomisation buys; on asymmetric instances it is measurably suboptimal).
    det = np.zeros(env.game.n_routes); det[env.shortest_route_index()] = 1.0
    _, expl_sp = best_response_attacker(env.game, det)
    uni = np.ones(env.game.n_routes) / env.game.n_routes
    _, expl_uni = best_response_attacker(env.game, uni)
    print(f"[shortest_path] exploitability = {expl_sp:.3f} (deterministic; the operational default)")
    print(f"[uniform]       exploitability = {expl_uni:.3f} (uncalibrated mixing reference)\n")

    print("[vanilla] training defender with NO adversary (nominal travel-cost objective)...")
    vprot, _, vseq = train_defender(env, sorties=args.sorties, switch_every=args.switch_every,
                                    batch_size=args.batch_size, seed=args.seed, adversarial=False,
                                    eval_every=0, sol=sol, reward_scale=args.reward_scale,
                                    lr_actor=args.lr_actor, window=args.window)
    vfin = final_metrics(vprot, env, vseq, args.window)
    print(f"[vanilla] final expl: policy {vfin['expl_policy']:.3f} | window {vfin['expl_window']:.3f} "
          f"| avg {vfin['expl_avg']:.3f}\n")

    print("[sacred] training defender vs the ORACLE best-response interdictor (fictitious play)...")
    sprot, hist, sseq = train_defender(env, sorties=args.sorties, switch_every=args.switch_every,
                                       batch_size=args.batch_size, seed=args.seed, adversarial=True,
                                       eval_every=args.eval_every, sol=sol, reward_scale=args.reward_scale,
                                       lr_actor=args.lr_actor, window=args.window)
    sfin = final_metrics(sprot, env, sseq, args.window)
    print(f"\n=== RESULT (Kaliningrad {s}->{t}, K={args.K}, band={band}, seed={args.seed}) ===")
    print(f"  arm             expl_policy  expl_window  expl_avg")
    print(f"  shortest_path   {expl_sp:11.3f}  {expl_sp:11.3f}  {expl_sp:8.3f}")
    print(f"  uniform         {expl_uni:11.3f}  {expl_uni:11.3f}  {expl_uni:8.3f}")
    print(f"  vanilla         {vfin['expl_policy']:11.3f}  {vfin['expl_window']:11.3f}  {vfin['expl_avg']:8.3f}")
    print(f"  sacred          {sfin['expl_policy']:11.3f}  {sfin['expl_window']:11.3f}  {sfin['expl_avg']:8.3f}")
    print(f"  equilibrium (loss_mixed) = {sol.value:.3f}; loss_det = {sol.loss_det:.3f}")
    print(f"  -> sacred policy-distribution distance to equilibrium: "
          f"{abs(sfin['expl_policy'] - sol.value):.3f}")
    if args.json_out:
        record = {"od": args.od, "K": args.K, "band": band, "seed": args.seed,
                  "sorties": args.sorties, "window": args.window,
                  "loss_det": sol.loss_det, "loss_mixed": sol.value,
                  "equilibrium_defender": sol.defender_strategy.tolist(),
                  "arms": {"shortest_path": {"expl_policy": expl_sp},
                           "uniform": {"expl_policy": expl_uni},
                           "vanilla": vfin, "sacred": {**sfin, "history": hist}}}
        Path(args.json_out).write_text(json.dumps(record, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
