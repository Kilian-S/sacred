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


def empirical_exploitability(prot, env) -> tuple[float, np.ndarray]:
    """Exploitability of the defender's empirical average play (the deployable mixed strategy)."""
    played = getattr(prot, "_played", None)
    if played is None or played.sum() == 0:
        d = np.ones(env.game.n_routes) / env.game.n_routes
    else:
        d = played / played.sum()
    _, expl = best_response_attacker(env.game, d)
    return expl, d


def _prot_transition(obs, first_hop, reward, mask) -> SMDPTransition:
    return SMDPTransition(agent="protagonist", state=obs, action={0: first_hop}, reward=reward,
                          next_state={}, done=True, elapsed_ticks=1,
                          action_mask={"protagonist": mask}, info={})


def train_defender(env, *, sorties, switch_every, batch_size, seed, adversarial, eval_every, sol,
                   reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    prot = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=reward_scale, lr_actor=lr_actor,
                          autotune_alpha=autotune_alpha, alpha_init=alpha_init, device="cpu")
    committed = None
    history = []
    played = np.zeros(env.game.n_routes)   # empirical route-play histogram (fictitious-play average)
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
        if adversarial:
            env.commit(committed)
            out = env.resolve_first_hop(fh)
        else:
            # no adversary: reward is the nominal travel cost only (drives to the shortest route).
            out = type("O", (), {"defender_reward": -env.config.travel_cost_weight * env.game.travel_cost[ri]})()
        prot.replay_buffer.push(_prot_transition(obs, fh, out.defender_reward, mask))
        prot.update(batch_size)
        if eval_every and (k + 1) % eval_every == 0:
            # exploitability of the empirical average play over a trailing window (the deployable
            # mixed strategy), the fictitious-play quantity that converges to the equilibrium.
            w = min(k + 1, 1000)
            recent = _recent_hist(env, prot, played, k)  # trailing-window empirical distribution
            _, expl_avg = best_response_attacker(env.game, recent)
            history.append((k + 1, expl_avg))
            print(f"    sortie {k+1:5d}: exploitability(avg-play) {expl_avg:.3f}  "
                  f"(loss_mixed={sol.value:.3f}, loss_det={sol.loss_det:.3f})", flush=True)
    prot._played = played  # expose the empirical strategy for final eval
    return prot, history


def _recent_hist(env, prot, played, k):
    """Empirical route distribution over all sorties so far (running average)."""
    tot = played.sum()
    return played / tot if tot > 0 else np.ones(env.game.n_routes) / env.game.n_routes


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
    args = p.parse_args()
    torch.set_num_threads(4)
    s, t = args.od.split("-")
    env = make_interdiction_env(od=(s, t), K=args.K, interception_loss=args.interception_loss,
                                travel_cost_weight=args.travel_cost_weight, k_extra_routes=args.k_extra)
    sol = solve(env.game)
    print(f"Interdiction {s}->{t} K={args.K}: {env.game.n_routes} routes, {len(env.first_hops)} first "
          f"hops; oracle loss_det={sol.loss_det:.3f}, loss_mixed={sol.value:.3f}, gap={sol.gap:.3f}\n")

    # shortest-path reference (deterministic).
    det = np.zeros(env.game.n_routes); det[env.shortest_route_index()] = 1.0
    _, expl_sp = best_response_attacker(env.game, det)
    print(f"[shortest_path] exploitability = {expl_sp:.3f} (deterministic; the operational default)\n")

    print("[vanilla] training defender with NO adversary (nominal travel-cost objective)...")
    vprot, _ = train_defender(env, sorties=args.sorties, switch_every=args.switch_every,
                              batch_size=args.batch_size, seed=args.seed, adversarial=False,
                              eval_every=0, sol=sol, reward_scale=args.reward_scale, lr_actor=args.lr_actor)
    expl_vanilla, dv = empirical_exploitability(vprot, env)
    print(f"[vanilla] final exploitability (avg play) = {expl_vanilla:.3f}\n")

    print("[sacred] training defender vs the ORACLE best-response interdictor (fictitious play)...")
    sprot, hist = train_defender(env, sorties=args.sorties, switch_every=args.switch_every,
                                 batch_size=args.batch_size, seed=args.seed, adversarial=True,
                                 eval_every=args.eval_every, sol=sol, reward_scale=args.reward_scale, lr_actor=args.lr_actor)
    expl_sacred, ds = empirical_exploitability(sprot, env)
    print(f"\n=== RESULT (Kaliningrad {s}->{t}, K={args.K}) ===")
    print(f"  shortest_path exploitability : {expl_sp:.3f}")
    print(f"  vanilla-SAC   exploitability : {expl_vanilla:.3f}")
    print(f"  SACRED        exploitability : {expl_sacred:.3f}   (equilibrium loss_mixed = {sol.value:.3f})")
    print(f"  -> adversarial training cut interception {expl_sp:.0%} -> {expl_sacred:.0%}"
          f" (distance to equilibrium {abs(expl_sacred - sol.value):.3f})")


if __name__ == "__main__":
    main()
