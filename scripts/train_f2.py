#!/usr/bin/env python3
"""F2 (gen20): ONE clean LEARNED-INTERDICTOR co-evolution attempt (Obj-1's "environment-altering
antagonist AGENT", which is an ORACLE best response in every banked positive result).

A learned antagonist (a menu-select SAC policy over the K=1 interdiction sets = candidate edges,
scored by its edges' node embeddings through the shared GNN) replaces the oracle BR as the SPARRING
PARTNER during training, co-evolving with the fleet-route defender (ATLA / simultaneous learning).
**EVALUATION STAYS ORACLE-BR PORTFOLIO-MAX** in every reported row - a weak learned attacker must
not be able to flatter the defender. The defender arm is the gen14 fleet-route config verbatim, so
its best-checkpoint TAP is directly comparable to the oracle-trained headline (0.256 on 35-159).

HARD GATE (pre-registered, Kilian): ONE attempt, no chase, whatever the outcome. PASS = the
learned-adversary-trained defender's best-checkpoint TAP (under the ORACLE BR) lands within a stated
margin of the oracle-trained reference; FAIL = the honest oracle-bounded sentence for Obj-1, with
the learned attacker's strength (its exploitation vs the oracle BR's) reported.

Run: PYTHONPATH=. python scripts/train_f2.py --sorties 4000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.train_multiconvoy import _transition, exact_fleet_occ_dist, route_one
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, objective_value, solve_multiconvoy)
from src.baselines.multiconvoy_planners import classical_baselines
from src.env.smdp_wrapper import SMDPTransition
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

TAP_K = 5


def antag_menu_idx(env):
    """Per-interdiction-set node-index groups (for the antagonist's menu head): iset j -> the sorted
    featurise-row indices of the nodes on its edges. K=1 -> each iset is one edge (2 nodes)."""
    obs = env.observe()
    pos = {str(n): i for i, n in enumerate(sorted(obs["nodes"].keys()))}
    groups = []
    for iset in env.game.interdiction_sets:
        nodes = set()
        for edge in iset:
            for n in edge:
                nodes.add(n)
        groups.append(torch.tensor(sorted(pos[str(n)] for n in nodes if str(n) in pos),
                                   dtype=torch.long))
    return groups


def antag_pick(antag, obs, n_isets, deterministic=False):
    """The learned antagonist selects an interdiction-set index (menu-select over isets)."""
    mask = {0: list(range(n_isets))}
    return int(antag.select_action(obs, mask, deterministic=deterministic)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--od", default="35-159"); p.add_argument("--N", type=int, default=3)
    p.add_argument("--K", type=int, default=1); p.add_argument("--k-extra", type=int, default=8)
    p.add_argument("--band", default="0.15,0.95"); p.add_argument("--sorties", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=200); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--leader-ent-frac", type=float, default=0.5)
    p.add_argument("--leader-alpha-floor", type=float, default=0.20)
    p.add_argument("--antag-ent-frac", type=float, default=0.5,
                   help="antagonist target-entropy fraction (keep it exploring the iset menu)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json-out", default=""); p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    s, t = args.od.split("-"); band = tuple(float(x) for x in args.band.split(","))
    env = make_multiconvoy_env(od=(s, t), N=args.N, K=args.K, k_extra_routes=args.k_extra,
                               menu_select=True, edge_vuln_band=band,
                               interception_loss=args.interception_loss, seed=args.seed)
    R = env.game.n_routes; n_isets = len(env.game.interdiction_sets); N = args.N
    sol = solve_multiconvoy(env.game, N, "mission"); base = classical_baselines(env.game, N, "mission")
    print(f"[F2] {args.od} R={R} isets={n_isets} | oracle: ALNS {base['alns']:.3f} eq {sol.loss_mixed:.3f} "
          f"| oracle-trained ref (gen14 35-159) best-ckpt TAP ~0.256", flush=True)

    # DEFENDER: gen14 fleet-route menu-select (role alphas, floor), comparable to the headline.
    dfn = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                         reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                         device="cpu", role_alpha=True, lr_alpha=5e-3, alpha_floor=args.leader_alpha_floor)
    dmenu = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]
    for net in (dfn.actor, dfn.q1, dfn.q2, dfn.target_q1, dfn.target_q2):
        net.menu_routes = dmenu
    leader_te = args.leader_ent_frac * math.log(R)

    # ANTAGONIST: a LEARNED menu-select policy over the K=1 interdiction sets (candidate edges),
    # scored by its edges' node embeddings through its own GNN. Maximises defender mission-failure.
    ant = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                         reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                         device="cpu", target_entropy=args.antag_ent_frac * math.log(n_isets))
    amenu = antag_menu_idx(env)
    for net in (ant.actor, ant.q1, ant.q2, ant.target_q1, ant.target_q2):
        net.menu_routes = amenu

    def antag_obs():  # the antagonist observes the graph with the base as its reference node
        env.reset(); o = dict(env.observe()); o["active_truck"] = 0
        # strip the DEFENDER's route menu the env attaches, else select_action's per-instance
        # override clobbers the antagonist's ISET menu (amenu) with it.
        o.pop("menu_route_node_idx", None); o.pop("menu_route_feats", None)
        return o

    hist = []
    for k in range(args.sorties):
        env.reset()
        # antagonist commits an iset (learned)
        aobs = antag_obs()
        j = antag_pick(ant, aobs, n_isets)
        env.commit(j)
        # defender routes the fleet (fleet-route stack) vs the committed iset
        steps, occ, _ = route_one(dfn, env, fleet_route=True)
        pay = env.game.payoff[:, j]
        mission = objective_value(np.asarray(occ), pay, N, env.config.objective, env.config.threshold_m)
        d_reward = -args.interception_loss * mission
        a_reward = +args.interception_loss * mission
        # push defender transitions (fleet-route: role entropy on the leader; followers copy)
        for obs_j, ci_j, _, _ in steps:
            obs_j["target_entropy"] = leader_te if ci_j == 0 else args.leader_ent_frac * 0.1 * math.log(R)
            obs_j["alpha_group"] = 0 if ci_j == 0 else 1
        for i, (obs, ci, hop, mask) in enumerate(steps):
            last = i == N - 1
            nobs, nci, nmask = ((steps[i + 1][0], steps[i + 1][1], steps[i + 1][3])
                                if not last else (None, None, None))
            dfn.replay_buffer.push(_transition(obs, ci, hop, mask, d_reward if last else 0.0,
                                               nobs, nci, nmask, last))
        dfn.update(args.batch_size)
        # push antagonist transition (single decision, terminal)
        ant.replay_buffer.push(SMDPTransition(
            agent="protagonist", state=aobs, action={0: j}, reward=a_reward, next_state={},
            done=True, elapsed_ticks=1, action_mask={"protagonist": {0: list(range(n_isets))}}, info={}))
        ant.update(args.batch_size)

        if (k + 1) % args.eval_every == 0:
            # DEFENDER exploitability under the ORACLE BR (portfolio-max discipline), exact.
            d, _ = exact_fleet_occ_dist(dfn, env)
            _, d_expl = best_response_attacker_multi(env.obj_matrix, d)
            # ANTAGONIST strength: its exploitation of the CURRENT defender vs the oracle BR's.
            # sample the antagonist's iset distribution, score its expected mission-failure on d.
            with torch.no_grad():
                ao = antag_obs()
                pyg = featurize_state(ao, 0).to(ant.device)
                pyg.x = _clip_x(pyg.x, ant.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, ant.edge_in_dim)
                n2i = node_index_map(ao); active = n2i[ao["trucks"][0]["current_node"]]
                ant.actor.eval(); aprobs, _ = ant.actor(pyg, active, list(range(n_isets))); ant.actor.train()
            a_exploit = float(d @ env.obj_matrix @ aprobs.numpy())  # learned antag's exploitation
            hist.append((k + 1, float(d_expl), a_exploit, float(dfn.alpha), float(ant.alpha)))
            if args.ckpt_dir:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(dfn.actor.state_dict(), str(Path(args.ckpt_dir) / f"defender_ep{k+1}.pt"))
            print(f"  sortie {k+1:5d}: defender expl (oracle BR) {d_expl:.3f} | learned-antag exploit "
                  f"{a_exploit:.3f} (oracle BR {d_expl:.3f}) | alpha d{dfn.alpha:.2f}/a{ant.alpha:.2f}",
                  flush=True)

    best = min((h[1] for h in hist), default=float("nan"))
    best_at = next((h[0] for h in hist if h[1] == best), None)
    # learned-antag strength ratio at the best-defender checkpoint (how close to the oracle BR)
    tail = hist[-min(len(hist), 6):]
    ant_ratio = float(np.mean([h[2] / max(h[1], 1e-9) for h in tail])) if tail else float("nan")
    print(f"\n=== F2 ({args.od}, seed {args.seed}) ===")
    print(f"  defender best-ckpt expl (oracle BR) {best:.3f} @ sortie {best_at} "
          f"(oracle-trained ref ~0.256, ALNS {base['alns']:.3f}, eq {sol.loss_mixed:.3f})")
    print(f"  learned antagonist strength: exploits the defender to {ant_ratio:.2f}x the oracle BR "
          f"(1.0 = as strong as the oracle interdictor)")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"od": args.od, "seed": args.seed, "best_defender_expl": best, "best_at": best_at,
             "alns": base["alns"], "eq": sol.loss_mixed, "oracle_trained_ref": 0.256,
             "antag_strength_ratio": ant_ratio, "history": hist}, indent=2))


if __name__ == "__main__":
    main()
