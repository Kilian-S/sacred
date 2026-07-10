"""Multi-convoy FLEET TRAVEL COST column + vanilla best-checkpoint row (eval-only; closes
CRITIQUE_PREFREEZE §3.4-3.5). For each arm the expected fleet travel cost per sortie (N x expected
route cost under its occupancy mixture) is reported beside its exploitability, so the ladder gains
the cost axis the single-convoy tables always had:
  * pre-fix headline (gen09-HEADLINE best-ckpt, exact re-eval mixtures from the saved ckpts);
  * post-fix gen10-MC best-ckpt mixtures;
  * ALNS plan (its chosen deterministic assignment);
  * shortest-path stack; equilibrium mixture.
Also: vanilla best-checkpoint TAP (selection symmetry with SACRED) from the gen10-VAN history.

Run: PYTHONPATH=. .venv/bin/python scratch/fleet_cost_probe.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.baselines.multiconvoy_planners import alns_fleet_planner, shortest_path_fleet
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

TAP_K = 5
env = make_multiconvoy_env(od=("62", "97"), N=3, K=1, k_extra_routes=8, menu_select=True,
                           edge_vuln_band=(0.15, 0.95), interception_loss=10.0, seed=0)
R, N = env.game.n_routes, env.config.N
cost = np.asarray(env.game.travel_cost)
env.reset(); obs = env.observe()
# TWO indexing conventions: post-fix checkpoints were trained with featurize's SORTED row order;
# PRE-FIX checkpoints (gen09) memorised the legacy insertion-order (permuted) readout and must be
# evaluated under it (the gen09_exact_reeval convention) or their learned weights are meaningless.
menu_sorted = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]
_ins = {str(n): i for i, n in enumerate(obs["nodes"].keys())}
menu_legacy = [torch.tensor([_ins[str(n)] for n in route if str(n) in _ins], dtype=torch.long)
               for route in env.game.routes]


def leader_dist(actor_state, legacy_order=False):
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4, device="cpu")
    prot.actor.menu_routes = menu_legacy if legacy_order else menu_sorted
    if any(k == "follow_w" for k in actor_state):
        prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    if any(k == "route_feat_w" for k in actor_state):
        prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2))
        c = np.asarray(env.game.travel_cost, float); v = env.game.payoff.max(axis=1)
        mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
        prot.actor.route_feats = torch.tensor(np.stack([mm(c), mm(v)], axis=1), dtype=torch.float32)
    if any(k == "route_bias" for k in actor_state):
        prot.actor.route_bias = torch.nn.Parameter(torch.zeros(R))
    prot.actor.load_state_dict(actor_state)
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    active = (_ins[str(obs["trucks"][0]["current_node"])] if legacy_order
              else node_index_map(obs)[obs["trucks"][0]["current_node"]])
    prot.actor.eval()
    with torch.no_grad():
        p, _ = prot.actor(pyg, active, list(range(R)), torch.zeros(R))
    return p.numpy()


def best_ckpt_fleet_cost(run_dir_pattern, label, legacy_order=False):
    out = []
    for seed in (0, 1, 2):
        ckpt_dir = Path(run_dir_pattern.format(seed=seed))
        ckpts = sorted(ckpt_dir.glob("actor_ep*.pt"),
                       key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
        if not ckpts:
            continue
        pol, best = [], None
        from src.baselines.multiconvoy_oracle import best_response_attacker_multi
        for cp in ckpts:
            lead = leader_dist(torch.load(cp, map_location="cpu"), legacy_order=legacy_order)
            occ = np.zeros(len(env.occupancies))
            for r in range(R):
                occ[env._occ_index[tuple(N if i == r else 0 for i in range(R))]] = lead[r]
            pol.append((occ, lead))
            tap_occ = np.mean([o for o, _ in pol[-TAP_K:]], axis=0)
            tap_lead = np.mean([l for _, l in pol[-TAP_K:]], axis=0)
            _, expl = best_response_attacker_multi(env.obj_matrix, tap_occ)
            if best is None or expl < best[0]:
                best = (expl, tap_lead)
        fleet_cost = N * float(best[1] @ cost)
        out.append((best[0], fleet_cost))
        print(f"  {label} seed {seed}: best-ckpt TAP {best[0]:.3f} | fleet cost {fleet_cost:.1f}")
    if out:
        print(f"  {label} MEAN: TAP {np.mean([o[0] for o in out]):.3f} | "
              f"fleet cost {np.mean([o[1] for o in out]):.1f} +/- {np.std([o[1] for o in out]):.1f}")
    return out


print("=== fleet travel cost per sortie (62->97 k8, N=3; route costs "
      f"{cost.min():.1f}-{cost.max():.1f}) ===")
sol = solve_multiconvoy(env.game, N, "mission")
eq_lead = np.zeros(R)
for i, occ in enumerate(sol.occupancies):
    nz = [r for r, c in enumerate(occ) if c > 0]
    if len(nz) == 1 and occ[nz[0]] == N:
        eq_lead[nz[0]] += sol.defender_strategy[i]
# the equilibrium may put mass on non-stacked occupancies; use the full mixture expected cost:
eq_cost = float(sum(sol.defender_strategy[i] * sum(occ[r] * cost[r] for r in range(R))
                    for i, occ in enumerate(sol.occupancies)))
sp = shortest_path_fleet(env.game, N)
sp_cost = float(sum(cost[r] for r in sp))
plan = alns_fleet_planner(env.game, N, "mission", seed=0)
alns_cost = float(sum(cost[r] for r in plan.assignment))
print(f"  shortest-path stack: cost {sp_cost:.1f} (expl 0.973)")
print(f"  ALNS plan {plan.assignment}: cost {alns_cost:.1f} (expl {plan.exploitability:.3f})")
print(f"  equilibrium mixture: expected cost {eq_cost:.1f} (expl {sol.loss_mixed:.3f})")
print("\npre-fix gen09-HEADLINE (exact re-eval of saved ckpts; LEGACY indexing = their training convention):")
pre = best_ckpt_fleet_cost("models/runs/gen09_multiconvoy/headline_seed{seed}_ckpts", "pre-fix",
                           legacy_order=True)
print("\npost-fix gen10-MC:")
post = best_ckpt_fleet_cost("models/runs/gen10_postfix/mc_seed{seed}_ckpts", "post-fix")

print("\n=== vanilla best-checkpoint TAP (selection symmetry; gen10-VAN seed 0) ===")
van = json.load(open("models/runs/gen10_postfix/van_seed0.json"))["vanilla"]
taps = [h[2] for h in van["history"]]
print(f"  vanilla per-eval TAP min {min(taps):.3f} @ eval {int(np.argmin(taps)) + 1} | final {taps[-1]:.3f}")
json.dump({"sp_cost": sp_cost, "alns_cost": alns_cost, "eq_cost": eq_cost,
           "prefix": pre, "postfix": post, "vanilla_best_tap": min(taps)},
          open("models/runs/fleet_cost_probe.json", "w"), indent=2)
print("[written] models/runs/fleet_cost_probe.json")
