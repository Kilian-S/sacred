"""EXACT re-evaluation of the gen09-HEADLINE best-checkpoint (eval-only, NO training).

The locked multi-convoy headline (best-checkpoint TAP 0.283 +/- 0.021) was measured with a
400-sample Monte-Carlo occupancy estimate (policy_occ_dist) and the best checkpoint was min-selected
over per-eval readings of that SAME noisy estimator, which (a) adds multinomial sampling noise of
the same order as the reported seed spread and (b) biases the selected minimum downward (selection
on noise). In FLEET-ROUTE mode the occupancy distribution is EXACT and cheap: the fleet stacks on
the leader, so occ-dist = the leader's route distribution (one forward pass) mapped onto the
stacked occupancies. This probe loads every saved per-eval actor checkpoint of the 3 headline seeds,
computes the EXACT per-eval occupancy distribution, rebuilds the TAP series (mean of the trailing
TAP_K=5 exact distributions), and reports the exact best-checkpoint TAP per seed, next to the
MC values recorded in the ledger.

Run: PYTHONPATH=. .venv/bin/python scratch/gen09_exact_reeval.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from src.agents.networks import featurize_state
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.multiconvoy_oracle import best_response_attacker_multi
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

TAP_K = 5
RUN_DIR = Path("models/runs/gen09_multiconvoy")

env = make_multiconvoy_env(od=("62", "97"), N=3, K=1, k_extra_routes=8, menu_select=True,
                           edge_vuln_band=(0.15, 0.95), interception_loss=10.0, seed=0)
R = env.game.n_routes
N = env.config.N
# stacked-occupancy index for each route r: occupancy = N convoys all on r.
stack_occ_idx = [env._occ_index[tuple(N if i == r else 0 for i in range(R))] for r in range(R)]

env.reset()
obs = env.observe()
menu = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]


def exact_leader_dist(actor_state: dict) -> np.ndarray:
    """Leader route distribution from a saved actor checkpoint (exact, one forward pass)."""
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
                          autotune_alpha=True, alpha_init=1.0, device="cpu")
    prot.actor.menu_routes = menu
    if any(k == "follow_w" for k in actor_state):
        prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.load_state_dict(actor_state)
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    node_ids = list(obs["nodes"].keys())
    n2i = {n: i for i, n in enumerate(node_ids)}
    active = n2i[obs["trucks"][0]["current_node"]]
    taken = torch.zeros(R)  # leader decision: no earlier convoys
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, active, list(range(R)), taken)
    return probs.numpy()


print(f"gen09-HEADLINE exact re-eval: 62->97 k8 N={N} K=1 fleet-route, {R} routes")
print(f"oracle: loss_mixed n/a here; exploitability under oracle BR interdictor\n")

best_taps = []
for seed in (0, 1, 2):
    ckpt_dir = RUN_DIR / f"headline_seed{seed}_ckpts"
    ckpts = sorted(ckpt_dir.glob("actor_ep*.pt"),
                   key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    if not ckpts:
        print(f"seed {seed}: NO checkpoints found under {ckpt_dir}"); continue
    pol_hist, rows = [], []
    for cp in ckpts:
        lead = exact_leader_dist(torch.load(cp, map_location="cpu"))
        occ_dist = np.zeros(len(env.occupancies))
        for r in range(R):
            occ_dist[stack_occ_idx[r]] = lead[r]
        pol_hist.append(occ_dist)
        _, expl = best_response_attacker_multi(env.obj_matrix, occ_dist)
        _, expl_tap = best_response_attacker_multi(env.obj_matrix, np.mean(pol_hist[-TAP_K:], axis=0))
        rows.append((int(re.search(r"ep(\d+)", cp.name).group(1)), expl, expl_tap))
    best_tap = min(r[2] for r in rows)
    best_at = next(r[0] for r in rows if r[2] == best_tap)
    best_exp = min(r[1] for r in rows)
    best_exp_at = next(r[0] for r in rows if r[1] == best_exp)
    best_taps.append(best_tap)
    # ledger MC values for comparison
    mc = json.load(open(RUN_DIR / f"headline_seed{seed}.json"))["fleet_route"]
    print(f"seed {seed}: EXACT best-ckpt TAP {best_tap:.3f} @ sortie {best_at} "
          f"(exact best single-ckpt {best_exp:.3f} @ {best_exp_at}) | "
          f"ledger MC best TAP {mc['best_tap']:.3f} @ {mc['best_tap_sortie']}")
    print("   per-eval exact (sortie, expl, TAP): "
          + " ".join(f"({s},{e:.2f},{t:.2f})" for s, e, t in rows))

if best_taps:
    print(f"\nEXACT best-checkpoint TAP mean {np.mean(best_taps):.3f} +/- {np.std(best_taps):.3f} "
          f"(pop std, {len(best_taps)} seeds)   [ledger MC: 0.283 +/- 0.021]")
