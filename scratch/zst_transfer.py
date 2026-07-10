"""ZST step 0 (aim-level promise; F4): zero-shot transfer of the post-fix single-convoy SACRED
policy to a HELD-OUT OD pair, scored against that pair's own oracle ladder. EVAL-ONLY.

Source: the gen10-SC-config sacred actor retrained with --save-actor (33->71, k8 shared-edge, hard
interception, walk mode, smooth FP; post-node-ordering-fix so the embeddings are meaningful).
Held-out target: **110->135 k8 hard** = the B2-S instance pre-registered in the gen08 ledger and
never run (anchors there: equilibrium 0.333, uniform 0.818, best cost-mixture >= 0.862, shortest
1.000), so this one evaluation closes B2-S and ZST step 0 together.

Readout per OD: the policy's EXACT deployable route mixture (trie branch product) under the oracle
best-response interdictor, beside shortest/uniform/equilibrium anchors, a random-init actor
reference (transfer must beat an untrained net to mean anything), and the home-OD sanity row.
Expectation set honestly (CRITIQUE_PREFREEZE §7): partial transfer at best; the policy has never
seen the target OD's geometry, and hard interception carries no vulnerability map to condition on.

Run: PYTHONPATH=. .venv/bin/python scratch/zst_transfer.py <path/to/actor.pt>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.interdiction import make_interdiction_env

ACTOR = sys.argv[1] if len(sys.argv) > 1 else "models/runs/gen11_night/zst_source_actor.pt"
HOME = ("33", "71")
HELD_OUT = ("110", "135")


def hop_probs(prot, obs, allowed):
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim); pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    active = n2i[obs["trucks"][0]["current_node"]]
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, active, [n2i[n] for n in allowed])
    return {allowed[i]: float(probs[i]) for i in range(len(allowed))}


def make_prot(seed=0):
    torch.manual_seed(seed)
    return ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu")


def eval_on(prot, od):
    env = make_interdiction_env(od=od, K=1, k_extra_routes=8)
    sol = solve(env.game)
    d = env.walk_distribution(lambda node, allowed: hop_probs(prot, env.observe_at(node), allowed))
    _, expl = best_response_attacker(env.game, d)
    det = np.zeros(env.game.n_routes); det[env.shortest_route_index()] = 1.0
    _, sp = best_response_attacker(env.game, det)
    uni = np.ones(env.game.n_routes) / env.game.n_routes
    _, un = best_response_attacker(env.game, uni)
    cost = float(np.asarray(d) @ env.game.travel_cost)
    return {"od": f"{od[0]}-{od[1]}", "expl": float(expl), "cost": cost, "shortest": float(sp),
            "uniform": float(un), "equilibrium": float(sol.value), "dist": np.asarray(d).tolist()}


trained = make_prot()
trained.actor.load_state_dict(torch.load(ACTOR, map_location="cpu"))
rows = {"home_sanity": eval_on(trained, HOME),
        "transfer": eval_on(trained, HELD_OUT),
        "random_init_reference": eval_on(make_prot(seed=123), HELD_OUT)}

print("=== ZST STEP 0: zero-shot transfer of the post-fix single-convoy policy ===")
print(f"source actor: {ACTOR} (trained on {HOME[0]}->{HOME[1]} k8 hard, walk, smooth FP)")
for name, r in rows.items():
    print(f"\n[{name}] OD {r['od']}: policy expl {r['expl']:.3f} @ cost {r['cost']:.1f}  "
          f"(shortest {r['shortest']:.3f} | uniform {r['uniform']:.3f} | eq {r['equilibrium']:.3f})")
tr, rnd = rows["transfer"], rows["random_init_reference"]
print(f"\nTRANSFER READ: held-out expl {tr['expl']:.3f} vs random-init {rnd['expl']:.3f}, "
      f"uniform {tr['uniform']:.3f}, shortest {tr['shortest']:.3f}, eq {tr['equilibrium']:.3f}")
with open("models/runs/zst_step0.json", "w") as f:
    json.dump(rows, f, indent=2)
print("[written] models/runs/zst_step0.json")
