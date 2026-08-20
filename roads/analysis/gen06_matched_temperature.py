#!/usr/bin/env python3
"""gen06: matched-temperature diagnostic for the gen06 selected checkpoints. The vanilla and
scripted-adversarial arms converged at different SAC temperatures, so this evaluates both arms
at matched sampling temperatures (tau=1.0, tau=0.5, and argmax) on 30 paired test instances per
pair, to check whether the robustness gap is a temperature artefact rather than a difference in
what the policies know.

Run: PYTHONPATH=. .venv/bin/python analysis/gen06_matched_temperature.py  (~5-15 min, eval only)
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import torch

from scripts.evaluate_portfolio import (
    TEST_SEED_BASE,
    _load_protagonist,
    _problem_setup,
    run_matrix,
)
from src.agents.networks import featurize_state
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x

PAIRS = {
    "pair0": {
        "vanilla": "models/runs/gen06_dynassign_matrix/vanilla_seed0/snapshots/protagonist_ep750.pt",
        "scripted": "models/runs/gen06_dynassign_matrix/dynassign_scripted_seed0/snapshots/protagonist_ep450.pt",
    },
    "pair1": {
        "vanilla": "models/runs/gen06_dynassign_matrix/vanilla_seed1/snapshots/protagonist_ep100.pt",
        "scripted": "models/runs/gen06_dynassign_matrix/dynassign_scripted_seed1/snapshots/protagonist_ep200.pt",
    },
    "pair2": {
        "vanilla": "models/runs/gen06_dynassign_matrix/vanilla_seed2/snapshots/protagonist_ep100.pt",
        "scripted": "models/runs/gen06_dynassign_matrix/dynassign_scripted_seed2/snapshots/protagonist_ep600.pt",
    },
}
TAUS: list[float | None] = [1.0, 0.5, None]  # None = argmax
ATTACKS = ["none", "targeted"]
INSTANCES = 30


def _select_with_temperature(agent: ProtagonistSAC, observation, action_mask,
                             tau: float | None):
    """Temperature-controlled variant of ProtagonistSAC.select_action (tau=None means argmax)."""
    active_truck = observation.get("active_truck")
    if active_truck is None or active_truck not in action_mask:
        return {}
    allowed_nodes = action_mask[active_truck]
    if not allowed_nodes:
        return {}
    pyg_data = featurize_state(observation, active_truck).to(agent.device)
    pyg_data.x = _clip_x(pyg_data.x, agent.node_in_dim)
    pyg_data.edge_attr = _clip_ea(pyg_data.edge_attr, agent.edge_in_dim)
    node_ids = list(observation["nodes"].keys())
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    active_idx = node_to_idx[observation["trucks"][active_truck]["current_node"]]
    mask_idxs = [node_to_idx[nid] for nid in allowed_nodes]
    agent.actor.eval()
    with torch.no_grad():
        probs, _ = agent.actor(pyg_data, active_idx, mask_idxs)
    agent.actor.train()
    if len(probs) == 0:
        return {}
    if tau is None:
        action_idx = torch.argmax(probs).item()
    else:
        scaled = torch.clamp(probs, min=1e-12) ** (1.0 / tau)
        scaled = scaled / scaled.sum()
        action_idx = torch.distributions.Categorical(scaled).sample().item()
    return {active_truck: allowed_nodes[action_idx]}


def temp_policy(smdp, agent: ProtagonistSAC, tau: float | None):
    """Temperature-controlled variant of the sac_protagonist_policy in scripts/evaluate_portfolio.py."""
    def policy(event):
        env = smdp.env
        mask = event.protagonist_action_mask
        nodes = event.observation["nodes"]
        actions: dict = {}
        claimed: set = set()
        projected = dict(event.observation)
        projected["trucks"] = {tid: dict(t) for tid, t in event.observation["trucks"].items()}
        for tid in sorted(mask):
            truck = env.trucks[tid]
            is_routing = smdp.config.routing_mode == "hybrid" and truck.assigned_target is not None
            opts = [n for n in mask[tid] if (is_routing or n not in claimed)]
            if not opts:
                continue
            truck_mask = {tid: opts}
            projected["active_truck"] = tid
            projected["allowed_destinations"] = {"protagonist": dict(truck_mask)}
            chosen = _select_with_temperature(agent, projected, truck_mask, tau)
            actions.update(chosen)
            node = chosen.get(tid)
            if node is not None:
                projected["trucks"][tid]["destination"] = node
                projected["trucks"][tid]["current_node"] = None
                if not is_routing and nodes[node].get("demand", 0.0) > 0.0:
                    claimed.add(node)
        return actions
    return policy


def _mean_ci(xs: list[float]) -> tuple[float, float]:
    m = statistics.mean(xs)
    ci = 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, ci


def main() -> None:
    cfg, make_env_for_seed, _greedy, _ = _problem_setup("dynassign", arrival_rate=0.06)
    seeds = [TEST_SEED_BASE + i for i in range(INSTANCES)]
    out: dict = {}
    t0 = time.time()
    for tau in TAUS:
        tau_key = "argmax" if tau is None else f"tau{tau}"
        out[tau_key] = {}
        for pair, arms in PAIRS.items():
            factories = {}
            for arm, path in arms.items():
                agent = _load_protagonist(path)
                factories[arm] = lambda smdp, _a=agent, _t=tau: temp_policy(smdp, _a, _t)
            res = run_matrix(factories, ATTACKS, {}, cfg, make_env_for_seed, seeds,
                             rollouts=1, quiet=True)
            entry = {}
            for arm in arms:
                w0 = res[arm]["none"]
                wt = res[arm]["targeted"]
                d = [a - b for a, b in zip(wt, w0)]
                entry[arm] = {"W_none": _mean_ci(w0), "W_targeted": _mean_ci(wt),
                              "D_targeted": _mean_ci(d), "raw": res[arm]}
            dd = [
                (tv - nv) - (ts - ns)
                for tv, nv, ts, ns in zip(res["vanilla"]["targeted"], res["vanilla"]["none"],
                                          res["scripted"]["targeted"], res["scripted"]["none"])
            ]
            entry["dD_targeted"] = _mean_ci(dd)
            entry["_dd_raw"] = dd
            out[tau_key][pair] = entry
            m, ci = entry["dD_targeted"]
            print(f"[{time.time() - t0:7.1f}s] {tau_key:7s} {pair}: "
                  + " | ".join(
                      f"{arm} W0={entry[arm]['W_none'][0]:.0f} D_tgt={entry[arm]['D_targeted'][0]:+.0f}"
                      for arm in arms)
                  + f" | dD_targeted={m:+.0f} ± {ci:.0f}")
        pooled = [d for pair in PAIRS for d in out[tau_key][pair]["_dd_raw"]]
        pm, pci = _mean_ci(pooled)
        out[tau_key]["pooled_dD_targeted"] = (pm, pci)
        print(f"          {tau_key:7s} POOLED dD_targeted = {pm:+.0f} ± {pci:.0f} (n={len(pooled)})")

    for tau_key in out:  # strip raw lists from the JSON summary, keep pooled + per-pair stats
        for pair in list(out[tau_key]):
            if isinstance(out[tau_key][pair], dict):
                out[tau_key][pair].pop("_dd_raw", None)
                for arm in PAIRS[pair] if pair in PAIRS else []:
                    out[tau_key][pair][arm].pop("raw", None)
    path = Path("analysis/gen06_matched_temperature.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
