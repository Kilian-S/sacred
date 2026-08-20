#!/usr/bin/env python3
"""Computes the gen45 worst-case-versus-committing premium, evaluation only.

For each confirmation seed's validation-selected checkpoint and each gated field, takes the
policy's stationary marginal route mixture m under its own window chain, lets the enemy abandon
the doctrine and commit to the single site maximising the stacked-fleet damage of m, and reports
that worst case against the field's static equilibrium value. The stacked equilibrium mixture's
own worst case is reported beside it, separating what is policy-specific from what stacking
itself costs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_gen45_unified import Inst, make_base, policy_dists
from src.agents.sac import ProtagonistSAC

OUT = Path("models/runs/gen45_unified")


def make_prot():
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3, alpha_floor=0.20)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    return prot


def marginal(inst, dists):
    Sn = len(inst.wins)
    pi = np.full(Sn, 1.0 / Sn)
    for _ in range(600):
        flow = pi[:, None] * dists
        nxt = np.zeros(Sn)
        np.add.at(nxt, inst.g.succ.ravel(), flow.ravel())
        nxt = 0.5 * nxt + 0.5 * pi
        if np.abs(nxt - pi).max() < 1e-13:
            pi = nxt
            break
        pi = nxt
    return (pi[:, None] * dists).sum(axis=0)


def main():
    torch.set_num_threads(2)
    base = make_base()
    insts = [Inst(base, f"gated{45200 + s}", 45200 + s) for s in range(6)]
    out, prems = {}, []
    for s in (10, 11, 12):
        d = json.loads((OUT / f"confirm_seed{s}.json").read_text())
        sel = min(d["history"], key=lambda h: h[1])[0]
        prot = make_prot()
        prot.actor.load_state_dict(
            torch.load(OUT / f"confirm_seed{s}_ckpts" / f"actor_ep{sel}.pt"))
        row = {}
        for inst in insts:
            m = marginal(inst, policy_dists(prot, inst))
            worst = float((m @ inst.g.dmg).max())
            eq_m = float((inst.g.d_eq @ inst.g.dmg).max())
            prem = worst / inst.g.eq_static
            row[inst.name] = {"worst": worst, "eq_static": inst.g.eq_static,
                              "eq_stack_worst": eq_m, "premium": prem,
                              "eq_stack_premium": eq_m / inst.g.eq_static}
            prems.append(prem)
        out[f"seed{s}"] = {"selected": sel, "fields": row}
        print(f"seed{s} (sel@{sel}): premiums "
              + " ".join(f"{row[i.name]['premium']:.2f}" for i in insts)
              + f"  mean {np.mean([row[i.name]['premium'] for i in insts]):.2f}", flush=True)
    eqp = [out["seed10"]["fields"][i.name]["eq_stack_premium"] for i in insts]
    print(f"stacked-equilibrium reference premiums: "
          + " ".join(f"{p:.2f}" for p in eqp) + f"  mean {np.mean(eqp):.2f}")
    print(f"POOLED policy premium over 18 cells: {np.mean(prems):.2f}")
    json.dump(out, open(OUT / "worstcase.json", "w"), indent=1)
    print("[written] models/runs/gen45_unified/worstcase.json")


if __name__ == "__main__":
    main()
