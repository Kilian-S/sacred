#!/usr/bin/env python3
"""gen27 worst-case row (EVAL-ONLY; pre-registered reported row in the gen27 ledger).

For each held-out Gdansk instance: run the select-on-train checkpoint policy against the
pattern-of-life adversary for n sorties, collect its MARGINAL route distribution, and score
that marginal's single-shot stacked exploitability under the instance's ORACLE best response,
vs the instance's single-shot stacked equilibrium V_eq (the gen19 worst-case row, zero-shot).
Answers: what does exploiting the adaptive adversary cost against a worst-case non-adaptive
one? (gen19 single-instance premium was +6%.)

Run: PYTHONPATH=. .venv/bin/python scratch/gen27_worstcase.py <seed.json> <ckpt_dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import deque

import numpy as np
import torch

from scripts.train_b1lite1 import softmax_br
from scripts.train_dyn_generalist import build_obs, pick_route, prep_instance
from scripts.train_generalist import sample_instances
from src.agents.sac import ProtagonistSAC
from src.baselines.multiconvoy_oracle import _row_minimiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_json"); ap.add_argument("ckpt_dir")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--json-out", default="models/runs/gen27_dyn_generalist/worstcase.json")
    a = ap.parse_args()
    torch.set_num_threads(4)
    d = json.load(open(a.seed_json))
    sel_sortie = d["select_on_train"]["sortie"]
    cks = {int(re.search(r"ep(\d+)", c).group(1)): c
           for c in glob.glob(f"{a.ckpt_dir}/actor_ep*.pt")}
    ck = cks[min(cks, key=lambda e: abs(e - sel_sortie))]
    w, tau = d["w"], d["tau"]

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu")
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    prot.actor.load_state_dict(torch.load(ck, map_location="cpu"))

    rows = []
    for it in sample_instances(6, 3, 1, (0.15, 0.95), 8, 0, city="gdansk"):
        prep_instance(it, tau, w)
        window = deque(maxlen=w)
        marg = np.zeros(it.nR)
        for _ in range(a.n):
            counts = np.bincount(list(window), minlength=it.nR).astype(float)
            obs = build_obs(it, counts, w)
            r = pick_route(prot, obs, it.nR)
            marg[r] += 1.0
            window.append(r)
        marg /= marg.sum()
        # single-shot stacked game: worst-case of the marginal vs the stacked equilibrium value
        v_eq, _ = _row_minimiser(it.L)
        worst = float((marg @ it.L).max())
        rows.append({"od": f"{it.od[0]}-{it.od[1]}", "v_eq": round(v_eq, 4),
                     "worst_case_of_marginal": round(worst, 4),
                     "premium": round(worst / v_eq, 3)})
        print(f"  {it.od}: worst-case of marginal {worst:.3f} vs V_eq {v_eq:.3f} "
              f"(premium {worst / v_eq:.2f}x)")
    json.dump({"checkpoint": ck, "rows": rows}, open(a.json_out, "w"), indent=2)
    print(f"[written] {a.json_out}")


if __name__ == "__main__":
    main()
