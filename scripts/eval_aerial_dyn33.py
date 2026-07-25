#!/usr/bin/env python3
"""gen33 METRIC 2 EVAL: the 6 held-out cells at the validation-selected checkpoint.

Cells (pinned pre-training): {ukraine, narva} x {oracle-K1, oracle-K3 (the banked screen
winners), gen32-doctrine flat}; each = mean exact chain value over pristine fields 4100-4102.
One invocation evaluates ONE arm-seed (or --untrained) so the ten evaluations parallelise;
results merge in the fold step. Checkpoint choice = argmin validation value from the run's
history (select-on-val, pinned)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_aerial_dyn33 import ForceInst, policy_value
from src.agents.sac import ProtagonistSAC
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre
from src.redforce_score import GEN32_DOCTRINE, ScoreBase

SIGMA0 = 8.0
FIELDS = (4100, 4101, 4102)
THEATRES = {"ukraine": "data/maps/theatre_ukraine_vec.json",
            "narva": "data/maps/theatre_narva_vec.json"}


def build_cells():
    screen = json.load(open("models/runs/gen33_score_screen.json"))
    lat_ref = lateral_width(load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json"))
    cells = {}
    for name, path in THEATRES.items():
        base = ScoreBase(path, lat_ref=lat_ref)
        anc = screen["anchors"][name]
        enemies = {
            "oracleK1": (anc["single"]["oracle_sites"],
                         [tuple(d) for d in anc["single"]["oracle_doctrine"]],
                         SIGMA0 * base.scale),
            "oracleK3": (anc["coordinated"]["oracle_sites"],
                         [tuple(d) for d in anc["coordinated"]["oracle_doctrine"]],
                         SIGMA0 * base.scale),
            "gen32doc": ([0], [GEN32_DOCTRINE], None),
        }
        for ename, (sites, doc, sig) in enemies.items():
            cells[f"{name}|{ename}"] = [ForceInst(base, f"{name}{f}", f, sites, doc, sig)
                                        for f in FIELDS]
    return cells


def fresh_prot():
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3, alpha_floor=0.20)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    return prot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="", help="llm|random|single; empty with --untrained")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--untrained", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    cells = build_cells()
    prot = fresh_prot()
    tag = "untrained"
    meta = {}
    if not args.untrained:
        run_json = Path(f"models/runs/gen33_curriculum/{args.arm}_s{args.seed}.json")
        if run_json.exists():
            best = min(json.load(open(run_json))["history"], key=lambda h: h[1])
            sortie, val = int(best[0]), float(best[1])
        else:                              # run ended early: latest checkpoint, val from the log
            cks = sorted(Path(f"models/runs/gen33_curriculum/{args.arm}_s{args.seed}_ckpts")
                         .glob("actor_ep*.pt"), key=lambda p: int(p.stem.split("ep")[1]))
            sortie = int(cks[-1].stem.split("ep")[1])
            val = float("nan")
            for line in open(f"models/runs/gen33_curriculum/{args.arm}_s{args.seed}.log"):
                if f"sortie   {sortie}" in line or f"sortie {sortie:6d}" in line:
                    val = float(line.split("VAL")[1].split()[0])
        ck = f"models/runs/gen33_curriculum/{args.arm}_s{args.seed}_ckpts/actor_ep{sortie}.pt"
        prot.actor.load_state_dict(torch.load(ck))
        tag = f"{args.arm}_s{args.seed}"
        meta = {"ckpt_sortie": sortie, "val_at_ckpt": val}
    rows = {}
    for cname, insts in cells.items():
        rows[cname] = float(np.mean([policy_value(prot, it) for it in insts]))
        print(f"  [{tag}] {cname}: {rows[cname]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    out = {"tag": tag, **meta, "cells": rows, "pooled": float(np.mean(list(rows.values())))}
    Path("models/runs/gen33_curriculum").mkdir(parents=True, exist_ok=True)
    Path(f"models/runs/gen33_curriculum/heldout_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[{tag}] pooled {out['pooled']:.4f} -> heldout_{tag}.json [{time.time()-t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
