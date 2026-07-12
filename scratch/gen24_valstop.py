#!/usr/bin/env python3
"""gen24 completion (EVAL-ONLY): the fair "early stopping on a proper validation set" row.

The gen24 select-on-train rule picks the overfit endpoint (distillation fits train to ~1.1x while
held-out degrades), which an examiner will call an unfair handicap: the standard supervised recipe
is early stopping on VALIDATION data. This script samples fresh instances from the TRAIN cities
(pool-seed 1; overlap with the pool-seed-0 instances dropped), evaluates every saved checkpoint on
them, selects the checkpoint with the best validation mean ratio, and reports the held-out Gdansk
ratio there (TAP over the centred checkpoint window, the standing deployable estimator). Applied
SYMMETRICALLY to the gen24 distillation seeds AND the gen16 adversarial seeds.
"""
from __future__ import annotations

import glob
import json
import re

import numpy as np
import torch

from scripts.train_generalist import exact_ratio, sample_instances
from src.agents.sac import ProtagonistSAC
from src.baselines.multiconvoy_oracle import best_response_attacker_multi

N, K, KX, BAND = 3, 1, 8, (0.15, 0.95)
TRAIN_CITIES = ["kaliningrad", "east_london", "istanbul"]


def _mkprot(state=None):
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu", role_alpha=True)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2))
    prot.actor.route_feats = None
    if state is not None:
        prot.actor.load_state_dict(state)
    return prot


def ckpts(d):
    return dict(sorted((int(re.search(r"ep(\d+)", c).group(1)), c)
                       for c in glob.glob(f"{d}/actor_ep*.pt")))


def tap_at(cks: dict, ep: int, it) -> float:
    eps = sorted(cks)
    i = eps.index(ep)
    window = eps[max(0, i - 1):i + 2]
    ds = [exact_ratio(_mkprot(torch.load(cks[e], map_location="cpu")), it)[1] for e in window]
    tap = np.mean(ds, axis=0)
    _, expl = best_response_attacker_multi(it.env.obj_matrix, tap)
    return float(expl) / it.eq


def main():
    torch.set_num_threads(4)
    # validation pool: fresh train-city instances, pool-seed 1, overlap dropped
    train0 = []
    for c in TRAIN_CITIES:
        train0 += [(c, i.od) for i in sample_instances(6, N, K, BAND, KX, 0, city=c)]
    val = []
    for c in TRAIN_CITIES:
        for it in sample_instances(3, N, K, BAND, KX, 1, city=c):
            if (c, it.od) not in train0:
                val.append(it)
    test = sample_instances(6, N, K, BAND, KX, 0, city="gdansk")
    print(f"[valstop] {len(val)} validation instances (train cities, pool-seed 1): "
          f"{[(i.city, i.od) for i in val]}", flush=True)

    out = {}
    for label, pattern in (("distill", "models/runs/gen24_distill/seed{s}_ckpts"),
                           ("adversarial-gen16", "models/runs/gen16_multicity/seed{s}_ckpts")):
        rows = []
        for s in (0, 1, 2):
            cks = ckpts(pattern.format(s=s))
            # single-checkpoint val curve (selection signal), then held-out TAP at the argmin
            val_curve = {}
            for ep, path in cks.items():
                prot = _mkprot(torch.load(path, map_location="cpu"))
                val_curve[ep] = float(np.mean([exact_ratio(prot, it)[0] for it in val]))
            sel = min(val_curve, key=val_curve.get)
            ho = float(np.mean([tap_at(cks, sel, it) for it in test]))
            rows.append((s, sel, val_curve[sel], ho))
            print(f"{label} seed {s}: val-selected step {sel} (val {val_curve[sel]:.3f}) -> "
                  f"held-out Gdansk TAP ratio {ho:.3f}", flush=True)
        means = float(np.mean([r[3] for r in rows]))
        out[label] = {"rows": rows, "holdout_mean": means}
        print(f"{label}: held-out mean (val-early-stopped) {means:.3f}", flush=True)

    json.dump(out, open("models/runs/gen24_distill/valstop.json", "w"), indent=2)
    print("[written] models/runs/gen24_distill/valstop.json")


if __name__ == "__main__":
    main()
