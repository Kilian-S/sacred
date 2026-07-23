#!/usr/bin/env python3
"""gen34 family refs (oracle-only): per-instance exact yardsticks for the hidden-adversary act.

For every gen34 pool instance (gen27 recipe: 3 cities x 6 train + 6 held-out Gdansk,
pool-seed 0), computes the pre-registered exact references via scratch/dyn_exact.py:
per-member omni optima, the type-blind cap (Karp on the mixture-averaged cost), the inference
gap, and the naive-rule rows. Written once to models/runs/gen34_hidden_adversary/
family_refs.json; the trainer LOADS this file (auditable, no solver in the trainer).

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. nice .venv/bin/python scratch/gen34_refs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scratch.critique_followup_probes import antirepeat_value, disjoint_subset, rotation_value
from scratch.dyn_exact import build_window_mdp, greedy_policy_from_rvi, karp_mmc, policy_value_exact
from scratch.gen34_family_probe import member_fns
from scripts.train_b1lite1 import stacked_L
from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import _row_minimiser

torch.set_num_threads(1)
N, K, BAND, KX, W, TAU = 3, 1, (0.15, 0.95), 8, 3, 0.15


def refs_for(env, city, od):
    L = stacked_L(env.game, N)
    R = L.shape[0]
    _, eq = _row_minimiser(L)
    fns = member_fns(L, eq)
    names = list(fns)
    n, pw = R ** W, R ** (W - 1)
    costs = {nm: build_window_mdp(L, TAU, W, member_fn=fns[nm])[0] for nm in names}
    omni = {nm: karp_mmc(costs[nm], n, R, pw) for nm in names}
    mix = np.mean([costs[nm] for nm in names], axis=0)
    blind = karp_mmc(mix, n, R, pw)
    dis = disjoint_subset([set(e) for e in env.game.route_edges])
    anti = float(np.mean([policy_value_exact(
        _anti_pol(dis, n, R, W), costs[nm], n, R, pw) for nm in names]))
    rot = float(np.mean([policy_value_exact(
        _det_cycle_pol(dis, n, R, W), costs[nm], n, R, pw) for nm in names]))
    iid = float(np.mean([policy_value_exact(
        np.broadcast_to(eq, (n, R)).copy(), costs[nm], n, R, pw) for nm in names]))
    return dict(city=city, od=list(od), R=R, m=len(dis),
                omni={nm: float(v) for nm, v in omni.items()},
                omni_cap=float(np.mean(list(omni.values()))), blind_cap=float(blind),
                inference_gap=float(blind - np.mean(list(omni.values()))),
                anti_repeat_mixture=anti, rotation_mixture=rot, iid_eq_mixture=iid)


def _decode(n, R, w):
    dec = np.empty((n, w), dtype=np.int64)
    x = np.arange(n)
    for i in range(w):
        dec[:, w - 1 - i] = x % R
        x = x // R
    return dec


def _anti_pol(dis, n, R, w):
    dec = _decode(n, R, w)
    pol = np.zeros((n, R))
    for s in range(n):
        allowed = [r for r in dis if r not in dec[s]] or list(dis)
        pol[s, allowed] = 1.0 / len(allowed)
    return pol


def _det_cycle_pol(dis, n, R, w):
    dec = _decode(n, R, w)
    nxt_in_cycle = {dis[i]: dis[(i + 1) % len(dis)] for i in range(len(dis))}
    pol = np.zeros((n, R))
    for s in range(n):
        pol[s, nxt_in_cycle.get(dec[s, -1], dis[0])] = 1.0
    return pol


def main():
    out = {}
    pools = []
    for c in ("kaliningrad", "east_london", "istanbul"):
        pools += [(it, "train") for it in sample_instances(6, N, K, BAND, KX, 0, city=c)]
    pools += [(it, "test") for it in sample_instances(6, N, K, BAND, KX, 0, city="gdansk")]
    for it, split in pools:
        key = f"{it.city}:{it.od[0]}-{it.od[1]}"
        r = refs_for(it.env, it.city, it.od)
        r["split"] = split
        out[key] = r
        print(f"{split} {key}: blind {r['blind_cap']:.4f} omni {r['omni_cap']:.4f} "
              f"gap {r['inference_gap']:.4f} ({r['blind_cap']/r['omni_cap']:.2f}x) "
              f"anti {r['anti_repeat_mixture']:.4f}", flush=True)
    Path("models/runs/gen34_hidden_adversary").mkdir(parents=True, exist_ok=True)
    with open("models/runs/gen34_hidden_adversary/family_refs.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote models/runs/gen34_hidden_adversary/family_refs.json")


if __name__ == "__main__":
    main()
