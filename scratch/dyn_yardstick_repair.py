#!/usr/bin/env python3
"""YARDSTICK REPAIR (oracle-only, no training; 2026-07-23): corrected history_opt for the
banked dynamic acts.

scratch/gen35_mmc_check.py established that oracle_refs' undamped RVI `history_opt` is wrong
on every cell tested (two independent exact methods, Karp min-mean-cycle and damped RVI, agree
with each other and disagree with it). This probe recomputes the CORRECTED references for:

  1. gen19's instance (35-159, N=3 K=1, w=3 tau=0.15) - the banked "SACRED 0.050 ~ history_opt
     0.049" STRONG sentence is restated against the exact optimum.
  2. gen27's full pools (18 train + 6 held-out Gdansk instances, pool-seed 0, byte-identical
     construction) - the reported "x history_opt" rows are restated; the PRIMARY (ratio to
     iid_eq) is UNAFFECTED (iid_eq is exact enumeration, no RVI involved).

Identity checks before anything is restated: per instance, the stored seed0.json refs must
match this probe's recomputation (history_opt to 1e-9 against the same buggy code -> instance
identity; iid_eq within LP-degeneracy wobble).

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/dyn_yardstick_repair.py
Writes models/runs/dyn_yardstick_repair.json
"""
from __future__ import annotations

import json

import numpy as np
import torch

from scratch.critique_followup_probes import (
    antirepeat_value, disjoint_subset, rotation_value)
from scratch.dyn_exact import history_opt_exact
from scripts.train_b1lite1 import oracle_refs, stacked_L
from scripts.train_generalist import sample_instances
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(1)
N, K, BAND, KX, W, TAU = 3, 1, (0.15, 0.95), 8, 3, 0.15
GEN19_BANKED = dict(sacred=0.050, no_window=0.148, ledger_history_opt=0.049)


def instance_row(env, tag):
    L = stacked_L(env.game, N)
    buggy = oracle_refs(L, TAU, W)
    exact = history_opt_exact(L, TAU, W)
    dis = disjoint_subset([set(e) for e in env.game.route_edges])
    rot = min(rotation_value(list(o), L, TAU, W)
              for o in {tuple(dis), tuple(reversed(dis))})
    anti = antirepeat_value(dis, L, TAU, W)
    return dict(tag=tag, R=L.shape[0], m=len(dis),
                iid_eq=buggy["iid_eq"], static_det=buggy["static_det"],
                history_opt_buggy=buggy["history_opt"], history_opt_exact=exact,
                rotation=rot, anti_repeat=anti,
                buggy_over_exact=buggy["history_opt"] / exact), L


def main():
    out = {}

    env19 = make_multiconvoy_env(od=("35", "159"), N=N, K=K, k_extra_routes=KX,
                                 menu_select=True, edge_vuln_band=BAND, interception_loss=10.0)
    row19, _ = instance_row(env19, "gen19 35-159")
    row19.update(banked=GEN19_BANKED,
                 sacred_over_exact=GEN19_BANKED["sacred"] / row19["history_opt_exact"],
                 rotation_over_exact=row19["rotation"] / row19["history_opt_exact"])
    out["gen19"] = row19
    print(f"gen19 35-159: exact {row19['history_opt_exact']:.4f} vs buggy "
          f"{row19['history_opt_buggy']:.4f} (ledger 0.049) | SACRED 0.050 -> "
          f"{row19['sacred_over_exact']:.3f}x exact | rotation {row19['rotation']:.4f} = "
          f"{row19['rotation_over_exact']:.3f}x exact", flush=True)

    cities = ["kaliningrad", "east_london", "istanbul"]
    train = []
    for c in cities:
        train += sample_instances(6, N, K, BAND, KX, 0, city=c)
    test = sample_instances(6, N, K, BAND, KX, 0, city="gdansk")

    stored = json.load(open("models/runs/gen27_dyn_generalist/seed0.json"))
    stored_test = {tuple(r["od"]): r for r in stored["test_refs"]}
    stored_train = {tuple(r["od"]): r for r in stored["train_refs"]}

    rows = {"train": [], "test": []}
    for tag, pool, ref in (("train", train, stored_train), ("test", test, stored_test)):
        for it in pool:
            r, L = instance_row(it.env, f"{it.city} {it.od[0]}-{it.od[1]}")
            sr = ref.get((str(it.od[0]), str(it.od[1]))) or ref.get(tuple(it.od))
            if sr is None:
                r["identity"] = "NO STORED MATCH"
            else:
                r["identity"] = dict(
                    hist_buggy_match=abs(sr["history_opt"] - r["history_opt_buggy"]) < 1e-9,
                    iid_eq_stored=sr["iid_eq"],
                    iid_eq_rel_dev=abs(sr["iid_eq"] - r["iid_eq"]) / sr["iid_eq"])
            r["city"] = it.city
            r["od"] = list(it.od)
            rows[tag].append(r)
            idm = r.get("identity")
            ok = idm["hist_buggy_match"] if isinstance(idm, dict) else idm
            print(f"  {tag} {it.city} {it.od}: exact {r['history_opt_exact']:.4f} "
                  f"buggy {r['history_opt_buggy']:.4f} ({r['buggy_over_exact']:.3f}x) "
                  f"m={r['m']} rot {r['rotation']:.4f} anti {r['anti_repeat']:.4f} "
                  f"identity={ok}", flush=True)
    out["gen27_refs"] = rows

    restate = []
    for s in (0, 1, 2):
        d = json.load(open(f"models/runs/gen27_dyn_generalist/seed{s}.json"))
        sel = d["select_on_train"]
        ods = [tuple(r["od"]) for r in d["test_refs"]]
        ex = {tuple(r["od"]): r["history_opt_exact"] for r in rows["test"]}
        bg = {tuple(r["od"]): r["history_opt_buggy"] for r in rows["test"]}
        per = []
        for od, loss in zip(ods, sel["test_losses"]):
            per.append(dict(od=list(od), loss=loss,
                            ratio_exact=loss / ex[od], ratio_buggy=loss / bg[od]))
        restate.append(dict(seed=s, sortie=sel["sortie"],
                            mean_ratio_exact=float(np.mean([p["ratio_exact"] for p in per])),
                            mean_ratio_buggy=float(np.mean([p["ratio_buggy"] for p in per])),
                            per_od=per))
        print(f"gen27 seed{s} select-on-train: ratio-to-EXACT-opt "
              f"{restate[-1]['mean_ratio_exact']:.3f} (was {restate[-1]['mean_ratio_buggy']:.3f} "
              f"vs buggy)", flush=True)
    pooled_e = float(np.mean([r["mean_ratio_exact"] for r in restate]))
    pooled_b = float(np.mean([r["mean_ratio_buggy"] for r in restate]))
    comp = []
    for r in rows["test"]:
        comp.append(dict(od=r["od"],
                         rot_over_exact=r["rotation"] / r["history_opt_exact"],
                         anti_over_exact=r["anti_repeat"] / r["history_opt_exact"]))
    out["gen27_restated"] = dict(per_seed=restate, pooled_ratio_exact=pooled_e,
                                 pooled_ratio_buggy=pooled_b, composed_rules=comp)
    print(f"gen27 POOLED ratio-to-optimum: EXACT {pooled_e:.3f} (buggy yardstick said "
          f"{pooled_b:.3f}) | composed rot/exact "
          f"{[round(c['rot_over_exact'], 2) for c in comp]}", flush=True)

    with open("models/runs/dyn_yardstick_repair.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote models/runs/dyn_yardstick_repair.json")


if __name__ == "__main__":
    main()
