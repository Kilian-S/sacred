#!/usr/bin/env python3
"""gen41 screen 2a (ORACLE-ONLY): the full-menu rotation (Tier 0) on the 24 selected
instances at (w=6, K=2); bar full-rot/opt_core >= 1.35; failures swapped for the
next-ranked passer clearing every bar. Pre-registered in gen41_deepwindow_zst.md.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     scratch/gen41_fullrot_screen.py
Updates models/runs/gen41_pool_screen.json (fullrot fields + any swaps).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from scratch.critique_followup_probes import disjoint_subset, rotation_value
from scratch.dyn_exact import karp_mmc
from scratch.gen40_dyn_sensitivity import TAU, BAND, N, enum_windows, window_losses
from scripts.train_b1lite1 import stacked_L
from scripts.train_generalist import CITY_PATHS
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(1)
W, K, KX = 6, 2, 12
BAR = 1.35


def full_rotation_row(city, od):
    np_, ep = CITY_PATHS[city]
    env = make_multiconvoy_env(tuple(od), N=N, K=K, k_extra_routes=KX, menu_select=True,
                               edge_vuln_band=BAND, nodes_path=np_, edges_path=ep)
    game = env.game
    L = stacked_L(game, N)
    R = game.n_routes
    dis = disjoint_subset([set(e) for e in game.route_edges])
    _, counts_c = enum_windows(dis, W, R)
    lw_c = window_losses(counts_c, L, TAU)
    opt_core = karp_mmc(lw_c[:, dis], 3 ** W, 3, 3 ** (W - 1))
    rng = np.random.default_rng(0)
    orders = [list(range(R))] + [list(rng.permutation(R)) for _ in range(20)]
    fr = min(rotation_value(o, L, TAU, W) for o in orders)
    return fr, opt_core, fr / max(opt_core, 1e-12)


def main():
    t0 = time.time()
    d = json.load(open("models/runs/gen41_pool_screen.json"))
    swaps = []
    for city, blob in d["cities"].items():
        rows_by_od = {tuple(r["od"]): r for r in blob["candidates"]}
        selected = [tuple(od) for od in blob["selected"]]
        final = []
        for od in selected:
            fr, opt, ratio = full_rotation_row(city, od)
            rows_by_od[od]["full_rotation"] = fr
            rows_by_od[od]["fullrot_over_opt"] = ratio
            status = "PASS" if ratio >= BAR else "FAIL"
            print(f"{city} {od[0]}-{od[1]}: full-rot {fr:.4f} opt {opt:.4f} "
                  f"ratio {ratio:.2f} {status}", flush=True)
            if ratio >= BAR:
                final.append(od)
                continue
            # replacement hunt: next-ranked passers clearing original bars AND this one
            cand = sorted([r for r in blob["candidates"] if r["passes"]
                           and tuple(r["od"]) not in selected
                           and tuple(r["od"]) not in [tuple(x) for x in final]],
                          key=lambda r: -r["rule_over_opt"])
            replaced = False
            for r in cand:
                od2 = tuple(r["od"])
                fr2, opt2, ratio2 = full_rotation_row(city, od2)
                r["full_rotation"] = fr2
                r["fullrot_over_opt"] = ratio2
                print(f"    candidate {od2[0]}-{od2[1]}: full-rot ratio {ratio2:.2f}"
                      f"{' -> SWAP IN' if ratio2 >= BAR else ''}", flush=True)
                if ratio2 >= BAR:
                    final.append(od2)
                    swaps.append((city, od, od2))
                    replaced = True
                    break
            if not replaced:
                final.append(od)
                swaps.append((city, od, None))
                print(f"    NO passing replacement found; {od[0]}-{od[1]} kept, "
                      f"flagged", flush=True)
        blob["selected"] = [list(od) for od in final]
    d["fullrot_screen"] = dict(bar=BAR, swaps=[[c, list(a), list(b) if b else None]
                                               for c, a, b in swaps],
                               secs=round(time.time() - t0, 1))
    with open("models/runs/gen41_pool_screen.json", "w") as f:
        json.dump(d, f, indent=1)
    print(f"done: {len(swaps)} swaps/flags ({round(time.time() - t0, 1)}s)", flush=True)


if __name__ == "__main__":
    main()
