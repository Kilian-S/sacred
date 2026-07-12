#!/usr/bin/env python3
"""A8 (ORACLE-ONLY): the prevalence figure.

Over sampled high-connectivity OD pairs in all four cities (the standing screen: deg >= 3, 3-6
base routes, k8 menus, R in [10,14], eq >= 0.05), compute per OD at N=3, K=1, mission:
  - loss_det / equilibrium  (the headroom calibrated play can exploit)
  - uniform-stack / equilibrium (the headroom over naive randomisation)
Mark the two headline instances (Kaliningrad 35-159 and 62-97). Answers "how often does
calibrated mixing matter, and were the headline instances cherry-picked?" descriptively.
Pre-registration: experiments/a6_a7_a8_completions.md §A8.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import best_response_attacker_multi
from src.envs.multiconvoy_interdiction import make_multiconvoy_env
from src.baselines.multiconvoy_oracle import solve_multiconvoy

N, K, KX, BAND = 3, 1, 8, (0.15, 0.95)
CITIES = ["kaliningrad", "east_london", "istanbul", "gdansk"]
PER_CITY = 40


def uniform_stack_ratio(env, eq):
    R = env.game.n_routes
    d = np.zeros(len(env.occupancies))
    for r in range(R):
        d[env._occ_index[tuple(N if i == r else 0 for i in range(R))]] = 1.0 / R
    _, expl = best_response_attacker_multi(env.obj_matrix, d)
    return float(expl) / eq


def main():
    torch.set_num_threads(4)
    rows = []
    for city in CITIES:
        insts = sample_instances(PER_CITY, N, K, BAND, KX, seed=3, city=city)
        for it in insts:
            rows.append({"city": city, "od": f"{it.od[0]}-{it.od[1]}",
                         "det_eq": it.loss_det / it.eq,
                         "unif_eq": uniform_stack_ratio(it.env, it.eq)})
        print(f"{city}: {len(insts)} ODs", flush=True)

    # headline instances (marked)
    marks = {}
    for od in (("35", "159"), ("62", "97")):
        env = make_multiconvoy_env(od=od, N=N, K=K, k_extra_routes=KX, menu_select=True,
                                   edge_vuln_band=BAND, interception_loss=10.0, seed=0)
        sol = solve_multiconvoy(env.game, N, "mission")
        marks[f"{od[0]}-{od[1]}"] = {"det_eq": sol.loss_det / sol.loss_mixed,
                                     "unif_eq": uniform_stack_ratio(env, sol.loss_mixed)}
        print(f"headline {od}: det/eq {marks[f'{od[0]}-{od[1]}']['det_eq']:.2f}, "
              f"unif-stack/eq {marks[f'{od[0]}-{od[1]}']['unif_eq']:.2f}", flush=True)

    de = np.array([r["det_eq"] for r in rows])
    ue = np.array([r["unif_eq"] for r in rows])
    q = lambda x: np.percentile(x, [10, 25, 50, 75, 90])
    print(f"loss_det/eq quantiles (10/25/50/75/90): {np.round(q(de), 2)}")
    print(f"uniform-stack/eq quantiles:            {np.round(q(ue), 2)}")
    print(f"fraction with det/eq >= 2 (material calibration headroom): {(de >= 2).mean():.2f}")
    print(f"fraction with unif/eq >= 1.5 (naive randomisation clearly suboptimal): {(ue >= 1.5).mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, mk, title in ((axes[0], de, "det_eq", "loss_det / equilibrium"),
                                (axes[1], ue, "unif_eq", "uniform-stack / equilibrium")):
        ax.hist(vals, bins=24, color="#4477aa", alpha=0.85)
        for name, m in marks.items():
            ax.axvline(m[mk], color="crimson", lw=1.4)
            ax.text(m[mk], ax.get_ylim()[1] * 0.9, name, rotation=90, fontsize=7, color="crimson")
        ax.set_xlabel(title)
        ax.set_ylabel("OD count (4 cities)")
    fig.suptitle("Prevalence of calibration headroom over high-connectivity ODs (A8)")
    fig.tight_layout()
    fig.savefig("assets/prevalence.png", dpi=150)
    json.dump({"rows": rows, "headlines": marks}, open("models/runs/a8_prevalence.json", "w"),
              indent=2)
    print("[written] assets/prevalence.png + models/runs/a8_prevalence.json")


if __name__ == "__main__":
    main()
