#!/usr/bin/env python3
"""B3 (ORACLE-ONLY): the risk-aversion spectrum - the price of predictability as a function of
loss-aversion. Pre-registration: experiments/b3_b4_oracle.md §B3."""
from __future__ import annotations

import json

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

BAND, KX = (0.15, 0.95), 8


def sweep(game, N):
    """[(label, aversion_rank, det, eq, det/eq)] from risk-neutral to maximally loss-averse."""
    rows = []
    sol = solve_multiconvoy(game, N, "linear")
    rows.append(("linear", 0, sol.loss_det, sol.loss_mixed, sol.loss_det / sol.loss_mixed))
    for rank, m in enumerate(range(N, 0, -1), start=1):   # m=N least averse ... m=1 mission
        sol = solve_multiconvoy(game, N, "threshold", m=m)
        rows.append((f"P(>={m} of {N})", rank, sol.loss_det, sol.loss_mixed,
                     sol.loss_det / max(sol.loss_mixed, 1e-9)))
    return rows


def main():
    torch.set_num_threads(4)
    out = {"headline": {}, "population": {}}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for od, N, style in ((("35", "159"), 3, "o-"), (("62", "97"), 3, "s-"),
                         (("35", "159"), 5, "o--"), (("62", "97"), 5, "s--")):
        env = make_multiconvoy_env(od=od, N=N, K=1, k_extra_routes=KX, menu_select=True,
                                   edge_vuln_band=BAND, interception_loss=10.0, seed=0)
        rows = sweep(env.game, N)
        key = f"{od[0]}-{od[1]} N={N}"
        out["headline"][key] = [(r[0], r[2], r[3], r[4]) for r in rows]
        ax.plot([r[1] for r in rows], [r[4] for r in rows], style, label=key)
        print(f"{key}: " + " | ".join(f"{r[0]}: det/eq {r[4]:.2f}" for r in rows), flush=True)

    # population median curve (Kaliningrad screen ODs, N=3)
    insts = sample_instances(40, 3, 1, BAND, KX, seed=3, city="kaliningrad")
    curves = {lab: [] for lab in ("linear", "P(>=3 of 3)", "P(>=2 of 3)", "P(>=1 of 3)")}
    for it in insts:
        for r in sweep(it.env.game, 3):
            curves[r[0]].append(r[4])
    med = {lab: float(np.median(v)) for lab, v in curves.items()}
    out["population"] = {lab: {"median": med[lab],
                               "q25": float(np.percentile(v, 25)),
                               "q75": float(np.percentile(v, 75))}
                         for lab, v in curves.items()}
    ax.plot(range(4), [med[lab] for lab in curves], "k^-", lw=2,
            label="population median (40 ODs, N=3)")
    print("population medians:", {k: round(v, 2) for k, v in med.items()}, flush=True)

    ax.set_xticks(range(6))
    ax.set_xticklabels(["risk-\nneutral", "m=N", "m=N-1", "m=N-2", "m=N-3", "m=1"][:6], fontsize=8)
    ax.set_xlabel("loss-aversion (left = risk-neutral, right = mission: any loss fails)")
    ax.set_ylabel("loss_det / equilibrium (the price of predictability)")
    ax.axhline(1, color="k", lw=0.8)
    ax.legend(fontsize=7)
    ax.set_title("B3: the deterministic-vs-mixed gap as a function of loss-aversion")
    fig.tight_layout()
    fig.savefig("assets/b3_risk_spectrum.png", dpi=150)
    json.dump(out, open("models/runs/b3_risk_spectrum.json", "w"), indent=2)
    print("[written] assets/b3_risk_spectrum.png + models/runs/b3_risk_spectrum.json")


if __name__ == "__main__":
    main()
