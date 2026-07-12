#!/usr/bin/env python3
"""A7 (EVAL-ONLY): per-OD gap-closure restatement of the transfer ladder + the decay figure.

Gap closure = (loss_det - policy) / (loss_det - equilibrium): the fraction of the
deterministic-to-equilibrium gap actually closed (1 = equilibrium play, 0 = no better than the
deterministic-class optimum, negative = worse). Ratios flatter thin-headroom cells; this is the
metric that measures what the thesis claims. Computed exactly from saved artefacts at the
standing deployable read of each act (select-on-train for gen16/gen22 dual-selection acts).
Pre-registration: experiments/a6_a7_a8_completions.md §A7.
"""
from __future__ import annotations

import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def per_od_closure_from_json(path: str, select: str = "train"):
    """gen15/gen16/gen22 JSONs: history rows (step, tr_m, te_m, te_v, ...); te_v aligns with
    test_ods order; test_refs keyed 'u-v' hold {eq, loss_det}."""
    d = json.load(open(path))
    hist = d["history"]
    row = min(hist, key=lambda h: h[1] if select == "train" else h[2])
    te_v = row[3]
    ods = [f"{od[1][0]}-{od[1][1]}" if isinstance(od[0], str) and len(od) == 2 and isinstance(od[1], (list, tuple))
           else f"{od[0]}-{od[1]}" for od in d["test_ods"]]
    # test_ods entries are (city, (u, v)) in multi-city runs and (u, v) in single-city runs
    fixed = []
    for od in d["test_ods"]:
        if len(od) == 2 and isinstance(od[1], (list, tuple)):
            fixed.append(f"{od[1][0]}-{od[1][1]}")
        else:
            fixed.append(f"{od[0]}-{od[1]}")
    ods = fixed
    out = []
    for od, r in zip(ods, te_v):
        ref = d["test_refs"][od]
        D = ref["loss_det"] / ref["eq"]
        out.append((od, r, (D - r) / (D - 1.0)))
    return out, row[0]


def main():
    acts = []

    # trained headline instances (constants from the gen14 ledger)
    acts.append(("trained MC 35-159 (gen14 n=10)", [(0.699 - 0.256) / (0.699 - 0.206)]))
    acts.append(("trained SC 33-71 (gen14 n=10)", [(1.000 - 0.310) / (1.000 - 0.167)]))

    # gen15: held-out ODs, same graph (3 seeds; select-on-train == select-on-test there)
    cl = []
    for s in (0, 1, 2):
        rows, step = per_od_closure_from_json(f"models/runs/gen15_generalist/seed{s}.json")
        cl += [c for _, _, c in rows]
    acts.append(("held-out ODs, same graph (gen15)", cl))

    # gen16: held-out city Gdansk (select-on-train)
    cl = []
    for s in (0, 1, 2):
        rows, step = per_od_closure_from_json(f"models/runs/gen16_multicity/seed{s}.json")
        cl += [c for _, _, c in rows]
    acts.append(("held-out CITY Gdansk (gen16)", cl))

    # gen22: held-out Istanbul (select-on-train)
    cl = []
    for s in (0, 1, 2):
        rows, step = per_od_closure_from_json(f"models/runs/gen22_rotation/seed{s}.json")
        cl += [c for _, _, c in rows]
    acts.append(("rotation: held-out ISTANBUL (gen22)", cl))

    # whole-Kyiv (single-checkpoint eval artefact)
    k = json.load(open("models/runs/a2_graph_transfer_kyiv.json"))
    cl = []
    for r in k["rows"]:
        D = r["loss_det"] / r["eq"]
        cl.append((D - r["gen_ratio"]) / (D - 1.0))
    acts.append(("whole-Kyiv scale row (single-ckpt)", cl))

    print(f"{'act':45s} {'mean':>7s} {'median':>7s} {'<=0 cells':>10s} {'n':>4s}")
    table = []
    for name, cs in acts:
        cs = np.asarray(cs, float)
        print(f"{name:45s} {cs.mean():7.3f} {np.median(cs):7.3f} "
              f"{int((cs <= 0).sum()):10d} {len(cs):4d}")
        table.append({"act": name, "mean": float(cs.mean()), "median": float(np.median(cs)),
                      "cells_leq_0": int((cs <= 0).sum()), "n": int(len(cs)),
                      "per_cell": [round(float(c), 3) for c in cs]})

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(acts))
    means = [np.mean(cs) for _, cs in acts]
    ax.bar(xs, means, color="#4477aa", alpha=0.85)
    for x, (name, cs) in zip(xs, acts):
        cs = np.asarray(cs, float)
        ax.scatter(np.full(len(cs), x) + np.random.default_rng(0).uniform(-0.12, 0.12, len(cs)),
                   cs, s=12, color="#222222", zorder=3, alpha=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="green", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace(" (", "\n(") for n, _ in acts], fontsize=7)
    ax.set_ylabel("gap closure  (1 = equilibrium, 0 = deterministic optimum)")
    ax.set_title("Calibration content vs transfer distance (gap-closure restatement, A7)")
    fig.tight_layout()
    fig.savefig("assets/transfer_gap_closure.png", dpi=150)
    json.dump(table, open("models/runs/a7_gap_closure.json", "w"), indent=2)
    print("[written] assets/transfer_gap_closure.png + models/runs/a7_gap_closure.json")


if __name__ == "__main__":
    main()
