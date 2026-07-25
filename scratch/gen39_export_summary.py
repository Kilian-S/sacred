#!/usr/bin/env python3
"""gen39: export a COMMITTABLE summary of a screen run.

models/ is gitignored (repo convention), so the 63 MB per-cell JSONs live on disk only. This drops
the per-arm ladders (the bulk) and keeps one row per cell: the axes, both memories' headline
numbers and the winning arm names. Small enough to version, complete enough to redraw every table
in the ledger.

    PYTHONPATH=. python scratch/gen39_export_summary.py \
        --in models/runs/gen39_screen2.json --out results/gen39_screen2_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os

COLS = ["tag", "map", "range_mult", "seed", "K", "kind", "hidden_leth", "conceal_reach",
        "n_conceal", "phi", "R", "H", "mean_known", "eq_static"]
PER = ["opt", "cap", "blind", "revealed", "G1", "G2", "G_conceal", "degenerate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="models/runs/gen39_screen2.json")
    ap.add_argument("--out", dest="dst", default="results/gen39_screen2_summary.csv")
    a = ap.parse_args()
    rows = json.load(open(a.src))
    os.makedirs(os.path.dirname(a.dst), exist_ok=True)
    head = COLS + [f"{m}_{k}" for m in ("forgetful", "persistent") for k in PER] \
        + ["persistent_blind_arm", "persistent_revealed_arm"] \
        + [f"opt_T{t}" for t in (10, 20, 40, 80)]
    with open(a.dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for r in rows:
            tag = r["tag"]
            base = [tag, tag.split("x")[0], tag.split("x")[1].split("cr")[0], r["seed"], r["K"],
                    r["kind"], r["hidden_leth"], round(r["conceal_reach"], 4), r["n_conceal"],
                    round(r["phi"], 4), r["R"], r["H"], round(r["mean_known"], 4),
                    round(r["eq_static"], 6)]
            for m in ("forgetful", "persistent"):
                base += [round(r[m][k], 6) if isinstance(r[m][k], float) else r[m][k] for k in PER]
            base += [r["persistent"]["blind_arm"], r["persistent"]["revealed_arm"]]
            base += [round(r["opt_curve"][str(t)], 6) for t in (10, 20, 40, 80)]
            w.writerow(base)
    print(f"[written] {a.dst} ({len(rows)} rows, {os.path.getsize(a.dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
