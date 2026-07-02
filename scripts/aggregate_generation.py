#!/usr/bin/env python3
"""Aggregate a generation's seeded runs into mean +/- std (the deliverable, not raw TB curves).

Reads every run under logs/tb_runs/<group>/, groups by config (the run name minus the
``_seed<k>`` suffix), and reports mean/std/min/max across seeds for the key metrics — so you
read one table instead of N wiggly curves. Optionally writes a mean+/-band plot per config.

    PYTHONPATH=. python scripts/aggregate_generation.py --group gen01_erb_ablation
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Metrics summarised as the windowed mean over the last `--window` eval/episode points.
SCALAR_TAGS = [
    "Eval/gap_atk", "Eval/gap_noatk", "Eval/learned_atk", "Eval/greedy_atk",
    "Episode/Total_Wait", "Value/Protagonist_Q_Spread", "Value/Protagonist_Entropy",
    "Value/Antagonist_Q",
]


def load_scalar(run_dir: str, tag: str):
    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    return np.array([p.value for p in ea.Scalars(tag)])


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate a generation into mean+/-std.")
    p.add_argument("--group", required=True)
    p.add_argument("--log-dir", default="logs/tb_runs")
    p.add_argument("--window", type=int, default=4, help="windowed mean over the last N points")
    p.add_argument("--plot", action="store_true", help="save a mean+/-band plot of gap_atk per config")
    args = p.parse_args()

    group_dir = os.path.join(args.log_dir, args.group)
    run_dirs = [d for d in glob.glob(os.path.join(group_dir, "*")) if os.path.isdir(d)]
    if not run_dirs:
        raise SystemExit(f"no runs under {group_dir}")

    # group by config = run name without the _seed<k> suffix
    by_config: dict[str, list[str]] = defaultdict(list)
    for d in run_dirs:
        config = re.sub(r"_seed\d+$", "", os.path.basename(d))
        by_config[config].append(d)

    print(f"Generation: {args.group}  (window = last {args.window} points)\n")
    for config in sorted(by_config):
        dirs = sorted(by_config[config])
        print(f"=== {config}  ({len(dirs)} seeds) ===")
        for tag in SCALAR_TAGS:
            per_seed = []
            for d in dirs:
                v = load_scalar(d, tag)
                if v is not None and len(v):
                    per_seed.append(float(np.mean(v[-args.window:])))
            if not per_seed:
                continue
            arr = np.array(per_seed)
            print(f"  {tag:30s} mean={arr.mean():8.1f}  std={arr.std():6.1f}  "
                  f"min={arr.min():8.1f}  max={arr.max():8.1f}  (n={len(arr)})")
        print()

    # headline comparison on gap_atk (neg = learned beats greedy under attack)
    print("HEADLINE — Eval/gap_atk (negative = RL beats greedy under attack):")
    for config in sorted(by_config):
        vals = []
        for d in sorted(by_config[config]):
            v = load_scalar(d, "Eval/gap_atk")
            if v is not None and len(v):
                vals.append(float(np.mean(v[-args.window:])))
        if vals:
            a = np.array(vals)
            verdict = "BEATS greedy" if a.mean() < 0 else "does not beat"
            print(f"  {config:18s} gap_atk = {a.mean():+.0f} +/- {a.std():.0f}  -> {verdict}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for config in sorted(by_config):
            curves = [load_scalar(d, "Eval/gap_atk") for d in sorted(by_config[config])]
            curves = [c for c in curves if c is not None and len(c)]
            if not curves:
                continue
            L = min(len(c) for c in curves)
            stack = np.stack([c[:L] for c in curves])
            m, s = stack.mean(0), stack.std(0)
            x = np.arange(L)
            plt.plot(x, m, label=config)
            plt.fill_between(x, m - s, m + s, alpha=0.2)
        plt.axhline(0, color="k", lw=0.8, ls="--")
        plt.xlabel("eval point"); plt.ylabel("gap_atk (neg = RL wins)"); plt.legend()
        plt.title(f"{args.group}: gap_atk mean +/- std across seeds")
        out = f"experiments/{args.group}_gap_atk.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
