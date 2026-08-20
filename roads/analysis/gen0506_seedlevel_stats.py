#!/usr/bin/env python3
"""A3.4 (ROADMAP): seed-level sensitivity statistics for the gen05 and gen06 primaries.

Post-hoc dual-reporting note (primaries untouched). The pre-registered primaries pooled paired
instances across the 3 seed pairings (n=72/90). This script computes the conservative
seed-level view: per-pairing mean dD (n=3 pairings as the unit), mean/SD/SEM, the t(2) 95% CI,
and the one-sided sign probability, straight from the raw per-instance results in the portfolio
JSONs (verifying the ledgered per-pairing numbers on the way).

Run: PYTHONPATH=. .venv/bin/python analysis/gen0506_seedlevel_stats.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

T2_975 = 4.302653  # t distribution, df=2, two-sided 95%


def pairing_dd(path: Path, attack: str) -> tuple[float, float]:
    """Mean and 95% CI half-width of dD = D(vanilla) - D(scripted) on paired instances."""
    d = json.load(open(path))["results"]
    dv = [w - b for w, b in zip(d["vanilla"][attack], d["vanilla"]["none"])]
    ds = [w - b for w, b in zip(d["scripted"][attack], d["scripted"]["none"])]
    diffs = [x - y for x, y in zip(dv, ds)]
    mean = statistics.mean(diffs)
    sem = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return mean, 1.96 * sem


def seed_level(name: str, files: list[Path], attack: str) -> None:
    means = []
    print(f"\n=== {name}, attack = {attack} ===")
    for f in files:
        m, ci = pairing_dd(f, attack)
        means.append(m)
        print(f"  {f.name}: dD = {m:+8.1f} ± {ci:6.1f} (95% CI, instance level)")
    mu = statistics.mean(means)
    sd = statistics.stdev(means)
    sem = sd / math.sqrt(len(means))
    half = T2_975 * sem
    neg = sum(1 for m in means if m < 0)
    sign_p = 0.5 ** len(means)  # one-sided, all-same-sign
    print(f"  SEED LEVEL (n={len(means)} pairings): mean {mu:+.1f}, SD {sd:.1f}, "
          f"t(2) 95% CI [{mu - half:+.1f}, {mu + half:+.1f}] "
          f"({'excludes' if (mu - half) * (mu + half) > 0 else 'INCLUDES'} zero); "
          f"sign consistency {neg}/{len(means)} negative (one-sided p = {sign_p:.3f})")


def main() -> None:
    exp = Path("experiments")
    gen06 = [exp / f"gen06_portfolio_pair{k}.json" for k in range(3)]
    gen05 = [exp / f"gen05_portfolio_pair{k}.json" for k in range(3)]
    seed_level("gen06 primary (dD_targeted)", gen06, "targeted")
    seed_level("gen06 secondary (dD_pathrand)", gen06, "pathrand")
    seed_level("gen05 primary (dD_gateway)", gen05, "gateway")


if __name__ == "__main__":
    main()
