"""Summarise gen11 arms vs the pre-registered bars (run when models/runs/gen11_menuhead/DONE)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("models/runs/gen11_menuhead")
BARS = {"pass": 0.295, "baseline_plateau": 0.447}
print("arm | best-ckpt TAP per seed | mean +/- std | best single-ckpt mean | final TAP mean")
means = {}
for arm in ("B", "C", "D", "E"):
    vals, singles, finals = [], [], []
    for s in (0, 1, 2):
        f = OUT / f"{arm}_seed{s}.json"
        if not f.exists():
            continue
        d = json.load(open(f))["fleet_route"]
        vals.append(d["best_tap"]); singles.append(d["best_expl"]); finals.append(d["expl_tap"])
    if not vals:
        print(f"  {arm}: (missing)"); continue
    means[arm] = float(np.mean(vals))
    print(f"  {arm}: {['%.3f' % v for v in vals]} | {np.mean(vals):.3f} +/- {np.std(vals):.3f} | "
          f"{np.mean(singles):.3f} | {np.mean(finals):.3f}")
print(f"\nbars: PASS <= {BARS['pass']} (new multi-convoy headline, Kilian pre-authorised); "
      f"gen10-MC plateau {BARS['baseline_plateau']}")
if means:
    best_arm = min(means, key=means.get)
    print(f"lowest mean: arm {best_arm} at {means[best_arm]:.3f} -> "
          f"{'PASS' if means[best_arm] <= BARS['pass'] else 'partial' if means[best_arm] < BARS['baseline_plateau'] else 'no improvement'}")
