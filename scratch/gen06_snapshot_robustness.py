#!/usr/bin/env python3
"""A3.2 (ROADMAP): robustness-vs-training-time for all gen06 arms.

Post-hoc analysis of the CLOSED gen06 generation (primary untouched). Evaluates EVERY protagonist
snapshot of all six runs under the pathrand and targeted attackers on the 8 VALIDATION instances
(seed base 20_000_019; test instances stay untouched), i.e. the same machinery checkpoint
selection used, swept over training time and both aimed attacks. Motivated by the selection
outcome (two of three vanilla arms selected ep100): does aimed-attack robustness DECLINE with
training time, and does it decline differently in the adversarially-trained arms?

Run: PYTHONPATH=. .venv/bin/python scratch/gen06_snapshot_robustness.py  (~10-20 min, eval only)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.evaluate_portfolio import (
    VAL_SEED_BASE,
    _problem_setup,
    select_best_under_attack,
)

RUNS = [
    "vanilla_seed0", "vanilla_seed1", "vanilla_seed2",
    "dynassign_scripted_seed0", "dynassign_scripted_seed1", "dynassign_scripted_seed2",
]
ATTACKERS = ["pathrand", "targeted"]
INSTANCES = 8


def main() -> None:
    cfg, make_env_for_seed, greedy_factory, _ = _problem_setup("dynassign", arrival_rate=0.06)
    seeds = [VAL_SEED_BASE + i for i in range(INSTANCES)]
    out: dict[str, dict[str, list[dict]]] = {}
    t0 = time.time()
    for run in RUNS:
        run_dir = f"models/runs/gen06_dynassign_matrix/{run}"
        out[run] = {}
        for attacker in ATTACKERS:
            ranked = select_best_under_attack(run_dir, cfg, make_env_for_seed, greedy_factory,
                                              seeds, rollouts=1, attacker=attacker)
            by_ep = sorted(ranked, key=lambda d: d["ep"])
            out[run][attacker] = [
                {"ep": r["ep"], "wait": r["val_attacked_wait"], "sem": r["sem"]} for r in by_ep
            ]
            print(f"[{time.time() - t0:7.1f}s] {run} / {attacker}:")
            print("   " + " ".join(f"ep{r['ep']}:{r['wait']:.0f}" for r in out[run][attacker]))

    path = Path("scratch/gen06_snapshot_robustness.json")
    path.write_text(json.dumps({"instances": INSTANCES, "seed_base": VAL_SEED_BASE,
                                "results": out}, indent=1))
    print(f"\nWrote {path}")

    # Compact trend summary: early (ep50-200) vs late (ep650-800) window means per run/attacker.
    print("\n=== attacked validation wait, early (ep50-200) vs late (ep650-800) ===")
    for run in RUNS:
        for attacker in ATTACKERS:
            rows = out[run][attacker]
            early = [r["wait"] for r in rows if 50 <= r["ep"] <= 200]
            late = [r["wait"] for r in rows if 650 <= r["ep"] <= 800]
            if early and late:
                e = sum(early) / len(early)
                l = sum(late) / len(late)
                print(f"{run:28s} {attacker:9s} early {e:8.0f} -> late {l:8.0f} "
                      f"({(l - e) / e * 100:+5.1f}%)")


if __name__ == "__main__":
    main()
