#!/usr/bin/env python3
"""A3.1 (ROADMAP): reproduce the gen06 arm-comparison training telemetry.

Post-hoc analysis of the CLOSED gen06 generation (primary untouched): windowed means of key
training scalars, vanilla vs scripted-adversarial arms, from the tfevents on disk. First
reported as a session analysis on 2026-07-06 (DIRECTION.md §4); this script is the committed
reproduction. Output feeds the gen06 ledger's post-hoc appendix.

Run: PYTHONPATH=. .venv/bin/python analysis/gen06_telemetry_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/tb_runs/gen06_dynassign_matrix")
RUNS = [
    "vanilla_seed0", "vanilla_seed1", "vanilla_seed2",
    "dynassign_scripted_seed0", "dynassign_scripted_seed1", "dynassign_scripted_seed2",
]
TAGS = [
    "Params/Protagonist_Alpha",
    "Value/Protagonist_Entropy",
    "Value/Protagonist_Q_Spread",
    "Value/Protagonist_Q",
    "Loss/Protagonist_Critic",
    "Episode/Total_Wait",
    "Episode/Delivery_Rate",
    "Episode/Final_Queue",
]
WINDOWS = [(1, 100), (350, 450), (700, 800)]


def windowed_means(run: str, tag: str) -> list[float | None]:
    events = sorted((ROOT / run).glob("**/events.out.tfevents.*"))
    if not events:
        return [None] * len(WINDOWS)
    acc = EventAccumulator(str(events[0].parent), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return [None] * len(WINDOWS)
    scalars = acc.Scalars(tag)
    steps = np.array([s.step for s in scalars])
    values = np.array([s.value for s in scalars])
    out = []
    for lo, hi in WINDOWS:
        m = (steps >= lo) & (steps <= hi)
        out.append(float(values[m].mean()) if m.any() else None)
    return out


def main() -> None:
    table: dict[str, dict[str, list[float | None]]] = {}
    for tag in TAGS:
        table[tag] = {run: windowed_means(run, tag) for run in RUNS}
        header = " / ".join(f"ep{lo}-{hi}" for lo, hi in WINDOWS)
        print(f"\n=== {tag} (windowed means: {header}) ===")
        for run in RUNS:
            vals = table[tag][run]
            cells = " ".join("     n/a" if v is None else f"{v:9.3f}" for v in vals)
            print(f"{run:28s} {cells}")

    out = Path("analysis/gen06_telemetry.json")
    out.write_text(json.dumps({"windows": WINDOWS, "table": table}, indent=1))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
