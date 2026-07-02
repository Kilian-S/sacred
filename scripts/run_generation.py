#!/usr/bin/env python3
"""Launch a *generation* of seeded training runs in parallel (experiment management).

A "generation" = one git-committed code state + one experiment group answering one question,
containing one-or-more config recipes x several seeds. Runs nest under
logs/tb_runs/<group>/ and models/runs/<group>/ (TensorBoard groups them), and a ledger with
the git SHA + spec is written to experiments/<group>.md so runs stay reproducible and
comparable (only compare *within* a generation / code state).

Example (the gen01 ERB ablation: assign with vs without ERB seeding, 3 seeds each = 6 runs):
    PYTHONPATH=. python scripts/run_generation.py --group gen01_erb_ablation \
        --configs assign_erb,assign_noerb --seeds 0,1,2 \
        --episodes 1000 --switch-every 50 --eval-every 50 --threads 3 --max-concurrent 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Config recipes: name -> extra train_sacred.py args that define that condition.
RECIPES = {
    "assign_erb": ["--problem", "assign", "--erb-path", "data/erb_assign.pt"],
    "assign_noerb": ["--problem", "assign", "--preseed-buffer", "False"],
    "stage0": ["--problem", "stage0", "--preseed-buffer", "False"],
    # Stage 1.5 dynamic assignment at the gate's rho~1 point + load-scaled antagonist budget.
    "dynassign": ["--problem", "dynassign", "--arrival-rate", "0.06",
                  "--congestion-budget", "4000", "--preseed-buffer", "False"],
    # gen03 Phase-1: the NON-adversarial control (identical to dynassign but the antagonist is
    # inert and the protagonist trains every episode) — see experiments/gen03_robustness_dynassign.md.
    "vanilla": ["--problem", "dynassign", "--arrival-rate", "0.06",
                "--congestion-budget", "4000", "--preseed-buffer", "False", "--vanilla"],
}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_ledger(group: str, configs: list[str], seeds: list[int], common: list[str]) -> None:
    os.makedirs("experiments", exist_ok=True)
    path = f"experiments/{group}.md"
    if os.path.exists(path):
        # A hand-written ledger (pre-registered metric etc.) already exists — append the launch
        # record instead of clobbering it.
        with open(path, "a") as f:
            f.write(f"\n## Launch record ({time.strftime('%Y-%m-%d %H:%M')})\n\n")
            f.write(f"- **git SHA:** `{git_sha()}`\n")
            f.write(f"- **configs:** {', '.join(configs)}  **seeds:** {seeds}\n")
            f.write(f"- **common args:** `{' '.join(common)}`\n")
        print(f"Appended launch record to existing ledger {path}")
        return
    with open(path, "w") as f:
        f.write(f"# Generation: {group}\n\n")
        f.write(f"- **git SHA:** `{git_sha()}` (runs are only comparable within this code state)\n")
        f.write(f"- **date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- **configs:** {', '.join(configs)}\n")
        f.write(f"- **seeds:** {seeds}\n")
        f.write(f"- **common args:** `{' '.join(common)}`\n\n")
        f.write("## Question\n\n_(fill in: what does this generation test?)_\n\n")
        f.write("## Result\n\n_(fill in after `aggregate_generation.py`: mean +/- std of gap_atk, etc.)_\n")
    print(f"Wrote ledger {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Launch a generation of seeded training runs in parallel.")
    p.add_argument("--group", required=True)
    p.add_argument("--configs", default="assign_erb,assign_noerb", help="comma-separated recipe names")
    p.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--switch-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=3, help="torch threads per run (total <= 10 cores)")
    p.add_argument("--max-concurrent", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="print commands without launching")
    args = p.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    for c in configs:
        if c not in RECIPES:
            sys.exit(f"unknown config recipe {c!r}; known: {list(RECIPES)}")

    common = [
        "--episodes", str(args.episodes), "--switch-every", str(args.switch_every),
        "--batch-size", str(args.batch_size), "--hidden-dim", str(args.hidden_dim),
        "--device", "cpu", "--eval-every", str(args.eval_every),
        "--group", args.group, "--threads", str(args.threads),
    ]
    if not args.dry_run:
        write_ledger(args.group, configs, seeds, common)

    jobs = [(c, s) for c in configs for s in seeds]
    print(f"Generation {args.group}: {len(jobs)} runs, {args.threads} threads each, "
          f"max {args.max_concurrent} concurrent ({args.threads * args.max_concurrent} cores).")

    os.makedirs(f"experiments/{args.group}", exist_ok=True)
    running: list[tuple] = []
    for config, seed in jobs:
        # throttle to max_concurrent
        while len(running) >= args.max_concurrent:
            for proc, name, fh in running[:]:
                if proc.poll() is not None:
                    fh.close()
                    print(f"  finished: {name} (exit {proc.returncode})")
                    running.remove((proc, name, fh))
            if len(running) >= args.max_concurrent:
                time.sleep(5)

        cmd = [sys.executable, "scripts/train_sacred.py", "--tag", config, "--seed", str(seed)] + RECIPES[config] + common
        name = f"{config}_seed{seed}"
        if args.dry_run:
            print("  DRY:", " ".join(cmd))
            continue
        log = open(f"experiments/{args.group}/{name}.log", "w")
        env = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS=str(args.threads))
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        running.append((proc, name, log))
        print(f"  launched: {name} (pid {proc.pid})")
        time.sleep(2)  # stagger startup

    for proc, name, fh in running:
        proc.wait(); fh.close()
        print(f"  finished: {name} (exit {proc.returncode})")
    print(f"\nGeneration {args.group} complete. Aggregate with:\n"
          f"  PYTHONPATH=. python scripts/aggregate_generation.py --group {args.group}")


if __name__ == "__main__":
    main()
