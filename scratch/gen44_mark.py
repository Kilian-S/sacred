#!/usr/bin/env python3
"""gen44 MARKER: the pre-registered reads of the authoring-budget sweep
(`experiments/gen44_budget_sweep.md`).

  1 per configuration and budget: median best-so-far over the 9 searches + bootstrap 95% CI
  2 separation: per-(field,repeat) PAIRED bootstrap CIs between configurations at each budget
  3 the decision row: does any budget separate an LLM pair beyond repeat noise, and do both
    arms at that budget still sit at or above the trainable knee 0.022
  4 knee row: fraction of searches whose best-at-b falls below 0.022

    PYTHONPATH=. ../sacred/.venv/bin/python scratch/gen44_mark.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path("models/runs/gen44_sweep")
MARKS = (2, 4, 8, 16)
KNEE = 0.022
ORDER = ["qwen35-2b", "qwen35-4b", "qwen35-9b", "qwen35-27b", "qwen3-27b",
         "qwen3-27b_think", "llama-3.3-70b", "local16", "random16"]
PAIRS = [("qwen35-2b", "qwen35-27b", "size 2B vs 27B"),
         ("qwen35-4b", "qwen35-27b", "size 4B vs 27B"),
         ("qwen35-27b", "qwen3-27b", "generation 3.5 vs 3.6"),
         ("qwen3-27b", "qwen3-27b_think", "thinking off vs on"),
         ("llama-3.3-70b", "qwen3-27b_think", "llama vs crown-thinking"),
         ("local16", "qwen3-27b_think", "hill-climb vs crown-thinking"),
         ("local16", "llama-3.3-70b", "hill-climb vs llama (the banked step-5 pair)")]
RNG = np.random.default_rng(0)


def load():
    got = {}
    for c in ORDER:
        p = D / f"{c}.json"
        if p.exists():
            got[c] = json.loads(p.read_text())
    return got


def vec(run, b):
    """best-at-budget b for each search, keyed so pairs align on (field, repeat)."""
    return {k: (v["at"][str(b)] if v["at"][str(b)] is not None else np.nan)
            for k, v in sorted(run.items())}


def boot(d, n=20000):
    if len(d) == 0 or np.all(np.isnan(d)):
        return float("nan"), float("nan")
    m = np.array([np.mean(RNG.choice(d, len(d))) for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    got = load()
    print(f"configurations present: {len(got)}/{len(ORDER)}"
          + (f" (missing {[c for c in ORDER if c not in got]})" if len(got) < len(ORDER) else ""))
    if not got:
        return

    print(f"\n{'=' * 96}\nBEST-AT-BUDGET, median over searches [bootstrap 95% CI]; "
          f"knee = {KNEE}\n{'=' * 96}")
    hdr = f'{"config":<18}{"n":>3}' + "".join(f'{"@" + str(b):>22}' for b in MARKS)
    print(hdr)
    for c in ORDER:
        if c not in got:
            continue
        row = f'{c:<18}{len(got[c]):>3}'
        for b in MARKS:
            v = np.array([x for x in vec(got[c], b).values()], float)
            v = v[~np.isnan(v)]
            lo, hi = boot(v)
            row += f'{np.median(v):>10.4f} [{lo:.3f},{hi:.3f}]'
        print(row)

    print(f"\n{'=' * 96}\nKNEE ROW: share of searches whose best-at-b is BELOW {KNEE} "
          f"(a budget that starves everyone answers nothing)\n{'=' * 96}")
    print(f'{"config":<18}' + "".join(f'{"@" + str(b):>10}' for b in MARKS))
    for c in ORDER:
        if c not in got:
            continue
        row = f'{c:<18}'
        for b in MARKS:
            v = np.array([x for x in vec(got[c], b).values()], float)
            row += f'{np.mean(v[~np.isnan(v)] < KNEE):>10.0%}'
        print(row)

    print(f"\n{'=' * 96}\nSEPARATION: per-(field,repeat) paired difference, "
          f"positive = SECOND config stronger\n{'=' * 96}")
    any_sep = []
    for a, b_, label in PAIRS:
        if a not in got or b_ not in got:
            continue
        print(f'  {label}')
        for bud in MARKS:
            va, vb = vec(got[a], bud), vec(got[b_], bud)
            keys = [k for k in va if k in vb
                    and not np.isnan(va[k]) and not np.isnan(vb[k])]
            d = np.array([vb[k] - va[k] for k in keys])
            lo, hi = boot(d)
            sep = lo > 0 or hi < 0
            if sep:
                any_sep.append((label, bud, float(np.median([va[k] for k in keys])),
                                float(np.median([vb[k] for k in keys]))))
            print(f'      @{bud:<3} pairs {len(keys):>2}  mean {d.mean():+.4f}  '
                  f'CI [{lo:+.4f}, {hi:+.4f}]  '
                  f'{"SEPARATED" if sep else "indistinguishable"}')

    print(f"\n{'=' * 96}\nTHE PRE-REGISTERED DECISION\n{'=' * 96}")
    if not any_sep:
        print("  No configuration pair separates at ANY tested budget.")
        print("  Licensed: 'curriculum authorship is insensitive to the author's reasoning")
        print("  strength at every tested search budget, measured with repeats'.")
    else:
        for label, bud, ma, mb in any_sep:
            usable = min(ma, mb) >= KNEE
            print(f"  SEPARATED: {label} at budget {bud} "
                  f"(medians {ma:.4f} / {mb:.4f}) -> "
                  f"{'USABLE, both above the knee' if usable else 'NOT usable, an arm is below the knee'}")
        print("  Any future training comparison must use a separating, usable budget.")


if __name__ == "__main__":
    main()
