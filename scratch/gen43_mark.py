#!/usr/bin/env python3
"""gen43 EXAM MARKER: the three pinned marks, the pinned per-item paired contrasts with
bootstrap CIs, and the size-Spearman. Reads whatever config papers exist and reports the rest
as pending, so it can be run repeatedly as the night's papers land.

Pinned by `experiments/gen43_exam.md`:
  (a) mean share of ceiling over non-format-fail items
  (b) items solved EXACTLY (a count)
  (c) mean percentile of the chosen combination in the item's full value table
  contrasts, per-item PAIRED with bootstrap CIs: 4B-9B, 9B-27B, 3.5-27B vs crown-off
  (generation), crown off vs on (thinking), llama vs the 27Bs (reference, never in family
  statistics); Spearman of score vs parameter count over the 3.5 rungs.
  No superiority sentence below a paired CI excluding zero; both directions reportable.
  Format-fail counts are first-class rows.

    PYTHONPATH=. ../sacred/.venv/bin/python scratch/gen43_mark.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path("models/runs/gen43_exam")
RUNG_B = {"qwen35-2b": 2.0, "qwen35-4b": 4.0, "qwen35-9b": 9.0, "qwen35-27b": 27.0}
CONFIGS = ["qwen35-2b", "qwen35-4b", "qwen35-9b", "qwen35-27b",
           "qwen3-27b", "qwen3-27b_think", "llama-3.3-70b"]
CONTRASTS = [("qwen35-4b", "qwen35-9b", "size 4B->9B"),
             ("qwen35-9b", "qwen35-27b", "size 9B->27B"),
             ("qwen35-2b", "qwen35-27b", "size CUMULATIVE 2B->27B"),
             ("qwen35-4b", "qwen35-27b", "size CUMULATIVE 4B->27B"),
             ("qwen35-27b", "qwen3-27b", "GENERATION 3.5->3.6 at 27B"),
             ("qwen3-27b", "qwen3-27b_think", "THINKING off->on (crown)"),
             ("llama-3.3-70b", "qwen3-27b", "reference: llama-70B vs crown"),
             ("llama-3.3-70b", "qwen35-27b", "reference: llama-70B vs 3.5-27B")]
RNG = np.random.default_rng(0)


def load():
    got = {}
    for c in CONFIGS:
        p = D / f"{c}.json"
        if p.exists():
            got[c] = json.loads(p.read_text())
    return got


def boot_paired(d, n=20000):
    if len(d) == 0:
        return float("nan"), float("nan")
    m = np.array([np.mean(RNG.choice(d, len(d))) for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    got = load()
    missing = [c for c in CONFIGS if c not in got]
    print(f"papers present: {len(got)}/{len(CONFIGS)}")
    if missing:
        print(f"pending: {', '.join(missing)}")
    if not got:
        return
    print(f"\n{'=' * 92}\nMARKS (a) share of ceiling  (b) solved exactly  (c) mean percentile"
          f"\n{'=' * 92}")
    print(f'{"config":<18}{"n":>4}{"fmt-fail":>9}{"share":>9}{"solved":>8}{"pct":>8}')
    for c in CONFIGS:
        if c not in got:
            continue
        s = got[c]["summary"]
        sh = "  n/a" if s["mean_share"] is None else f'{s["mean_share"]:.3f}'
        pc = "  n/a" if s["mean_pct"] is None else f'{s["mean_pct"]:.3f}'
        print(f'{c:<18}{s["n"]:>4}{s["format_fail"]:>9}{sh:>9}{s["solved"]:>8}{pc:>8}')

    print(f"\n{'=' * 92}\nPER-ITEM PAIRED CONTRASTS (share of ceiling; positive = the SECOND "
          f"config scores higher)\n{'=' * 92}")
    for a, b, label in CONTRASTS:
        if a not in got or b not in got:
            print(f'  {label:<32} pending ({a if a not in got else b} not sat)')
            continue
        ra = {r["id"]: r for r in got[a]["rows"]}
        rb = {r["id"]: r for r in got[b]["rows"]}
        ids = [i for i in ra if ra[i]["status"] == "ok" and rb.get(i, {}).get("status") == "ok"]
        d = np.array([rb[i]["share"] - ra[i]["share"] for i in ids])
        lo, hi = boot_paired(d)
        sep = "SEPARATED" if (lo > 0 or hi < 0) else "indistinguishable"
        print(f'  {label:<32} pairs {len(ids):>3}  mean diff {d.mean():+.4f}  '
              f'CI [{lo:+.4f}, {hi:+.4f}]  {sep}')

    rung = [(RUNG_B[c], got[c]["summary"]["mean_share"]) for c in RUNG_B
            if c in got and got[c]["summary"]["mean_share"] is not None]
    if len(rung) >= 3:
        from itertools import permutations

        from scipy.stats import spearmanr
        x, y = zip(*rung)
        rho, _ = spearmanr(x, y)
        # scipy's asymptotic p is meaningless at n=4 (rho=1 -> t=inf -> p=0). Exact
        # permutation p instead: the share of orderings at least this monotone.
        allr = [abs(spearmanr(x, p_)[0]) for p_ in permutations(y)]
        p_exact = float(np.mean([r >= abs(rho) - 1e-12 for r in allr]))
        print(f'\nSpearman(share vs parameter count) over {len(rung)} Qwen3.5 rungs: '
              f'rho {rho:+.3f}, exact permutation p {p_exact:.3f} '
              f'({len(allr)} orderings; the floor at n={len(rung)} is {2/len(allr):.3f})')
    else:
        print(f'\nSpearman: pending ({len(rung)}/4 Qwen3.5 rungs sat)')

    print("\nSOLVED-EXACTLY counts (a count, so not subject to the median's small-sample wobble):")
    for c in CONFIGS:
        if c in got:
            print(f'  {c:<18}{got[c]["summary"]["solved"]:>3}/40')


if __name__ == "__main__":
    main()
