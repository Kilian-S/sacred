#!/usr/bin/env python3
"""gen39 step 5C: score the qwenthink16 arm against the PRE-REGISTERED clauses.

Pinned by the step-5c pre-registration:
  PRIMARY  the qwenthink16-trained defender is below the tuned control on >= 4/6 held-out
           cells AND pooled, on >= 2/3 seeds (the act's original clause, new arm).
  PAIRED   pooled paired difference vs llm16 and vs local16, mean +/- sd across seeds; no
           superiority bar. Differences within noise license only "authorship
           indistinguishable".
  MECHANISM the arm's curriculum-strength row joins the table.
The narva test set is UNCHANGED, so these numbers are directly comparable with the banked
n=3 table.

    PYTHONPATH=. ../sacred/.venv/bin/python scratch/gen39_step5c_score.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

O = Path("models/runs/gen39_step5")
ARMS = ["qwenthink16", "llm16", "local16", "random16", "tuned"]
SEEDS = (0, 1, 2)


def sel(arm, s):
    p = O / f"{arm}_seed{s}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return min(r["history"], key=lambda h: h["val"])


def curriculum_strength():
    """Median over training fields of the best authored force's irreducible threat."""
    out = {}
    for f, key in (("curricula.json", None), ("curricula_qwenthink.json", "qwenthink16")):
        p = O / f
        if not p.exists():
            continue
        cur = json.loads(p.read_text())
        for arm, fields in cur.items():
            if key and arm != key and arm in out:
                continue
            best = [max(v[1] for v in entries) if entries and isinstance(entries[0], list)
                    else None for entries in fields.values()]
            best = [b for b in best if b is not None]
            if best:
                out[arm] = float(np.median(best))
    return out


def main():
    rows = {}
    for a in ARMS:
        for s in SEEDS:
            b = sel(a, s)
            if b is not None:
                rows[(a, s)] = np.array(b["cells"], float)
    have = sorted({a for a, _ in rows})
    print(f'{"arm":12s} {"seed":>4s} {"sel@":>6s} {"VAL":>5s} | per-cell held-out damage')
    for a in ARMS:
        for s in SEEDS:
            if (a, s) not in rows:
                continue
            b = sel(a, s)
            print(f'{a:12s} {s:4d} {b["sortie"]:6d} {b["val"]:5.2f} | '
                  + " ".join(f"{c:.4f}" for c in rows[(a, s)]))

    print("\n=== pooled arm means (validation-selected) ===")
    pooled = {}
    for a in have:
        v = [rows[(a, s)].mean() for s in SEEDS if (a, s) in rows]
        pooled[a] = float(np.mean(v))
        print(f'  {a:12s} {pooled[a]:.4f}   (seeds ' + ' '.join(f'{x:.4f}' for x in v) + ')')

    if "qwenthink16" not in have:
        print("\nqwenthink16 runs not present yet; primary and paired rows pending.")
        return

    print("\n=== PRIMARY: qwenthink16 below TUNED on >=4/6 cells AND pooled, on >=2/3 seeds ===")
    ok = 0
    for s in SEEDS:
        if ("qwenthink16", s) not in rows or ("tuned", s) not in rows:
            continue
        Q, T = rows[("qwenthink16", s)], rows[("tuned", s)]
        c = int((Q < T).sum())
        p = Q.mean() < T.mean()
        good = c >= 4 and p
        ok += good
        print(f'  seed {s}: qwenthink16 {Q.mean():.4f} vs tuned {T.mean():.4f}, beats {c}/6, '
              f'pooled {p} -> {"PASS" if good else "FAIL"}')
    print(f'  VERDICT: {ok}/3 seeds -> PRIMARY {"PASS" if ok >= 2 else "FAIL"}')

    print("\n=== PAIRED READOUT (no superiority bar; negative = qwenthink16 better) ===")
    for ctrl in ("llm16", "local16", "random16", "tuned"):
        d = [rows[("qwenthink16", s)].mean() - rows[(ctrl, s)].mean()
             for s in SEEDS if (ctrl, s) in rows and ("qwenthink16", s) in rows]
        if not d:
            continue
        d = np.array(d)
        beats = sum(int((rows[("qwenthink16", s)] < rows[(ctrl, s)]).sum()) for s in SEEDS
                    if (ctrl, s) in rows and ("qwenthink16", s) in rows)
        n_cells = 6 * len(d)
        verdict = ("indistinguishable" if abs(d.mean()) <= d.std(ddof=0)
                   else "separated beyond seed spread")
        print(f'  vs {ctrl:10s} paired {d.mean():+.4f} +/- {d.std(ddof=0):.4f} '
              f'(seeds {" ".join(f"{x:+.4f}" for x in d)}), cells won {beats}/{n_cells} '
              f'-> {verdict}')

    cs = curriculum_strength()
    if cs:
        print("\n=== MECHANISM: curriculum strength (median over training fields) -> defender ===")
        for a in ARMS:
            if a in cs and a in pooled:
                print(f'  {a:12s} curriculum {cs[a]:.4f} -> defender {pooled[a]:.4f}')

    print("\nReminder of the pre-committed consequence: ties everywhere CONFIRM the gen42 NULL "
          "prediction, licensing\n  'curriculum authorship is insensitive to the author's "
          "reasoning strength at matched search budget'.\nNo 'best author' sentence exists "
          "unless it survives the paired columns on >= 2/3 seeds.")


if __name__ == "__main__":
    main()
