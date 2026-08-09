#!/usr/bin/env python3
"""gen45 attempt-wave gate (mechanical form of the pre-registered 'read the attempt
diagnostics before spending the confirmation wave').

The confirmation wave evaluates the PRISTINE gated set once, so it may only fire on an
unambiguously strong attempt. This script reads the three attempt artefacts, takes each
seed's VALIDATION-SELECTED eval point (best VAL ratio, the protocol's own selection rule)
and requires, at that point, beats-CAP on both dev fields AND beats-BLIND on both, for all
three seeds. Anything less exits non-zero and the confirmation stays unspent for the analyst
to judge by eye.

Exit 0 = PASS (fire the confirmation), 1 = FAIL/AMBIGUOUS (do not fire), 2 = artefacts absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("models/runs/gen45_unified")
SEEDS = (0, 1, 2)
N_DEV = 2


def main() -> int:
    rows, ok = [], True
    for s in SEEDS:
        p = OUT / f"attempt_seed{s}.json"
        if not p.exists():
            print(f"[gate] MISSING {p}")
            return 2
        hist = json.loads(p.read_text())["history"]
        if not hist:
            print(f"[gate] EMPTY history for seed {s}")
            return 2
        # history entries: (sortie, va, te, beats, bblind, rows, fw, alpha); VAL lower = better
        best = min(hist, key=lambda h: h[1])
        sortie, va, te, beats, bblind = best[0], best[1], best[2], best[3], best[4]
        fw, alpha = best[6], best[7]
        good = (beats == N_DEV) and (bblind == N_DEV)
        ok &= good
        rows.append((s, sortie, va, te, beats, bblind, fw, alpha, good))

    print(f"{'seed':>4} {'val-sel@':>9} {'VAL':>6} {'devratio':>8} {'beatCAP':>8} "
          f"{'beatBLIND':>10} {'alpha':>6}  rw[expo,recency,doctrine]   verdict")
    for s, sortie, va, te, beats, bblind, fw, alpha, good in rows:
        print(f"{s:>4} {sortie:>9} {va:>6.3f} {te:>8.3f} {beats:>6}/{N_DEV} "
              f"{bblind:>8}/{N_DEV} {alpha:>6.2f}  "
              f"[{fw[0]:+.2f},{fw[1]:+.2f},{fw[2]:+.2f}]  {'PASS' if good else 'FAIL'}")
    print(f"[gate] {'PASS' if ok else 'FAIL'}: "
          f"{sum(r[8] for r in rows)}/{len(rows)} seeds beat both dev objects at their "
          f"validation-selected checkpoint")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
