#!/usr/bin/env python3
"""Scores the gen45 confirmation artefacts against the pre-registered bars.

Selection is by validation (fields 45400-45403) and never by the gated set, so nothing here
selects on test. For each arm the validation-selected checkpoint is taken and its gated-field
values are scored against that field's own exact references.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("models/runs/gen45_unified")
SEEDS = (10, 11, 12)


def load(path: Path):
    d = json.loads(path.read_text())
    hist = d["history"]
    best = min(hist, key=lambda h: h[1])          # h[1] = VAL ratio, lower is better
    return d, best, hist[-1]


def gated_rows(entry, refs):
    """{field: (value, cap, best_blind, fitted, hist_opt)} for the gated fields only."""
    rows = entry[5]
    out = {}
    for name, val in rows.items():
        if name.startswith("gated"):
            r = refs[name]
            out[name] = (val, r["cap"], r["best_blind"], r["fitted"], r["hist_opt"])
    return dict(sorted(out.items()))


def main() -> int:
    arms = {}
    for s in SEEDS:
        p = OUT / f"confirm_seed{s}.json"
        if not p.exists():
            print(f"[score] MISSING {p}")
            return 2
        arms[f"seed{s}"] = load(p)
    pblind = OUT / "blind_seed10.json"
    blind = load(pblind) if pblind.exists() else None

    print("=" * 100)
    print("PER-SEED, at the VALIDATION-selected checkpoint (selection never touches the "
          "gated set)")
    print("=" * 100)
    all_ratios, all_optr, beats_total, blindbeat_total, cells = [], [], 0, 0, 0
    for name, (d, best, final) in arms.items():
        g = gated_rows(best, d["refs"])
        beats = sum(v < c for v, c, _, _, _ in g.values())
        bb = sum(v < b for v, _, b, _, _ in g.values())
        ratios = [v / c for v, c, _, _, _ in g.values()]
        optr = [v / o for v, _, _, _, o in g.values()]
        beats_total += beats
        blindbeat_total += bb
        cells += len(g)
        all_ratios += ratios
        all_optr += optr
        print(f"\n{name}  val-selected @ sortie {best[0]}  (VAL {best[1]:.3f}, "
              f"alpha {best[7]:.2f}, rw [{best[6][0]:+.2f}, {best[6][1]:+.2f}, "
              f"{best[6][2]:+.2f}])")
        print(f"  {'field':<14}{'policy':>9}{'CAP':>9}{'ratio':>8}{'blindfam':>10}"
              f"{'fitted':>9}{'optimum':>9}{'x opt':>8}")
        for f, (v, c, b, fi, o) in g.items():
            print(f"  {f:<14}{v:>9.4f}{c:>9.4f}{v / c:>8.3f}{b:>10.4f}{fi:>9.4f}"
                  f"{o:>9.4f}{v / o:>8.2f}")
        print(f"  beats CAP {beats}/{len(g)}   beats payoff-blind family {bb}/{len(g)}   "
              f"mean ratio-to-cap {sum(ratios) / len(ratios):.3f}   "
              f"mean x optimum {sum(optr) / len(optr):.2f}")
        print(f"  drift: selected VAL {best[1]:.3f} -> final VAL {final[1]:.3f} "
              f"(sortie {final[0]})")

    pooled_cap = sum(all_ratios) / len(all_ratios)
    pooled_opt = sum(all_optr) / len(all_optr)
    seed_pass = sum(
        1 for (d, best, _) in arms.values()
        if sum(v < c for v, c, _, _, _ in gated_rows(best, d["refs"]).values()) >= 4)

    print("\n" + "=" * 100)
    print("VERDICT AGAINST THE PRE-REGISTERED BARS")
    print("=" * 100)
    primary = seed_pass >= 2 and pooled_cap < 1.0
    strong = pooled_opt <= 2.5
    print(f"PRIMARY  >=4/6 gated fields on >=2/3 seeds AND pooled below cap: "
          f"{seed_pass}/3 seeds qualify, pooled ratio-to-cap {pooled_cap:.3f} "
          f"-> {'PASS' if primary else 'FAIL'}")
    print(f"STRONG   pooled <= 2.5x the exact optimum: {pooled_opt:.2f}x "
          f"-> {'PASS' if strong else 'FAIL'}")
    if blind is not None:
        bd, bbest, bfinal = blind
        bg = gated_rows(bbest, bd["refs"])
        bbeats = sum(v < c for v, c, _, _, _ in bg.values())
        bratio = sum(v / c for v, c, _, _, _ in bg.values()) / len(bg)
        causal = bbeats == 0
        print(f"CAUSAL   blinded control beats the cap 0/6: {bbeats}/{len(bg)} at "
              f"{bratio:.3f}x cap (rw [{bbest[6][0]:+.2f}, {bbest[6][1]:+.2f}, "
              f"{bbest[6][2]:+.2f}]) -> {'PASS' if causal else 'FAIL'}")
    else:
        causal = False
        print("CAUSAL   blinded control artefact ABSENT")
    print(f"REPORTED beats the payoff-blind rule family on {blindbeat_total}/{cells} "
          f"seed-field cells")
    print(f"\nOVERALL: PRIMARY {'PASS' if primary else 'FAIL'}, "
          f"STRONG {'PASS' if strong else 'FAIL'}, "
          f"CAUSAL {'PASS' if causal else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
