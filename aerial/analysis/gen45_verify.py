#!/usr/bin/env python3
"""Recomputes the banked gen45 confirmation figures from the raw artefacts, with selection
logic written independently of analysis/gen45_score.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("models/runs/gen45_unified")


def sel(hist):
    return min(hist, key=lambda h: h[1])          # validation ratio, lower is better


def main():
    refs0 = None
    all_r, all_o, table = [], [], {}
    for s in (10, 11, 12):
        d = json.loads((OUT / f"confirm_seed{s}.json").read_text())
        refs = {k: v for k, v in d["refs"].items() if k.startswith("gated")}
        if refs0 is None:
            refs0 = refs
        else:
            same = all(abs(refs[k][f] - refs0[k][f]) < 1e-12
                       for k in refs for f in refs[k])
            print(f"seed{s}: gated refs identical to seed10's -> {same}")
        b = sel(d["history"])
        rows = {k: v for k, v in b[5].items() if k.startswith("gated")}
        beats = sum(rows[k] < refs[k]["cap"] for k in rows)
        bb = sum(rows[k] < refs[k]["best_blind"] for k in rows)
        r = [rows[k] / refs[k]["cap"] for k in rows]
        o = [rows[k] / refs[k]["hist_opt"] for k in rows]
        all_r += r; all_o += o
        fin = d["history"][-1]
        print(f"seed{s}: sel@{b[0]} VAL {b[1]:.3f} beatsCAP {beats}/6 beatsBLINDfam {bb}/6 "
              f"meanratio {np.mean(r):.3f} meanxopt {np.mean(o):.2f} "
              f"drift {b[1]:.3f}->{fin[1]:.3f} rw[{b[6][0]:+.2f},{b[6][1]:+.2f},{b[6][2]:+.2f}] "
              f"alpha {b[7]:.2f}")
        for k in sorted(rows):
            table.setdefault(k, {})[f"s{s}"] = rows[k]

    print(f"\nPOOLED ratio-to-cap {np.mean(all_r):.3f}   pooled x-optimum {np.mean(all_o):.2f}")

    d = json.loads((OUT / "blind_seed10.json").read_text())
    refs = {k: v for k, v in d["refs"].items() if k.startswith("gated")}
    hist = d["history"]
    fw_max = max(max(abs(h[6][1]), abs(h[6][2])) for h in hist)
    b = sel(hist)
    rows = {k: v for k, v in b[5].items() if k.startswith("gated")}
    beats = sum(rows[k] < refs[k]["cap"] for k in rows)
    r = [rows[k] / refs[k]["cap"] for k in rows]
    print(f"BLIND:  sel@{b[0]} beatsCAP {beats}/6 meanratio {np.mean(r):.3f} "
          f"max|rw[recency,doctrine]| over ALL {len(hist)} evals = {fw_max:.6f} "
          f"(blind flag in json: {d['blind']})")

    print("\nper-field refs + selected policy values (ledger cross-check):")
    for k in sorted(refs0):
        rr = refs0[k]
        print(f"  {k}: CAP {rr['cap']:.4f} blindfam {rr['best_blind']:.4f} "
              f"fitted {rr['fitted']:.4f} opt {rr['hist_opt']:.4f} | "
              + " ".join(f"{table[k][f's{s}']:.4f}" for s in (10, 11, 12)))
    fr = [np.mean([rr["fitted"] / table[k][f"s{s}"] for s in (10, 11, 12)])
          for k, rr in refs0.items()]
    print(f"fitted-vs-policy: fitted ahead by factor "
          f"{np.mean([1/x for x in fr]):.2f} (policy/fitted)")

    print("\nATTEMPT wave (diagnostic):")
    for s in (0, 1, 2):
        d = json.loads((OUT / f"attempt_seed{s}.json").read_text())
        b = sel(d["history"])
        fin = d["history"][-1]
        print(f"  seed{s}: sel@{b[0]} VAL {b[1]:.3f} drift->{fin[1]:.3f} "
              f"rw[{b[6][0]:+.2f},{b[6][1]:+.2f},{b[6][2]:+.2f}] alpha {b[7]:.2f}")


if __name__ == "__main__":
    main()
