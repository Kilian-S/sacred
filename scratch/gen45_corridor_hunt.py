#!/usr/bin/env python3
"""gen45 Phase 0: the corridor hunt on the UNIFIED real-corridor game (ORACLE-ONLY, free;
pre-registered gates G1-G2 + fail branch in experiments/gen45_unified_corridor.md).

Substrate FROZEN to gen39's values: kgd_gvardeysk, terrain v2 (hidden_leth 1.0, conceal_reach
0.85), range_scale 0.7, 200 quota sites, spacing 2.0, standoff 4.0; hidden field = the gen39
MULTIPLIER draw (band 0.55-1.0) on terrain lethality. Enemy = the gen32 anticipatory doctrine
(DynTheatre, verbatim import) running on ConcealBase, i.e. the gen39 machinery's flat
full-map-relocation limit. The pre-registered freedom is enemy-behavioural only (q, tau,
w in {2, 3}); w=2 is the preferred pin and w=3 runs only if w=2 fails its gates.

SELF-CHECK (hard assert, runs before any gate number is read): ConcealDyn with one team,
sigma_r huge and same_class=False must reproduce DynTheatre's aim matrix, stepdmg and
history_opt on the same (base, field) - the flat-limit regression the ledger requires.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from scratch.gen32_theatre_hunt import DynTheatre, rule_family
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import terrain_v2

PATH = "data/maps/theatre_kgd_gvardeysk_vec.json"
RANGE_SCALE = 0.7                      # gen39's RM at the kgd reference (lateral factor 1.0)
CONCEAL_REACH = 0.85                   # gen39 pinned table
DOC32 = (0.6, 0.2, 0.3)                # q_rep, q_flee, q_ar (frozen components)
TAU = 0.10
HUNT_FIELDS = tuple(range(45001, 45013))     # burned by this hunt, per the ledger
G1_BAR, G2_BAR, G2_COUNT = 2.0, 1.25, 10     # pre-registered


def make_base() -> ConcealBase:
    return ConcealBase(PATH, terrain=terrain_v2(hidden_leth=1.0, conceal_reach=CONCEAL_REACH),
                       range_scale=RANGE_SCALE, spacing_km=2.0, standoff_km=4.0, n_sites=200)


def lethality_for(base: ConcealBase, seed: int) -> np.ndarray:
    """Terrain lethality x the gen39 multiplier draw (band 0.55-1.0); hidden_leth pinned 1.0."""
    return base.lethality(resample_field(base.coords, seed), hidden_leth=1.0)


def selfcheck(base: ConcealBase, seed: int = 45001, w: int = 2) -> None:
    pp = lethality_for(base, seed)
    g32 = DynTheatre(base, pp, w, TAU, *DOC32)
    g39 = ConcealDyn(base, pp, [0], w=w, tau=TAU, q_rep=DOC32[0], q_flee=DOC32[1],
                     q_ar=DOC32[2], sigma_r=1e6, same_class=False)
    d_step = float(np.abs(g32.stepdmg - g39.stepdmg).max())
    d_opt = abs(g32.history_opt() - g39.history_opt())
    print(f"[selfcheck] flat-limit anchor on field {seed} w={w}: "
          f"max|stepdmg diff|={d_step:.3e}, |history_opt diff|={d_opt:.3e}", flush=True)
    assert d_step < 1e-8 and d_opt < 1e-8, "flat-limit anchor FAILED; do not read gate numbers"


def cell(base: ConcealBase, seed: int, w: int) -> dict:
    t0 = time.time()
    g = DynTheatre(base, lethality_for(base, seed), w, TAU, *DOC32)
    rows = rule_family(g)
    hopt = g.history_opt()
    blind = [k for k in rows if k.startswith(("anti_", "rot_"))]
    bb = min(rows[k] for k in blind)
    bbn = min(blind, key=lambda k: rows[k])
    fit = min(rows[k] for k in ("myopic_dodge", "softdodge*fit", "composed*fit"))
    cap = min(rows["iid_eq"], rows["static_localopt*fit"])
    g1, g2, g3 = cap / max(hopt, 1e-9), bb / max(hopt, 1e-9), fit / max(hopt, 1e-9)
    print(f"field {seed} w{w}: cap={cap:.4f} blind={bb:.4f}({bbn[:18]}) fit={fit:.4f} "
          f"hopt={hopt:.4f} | G1={g1:.2f} G2={g2:.2f} G3={g3:.2f} eq={g.eq_static:.4f} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return {"seed": seed, "w": w, "tau": TAU, "q": DOC32, "cap": cap, "best_blind": bb,
            "best_blind_rule": bbn, "fit": fit, "hist_opt": hopt,
            "G1": g1, "G2": g2, "G3": g3, "eq_static": g.eq_static, "rows": rows}


def gates(cells: list[dict], w: int) -> bool:
    cs = [c for c in cells if c["w"] == w]
    g1_min = min(c["G1"] for c in cs)
    g2_pass = sum(c["G2"] >= G2_BAR for c in cs)
    ok = g1_min >= G1_BAR and g2_pass >= G2_COUNT
    print(f"[gates w={w}] G1 min {g1_min:.2f} (bar {G1_BAR}) | G2 >= {G2_BAR} on "
          f"{g2_pass}/{len(cs)} (bar {G2_COUNT}) | G3 median "
          f"{float(np.median([c['G3'] for c in cs])):.2f} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=0, help="0 = w=2 first, w=3 only if w=2 fails")
    ap.add_argument("--out", default="models/runs/gen45_hunt.json")
    args = ap.parse_args()

    t0 = time.time()
    base = make_base()
    n_cls = {k: sum(1 for c in base.cls if c == k) for k in sorted(set(base.cls))}
    print(f"[gen45] {base.th.name}: R={base.R} lanes={len(base.lane_idx)} H={base.H} "
          f"sites={n_cls} range_scale={RANGE_SCALE} cr={CONCEAL_REACH} "
          f"[{time.time()-t0:.0f}s build]", flush=True)
    selfcheck(base)

    cells, pinned = [], None
    for w in ([args.w] if args.w else [2, 3]):
        for seed in HUNT_FIELDS:
            cells.append(cell(base, seed, w))
            json.dump(cells, open(args.out, "w"), indent=1)
        if gates(cells, w):
            pinned = w
            break
    json.dump({"cells": cells, "pinned_w": pinned,
               "config": {"range_scale": RANGE_SCALE, "conceal_reach": CONCEAL_REACH,
                          "q": DOC32, "tau": TAU, "n_sites": 200, "theatre": "kgd_gvardeysk"}},
              open(args.out, "w"), indent=1)
    print(f"[done] pinned_w={pinned} -> {args.out} [{time.time()-t0:.0f}s total]", flush=True)
    if pinned is None:
        print("[FAIL BRANCH] no operating point passed inside the pre-declared freedom at "
              "DOC32/tau=0.10; q/tau exploration is a separate deliberate step per the ledger",
              flush=True)


if __name__ == "__main__":
    main()
