#!/usr/bin/env python3
"""gen36 DESIGN PROBE (oracle-only, no training): distillation labels for the gen29 reopening.

gen29's trained half failed both tiers (density starvation + FP instability; blinded ~
sighted). The ledger pre-committed two untaken options: the DISTILLATION control (train the
same policy class to imitate the exact coordinated joint mixture per instance) and ONE dense
per-stream-credit self-play re-aim. This probe verifies the distillation ingredient:

  1. ANCHOR REPRODUCTION (the gen30 dogma): rebuild screened cells from
     models/runs/gen29_screen.json byte-identically (same builders) and reproduce the stored
     eq / cap scalars BEFORE any new number is read.
  2. LABELS: extract the joint optimal MIXTURE dstar (= _row_minimiser(M)[1], discarded by the
     screen) per cell; report support size, entropy fraction, LP wall time.
  3. FACTORISATION: check dstar reshapes to the per-stream `shape` and yields proper
     sequential conditionals P(r0), P(r1|r0), P(r2|r0,r1) - the exact target objects for the
     trainer's stream-sequential policy head.

Run (from the gen29 worktree): OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
    /Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python scratch/gen36_label_probe.py
Writes models/runs/gen36_label_probe.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from scratch.b4_joint_napkin_probe import best_m_pairings
from scratch.b4_multiod_probe import build_graph
from scratch.b4_widen_probe import joint_payoff, survival_mats
from src.baselines.interdiction_oracle import build_route_set
from src.baselines.multiconvoy_oracle import _row_minimiser

KX = 8
TOL = 2e-3   # LP-degeneracy wobble allowance on eq (SYSTEM.md dogma d)


def probe_cell(G, spec, tag):
    t0 = time.time()
    s, targets = spec["s"], spec["targets"]
    rsets = [build_route_set(G, s, t, KX, "w") for t in targets]
    S_list, cand = survival_mats(G, rsets)
    isets = [(e,) for e in range(len(cand))]
    M, shape = joint_payoff(S_list, isets)
    v_joint, dstar = _row_minimiser(M)
    v_cap = min(best_m_pairings(M).values())
    lp_secs = time.time() - t0
    eq_dev = abs(v_joint - spec["eq"]) / spec["eq"]
    cap_dev = abs(v_cap - spec["cap"]) / spec["cap"]
    support = int((dstar > 1e-6).sum())
    p = dstar[dstar > 1e-12]
    entf = float(-(p * np.log(p)).sum() / np.log(len(dstar)))
    joint = dstar.reshape(shape)
    p0 = joint.sum(axis=(1, 2))
    cond_ok = abs(p0.sum() - 1.0) < 1e-9
    n_cond1 = int((joint.sum(axis=2) > 1e-9).sum())
    row = dict(tag=tag, s=s, targets=targets, R=list(shape), n_joint=int(M.shape[0]),
               E=len(cand), eq_stored=spec["eq"], eq_recomputed=float(v_joint),
               eq_rel_dev=float(eq_dev), cap_stored=spec["cap"], cap_recomputed=float(v_cap),
               cap_rel_dev=float(cap_dev), anchor_ok=bool(eq_dev < TOL and cap_dev < TOL),
               label_support=support, label_ent_frac=entf, factorisation_ok=bool(cond_ok),
               nonzero_stream1_conditionals=n_cond1, lp_secs=round(lp_secs, 2))
    print(f"{tag}: eq {v_joint:.4f} (stored {spec['eq']:.4f}, dev {eq_dev:.1e}) "
          f"cap {v_cap:.4f} (dev {cap_dev:.1e}) ANCHOR={'OK' if row['anchor_ok'] else 'FAIL'} | "
          f"label support {support}/{M.shape[0]} entf {entf:.3f} | {lp_secs:.1f}s", flush=True)
    return row


def main():
    screen = json.load(open("models/runs/gen29_screen.json"))
    G = build_graph()
    rows = []
    rows.append(probe_cell(G, screen["headline"], "HEADLINE"))
    for i, spec in enumerate(screen["pool"][:3]):
        rows.append(probe_cell(G, spec, f"pool{i}"))
    for i, spec in enumerate(screen["held_out"][:2]):
        rows.append(probe_cell(G, spec, f"held_out{i}"))
    n_ok = sum(r["anchor_ok"] for r in rows)
    total_lp = sum(r["lp_secs"] for r in rows)
    print(f"\nanchors OK {n_ok}/{len(rows)}; total LP time {total_lp:.1f}s for {len(rows)} "
          f"cells -> full 26-cell label pass ~{total_lp / len(rows) * 26:.0f}s")
    with open("models/runs/gen36_label_probe.json", "w") as f:
        json.dump(dict(rows=rows, anchors_ok=f"{n_ok}/{len(rows)}"), f, indent=1)
    print("wrote models/runs/gen36_label_probe.json")


if __name__ == "__main__":
    main()
