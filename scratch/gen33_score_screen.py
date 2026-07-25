"""gen33_llm_adversary SCORER CONFIRMATION SCREEN (oracle-only, no LLM scores touched).

Confirms the pinned enemy semantics (src/redforce_score.py) BEFORE any live force is scored,
and anchors the metric-1 baseline ladder. Pre-written decision rules (pinned here, before any
number is seen):

  CONSISTENCY  flat-prior single-agent (0.7,0.3,0) tau .10 w2 must reproduce the gen32
               DynTheatre history_opt (|rel diff| < 1e-6 on 3 field seeds).
  SIGMA RULE   the concentration scale sigma0 (km at kgd scale, scaled per theatre like ranges)
               is chosen on kgd seed 5100 from the grid (2, 4, 8) as the LARGEST value with
               (S2) placement sensitivity: (oracle - random_mean)/random_mean >= 0.05
                    on BOTH phases (K=1, K=3),
               subject to
               (S1) non-collapse: heuristic value >= 0.02 and >= 0.4x its flat-prior value
                    (the concentration must not gut the doctrine contest, the Phase 0 B lesson).
               If no grid value passes both, the semantics FAIL the screen (a writable result).
  ANCHORS      per (theatre, phase): random floor = mean+/-sd over 16 draws x 3 seeds
               (5100-5102); heuristic = mean over 3 seeds; oracle = search on seed 5100
               (disclosed budget: 48 random + per-agent site ascent + doctrine sweep),
               winner re-scored on the 3 seeds. These are the metric-1 ladder rows, written to
               the ledger BEFORE any LLM force is scored.
"""
import json
import time

import numpy as np

from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre
from src.redforce_score import (ScoreBase, force_value, heuristic_force, oracle_force,
                                random_force)

THEATRES = {
    "kgd": "data/maps/theatre_kgd_gvardeysk_vec.json",
    "ukraine": "data/maps/theatre_ukraine_vec.json",
    "narva": "data/maps/theatre_narva_vec.json",
}
SEEDS = (5100, 5101, 5102)
LAT_REF = lateral_width(load_vec_theatre(THEATRES["kgd"]))


def consistency(base):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("g32", "scratch/gen32_theatre_hunt.py")
    g32 = importlib.util.module_from_spec(spec)
    sys.modules["g32"] = g32
    spec.loader.exec_module(g32)
    tb = g32.TheatreBase()
    ok = True
    for seed in SEEDS:
        ref = g32.DynTheatre(tb, g32.resample_field(tb.coords, seed), 2, 0.10, 0.7, 0.3)
        v = force_value(base.field(seed), [0], [(0.7, 0.3, 0.0, 0.10, 2)], None)
        r = ref.history_opt()
        ok &= abs(v - r) / max(r, 1e-9) < 1e-6
        print(f"  consistency seed{seed}: ours={v:.6f} gen32={r:.6f} "
              f"{'OK' if abs(v - r) / max(r, 1e-9) < 1e-6 else 'MISMATCH'}")
    return ok


def sigma_row(base, sigma, K, n_rand=8):
    fc = base.field(5100)
    rng = np.random.default_rng(100)
    rand = np.mean([force_value(fc, *random_force(base, K, rng), sigma) for _ in range(n_rand)])
    heur = force_value(fc, *heuristic_force(base, K), sigma)
    _, _, orac = oracle_force(base, K, sigma, n_random=24, n_ascent=8)
    return rand, heur, orac


def pick_sigma(base):
    flat = {K: force_value(base.field(5100), *heuristic_force(base, K), None) for K in (1, 3)}
    rows = {}
    for sigma in (2.0, 4.0, 8.0):
        s = sigma * base.scale
        for K in (1, 3):
            rand, heur, orac = sigma_row(base, s, K)
            sens = (orac - rand) / max(rand, 1e-9)
            rows[(sigma, K)] = dict(rand=float(rand), heur=float(heur), orac=float(orac),
                                    sens=float(sens),
                                    s1=bool(heur >= 0.02 and heur >= 0.4 * flat[K]),
                                    s2=bool(sens >= 0.05))
            print(f"  sigma{sigma:>4} K{K}: rand={rand:.4f} heur={heur:.4f} orac={orac:.4f} "
                  f"sens={sens:+.2f} flat_heur={flat[K]:.4f} "
                  f"S1={'y' if rows[(sigma, K)]['s1'] else 'N'} "
                  f"S2={'y' if rows[(sigma, K)]['s2'] else 'N'}", flush=True)
    chosen = None
    for sigma in (8.0, 4.0, 2.0):                       # largest first
        if all(rows[(sigma, K)]["s1"] and rows[(sigma, K)]["s2"] for K in (1, 3)):
            chosen = sigma
            break
    return chosen, rows


def anchors(base, name, sigma0):
    s = sigma0 * base.scale
    out = {}
    for K, phase in ((1, "single"), (3, "coordinated")):
        t0 = time.time()
        rng = np.random.default_rng(200)
        rand = [np.mean([force_value(base.field(sd), *random_force(base, K, rng), s)
                         for sd in SEEDS]) for _ in range(16)]
        heur = np.mean([force_value(base.field(sd), *heuristic_force(base, K), s)
                        for sd in SEEDS])
        osites, odoc, _ = oracle_force(base, K, s)
        orac = np.mean([force_value(base.field(sd), osites, odoc, s) for sd in SEEDS])
        out[phase] = dict(random_mean=float(np.mean(rand)), random_sd=float(np.std(rand)),
                          heuristic=float(heur), oracle=float(orac),
                          oracle_sites=[int(x) for x in osites],
                          oracle_doctrine=[list(map(float, d)) for d in odoc],
                          eq_static=float(np.mean([base.field(sd).eq_static for sd in SEEDS])))
        print(f"  {name} {phase} K={K}: random={np.mean(rand):.4f}+/-{np.std(rand):.4f} "
              f"heuristic={heur:.4f} oracle={orac:.4f} "
              f"eq_static={out[phase]['eq_static']:.3f} [{time.time()-t0:.0f}s]", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    bases = {n: ScoreBase(p, lat_ref=None if n == "kgd" else LAT_REF)
             for n, p in THEATRES.items()}
    print("=== CONSISTENCY (flat prior reproduces gen32 DynTheatre) ===")
    ok = consistency(bases["kgd"])
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    print("\n=== SIGMA GRID (kgd seed 5100; rule pinned in the header) ===")
    sigma0, grid = pick_sigma(bases["kgd"])
    print(f"  -> sigma0 = {sigma0}")
    if sigma0 is None:
        raise SystemExit("SEMANTICS FAIL THE SCREEN (no sigma passes S1+S2); writable result")
    print("\n=== ANCHOR LADDER (all theatres, both phases, 3 seeds) ===")
    res = {"sigma0": sigma0, "consistency": ok,
           "grid": {f"{s}_{K}": v for (s, K), v in grid.items()}, "anchors": {}}
    for n, b in bases.items():
        res["anchors"][n] = anchors(b, n, sigma0)
    json.dump(res, open("models/runs/gen33_score_screen.json", "w"), indent=1)
    print(f"\n[written] models/runs/gen33_score_screen.json [{time.time()-t0:.0f}s total]")
