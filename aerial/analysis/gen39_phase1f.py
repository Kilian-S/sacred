#!/usr/bin/env python3
"""gen39 Phase 1f: SAMPLE EFFICIENCY IN A SPACE TOO LARGE TO ENUMERATE (Kilian, 2026-07-27).

Phase 1e closed the interface questions: briefing was not the problem, grounding was and is fixed
(91%), and on a 165-option menu the residual gap is combinatorial search - where an LLM cannot
earn its place, because brute force is free there.

This phase asks the question where it CAN. The real emplacement space on narva is 200 candidate
sites = 1.3 million three-team combinations; no planner enumerates that, and neither can we. So
every method gets the SAME BUDGET of exact evaluations and we race them:

  llm     the model proposes candidate forces from a map digest + the running leaderboard of what
          it has tried and scored (its own history only: no other arm's results);
  random  triples drawn uniformly from the 200 sites;
  local   the standing algorithmic baseline: greedy max-min seed + steepest-descent single-site
          swaps (`ConcealBase.best_laydown`'s own strategy), restarted when it converges;
  greedy  top-K by individual site threat, re-evaluated (the naive planner's shortcut).

Every arm is scored by the SAME exact evaluator on the SAME fields, and the budget is counted in
EXACT EVALUATIONS, which is the currency that matters operationally (each one is a full mission
solve). The LLM's model calls are reported separately and are NOT charged to the budget: the
claim under test is "does reading the map and reasoning in words find good forces in fewer
SIMULATIONS", not "is it cheap in tokens".

  BARS (fixed before any call):
    S1 at the shared budget, the LLM arm's BEST force beats the random arm's best;
    S2 it also beats the local-search arm's best (the algorithmic incumbent);
    S3 it reaches >= 0.0215, the tuned-doctrine curriculum's level (operational relevance).
  Reported: the whole budget curve (best-so-far vs evaluations), per model, plus the tuned
  doctrine and the 165-slot restricted ceiling as context lines.

    PYTHONPATH=. python analysis/gen39_phase1f.py --budget 96 --rounds 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from analysis.gen39_compose import BASE_URL, FIELDS, K, KEY, MODELS, OUTDIR, g33, narva_base
from src.envs.aerial_conceal import ConcealDyn, resample_field
from src.redforce import serialise_theatre

W, TAU, T_MISSION = 2, 0.10, 40
DOC = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3)      # doctrine held FIXED across arms: positions only
TUNED_BAR, SLOT_CEIL = 0.0215, 0.0278
OUT = Path("models/runs/gen39_phase1f.json")

SCHEMA = {"type": "object", "properties": {"forces": {"type": "array", "items": {
    "type": "object", "properties": {
        "sites": {"type": "array", "items": {"type": "integer"}, "minItems": K, "maxItems": K},
        "why": {"type": "string"}},
    "required": ["sites", "why"]}}}, "required": ["forces"]}


# --- the shared exact evaluator ------------------------------------------------------------------
_CTX: dict = {}


def _init():
    _CTX["base"], _, _ = narva_base()
    _CTX["pp"] = {f: _CTX["base"].lethality(resample_field(_CTX["base"].coords, f),
                                            hidden_leth=1.0) for f in FIELDS}


def _eval(spec):
    sites, field = spec
    base = _CTX["base"]
    g = ConcealDyn(base, _CTX["pp"][field], np.asarray(sites, int), w=W, tau=TAU, **DOC)
    return tuple(sites), field, float(g.episodic(T=T_MISSION))


def evaluate(pool, triples):
    """Exact irreducible threat, median over the fields. Costs len(triples) budget units."""
    specs = [(t, f) for t in triples for f in FIELDS]
    agg: dict = {}
    for t, f, v in pool.imap_unordered(_eval, specs, chunksize=3):
        agg.setdefault(t, []).append(v)
    return {t: float(np.median(v)) for t, v in agg.items()}


# --- the map digest the LLM reads ----------------------------------------------------------------

def map_digest(base, pp, n_show=200):
    """A factual site list: index, ground, reach, lethality, whether it reveals, position along
    and across the corridor, and how many routes it threatens. Descriptive only."""
    v = base.th.target - base.th.base
    u = v / (np.linalg.norm(v) + 1e-9)
    nrm = np.array([-u[1], u[0]])
    along = ((base.coords - base.th.base) @ u) / (float(v @ u) + 1e-9)
    across = (base.coords - base.th.base) @ nrm
    thr = base.threat_rank(pp)
    lines = ["SITE CATALOGUE (every emplacement available on this map):",
             " idx | ground | reach | leth | reveals | along-corridor | across | routes it can hit"]
    for i in np.argsort(-thr)[:n_show]:
        c = base.cls[i]
        g = ConcealDyn(base, pp, np.array([i]), w=W, tau=TAU)
        d = g.dmg_j[0]
        nr = int((d > 0.02 * d.max()).sum())
        lines.append(f" {int(i):3d} | {c:6s} | {base.rr[i]:4.1f} | {base.pp_base[i]:.2f} | "
                     f"{'yes' if not base.concealed[i] else ' no':>7s} | {along[i]:14.2f} | "
                     f"{across[i]:6.1f} | {nr:3d}")
    return "\n".join(lines)


def llm_prompt(digest, history, budget_left):
    h = ""
    if history:
        top = sorted(history, key=lambda x: -x[1])[:12]
        h = ("\n\nWHAT YOU HAVE TRIED SO FAR (your own proposals, exactly scored; higher is "
             "better):\n" + "\n".join(f"  sites {list(t)} -> {v:.4f}" for t, v in top)
             + f"\n  (you have proposed {len(history)} forces; your best is "
               f"{max(v for _, v in history):.4f})")
    return f"""{digest}{h}

You are choosing where to post {K} air-defence teams among these sites. Your force is scored by
how much damage it does to a drone flight that KNOWS where all your teams are and picks its
safest route: your force is only as strong as the flight's cheapest remaining option. Teams that
cluster together, or that all cover the same routes, leave the flight a free lane.

Propose {budget_left} DIFFERENT candidate forces (each a list of {K} site indices from the
catalogue). Vary them: this is a search, and you are choosing where to spend expensive
simulations. Give one short reason each. Emit ONLY the structured JSON."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=96, help="exact evaluations per arm")
    ap.add_argument("--rounds", type=int, default=8, help="LLM proposal rounds")
    a = ap.parse_args()
    import multiprocessing as mp_
    base, terr, sc = narva_base()
    pp0 = base.lethality(resample_field(base.coords, FIELDS[0]), hidden_leth=1.0)
    H = base.H
    digest = map_digest(base, pp0)
    (OUTDIR / "digest_phase1f.txt").write_text(digest)
    per_round = max(1, a.budget // a.rounds)
    rng = np.random.default_rng(0)
    print(f"[1f] {H} sites -> {H*(H-1)*(H-2)//6:,} possible forces; budget {a.budget} exact "
          f"evaluations per arm; {len(FIELDS)} fields per evaluation\n")
    curves: dict = {}

    with mp_.get_context("spawn").Pool(9, initializer=_init) as P:
        # --- random ------------------------------------------------------------------------
        tri = [tuple(sorted(int(x) for x in rng.choice(H, K, replace=False)))
               for _ in range(a.budget)]
        sc_r = evaluate(P, tri)
        curves["random"] = [max(sc_r[t] for t in tri[:i + 1]) for i in range(len(tri))]
        print(f"  random   best {max(sc_r.values()):.4f}", flush=True)

        # --- greedy top-K by individual threat (cheap incumbent) ---------------------------
        thr = base.threat_rank(pp0)
        order = np.argsort(-thr)
        tri_g = [tuple(sorted(int(x) for x in order[i:i + K])) for i in range(a.budget)]
        sc_g = evaluate(P, list(dict.fromkeys(tri_g)))
        best_g, cg = -1.0, []
        for t in tri_g:
            best_g = max(best_g, sc_g.get(t, 0.0)); cg.append(best_g)
        curves["greedy"] = cg
        print(f"  greedy   best {max(sc_g.values()):.4f}", flush=True)

        # --- local search: greedy max-min seed + steepest-descent swaps --------------------
        used, cl, best_l = 0, [], -1.0
        cur = tuple(sorted(int(x) for x in rng.choice(H, K, replace=False)))
        val = evaluate(P, [cur])[cur]; used += 1; best_l = val; cl.append(best_l)
        while used < a.budget:
            cands = []
            for s in range(K):
                for _ in range(6):
                    c = list(cur); c[s] = int(rng.integers(H))
                    if len(set(c)) == K:
                        cands.append(tuple(sorted(c)))
            cands = list(dict.fromkeys(cands))[:min(18, a.budget - used)]
            if not cands:
                break
            got = evaluate(P, cands); used += len(cands)
            bt = max(got, key=got.get)
            if got[bt] > val:
                cur, val = bt, got[bt]
            else:
                cur = tuple(sorted(int(x) for x in rng.choice(H, K, replace=False)))
                val = evaluate(P, [cur])[cur]; used += 1
            best_l = max(best_l, val)
            cl += [best_l] * min(len(cands), a.budget - len(cl))
        curves["local"] = cl[:a.budget]
        print(f"  local    best {best_l:.4f} ({used} evals)", flush=True)

        # --- the LLM arms ------------------------------------------------------------------
        for model in MODELS:
            hist, curve, best = [], [], -1.0
            t0 = time.time()
            for rnd in range(a.rounds):
                left = min(per_round, a.budget - len(hist))
                if left <= 0:
                    break
                prompt = llm_prompt(digest, hist, left)
                tri_l = []
                for _ in range(2):
                    try:
                        txt, _m = g33.call_openai(BASE_URL, KEY, model,
                                                  "You are an air-defence planner running a search.",
                                                  prompt, schema=SCHEMA, max_tokens=3000,
                                                  temperature=0.9, timeout=900)
                        obj = g33._extract_json(txt)
                        for f in obj.get("forces", []):
                            s = [int(x) for x in f.get("sites", []) if 0 <= int(x) < H]
                            if len(set(s)) == K:
                                tri_l.append(tuple(sorted(set(s))))
                        if tri_l:
                            break
                    except Exception:                                  # noqa: BLE001
                        continue
                tri_l = [t for t in dict.fromkeys(tri_l) if t not in dict(hist)][:left]
                if not tri_l:                       # a dud round still costs nothing but a call
                    continue
                got = evaluate(P, tri_l)
                for t in tri_l:
                    hist.append((t, got[t]))
                    best = max(best, got[t])
                    curve.append(best)
                print(f"  {model[:12]:12s} round {rnd}: proposed {len(tri_l)}, best so far "
                      f"{best:.4f} ({len(hist)}/{a.budget} evals, {(time.time()-t0)/60:.1f} min)",
                      flush=True)
            curves[f"llm:{model}"] = curve

    n = min(len(v) for v in curves.values() if v)
    print(f"\n{'=' * 88}\nPHASE 1F: best force found vs exact-evaluation budget "
          f"(tuned doctrine {TUNED_BAR:.4f}; 165-slot ceiling {SLOT_CEIL:.4f})\n{'=' * 88}")
    marks = [m for m in (8, 16, 32, 48, 64, 96) if m <= n]
    print(f'{"arm":22s} ' + " ".join(f'{"@" + str(m):>9s}' for m in marks) + f'{"  final":>10s}')
    for k, v in curves.items():
        print(f'{k:22s} ' + " ".join(f'{v[m-1]:9.4f}' for m in marks) + f'{v[n-1]:10.4f}')
    OUT.write_text(json.dumps({k: v for k, v in curves.items()}, indent=1))
    lb = max(curves.get(f"llm:{m}", [0])[-1] for m in MODELS)
    print(f'\nS1 LLM best > random best: '
          f'{"PASS" if lb > curves["random"][n-1] else "FAIL"} '
          f'({lb:.4f} vs {curves["random"][n-1]:.4f})')
    print(f'S2 LLM best > local-search best: '
          f'{"PASS" if lb > curves["local"][-1] else "FAIL"} ({lb:.4f} vs {curves["local"][-1]:.4f})')
    print(f'S3 LLM best >= tuned doctrine {TUNED_BAR}: '
          f'{"PASS" if lb >= TUNED_BAR else "FAIL"} ({lb:.4f})')
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
