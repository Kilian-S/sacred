#!/usr/bin/env python3
"""gen39 Phase 1c: can the LLM compose a force that is dangerous even to a defender that KNOWS
where it is? (Kilian, 2026-07-27. Oracle + model calls only, no training.)

Phase 1 measured the mechanism: curriculum value tracks the opponent's IRREDUCIBLE threat (damage
against a defender with perfect knowledge), and the LLM's forces are concealment gambits - strong
against a searching defender, near-harmless against a knowing one (median 0.0008-0.0019 vs the
heuristic curriculum's 0.0215). Three arms test whether that is a capability ceiling or a
BRIEFING failure. All three keep the placer, terrain, budget and scorer of step 2, so only the
composition process differs.

  robust   the brief gains ONE constraint: the force must stay dangerous against a defender that
           knows exactly where every team is. Nothing else changes.
  iter     three rounds of feedback: compose -> score exactly -> tell the model its two yardstick
           numbers and which routes escaped -> revise. Matches the tuning budget the gen32
           doctrine received over two generations (the fair-fight argument).
  curated  best-of-N over the banked 61-force population, ranked by irreducible threat: the
           "LLM proposes, oracle selects" pattern (the gen38 shape). FREE, no new calls.

BAR (fixed before the calls, per house rules): an arm justifies a Phase-2 training run only if
its forces reach ~0.0215 median irreducible threat, the heuristic curriculum's level. Anything
below that cannot overturn step 3 and is reported as a measured ceiling.

    PYTHONPATH=. python scratch/gen39_phase1c.py --robust --iter --curated
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from scratch.gen39_compose import (BASE_URL, FIELDS, K, KEY, MODELS, OUTDIR, doctrines_of, g33,
                                   narva_base, place, score_force)
from src.envs.aerial_conceal import resample_field
from src.redforce import force_schema, serialise_theatre

N_ROBUST, ITER_ROUNDS, N_ITER = 16, 3, 6
HEUR_BAR = 0.0215                      # the heuristic curriculum's irreducible threat

ROBUST_CLAUSE = """

CRITICAL CONSTRAINT ON THIS FORCE. The enemy flight is not a naive searcher: it LOCATES a team
that engages from revealing ground and REMEMBERS it for the rest of the mission, and it is a
trained planner that will route around everything it knows about. A force that is only dangerous
while it stays unfound is worth little. Design so that your force still inflicts losses against a
planner that knows EXACTLY where all {k} of your teams are: it must be able to hold or deny the
ground the flight has to cross, not merely ambush from hiding. State in each rationale why that
team still hurts once it is known."""


def brief(base_th, scale, extra="", terrain=None):
    system, user = serialise_theatre(base_th, phase="coordinated", K=K, range_scale=scale,
                                     terrain=terrain)
    return system, user + extra


def gen(model, system, user, schema, temperature=0.8):
    txt, mode = g33.call_openai(BASE_URL, KEY, model, system, user, schema=schema,
                                max_tokens=3000, temperature=temperature, timeout=900)
    obj = g33._extract_json(txt)
    return obj, g33.validate_force(obj), mode


# --- exact scoring (shared) --------------------------------------------------------------------
_CTX: dict = {}


def _init():
    _CTX["base"], _CTX["t"], _CTX["sc"] = narva_base()


def _task(spec):
    key, force, field = spec
    base = _CTX["base"]
    pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    o, b, ob, cov = score_force(base, pp, place(force, base, pp), doctrines_of(force))
    return key, (o, ob, cov)


def score_many(forces: dict, workers=9):
    """forces: key -> force dict. Returns key -> (median irreducible, median vs-observing, cover)."""
    import multiprocessing as mp_
    specs = [(k, f, fld) for k, f in forces.items() for fld in FIELDS]
    agg: dict = {}
    with mp_.get_context("spawn").Pool(workers, initializer=_init) as P:
        for k, v in P.imap_unordered(_task, specs, chunksize=4):
            agg.setdefault(k, []).append(v)
    return {k: tuple(np.median(np.array(v), axis=0)) for k, v in agg.items()}


def cover_share(force):
    t = [a["emplacement_zone"]["terrain"] for a in force["agents"]]
    return sum(x in ("forest", "urban") for x in t) / len(t)


def run_robust(base, th, sc):
    system, user = brief(th, sc, ROBUST_CLAUSE.format(k=K), terrain=base.terrain)
    (OUTDIR / "brief_robust.txt").write_text(system + "\n\n---\n\n" + user)
    schema = force_schema(base.terrain)
    recs = []

    def one(m, j):
        try:
            obj, errs, mode = gen(m, system, user, schema)
            return {"model": m, "j": j, "force": obj, "errors": errs, "mode": mode}
        except Exception as e:                                        # noqa: BLE001
            return {"model": m, "j": j, "force": None, "errors": [str(e)], "mode": "error"}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, m, j) for m in MODELS for j in range(N_ROBUST // 2)]
        for f in as_completed(futs):
            r = f.result()
            recs.append(r)
            print(f"  [robust] {r['model']} #{r['j']}: "
                  f"{'ok' if r['force'] and not r['errors'] else 'BAD'}", flush=True)
    (OUTDIR / "forces_robust.json").write_text(json.dumps(recs, indent=1))
    return recs


def run_iter(base, th, sc):
    """Three rounds of compose -> exact score -> feedback -> revise, per model."""
    schema = force_schema(base.terrain)
    system, user0 = brief(th, sc, terrain=base.terrain)
    history = []
    live = {}
    for m in MODELS:
        for j in range(N_ITER // 2):
            live[f"{m}#{j}"] = dict(model=m, j=j, msgs=user0, force=None)
    for rnd in range(ITER_ROUNDS):
        def one(key, st):
            try:
                obj, errs, _ = gen(st["model"], system, st["msgs"], schema)
                return key, obj, errs
            except Exception as e:                                    # noqa: BLE001
                return key, None, [str(e)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for key, obj, errs in [f.result() for f in
                                   [ex.submit(one, k, v) for k, v in live.items()]]:
                if obj and not errs and len(obj.get("agents", [])) == K:
                    live[key]["force"] = obj
        scored = score_many({k: v["force"] for k, v in live.items() if v["force"]})
        for key, (irr, obs, cov) in scored.items():
            st = live[key]
            history.append(dict(round=rnd, key=key, irreducible=float(irr), observing=float(obs),
                                cover=cover_share(st["force"]), force=st["force"]))
            st["msgs"] = (user0 + f"""

YOUR PREVIOUS FORCE AND HOW IT SCORED (exact simulation, {len(FIELDS)} threat fields):
{json.dumps(st['force'], indent=1)[:1500]}

  - damage against a defender that must SEARCH for you: {obs:.4f}
  - damage against a defender that KNOWS where all your teams are: {irr:.4f}   <-- the number to raise
  - your force sat {cover_share(st['force']):.0%} in cover

A reference force built by our own planner reaches {HEUR_BAR:.4f} on the second number. Yours is
{irr / HEUR_BAR:.0%} of that. Revise the force to raise it, keeping the same {K} teams. Emit ONLY
the new structured force.""")
        med = np.median([h["irreducible"] for h in history if h["round"] == rnd])
        print(f"  [iter] round {rnd}: median irreducible {med:.5f} "
              f"({med / HEUR_BAR:.0%} of the heuristic bar)", flush=True)
    (OUTDIR / "forces_iter.json").write_text(json.dumps(history, indent=1))
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--iter", action="store_true")
    ap.add_argument("--curated", action="store_true")
    a = ap.parse_args()
    base, t, sc = narva_base()
    base.terrain = t
    th, scale = base.th, sc * 0.7
    t0 = time.time()
    out = {}

    if a.robust:
        recs = run_robust(base, th, scale)
        good = {f"{r['model']}#{r['j']}": r["force"] for r in recs
                if r["force"] and not r["errors"] and len(r["force"]["agents"]) == K}
        sc_r = score_many(good)
        out["robust"] = [dict(key=k, irreducible=float(v[0]), observing=float(v[1]),
                              cover=cover_share(good[k])) for k, v in sc_r.items()]
    if a.iter:
        out["iter"] = run_iter(base, th, scale)
    if a.curated:
        rows = json.loads(Path("models/runs/gen39_phase1b_scores.json").read_text())
        top = sorted(rows, key=lambda r: -r["opt"])[:12]
        out["curated"] = [dict(key=f'{r["model"]}#{r["j"]}', irreducible=r["opt"],
                               observing=r["obs"], cover=r["hidden"]) for r in top]

    print(f"\n{'=' * 78}\nPHASE 1C RESULT (bar = {HEUR_BAR:.4f} irreducible threat)\n{'=' * 78}")
    print(f'{"arm":28s} {"n":>3s} {"median irreducible":>19s} {"best":>9s} {"% of bar":>9s} {"cover":>7s}')
    base_rows = json.loads(Path("models/runs/gen39_phase1b_scores.json").read_text())
    print(f'{"step-2/1b population":28s} {len(base_rows):3d} '
          f'{np.median([r["opt"] for r in base_rows]):19.5f} '
          f'{max(r["opt"] for r in base_rows):9.4f} '
          f'{np.median([r["opt"] for r in base_rows]) / HEUR_BAR:8.0%} '
          f'{np.mean([r["hidden"] for r in base_rows]):6.0%}')
    for k, v in out.items():
        if k == "iter":
            for rnd in range(ITER_ROUNDS):
                s = [h for h in v if h["round"] == rnd]
                if s:
                    print(f'{"iter round " + str(rnd):28s} {len(s):3d} '
                          f'{np.median([x["irreducible"] for x in s]):19.5f} '
                          f'{max(x["irreducible"] for x in s):9.4f} '
                          f'{np.median([x["irreducible"] for x in s]) / HEUR_BAR:8.0%} '
                          f'{np.mean([x["cover"] for x in s]):6.0%}')
        else:
            print(f'{k:28s} {len(v):3d} {np.median([x["irreducible"] for x in v]):19.5f} '
                  f'{max(x["irreducible"] for x in v):9.4f} '
                  f'{np.median([x["irreducible"] for x in v]) / HEUR_BAR:8.0%} '
                  f'{np.mean([x["cover"] for x in v]):6.0%}')
    Path("models/runs/gen39_phase1c.json").write_text(json.dumps(out, indent=1))
    print(f"\n[{(time.time() - t0) / 60:.1f} min] written models/runs/gen39_phase1c.json")


if __name__ == "__main__":
    main()
