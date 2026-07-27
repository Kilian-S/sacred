#!/usr/bin/env python3
"""gen39 step 5 PREP: build the four STRONG curricula and the strong test set (oracle + model
calls only, no training). Pinned by the step-5 pre-registration in the ledger.

Per training field (1000-1015) and per test field (6100-6105), each arm runs its OWN search at a
MATCHED budget of 16 exact evaluations and keeps its top-3 laydowns:

  llm16     llama-3.3-70b proposes from the site catalogue + its running leaderboard
  local16   greedy seed + steepest-descent single-site swaps
  random16  uniform triples
  tuned     the step-3 control unchanged (`choose_force` + gen32 doctrine, 3 archetypes)

Doctrine is FROZEN to gen32 in every arm (free-gate Part A: LLM-written doctrine scores 0.53-0.75x
on the same positions). Output: models/runs/gen39_step5/curricula.json, consumed by the trainer.

    PYTHONPATH=. python scratch/gen39_step5_prep.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from scratch.gen39_compose import BASE_URL, K, KEY, g33
from scratch.gen39_phase1f import SCHEMA, llm_prompt, map_digest
from scripts.train_gen39_conceal import DOC32, TEST_FIELDS, TRAIN_FIELDS, W, TAU, narva_base
from src.envs.aerial_conceal import ConcealDyn, choose_force, resample_field

BUDGET, KEEP, T_MISSION = 16, 3, 40
OUT = Path("models/runs/gen39_step5/curricula.json")

_CTX: dict = {}


def _init():
    _CTX["base"] = narva_base()


def _eval(spec):
    sites, field = spec
    base = _CTX["base"]
    pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    g = ConcealDyn(base, pp, np.asarray(sites, int), w=W, tau=TAU, **DOC32)
    return tuple(sites), float(g.episodic(T=T_MISSION))


def score(pool, triples, field):
    return dict(pool.imap_unordered(_eval, [(t, field) for t in triples], chunksize=2))


def llm16(base, digest, pool, field):
    hist = []
    for _ in range(4):
        left = BUDGET - len(hist)
        if left <= 0:
            break
        tri = []
        for _try in range(2):
            try:
                txt, _m = g33.call_openai(BASE_URL, KEY, "llama-3.3-70b",
                                          "You are an air-defence planner running a search.",
                                          llm_prompt(digest, hist, min(4, left)), schema=SCHEMA,
                                          max_tokens=2500, temperature=0.9, timeout=900)
                for f in g33._extract_json(txt).get("forces", []):
                    s = [int(x) for x in f.get("sites", []) if 0 <= int(x) < base.H]
                    if len(set(s)) == K:
                        tri.append(tuple(sorted(set(s))))
                if tri:
                    break
            except Exception:                                          # noqa: BLE001
                continue
        tri = [t for t in dict.fromkeys(tri) if t not in dict(hist)][:left]
        if not tri:
            break
        got = score(pool, tri, field)
        hist += list(got.items())
    return hist


def local16(base, pool, field, rng):
    used, hist = 0, []
    cur = tuple(sorted(int(x) for x in rng.choice(base.H, K, replace=False)))
    v = score(pool, [cur], field)[cur]; used += 1; hist.append((cur, v))
    while used < BUDGET:
        cands = []
        for s in range(K):
            for _ in range(5):
                c = list(cur); c[s] = int(rng.integers(base.H))
                if len(set(c)) == K:
                    cands.append(tuple(sorted(c)))
        cands = list(dict.fromkeys(cands))[:min(5, BUDGET - used)]
        if not cands:
            break
        got = score(pool, cands, field); used += len(cands)
        hist += list(got.items())
        b = max(got, key=got.get)
        if got[b] > v:
            cur, v = b, got[b]
        else:
            cur = tuple(sorted(int(x) for x in rng.choice(base.H, K, replace=False)))
            v = score(pool, [cur], field)[cur]; used += 1; hist.append((cur, v))
    return hist


def random16(base, pool, field, rng):
    tri = [tuple(sorted(int(x) for x in rng.choice(base.H, K, replace=False)))
           for _ in range(BUDGET)]
    return list(score(pool, list(dict.fromkeys(tri)), field).items())


def main():
    import multiprocessing as mp_
    base = narva_base()
    pp0 = base.lethality(resample_field(base.coords, 1000), hidden_leth=1.0)
    digest = map_digest(base, pp0)
    fields = list(TRAIN_FIELDS) + list(TEST_FIELDS)
    out = {"llm16": {}, "local16": {}, "random16": {}, "tuned": {}}
    t0 = time.time()
    with mp_.get_context("spawn").Pool(9, initializer=_init) as P:
        for fi, field in enumerate(fields):
            rng = np.random.default_rng(field)
            pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
            for arm, fn in (("llm16", lambda: llm16(base, digest, P, field)),
                            ("local16", lambda: local16(base, P, field, rng)),
                            ("random16", lambda: random16(base, P, field, rng))):
                h = sorted(fn(), key=lambda x: -x[1])[:KEEP]
                out[arm][str(field)] = [[list(map(int, t)), float(v)] for t, v in h]
            tn = []
            for kind in ("open", "hidden", "mixed"):
                L, g, _ = choose_force(base, pp, kind, K, np.random.default_rng(field),
                                       w=W, tau=TAU, doctrine=DOC32)
                tn.append([[int(x) for x in L], float(g.episodic(T=T_MISSION))])
            out["tuned"][str(field)] = sorted(tn, key=lambda x: -x[1])[:KEEP]
            print(f"  field {field}: " + " ".join(
                f"{a} {out[a][str(field)][0][1]:.4f}" for a in out)
                + f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n{'arm':10s} {'train-field median best':>24s} {'test-field median best':>24s}")
    for a in out:
        tr = np.median([out[a][str(f)][0][1] for f in TRAIN_FIELDS])
        te = np.median([out[a][str(f)][0][1] for f in TEST_FIELDS])
        print(f"{a:10s} {tr:24.4f} {te:24.4f}")
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
