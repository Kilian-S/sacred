#!/usr/bin/env python3
"""gen39 FREE GATE before the strong-curriculum training programme (Kilian, 2026-07-27).

Two questions, both oracle-only, both decided before any training CPU:

A. DOCTRINE ATTRIBUTION. On the best positions a 16-evaluation search finds, how much of the
   curriculum strength is the POSITIONS (geometry) and how much is the BEHAVIOUR (doctrine)? We
   score three variants on the SAME winning sites: gen32 doctrine (Phase 1f's frozen recipe),
   an LLM-WRITTEN doctrine for those sites, and the LLM choosing BOTH. This says which variant
   carries the strongest curriculum and how much credit is the LLM's, before we build arms
   around it. (Kilian's addition to the gate.)

B. FOUR-MAP CURRICULUM STRENGTH. Build the 16-evaluation curricula (LLM-proposed / local-search /
   random) on ALL FOUR theatres under the UNIFORM pinned table (comparable across maps; kgd sits
   off its own operating point, disclosed), and verify the mechanism's ordering - strong search
   >> tuned doctrine 0.0215 >> nothing - holds off Narva. If it does not hold on a map, we learn
   that here for free rather than after a night of training.

Everything is exact irreducible threat (damage against a defender that KNOWS the laydown), median
over the fields. No training.

    PYTHONPATH=. python scratch/gen39_freegate.py --maps narva,kgd_gvardeysk,ukraine,fulda
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

from scratch.gen39_compose import BASE_URL, FIELDS, K, KEY, MODELS, OUTDIR, g33
from scratch.gen39_phase1f import map_digest, llm_prompt, SCHEMA
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2
from src.redforce import TAU_BIN

W, TAU, T_MISSION = 2, 0.10, 40
CR, RM = 0.85, 0.7
DOC32 = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3)
TUNED_BAR = 0.0215
PATH = "data/maps/theatre_%s_vec.json"
OUT = Path("models/runs/gen39_freegate.json")
BUDGET = 16                                       # the operating budget Kilian chose from 1f


def base_for(name):
    sc = lateral_width(load_vec_theatre(PATH % name)) / lateral_width(
        load_vec_theatre(PATH % "kgd_gvardeysk"))
    return ConcealBase(PATH % name, terrain=terrain_v2(hidden_leth=1.0, conceal_reach=CR),
                       range_scale=sc * RM, spacing_km=2.0 * sc, standoff_km=4.0 * sc,
                       n_sites=200)


_CTX: dict = {}


def _init(name):
    _CTX["base"] = base_for(name)
    _CTX["pp"] = {f: _CTX["base"].lethality(resample_field(_CTX["base"].coords, f),
                                            hidden_leth=1.0) for f in FIELDS}


def _eval(spec):
    sites, field, doc = spec
    base = _CTX["base"]
    d = None if doc == "g32" else doc
    kw = DOC32 if doc == "g32" else {}
    g = ConcealDyn(base, _CTX["pp"][field], np.asarray(sites, int), w=W, tau=TAU,
                   doctrines=d, **kw)
    return tuple(sites), field, doc if isinstance(doc, str) else "llm", float(g.episodic(T=T_MISSION))


def score(pool, triples, doc="g32"):
    specs = [(t, f, doc) for t in triples for f in FIELDS]
    agg: dict = {}
    for t, f, _dl, v in pool.imap_unordered(_eval, specs, chunksize=3):
        agg.setdefault(t, []).append(v)
    return {t: float(np.median(v)) for t, v in agg.items()}


def llm_search(name, base, digest, pool, rounds=4):
    """16-evaluation LLM-proposed search (llama; the Phase-1f arm), returns (best_sites, curve)."""
    hist, best, bt = [], -1.0, None
    per = max(1, BUDGET // rounds)
    for rnd in range(rounds):
        left = min(per, BUDGET - len(hist))
        if left <= 0:
            break
        tri = []
        for _ in range(2):
            try:
                txt, _m = g33.call_openai(BASE_URL, KEY, "llama-3.3-70b",
                                          "You are an air-defence planner running a search.",
                                          llm_prompt(digest, hist, left), schema=SCHEMA,
                                          max_tokens=3000, temperature=0.9, timeout=900)
                obj = g33._extract_json(txt)
                for f in obj.get("forces", []):
                    s = [int(x) for x in f.get("sites", []) if 0 <= int(x) < base.H]
                    if len(set(s)) == K:
                        tri.append(tuple(sorted(set(s))))
                if tri:
                    break
            except Exception:                                          # noqa: BLE001
                continue
        tri = [t for t in dict.fromkeys(tri) if t not in dict(hist)][:left]
        if not tri:
            continue
        got = score(pool, tri, "g32")
        for t in tri:
            hist.append((t, got[t]))
            if got[t] > best:
                best, bt = got[t], t
    return bt, best, hist


def local_search(base, pool, rng):
    used, best, cur = 0, -1.0, tuple(sorted(int(x) for x in rng.choice(base.H, K, replace=False)))
    val = score(pool, [cur], "g32")[cur]; used += 1; best = val; bt = cur
    while used < BUDGET:
        cands = []
        for s in range(K):
            for _ in range(6):
                c = list(cur); c[s] = int(rng.integers(base.H))
                if len(set(c)) == K:
                    cands.append(tuple(sorted(c)))
        cands = list(dict.fromkeys(cands))[:min(9, BUDGET - used)]
        if not cands:
            break
        got = score(pool, cands, "g32"); used += len(cands)
        b = max(got, key=got.get)
        if got[b] > val:
            cur, val = b, got[b]
        else:
            cur = tuple(sorted(int(x) for x in rng.choice(base.H, K, replace=False)))
            val = score(pool, [cur], "g32")[cur]; used += 1
        if val > best:
            best, bt = val, cur
    return bt, best


def random_search(base, pool, rng):
    tri = [tuple(sorted(int(x) for x in rng.choice(base.H, K, replace=False)))
           for _ in range(BUDGET)]
    got = score(pool, tri, "g32")
    bt = max(got, key=got.get)
    return bt, got[bt]


def ask_llm_doctrine(base, digest, sites):
    """Part A: ask the model to WRITE a doctrine for a force ALREADY PLACED at `sites`."""
    desc = []
    for i in sites:
        g = ConcealDyn(base, base.lethality(resample_field(base.coords, FIELDS[0]),
                                            hidden_leth=1.0), np.array([i]), w=W, tau=TAU)
        d = g.dmg_j[0]
        desc.append(f"site {int(i)} ({base.cls[i]}, threatens routes "
                    f"{sorted(int(r) for r in np.where(d > 0.02 * d.max())[0])})")
    prompt = ("Your force of 3 air-defence teams is already placed:\n  " + "\n  ".join(desc)
              + "\n\nWrite a DOCTRINE for each team (punish_pattern / anticipate_flight / "
              "hold_static, summing to ~1, plus decisiveness decisive|balanced|hedged and memory "
              "1-2). The enemy flight reacts to its own recent routes; choose behaviour that makes "
              "your fixed force hardest to evade. Emit ONLY a JSON object "
              '{"doctrines":[{...},{...},{...}]}.')
    for _ in range(2):
        try:
            txt, _m = g33.call_openai(BASE_URL, KEY, "llama-3.3-70b",
                                      "You are an air-defence doctrine planner.", prompt,
                                      max_tokens=2000, temperature=0.7, timeout=900)
            obj = g33._extract_json(txt)
            ds = obj["doctrines"][:K]
            out = []
            for d in ds:
                tot = (d.get("punish_pattern", 0) + d.get("anticipate_flight", 0)
                       + d.get("hold_static", 0)) or 1.0
                out.append(dict(q_rep=d.get("punish_pattern", 0) / tot,
                                q_flee=d.get("anticipate_flight", 0) / tot,
                                q_hold=d.get("hold_static", 0) / tot,
                                tau=TAU_BIN.get(d.get("decisiveness", "balanced"), TAU),
                                w=int(np.clip(d.get("memory", W), 1, W))))
            if len(out) == K:
                return out
        except Exception:                                              # noqa: BLE001
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="narva,kgd_gvardeysk,ukraine,fulda")
    a = ap.parse_args()
    import multiprocessing as mp_
    maps = a.maps.split(",")
    result = {}
    t0 = time.time()
    for name in maps:
        base = base_for(name)
        pp0 = base.lethality(resample_field(base.coords, FIELDS[0]), hidden_leth=1.0)
        digest = map_digest(base, pp0)
        rng = np.random.default_rng(hash(name) % 2**31)
        with mp_.get_context("spawn").Pool(9, initializer=_init, initargs=(name,)) as P:
            llm_sites, llm_best, _hist = llm_search(name, base, digest, P)
            loc_sites, loc_best = local_search(base, P, rng)
            rnd_sites, rnd_best = random_search(base, P, rng)
            # PART A (on this map's LLM-best positions): three doctrine variants
            doc_g32 = score(P, [llm_sites], "g32")[llm_sites] if llm_sites else None
            llm_doc = ask_llm_doctrine(base, digest, llm_sites) if llm_sites else None
            doc_llm = (score(P, [llm_sites], llm_doc)[llm_sites]
                       if (llm_sites and llm_doc) else None)
        result[name] = dict(llm_best=llm_best, local_best=loc_best, random_best=rnd_best,
                            llm_sites=[int(x) for x in llm_sites] if llm_sites else None,
                            doctrine_g32=doc_g32, doctrine_llm=doc_llm)
        print(f"[{name}] LLM {llm_best:.4f} | local {loc_best:.4f} | random {rnd_best:.4f} | "
              f"tuned-bar {TUNED_BAR} || doctrine on LLM sites: g32 {doc_g32} llm {doc_llm} "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    OUT.write_text(json.dumps(result, indent=1))
    print(f"\n{'=' * 90}\nFREE GATE (irreducible threat; tuned doctrine {TUNED_BAR}; higher = "
          f"stronger curriculum)\n{'=' * 90}")
    print(f'{"map":16s} {"LLM-16":>8s} {"local-16":>9s} {"random-16":>10s} {"tuned":>7s} | '
          f'{"ordering holds?":>16s}')
    for name in maps:
        r = result[name]
        holds = r["llm_best"] > TUNED_BAR and r["local_best"] > TUNED_BAR
        print(f'{name:16s} {r["llm_best"]:8.4f} {r["local_best"]:9.4f} {r["random_best"]:10.4f} '
              f'{TUNED_BAR:7.4f} | {"YES" if holds else "NO":>16s}')
    print(f'\nPART A - doctrine attribution on the LLM-best positions (Narva primary):')
    r = result[maps[0]]
    if r["doctrine_g32"] and r["doctrine_llm"]:
        print(f'  same positions: gen32 doctrine {r["doctrine_g32"]:.4f} vs LLM-written '
              f'{r["doctrine_llm"]:.4f} -> LLM doctrine is '
              f'{r["doctrine_llm"]/r["doctrine_g32"]:.2f}x')
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
