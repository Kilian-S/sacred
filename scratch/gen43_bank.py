#!/usr/bin/env python3
"""gen43 BANK BUILDER: the forty-question placement exam (oracle-only, no model calls).
Pinned by the gen43 pre-registration. Each item is a (theatre, field, S-slot subset, K)
puzzle with EVERY combination valued exactly; ceiling, median and the full table are saved,
and the prompt is built once here so every config sits byte-identical papers.

    PYTHONPATH=. ../sacred/.venv/bin/python scratch/gen43_bank.py
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from scratch.gen39_compose import CR
from scratch.gen39_phase1e import catalogue_text, slot_coverage, slot_table
from scratch.gen39_zeroshot import base_for
from src.envs.aerial_conceal import resample_field
from src.envs.aerial_theatre_vec import terrain_v2
from scratch.gen39_phase1e import score_sites

MAPS = ("narva", "kgd_gvardeysk", "ukraine", "fulda")
SK_CYCLE = ((6, 2), (8, 2), (8, 3), (10, 3), (12, 3))
TARGET, FIELD0 = 40, 43000
CEIL_MIN, CEIL_OVER_MED = 0.010, 2.0
OUT = Path("models/runs/gen43_exam/bank.json")

TASK = """

TASK: choose exactly {k} slots from the catalogue above (one per team, no slot twice). Your
goal is the force that stays most dangerous against a flight that already KNOWS where every
team is: read the coverage lists and leave the flight no cheap option. Reply as JSON of
exactly this form: {{"slots": ["class/region", "class/region"{ell}]}}"""

_CTX: dict = {}


def _init():
    _CTX["bases"] = {}


def _eval(spec):
    name, field, sites = spec
    b = _CTX["bases"].get(name)
    if b is None:
        b = _CTX["bases"][name] = base_for(name)
    pp = b.lethality(resample_field(b.coords, field), hidden_leth=1.0)
    return sites, float(score_sites(b, pp, list(sites))[0])


def main():
    import multiprocessing as mp_
    terr = terrain_v2(hidden_leth=1.0, conceal_reach=CR)
    bases = {m: base_for(m) for m in MAPS}
    items, skipped = [], []
    idx, t0 = 0, time.time()
    with mp_.get_context("spawn").Pool(6, initializer=_init) as P:
        while len(items) < TARGET:
            name = MAPS[idx % len(MAPS)]
            S, K = SK_CYCLE[idx % len(SK_CYCLE)]
            field = FIELD0 + idx
            rng = np.random.default_rng(4300 + idx)
            idx += 1
            b = bases[name]
            pp = b.lethality(resample_field(b.coords, field), hidden_leth=1.0)
            slots = slot_table(b, pp)
            if len(slots) < S:
                skipped.append(dict(map=name, field=field, S=S, K=K, why="too_few_slots"))
                continue
            sub = sorted(rng.choice(sorted(slots), S, replace=False).tolist())
            subd = {s: slots[s] for s in sub}
            combos = [tuple(c) for c in itertools.combinations(sub, K)]
            specs = [(name, field, tuple(subd[s] for s in c)) for c in combos]
            got = dict(P.imap_unordered(_eval, specs, chunksize=4))
            vals = [float(got[tuple(subd[s] for s in c)]) for c in combos]
            ceil, medv = max(vals), float(np.median(vals))
            if ceil < CEIL_MIN or ceil < CEIL_OVER_MED * medv:
                skipped.append(dict(map=name, field=field, S=S, K=K, why="degenerate",
                                    ceiling=ceil, median=medv))
                continue
            cov = slot_coverage(b, pp, subd)
            prompt = catalogue_text(b, subd, cov, terr) + TASK.format(
                k=K, ell=", ..." if K > 2 else "")
            items.append(dict(id=len(items), map=name, field=field, S=S, K=K, slots=subd,
                              table=[[list(c), v] for c, v in zip(combos, vals)],
                              ceiling=ceil, median=medv,
                              best=list(combos[int(np.argmax(vals))]), prompt=prompt))
            print(f"  item {len(items):2d}/{TARGET}: {name} f{field} S={S} K={K} "
                  f"combos={len(combos)} ceil={ceil:.4f} med={medv:.4f} "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(items=items, skipped=skipped), indent=1))
    per_map = {m: sum(1 for i in items if i['map'] == m) for m in MAPS}
    print(f"\n[written] {OUT}: {len(items)} items ({per_map}), {len(skipped)} screened out")


if __name__ == "__main__":
    main()
