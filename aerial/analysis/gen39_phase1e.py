#!/usr/bin/env python3
"""Hands the composing model a grounded action space and a like-for-like ceiling.

The model receives a catalogue of every (class, region) slot on the map, each annotated with the
routes a team posted there would threaten, its reach and lethality, and whether it reveals itself,
and then picks three slots and a doctrine per team, so its choice determines coverage readably.
The catalogue stays descriptive: it says what each slot covers, never which combination to choose,
and no counterfactual scores are given. Because the unrestricted optimiser searches thousands of
exact site combinations while the model picks three slots from a dozen, every three-slot
combination from the same catalogue is scored exhaustively under the same doctrine and the model
is reported against that restricted best.

    PYTHONPATH=. python analysis/gen39_phase1e.py --n 4 --rounds 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from analysis.gen39_compose import (BASE_URL, FIELDS, K, KEY, MODELS, OUTDIR, g33, narva_base)
from analysis.gen39_phase1d import after_action, grounding, parse_intent
from src.envs.aerial_conceal import ConcealDyn, resample_field
from src.redforce import TAU_BIN, force_schema, serialise_theatre

W, TAU, T_MISSION = 2, 0.10, 40
N_ROUTES = 26                     # narva menu size, recomputed at run time
UNRESTRICTED_BAR = 0.0215
REGIONS = ("near_base", "mid_corridor", "near_target_standoff")
CLASSES = ("open", "field", "forest", "urban")
OUT = Path("models/runs/gen39_phase1e.json")


def slot_table(base, pp):
    """Builds the slot catalogue, mapping each existing (class, region) slot to its site.

    The site is the highest-threat one in that slot, which is the placer's own rule, so the
    catalogue matches where a team would actually stand.
    """
    thr = base.threat_rank(pp)
    v = base.th.target - base.th.base
    u = v / (np.linalg.norm(v) + 1e-9)
    frac = np.clip(((base.coords - base.th.base) @ u) / (float(v @ u) + 1e-9), 0, 1)
    lohi = {"near_base": (0.0, 0.34), "mid_corridor": (0.34, 0.66),
            "near_target_standoff": (0.66, 1.0)}
    slots = {}
    for c in CLASSES:
        for r in REGIONS:
            lo, hi = lohi[r]
            pool = [i for i, cl in enumerate(base.cls) if cl == c and lo <= frac[i] <= hi]
            if not pool:
                continue
            site = int(max(pool, key=lambda i: thr[i]))
            slots[f"{c}/{r}"] = site
    return slots


def slot_coverage(base, pp, slots):
    """Routes each slot's team would threaten, computed exactly with one single-team game per slot."""
    cov = {}
    for name, site in slots.items():
        g = ConcealDyn(base, pp, np.array([site]), w=W, tau=TAU)
        d = g.dmg_j[0]
        cov[name] = sorted(int(r) for r in np.where(d > 0.02 * d.max())[0])
    return cov


def catalogue_text(base, slots, cov, terr):
    lines = ["YOUR AVAILABLE EMPLACEMENT SLOTS. Each line is a place a team can be posted, and "
             "the routes it would then threaten (computed exactly on this map):",
             "  slot                          | reach | lethality | reveals? | routes it threatens"]
    for name in sorted(slots):
        c = name.split("/")[0]
        spec = terr[c]
        lines.append(f"  {name:29s} | {spec['r_km']:4.1f}km | {spec['p_max']:9.2f} | "
                     f"{'YES' if spec.get('reveal', True) else ' no':>8s} | {cov[name]}")
    allr = sorted(set().union(*cov.values()))
    lines.append(f"\nThe flight chooses among routes {allr[0]}-{allr[-1]}. A route threatened by "
                 f"NO team is a free lane and costs the flight nothing: your force is only as "
                 f"dangerous as the flight's safest remaining option.")
    return "\n".join(lines)


TASK = """

TASK: choose exactly {k} slots from the catalogue above (one per team, no slot twice) and a
doctrine mix for each team. You are choosing which routes your force denies: read the coverage
lists and pick a combination that leaves the flight no cheap option.

Put the slot you chose for each team in emplacement_zone (terrain = the part before the '/',
region = the part after). End the FIRST agent's rationale with a line of exactly this form
  INTENDED_ROUTES: 3,7,12
listing every route your combination threatens (you can read this off the catalogue)."""


def force_from(agents_slots, doctrines):
    return {"agents": [{"archetype": "blocker",
                        "emplacement_zone": {"terrain": s.split("/")[0], "region": s.split("/")[1]},
                        "doctrine": d, "decisiveness": "balanced", "memory": W,
                        "rationale": "enumerated"} for s, d in zip(agents_slots, doctrines)]}


def score_sites(base, pp, sites, doctrines=None):
    g = ConcealDyn(base, pp, np.asarray(sites, int), w=W, tau=TAU, doctrines=doctrines)
    sup = g.blind_supports()
    blind = min(g.episodic(rule=lambda i, m, p, M=np.asarray(g._anti(d), float): M, T=T_MISSION)
                for d in sup.values())
    obs = min([blind] + [g.episodic_rule(d, anti_repeat=a, softness=s, topm=t, T=T_MISSION)
                         for d in sup.values() for a in (False, True)
                         for s, t in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3), (0.0, 5))])
    return float(g.episodic(T=T_MISSION)), float(obs)


_CTX: dict = {}


def _init():
    _CTX["base"], _CTX["terr"], _ = narva_base()


def _ceiling_task(spec):
    combo, field = spec
    base = _CTX["base"]
    pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    slots = slot_table(base, pp)
    sites = [slots[s] for s in combo]
    return combo, field, score_sites(base, pp, sites)[0]


def _force_task(spec):
    key, slotnames, doctrines, field = spec
    base = _CTX["base"]
    pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    slots = slot_table(base, pp)
    sites = [slots[s] for s in slotnames if s in slots]
    if len(sites) < K:
        return key, field, None
    return key, field, score_sites(base, pp, sites, doctrines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="lineages per model")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    import multiprocessing as mp_
    base, terr, sc = narva_base()
    pp0 = base.lethality(resample_field(base.coords, FIELDS[0]), hidden_leth=1.0)
    slots = slot_table(base, pp0)
    cov = slot_coverage(base, pp0, slots)
    cat = catalogue_text(base, slots, cov, terr)
    n_routes = len(ConcealDyn(base, pp0, np.array([list(slots.values())[0]]), w=W,
                              tau=TAU).dmg_j[0])
    globals()["N_ROUTES"] = n_routes
    print(f"[1e] {len(slots)} slots, {n_routes} routes: {sorted(slots)}\n")

    # --- the fair ceiling: every 3-slot combination, exactly scored, same doctrine -------------
    combos = list(itertools.combinations(sorted(slots), K))
    specs = [(c, f) for c in combos for f in FIELDS]
    agg: dict = {}
    with mp_.get_context("spawn").Pool(9, initializer=_init) as P:
        for combo, field, v in P.imap_unordered(_ceiling_task, specs, chunksize=4):
            agg.setdefault(combo, []).append(v)
    ceil = {c: float(np.median(v)) for c, v in agg.items()}
    best_combo = max(ceil, key=ceil.get)
    CEIL = ceil[best_combo]
    rnd_med = float(np.median(list(ceil.values())))
    print(f"[1e] restricted ceiling over {len(combos)} slot combinations: {CEIL:.4f} "
          f"({best_combo}); median (= a random slot choice) {rnd_med:.4f}; "
          f"unrestricted optimiser {UNRESTRICTED_BAR:.4f}\n")

    schema = force_schema(terr)
    system, user0 = serialise_theatre(base.th, phase="coordinated", K=K,
                                      range_scale=sc * 0.7, terrain=terr)
    user0 = user0 + "\n\n" + cat + TASK.format(k=K)
    (OUTDIR / "brief_phase1e.txt").write_text(system + "\n\n---\n\n" + user0)

    live = {f"{m}#{i}": dict(model=m, prompt=user0, hist=[]) for m in MODELS for i in range(a.n)}
    log = []
    t0 = time.time()
    for rnd in range(a.rounds):
        def one(item):
            key, st = item
            for _ in range(2):
                try:
                    txt, _m = g33.call_openai(BASE_URL, KEY, st["model"], system, st["prompt"],
                                              schema=schema, max_tokens=3000, temperature=0.8,
                                              timeout=900)
                    obj = g33._extract_json(txt)
                    if not g33.validate_force(obj) and len(obj.get("agents", [])) == K:
                        return key, obj
                except Exception as e:                                 # noqa: BLE001
                    print(f"  [1e call FAILED] {key}: {type(e).__name__}: {e}", flush=True)
                    continue
            return key, None
        with ThreadPoolExecutor(max_workers=8) as ex:
            got = {k: v for k, v in ex.map(one, list(live.items())) if v}
        specs, meta = [], {}
        for key, force in got.items():
            sl = [f"{x['emplacement_zone']['terrain']}/{x['emplacement_zone']['region']}"
                  for x in force["agents"]]
            doc = [dict(q_rep=x["doctrine"]["punish_pattern"],
                        q_flee=x["doctrine"]["anticipate_flight"],
                        q_hold=x["doctrine"]["hold_static"],
                        tau=TAU_BIN.get(x.get("decisiveness", "balanced"), TAU), w=W)
                   for x in force["agents"]]
            meta[key] = (force, sl, doc)
            for f in FIELDS:
                specs.append((key, sl, doc, f))
        agg2: dict = {}
        with mp_.get_context("spawn").Pool(9, initializer=_init) as P:
            for key, field, v in P.imap_unordered(_force_task, specs, chunksize=3):
                if v:
                    agg2.setdefault(key, []).append(v)
        for key, (force, sl, doc) in meta.items():
            if key not in agg2:
                continue
            irr, obs = np.median(np.array(agg2[key]), axis=0)
            # grounding: declared routes vs the catalogue's truth for the slots it chose
            truth = sorted(set().union(*[set(cov[s]) for s in sl if s in cov])) if sl else []
            intent = parse_intent(force)
            gr = (len(set(intent) & set(truth)) / max(len(set(intent) | set(truth)), 1)
                  if intent is not None else None)
            free = [r for r in range(N_ROUTES) if r not in truth]
            rec = dict(round=rnd, key=key, model=live[key]["model"], slots=sl,
                       irreducible=float(irr), observing=float(obs), grounding=gr,
                       n_free=len(free), valid_slots=sum(s in cov for s in sl), force=force)
            log.append(rec)
            live[key]["hist"].append(rec)
            live[key]["prompt"] = (
                user0 + f"\n\nYOUR PREVIOUS CHOICE: slots {sl}\n"
                f"  it threatened routes {truth}\n  free lanes left: {free}\n"
                f"  damage against a defender that must SEARCH: {obs:.4f}\n"
                f"  damage against a defender that KNOWS your positions: {irr:.4f}\n"
                f"  the best possible 3-slot choice on this map reaches {CEIL:.4f}\n"
                "Choose again, better. Emit ONLY the structured force.")
        r = [x for x in log if x["round"] == rnd]
        gs = [x["grounding"] for x in r if x["grounding"] is not None]
        print(f"  [round {rnd}] n={len(r)} irreducible median "
              f"{np.median([x['irreducible'] for x in r]):.4f} "
              f"({np.median([x['irreducible'] for x in r]) / CEIL:.0%} of ceiling) | grounding "
              f"{(np.mean(gs) if gs else float('nan')):.0%} | free lanes "
              f"{np.mean([x['n_free'] for x in r]):.1f} | valid slots "
              f"{np.mean([x['valid_slots'] for x in r]):.1f}/{K} | {(time.time()-t0)/60:.1f} min",
              flush=True)

    OUT.write_text(json.dumps(dict(ceiling=CEIL, best_combo=best_combo, random_median=rnd_med,
                                   slots=sorted(slots), coverage=cov, log=log), indent=1))
    print(f"\n{'=' * 86}\nPHASE 1E: grounded action space, fair ceiling {CEIL:.4f} "
          f"(random slot choice {rnd_med:.4f}; unrestricted {UNRESTRICTED_BAR:.4f})\n{'=' * 86}")
    for m in list(MODELS) + ["ALL"]:
        r = [x for x in log if m == "ALL" or x["model"] == m]
        if not r:
            continue
        gs = [x["grounding"] for x in r if x["grounding"] is not None]
        med = np.median([x["irreducible"] for x in r])
        best = max(x["irreducible"] for x in r)
        print(f'{m:16s} n={len(r):3d}  median {med:.4f} ({med / CEIL:5.0%} of ceiling)  '
              f'best {best:.4f} ({best / CEIL:5.0%})  grounding '
              f'{(np.mean(gs) if gs else float("nan")):4.0%}  free lanes '
              f'{np.mean([x["n_free"] for x in r]):.1f}')
    allr = [x["irreducible"] for x in log]
    print(f'\nG  grounding >= 80%: '
          f'{"PASS" if np.mean([x["grounding"] for x in log if x["grounding"] is not None]) >= 0.8 else "FAIL"}')
    print(f'C1 median >= 60% of ceiling: '
          f'{"PASS" if np.median(allr) >= 0.6 * CEIL else "FAIL"} '
          f'({np.median(allr) / CEIL:.0%})')
    print(f'C2 best beats a random slot choice: '
          f'{"PASS" if max(allr) > rnd_med else "FAIL"} ({max(allr):.4f} vs {rnd_med:.4f})')
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
