#!/usr/bin/env python3
"""gen39 Phase 1d: THE FEEDBACK LOOP THE LLM ACTUALLY DESERVES (Kilian, 2026-07-27).

Kilian's argument, accepted: a heuristic cannot act on feedback and the optimiser behind it only
searches blindly; reading a report and reasoning about it is the ONE capability a language model
has that the alternatives do not, and Phase 1c never tested it - it handed the model a grade (two
scalars), not an account of the battle. This gives it a real AFTER-ACTION REPORT each round.

The report is DIAGNOSIS, never PRESCRIPTION (the binding design line): it says what happened, and
never which site to move to. The moment a counterfactual "move team 2 here and damage rises to X"
enters the prompt, the optimiser is solving the problem and the model is transcribing it.

Per round the model receives:
  * the mission outcome: total damage, and the per-serial decay curve (does the force fade as it
    is located?);
  * the ROUTE TABLE: for every route, its cost to the flight, which of its teams threaten it, and
    how much of the defender's flying went down it;
  * the FREE-LANE list: routes no team threatens, and the cost of the flight's safest option (the
    single number the position optimiser maximises);
  * per TEAM: damage contributed, engagements, whether and when it was located, routes covered,
    and overlap with its team-mates (redundant vs complementary);
  * its own history: every previous force and its two scores.

GROUNDING CHECK (Kilian's addition): each force must also declare `intended_routes`, the route
numbers it believes it will threaten. We score that against the truth. This separates the two
failure modes: a model that misreads the report (low grounding) is failing at comprehension; a
model that reads it correctly and still cannot close the lanes (high grounding, flat threat) is
failing at combinatorial-geometric reasoning, which is the claim Phase 1c over-reached on.

BARS (fixed before any call):
  B1 free lanes fall materially across rounds (the report's core signal is acted on);
  B2 median irreducible threat rises toward the heuristic curriculum's 0.0215;
  B3 the best evolved force beats the heuristic force against a TRAINED SACRED defender
     (the matchup the curriculum question actually turns on; scored offline from the banked
     step-3 checkpoints, so it costs no training).
  Grounding is REPORTED per round, never gated.

    PYTHONPATH=. python analysis/gen39_phase1d.py --rounds 6 --n 3
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from analysis.gen39_compose import (BASE_URL, FIELDS, K, KEY, MODELS, OUTDIR, doctrines_of, g33,
                                   narva_base, place, score_force)
from src.envs.aerial_conceal import ConcealDyn, resample_field
from src.redforce import force_schema, serialise_theatre

W, TAU, T_MISSION = 2, 0.10, 40
HEUR_BAR = 0.0215
REPORT_FIELD = FIELDS[0]          # the field the narrative report is drawn from (scores use all)
OUT = Path("models/runs/gen39_phase1d.json")


# --- the after-action report -------------------------------------------------------------------

def after_action(base, force, field):
    """Exact per-route / per-team account of one mission. Diagnosis only."""
    pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    sites = place(force, base, pp)
    g = ConcealDyn(base, pp, sites, w=W, tau=TAU, doctrines=doctrines_of(force))
    R, k = g.R, len(g.L)
    # per-team threat per route (the concentration-weighted damage), and coverage
    dmg_j = g.dmg_j                                   # [k, R]
    covers = [sorted(int(r) for r in np.where(dmg_j[j] > 0.02 * dmg_j.max())[0]) for j in range(k)]
    # the flight's per-route cost under the enemy's actual aim, and its safest option
    route_cost = g.stepdmg.mean(axis=0)               # [R] average over track states
    safest = int(np.argmin(route_cost))
    free = [int(r) for r in range(R) if all(r not in c for c in covers)]
    # what the defender actually flies: the best observing rule's stationary route use
    sup = g.blind_supports()
    best_d, best_v = None, np.inf
    for d in sup.values():
        for anti in (False, True):
            for s, t in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3), (0.0, 5)):
                v = g.episodic_rule(d, anti_repeat=anti, softness=s, topm=t, T=T_MISSION)
                if v < best_v:
                    best_v, best_d = v, (d, anti, s, t)
    d, anti, s, t = best_d
    use = np.asarray(g._anti(d) if anti else np.broadcast_to(d, (len(g.states), R))).mean(axis=0)
    decay = g.episodic(horizons=(5, 10, 20, 40), rule=lambda i, m, p, M=np.asarray(
        g._anti(d), float): M)
    exposed_by = {}
    for j in range(k):
        rs = np.where(g.revealable[:, j])[0]
        exposed_by[j] = [int(x) for x in rs]
    return dict(g=g, route_cost=route_cost, use=use, covers=covers, free=free, safest=safest,
                safest_cost=float(route_cost[safest]), decay=decay, exposed_by=exposed_by,
                team_dmg=[float(x) for x in dmg_j.sum(axis=1)], R=R, k=k,
                cls=[base.cls[i] for i in sites])


def report_text(aa, force, scores, history):
    R, k = aa["R"], aa["k"]
    order = np.argsort(-aa["route_cost"])
    lines = ["ROUTE TABLE (all %d routes the flight can choose; cost = losses it takes there, "
             "'used' = share of the defender's flying that went down it):" % R,
             "  route |  cost  | used  | threatened by"]
    for r in order:
        who = [f"team{j}" for j in range(k) if int(r) in aa["covers"][j]]
        lines.append(f"  {int(r):5d} | {aa['route_cost'][r]:.4f} | {100*aa['use'][r]:4.0f}% | "
                     f"{','.join(who) if who else 'NOBODY (free lane)'}")
    lines.append("")
    lines.append(f"FREE LANES (threatened by no team): {aa['free'] if aa['free'] else 'none'}")
    lines.append(f"The flight's SAFEST option is route {aa['safest']} at cost "
                 f"{aa['safest_cost']:.4f}. Raising the cost of the flight's safest option is what "
                 f"makes a force irreducibly dangerous.")
    lines.append("")
    lines.append("PER TEAM:")
    for j in range(k):
        a = force["agents"][j]
        ex = aa["exposed_by"][j]
        lines.append(
            f"  team{j} ({a['emplacement_zone']['terrain']} / "
            f"{a['emplacement_zone']['region']}, actually stood on {aa['cls'][j]}): "
            f"threat {aa['team_dmg'][j]:.4f}; covers routes {aa['covers'][j] or 'NONE'}; "
            + (f"gives its position away if the flight uses routes {ex}"
               if ex else "never gives its position away (concealed ground)"))
    ov = []
    for i in range(k):
        for j in range(i + 1, k):
            sh = set(aa["covers"][i]) & set(aa["covers"][j])
            if sh:
                ov.append(f"team{i}+team{j} both cover {sorted(sh)}")
    lines.append("  OVERLAP: " + ("; ".join(ov) if ov else "none - the three teams cover "
                                                           "different routes"))
    lines.append("")
    lines.append("MISSION DECAY (mean damage per serial as the mission runs; a force that is "
                 "located fades):")
    lines.append("  " + "  ".join(f"first {t} serials: {v:.4f}" for t, v in aa["decay"].items()))
    lines.append("")
    lines.append(f"YOUR TWO SCORES: against a defender that must SEARCH {scores[1]:.4f}; "
                 f"against a defender that KNOWS where all your teams are {scores[0]:.4f}. "
                 f"Our own position optimiser reaches {HEUR_BAR:.4f} on the second.")
    if history:
        lines.append("\nYOUR PREVIOUS ATTEMPTS:")
        for h in history:
            gr = ("not stated" if h["grounding"] is None else f"{h['grounding']:.0%}")
            lines.append(f"  round {h['round']}: searching {h['observing']:.4f}, knowing "
                         f"{h['irreducible']:.4f}, free lanes {h['n_free']}, "
                         f"grounding {gr}, terrain "
                         f"{[x['emplacement_zone']['terrain'] for x in h['force']['agents']]}")
    return "\n".join(lines)


GROUND_CLAUSE = """

ALSO REQUIRED: in the rationale of the FIRST agent, end with a line of exactly this form
  INTENDED_ROUTES: 3,7,12
listing every route number you believe your force will threaten. This is checked against the
simulation; it is how we tell a misread report from a hard problem. Be honest, not optimistic."""


def parse_intent(force):
    for a in force.get("agents", []):
        txt = a.get("rationale", "")
        if "INTENDED_ROUTES" in txt:
            tail = txt.split("INTENDED_ROUTES")[-1].lstrip(": ").split("\n")[0]
            out = []
            for tok in tail.replace(";", ",").split(","):
                tok = "".join(c for c in tok if c.isdigit())
                if tok:
                    out.append(int(tok))
            return sorted(set(out))
    return None


def grounding(intent, aa):
    """Jaccard between the routes the model SAID it would threaten and the truth."""
    if intent is None:
        return None
    truth = set().union(*[set(c) for c in aa["covers"]]) if aa["covers"] else set()
    if not truth and not intent:
        return 1.0
    return len(set(intent) & truth) / max(len(set(intent) | truth), 1)


# --- exact scoring across the field set ---------------------------------------------------------
_CTX: dict = {}


def _init():
    _CTX["base"], _, _ = narva_base()


def _task(spec):
    key, force, field = spec
    base = _CTX["base"]
    pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    o, b, ob, cov = score_force(base, pp, place(force, base, pp), doctrines_of(force))
    return key, (o, ob)


def score_many(forces, workers=9):
    import multiprocessing as mp_
    specs = [(k, f, fld) for k, f in forces.items() for fld in FIELDS]
    agg: dict = {}
    with mp_.get_context("spawn").Pool(workers, initializer=_init) as P:
        for k, v in P.imap_unordered(_task, specs, chunksize=3):
            agg.setdefault(k, []).append(v)
    return {k: tuple(np.median(np.array(v), axis=0)) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--n", type=int, default=3, help="parallel lineages per model")
    a = ap.parse_args()
    base, terr, sc = narva_base()
    schema = force_schema(terr)
    system, user0 = serialise_theatre(base.th, phase="coordinated", K=K,
                                      range_scale=sc * 0.7, terrain=terr)
    user0 += GROUND_CLAUSE
    (OUTDIR / "brief_phase1d.txt").write_text(system + "\n\n---\n\n" + user0)

    live = {f"{m}#{i}": dict(model=m, prompt=user0, force=None, hist=[])
            for m in MODELS for i in range(a.n)}
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
                    print(f"  [1d call FAILED] {key}: {type(e).__name__}: {e}", flush=True)
                    continue
            return key, None
        with ThreadPoolExecutor(max_workers=8) as ex:
            for key, obj in ex.map(one, list(live.items())):
                if obj:
                    live[key]["force"] = obj
        ok = {k: v["force"] for k, v in live.items() if v["force"]}
        if not ok:
            print(f"  [round {rnd}] no valid forces", flush=True)
            continue
        scored = score_many(ok)
        for key, force in ok.items():
            irr, obs = scored[key]
            aa = after_action(base, force, REPORT_FIELD)
            gr = grounding(parse_intent(force), aa)
            rec = dict(round=rnd, key=key, model=live[key]["model"], irreducible=float(irr),
                       observing=float(obs), n_free=len(aa["free"]),
                       safest_cost=aa["safest_cost"], grounding=gr,
                       terrain=[x["emplacement_zone"]["terrain"] for x in force["agents"]],
                       force=force)
            log.append(rec)
            live[key]["hist"].append(rec)
            live[key]["prompt"] = (
                user0 + "\n\n" + "=" * 70
                + f"\nAFTER-ACTION REPORT ON YOUR ROUND-{rnd} FORCE\n" + "=" * 70 + "\n"
                + report_text(aa, force, (irr, obs), live[key]["hist"][:-1])
                + "\n\nNow issue a REVISED force of the same 3 teams. Emit ONLY the structured "
                  "force, with the INTENDED_ROUTES line in the first rationale.")
        r = [x for x in log if x["round"] == rnd]
        print(f"  [round {rnd}] irreducible median {np.median([x['irreducible'] for x in r]):.5f} "
              f"({np.median([x['irreducible'] for x in r]) / HEUR_BAR:.0%} of bar) | free lanes "
              f"{np.mean([x['n_free'] for x in r]):.1f} | grounding "
              f"{np.mean([x['grounding'] for x in r if x['grounding'] is not None] or [float('nan')]):.0%}"
              f" | {(time.time() - t0) / 60:.1f} min", flush=True)

    OUT.write_text(json.dumps(log, indent=1))
    print(f"\n{'=' * 84}\nPHASE 1D (bar {HEUR_BAR:.4f}; grounding reported, never gated)\n{'=' * 84}")
    print(f'{"round":>6s} {"n":>3s} {"irreducible":>12s} {"% bar":>6s} {"vs searcher":>12s} '
          f'{"free lanes":>11s} {"safest cost":>12s} {"grounding":>10s}')
    for rnd in range(a.rounds):
        r = [x for x in log if x["round"] == rnd]
        if not r:
            continue
        gs = [x["grounding"] for x in r if x["grounding"] is not None]
        print(f'{rnd:6d} {len(r):3d} {np.median([x["irreducible"] for x in r]):12.5f} '
              f'{np.median([x["irreducible"] for x in r]) / HEUR_BAR:5.0%} '
              f'{np.median([x["observing"] for x in r]):12.4f} '
              f'{np.mean([x["n_free"] for x in r]):11.1f} '
              f'{np.median([x["safest_cost"] for x in r]):12.4f} '
              f'{(np.mean(gs) if gs else float("nan")):9.0%}')
    best = max(log, key=lambda x: x["irreducible"])
    print(f'\nBEST FORCE: {best["key"]} round {best["round"]}, irreducible {best["irreducible"]:.4f} '
          f'({best["irreducible"] / HEUR_BAR:.0%} of bar), terrain {best["terrain"]}, '
          f'free lanes {best["n_free"]}')
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
