#!/usr/bin/env python3
"""gen39 step 2: three composers write DOCTRINE + ROLES; one fixed algorithm places
(experiments/gen39_concealment.md, step-2 pre-registration + the implementation pins).

Arms (identical budget, terrain, weapons table and placer; only the composition differs):
  llm        llama-3.3-70b and qwen3-27b, per-model reporting, guided JSON (gen33 machinery);
  random     doctrine mixes drawn uniformly on the simplex, postures random;
  heuristic  the pinned gen32 doctrine replicated across the teams (the screen's own enemy:
             the per-team path reproduces it exactly, the tested regression anchor);
  relabel    the BINDING CONTROL: the same calls under a brief whose forest and open rows are
             SWAPPED; the model's terrain choices must materially change or the terrain-reasoning
             claim is not licensed (composition-without-terrain-content re-scope, as gen33).

Scoring (fixed before any call): every force is placed by the same placer (best-threat site of
its stated terrain in its stated region, no site reused) and scored EXACTLY on the pinned narva
cell under persistent memory, T=40. PRIMARY scorer = the best OBSERVING defender (the rule
family that uses what its track has spotted); the omniscient optimum and the blind value are
reported beside it. The oracle-searched best force (choose_force) is the reported ceiling.

    PYTHONPATH=. python analysis/gen39_compose.py --dry          # offline pipeline validation
    PYTHONPATH=. python analysis/gen39_compose.py --live         # the ~32-call live run
    PYTHONPATH=. python analysis/gen39_compose.py --score        # (re)score saved forces
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from src.envs.aerial_conceal import ConcealBase, ConcealDyn, choose_force, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2
from src.redforce import TAU_BIN, dry_force, force_schema, serialise_theatre

_spec = importlib.util.spec_from_file_location("g33", "scripts/gen33_generate_force.py")
g33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g33)

# --- the pinned operating point (ledger, 2026-07-26) -------------------------------------------
MAP, CR, RM, K = "narva", 0.85, 0.7, 3
PATH = "data/maps/theatre_%s_vec.json"
FIELDS = (5100, 5101, 5102)          # the cell's own fields; pristine 61xx stay for step 3
W, TAU = 2, 0.10
DOC32 = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3, tau=TAU, w=W)
MODELS = ("llama-3.3-70b", "qwen3-27b")
N_LLM, N_RANDOM = 8, 20
N_LLM_BIG = 32          # Phase 1b: a richer population, per model, for the curriculum question
BASE_URL, KEY = "http://cv-iits-w05.tail5b8d80.ts.net:8080/v1", "iits-local-key"
OUTDIR = Path("models/runs/gen39_compose")


def narva_base():
    sc = lateral_width(load_vec_theatre(PATH % MAP)) / lateral_width(
        load_vec_theatre(PATH % "kgd_gvardeysk"))
    t = terrain_v2(hidden_leth=1.0, conceal_reach=CR)
    return ConcealBase(PATH % MAP, terrain=t, range_scale=sc * RM, spacing_km=2.0 * sc,
                       standoff_km=4.0 * sc, n_sites=200), t, sc


def relabelled(t: dict) -> dict:
    """The control brief's table: forest and open SWAP characteristics (reach, lethality,
    reveal, los); emplaceability kept. Used ONLY for the brief text; placement and scoring stay
    on the true table."""
    s = {k: dict(v) for k, v in t.items()}
    for key in ("r_km", "p_max", "reveal", "los"):
        s["forest"][key], s["open"][key] = s["open"][key], s["forest"][key]
    return s


def place(force, base, pp):
    """The one placer every arm shares: per team, the highest-threat unused site of its stated
    terrain in its stated region; fall back to its terrain anywhere, then to any unused site."""
    thr = base.threat_rank(pp)
    v = base.th.target - base.th.base
    u = v / (np.linalg.norm(v) + 1e-9)
    frac = np.clip(((base.coords - base.th.base) @ u) / (float(v @ u) + 1e-9), 0, 1)
    cls = list(base.cls)
    taken, sites = set(), []
    for a in force["agents"]:
        terr = a["emplacement_zone"]["terrain"]
        region = a["emplacement_zone"]["region"]
        elig = [i for i, c in enumerate(cls)
                if (c == terr or (terr == "open" and c == "field")) and i not in taken]
        lo, hi = {"near_base": (0.0, 0.34), "near_target_standoff": (0.66, 1.0)}.get(
            region, (0.34, 0.66))
        pool = [i for i in elig if lo <= frac[i] <= hi] or elig \
            or [i for i in range(base.H) if i not in taken]
        site = int(max(pool, key=lambda i: thr[i]))
        taken.add(site)
        sites.append(site)
    return np.array(sites, dtype=int)


def doctrines_of(force):
    """Schema doctrine -> per-team ConcealDyn doctrine: punish->q_rep, anticipate->q_flee,
    hold->q_hold (normalised); tau from decisiveness; memory clamped to the game's w."""
    out = []
    for a in force["agents"]:
        d = a["doctrine"]
        tot = (d["punish_pattern"] + d["anticipate_flight"] + d["hold_static"]) or 1.0
        out.append(dict(q_rep=d["punish_pattern"] / tot, q_flee=d["anticipate_flight"] / tot,
                        q_hold=d["hold_static"] / tot, tau=TAU_BIN.get(
                            a.get("decisiveness", "balanced"), TAU),
                        w=int(np.clip(a.get("memory", W), 1, W))))
    return out


def score_force(base, pp, sites, doctrines):
    """(omniscient optimum, best blind rule, best OBSERVING rule, implied coverage)."""
    g = ConcealDyn(base, pp, sites, w=W, tau=TAU, doctrines=doctrines)
    sup = g.blind_supports()
    blind = min(g.episodic(rule=lambda i, m, p, M=np.asarray(g._anti(d), float): M, T=40)
                for d in sup.values())
    obs = min([blind] + [g.episodic_rule(d, anti_repeat=a, softness=s, topm=t, T=40)
                         for d in sup.values() for a in (False, True)
                         for s, t in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3), (0.0, 5))])
    zone = g.prior_j >= 0.05 * g.prior_j.max(axis=1, keepdims=True)
    cover = float((((base.expo.astype(int) @ zone.T.astype(int)) > 0).any(axis=1)).mean())
    return float(g.episodic(T=40)), float(blind), float(obs), cover


def random_force(rng, k=K):
    f = {"agents": []}
    for _ in range(k):
        d = rng.dirichlet([1, 1, 1])
        f["agents"].append({
            "archetype": str(rng.choice(["sniper_overwatch", "ambusher", "anticipator",
                                         "blocker", "forward_picket"])),
            "emplacement_zone": {"terrain": str(rng.choice(["open", "field", "forest", "urban"])),
                                 "region": str(rng.choice(["near_base", "mid_corridor",
                                                           "near_target_standoff", "chokepoint"]))},
            "doctrine": {"punish_pattern": float(d[0]), "anticipate_flight": float(d[1]),
                         "hold_static": float(d[2])},
            "decisiveness": str(rng.choice(["decisive", "balanced", "hedged"])),
            "memory": int(rng.integers(1, 3)),
            "team_id": 0, "team_role": str(rng.choice(["bait", "block", "cover", "anchor"])),
            "rationale": "random composition"})
    return f


def heuristic_force():
    """The pinned gen32 doctrine replicated; reach posture, regions spread (pinned pre-call)."""
    f = {"agents": []}
    for reg in ("near_base", "mid_corridor", "near_target_standoff"):
        f["agents"].append({
            "archetype": "blocker",
            "emplacement_zone": {"terrain": "open", "region": reg},
            "doctrine": {"punish_pattern": 0.6, "anticipate_flight": 0.2, "hold_static": 0.0},
            "decisiveness": "balanced", "memory": W, "team_id": 0, "team_role": "block",
            "rationale": "gen32 pinned doctrine, replicated"})
    return f


def heuristic_doctrines():
    return [dict(DOC32)] * K          # gen32 exactly, incl. q_ar (the regression-anchor path)


def generate_llm(arm, n, temperature=0.8):
    base, t, sc = narva_base()
    table = relabelled(t) if arm == "relabel" else t
    system, user = serialise_theatre(base.th, phase="coordinated", K=K,
                                     range_scale=sc * RM, terrain=table)
    schema = force_schema(t)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"brief_{arm}.txt").write_text(system + "\n\n---\n\n" + user)
    recs = []

    def one(model, j):
        txt, mode = g33.call_openai(BASE_URL, KEY, model, system, user, schema=schema,
                                    max_tokens=3000, temperature=temperature, timeout=900)
        obj = g33._extract_json(txt)
        errs = g33.validate_force(obj)
        return {"model": model, "arm": arm, "j": j, "mode": mode, "force": obj,
                "errors": errs, "raw": txt}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, m, j): (m, j) for m in MODELS for j in range(n)}
        for fu in as_completed(futs):
            m, j = futs[fu]
            try:
                r = fu.result()
            except Exception as e:                                   # noqa: BLE001
                r = {"model": m, "arm": arm, "j": j, "mode": "error", "force": None,
                     "errors": [str(e)], "raw": ""}
            recs.append(r)
            ok = "ok" if r["force"] and not r["errors"] else "BAD"
            print(f"  [{arm}] {m} #{j}: {ok} ({r['mode']})", flush=True)
    (OUTDIR / f"forces_{arm}.json").write_text(json.dumps(recs, indent=1))
    return recs


# --- scoring is pool-parallel: one task = (force-key, field) -----------------------------------
_CTX: dict = {}


def _pool_init():
    _CTX["base"], _CTX["t"], _CTX["sc"] = narva_base()


def _pool_task(spec):
    key, field, force, doctrines, kind = spec
    base = _CTX["base"]
    pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    if kind == "oracle":
        L, g, picker = choose_force(base, pp, force, K, np.random.default_rng(0),
                                    w=W, tau=TAU, doctrine=dict(q_rep=0.6, q_flee=0.2, q_ar=0.3))
        sup = g.blind_supports()
        blind = min(g.episodic(rule=lambda i, m, p, M=np.asarray(g._anti(d), float): M, T=40)
                    for d in sup.values())
        obs = min([blind] + [g.episodic_rule(d, anti_repeat=a, softness=s, topm=t2, T=40)
                             for d in sup.values() for a in (False, True)
                             for s, t2 in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3),
                                           (0.0, 5))])
        zone = g.prior_j >= 0.05 * g.prior_j.max(axis=1, keepdims=True)
        cover = float((((base.expo.astype(int) @ zone.T.astype(int)) > 0).any(axis=1)).mean())
        return key, field, (float(g.episodic(T=40)), float(blind), float(obs), cover,
                            [int(x) for x in L], picker)
    sites = place(force, base, pp)
    o, b, ob, cov = score_force(base, pp, sites, doctrines)
    return key, field, (o, b, ob, cov, [int(x) for x in sites], "")


def score_all():
    import multiprocessing as mp_
    specs = []
    forces = {}
    for arm in ("llm", "relabel"):
        p = OUTDIR / f"forces_{arm}.json"
        if not p.exists():
            continue
        for r in json.load(open(p)):
            if not r["force"] or r["errors"]:
                continue
            if len(r["force"].get("agents", [])) != K:      # the budget is part of the contract
                print(f"  [skip] {arm}:{r['model']}:{r['j']} emitted "
                      f"{len(r['force'].get('agents', []))} teams (need {K})")
                continue
            key = f"{arm}:{r['model']}:{r['j']}"
            forces[key] = (r["force"], doctrines_of(r["force"]), "llm")
    rng = np.random.default_rng(0)
    for j in range(N_RANDOM):
        f = random_force(rng)
        forces[f"random::{j}"] = (f, doctrines_of(f), "random")
    forces["heuristic::0"] = (heuristic_force(), heuristic_doctrines(), "heuristic")
    for kind in ("open", "hidden", "mixed"):
        forces[f"oracle:{kind}:0"] = (kind, None, "oracle")
    for key, (force, doc, kind) in forces.items():
        for field in FIELDS:
            specs.append((key, field, force, doc, kind))
    print(f"[score] {len(forces)} forces x {len(FIELDS)} fields = {len(specs)} tasks")
    with mp_.get_context("spawn").Pool(10, initializer=_pool_init) as P:
        res = {}
        for key, field, val in P.imap_unordered(_pool_task, specs,
                                                chunksize=max(1, len(specs) // 30)):
            res.setdefault(key, {})[field] = val
    (OUTDIR / "scores.json").write_text(json.dumps(res, indent=1))
    report(res)


def report(res):
    med = lambda ks, i: float(np.median([res[k][f][i] for k in ks for f in FIELDS]))  # noqa: E731
    per_field = lambda k, i: [float(np.median([res[k][f][i]])) for f in FIELDS]       # noqa: E731
    groups = {}
    for k in res:
        arm = k.split(":")[0] + (":" + k.split(":")[1] if k.startswith(("llm", "relabel"))
                                 else "")
        groups.setdefault(arm, []).append(k)
    print(f"\n{'arm':24s} {'n':>3s} {'vs OBSERVING (primary)':>22s} {'vs omniscient':>13s} "
          f"{'blind':>7s} {'coverage':>8s}")
    for arm, ks in sorted(groups.items()):
        print(f"{arm:24s} {len(ks):3d} {med(ks, 2):22.4f} {med(ks, 0):13.4f} "
              f"{med(ks, 1):7.4f} {med(ks, 3):8.2f}")
    heur = groups.get("heuristic", [])
    rnd = groups.get("random", [])
    if heur and rnd:
        h_pf = [np.median([res[k][f][2] for k in heur]) for f in FIELDS]
        r_mean = float(np.mean([res[k][f][2] for k in rnd for f in FIELDS]))
        for m in MODELS:
            ks = groups.get(f"llm:{m}", [])
            if not ks:
                continue
            l_pf = [np.median([res[k][f][2] for k in ks]) for f in FIELDS]
            wins = sum(1 for lf, hf in zip(l_pf, h_pf) if lf > hf)
            pooled = med(ks, 2) > med(heur, 2)
            above_rnd = med(ks, 2) > r_mean
            print(f"\n[BAR] {m}: beats heuristic {wins}/3 fields, pooled {'PASS' if pooled else 'FAIL'}, "
                  f"above random mean {'PASS' if above_rnd else 'FAIL'} "
                  f"(llm {med(ks, 2):.4f} vs heur {med(heur, 2):.4f} vs random-mean {r_mean:.4f})")
    for m in MODELS:                                     # the binding relabel control
        ks_n = groups.get(f"llm:{m}", [])
        ks_c = groups.get(f"relabel:{m}", [])
        if ks_n and ks_c:
            p = OUTDIR
            def terr_dist(arm):
                recs = json.load(open(p / f"forces_{arm}.json"))
                tt = [a["emplacement_zone"]["terrain"] for r in recs
                      if r["model"] == m and r["force"] and not r["errors"]
                      for a in r["force"]["agents"]]
                return {c: tt.count(c) / max(len(tt), 1) for c in ("open", "field", "forest",
                                                                   "urban")}
            print(f"[CONTROL] {m}: terrain choices normal {terr_dist('llm')} vs relabelled "
                  f"{terr_dist('relabel')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--big", action="store_true", help="Phase 1b: 32 forces per model")
    a = ap.parse_args()
    if a.dry:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        recs = [{"model": m, "arm": "llm", "j": j, "mode": "dry",
                 "force": dry_force(K, seed=j, coordinated=True), "errors": [], "raw": ""}
                for m in MODELS for j in range(2)]
        (OUTDIR / "forces_llm.json").write_text(json.dumps(recs, indent=1))
        score_all()
        return
    if a.big:                       # Phase 1b: 32 per model, no relabel arm (already banked)
        t0 = time.time()
        generate_llm("big", N_LLM_BIG)
        print(f"[big] generation done in {(time.time() - t0) / 60:.1f} min")
        return
    if a.live:
        t0 = time.time()
        generate_llm("llm", N_LLM)
        generate_llm("relabel", N_LLM)
        print(f"[live] generation done in {(time.time() - t0) / 60:.1f} min")
        score_all()
        return
    if a.score:
        score_all()
        return
    ap.error("give --dry / --live / --big / --score")


if __name__ == "__main__":
    main()
