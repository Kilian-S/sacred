#!/usr/bin/env python3
"""Scores the Narva-trained defenders on theatres they have never seen.

All twelve validation-selected checkpoints, four arms by three seeds, are scored on the other three
theatres. Every map gets its own strong test set built exactly as Narva's was, with four enemy
families each authored by its own 16-evaluation search at the same matched budget, plus the
oracle-searched force as the ceiling row. Doctrine is frozen throughout and nothing is retrained,
so what is measured is transfer. The self-check rebuilds Narva's test instances from the committed
curricula and re-evaluates a checkpoint through this harness, whose values must reproduce the
training run log before any map row can be trusted.

    PYTHONPATH=. python analysis/gen39_zeroshot.py --selfcheck
    PYTHONPATH=. python analysis/gen39_zeroshot.py --maps kgd_gvardeysk,ukraine,fulda
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from analysis.gen39_compose import BASE_URL, K, KEY, g33
from analysis.gen39_phase1f import SCHEMA, llm_prompt, map_digest
from scripts.train_gen39_conceal import (DOC32, Inst, N, TAU, TEST_FIELDS, W, _mm, policy_value)
from src.agents.sac import ProtagonistSAC
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, choose_force, resample_field
from src.envs.aerial_theatre_env import TheatreEnv
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

CR, RM, BUDGET, T_MISSION = 0.85, 0.7, 16, 40
PATH = "data/maps/theatre_%s_vec.json"
ARMS = ("llm16", "local16", "random16", "tuned")
STEP5 = Path("models/runs/gen39_step5")
OUT = Path("models/runs/gen39_zeroshot.json")


def base_for(name):
    """Builds a theatre exactly as the trainer does, parameterised by map.

    Includes the public-exposure head column the policy reads.
    """
    sc = lateral_width(load_vec_theatre(PATH % name)) / lateral_width(
        load_vec_theatre(PATH % "kgd_gvardeysk"))
    base = ConcealBase(PATH % name, terrain=terrain_v2(hidden_leth=1.0, conceal_reach=CR),
                       range_scale=sc * RM, spacing_km=2.0 * sc, standoff_km=4.0 * sc,
                       n_sites=200)
    S_pub = base.survival(base.pp_base)
    base.expo_pub = _mm((1.0 - S_pub ** N).max(axis=1))
    return base


# --- per-map 16-evaluation searches (the four families) ------------------------------------------
_CTX: dict = {}


def _init(name):
    _CTX["base"] = base_for(name)


def _eval(spec):
    sites, field = spec
    base = _CTX["base"]
    pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    g = ConcealDyn(base, pp, np.asarray(sites, int), w=W, tau=TAU, **DOC32)
    return tuple(sites), float(g.episodic(T=T_MISSION))


def score(pool, triples, field):
    return dict(pool.imap_unordered(_eval, [(t, field) for t in triples], chunksize=2))


def search_llm(base, digest, pool, field):
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
        hist += list(score(pool, tri, field).items())
    return hist


def search_local(base, pool, field, rng):
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
        got = score(pool, cands, field); used += len(cands); hist += list(got.items())
        b = max(got, key=got.get)
        if got[b] > v:
            cur, v = b, got[b]
        else:
            cur = tuple(sorted(int(x) for x in rng.choice(base.H, K, replace=False)))
            v = score(pool, [cur], field)[cur]; used += 1; hist.append((cur, v))
    return hist


def search_random(base, pool, field, rng):
    tri = [tuple(sorted(int(x) for x in rng.choice(base.H, K, replace=False)))
           for _ in range(BUDGET)]
    return list(score(pool, list(dict.fromkeys(tri)), field).items())


def build_test_set(name, base, pool):
    """Six pristine fields, each with the best force of every family plus the oracle ceiling."""
    pp0 = base.lethality(resample_field(base.coords, TEST_FIELDS[0]), hidden_leth=1.0)
    digest = map_digest(base, pp0)
    cells = []
    for f in TEST_FIELDS:
        rng = np.random.default_rng(f)
        pp = base.lethality(resample_field(base.coords, f), hidden_leth=1.0)
        best = {}
        best["llm16"] = max(search_llm(base, digest, pool, f), key=lambda x: x[1])
        best["local16"] = max(search_local(base, pool, f, rng), key=lambda x: x[1])
        best["random16"] = max(search_random(base, pool, f, rng), key=lambda x: x[1])
        tn = []
        for kind in ("open", "hidden", "mixed"):
            L, g, _ = choose_force(base, pp, kind, K, np.random.default_rng(f),
                                   w=W, tau=TAU, doctrine=DOC32)
            tn.append((tuple(int(x) for x in L), float(g.episodic(T=T_MISSION))))
        best["tuned"] = max(tn, key=lambda x: x[1])
        cell = [Inst(base, f"{name}-te{f}-{fam}", f, sites=list(best[fam][0])).refs()
                for fam in ARMS]
        cell.append(Inst(base, f"{name}-te{f}-oracle", f,
                         sites=list(max(best.values(), key=lambda x: x[1])[0])).refs())
        cells.append(cell)
        print(f"    field {f}: " + " ".join(f"{a} {best[a][1]:.4f}" for a in ARMS), flush=True)
    return cells


def load_ckpts():
    out = []
    for arm in ARMS:
        for s in (0, 1, 2):
            p = STEP5 / f"{arm}_seed{s}.json"
            if not p.exists():
                continue
            run = json.loads(p.read_text())
            srt = min(run["history"], key=lambda h: h["val"])["sortie"]
            ck = STEP5 / f"{arm}_seed{s}_ckpts" / f"actor_ep{srt}.pt"
            if not ck.exists():
                continue
            prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2,
                                  heads=4, reward_scale=1.0, device="cpu", role_alpha=True)
            prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
            prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(3))
            prot.actor.route_feats = None
            prot.actor.load_state_dict(torch.load(ck, weights_only=True))
            out.append((arm, s, prot))
    return out


def selfcheck():
    """Re-evaluates one checkpoint on Narva's test instances rebuilt from the committed curricula.

    The values must match the training run log, otherwise the harness is wrong.
    """
    base = base_for("narva")
    cur = json.loads((STEP5 / "curricula.json").read_text())
    run = json.loads((STEP5 / "llm16_seed0.json").read_text())
    b = min(run["history"], key=lambda h: h["val"])
    ck = STEP5 / f"llm16_seed0_ckpts" / f"actor_ep{b['sortie']}.pt"
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, device="cpu", role_alpha=True)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(3))
    prot.actor.route_feats = None
    prot.actor.load_state_dict(torch.load(ck, weights_only=True))
    env, got = None, []
    for f in TEST_FIELDS:
        cell = []
        for fam in ARMS:
            it = Inst(base, f"te{f}-{fam}", f, sites=cur[fam][str(f)][0][0])
            if env is None:
                env = TheatreEnv(base.menu, it.g.game, it.S_field, N=N)
            cell.append(policy_value(prot, it, env))
        pp = base.lethality(resample_field(base.coords, f), hidden_leth=1.0)
        bb = None
        for kind in ("open", "hidden", "mixed"):
            it = Inst(base, f"te{f}-oracle({kind})", f, archetype=kind)
            v = it.g.episodic(T=40)
            if bb is None or v > bb[0]:
                bb = (v, it)
        cell.append(policy_value(prot, bb[1], env))
        got.append(float(np.mean(cell)))
    ref = b["cells"]
    print("  harness :", " ".join(f"{x:.4f}" for x in got))
    print("  step-5  :", " ".join(f"{x:.4f}" for x in ref))
    d = max(abs(a - c) for a, c in zip(got, ref))
    print(f"  max abs difference {d:.5f} -> {'OK' if d < 5e-3 else 'MISMATCH'}")
    return d < 5e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="kgd_gvardeysk,ukraine,fulda")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        raise SystemExit(0 if selfcheck() else 1)
    import multiprocessing as mp_
    ckpts = load_ckpts()
    print(f"[zs] {len(ckpts)} validation-selected checkpoints loaded\n")
    result = {}
    t0 = time.time()
    for name in a.maps.split(","):
        print(f"[{name}] building the strong test set ({(time.time()-t0)/60:.1f} min)", flush=True)
        base = base_for(name)
        with mp_.get_context("spawn").Pool(9, initializer=_init, initargs=(name,)) as P:
            cells = build_test_set(name, base, P)
        env = TheatreEnv(base.menu, cells[0][0].g.game, cells[0][0].S_field, N=N)
        rows: dict = {}
        for arm, s, prot in ckpts:
            per_cell = [float(np.mean([policy_value(prot, it, env) for it in cell]))
                        for cell in cells]
            rows.setdefault(arm, []).append(per_cell)
        refs = dict(cap=[float(np.mean([it.cap for it in c])) for c in cells],
                    obs=[float(np.mean([it.obs_ref for it in c])) for c in cells],
                    opt=[float(np.mean([it.opt for it in c])) for c in cells])
        result[name] = dict(rows={k: v for k, v in rows.items()}, refs=refs)
        pooled = {k: float(np.mean(v)) for k, v in rows.items()}
        print(f"[{name}] " + "  ".join(f"{k} {pooled[k]:.4f}" for k in ARMS)
              + f"   ({(time.time()-t0)/60:.1f} min)", flush=True)
        OUT.write_text(json.dumps(result, indent=1))

    print(f"\n{'=' * 92}\nZERO-SHOT: Narva-trained defenders on unseen theatres "
          f"(pooled over 3 seeds x 6 fields; lower = better)\n{'=' * 92}")
    print(f'{"map":16s} ' + " ".join(f'{a:>10s}' for a in ARMS)
          + f' | {"llm16 vs tuned":>15s} {"llm16 vs local16":>17s}')
    for name, r in result.items():
        p = {k: np.mean(v) for k, v in r["rows"].items()}
        seeds_l = np.mean(r["rows"]["llm16"], axis=1)
        seeds_t = np.mean(r["rows"]["tuned"], axis=1)
        seeds_c = np.mean(r["rows"]["local16"], axis=1)
        print(f'{name:16s} ' + " ".join(f'{p[a]:10.4f}' for a in ARMS)
              + f' | {int((seeds_l < seeds_t).sum())}/3 seeds     '
                f'{int((seeds_l < seeds_c).sum())}/3 seeds')
    print(f"\n[written] {OUT}  [{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
