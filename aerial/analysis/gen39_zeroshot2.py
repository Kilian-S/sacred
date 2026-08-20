#!/usr/bin/env python3
"""Rebuilds the unseen-map test sets with their laydowns saved and scores all fifteen checkpoints
on them.

`gen39_zeroshot.py` saved scores but not the laydowns it scored, and the LLM search is stochastic,
so its per-map test sets cannot be reconstructed or extended. Each unseen map's test set is
therefore rebuilt once with laydowns persisted, all fifteen checkpoints are scored on the fresh
sets, and comparisons are paired only within them. The test families stay the original four plus
the oracle ceiling row, every search function is imported from the earlier harness rather than
re-implemented, and the build is incremental per (map, family), so nothing already saved is
recomputed.

    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen39_zeroshot2.py --selfcheck
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen39_zeroshot2.py --build \
        --maps kgd_gvardeysk,ukraine,fulda --families local16,random16,tuned --workers 6
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen39_zeroshot2.py --build \
        --maps kgd_gvardeysk,ukraine,fulda --families llm16
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen39_zeroshot2.py --score --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from analysis.gen39_zeroshot import (ARMS, STEP5, _init, base_for, load_ckpts, search_llm,
                                    search_local, search_random, selfcheck)
from analysis.gen39_phase1f import map_digest
from scripts.train_gen39_conceal import (DOC32, Inst, N, TAU, TEST_FIELDS, W, policy_value)
from src.envs.aerial_conceal import choose_force, resample_field
from src.envs.aerial_theatre_env import TheatreEnv

T_MISSION = 40
DEF_ARMS = ARMS + ("qwenthink16",)          # 15 defenders: 5 arms x 3 seeds
BUILD = Path("models/runs/gen39_step5/zeroshot2_build.json")
OUT = Path("models/runs/gen39_step5/zeroshot2.json")


def _load(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def build(maps, families, workers):
    """Per-(map, field, family) best laydown + value, persisted after every field."""
    art = _load(BUILD, {})
    for name in maps:
        art.setdefault(name, {})
        base = base_for(name)
        pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(name,))
        try:
            digest = None
            for f in TEST_FIELDS:
                key = str(f)
                art[name].setdefault(key, {})
                todo = [fam for fam in families if fam not in art[name][key]]
                if not todo:
                    print(f"  {name} field {f}: already stored {sorted(art[name][key])}",
                          flush=True)
                    continue
                t0 = time.time()
                rng = np.random.default_rng(f)
                pp = base.lethality(resample_field(base.coords, f), hidden_leth=1.0)
                for fam in todo:
                    if fam == "llm16":
                        if digest is None:
                            pp0 = base.lethality(resample_field(base.coords, TEST_FIELDS[0]),
                                                 hidden_leth=1.0)
                            digest = map_digest(base, pp0)
                        hist = search_llm(base, digest, pool, f)
                    elif fam == "local16":
                        hist = search_local(base, pool, f, rng)
                    elif fam == "random16":
                        hist = search_random(base, pool, f, rng)
                    elif fam == "tuned":
                        hist = []
                        for kind in ("open", "hidden", "mixed"):
                            L, g, _ = choose_force(base, pp, kind, 3, np.random.default_rng(f),
                                                   w=W, tau=TAU, doctrine=DOC32)
                            hist.append((tuple(int(x) for x in L), float(g.episodic(T=T_MISSION))))
                    else:
                        raise SystemExit(f"unknown family {fam}")
                    if not hist:
                        print(f"  {name} field {f} {fam}: EMPTY SEARCH (recorded as null)",
                              flush=True)
                        art[name][key][fam] = None
                        continue
                    sites, val = max(hist, key=lambda x: x[1])
                    art[name][key][fam] = dict(sites=[int(x) for x in sites], value=float(val),
                                               n_evaluated=len(hist))
                BUILD.parent.mkdir(parents=True, exist_ok=True)
                BUILD.write_text(json.dumps(art, indent=1))
                done = {k: (v["value"] if v else None) for k, v in art[name][key].items()}
                print(f"  {name} field {f}: " + " ".join(
                    f"{k} {v:.4f}" for k, v in done.items() if v is not None)
                    + f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
        finally:
            pool.close()
            pool.join()
    print(f"[written] {BUILD}")


def score(workers, out_path=None, only_maps=None):
    """Scores all fifteen checkpoints on the fresh sets.

    A cell is the mean over the four family instances plus the oracle-ceiling instance.
    ``out_path`` lets one process per map run concurrently without racing the artefact; the
    per-map files are merged by --merge.
    """
    OUT = Path(out_path) if out_path else globals()["OUT"]
    art = _load(BUILD, {})
    maps = [m for m in art if all(
        all(f in art[m].get(str(fl), {}) and art[m][str(fl)][f] for f in ARMS)
        for fl in TEST_FIELDS)]
    if only_maps:
        maps = [m for m in maps if m in only_maps]
    if not maps:
        raise SystemExit("no (selected) map has all four families stored yet; run --build first")
    print(f"scoring maps: {maps}")
    ckpts = load_ckpts_all()
    print(f"checkpoints loaded: {len(ckpts)}")
    out = _load(OUT, {})
    for name in maps:
        base = base_for(name)
        env = None
        rows = out.setdefault(name, {})
        for arm, s, prot in ckpts:
            tag = f"{arm}_seed{s}"
            if tag in rows:
                continue
            t0 = time.time()
            cells = []
            for f in TEST_FIELDS:
                cell = []
                for fam in ARMS:
                    rec = art[name][str(f)][fam]
                    it = Inst(base, f"{name}-te{f}-{fam}", f, sites=list(rec["sites"]))
                    if env is None:
                        env = TheatreEnv(base.menu, it.g.game, it.S_field, N=N)
                    cell.append(policy_value(prot, it, env))
                bestfam = max((a for a in ARMS), key=lambda a: art[name][str(f)][a]["value"])
                it = Inst(base, f"{name}-te{f}-oracle", f,
                          sites=list(art[name][str(f)][bestfam]["sites"]))
                cell.append(policy_value(prot, it, env))
                cells.append(float(np.mean(cell)))
            rows[tag] = dict(cells=cells, mean=float(np.mean(cells)))
            OUT.write_text(json.dumps(out, indent=1))
            print(f"  {name} {tag}: {np.mean(cells):.4f}  "
                  + " ".join(f"{c:.4f}" for c in cells) + f"  [{(time.time()-t0)/60:.1f} min]",
                  flush=True)
    print(f"[written] {OUT}")


def load_ckpts_all():
    """Extends `load_ckpts`, which covers four arms, with the qwenthink16 seeds."""
    import torch

    from src.agents.sac import ProtagonistSAC
    out = list(load_ckpts())
    for s in (0, 1, 2):
        p = STEP5 / f"qwenthink16_seed{s}.json"
        if not p.exists():
            print(f"  [skip] qwenthink16 seed {s}: no run json yet")
            continue
        run = json.loads(p.read_text())
        srt = min(run["history"], key=lambda h: h["val"])["sortie"]
        ck = STEP5 / f"qwenthink16_seed{s}_ckpts" / f"actor_ep{srt}.pt"
        if not ck.exists():
            print(f"  [skip] qwenthink16 seed {s}: checkpoint {ck.name} missing")
            continue
        prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2,
                              heads=4, reward_scale=1.0, device="cpu", role_alpha=True)
        prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        prot.actor.route_feats = None
        prot.actor.load_state_dict(torch.load(ck, weights_only=True))
        out.append(("qwenthink16", s, prot))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--maps", default="kgd_gvardeysk,ukraine,fulda")
    ap.add_argument("--families", default="local16,random16,tuned")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None, help="per-map score file (avoids concurrent races)")
    ap.add_argument("--merge", action="store_true", help="merge per-map score files into OUT")
    a = ap.parse_args()
    if a.selfcheck:
        raise SystemExit(0 if selfcheck() else 1)
    if a.build:
        build([m for m in a.maps.split(",") if m],
              [f for f in a.families.split(",") if f], a.workers)
    if a.score:
        score(a.workers, a.out, [m for m in a.maps.split(',') if m])
    if a.merge:
        merged = _load(OUT, {})
        for p in sorted(OUT.parent.glob("zeroshot2_score_*.json")):
            for m, rows in json.loads(p.read_text()).items():
                merged.setdefault(m, {}).update(rows)
        OUT.write_text(json.dumps(merged, indent=1))
        print(f"[merged] {OUT}: " + ", ".join(f"{m} {len(r)} ckpts" for m, r in merged.items()))
    if not (a.build or a.score or a.merge):
        ap.error("choose --selfcheck, --build, --score or --merge")


if __name__ == "__main__":
    main()
