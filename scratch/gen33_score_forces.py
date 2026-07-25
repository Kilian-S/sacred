"""gen33_llm_adversary METRIC 1 + DIVERSITY: score the banked LLM forces against the anchored
ladder (oracle-only; semantics + anchors pinned by scratch/gen33_score_screen.py BEFORE this ran).

Per (model, theatre, phase): each of the 8 forces scored as mean best-response damage over the
3 pinned field seeds at sigma0=8km (theatre-scaled); population mean/sd/min/max vs the anchor
rows; BAR (pre-registered) = population mean > heuristic on >= 2/3 theatres AND pooled, both
phases, per model. Diversity: normalised archetype entropy, mean pairwise doctrine L1, mean
pairwise placement distance (vs the random-floor reference). NO pooling across models.
"""
import itertools
import json
import time

import numpy as np

from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre
from src.redforce_score import ScoreBase, force_value, random_force

THEATRES = {
    "kgd": "data/maps/theatre_kgd_gvardeysk_vec.json",
    "ukraine": "data/maps/theatre_ukraine_vec.json",
    "narva": "data/maps/theatre_narva_vec.json",
}
MODELS = ("llama-3.3-70b", "qwen3-27b")
PHASES = ("single", "coordinated")
SEEDS = (5100, 5101, 5102)
SIGMA0 = 8.0
LAT_REF = lateral_width(load_vec_theatre(THEATRES["kgd"]))
SCREEN = json.load(open("models/runs/gen33_score_screen.json"))


def cell_scores(base, model, name, phase):
    art = json.load(open(f"models/runs/gen33_forces/force_{model}_{name}_{phase}.json"))
    sigma = SIGMA0 * base.scale
    vals, agents, sites_all = [], [], []
    for rec in art["forces"]:
        sites = rec["resolved"]["sites"]
        doctrine = [tuple(d) for d in rec["resolved"]["doctrine"]]
        v = float(np.mean([force_value(base.field(sd), sites, doctrine, sigma) for sd in SEEDS]))
        vals.append(v)
        agents += rec["force"]["agents"]
        sites_all += list(sites)
    return np.array(vals), agents, sites_all


def diversity(base, agents, sites_all, K):
    cnt = np.array([sum(a["archetype"] == k for a in agents) for k in
                    ("sniper_overwatch", "ambusher", "anticipator", "blocker", "forward_picket")],
                   dtype=float)
    p = cnt / cnt.sum()
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum() / np.log(5))
    qs = np.array([[a["doctrine"]["punish_pattern"], a["doctrine"]["anticipate_flight"],
                    a["doctrine"]["hold_static"]] for a in agents], dtype=float)
    qs = qs / np.clip(qs.sum(axis=1, keepdims=True), 1e-9, None)
    dl1 = float(np.mean([np.abs(a - b).sum() for a, b in itertools.combinations(qs, 2)]))
    xy = base.coords[np.array(sites_all)]
    pd = float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(xy, 2)]))
    rng = np.random.default_rng(7)
    rxy = base.coords[np.concatenate([random_force(base, K, rng)[0] for _ in range(16)])]
    rpd = float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(rxy, 2)]))
    return dict(arch_entropy=ent, doctrine_l1=dl1, placement_km=pd, placement_random_km=rpd)


if __name__ == "__main__":
    t0 = time.time()
    bases = {n: ScoreBase(p, lat_ref=None if n == "kgd" else LAT_REF)
             for n, p in THEATRES.items()}
    out = {"sigma0": SIGMA0, "seeds": list(SEEDS), "cells": {}, "verdicts": {}}
    for model in MODELS:
        for phase in PHASES:
            wins, pool_llm, pool_heur, pool_rand = 0, [], [], []
            for name in THEATRES:
                anc = SCREEN["anchors"][name][phase]
                vals, agents, sites_all = cell_scores(bases[name], model, name, phase)
                K = 1 if phase == "single" else 3
                div = diversity(bases[name], agents, sites_all, K)
                beat = bool(vals.mean() > anc["heuristic"])
                wins += beat
                pool_llm.append(vals.mean()); pool_heur.append(anc["heuristic"])
                pool_rand.append(anc["random_mean"])
                out["cells"][f"{model}|{name}|{phase}"] = dict(
                    mean=float(vals.mean()), sd=float(vals.std()), min=float(vals.min()),
                    max=float(vals.max()), values=[float(v) for v in vals],
                    heuristic=anc["heuristic"], random_mean=anc["random_mean"],
                    oracle=anc["oracle"], beats_heuristic=beat, **div)
                print(f"{model:14s} {name:8s} {phase:11s}: pop {vals.mean():.4f}+/-{vals.std():.4f} "
                      f"[{vals.min():.4f},{vals.max():.4f}] | heur {anc['heuristic']:.4f} "
                      f"rand {anc['random_mean']:.4f} orac {anc['oracle']:.4f} | "
                      f"{'BEATS-HEUR' if beat else 'below'} | ent={div['arch_entropy']:.2f} "
                      f"dL1={div['doctrine_l1']:.2f} pkm={div['placement_km']:.1f}"
                      f"/{div['placement_random_km']:.1f}", flush=True)
            pooled = bool(np.mean(pool_llm) > np.mean(pool_heur))
            out["verdicts"][f"{model}|{phase}"] = dict(
                theatre_wins=int(wins), pooled_llm=float(np.mean(pool_llm)),
                pooled_heuristic=float(np.mean(pool_heur)),
                pooled_random=float(np.mean(pool_rand)), pooled_beats=pooled,
                bar=bool(wins >= 2 and pooled))
            print(f"  -> {model} {phase}: theatres {wins}/3, pooled "
                  f"{np.mean(pool_llm):.4f} vs heur {np.mean(pool_heur):.4f} "
                  f"(rand {np.mean(pool_rand):.4f}) => "
                  f"{'BAR MET' if wins >= 2 and pooled else 'BAR NOT MET'}", flush=True)
    json.dump(out, open("models/runs/gen33_force_scores.json", "w"), indent=1)
    print(f"[written] models/runs/gen33_force_scores.json [{time.time()-t0:.0f}s]")
