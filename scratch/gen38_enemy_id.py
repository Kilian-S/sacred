#!/usr/bin/env python3
"""gen38 Step V1 (pre-registered: experiments/gen38_llm_enemy_id.md): does an LLM classify the
enemy doctrine from a behavioural intelligence narrative well enough that deploying the matching
exact counter crosses the gen34 type-blind wall?

Oracle-exact operational eval, no training. Per held-out Gdansk cell (gen34 pool): build the 5
members' cost matrices + exact specialists (dyn_exact), pull blind_cap/omni_cap. Classify each
of 20 narratives with (a) the LLM (3 draws, temp 0.2), (b) the keyword control, (c) random
(analytic). The assisted defender plays specialist[predicted] vs the true type; value =
mean over narratives of policy_value_exact(specialist[pred], cost[true]).

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen38_enemy_id.py
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from scratch.dyn_exact import build_window_mdp, greedy_policy_from_rvi, karp_mmc, policy_value_exact
from scratch.gen34_family_probe import member_fns
from scratch.gen38_narratives import (
    DOCTRINE_BRIEF, MEMBERS, NARRATIVES, all_labelled, keyword_classify)
from scripts.train_b1lite1 import stacked_L
from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import _row_minimiser

torch.set_num_threads(1)
N, K, BAND, KX, W, TAU = 3, 1, (0.15, 0.95), 8, 3, 0.15
BASE, KEY, MODEL = "http://localhost:18080/v1/chat/completions", "iits-local-key", "llama-3.3-70b"
DRAWS = 3
OUT = Path("models/runs/gen38_llm_enemy_id")
TR = OUT / "transcripts"


def instance_apparatus(env):
    L = stacked_L(env.game, N)
    R = L.shape[0]
    _, eq = _row_minimiser(L)
    fns = member_fns(L, eq)
    n, pw = R ** W, R ** (W - 1)
    costs, spec, omni = {}, {}, {}
    for nm in MEMBERS:
        c = build_window_mdp(L, TAU, W, member_fn=fns[nm])[0]
        costs[nm] = c
        spec[nm] = greedy_policy_from_rvi(c, n, R, pw)
        omni[nm] = karp_mmc(c, n, R, pw)
    mix = np.mean([costs[nm] for nm in MEMBERS], axis=0)
    blind_policy = greedy_policy_from_rvi(mix, n, R, pw)
    blind_cap = karp_mmc(mix, n, R, pw)
    return dict(costs=costs, spec=spec, omni=omni, blind_policy=blind_policy,
                blind_cap=blind_cap, omni_cap=float(np.mean(list(omni.values()))),
                n=n, R=R, pw=pw)


def op_value(app, pred_type, true_type):
    return policy_value_exact(app["spec"][pred_type], app["costs"][true_type],
                              app["n"], app["R"], app["pw"])


def op_value_hedge(app, pred_type, conf, true_type, thresh=0.5):
    pol = app["spec"][pred_type] if conf >= thresh else app["blind_policy"]
    return policy_value_exact(pol, app["costs"][true_type], app["n"], app["R"], app["pw"])


def llm_classify(narr):
    brief = "\n".join(f"- {DOCTRINE_BRIEF[m]}" for m in MEMBERS)
    prompt = ("You are an intelligence analyst. Below is your doctrine catalogue of five enemy "
              "behaviours, then a field assessment of the enemy we face. Identify which single "
              "doctrine best matches the assessment.\n\nDOCTRINE CATALOGUE:\n" + brief +
              "\n\nFIELD ASSESSMENT:\n" + narr + "\n\nReturn STRICT JSON only: "
              '{"type": "<one of reactive|sharp|anticipatory|doctrine|scattergun>", '
              '"confidence": <0..1>, "reasoning": "<one sentence>"}. No other text.')
    body = json.dumps(dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                           temperature=0.2, max_tokens=300)).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    s, e = content.find("{"), content.rfind("}")
    obj = json.loads(content[s:e + 1])
    typ = str(obj.get("type", "")).strip().lower()
    if typ not in MEMBERS:
        typ = next((m for m in MEMBERS if m in typ), "reactive")
    conf = float(obj.get("confidence", 0.5))
    return typ, conf, content


def main():
    TR.mkdir(parents=True, exist_ok=True)
    test = sample_instances(6, N, K, BAND, KX, 0, city="gdansk")
    apps = {}
    for it in test:
        key = f"{it.city}:{it.od[0]}-{it.od[1]}"
        apps[key] = instance_apparatus(it.env)
        print(f"[apparatus] {key}: blind_cap {apps[key]['blind_cap']:.4f} "
              f"omni_cap {apps[key]['omni_cap']:.4f}", flush=True)

    labelled = all_labelled()  # (type, idx, narrative)

    # --- classify: LLM (DRAWS each) + keyword ---
    llm_preds, kw_preds = [], []
    for (true_type, idx, narr) in labelled:
        kw_preds.append((true_type, keyword_classify(narr)))
        draws = []
        for d in range(DRAWS):
            for attempt in range(3):
                try:
                    typ, conf, raw = llm_classify(narr)
                    draws.append(dict(pred=typ, conf=conf, raw=raw[:800]))
                    break
                except Exception as ex:  # noqa: BLE001
                    if attempt == 2:
                        draws.append(dict(pred="reactive", conf=0.0, error=str(ex)))
                    time.sleep(2)
        (TR / f"{true_type}_{idx}.json").write_text(json.dumps(
            {"true": true_type, "narrative": narr, "draws": draws}, indent=1))
        for dr in draws:
            llm_preds.append((true_type, dr["pred"], dr["conf"]))
        print(f"  {true_type}#{idx}: LLM {[d['pred'] for d in draws]} kw {kw_preds[-1][1]}",
              flush=True)

    # --- accuracy + confusion ---
    def acc(pairs):
        return float(np.mean([t == p for t, p in pairs]))
    llm_acc = acc([(t, p) for t, p, _ in llm_preds])
    kw_acc = acc(kw_preds)
    conf = {a: {b: 0 for b in MEMBERS} for a in MEMBERS}
    for t, p, _ in llm_preds:
        conf[t][p] += 1

    # --- operational value, pooled over held-out cells ---
    def pooled_op(pred_fn):
        per_cell = []
        for name, app in apps.items():
            vals = [pred_fn(app, t, p, c) for (t, p, c) in _iter_preds()]
            per_cell.append(float(np.mean(vals)))
        return per_cell

    def _iter_preds():
        return [(t, p, c) for t, p, c in llm_preds]

    llm_commit = {name: float(np.mean([op_value(app, p, t) for t, p, c in llm_preds]))
                  for name, app in apps.items()}
    llm_hedge = {name: float(np.mean([op_value_hedge(app, p, c, t) for t, p, c in llm_preds]))
                 for name, app in apps.items()}
    kw_commit = {name: float(np.mean([op_value(app, p, t) for t, p in kw_preds]))
                 for name, app in apps.items()}
    # random classifier: analytic expectation = mean over pred of specialist[pred] vs true,
    # averaged uniformly over the 5 preds and over true-type frequency (uniform, 4 each)
    rand_commit = {}
    for name, app in apps.items():
        tot = 0.0
        for t in MEMBERS:
            for p in MEMBERS:
                tot += op_value(app, p, t)
        rand_commit[name] = tot / (len(MEMBERS) ** 2)
    omni = {name: app["omni_cap"] for name, app in apps.items()}
    blind = {name: app["blind_cap"] for name, app in apps.items()}

    cells = list(apps)
    def pooled(dd):
        return float(np.mean([dd[c] for c in cells]))
    crosses = sum(1 for c in cells if llm_commit[c] < blind[c])
    kw_crosses = sum(1 for c in cells if kw_commit[c] < blind[c])

    result = dict(
        llm_accuracy=llm_acc, keyword_accuracy=kw_acc, draws=DRAWS,
        confusion=conf,
        pooled=dict(blind_cap=pooled(blind), omni_cap=pooled(omni),
                    llm_commit=pooled(llm_commit), llm_hedge=pooled(llm_hedge),
                    keyword_commit=pooled(kw_commit), random_commit=pooled(rand_commit)),
        per_cell=dict(blind=blind, omni=omni, llm_commit=llm_commit, llm_hedge=llm_hedge,
                      keyword_commit=kw_commit, random_commit=rand_commit),
        llm_crosses_blind=f"{crosses}/6", keyword_crosses_blind=f"{kw_crosses}/6")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v1_result.json").write_text(json.dumps(result, indent=1))

    print("\n=== gen38 V1 RESULT ===")
    print(f"classification accuracy: LLM {llm_acc:.3f} vs keyword {kw_acc:.3f} vs random 0.200")
    print("confusion (rows=true, cols=LLM pred):")
    print("           " + " ".join(f"{m[:5]:>6}" for m in MEMBERS))
    for t in MEMBERS:
        print(f"  {t:10s} " + " ".join(f"{conf[t][p]:>6d}" for p in MEMBERS))
    p = result["pooled"]
    print(f"\npooled operational value (lower=better):")
    print(f"  blind_cap (the WALL)     {p['blind_cap']:.4f}")
    print(f"  omni_cap  (perfect ID)   {p['omni_cap']:.4f}")
    print(f"  LLM commit-to-argmax     {p['llm_commit']:.4f}  (crosses wall on "
          f"{crosses}/6 cells; PRIMARY bar >=4/6)")
    print(f"  LLM confidence-hedged    {p['llm_hedge']:.4f}")
    print(f"  keyword commit           {p['keyword_commit']:.4f}  (crosses {kw_crosses}/6)")
    print(f"  random commit            {p['random_commit']:.4f}")
    strong = p['llm_commit'] <= p['omni_cap'] * 1.15
    print(f"\nPRIMARY (LLM commit < blind on >=4/6): {'PASS' if crosses >= 4 else 'FAIL'} "
          f"({crosses}/6)")
    print(f"STRONG (LLM commit <= 1.15x omni_cap): {'PASS' if strong else 'no'}")
    print(f"COMPARATIVE (LLM acc>kw AND LLM val<kw val): "
          f"{'PASS' if (llm_acc > kw_acc and p['llm_commit'] < p['keyword_commit']) else 'no'}")
    print(f"\nwrote {OUT}/v1_result.json + transcripts/")


if __name__ == "__main__":
    main()
