#!/usr/bin/env python3
"""gen38 V1 ROBUSTNESS row (ungated, disclosed; the skeptic's attack on the clean 100%).

V1 hit 100% on clean author-written narratives. This probes whether that is fragile by
degrading the intel to messy/realistic forms, PROGRAMMATICALLY (not re-authored, to avoid
gaming):
  - TERSE:      first sentence only (minimal signal).
  - DISTRACTOR: the full narrative + one CONFLICTING sentence lifted from a DIFFERENT
                doctrine's narrative (contradictory intel, the realistic case). The true label
                is still the base narrative's type; a surface pattern-matcher should break.
  - BOTH:       terse + a distractor sentence.
Accuracy under each condition, LLM vs keyword, + the operational value (pooled held-out) so the
fragility is priced against the gen34 wall.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen38_robustness.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from scratch.gen38_enemy_id import instance_apparatus, llm_classify, op_value
from scratch.gen38_narratives import MEMBERS, NARRATIVES, keyword_classify
from scripts.train_generalist import sample_instances

N, K, BAND, KX = 3, 1, (0.15, 0.95), 8
DRAWS = 2
OUT = Path("models/runs/gen38_llm_enemy_id")
rng = np.random.default_rng(38)


def perturb(true_type, idx, kind):
    base = NARRATIVES[true_type][idx]
    first = base.split(". ")[0].strip().rstrip(".") + "."
    if kind == "terse":
        return first
    # distractor: one sentence from a DIFFERENT type
    others = [m for m in MEMBERS if m != true_type]
    dt = others[int(rng.integers(len(others)))]
    ds = NARRATIVES[dt][int(rng.integers(len(NARRATIVES[dt])))].split(". ")[0].strip()
    ds = ds.rstrip(".") + "."
    if kind == "distractor":
        return base + " However, one report adds: " + ds
    return first + " However, one report adds: " + ds  # both


def classify_set(kind):
    llm_preds, kw_preds = [], []
    recs = []
    for true_type in MEMBERS:
        for idx in range(len(NARRATIVES[true_type])):
            text = perturb(true_type, idx, kind)
            kw_preds.append((true_type, keyword_classify(text)))
            draws = []
            for _ in range(DRAWS):
                for attempt in range(3):
                    try:
                        typ, conf, _ = llm_classify(text)
                        draws.append((typ, conf)); break
                    except Exception as ex:  # noqa: BLE001
                        if attempt == 2:
                            draws.append(("reactive", 0.0))
                        time.sleep(2)
            for typ, conf in draws:
                llm_preds.append((true_type, typ, conf))
            recs.append(dict(true=true_type, kind=kind, text=text,
                             llm=[d[0] for d in draws], kw=kw_preds[-1][1]))
    return llm_preds, kw_preds, recs


def main():
    test = sample_instances(6, N, K, BAND, KX, 0, city="gdansk")
    apps = {f"{it.city}:{it.od[0]}-{it.od[1]}": instance_apparatus(it.env) for it in test}
    blind = float(np.mean([a["blind_cap"] for a in apps.values()]))
    omni = float(np.mean([a["omni_cap"] for a in apps.values()]))

    out = {"conditions": {}, "blind_cap": blind, "omni_cap": omni}
    all_recs = []
    for kind in ("terse", "distractor", "both"):
        lp, kp, recs = classify_set(kind)
        all_recs += recs
        llm_acc = float(np.mean([t == p for t, p, _ in lp]))
        kw_acc = float(np.mean([t == p for t, p in kp]))
        llm_op = float(np.mean([np.mean([op_value(a, p, t) for t, p, _ in lp])
                                for a in apps.values()]))
        kw_op = float(np.mean([np.mean([op_value(a, p, t) for t, p in kp])
                               for a in apps.values()]))
        crosses = sum(1 for a in apps.values()
                      if np.mean([op_value(a, p, t) for t, p, _ in lp]) < a["blind_cap"])
        out["conditions"][kind] = dict(llm_acc=llm_acc, kw_acc=kw_acc, llm_op=llm_op,
                                       kw_op=kw_op, llm_crosses=f"{crosses}/6")
        print(f"{kind:10s}: LLM acc {llm_acc:.3f} (kw {kw_acc:.3f}) | LLM op {llm_op:.4f} "
              f"(blind {blind:.4f}, omni {omni:.4f}; kw op {kw_op:.4f}) | crosses {crosses}/6",
              flush=True)
    (OUT / "robustness.json").write_text(json.dumps(out, indent=1))
    (OUT / "robustness_transcripts.json").write_text(json.dumps(all_recs, indent=1))
    print(f"\nwrote {OUT}/robustness.json")


if __name__ == "__main__":
    main()
