#!/usr/bin/env python3
"""gen39 step 5c PREP: the qwenthink16 curriculum (Qwen3.6-27B, thinking ON, positions only;
oracle + model calls, no training). Pinned by the step-5c pre-registration in the ledger.

Identical machinery to step5_prep's llm16 arm (4 rounds x <=4 proposals, matched budget of 16
exact evaluations per field, top-3 kept, gen32 doctrine frozen) with exactly two changes: the
model is qwen3-27b (Qwen3.6-27B) called with chat_template_kwargs.enable_thinking=true and
max_tokens 8000 (the thinking trace draws from the same budget; the gen39_phase1e_thinking
precedent), and transport failures print loudly before the retry (the false-zero lesson).

Output: models/runs/gen39_step5/curricula_qwenthink.json = the four banked family keys copied
byte-identically from curricula.json (build_pools_step5 reads the TEST-SET families out of the
same file, so the banked test set is preserved exactly) plus the new "qwenthink16" key.

    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen39_step5c_prep.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import analysis.gen39_step5_prep as p5
from analysis.gen39_compose import BASE_URL, K, KEY
from analysis.gen39_phase1f import SCHEMA, llm_prompt, map_digest
from scripts.train_gen39_conceal import TEST_FIELDS, TRAIN_FIELDS, narva_base
from src.envs.aerial_conceal import resample_field

MODEL, THINK_TOKENS = "qwen3-27b", 16000
OUT = Path("models/runs/gen39_step5/curricula_qwenthink.json")
BANKED = Path("models/runs/gen39_step5/curricula.json")
PROGRESS = Path("models/runs/gen39_step5/qwenthink_progress.json")


def call_thinking(base, key, model, system, user, schema=None, max_tokens=2500,
                  temperature=0.9, timeout=900):
    import requests
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": THINK_TOKENS, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": True}}
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "red_force", "schema": schema}}
    try:
        r = requests.post(base.rstrip("/") + "/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {key}",
                                   "content-type": "application/json"}, timeout=timeout)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        return ch["message"].get("content") or "", ch.get("finish_reason", "")
    except Exception as e:                                              # noqa: BLE001
        print(f"  [call FAILED] {type(e).__name__}: {str(e)[:200]}", flush=True)
        raise


def qwenthink16(base, digest, pool, field):
    hist = []
    for _ in range(4):
        left = p5.BUDGET - len(hist)
        if left <= 0:
            break
        tri = []
        for _try in range(3):
            try:
                txt, fr = call_thinking(BASE_URL, KEY, MODEL,
                                        "You are an air-defence planner running a search.",
                                        llm_prompt(digest, hist, min(4, left)), schema=SCHEMA,
                                        temperature=0.9, timeout=900)
                for f in p5.g33._extract_json(txt).get("forces", []):
                    s = [int(x) for x in f.get("sites", []) if 0 <= int(x) < base.H]
                    if len(set(s)) == K:
                        tri.append(tuple(sorted(set(s))))
                if tri:
                    break
                print(f"  [unusable reply] field {field} try {_try}: finish={fr} "
                      f"content_len={len(txt)}", flush=True)
            except Exception as e:                                     # noqa: BLE001
                print(f"  [parse FAILED] field {field} try {_try}: {type(e).__name__}",
                      flush=True)
                continue
        tri = [t for t in dict.fromkeys(tri) if t not in dict(hist)][:left]
        if not tri:
            break
        got = p5.score(pool, tri, field)
        hist += list(got.items())
    return hist


def main():
    import multiprocessing as mp_
    base = narva_base()
    pp0 = base.lethality(resample_field(base.coords, 1000), hidden_leth=1.0)
    digest = map_digest(base, pp0)
    banked = json.loads(BANKED.read_text())
    fields = list(TRAIN_FIELDS) + list(TEST_FIELDS)
    new: dict = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    if new:
        print(f"  resuming: {len(new)} fields already done", flush=True)
    t0 = time.time()
    with mp_.get_context("spawn").Pool(9, initializer=p5._init) as P:
        for field in fields:
            if str(field) in new:
                continue
            h = sorted(qwenthink16(base, digest, P, field), key=lambda x: -x[1])[:p5.KEEP]
            if not h:
                raise RuntimeError(f"field {field}: zero usable proposals - transport or "
                                   f"model failure, not a curriculum")
            new[str(field)] = [[list(map(int, t)), float(v)] for t, v in h]
            PROGRESS.write_text(json.dumps(new, indent=1))
            print(f"  field {field}: qwenthink16 best {new[str(field)][0][1]:.4f} "
                  f"(kept {len(h)})  [{(time.time()-t0)/60:.1f} min]", flush=True)
    out = dict(banked)
    out["qwenthink16"] = new
    OUT.write_text(json.dumps(out, indent=1))
    check = json.loads(OUT.read_text())
    assert all(check[a] == banked[a] for a in ("llm16", "local16", "random16", "tuned"))
    tr = np.median([new[str(f)][0][1] for f in TRAIN_FIELDS])
    te = np.median([new[str(f)][0][1] for f in TEST_FIELDS])
    print(f"\nqwenthink16 train-field median best {tr:.4f}  test-field median best {te:.4f}")
    print(f"[written] {OUT} (four banked family keys verified byte-identical)")


if __name__ == "__main__":
    main()
