#!/usr/bin/env python3
"""Sweeps the gen44 authoring budget: whether the author's reasoning strength ever separates, and
at what search budget. The step-5 authoring loop, model-parameterised and repeated so the readings
carry error bars. Evaluation only, with LLM proposals and exact scoring but no training; the
search, prompt, temperature, doctrine and operating point are imported from step 5 rather than
re-implemented.

    # a mounted rung, inside its existing window
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen44_budget_sweep.py --config qwen35-4b \
        --base http://<llm-host>:8006/v1
    # gateway configurations
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen44_budget_sweep.py --config llama-3.3-70b
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen44_budget_sweep.py --config qwen3-27b --thinking on
    # the no-LLM controls
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen44_budget_sweep.py --config local16
"""
from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from analysis.gen39_compose import KEY, g33
from analysis.gen39_phase1f import SCHEMA, llm_prompt, map_digest
from analysis.gen39_zeroshot import _init, base_for, score, search_local, search_random
from src.envs.aerial_conceal import resample_field

FIELDS = (1000, 1001, 1002)
REPEATS = 3
BUDGET = 16
K = 3
MARKS = (2, 4, 8, 16)
OUTDIR = Path("models/runs/gen44_sweep")
GATEWAY = os.environ.get("SACRED_LLM_BASE", "")


def search_llm_cfg(base, digest, pool, field, model, url, thinking, rng_tag):
    """Runs the step-5 search_llm loop against a named model, printing rather than swallowing
    call failures."""
    hist = []
    for _round in range(4):
        left = BUDGET - len(hist)
        if left <= 0:
            break
        tri = []
        for attempt in range(2):
            try:
                body_extra = dict(max_tokens=16000) if thinking else dict(max_tokens=2500)
                txt, _m = call(url, model, llm_prompt(digest, hist, min(4, left)), thinking,
                               **body_extra)
                for f in g33._extract_json(txt).get("forces", []):
                    s = [int(x) for x in f.get("sites", []) if 0 <= int(x) < base.H]
                    if len(set(s)) == K:
                        tri.append(tuple(sorted(set(s))))
                if tri:
                    break
            except Exception as e:                                     # noqa: BLE001
                print(f"    [{rng_tag} round {_round} attempt {attempt}] "
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)
        tri = [t for t in dict.fromkeys(tri) if t not in dict(hist)][:left]
        if not tri:
            break
        hist += list(score(pool, tri, field).items())
    return hist


def call(url, model, prompt, thinking, max_tokens=2500):
    import requests
    body = {"model": model,
            "messages": [{"role": "system",
                          "content": "You are an air-defence planner running a search."},
                         {"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.9,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "forces", "schema": SCHEMA}}}
    r = requests.post(url.rstrip("/") + "/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {KEY}",
                               "content-type": "application/json"}, timeout=900)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return (ch.get("message") or {}).get("content") or "", ch.get("finish_reason", "")


def running_best(hist):
    """Best-so-far after each evaluation, in the order the searcher spent its budget."""
    out, b = [], 0.0
    for _sites, v in hist:
        b = max(b, float(v))
        out.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="served model name, or 'local16' / 'random16' for the no-LLM controls")
    ap.add_argument("--base", default=GATEWAY)
    ap.add_argument("--thinking", choices=("off", "on"), default="off")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    thinking = a.thinking == "on"
    tag = a.config + ("_think" if thinking else "")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{tag}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}

    base = base_for("narva")
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=("narva",))
    try:
        pp0 = base.lethality(resample_field(base.coords, FIELDS[0]), hidden_leth=1.0)
        digest = map_digest(base, pp0)
        for f in FIELDS:
            for rep in range(REPEATS):
                key = f"{f}_r{rep}"
                if key in done:
                    print(f"  {tag} {key}: already stored", flush=True)
                    continue
                t0 = time.time()
                rng = np.random.default_rng(f * 100 + rep)
                if a.config == "local16":
                    hist = search_local(base, pool, f, rng)
                elif a.config == "random16":
                    hist = search_random(base, pool, f, rng)
                else:
                    hist = search_llm_cfg(base, digest, pool, f, a.config, a.base, thinking, key)
                rb = running_best(hist)
                done[key] = dict(
                    n=len(hist), running_best=rb,
                    at={str(b): (rb[min(b, len(rb)) - 1] if rb else None) for b in MARKS},
                    history=[[list(map(int, s)), float(v)] for s, v in hist])
                out_path.write_text(json.dumps(done, indent=1))
                at = done[key]["at"]
                print(f"  {tag} field {f} rep {rep}: n={len(hist):2d} "
                      + " ".join(f"@{b} {at[str(b)]:.4f}" if at[str(b)] is not None else f"@{b} -"
                                 for b in MARKS)
                      + f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
    finally:
        pool.close()
        pool.join()
    print(f"[written] {out_path}  ({len(done)}/{len(FIELDS)*REPEATS} searches)")


if __name__ == "__main__":
    main()
