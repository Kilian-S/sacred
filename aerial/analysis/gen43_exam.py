#!/usr/bin/env python3
"""Sits one model configuration on the gen43 forty-question bank: one call per item with a single
retry on parse or format failure, non-thinking arms at temperature 0 and the thinking arm at 0.6,
seed 0 and max_tokens 16000. Traces are saved.

    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen43_exam.py --model qwen3-27b
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen43_exam.py --model qwen3-27b --thinking on
    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen43_exam.py --model qwen35-2b \
        --base http://cv-iits-w05.tail5b8d80.ts.net:8005/v1
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

BANK = Path("models/runs/gen43_exam/bank.json")
KEY = "iits-local-key"
SYSTEM = "You are an air-defence planner choosing emplacements."


def call(base, model, prompt, k, thinking, temperature=None, seed=None):
    """Puts one exam question to the model.

    Leaving temperature and seed as None reproduces the pinned decoding: 0.6 with seed 0 for the
    thinking arm, 0.0 and no seed otherwise.
    """
    import requests
    schema = {"type": "object",
              "properties": {"slots": {"type": "array", "items": {"type": "string"},
                                       "minItems": k, "maxItems": k}},
              "required": ["slots"]}
    body = {"model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "max_tokens": 16000 if thinking else 1000,
            "temperature": (0.6 if thinking else 0.0) if temperature is None else temperature,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "slot_choice", "schema": schema}}}
    if thinking:
        body["seed"] = 0 if seed is None else seed
    r = requests.post(base.rstrip("/") + "/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {KEY}",
                               "content-type": "application/json"}, timeout=900)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    msg = ch.get("message") or {}
    return (msg.get("content") or "", ch.get("finish_reason", ""),
            msg.get("reasoning") or "", r.json().get("usage"))


def parse_choice(txt, item):
    m = re.search(r"\{.*\}", txt, re.S)
    obj = json.loads(m.group(0))
    names = [str(s).strip().lower() for s in obj["slots"]]
    valid = set(s.lower() for s in item["slots"])
    if len(names) != item["K"] or len(set(names)) != item["K"] or not set(names) <= valid:
        raise ValueError(f"bad slot choice {names}")
    return sorted(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://cv-iits-w05.tail5b8d80.ts.net:8080/v1")
    ap.add_argument("--thinking", choices=("off", "on"), default="off")
    ap.add_argument("--out", default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="amendment rows only; default reproduces the pinned decoding")
    ap.add_argument("--seed", type=int, default=None,
                    help="amendment rows only; default is the pinned seed 0")
    a = ap.parse_args()
    think = a.thinking == "on"
    tag = a.model + ("_think" if think else "")
    out = Path(a.out or f"models/runs/gen43_exam/{tag}.json")
    bank = json.loads(BANK.read_text())["items"]
    rows, traces = [], []
    t0 = time.time()
    for it in bank:
        table = {tuple(sorted(s.lower() for s in c)): v for c, v in it["table"]}
        rec = dict(id=it["id"], map=it["map"], S=it["S"], K=it["K"], status="format_fail",
                   share=None, solved=False, pct=None, choice=None)
        for attempt in range(2):
            try:
                txt, fr, reasoning, usage = call(a.base, a.model, it["prompt"], it["K"], think,
                                                 a.temperature, a.seed)
                traces.append(dict(id=it["id"], attempt=attempt, finish=fr, usage=usage,
                                   reasoning_len=len(reasoning), content=txt[:2000]))
                names = tuple(parse_choice(txt, it))
                v = table[names]
                vals = np.array(sorted(table.values()))
                rec.update(status="ok", choice=list(names), share=v / it["ceiling"],
                           solved=bool(abs(v - it["ceiling"]) < 1e-9),
                           pct=float(np.searchsorted(vals, v, side="right") / len(vals)))
                break
            except Exception as e:                                     # noqa: BLE001
                print(f"  [item {it['id']} attempt {attempt}] {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
        rows.append(rec)
        if rec["status"] == "ok":
            print(f"  item {it['id']:2d} {it['map']:14s} S{it['S']}K{it['K']}: share "
                  f"{rec['share']:.2f} pct {rec['pct']:.2f}{' SOLVED' if rec['solved'] else ''}"
                  f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    summary = dict(model=a.model, thinking=a.thinking, base=a.base, n=len(rows),
                   temperature=a.temperature, seed=a.seed,
                   format_fail=len(rows) - len(ok),
                   mean_share=float(np.mean([r["share"] for r in ok])) if ok else None,
                   solved=sum(r["solved"] for r in ok),
                   mean_pct=float(np.mean([r["pct"] for r in ok])) if ok else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, rows=rows, traces=traces), indent=1))
    print(f"\n== {tag}: {summary['solved']}/{len(rows)} solved, mean share "
          f"{summary['mean_share']}, mean pct {summary['mean_pct']}, "
          f"format-fail {summary['format_fail']} ==\n[written] {out}")


if __name__ == "__main__":
    main()
