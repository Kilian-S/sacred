#!/usr/bin/env python3
"""Runs gen39 phase 1e with the qwen3-27b reasoning mode switched on, everything else unchanged.
Only llama-3.3-70b is dropped, since it has no thinking mode. Raising max_tokens from 3000 to
8000 is a forced co-change: the reasoning trace draws the same generation budget before the JSON
appears, so the original cap would guarantee truncation.

    PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen39_phase1e_thinking.py --n 4 --rounds 2
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import analysis.gen39_phase1e as p1e

MAX_TOKENS = 8000
OUT = Path("models/runs/gen39_phase1e_thinking.json")
TRACES = Path("models/runs/gen39_phase1e_thinking_traces.json")

# The raw Tailscale IP connect-times-out from `requests`; the MagicDNS name must be used instead.
BASE_URL_FIXED = "http://cv-iits-w05.tail5b8d80.ts.net:8080/v1"

_lock = threading.Lock()
_trace_log: list = []


def call_openai_thinking(base, key, model, system, user, schema=None, max_tokens=1500,
                         temperature=0.7, timeout=900):
    """Calls the model with reasoning enabled, re-raising failures rather than returning a zero."""
    import requests
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "red_force", "schema": schema}}
    try:
        r = requests.post(base.rstrip("/") + "/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {key}",
                                   "content-type": "application/json"},
                          timeout=timeout)
        r.raise_for_status()
        d = r.json()
        ch = d["choices"][0]
        msg = ch.get("message") or {}
        with _lock:
            _trace_log.append(dict(model=model, finish_reason=ch.get("finish_reason"),
                                   usage=d.get("usage"),
                                   reasoning=msg.get("reasoning") or "",
                                   content=msg.get("content") or ""))
        return msg.get("content") or "", ch.get("finish_reason", "")
    except Exception as e:                                             # noqa: BLE001
        print(f"  [call FAILED] {type(e).__name__}: {str(e)[:200]}", flush=True)
        raise


if __name__ == "__main__":
    p1e.g33.call_openai = call_openai_thinking
    p1e.MODELS = ("qwen3-27b",)
    p1e.OUT = OUT                      # never overwrite the banked thinking-OFF run
    p1e.BASE_URL = BASE_URL_FIXED
    print(f"[thinking-on probe] models={p1e.MODELS} max_tokens={MAX_TOKENS}\n"
          f"  endpoint {BASE_URL_FIXED}\n  -> {OUT}\n")
    try:
        p1e.main()
    finally:
        if _trace_log:
            TRACES.write_text(json.dumps(_trace_log, indent=1))
            print(f"[written] {TRACES}  ({len(_trace_log)} calls)")
        else:
            print("[warn] no calls were recorded", file=sys.stderr)
