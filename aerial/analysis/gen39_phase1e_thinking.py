#!/usr/bin/env python3
"""gen39 Phase 1e, THINKING-ON arm (probe, 2026-08-06).

WHY. The whole LLM arc called `qwen3-27b` through the workbench gateway on :8080. That gateway
injects `enable_thinking: False` into any request for a model whose `default_thinking` is false
(`gateway.py:128-132`), and the shared call site (`scripts/gen33_generate_force.py:68-89`) never
sets `chat_template_kwargs`. The box's own audit log confirms it: every one of the 213 qwen calls
on 22, 23, 26 and 27 July is recorded `thinking: off(default)`. The served model is
Qwen3.6-27B (`base_model: Qwen/Qwen3.6-27B`), a reasoning model, so the arc measured its
NON-thinking mode throughout. This probe asks what its thinking mode does on the same task.

WHAT CHANGES. Three things, and nothing else:
  1. MODELS restricted to qwen3-27b (llama-3.3-70b has no thinking mode; its rows already exist);
  2. `chat_template_kwargs={"enable_thinking": True}` on every call;
  3. max_tokens 3000 -> 8000. This is a FORCED co-change, not a free variable: on vLLM 0.23 the
     trace lands in `message.reasoning` but draws the same generation budget before the JSON
     appears in `content`, so the original cap would guarantee truncation. Disclosed as such.
Fields, map, K, W, TAU, temperature, brief, schema, rounds, lineages and the scoring path are
the pinned Phase 1e values, untouched.

PRE-REGISTERED BARS (fixed before the first call; comparators read from the banked
`models/runs/gen39_phase1e.json`, thinking OFF: median 0.0071 = 26% of ceiling, best 0.0113 =
41%, grounding 92%, free lanes 0.5; restricted ceiling 0.0278; random slot choice 0.0055):
  T1 MATERIALITY  median >= 0.0107 (1.5x the thinking-off median). If met, the thinking mode is
                  a live factor and every rung of any capability ladder must control for it.
  T2 PHASE-1E C1  median >= 0.0167 (60% of the restricted ceiling) - the bar thinking-off FAILED.
  T3 NO REGRESSION grounding >= 80% (the fixed interface must still work).
Reported beside them: best, free lanes, per-round, and completion-token / truncation counts read
back from the box's audit log.

DISCLOSURE, FIXED IN ADVANCE. n=8 (4 lineages x 2 rounds), one temperature, one map. This is a
DIAGNOSTIC PROBE, not a test. A positive licenses a powered run; it licenses no claim.

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

# The pinned BASE_URL (`http://100.88.32.88:8080/v1`, gen39_compose.py:57) NO LONGER ROUTES from
# Kilian's Mac: the raw Tailscale IP connect-times-out from Python while the MagicDNS name
# resolves instantly (measured 2026-08-06; `curl -4` to the IP still works, plain `requests` does
# not). The first attempt at this probe returned n=0 on every round because `gen39_phase1e.one()`
# swallows the ConnectTimeout in `except Exception: continue` and reports a clean zero. Any future
# run of the LLM harness from this machine would fail the same silent way.
BASE_URL_FIXED = "http://cv-iits-w05.tail5b8d80.ts.net:8080/v1"

_lock = threading.Lock()
_trace_log: list = []


def call_openai_thinking(base, key, model, system, user, schema=None, max_tokens=1500,
                         temperature=0.7, timeout=900):
    """The pinned call, with the reasoning mode switched ON and room for the trace.

    Failures are printed before re-raising, so a dead endpoint can never masquerade as a
    legitimate zero again.
    """
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
