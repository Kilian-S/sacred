#!/usr/bin/env python3
"""gen42 capability-ladder battery driver (pre-registered in experiments/gen42_capability_ladder.md).

Runs the fixed three-phase battery against ONE served model (a "rung") by wrapping the three
gen39 harnesses WITHOUT modifying them (the scratch/gen39_phase1e_thinking.py pattern: rebind
module attributes, then call the harness's own main() with a manipulated sys.argv):

  B-COMP  scratch/gen39_compose.py  --live with N_LLM=16  -> 16 baseline + 16 relabel-control
  B-SLOT  scratch/gen39_phase1e.py  --n 4 --rounds 2      -> 8 calls, grounded slot choice
  B-EFF   scratch/gen39_phase1f.py  --budget 96 --rounds 8 -> the 8/16/96 budget curve

Per phase the harness's MODELS, BASE_URL, OUT and OUTDIR are rebound; every call goes through one
explicit caller that ALWAYS states chat_template_kwargs.enable_thinking, so behaviour is identical
through the workbench gateway or straight to a bare vLLM port. Corrected-brief instruments only
(post the 2026-08-06 gen39 repair). Eval-only: no training anywhere.

    PYTHONPATH=. ../sacred/.venv/bin/python scratch/gen42_battery.py --rung qwen35-2b
    PYTHONPATH=. ../sacred/.venv/bin/python scratch/gen42_battery.py --rung qwen3-27b --thinking on
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

N_FORCES = 16              # B-COMP: forces per arm per rung (ledger: 16 + 16 relabel control)
SLOT_N, SLOT_ROUNDS = 4, 2  # B-SLOT: 4 lineages x 2 rounds = 8 calls
EFF_BUDGET, EFF_ROUNDS = 96, 8   # B-EFF: the banked 8/16/96 curve
THINK_TOKENS = 8000        # forced co-change when thinking is on (the trace draws the same budget)

_lock = threading.Lock()
_traces: list = []
_PHASE = ""


def make_caller(thinking: bool):
    """The pinned call site with the reasoning mode stated explicitly on every request.

    Failures are printed before re-raising: gen39_phase1e.one() and gen39_phase1f's proposal loop
    both swallow caller exceptions, so a dead endpoint would otherwise report a legitimate zero.
    """
    def call_openai(base, key, model, system, user, schema=None, max_tokens=1500,
                    temperature=0.7, timeout=900):
        import requests
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": THINK_TOKENS if thinking else max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
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
                _traces.append(dict(phase=_PHASE, model=model, finish_reason=ch.get("finish_reason"),
                                    usage=d.get("usage"), reasoning=msg.get("reasoning") or "",
                                    content=msg.get("content") or ""))
            return msg.get("content") or "", ch.get("finish_reason", "")
        except Exception as e:                                          # noqa: BLE001
            print(f"  [call FAILED] {_PHASE}/{model}: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            raise
    return call_openai


def phase_comp(args, caller):
    """B-COMP: the step-2 protocol verbatim (--live), one model, 16 forces per arm."""
    import scratch.gen39_compose as comp
    comp.g33.call_openai = caller          # g33 is one shared module object across all 3 harnesses
    comp.MODELS = (args.rung,)
    comp.BASE_URL = args.base
    comp.OUTDIR = args.outdir
    comp.N_LLM = N_FORCES
    sys.argv = ["gen39_compose", "--live"]
    comp.main()
    src = args.outdir / "scores.json"
    if src.exists():
        shutil.copyfile(src, args.outdir / "comp.json")


def phase_slot(args, caller):
    """B-SLOT: the corrected Phase-1e protocol verbatim, 4 lineages x 2 rounds."""
    import scratch.gen39_phase1e as p1e
    p1e.g33.call_openai = caller
    p1e.MODELS = (args.rung,)
    p1e.BASE_URL = args.base
    p1e.OUTDIR = args.outdir
    p1e.OUT = args.outdir / "slot.json"
    sys.argv = ["gen39_phase1e", "--n", str(SLOT_N), "--rounds", str(SLOT_ROUNDS)]
    p1e.main()


def phase_eff(args, caller):
    """B-EFF: the Phase-1f protocol at budget 96 (the 8/16/96 marks), one lineage."""
    import scratch.gen39_phase1f as p1f
    p1f.g33.call_openai = caller
    p1f.MODELS = (args.rung,)
    p1f.BASE_URL = args.base
    p1f.OUTDIR = args.outdir
    p1f.OUT = args.outdir / "eff.json"
    print("  NOTE phase1f has no flag to skip the random / greedy / local arms, so they are "
          "recomputed. They are model-independent and seeded (default_rng(0)) at the banked "
          f"budget {EFF_BUDGET}, so the curves reproduce the banked ones exactly.", flush=True)
    sys.argv = ["gen39_phase1f", "--budget", str(EFF_BUDGET), "--rounds", str(EFF_ROUNDS)]
    p1f.main()


PHASES = {"comp": phase_comp, "slot": phase_slot, "eff": phase_eff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, help="the served model name, e.g. qwen35-2b")
    ap.add_argument("--base", default="http://cv-iits-w05.tail5b8d80.ts.net:8080/v1")
    ap.add_argument("--thinking", choices=("off", "on"), default="off")
    ap.add_argument("--phases", default="comp,slot,eff")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    if not Path("data/maps").is_dir():
        sys.exit("run from the repo root: PYTHONPATH=. ../sacred/.venv/bin/python scratch/gen42_battery.py ...")
    tag = a.rung + ("_think" if a.thinking == "on" else "")
    a.outdir = Path(a.outdir or f"models/runs/gen42_ladder/{tag}")
    a.outdir.mkdir(parents=True, exist_ok=True)
    want = [p.strip() for p in a.phases.split(",") if p.strip()]
    bad = [p for p in want if p not in PHASES]
    if bad:
        sys.exit(f"unknown phase(s) {bad}; choose from {list(PHASES)}")
    caller = make_caller(a.thinking == "on")

    print(f"[gen42 battery] rung={a.rung} thinking={a.thinking} phases={want}\n"
          f"  endpoint {a.base}\n  -> {a.outdir}\n"
          f"  NOTE B-COMP's headline lands as scores.json and is copied to comp.json; the ledger "
          f"names it compose.json.\n", flush=True)

    global _PHASE
    done, failed, stamps = [], [], {}
    for name in ["comp", "slot", "eff"]:
        if name not in want:
            continue
        _PHASE = name
        t0 = time.time()
        stamps[name] = {"start": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"\n{'=' * 88}\n[phase {name}] {a.rung}\n{'=' * 88}", flush=True)
        try:
            PHASES[name](a, caller)
            done.append(name)
        except Exception as e:                                          # noqa: BLE001
            failed.append(name)
            print(f"[phase {name}] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        finally:
            stamps[name]["end"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            stamps[name]["minutes"] = round((time.time() - t0) / 60, 2)
            with _lock:
                tr = [t for t in _traces if t["phase"] == name]
            if tr:
                (a.outdir / f"calls_{name}.json").write_text(json.dumps(tr, indent=1))
                print(f"[written] {a.outdir / f'calls_{name}.json'}  ({len(tr)} calls)", flush=True)
            else:
                print(f"[warn] no calls recorded for phase {name}", file=sys.stderr)

    summary = dict(rung=a.rung, thinking=a.thinking, base=a.base, outdir=str(a.outdir),
                   phases_run=done, phases_failed=failed, timestamps=stamps,
                   settings=dict(n_forces=N_FORCES, slot_n=SLOT_N, slot_rounds=SLOT_ROUNDS,
                                 eff_budget=EFF_BUDGET, eff_rounds=EFF_ROUNDS,
                                 think_tokens=THINK_TOKENS if a.thinking == "on" else None))
    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n[gen42 battery] run {done}, failed {failed}\n[written] {a.outdir / 'summary.json'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
