#!/usr/bin/env python3
"""Generate the LLM red force for the gen33 experiment: serialise each theatre to a brief, ask the
model for a structured force, validate it, resolve it onto concrete sites and doctrine, and save.
The (model x theatre x phase x force-index) tasks fan out over a capped thread pool because the
calls are HTTP-bound, and `--provider dry` runs the whole pipeline against a synthetic force with
no model and no network.

  Dry:   PYTHONPATH=. <venv>/bin/python scripts/gen33_generate_force.py --provider dry
  Smoke: PYTHONPATH=. <venv>/bin/python scripts/gen33_generate_force.py --provider openai \
           --base http://100.88.32.88:8080/v1 --models llama-3.3-70b,qwen3-27b \
           --phases single,coordinated --n 1
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.envs.aerial_theatre_vec import build_theatre_game, lateral_width, load_vec_theatre
from src.redforce import (FORCE_SCHEMA, dry_force, resolve_force_to_sites, serialise_theatre)

THEATRES = {
    "kgd": "data/maps/theatre_kgd_gvardeysk_vec.json",
    "ukraine": "data/maps/theatre_ukraine_vec.json",
    "narva": "data/maps/theatre_narva_vec.json",
}


def validate_force(obj) -> list:
    """Light structural validation (no jsonschema dependency): returns a list of problems."""
    problems = []
    if not isinstance(obj, dict) or "agents" not in obj or not isinstance(obj["agents"], list):
        return ["missing agents[]"]
    for i, a in enumerate(obj["agents"]):
        for req in ("archetype", "emplacement_zone", "doctrine", "rationale"):
            if req not in a:
                problems.append(f"agent{i}: missing {req}")
        if "doctrine" in a and not all(k in a["doctrine"]
                                       for k in ("punish_pattern", "anticipate_flight",
                                                 "hold_static")):
            problems.append(f"agent{i}: doctrine incomplete")
    return problems


def _extract_json(text: str) -> dict:
    """Fallback parse: strip think-tags and fences, take the outermost {...} span."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j <= i:
        raise ValueError("no JSON object found")
    return json.loads(t[i:j + 1])


def call_openai(base, key, model, system, user, schema=None, max_tokens=1500, temperature=0.7,
                timeout=900):
    import requests
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if schema is not None:
        # The gateway ignores vLLM's guided_json extra but honours the OpenAI structured-output
        # form, so that is the only enforcement path available.
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "red_force", "schema": schema}}
    r = requests.post(base.rstrip("/") + "/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
                      timeout=timeout)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason", "")


def build_ctx(name, path, lat_ref) -> dict:
    """Per-theatre context (game build once, shared read-only by all tasks on the theatre)."""
    th = load_vec_theatre(path)
    scale = lateral_width(th) / lat_ref
    game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(
        th, K=1, n_lanes=14, n_terrain=12, spacing_km=2.0 * scale, standoff_km=4.0 * scale,
        range_scale=scale)
    site_exposure = (1.0 - S).mean(axis=0)           # per-SITE mean interception (best-value sites)
    cls = [th.classify(c) for c in coords]
    return {"th": th, "coords": coords, "cls": cls, "exposure": site_exposure, "scale": scale}


def run_task(t, a) -> dict:
    """Run one (model, theatre, phase, index) generation: call, parse, validate, resolve.

    The live path makes two guided attempts and then falls back to a few-shot prompt without the
    schema. An emitted agent count other than K is retryable, since the brief asks for exactly K
    teams.
    """
    rec = {"model": t["model"], "theatre": t["theatre"], "phase": t["phase"], "index": t["j"],
           "K": t["K"]}
    t0 = time.perf_counter()
    force, errors = None, []
    if a.provider == "dry":
        force = dry_force(K=t["K"], seed=1000 * t["j"] + a.seed,
                          coordinated=(t["phase"] == "coordinated"))
        rec["mode"] = "dry"
    else:
        for attempt, guided in enumerate((True, True, False)):
            user = t["user"]
            if not guided:
                user += ("\n\nReturn ONLY one JSON object matching this schema, no prose, no "
                         "code fences:\n" + json.dumps(FORCE_SCHEMA))
            try:
                raw, fin = call_openai(a.base, a.key, t["model"], t["system"], user,
                                       FORCE_SCHEMA if guided else None,
                                       max_tokens=a.max_tokens, temperature=a.temperature,
                                       timeout=a.timeout)
            except Exception as e:
                errors.append(f"a{attempt}({'guided' if guided else 'fallback'}): "
                              f"{type(e).__name__}: {str(e)[:200]}")
                continue
            mode = "guided" if guided else "fallback"
            try:
                cand = json.loads(raw)
            except Exception:
                try:
                    cand = _extract_json(raw)
                    mode += "+extracted"
                except Exception as e:
                    errors.append(f"a{attempt}: parse failed ({e}); finish={fin}; "
                                  f"raw[:120]={raw[:120]!r}")
                    continue
            problems = validate_force(cand)
            if not problems and len(cand["agents"]) != t["K"]:
                problems = [f"expected {t['K']} agents, got {len(cand['agents'])}"]
            if problems:
                errors.append(f"a{attempt}({mode}): {problems[:3]}")
                continue
            force, rec["mode"] = cand, mode
            break
    rec["latency_s"] = round(time.perf_counter() - t0, 2)
    if errors:
        rec["errors"] = errors
    if force is None:
        rec["valid"] = False
        return rec
    ctx = t["ctx"]
    rec["resolved"] = resolve_force_to_sites(force, ctx["th"], ctx["coords"], ctx["cls"],
                                             ctx["exposure"])
    rec["force"], rec["valid"] = force, True
    return rec


def summarise_and_save(records, a, out_dir):
    """Group by (model, theatre, phase), save one artefact each and print the summary rows.

    Results are never pooled across models.
    """
    groups: dict = {}
    for r in records:
        groups.setdefault((r["model"], r["theatre"], r["phase"]), []).append(r)
    for (model, name, phase), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["index"])
        valid = [r for r in rs if r.get("valid")]
        lat = [r["latency_s"] for r in rs]
        archs = {ag["archetype"] for r in valid for ag in r["force"]["agents"]}
        notes = [n for r in valid for n in r["resolved"]["notes"]]
        modes: dict = {}
        for r in rs:
            m = r.get("mode", "error")
            modes[m] = modes.get(m, 0) + 1
        out = {"theatre": name, "phase": phase, "K": rs[0]["K"], "provider": a.provider,
               "model": model, "n_valid": len(valid), "n_total": len(rs), "modes": modes,
               "latency_s": {"mean": round(float(np.mean(lat)), 2),
                             "max": round(float(np.max(lat)), 2)},
               "forces": [{"index": r["index"], "mode": r["mode"], "latency_s": r["latency_s"],
                           **({"errors": r["errors"]} if r.get("errors") else {}),
                           "force": r["force"], "resolved": r["resolved"]} for r in valid],
               "failures": [{"index": r["index"], "latency_s": r["latency_s"],
                             "errors": r.get("errors", [])} for r in rs if not r.get("valid")]}
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            safe = model.replace("/", "-")
            Path(out_dir, f"force_{safe}_{name}_{phase}.json").write_text(json.dumps(out, indent=1))
        print(f"{model:14s} {name:8s} {phase:11s} K={rs[0]['K']}: {len(valid)}/{len(rs)} valid, "
              f"archetypes={len(archs)}, resolve-fallbacks={len(notes)}, modes={modes}, "
              f"lat mean/max={np.mean(lat):.1f}/{np.max(lat):.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["dry", "openai"], default="dry")
    ap.add_argument("--base", default="", help="OpenAI-compatible base URL (e.g. http://100.88.32.88:8080/v1)")
    ap.add_argument("--key", default="iits-local-key")
    ap.add_argument("--model", default="dry-synthetic")
    ap.add_argument("--models", default="", help="comma-separated model list; all held in flight together")
    ap.add_argument("--theatre", default="all", choices=list(THEATRES) + ["all"])
    ap.add_argument("--phase", default="single", choices=["single", "coordinated"])
    ap.add_argument("--phases", default="", help="comma-separated subset of single,coordinated")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--n", type=int, default=3, help="forces per (model, theatre, phase) cell (the population)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=12, help="thread-pool cap (HTTP-bound tasks)")
    ap.add_argument("--out", default="models/runs/gen33_forces")
    a = ap.parse_args()
    models = [m.strip() for m in (a.models.split(",") if a.models else [a.model]) if m.strip()]
    phases = [p.strip() for p in (a.phases.split(",") if a.phases else [a.phase]) if p.strip()]
    assert all(p in ("single", "coordinated") for p in phases), phases
    names = list(THEATRES) if a.theatre == "all" else [a.theatre]
    t0 = time.time()
    lat_ref = lateral_width(load_vec_theatre(THEATRES["kgd"]))
    ctxs = {n: build_ctx(n, THEATRES[n], lat_ref) for n in names}
    tasks = []
    for name in names:
        for phase in phases:
            K = a.K if phase == "single" else max(a.K, 3)
            system, user = serialise_theatre(ctxs[name]["th"], phase=phase, K=K,
                                             range_scale=ctxs[name]["scale"])
            for model in models:
                for j in range(a.n):
                    tasks.append({"model": model, "theatre": name, "phase": phase, "j": j,
                                  "K": K, "system": system, "user": user, "ctx": ctxs[name]})
    workers = max(1, min(a.workers, len(tasks)))
    print(f"[gen33] {len(tasks)} tasks ({len(models)} models x {len(names)} theatres x "
          f"{len(phases)} phases x n={a.n}), {workers} workers", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_task, t, a) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            records.append(r)
            print(f"  [{r['model']} {r['theatre']} {r['phase']} #{r['index']}] "
                  f"{'ok' if r.get('valid') else 'FAIL'} {r['latency_s']}s mode={r.get('mode', '-')}",
                  flush=True)
    summarise_and_save(records, a, a.out)
    print(f"[done {time.time() - t0:.0f}s] provider={a.provider} "
          f"valid={sum(1 for r in records if r.get('valid'))}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
