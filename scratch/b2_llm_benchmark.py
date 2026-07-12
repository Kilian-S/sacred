#!/usr/bin/env python3
"""B2: the agentic-LLM exploitability benchmark (EVAL-ONLY; needs an API key to run live).

Three pre-registered registers on the 35-159 headline game (N=3, K=1, band 0.15-0.95, 12 routes):
  (a) deterministic  - "choose ONE route for the stacked fleet"            -> scored under oracle BR
  (b) stated-strategy - "output a probability distribution over the menu"  -> exact mixture under oracle BR
  (c) agentic-sequential - T sorties vs the gen19 pattern-of-life adversary (w=3, tau=0.15) with
      per-sortie interception feedback -> realised mean mission-failure vs the computable ladder
      (static_det 0.613 / iid_eq 0.147 / SACRED 0.050 / history_opt 0.049).

No tools are offered to the model (the no-tools register is the informative one: with code
execution a frontier model would simply solve the LP). Transcripts are logged verbatim as the
reproducibility record. `--provider dry` runs the full pipeline with a uniform-random synthetic
agent (validates scoring end-to-end without any API).

Usage:
  PYTHONPATH=. .venv/bin/python scratch/b2_llm_benchmark.py --provider dry
  PYTHONPATH=. .venv/bin/python scratch/b2_llm_benchmark.py --provider anthropic \
      --model claude-sonnet-5 --key-env ANTHROPIC_API_KEY --sorties 30
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

import numpy as np
import torch

from scripts.train_b1lite1 import softmax_br, stacked_L
from src.baselines.multiconvoy_oracle import best_response_attacker_multi, solve_multiconvoy
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

N, K, KX, BAND, OD = 3, 1, 8, (0.15, 0.95), ("35", "159")
W, TAU = 3, 0.15


def game_spec(env) -> str:
    g = env.game
    G = env.graph
    lines = [f"There are {g.n_routes} candidate routes from base to destination. "
             f"A hidden adversary places {K} ambush on one road segment (edge) before you move. "
             f"If your convoy fleet (all {N} convoys travel together on ONE route) crosses the "
             f"ambushed edge, each convoy is independently intercepted with the edge's "
             f"interception probability. The MISSION FAILS if at least one convoy is intercepted.",
             "Routes (edge: length, interception probability if ambushed there):"]
    emap = env.edge_vulnerability

    def ev(u, v):
        return emap.get((u, v), emap.get((v, u), emap.get(tuple(sorted((u, v), key=repr)), 0.0)))
    for i, r in enumerate(g.routes):
        segs = [f"[{u}-{v}: {G[u][v]['w']:.0f}m, p={ev(u, v):.2f}]" for u, v in zip(r, r[1:])]
        lines.append(f"Route {i}: " + " ".join(segs))
    return "\n".join(lines)


def call_llm(provider, model, key, messages, max_tokens=700):
    if provider == "anthropic":
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": model, "max_tokens": max_tokens,
                             "messages": messages}).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["content"][0]["text"]
    if provider == "openai":
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": model, "max_tokens": max_tokens,
                             "messages": messages}).encode(),
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    raise ValueError(provider)


def parse_route(text, R):
    m = re.findall(r"ROUTE\s*[:=]?\s*(\d+)", text.upper())
    if m and 0 <= int(m[-1]) < R:
        return int(m[-1])
    m = re.findall(r"\b(\d+)\b", text)
    for x in reversed(m):
        if 0 <= int(x) < R:
            return int(x)
    return None


def parse_dist(text, R):
    m = re.findall(r"(?:route\s*)?(\d+)\s*[:=]\s*(0?\.\d+|\d+(?:\.\d+)?%?)", text.lower())
    d = np.zeros(R)
    for i, v in m:
        i = int(i)
        if 0 <= i < R:
            d[i] = float(v.rstrip("%")) / (100.0 if v.endswith("%") else 1.0)
    return d / d.sum() if d.sum() > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["dry", "anthropic", "openai"], default="dry")
    ap.add_argument("--model", default="dry-uniform")
    ap.add_argument("--key-env", default="ANTHROPIC_API_KEY")
    ap.add_argument("--sorties", type=int, default=30)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    torch.set_num_threads(2)
    import os
    key = os.environ.get(a.key_env, "")
    rng = np.random.default_rng(0)

    env = make_multiconvoy_env(od=OD, N=N, K=K, k_extra_routes=KX, menu_select=True,
                               edge_vuln_band=BAND, interception_loss=10.0, seed=0)
    sol = solve_multiconvoy(env.game, N, "mission")
    R = env.game.n_routes
    L = stacked_L(env.game, N)
    spec = game_spec(env)
    transcript = []

    def ask(prompt):
        transcript.append({"prompt": prompt})
        if a.provider == "dry":
            # synthetic uniform agent: route = uniform draw; distribution = uniform
            reply = (f"ROUTE: {int(rng.integers(R))}" if "ROUTE:" in prompt
                     else "\n".join(f"{i}: {1/R:.4f}" for i in range(R)))
        else:
            reply = call_llm(a.provider, a.model, key,
                             [{"role": "user", "content": prompt}])
            time.sleep(1)
        transcript[-1]["reply"] = reply
        return reply

    out = {"model": a.model, "provider": a.provider,
           "anchors": {"loss_det": sol.loss_det, "eq": sol.loss_mixed,
                       "uniform_stack": None, "static_det_c": 0.613, "iid_eq_c": 0.147,
                       "sacred_c": 0.050, "history_opt_c": 0.049}}
    d_unif = np.zeros(len(env.occupancies))
    for r in range(R):
        d_unif[env._occ_index[tuple(N if i == r else 0 for i in range(R))]] = 1.0 / R
    _, u = best_response_attacker_multi(env.obj_matrix, d_unif)
    out["anchors"]["uniform_stack"] = float(u)

    # (a) deterministic register
    reply = ask(spec + "\n\nThe adversary knows your standing plan and will place its ambush "
                       "optimally against it. Choose the single best route for the fleet.\n"
                       "Answer with exactly one line: ROUTE: <number>")
    r = parse_route(reply, R)
    if r is not None:
        out["a_deterministic"] = {"route": r, "expl": float(L[r].max())}

    # (b) stated-strategy register
    reply = ask(spec + "\n\nYou may RANDOMISE: on each sortie the fleet samples one route from a "
                       "probability distribution you commit to. The adversary learns your "
                       "distribution and places its ambush optimally against it. State the best "
                       "distribution.\nAnswer with one line per route: <route>: <probability>")
    d = parse_dist(reply, R)
    if d is not None:
        dd = np.zeros(len(env.occupancies))
        for i in range(R):
            dd[env._occ_index[tuple(N if j == i else 0 for j in range(R))]] = d[i]
        _, expl = best_response_attacker_multi(env.obj_matrix, dd)
        out["b_stated"] = {"dist": [round(float(x), 4) for x in d], "expl": float(expl),
                           "ratio_to_eq": float(expl / sol.loss_mixed)}

    # (c) agentic-sequential register vs the pattern-of-life adversary (w=3, tau=0.15)
    counts = np.zeros(R)
    fails, choices = [], []
    for t in range(a.sorties):
        fb = ""
        if choices:
            fb = (f"\nHistory so far (sortie: route, mission outcome): " +
                  "; ".join(f"{i+1}: R{c} {'FAILED' if f else 'ok'}"
                            for i, (c, f) in enumerate(zip(choices, fails))))
        reply = ask(spec + f"\n\nYou fly repeated sorties. The adversary watches your last {W} "
                           f"route choices and tends to ambush where you have recently been "
                           f"(softmax of expected damage, temperature {TAU})." + fb +
                           f"\n\nSortie {t+1} of {a.sorties}: choose this sortie's route.\n"
                           f"Answer with exactly one line: ROUTE: <number>")
        c = parse_route(reply, R)
        if c is None:
            c = int(rng.integers(R))
        # adversary saw only the trailing window W of realised play
        window = np.zeros(R)
        for cc in choices[-W:]:
            window[cc] += 1
        br = softmax_br(window, L, TAU)
        j = int(rng.choice(len(br), p=br))
        fails.append(float(L[c, j]))
        choices.append(c)
        counts[c] += 1
    out["c_agentic"] = {"mean_mission_failure": float(np.mean(fails)),
                        "choices": choices, "n_sorties": a.sorties,
                        "repeat_rate_w": float(np.mean([c in choices[max(0, i - W):i]
                                                        for i, c in enumerate(choices) if i > 0]))}

    print(json.dumps({k: v for k, v in out.items() if k != "transcript"}, indent=2))
    path = a.json_out or f"models/runs/b2_llm/{a.model.replace('/', '_')}.json"
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    out["transcript"] = transcript
    json.dump(out, open(path, "w"), indent=2)
    print(f"[written] {path}")


if __name__ == "__main__":
    main()
