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
    # explicit shared-segment table (amendment 2026-07-16: the strategic structure must be
    # READABLE from the prompt; whether the model USES it is what the benchmark measures)
    shared = {}
    for i, re_ in enumerate(g.route_edges):
        for e in re_:
            shared.setdefault(e, []).append(i)
    rows = [f"  segment {'-'.join(sorted(e, key=repr))}: routes {', '.join(map(str, v))}"
            for e, v in sorted(shared.items(), key=lambda kv: repr(kv[0])) if len(v) >= 2]
    lines.append("SHARED SEGMENTS (same physical road used by several routes):")
    lines.extend(rows if rows else ["  (none)"])
    return "\n".join(lines)


SYSTEM_MSG = ("You are a convoy routing officer. Reason step by step in text only; you have no "
              "tools. End your answer with EXACTLY the requested answer line(s), nothing after.")


def gate_questions(env):
    """Neutral comprehension gate (NO strategic hints; the independence probe is post-decision).
    Returns (prompt_text, checker(reply) -> pass_count/3)."""
    g = env.game
    shared_with_0 = sorted({j for j, re_ in enumerate(g.route_edges)
                            if j != 0 and re_ & g.route_edges[0]})
    costs = [sum(env.graph[u][v]["w"] for u, v in zip(r, r[1:])) for r in g.routes]
    longest = int(max(range(g.n_routes), key=lambda i: costs[i]))
    worst_p = [float(g.payoff[i].max()) for i in range(g.n_routes)]
    safest = int(min(range(g.n_routes), key=lambda i: worst_p[i]))
    q = ("Before deciding, answer these comprehension questions from the data above:\n"
         "Q1: Which routes share at least one segment with route 0? (list of numbers; 'none')\n"
         "Q2: Which route has the greatest total length? (one number)\n"
         "Q3: Which route has the LOWEST worst-segment interception probability? (one number)\n"
         "Answer format, three lines exactly: Q1: ... / Q2: ... / Q3: ...")

    def check(reply):
        # LAST occurrence of each answer line (reasoning models restate the questions early;
        # the first-match version misgraded correct answers — live-test fix, 2026-07-16)
        ok = 0
        m = re.findall(r"q1[:\s]*([^\n]*)", reply, re.I)
        if m:
            got = sorted({int(x) for x in re.findall(r"\d+", m[-1])})
            ok += int(got == shared_with_0 or (not got and not shared_with_0))
        m = re.findall(r"q2[:\s]*[^\d]*(\d+)", reply, re.I)
        ok += int(bool(m) and int(m[-1]) == longest)
        m = re.findall(r"q3[:\s]*[^\d]*(\d+)", reply, re.I)
        ok += int(bool(m) and int(m[-1]) == safest)
        return ok
    return q, check


def call_llm(provider, model, key, messages, max_tokens=1000, base="", temperature=0.7):
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
        host = (base.rstrip("/") if base else "https://api.openai.com/v1")
        req = urllib.request.Request(
            host + "/chat/completions",
            data=json.dumps({"model": model, "max_tokens": max_tokens,
                             "temperature": temperature,
                             "messages": messages}).encode(),
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"})
        # long-reasoning models at 12k-token budgets need minutes, not 120 s (live test 2026-07-16)
        with urllib.request.urlopen(req, timeout=900) as r:
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
    ap.add_argument("--key", default="", help="literal API key (overrides --key-env; local box)")
    ap.add_argument("--base", default="", help="OpenAI-compatible base URL, e.g. http://localhost:18080/v1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="reply token budget (live test 2026-07-16: 1000 truncated qwen3-27b's "
                         "reasoning mid-decision; 3000 fits; same for all models, disclosed)")
    ap.add_argument("--sorties", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="episode RNG + label permutation seed")
    ap.add_argument("--print-prompts", action="store_true",
                    help="emit the exact prompts (system, gate, registers a/b/c) and exit; no API")
    ap.add_argument("--register", default="abc", help="subset of registers to run, e.g. 'b'")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    torch.set_num_threads(2)
    import os
    key = a.key or os.environ.get(a.key_env, "")
    rng = np.random.default_rng(a.seed)

    env = make_multiconvoy_env(od=OD, N=N, K=K, k_extra_routes=KX, menu_select=True,
                               edge_vuln_band=BAND, interception_loss=10.0, seed=0)
    sol = solve_multiconvoy(env.game, N, "mission")
    R = env.game.n_routes
    L = stacked_L(env.game, N)
    spec = game_spec(env)
    gate_q, gate_check = gate_questions(env)
    transcript = []

    def converse(messages):
        """One conversation turn: send the full message list, log verbatim, return the reply."""
        transcript.append({"messages_tail": messages[-1]["content"][:4000]})
        if a.provider == "dry":
            last = messages[-1]["content"]
            reply = (f"ROUTE: {int(rng.integers(R))}" if "ROUTE:" in last
                     else "Q1: none / Q2: 0 / Q3: 0" if "Q1:" in last
                     else "\n".join(f"{i}: {1/R:.4f}" for i in range(R)))
        else:
            reply = call_llm(a.provider, a.model, key, messages, max_tokens=a.max_tokens,
                             base=a.base, temperature=a.temperature)
            time.sleep(0.5)
        transcript[-1]["reply"] = reply
        return reply

    def fresh_convo():
        return [{"role": "system", "content": SYSTEM_MSG}]

    def run_gate(convo):
        convo.append({"role": "user", "content": spec + "\n\n" + gate_q})
        reply = converse(convo)
        convo.append({"role": "assistant", "content": reply})
        return gate_check(reply)

    PROBE = ("One final question (this does not change your committed answer): are there sets of "
             "routes that share NO segments with each other? If so, give one largest such set "
             "of route numbers.")

    REG_A = ("The adversary knows your standing plan and will place its ambush optimally "
             "against it. Choose the single best route for the fleet.\n"
             "Answer with exactly one line: ROUTE: <number>")
    REG_B = ("You may RANDOMISE: on each sortie the fleet samples one route from a probability "
             "distribution you commit to. The adversary learns your distribution and places its "
             "ambush optimally against it. State the best distribution.\n"
             "Answer with one line per route: <route>: <probability>")

    def reg_c_turn(t, total, choices, fails):
        fb = ""
        if choices:
            fb = ("\nHistory (sortie: route, mission outcome): " +
                  "; ".join(f"{i+1}: R{c} {'FAILED' if f else 'ok'}"
                            for i, (c, f) in enumerate(zip(choices, fails))))
        return (f"You fly repeated sorties. The adversary watches your last {W} route choices "
                f"and tends to ambush where you have recently been." + fb +
                f"\n\nSortie {t+1} of {total}: choose this sortie's route.\n"
                f"Answer with exactly one line: ROUTE: <number>")

    if a.print_prompts:
        print("=" * 30, "SYSTEM", "=" * 30); print(SYSTEM_MSG)
        print("=" * 30, "TURN 1 (gate; every conversation)", "=" * 22)
        print(spec + "\n\n" + gate_q)
        print("=" * 30, "REGISTER (a) decision turn", "=" * 26); print(REG_A)
        print("=" * 30, "REGISTER (b) decision turn", "=" * 26); print(REG_B)
        print("=" * 30, "REGISTER (c) first + later turn", "=" * 22)
        print(reg_c_turn(0, a.sorties, [], []))
        print("-" * 60)
        print(reg_c_turn(3, a.sorties, [2, 5, 2], [1.0, 0.0, 0.0]))
        print("=" * 30, "POST-DECISION PROBE (after scoring)", "=" * 18); print(PROBE)
        return

    out = {"model": a.model, "provider": a.provider,
           "anchors": {"loss_det": sol.loss_det, "eq": sol.loss_mixed,
                       "uniform_stack": None, "static_det_c": 0.613, "iid_eq_c": 0.147,
                       "sacred_c": 0.050, "history_opt_c": 0.049}}
    d_unif = np.zeros(len(env.occupancies))
    for r in range(R):
        d_unif[env._occ_index[tuple(N if i == r else 0 for i in range(R))]] = 1.0 / R
    _, u = best_response_attacker_multi(env.obj_matrix, d_unif)
    out["anchors"]["uniform_stack"] = float(u)

    # (a) deterministic register: gate -> decision -> post-probe (one conversation)
    if "a" in a.register:
        convo = fresh_convo()
        gate_a = run_gate(convo)
        convo.append({"role": "user", "content": REG_A})
        reply = converse(convo)
        convo.append({"role": "assistant", "content": reply})
        r = parse_route(reply, R)
        convo.append({"role": "user", "content": PROBE})
        probe_a = converse(convo)
        if r is not None:
            out["a_deterministic"] = {"route": r, "expl": float(L[r].max()),
                                      "gate": gate_a, "probe": probe_a[:1500]}

    # (b) stated-strategy register: gate -> decision -> post-probe (one conversation)
    d = None
    if "b" in a.register:
        convo = fresh_convo()
        gate_b = run_gate(convo)
        convo.append({"role": "user", "content": REG_B})
        reply = converse(convo)
        convo.append({"role": "assistant", "content": reply})
        d = parse_dist(reply, R)
        convo.append({"role": "user", "content": PROBE})
        probe_b = converse(convo)
    if d is not None:
        dd = np.zeros(len(env.occupancies))
        for i in range(R):
            dd[env._occ_index[tuple(N if j == i else 0 for j in range(R))]] = d[i]
        _, expl = best_response_attacker_multi(env.obj_matrix, dd)
        out["b_stated"] = {"dist": [round(float(x), 4) for x in d], "expl": float(expl),
                           "ratio_to_eq": float(expl / sol.loss_mixed),
                           "gate": gate_b, "probe": probe_b[:1500]}

    # (c) agentic-sequential register vs the pattern-of-life adversary (w=3, tau=0.15):
    # ONE conversation per episode (in-context adaptation is the phenomenon under test);
    # the mechanism is described QUALITATIVELY (giving the softmax formula would invite
    # in-head BR arithmetic, a different register; design decision ledgered 2026-07-16).
    counts = np.zeros(R)
    fails, choices = [], []
    if "c" in a.register:
        convo = fresh_convo()
        gate_c = run_gate(convo)
        for t in range(a.sorties):
            convo.append({"role": "user",
                          "content": reg_c_turn(t, a.sorties, choices,
                                                [f > 0.5 for f in fails])})
            reply = converse(convo)
            convo.append({"role": "assistant", "content": reply})
            c = parse_route(reply, R)
            if c is None:
                c = int(rng.integers(R))
            window = np.zeros(R)
            for cc in choices[-W:]:
                window[cc] += 1
            br = softmax_br(window, L, TAU)
            j = int(rng.choice(len(br), p=br))
            fails.append(float(L[c, j]))
            choices.append(c)
            counts[c] += 1
    if choices:
        out["c_agentic"] = {"mean_mission_failure": float(np.mean(fails)),
                        "choices": choices, "n_sorties": a.sorties,
                        "repeat_rate_w": float(np.mean([c in choices[max(0, i - W):i]
                                                        for i, c in enumerate(choices) if i > 0])),
                        "gate": gate_c}

    print(json.dumps({k: v for k, v in out.items() if k != "transcript"}, indent=2))
    path = a.json_out or f"models/runs/b2_llm/{a.model.replace('/', '_')}.json"
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    out["transcript"] = transcript
    json.dump(out, open(path, "w"), indent=2)
    print(f"[written] {path}")


if __name__ == "__main__":
    main()
