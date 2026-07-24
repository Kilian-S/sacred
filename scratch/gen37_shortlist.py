#!/usr/bin/env python3
"""gen37 shortlist builder + mechanism rows (pre-registered: experiments/gen37_reasoning_curation.md).

For every gen29 cell (16 train + 4 val + 6 held-out), builds three route-triple shortlists of
size M=50 (llm / random / heuristic) and computes the oracle-exact mechanism rows BEFORE any
training: CONTAINMENT (mass of the true optimal mixture dstar inside the shortlist) and
LP-OVER-SHORTLIST (exact game value restricted to the shortlist = the ceiling any policy in it
can reach), beside the full equilibrium and the fitted cap.

The LLM (llama-3.3-70b via the localhost:18080 tunnel) sees per-stream route summaries
(worst-edge vulnerability + compact edge-id sets) and the mission rules; NO payoff/equilibrium
numbers. One call/instance, temp 0.2, guided retry <= 3 on invalid JSON. Full transcripts +
shortlists committed under models/runs/gen37_reasoning_curation/.

Run (gen29 worktree): OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  /Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python scratch/gen37_shortlist.py
"""
from __future__ import annotations

import itertools
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from scripts.train_multiod_generalist import Inst
from src.baselines.multiconvoy_oracle import _row_minimiser

M = 50
BASE = "http://localhost:18080/v1/chat/completions"
KEY = "iits-local-key"
MODEL = "llama-3.3-70b"
OUT = Path("models/runs/gen37_reasoning_curation")
TRANSC = OUT / "transcripts"


def edge_id_map(env):
    ids = {}
    per_stream = []
    for f in range(env.F):
        rr = []
        for es in env.route_edges[f]:
            lst = []
            for e in es:
                if e not in ids:
                    ids[e] = len(ids)
                lst.append(ids[e])
            rr.append(sorted(lst))
        per_stream.append(rr)
    return per_stream  # per_stream[f][r] = sorted list of int edge ids


def triple_index(env, a, b, c):
    R = [len(rs) for rs in env.route_sets]
    return (a * R[1] + b) * R[2] + c


def containment(env, dstar, triples):
    return float(sum(dstar[triple_index(env, *t)] for t in triples))


def lp_over_shortlist(env, triples):
    M_full = env.obj_matrix
    rows = [triple_index(env, *t) for t in triples]
    sub = M_full[rows, :]
    v, _ = _row_minimiser(sub)
    return float(v)


def random_shortlist(env, seed):
    rng = np.random.default_rng(seed)
    R = [len(rs) for rs in env.route_sets]
    allt = list(itertools.product(range(R[0]), range(R[1]), range(R[2])))
    idx = rng.choice(len(allt), size=min(M, len(allt)), replace=False)
    return [list(allt[i]) for i in idx]


def heuristic_shortlist(env, edges):
    R = [len(rs) for rs in env.route_sets]
    wv = [env.worst_vuln[f] for f in range(env.F)]
    scored = []
    for a, b, c in itertools.product(range(R[0]), range(R[1]), range(R[2])):
        ea, eb, ec = set(edges[0][a]), set(edges[1][b]), set(edges[2][c])
        shared = max(len(ea & eb), len(ea & ec), len(eb & ec))     # worst pairwise overlap
        vuln = wv[0][a] + wv[1][b] + wv[2][c]
        scored.append((shared, vuln, [a, b, c]))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [t for _, _, t in scored[:M]]


def llm_call(messages, temperature, max_tokens=4096):
    body = json.dumps(dict(model=MODEL, messages=messages, temperature=temperature,
                           max_tokens=max_tokens)).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def build_prompt(env, edges):
    R = [len(rs) for rs in env.route_sets]
    lines = ["You are a route-coordination analyst. Three convoy streams (0,1,2) each pick ONE "
             "route. An ambush places ONE asset on a single edge; the MISSION FAILS if ANY of the "
             "three streams crosses the ambushed edge. Lower shared edges across the three chosen "
             "routes and lower per-route vulnerability both reduce failure. Your job: from all "
             f"route combinations, shortlist the {M} triples (r0,r1,r2) most worth keeping in a "
             "coordination playbook - diverse, mutually edge-disjoint where possible, low "
             "vulnerability. Edges are integer ids; two routes sharing an id share that road.",
             ""]
    for f in range(env.F):
        lines.append(f"Stream {f} routes (index: vuln, edges):")
        for r in range(R[f]):
            lines.append(f"  {r}: vuln={float(env.worst_vuln[f][r]):.2f} edges={edges[f][r]}")
        lines.append("")
    lines.append(f"Return STRICT JSON only: {{\"triples\": [[r0,r1,r2], ...]}} with exactly {M} "
                 f"distinct triples, r0 in 0..{R[0]-1}, r1 in 0..{R[1]-1}, r2 in 0..{R[2]-1}. "
                 "No prose.")
    return "\n".join(lines)


def parse_triples(text, env):
    R = [len(rs) for rs in env.route_sets]
    s = text.find("{"); e = text.rfind("}")
    obj = json.loads(text[s:e + 1])
    out, seen = [], set()
    for t in obj["triples"]:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        if 0 <= a < R[0] and 0 <= b < R[1] and 0 <= c < R[2] and (a, b, c) not in seen:
            seen.add((a, b, c)); out.append([a, b, c])
    return out


def llm_shortlist(env, edges, name):
    prompt = build_prompt(env, edges)
    msgs = [{"role": "user", "content": prompt}]
    transcript = {"instance": name, "prompt": prompt, "attempts": []}
    triples = []
    for attempt in range(3):
        try:
            content = llm_call(msgs, temperature=0.2)
        except (urllib.error.URLError, TimeoutError) as ex:
            transcript["attempts"].append({"error": str(ex)}); time.sleep(3); continue
        transcript["attempts"].append({"raw": content[:6000]})
        try:
            triples = parse_triples(content, env)
        except Exception as ex:  # noqa: BLE001
            triples = []
            transcript["attempts"][-1]["parse_error"] = str(ex)
        if len(triples) >= M:
            triples = triples[:M]; break
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"That was invalid or had {len(triples)} valid "
                 f"distinct triples. Return STRICT JSON with exactly {M} distinct valid triples."}]
    transcript["n_valid"] = len(triples)
    (TRANSC / f"{name}.json").write_text(json.dumps(transcript, indent=1))
    return triples


def main():
    TRANSC.mkdir(parents=True, exist_ok=True)
    sc = json.load(open("models/runs/gen29_screen.json"))
    cells = ([("train", c) for c in [sc["headline"]] + sc["pool"]]
             + [("val", c) for c in sc["validation"]]
             + [("held", c) for c in sc["held_out"]])
    arms = {"llm": {}, "random": {}, "heuristic": {}}
    mech = {"llm": [], "random": [], "heuristic": []}
    for gi, (split, spec) in enumerate(cells):
        it = Inst(spec)
        env = it.env
        name = f"{split}_{spec['s']}_{'-'.join(spec['targets'])}"
        v_eq, dstar = _row_minimiser(env.obj_matrix)
        edges = edge_id_map(env)
        sls = {"random": random_shortlist(env, 9000 + gi),
               "heuristic": heuristic_shortlist(env, edges),
               "llm": llm_shortlist(env, edges, name)}
        for arm, S in sls.items():
            if arm == "llm" and len(S) < M:
                # top up an under-length LLM list with heuristic picks (disclosed), never random
                extra = [t for t in heuristic_shortlist(env, edges) if t not in S]
                S = (S + extra)[:M]
            arms[arm][it.name] = S
            cont = containment(env, dstar, S)
            lp = lp_over_shortlist(env, S)
            mech[arm].append(dict(instance=it.name, split=split, n=len(S), containment=cont,
                                  lp_over_shortlist=lp, eq=float(v_eq), cap=float(it.cap),
                                  lp_ratio_eq=lp / float(v_eq)))
        print(f"{name}: eq {v_eq:.4f} cap {it.cap:.4f} | containment "
              f"llm {mech['llm'][-1]['containment']:.2f} heur "
              f"{mech['heuristic'][-1]['containment']:.2f} rand "
              f"{mech['random'][-1]['containment']:.2f} | LP/eq "
              f"llm {mech['llm'][-1]['lp_ratio_eq']:.3f} heur "
              f"{mech['heuristic'][-1]['lp_ratio_eq']:.3f} rand "
              f"{mech['random'][-1]['lp_ratio_eq']:.3f}", flush=True)

    for arm in arms:
        (OUT / f"shortlists_{arm}.json").write_text(json.dumps(arms[arm], indent=1))
    (OUT / "mechanism_rows.json").write_text(json.dumps(mech, indent=1))

    def summ(rows, split):
        r = [x for x in rows if x["split"] == split]
        return (float(np.mean([x["containment"] for x in r])),
                float(np.mean([x["lp_ratio_eq"] for x in r])))
    print("\n=== MECHANISM SUMMARY (containment, LP/eq) ===")
    for split in ("train", "val", "held"):
        for arm in ("llm", "heuristic", "random"):
            c, lp = summ(mech[arm], split)
            print(f"  {split:5s} {arm:9s}: containment {c:.3f}  LP-over-shortlist/eq {lp:.3f}")
    print(f"\nwrote {OUT}/shortlists_*.json + mechanism_rows.json + transcripts/")


if __name__ == "__main__":
    main()
