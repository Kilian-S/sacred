#!/usr/bin/env python3
"""gen37 decisive small-M ceiling test (oracle + LLM, NO training).

The M-sweep (scratch/gen37_mchoice.py) showed the reasoning-vs-naive question can only be
answered where the prune is aggressive (small M): there random's LP-over-shortlist ceiling is
bad (held pooled 1.48 @M=15, 1.63 @M=10, 1.93 @M=6), so a GOOD curator can separate. At M=50
random already retains the value (1.10) and the LLM's 50-list was worse than random (1.21 ~
random@20). This probe makes FRESH LLM calls at M in {10,15} for every gen29 cell and compares
the three arms' ceilings + containment head-to-head, which fully answers "is the LLM a good
route curator" WITHOUT spending any training compute. Training is launched only if the LLM
ceiling beats random at some M (the pre-registered mechanism clause deciding Tier interpretation).

Run (gen29 worktree): OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  /Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python scratch/gen37_smallM.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scratch.gen37_shortlist as g37
from scratch.gen37_shortlist import (
    Inst, _row_minimiser, build_prompt, containment, edge_id_map, llm_call,
    lp_over_shortlist, parse_triples, random_shortlist)
from scratch.gen37_mchoice import _heur

MS = [10, 15]
OUT = Path("models/runs/gen37_reasoning_curation")
TR = OUT / "transcripts_smallM"


def llm_shortlist_M(env, edges, name, M):
    g37.M = M                                     # build_prompt reads the module global
    prompt = build_prompt(env, edges)
    msgs = [{"role": "user", "content": prompt}]
    triples = []
    rec = {"instance": name, "M": M, "attempts": []}
    for _ in range(3):
        try:
            content = llm_call(msgs, temperature=0.2)
        except Exception as ex:  # noqa: BLE001
            rec["attempts"].append({"error": str(ex)}); continue
        rec["attempts"].append({"raw": content[:4000]})
        try:
            triples = parse_triples(content, env)[:M]
        except Exception as ex:  # noqa: BLE001
            triples = []; rec["attempts"][-1]["parse_error"] = str(ex)
        if len(triples) >= M:
            break
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Return STRICT JSON with exactly {M} distinct "
                 f"valid triples."}]
    rec["n_valid"] = len(triples)
    (TR / f"{name}_M{M}.json").write_text(json.dumps(rec, indent=1))
    return triples


def main():
    TR.mkdir(parents=True, exist_ok=True)
    sc = json.load(open("models/runs/gen29_screen.json"))
    cells = ([("train", c) for c in [sc["headline"]] + sc["pool"]]
             + [("val", c) for c in sc["validation"]]
             + [("held", c) for c in sc["held_out"]])
    rows = {M: {"llm": [], "random": [], "heuristic": []} for M in MS}
    for gi, (split, spec) in enumerate(cells):
        it = Inst(spec); env = it.env
        name = f"{split}_{spec['s']}_{'-'.join(spec['targets'])}"
        v_eq, dstar = _row_minimiser(env.obj_matrix)
        edges = edge_id_map(env)
        for M in MS:
            sls = {"random": random_shortlist(env, 9000 + gi)[:M],
                   "heuristic": _heur(env, edges, M),
                   "llm": llm_shortlist_M(env, edges, name, M)}
            for arm, S in sls.items():
                if not S:
                    continue
                rows[M][arm].append(dict(instance=it.name, split=split, n=len(S),
                                         lp_ratio_eq=lp_over_shortlist(env, S) / float(v_eq),
                                         containment=containment(env, dstar, S)))
        print(f"{name}: done", flush=True)

    json.dump(rows, open(OUT / "smallM_rows.json", "w"), indent=1, default=float)
    print("\n=== held-out pooled LP-over-shortlist / eq (lower = better curation) ===")
    print(f"{'M':>4} {'llm':>10} {'random':>10} {'heuristic':>10}  llm<rand cells")
    for M in MS:
        def pooled(arm):
            r = [x['lp_ratio_eq'] for x in rows[M][arm] if x['split'] == 'held']
            return float(np.mean(r)) if r else float('nan')
        hl = {x['instance']: x['lp_ratio_eq'] for x in rows[M]['llm'] if x['split'] == 'held'}
        hr = {x['instance']: x['lp_ratio_eq'] for x in rows[M]['random'] if x['split'] == 'held'}
        wins = sum(hl[k] < hr[k] for k in hl if k in hr)
        print(f"{M:>4} {pooled('llm'):>10.3f} {pooled('random'):>10.3f} "
              f"{pooled('heuristic'):>10.3f}  {wins}/{len(hl)}")
    print("\ncontainment (held pooled):")
    for M in MS:
        for arm in ("llm", "random", "heuristic"):
            c = [x['containment'] for x in rows[M][arm] if x['split'] == 'held']
            print(f"  M={M} {arm:9s} {float(np.mean(c)) if c else 0:.3f}")
    print(f"wrote {OUT}/smallM_rows.json + transcripts_smallM/")


if __name__ == "__main__":
    main()
