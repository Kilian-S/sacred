#!/usr/bin/env python3
"""Step-2 thinking row via the gen42 arms (pre-registered 2026-08-13; zero new calls).

Applies the BANKED step-2 headline aggregation (per-arm median over forces x fields of the
vs-best-observing-defender column, index 2 of each scores.json tuple, plus the per-field,
pooled, and above-random clauses and the relabel collapse) to three directories: the banked
step-2 record (the consistency anchor, which must reproduce exactly), and the gen42 crown
OFF and ON arms, whose generation followed the step-2 protocol verbatim on the corrected
brief. Clears the gen42 standing recomputation flag in the same pass.

Run: PYTHONPATH=. ../sacred/.venv/bin/python analysis/gen39_step2_think_rescore.py
Writes models/runs/gen42_ladder/step2_rescore.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

FIELDS = ["5100", "5101", "5102"]
DIRS = {
    "banked_step2": "models/runs/gen39_compose",
    "crown_off": "models/runs/gen42_ladder/qwen3-27b",
    "crown_on": "models/runs/gen42_ladder/qwen3-27b_think",
}
ANCHOR = {  # banked step-2 table, must reproduce to 4 dp before any new number is read
    "llm:llama-3.3-70b": 0.0747, "llm:qwen3-27b": 0.0613, "heuristic": 0.0603,
    "random": 0.0123, "relabel:llama-3.3-70b": 0.0059, "relabel:qwen3-27b": 0.0057,
}


def arm_of(key):
    parts = key.split(":")
    return parts[0] + (":" + parts[1] if key.startswith(("llm", "relabel")) else "")


def table(res):
    groups = {}
    for k in res:
        groups.setdefault(arm_of(k), []).append(k)
    med = lambda ks: float(np.median([res[k][f][2] for k in ks for f in FIELDS]))  # noqa: E731
    out = {}
    for arm, ks in sorted(groups.items()):
        pf = [float(np.median([res[k][f][2] for k in ks])) for f in FIELDS]
        out[arm] = {"n": len(ks), "pooled_median_vs_observing": med(ks), "per_field": pf,
                    "median_vs_omniscient": float(np.median([res[k][f][0] for k in ks
                                                             for f in FIELDS])),
                    "mean_coverage": float(np.mean([res[k][f][3] for k in ks
                                                    for f in FIELDS]))}
    rnd = groups.get("random", [])
    if rnd:
        out["_random_mean_vs_observing"] = float(np.mean([res[k][f][2] for k in rnd
                                                          for f in FIELDS]))
    return out


def bars(t, llm_arm, relabel_arm):
    heur = t["heuristic"]
    llm = t[llm_arm]
    wins = sum(1 for lf, hf in zip(llm["per_field"], heur["per_field"]) if lf > hf)
    return {"fields_beating_heuristic": f"{wins}/3",
            "pooled_beats_heuristic": llm["pooled_median_vs_observing"]
            > heur["pooled_median_vs_observing"],
            "above_random_mean": llm["pooled_median_vs_observing"]
            > t["_random_mean_vs_observing"],
            "relabel_collapse_factor": (llm["pooled_median_vs_observing"]
                                        / max(t[relabel_arm]["pooled_median_vs_observing"],
                                              1e-9))}


def main():
    out = {}
    for name, d in DIRS.items():
        res = json.load(open(pathlib.Path(d) / "scores.json"))
        t = table(res)
        out[name] = {"table": t}
        if "llm:qwen3-27b" in t:
            out[name]["bars_qwen"] = bars(t, "llm:qwen3-27b", "relabel:qwen3-27b")
        if "llm:llama-3.3-70b" in t:
            out[name]["bars_llama"] = bars(t, "llm:llama-3.3-70b", "relabel:llama-3.3-70b")

    anchor_verdicts = {}
    bt = out["banked_step2"]["table"]
    for arm, exp in ANCHOR.items():
        got = bt[arm]["pooled_median_vs_observing"]
        anchor_verdicts[arm] = "PASS" if abs(got - exp) < 5e-5 else \
            f"FAIL (got {got:.4f} vs banked {exp})"
    out["anchor_verdicts"] = anchor_verdicts
    out["anchor_all_pass"] = all(v == "PASS" for v in anchor_verdicts.values())

    print(json.dumps({k: v for k, v in out.items() if k != "banked_step2"}, indent=1))
    print("\nANCHORS:", json.dumps(anchor_verdicts, indent=1))
    for name in DIRS:
        t = out[name]["table"]
        q = t.get("llm:qwen3-27b")
        if q:
            print(f"{name:14s} qwen n={q['n']:2d} pooled {q['pooled_median_vs_observing']:.4f} "
                  f"per-field {[round(x, 4) for x in q['per_field']]} "
                  f"relabel {t['relabel:qwen3-27b']['pooled_median_vs_observing']:.4f}")
    path = pathlib.Path("models/runs/gen42_ladder/step2_rescore.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[written] {path}")


if __name__ == "__main__":
    main()
