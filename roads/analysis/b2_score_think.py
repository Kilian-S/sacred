#!/usr/bin/env python3
"""Score the B2 thinking-mode rerun.

Three cells (71-33, 35-159, Gdansk 249-95), qwen3-27b thinking ON. Registers (a)/(c) come
from the 16k gateway phase; register (b) comes from the direct-port 32k phase only (16k (b)
sidecars are excluded). Off-mode comparison rows come from the banked scored artefacts.
Validity rows report per-register n, empty-content turn retries, and gate means.

Run: PYTHONPATH=. .venv/bin/python analysis/b2_score_think.py
Writes models/runs/b2_llm/batch_think_scored.json
"""
from __future__ import annotations

import glob
import json

import numpy as np

CELLS = {
    "7133": {"dir": "batch_7133_think", "core": [0, 1, 2, 3, 4, 5], "R": 11,
             "anchors": {"loss_det": 0.4199, "eq": 0.1276, "uni_disj": 0.1666,
                         "uni_full": 0.2252, "sacred_b": 0.160,
                         "dyn_opt": 0.0313, "rot": 0.0387, "sacred_c": 0.0462,
                         "iid": 0.0967},
             "off": {"a": 0.572, "b": 0.254, "b_sd": 0.076, "c": 0.054, "c_sd": 0.043}},
    "35159": {"dir": "batch_35159_think", "core": [0, 1, 2, 3], "R": 12,
              "anchors": {"loss_det": 0.699, "eq": 0.206, "uni_disj": 0.250,
                          "uni_full": 0.442, "sacred_b": 0.256,
                          "dyn_opt": 0.0413, "rot": 0.0413, "sacred_c": 0.050,
                          "iid": 0.1468},
              "off": {"a": 0.841, "b": 0.523, "b_sd": 0.161, "c": 0.297, "c_sd": 0.176}},
    "gdansk": {"dir": "batch_gdansk_think", "core": None, "R": 10,
               "anchors": {"loss_det": 0.740, "eq": 0.302, "uni_disj": 0.333,
                           "uni_full": 0.694, "sacred_b": None,
                           "dyn_opt": 0.0723, "rot": 0.2069, "sacred_c": 0.098,
                           "iid": 0.223},
               "off": {"a": 0.867, "b": 0.354, "b_sd": 0.066, "c": 0.394, "c_sd": 0.047}},
}


def empties(d):
    return sum(1 for x in d.get("transcript", []) if x.get("empty_retry"))


def main():
    out = {}
    for tag, cfg in CELLS.items():
        base = f"models/runs/b2_llm/{cfg['dir']}"
        row = {"anchors": cfg["anchors"], "off_mode": cfg["off"]}
        # (a)
        ae, routes, gates, emp = [], [], [], 0
        for f in sorted(glob.glob(f"{base}/qwen3-27b_a_seed*.json")):
            d = json.load(open(f))
            a = d.get("a_deterministic")
            emp += empties(d)
            if a:
                ae.append(a["expl"]); routes.append(a["route"]); gates.append(a["gate"])
        row["a"] = {"n": len(ae), "mean": float(np.mean(ae)), "sd": float(np.std(ae)),
                    "routes": routes, "gate_mean": float(np.mean(gates)),
                    "empty_retries": emp}
        # (b): direct-port 32k files only (sidecars carry _16k suffix and are excluded)
        be, gates, core_mass, dists, emp = [], [], [], [], 0
        for f in sorted(glob.glob(f"{base}/qwen3-27b_b_seed*.json")):
            if "_16k" in f:
                continue
            d = json.load(open(f))
            b = d.get("b_stated")
            emp += empties(d)
            if b:
                p = np.array(b["dist"])
                be.append(b["expl"]); gates.append(b["gate"]); dists.append(b["dist"])
                if cfg["core"] is not None:
                    core_mass.append(float(p[cfg["core"]].sum()))
        row["b"] = {"n": len(be), "mean": float(np.mean(be)), "sd": float(np.std(be)),
                    "min": float(np.min(be)), "max": float(np.max(be)),
                    "per_seed": [round(float(x), 4) for x in be],
                    "gate_mean": float(np.mean(gates)),
                    "core_mass_mean": (float(np.mean(core_mass)) if core_mass else None),
                    "empty_retries": emp, "dists": dists}
        # (c)
        ce, reps, gates, emp = [], [], [], 0
        for f in sorted(glob.glob(f"{base}/qwen3-27b_c_seed*.json")):
            d = json.load(open(f))
            c = d.get("c_agentic")
            emp += empties(d)
            if c:
                ce.append(c["mean_mission_failure"]); reps.append(c["repeat_rate_w"])
                gates.append(c["gate"])
        row["c"] = {"n": len(ce), "mean": float(np.mean(ce)), "sd": float(np.std(ce)),
                    "best": float(np.min(ce)), "per_episode": [round(float(x), 4) for x in ce],
                    "repeat_mean": float(np.mean(reps)), "gate_mean": float(np.mean(gates)),
                    "empty_retries": emp}
        out[tag] = row
    json.dump(out, open("models/runs/b2_llm/batch_think_scored.json", "w"), indent=1)
    for tag, r in out.items():
        an, off = r["anchors"], r["off_mode"]
        print(f"\n=== {tag} (thinking ON vs off) ===")
        print(f"(a) n={r['a']['n']} {r['a']['mean']:.3f} (off {off['a']}) "
              f"gate {r['a']['gate_mean']:.1f}/3 routes {r['a']['routes']}")
        print(f"(b) n={r['b']['n']} {r['b']['mean']:.3f} +/- {r['b']['sd']:.3f} "
              f"[{r['b']['min']:.3f}, {r['b']['max']:.3f}] (off {off['b']} +/- {off['b_sd']}) "
              f"gate {r['b']['gate_mean']:.1f}/3 core-mass {r['b']['core_mass_mean']} "
              f"| anchors eq {an['eq']} uni-disj {an['uni_disj']} uni-full {an['uni_full']} "
              f"SACRED {an['sacred_b']}")
        print(f"(c) n={r['c']['n']} {r['c']['mean']:.3f} +/- {r['c']['sd']:.3f} "
              f"best {r['c']['best']:.3f} (off {off['c']} +/- {off['c_sd']}) "
              f"repeat {r['c']['repeat_mean']:.2f} gate {r['c']['gate_mean']:.1f}/3")
    print("\n[written] models/runs/b2_llm/batch_think_scored.json")


if __name__ == "__main__":
    main()
