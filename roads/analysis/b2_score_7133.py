#!/usr/bin/env python3
"""Score the B2 71-33 batch (pre-registered 2026-08-12; anchors in b2_7133_anchors.json).

Aggregates models/runs/b2_llm/batch_7133/*.json into per-model rows: register (a) mean
worst-case, (b) mean +/- sd exploitability with the pre-registered core-mass/distance
readout, (c) mean +/- sd realised mission failure with best episode and repeat rate.
Writes models/runs/b2_llm/batch_7133_scored.json.

Run: PYTHONPATH=. .venv/bin/python analysis/b2_score_7133.py
"""
from __future__ import annotations

import glob
import json

import numpy as np

CORE = [0, 1, 2, 3, 4, 5]          # disjoint core, from the anchor probe
R = 11
ANCHORS = json.load(open("models/runs/b2_llm/b2_7133_anchors.json"))["got"]
INV = np.zeros(R)
UNI = np.ones(R) / R


def build_ref_dists():
    # rebuild the inv-vuln disjoint stack weights exactly as the anchor probe did
    from analysis.critique_followup_probes import disjoint_subset
    from scripts.train_b1lite1 import stacked_L
    from scripts.train_generalist import CITY_PATHS
    from src.envs.multiconvoy_interdiction import make_multiconvoy_env
    nodes_path, edges_path = CITY_PATHS["kaliningrad"]
    env = make_multiconvoy_env(od=("71", "33"), N=3, K=1, k_extra_routes=8,
                               menu_select=True, edge_vuln_band=(0.15, 0.95),
                               interception_loss=10.0, seed=0,
                               nodes_path=nodes_path, edges_path=edges_path)
    L = stacked_L(env.game, 3)
    dis = disjoint_subset([set(e) for e in env.game.route_edges])
    assert dis == CORE, dis
    q = np.array([L[r].max() for r in dis])
    INV[np.asarray(dis)] = (1.0 / q) / (1.0 / q).sum()


def main():
    build_ref_dists()
    out = {"anchors": ANCHORS, "models": {}}
    for model in ["llama-3.3-70b", "qwen3-27b"]:
        rows = {}
        # (a)
        a_expl, a_routes, a_gates = [], [], []
        for f in sorted(glob.glob(f"models/runs/b2_llm/batch_7133/{model}_a_seed*.json")):
            d = json.load(open(f)).get("a_deterministic")
            if d:
                a_expl.append(d["expl"]); a_routes.append(d["route"]); a_gates.append(d["gate"])
        rows["a"] = {"n": len(a_expl), "mean_expl": float(np.mean(a_expl)),
                     "sd": float(np.std(a_expl)), "routes": a_routes,
                     "gate_mean": float(np.mean(a_gates))}
        # (b)
        b_expl, b_gates, core_mass, d_inv, d_uni, dists = [], [], [], [], [], []
        for f in sorted(glob.glob(f"models/runs/b2_llm/batch_7133/{model}_b_seed*.json")):
            d = json.load(open(f)).get("b_stated")
            if d:
                b_expl.append(d["expl"]); b_gates.append(d["gate"])
                p = np.array(d["dist"]); dists.append(d["dist"])
                core_mass.append(float(p[CORE].sum()))
                d_inv.append(float(np.abs(p - INV).sum() / 2))
                d_uni.append(float(np.abs(p - UNI).sum() / 2))
        rows["b"] = {"n": len(b_expl), "mean_expl": float(np.mean(b_expl)),
                     "sd": float(np.std(b_expl)), "gate_mean": float(np.mean(b_gates)),
                     "core_mass_mean": float(np.mean(core_mass)),
                     "tv_to_invvuln_stack": float(np.mean(d_inv)),
                     "tv_to_uniform_full": float(np.mean(d_uni)),
                     "dists": dists}
        # (c)
        c_mean, c_rep, c_gates = [], [], []
        for f in sorted(glob.glob(f"models/runs/b2_llm/batch_7133/{model}_c_seed*.json")):
            d = json.load(open(f)).get("c_agentic")
            if d:
                c_mean.append(d["mean_mission_failure"])
                c_rep.append(d["repeat_rate_w"]); c_gates.append(d["gate"])
        rows["c"] = {"n": len(c_mean), "mean": float(np.mean(c_mean)),
                     "sd": float(np.std(c_mean)), "best": float(np.min(c_mean)),
                     "worst": float(np.max(c_mean)), "per_episode": c_mean,
                     "repeat_rate_mean": float(np.mean(c_rep)),
                     "gate_mean": float(np.mean(c_gates))}
        out["models"][model] = rows
    json.dump(out, open("models/runs/b2_llm/batch_7133_scored.json", "w"), indent=2)
    for model, rows in out["models"].items():
        print(f"\n=== {model} ===")
        print(f"(a) n={rows['a']['n']} mean worst-case {rows['a']['mean_expl']:.3f} "
              f"(loss_det {ANCHORS['loss_det']:.3f}) routes {rows['a']['routes']}")
        print(f"(b) n={rows['b']['n']} expl {rows['b']['mean_expl']:.3f} +/- {rows['b']['sd']:.3f} "
              f"(v* {ANCHORS['v_eq']:.4f} = inv-vuln stack; uni-disj {ANCHORS['stack_uniform_disjoint']:.4f}; "
              f"uni-full {ANCHORS['stack_uniform_full']:.4f}) gate {rows['b']['gate_mean']:.1f}/3 "
              f"core-mass {rows['b']['core_mass_mean']:.2f} "
              f"TV(inv) {rows['b']['tv_to_invvuln_stack']:.2f} TV(uni) {rows['b']['tv_to_uniform_full']:.2f}")
        print(f"(c) n={rows['c']['n']} mean {rows['c']['mean']:.3f} +/- {rows['c']['sd']:.3f} "
              f"best {rows['c']['best']:.3f} "
              f"(opt {ANCHORS['dyn_opt']:.4f}; rotation {ANCHORS['dyn_rotation']:.4f}; "
              f"iid_eq {ANCHORS['dyn_iid_eq']:.4f}) repeat {rows['c']['repeat_rate_mean']:.2f}")
    print("\n[written] models/runs/b2_llm/batch_7133_scored.json")


if __name__ == "__main__":
    main()
