"""gen33 METRIC 3: doctrinal fidelity + the two binding controls, scored on the TRUE map.

- scrambled-terrain: population value must DEGRADE materially vs the banked normal population
  (same protocol: mean over seeds 5100-5102, sigma0=8km).
- renamed-map: population value AND composition (archetype + zone choices) must NOT change
  materially.
- fidelity rubric (normal populations): fraction of agents whose emplacement matches their
  archetype's canon (sniper_overwatch -> open/reach; ambusher -> forest/concealment; blocker ->
  chokepoint or terminal region; forward_picket -> near_base; anticipator -> unconstrained),
  plus placement value as fraction of the oracle ceiling (from the metric-1 artefact).
"""
import json
from collections import Counter

import numpy as np

from src.redforce_score import ScoreBase, force_value

SEEDS = (5100, 5101, 5102)
SIGMA0 = 8.0
MODELS = ("llama-3.3-70b", "qwen3-27b")
SCORES = json.load(open("models/runs/gen33_force_scores.json"))

CANON = {
    "sniper_overwatch": lambda a: a["emplacement_zone"]["terrain"] == "open"
    or a.get("terrain_preference") == "reach",
    "ambusher": lambda a: a["emplacement_zone"]["terrain"] == "forest"
    or a.get("terrain_preference") == "concealment",
    "blocker": lambda a: a["emplacement_zone"]["region"] in ("chokepoint",
                                                             "near_target_standoff"),
    "forward_picket": lambda a: a["emplacement_zone"]["region"] == "near_base",
    "anticipator": lambda a: True,
}


def pop_value(base, path):
    art = json.load(open(path))
    vals = []
    for rec in art["forces"]:
        sites = rec["resolved"]["sites"]
        doctrine = [tuple(d) for d in rec["resolved"]["doctrine"]]
        vals.append(np.mean([force_value(base.field(sd), sites, doctrine, SIGMA0 * base.scale)
                             for sd in SEEDS]))
    agents = [a for rec in art["forces"] for a in rec["force"]["agents"]]
    return np.array(vals), agents


def comp(agents):
    return dict(arch=Counter(a["archetype"] for a in agents),
                terr=Counter(a["emplacement_zone"]["terrain"] for a in agents),
                reg=Counter(a["emplacement_zone"]["region"] for a in agents))


if __name__ == "__main__":
    base = ScoreBase("data/maps/theatre_kgd_gvardeysk_vec.json")
    out = {"controls": {}, "fidelity": {}}
    for model in MODELS:
        for phase in ("single", "coordinated"):
            normal = SCORES["cells"][f"{model}|kgd|{phase}"]
            n_vals, n_agents = pop_value(
                base, f"models/runs/gen33_forces/force_{model}_kgd_{phase}.json")
            for control in ("scrambled", "renamed"):
                c_vals, c_agents = pop_value(
                    base,
                    f"models/runs/gen33_forces_controls/force_{model}_kgd-{control}_{phase}.json")
                dv = float(c_vals.mean() - n_vals.mean())
                rel = dv / max(n_vals.mean(), 1e-9)
                nc, cc = comp(n_agents), comp(c_agents)
                overlap = sum((nc["arch"] & cc["arch"]).values()) / max(
                    sum(nc["arch"].values()), 1)
                out["controls"][f"{model}|{phase}|{control}"] = dict(
                    normal_mean=float(n_vals.mean()), control_mean=float(c_vals.mean()),
                    delta=dv, rel=float(rel), arch_overlap=float(overlap),
                    control_terr=dict(cc["terr"]), normal_terr=dict(nc["terr"]),
                    control_reg=dict(cc["reg"]), normal_reg=dict(nc["reg"]))
                print(f"{model:14s} {phase:11s} {control:9s}: normal {n_vals.mean():.4f} -> "
                      f"control {c_vals.mean():.4f} (rel {rel:+.0%}) arch-overlap {overlap:.2f} "
                      f"terr {dict(cc['terr'])}", flush=True)
    print()
    for model in MODELS:
        rows = {}
        for name in ("kgd", "ukraine", "narva"):
            for phase in ("single", "coordinated"):
                art = json.load(open(f"models/runs/gen33_forces/force_{model}_{name}_{phase}.json"))
                agents = [a for rec in art["forces"] for a in rec["force"]["agents"]]
                fid = np.mean([CANON[a["archetype"]](a) for a in agents])
                cell = SCORES["cells"][f"{model}|{name}|{phase}"]
                rows[f"{name}|{phase}"] = dict(fidelity=float(fid),
                                               vs_oracle=float(cell["mean"] / cell["oracle"]))
        fid_all = float(np.mean([r["fidelity"] for r in rows.values()]))
        vso_all = float(np.mean([r["vs_oracle"] for r in rows.values()]))
        out["fidelity"][model] = dict(cells=rows, mean_fidelity=fid_all, mean_vs_oracle=vso_all)
        cells_txt = ", ".join(f"{k}={v['fidelity']:.2f}" for k, v in rows.items())
        print(f"{model}: archetype-terrain fidelity {fid_all:.2f} ({cells_txt}) "
              f"| placement value vs oracle {vso_all:.2f}", flush=True)
    json.dump(out, open("models/runs/gen33_controls_scores.json", "w"), indent=1)
    print("[written] models/runs/gen33_controls_scores.json")
