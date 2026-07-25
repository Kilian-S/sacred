"""gen33 METRIC 1 DEPLOYED ROW (reported, ungated): the banked LLM forces against the FIXED
gen32 trained policy (val-selected checkpoint of gen32_confirm seed10), kgd, field seed 5100,
per model per phase, no pooling. Deployed premium = policy-suffered damage / the force's
best-response value (from metric 1). Duplicate (sites, doctrine) forces are evaluated once."""
import json
import time

import numpy as np
import torch

from scripts.eval_aerial_dyn33 import fresh_prot
from scripts.train_aerial_dyn33 import ForceInst, policy_value
from src.redforce_score import ScoreBase

SEED = 5100
SIGMA0 = 8.0
MODELS = ("llama-3.3-70b", "qwen3-27b")

if __name__ == "__main__":
    t0 = time.time()
    hist = json.load(open("models/runs/gen32_confirm/seed10.json"))["history"]
    best = min(hist, key=lambda h: h[1])
    ck = f"models/runs/gen32_confirm/seed10_ckpts/actor_ep{best[0]}.pt"
    prot = fresh_prot()
    prot.actor.load_state_dict(torch.load(ck))
    print(f"[deployed] gen32 policy = seed10 actor_ep{best[0]} (val {best[1]:.3f})", flush=True)
    base = ScoreBase("data/maps/theatre_kgd_gvardeysk_vec.json")
    scores = json.load(open("models/runs/gen33_force_scores.json"))
    cache, out = {}, {}
    for model in MODELS:
        for phase in ("single", "coordinated"):
            art = json.load(open(f"models/runs/gen33_forces/force_{model}_kgd_{phase}.json"))
            vals = []
            for rec in art["forces"]:
                sites = tuple(rec["resolved"]["sites"])
                doctrine = tuple(tuple(d) for d in rec["resolved"]["doctrine"])
                key = (sites, doctrine)
                if key not in cache:
                    inst = ForceInst(base, "kgd5100", SEED, list(sites),
                                     [tuple(d) for d in doctrine], SIGMA0 * base.scale)
                    cache[key] = policy_value(prot, inst)
                vals.append(cache[key])
            hopt = scores["cells"][f"{model}|kgd|{phase}"]["mean"]
            out[f"{model}|{phase}"] = dict(policy_mean=float(np.mean(vals)),
                                           policy_sd=float(np.std(vals)),
                                           bestresp_mean=float(hopt),
                                           premium=float(np.mean(vals) / max(hopt, 1e-9)))
            print(f"{model:14s} {phase:11s}: policy {np.mean(vals):.4f}+/-{np.std(vals):.4f} "
                  f"vs best-resp {hopt:.4f} -> premium {np.mean(vals)/max(hopt,1e-9):.2f}x "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump({"ckpt": int(best[0]), "field_seed": SEED, "rows": out},
              open("models/runs/gen33_deployed_row.json", "w"), indent=1)
    print(f"[written] models/runs/gen33_deployed_row.json [{time.time()-t0:.0f}s]")
