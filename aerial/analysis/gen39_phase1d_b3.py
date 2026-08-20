"""Scores the gen39 phase-1d bar B3: evolved LLM forces against trained SACRED defenders,
compared with the heuristic force, over all six held-out fields."""
import json, os, numpy as np, torch
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
from pathlib import Path
from scripts.train_gen39_conceal import (Inst, TheatreEnv, narva_base, policy_value, N, DOC32,
                                         K, W, TAU, TEST_FIELDS)
from src.envs.aerial_conceal import choose_force, resample_field
from analysis.gen39_compose import place, doctrines_of
from src.agents.sac import ProtagonistSAC

base = narva_base()
log = json.loads(Path("models/runs/gen39_phase1d.json").read_text())
best = max(log, key=lambda x: x["irreducible"])
base0 = max([x for x in log if x["round"] == 0], key=lambda x: x["irreducible"])
FORCES = {"llm evolved (best)": best["force"], "llm round-0": base0["force"]}

insts, env = {}, None
for fld in TEST_FIELDS:
    pp = base.lethality(resample_field(base.coords, fld), hidden_leth=1.0)
    for lab, f in FORCES.items():
        insts[(lab, fld)] = Inst(base, f"{lab}@{fld}", fld, sites=place(f, base, pp),
                                 doctrines=doctrines_of(f))
    L, _, _ = choose_force(base, pp, "mixed", K, np.random.default_rng(fld), w=W, tau=TAU,
                           doctrine=DOC32)
    insts[("heuristic force", fld)] = Inst(base, f"heur@{fld}", fld, sites=L)
    if env is None:
        env = TheatreEnv(base.menu, insts[("heuristic force", fld)].g.game,
                         insts[("heuristic force", fld)].S_field, N=N)

ckpts = []
for arm in ("llm", "heuristic", "random"):
    for s in (0, 1, 2):
        run = json.loads(Path(f"models/runs/gen39_step3/{arm}_seed{s}.json").read_text())
        srt = min(run["history"], key=lambda h: h["val"])["sortie"]
        p = Path(f"models/runs/gen39_step3/{arm}_seed{s}_ckpts/actor_ep{srt}.pt")
        if p.exists():
            prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2,
                                  heads=4, reward_scale=1.0, device="cpu", role_alpha=True)
            prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
            prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(3))
            prot.actor.route_feats = None
            prot.actor.load_state_dict(torch.load(p, weights_only=True))
            ckpts.append((arm, s, prot))
print(f"{len(ckpts)} trained defenders x {len(TEST_FIELDS)} fields\n")
labs = list(FORCES) + ["heuristic force"]
print(f'{"field":>6s} ' + " ".join(f'{l[:16]:>17s}' for l in labs))
per = {l: [] for l in labs}
for fld in TEST_FIELDS:
    row = []
    for l in labs:
        v = float(np.mean([policy_value(p, insts[(l, fld)], env) for _, _, p in ckpts]))
        per[l].append(v); row.append(v)
    print(f'{fld:6d} ' + " ".join(f'{x:17.4f}' for x in row))
print()
h = np.mean(per["heuristic force"])
for l in FORCES:
    v, w = np.mean(per[l]), sum(a > b for a, b in zip(per[l], per["heuristic force"]))
    print(f'B3 "{l}": pooled {v:.4f} vs heuristic {h:.4f} ({v/h:.2f}x), beats it on {w}/6 fields'
          f' -> {"PASS" if v > h and w >= 4 else "FAIL"}')
