"""Export the real-terrain theatre for the interactive review view (headline aiming cell)."""
import json
import numpy as np
from src.baselines.aerial_lanes import lane_stack_distributions  # noqa (unused, kept parallel)
from src.baselines.interdiction_oracle import best_response_attacker
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_theatre import build_theatre_game, load_theatre, CLASS_NAME
import itertools
from math import factorial

N = 3
th = load_theatre("data/maps/theatre_kgd_gvardeysk.json")
game, menu, coords, rr, pp, S = build_theatre_game(th, K=1, menu_size=24, site_stride=3,
                                                   standoff_km=4.0, los=True)
sol = solve_multiconvoy(game, N, "mission")
_, M = objective_matrix(game, N, "mission", 1)
occs = list(itertools.combinations_with_replacement(range(game.n_routes), N))
V = np.zeros((len(occs), game.n_routes))
for i, c in enumerate(occs):
    for r in c: V[i, r] += 1
oidx = {tuple(int(x) for x in v): i for i, v in enumerate(V)}

def stack_occ(d):
    o = np.zeros(len(V))
    for r, p in enumerate(d):
        if p > 1e-12:
            v = np.zeros(game.n_routes); v[r] = N; o[oidx[tuple(int(x) for x in v)]] += p
    return o

exp = 1.0 - S.min(axis=1)
d_eq_routes = np.zeros(game.n_routes)                       # fleet-eq route marginal (stacked)
for i, o in enumerate(occs):
    if len(set(o)) == 1: d_eq_routes[o[0]] += sol.defender_strategy[i]
d_eq_routes = d_eq_routes / d_eq_routes.sum() if d_eq_routes.sum() > 0 else np.full(game.n_routes, 1/game.n_routes)
d_uni = np.full(game.n_routes, 1.0/game.n_routes)
d_inv = 1.0/np.clip(exp, 1e-9, None); d_inv /= d_inv.sum()

strategies = {"uniform_lanes": d_uni, "invrisk_lanes": d_inv, "equilibrium": d_eq_routes}
worst, brpos = {}, {}
for k, d in strategies.items():
    oc = stack_occ(d)
    per = oc @ M
    j = int(per.argmax())
    worst[k] = float(per[j])
    brpos[k] = list(game.interdiction_sets[j])           # committed hazard indices (K=1 -> 1)

out = dict(
    name=th.name, nrow=th.nrow, ncol=th.ncol, cell_km=th.cell_m/1000.0,
    grid=th.grid.tolist(), class_name={int(k): v for k, v in CLASS_NAME.items()},
    base=list(th.base), target=list(th.target),
    base_label="Kaliningrad", target_label="Gvardeysk",
    routes=[[[int(c[1]), int(c[0])] for c in r] for r in menu],   # [col=x, row=y]
    hazards=[[float(coords[h][0]/th.cell_m), float(coords[h][1]/th.cell_m),
              float(rr[h]/th.cell_m), float(pp[h])] for h in range(len(coords))],
    eq=round(float(sol.loss_mixed), 4), det=round(float(sol.loss_det), 4),
    strategies={k: [round(float(x), 5) for x in d] for k, d in strategies.items()},
    worst={k: round(v, 4) for k, v in worst.items()},
    br_pos=brpos, N=N,
)
json.dump(out, open("models/runs/theatre_view_data.json", "w"))
print("wrote models/runs/theatre_view_data.json | eq", out["eq"], "det", out["det"], "worst", out["worst"])
