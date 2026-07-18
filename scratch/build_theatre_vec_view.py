"""Export the continuous vector theatre + LOS footprints for the SVG operations map."""
import json, itertools
import numpy as np
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_theatre_vec import (load_vec_theatre, build_theatre_game,
                                         engagement_footprint, TERRAIN)
N = 3
th = load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json")
game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(th, K=1, n_lanes=14, n_terrain=12,
                                                             spacing_km=2.0, standoff_km=4.0)
sol = solve_multiconvoy(game, N, "mission")
_, M = objective_matrix(game, N, "mission", 1)
occs = list(itertools.combinations_with_replacement(range(game.n_routes), N))
V = np.zeros((len(occs), game.n_routes))
for i, c in enumerate(occs):
    for r in c: V[i, r] += 1
oidx = {tuple(int(x) for x in v): i for i, v in enumerate(V)}
def stack(d):
    o = np.zeros(len(V))
    for r, p in enumerate(d):
        if p > 1e-12:
            v = np.zeros(game.n_routes); v[r] = N; o[oidx[tuple(int(x) for x in v)]] += p
    return o
exp = 1.0 - S.min(axis=1)
d_eq = np.zeros(game.n_routes)
for i, o in enumerate(occs):
    if len(set(o)) == 1: d_eq[o[0]] += sol.defender_strategy[i]
d_eq = d_eq/d_eq.sum() if d_eq.sum() > 0 else np.full(game.n_routes, 1/game.n_routes)
d_uni = np.zeros(game.n_routes); d_uni[lane_idx] = 1/len(lane_idx)          # naive: lanes only
d_inv = np.zeros(game.n_routes); d_inv[lane_idx] = 1/np.clip(exp[lane_idx], 1e-9, None); d_inv /= d_inv.sum()
strat = {"uniform_lanes": d_uni, "invrisk_lanes": d_inv, "equilibrium": d_eq}
worst, brpos = {}, {}
for k, d in strat.items():
    per = stack(d) @ M; j = int(per.argmax())
    worst[k] = float(per[j]); brpos[k] = [int(x) for x in game.interdiction_sets[j]]
# LOS footprints for ALL candidate sites (viewshed polygons)
foot = [engagement_footprint(th, coords[h], rr[h], n_rays=72) for h in range(len(coords))]
vecd = json.load(open("data/maps/theatre_kgd_gvardeysk_vec.json"))
out = dict(
    W=th.W, H=th.H, base=th.base.tolist(), target=th.target.tolist(),
    base_label=vecd["base"]["label"], target_label=vecd["target"]["label"],
    classes={k: [[[round(x, 2), round(y, 2)] for x, y in ring] for ring in v]
             for k, v in vecd["classes"].items()},
    routes=[[[round(float(x), 2), round(float(y), 2)] for x, y in r[::2]] for r in menu],
    lane_idx=lane_idx,
    route_exposure=[round(float(e), 3) for e in exp],
    hazards=[[round(float(coords[h][0]), 2), round(float(coords[h][1]), 2),
              round(float(rr[h]), 2), round(float(pp[h]), 2)] for h in range(len(coords))],
    footprints=foot,
    eq=round(float(sol.loss_mixed), 4), det=round(float(sol.loss_det), 4),
    strategies={k: [round(float(x), 4) for x in d] for k, d in strat.items()},
    worst={k: round(v, 4) for k, v in worst.items()}, br_pos=brpos, N=N,
)
json.dump(out, open("models/runs/theatre_vec_view.json", "w"))
print("wrote view | menu", len(menu), "lanes", len(lane_idx), "sites", len(coords),
      "| eq", out["eq"], "det", out["det"], "naive", round(min(worst['uniform_lanes'],worst['invrisk_lanes']),3))
