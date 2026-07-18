"""Export the continuous vector theatre for the SVG operations-map artifact."""
import json, itertools
import numpy as np
from src.baselines.interdiction_oracle import best_response_attacker
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_theatre_vec import load_vec_theatre, build_theatre_game, TERRAIN
N = 3
th = load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json")
game, menu, coords, rr, pp, S = build_theatre_game(th, K=1, menu_size=24, spacing_km=2.0,
                                                   standoff_km=4.0, los=True)
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
d_uni = np.full(game.n_routes, 1/game.n_routes)
d_inv = 1/np.clip(exp, 1e-9, None); d_inv /= d_inv.sum()
strat = {"uniform_lanes": d_uni, "invrisk_lanes": d_inv, "equilibrium": d_eq}
worst, brpos = {}, {}
for k, d in strat.items():
    per = stack(d) @ M; j = int(per.argmax())
    worst[k] = float(per[j]); brpos[k] = [int(x) for x in game.interdiction_sets[j]]
vecd = json.load(open("data/maps/theatre_kgd_gvardeysk_vec.json"))
out = dict(
    W=th.W, H=th.H, base=th.base.tolist(), target=th.target.tolist(),
    base_label=vecd["base"]["label"], target_label=vecd["target"]["label"],
    classes={k: [[[round(x, 2), round(y, 2)] for x, y in ring] for ring in v]
             for k, v in vecd["classes"].items()},
    routes=[[[round(float(x), 2), round(float(y), 2)] for x, y in r[::2]] for r in menu],
    hazards=[[round(float(coords[h][0]), 2), round(float(coords[h][1]), 2),
              round(float(rr[h]), 2), round(float(pp[h]), 2)] for h in range(len(coords))],
    eq=round(float(sol.loss_mixed), 4), det=round(float(sol.loss_det), 4),
    strategies={k: [round(float(x), 4) for x in d] for k, d in strat.items()},
    worst={k: round(v, 4) for k, v in worst.items()}, br_pos=brpos, N=N,
)
json.dump(out, open("models/runs/theatre_vec_view.json", "w"))
print("wrote theatre_vec_view.json | eq", out["eq"], "det", out["det"], "worst", out["worst"],
      "| routes", len(out["routes"]), "hazards", len(out["hazards"]))
