"""Export gen28 v5 theatre data for the interactive review render: terrain grid, hazard field,
base/target/standoff, and four strategies (direct field crossing / safest corridor / inverse-
risk full menu = strongest naive / fleet-stack equilibrium) with their oracle mission-failure
and best-response hazard. N=3 fleet, K=1, mission. Non-degenerate theatres only."""
import json, itertools
import numpy as np
from src.baselines.interdiction_oracle import best_response_attacker, _row_minimiser
from src.baselines.multiconvoy_oracle import objective_matrix
from src.envs.aerial_terrain import generate_theatre, build_theatre_game

N = 3

def export(seed, ny=17, nx=40, step=0.75, R=36):
    th = generate_theatre(ny=ny, nx=nx, seed=seed)
    game, S, menu, centres, pmax, rad, exp = build_theatre_game(th, K=1, step=step, R=R, seed=0)
    Rn = game.n_routes
    occs = list(itertools.combinations_with_replacement(range(Rn), N))
    vecs = np.zeros((len(occs), Rn))
    for i, c in enumerate(occs):
        for r in c: vecs[i, r] += 1
    oidx = {tuple(int(x) for x in v): i for i, v in enumerate(vecs)}
    _, M = objective_matrix(game, N, "mission", 1)
    def stack(d):
        out = np.zeros(len(vecs))
        for r, p in enumerate(d):
            if p > 1e-12:
                v = np.zeros(Rn); v[r] = N; out[oidx[tuple(int(x) for x in v)]] += p
        return out
    exposure = 1.0 - S.min(axis=1)
    stack_rows = np.stack([M[oidx[tuple(int(x) for x in (np.eye(Rn)[r]*N))]] for r in range(Rn)])
    eqv, d_eq = _row_minimiser(stack_rows)
    invx = 1.0/np.clip(exposure, 1e-9, None); invx /= invx.sum()
    direct = np.zeros(Rn); direct[int(np.argmax(exposure))] = 1.0     # most-exposed = direct field
    corridor = np.zeros(Rn); corridor[int(np.argmin(exposure))] = 1.0 # safest single
    strat = {"direct": direct, "corridor": corridor, "invrisk": invx, "equilibrium": d_eq}
    out = dict(seed=seed, ny=ny, nx=nx, grid=th.grid.tolist(),
               base=list(th.lattice().base), target=list(th.lattice().target), safe_r=3.0,
               centres=[[round(float(x),2), round(float(y),2)] for x,y in centres],
               pmax=[round(float(x),3) for x in pmax], rad=[round(float(x),2) for x in rad],
               routes=[[[round(float(x),2), round(float(y),2)] for x,y in c.pts[::3]] for c in menu],
               eq=round(float(eqv),3), strat={}, worst={}, br={})
    for k, d in strat.items():
        j, v = best_response_attacker(game, d)
        out["strat"][k] = [round(float(x),4) for x in d]
        out["worst"][k] = round(float((stack(d) @ M).max()), 3)
        out["br"][k] = int(game.interdiction_sets[j][0])
    return out

seeds = [2, 6, 10, 4]
data = [export(s) for s in seeds]
json.dump(data, open("models/runs/gen28_terrain_view.json", "w"))
print("wrote", [f"seed{d['seed']}: eq={d['eq']} worst={d['worst']}" for d in data])
