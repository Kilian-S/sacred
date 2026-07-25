"""gen33_llm_adversary NON-DEGENERACY SCREEN (oracle-only, no training).

For each core theatre: build the static mission game at the coverage-fraction-scaled weapon range
and report the gates that decide whether it is a usable contest (eq value in band, leader entropy
materially < 1, a naive/eq gap), plus the compute envelope (site count, K=1 done, K=3 matrix size)
that sizes the phase-2 budget. Anchors the coverage fraction phi per map. Numbers go to the ledger.
"""
import itertools
import time
from math import comb

import numpy as np

from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_theatre_vec import build_theatre_game, lateral_width, load_vec_theatre

N = 3
THEATRES = {
    "kgd": "data/maps/theatre_kgd_gvardeysk_vec.json",
    "ukraine": "data/maps/theatre_ukraine_vec.json",
    "narva": "data/maps/theatre_narva_vec.json",
}
LAT_REF = lateral_width(load_vec_theatre(THEATRES["kgd"]))          # kgd = the range reference


def stack_od(d_routes, occ_index, R):
    """Occupancy distribution for the fleet stacking all N on route r w.p. d_routes[r].
    Occupancies are COUNT vectors of length R (N on the stacked route, 0 elsewhere)."""
    od = np.zeros(len(occ_index))
    for r in range(R):
        key = tuple(N if i == r else 0 for i in range(R))
        od[occ_index[key]] += d_routes[r]
    return od


def worst(od, obj):
    return float((od @ obj).max())                                  # attacker best response


def screen(name, path):
    th = load_vec_theatre(path)
    lw = lateral_width(th)
    scale = lw / LAT_REF
    t0 = time.time()
    game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(
        th, K=1, n_lanes=14, n_terrain=12,
        spacing_km=2.0 * scale, standoff_km=4.0 * scale, range_scale=scale)
    R, Hn = game.n_routes, len(coords)
    occs, obj = objective_matrix(game, N, "mission", 1)
    occ_index = {tuple(int(x) for x in o): i for i, o in enumerate(occs)}
    sol = solve_multiconvoy(game, N, "mission")
    d_eq = np.zeros(R)
    for i, o in enumerate(occs):
        if len(set(o)) == 1:
            d_eq[o[0]] += sol.defender_strategy[i]
    d_eq = d_eq / d_eq.sum() if d_eq.sum() > 0 else np.full(R, 1.0 / R)
    eq = float(sol.loss_mixed)
    p = d_eq[d_eq > 1e-9]
    ent = float(-(p * np.log(p)).sum() / np.log(R))                 # leader entropy H/lnR

    # naive defender family (each scored by the mission BR): uniform lanes, inv-risk lanes, best-k
    d_uni = np.zeros(R); d_uni[lane_idx] = 1.0 / len(lane_idx)
    exp = 1.0 - S.min(axis=1)
    d_inv = np.zeros(R); d_inv[lane_idx] = 1.0 / np.clip(exp[lane_idx], 1e-6, None)
    d_inv /= d_inv.sum()
    naive = [worst(stack_od(d_uni, occ_index, R), obj),
             worst(stack_od(d_inv, occ_index, R), obj)]
    bestk = np.inf
    for k in (2, 3, 4):
        for T in itertools.combinations(range(R), k):
            dk = np.zeros(R); dk[list(T)] = 1.0 / k
            bestk = min(bestk, worst(stack_od(dk, occ_index, R), obj))
    naive.append(bestk)
    best_naive = min(naive)

    phi = 2.0 * float(np.mean(rr)) / lw                             # coverage fraction
    k3 = R * comb(Hn, 3)                                            # phase-2 exact matrix size
    ok_eq = 0.02 < eq < 0.90
    ok_ent = ent < 0.95
    ok_gap = best_naive / eq > 1.10
    verdict = "USABLE" if (ok_eq and ok_ent and ok_gap) else "CHECK"
    print(f"{name:9s} scale={scale:4.2f} R={R:2d} sites={Hn:3d} phi={phi:4.2f} | "
          f"eq={eq:.3f} ent={ent:.2f} best_naive={best_naive:.3f} ({best_naive/eq:4.2f}x) | "
          f"K3_matrix={k3/1e6:5.0f}M {'ok' if k3 < 60e6 else 'BIG'} | "
          f"{verdict} [{time.time() - t0:.0f}s]", flush=True)
    return dict(name=name, scale=scale, R=R, sites=Hn, phi=phi, eq=eq, ent=ent,
                best_naive=best_naive, gap=best_naive / eq, k3_matrix=k3, verdict=verdict)


def _load_g32():
    """Load gen32's dynamic-doctrine machinery by path (avoids scratch-package issues)."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("g32", "scratch/gen32_theatre_hunt.py")
    g32 = importlib.util.module_from_spec(spec)
    sys.modules["g32"] = g32
    spec.loader.exec_module(g32)
    return g32


def dynamic_gate(name, path, g32, seeds=(5100, 5101, 5102)):
    """The trainability gate: build the DYNAMIC doctrine game (random field + the gen32 operating
    point q=(0.7,0.3) tau=0.10 w=2) at the scaled range, and report G1 (static capped), G2 (blind
    rules beatable), leader entropy. A degraded field breaks the static symmetry (ent<1)."""
    th = load_vec_theatre(path)
    scale = lateral_width(th) / LAT_REF
    base = g32.TheatreBase(path, range_scale=scale, spacing_km=2.0 * scale,
                           standoff_km=4.0 * scale)
    import numpy as _np
    g1s, g2s, ents, eqs = [], [], [], []
    for seed in seeds:
        field = g32.resample_field(base.coords, seed)
        g = g32.DynTheatre(base, field, 2, 0.10, 0.7, 0.3)
        rows = g32.rule_family(g)
        hopt = g.history_opt()
        blind = [k for k in rows if k.startswith(("anti_", "rot_"))]
        bb = min(rows[k] for k in blind)
        cap = min(rows["iid_eq"], rows["static_localopt*fit"])
        p = g.d_eq[g.d_eq > 1e-9]
        ent = float(-(p * _np.log(p)).sum() / _np.log(g.R))
        g1s.append(cap / max(hopt, 1e-9)); g2s.append(bb / max(hopt, 1e-9))
        ents.append(ent); eqs.append(g.eq_static)
    g1, g2 = float(_np.mean(g1s)), float(_np.mean(g2s))
    ent, eq = float(_np.mean(ents)), float(_np.mean(eqs))
    ok = g1 >= 1.4 and g2 >= 1.25 and ent < 0.95 and 0.02 < eq < 0.9
    print(f"{name:9s} scale={scale:4.2f} | eq_static={eq:.3f} leader_ent={ent:.2f} "
          f"G1(cap/opt)={g1:4.2f} G2(blind/opt)={g2:4.2f} | "
          f"{'DOCTRINE CONTEST' if ok else 'CHECK'}", flush=True)
    return dict(name=name, scale=scale, eq=eq, ent=ent, G1=g1, G2=g2, usable=ok)


if __name__ == "__main__":
    print(f"=== STATIC feasibility (loader/scaling/K3 envelope); range ref lat_w={LAT_REF:.0f}km "
          f"(kgd) ===\n")
    static_rows = [screen(n, p) for n, p in THEATRES.items()]
    print(f"\n=== DYNAMIC doctrine gate (the trainability screen; random field + q(0.7,0.3) "
          f"tau0.10 w2; gates G1>=1.4 G2>=1.25 ent<0.95) ===\n")
    g32 = _load_g32()
    dyn_rows = [dynamic_gate(n, p, g32) for n, p in THEATRES.items()]
