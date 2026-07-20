#!/usr/bin/env python3
"""gen32 Phase 0: the corridor hunt on the REAL Kaliningrad->Gvardeysk vec-theatre
(ORACLE-ONLY, free; the gen31 anticipatory-doctrine register transplanted onto real OSM
terrain). Pre-registered gates G1-G5 in experiments/gen32_theatre_dyn.md.

Substrate: the committed vector theatre (data/maps/theatre_kgd_gvardeysk_vec.json; 25 routes =
14 geometric lanes + 11 terrain-aware cover routes; 185 candidate AD sites on emplaceable
terrain outside terminal standoff; LOS-masked survival). The FIXED menu + sites are shared
across layouts; only the hidden per-site EFFECTIVENESS field is resampled (a spatially-
correlated RBF over the real site coordinates, rank-mapped into a band): 'which real positions
are hot today'. Enemy = the gen31 anticipatory mixed doctrine (q_rep punish the recent window
+ q_flee pre-aim at the obvious escape route), softmax(tau). Everything exact at w=2.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_theatre_vec import (build_theatre_game, load_vec_theatre,
                                         route_survival)

N = 3


def _mm(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


def resample_field(coords, seed, length_scale=6.0, band=(0.30, 0.95)):
    """Spatially-correlated hidden effectiveness field over the REAL site coordinates
    (RBF Gaussian draw, rank-mapped into the band; km length scale). Terrain still decides
    WHICH positions can emplace and their radii; this decides which are HOT today."""
    rng = np.random.default_rng(seed)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2.0 * length_scale ** 2)) + 1e-8 * np.eye(len(coords))
    g = rng.multivariate_normal(np.zeros(len(coords)), cov)
    ranks = np.argsort(np.argsort(g))
    lo, hi = band
    return lo + (hi - lo) * ranks / (len(coords) - 1)


class TheatreBase:
    """Loads the fixed theatre ONCE (menu, sites, radii, route topology). Field-specific games
    are cheap rebuilds (recompute survival + payoff for a resampled effectiveness field)."""

    def __init__(self, path="data/maps/theatre_kgd_gvardeysk_vec.json"):
        self.th = load_vec_theatre(path)
        game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(
            self.th, K=1, n_lanes=14, n_terrain=12, standoff_km=4.0)
        self.menu, self.coords, self.rr, self.lane_idx = menu, coords, rr, lane_idx
        self.routes, self.route_edges = game.routes, game.route_edges
        self.isets, self.travel = game.interdiction_sets, game.travel_cost
        self.R, self.H = game.n_routes, len(coords)

    def game_for(self, pp_field):
        S = np.stack([route_survival(self.th, self.menu[i], self.coords, self.rr, pp_field,
                                     los=True) for i in range(self.R)])
        logS = np.log(np.clip(S, 1e-300, 1.0))
        idx = np.asarray(self.isets, dtype=int)
        payoff = 1.0 - np.exp(logS[:, idx].sum(axis=2))
        game = InterdictionGame(self.routes, self.route_edges, self.isets, payoff,
                                self.travel, 1)
        return game, S


class DynTheatre:
    """One (field, doctrine, operating point) cell with exact machinery (gen31 math)."""

    def __init__(self, base: TheatreBase, pp_field, w, tau, q_rep, q_flee, q_ar=0.0):
        game, S = base.game_for(pp_field)
        self.R, self.w = base.R, w
        self.dmg = 1.0 - S ** N                                   # [R, H]
        sol = solve_multiconvoy(game, N, "mission")
        occs = list(itertools.combinations_with_replacement(range(self.R), N))
        d_eq = np.zeros(self.R)
        for i, o in enumerate(occs):
            if len(set(o)) == 1:
                d_eq[o[0]] += sol.defender_strategy[i]
        self.d_eq = d_eq / d_eq.sum() if d_eq.sum() > 0 else np.full(self.R, 1 / self.R)
        self.eq_static = float(sol.loss_mixed)
        self.lane_idx = base.lane_idx
        self.S = S
        R = self.R
        self.states = np.array(list(itertools.product(range(R), repeat=w)))
        Sn = len(self.states)
        Vw = self.dmg[self.states].mean(axis=1)                  # [Sn, H] damage vs recent play
        mask = np.zeros((Sn, R), bool)
        for k in range(w):
            mask[np.arange(Sn), self.states[:, k]] = True
        Zr = (Vw - Vw.max(axis=1, keepdims=True)) / tau
        Ar = np.exp(Zr); Ar /= Ar.sum(axis=1, keepdims=True)
        rflee = (Ar @ self.dmg.T).argmin(axis=1)                 # obvious myopic escape route
        # anti-repeat ANTICIPATION (the G2 lever): the enemy models a defender who avoids its
        # recent window (uniform over NON-window routes) and pre-aims at that spread, so any
        # blind rotation/anti-repeat rule is punished and only calibrated randomised play evades.
        ar = (~mask).astype(float)
        ar /= ar.sum(axis=1, keepdims=True)
        Var = ar @ self.dmg                                      # [Sn, H] damage vs a spreader
        Z = q_rep * Vw + q_flee * self.dmg[rflee] + q_ar * Var
        Zs = (Z - Z.max(axis=1, keepdims=True)) / tau
        A = np.exp(Zs); A /= A.sum(axis=1, keepdims=True)
        self.stepdmg = A @ self.dmg.T                            # [Sn, R]
        pows = R ** np.arange(w - 1, -1, -1)
        shifted = np.concatenate([self.states[:, 1:], np.zeros((Sn, 1), int)], axis=1)
        self.succ = (shifted @ pows)[:, None] + np.arange(R)[None, :]
        self.in_window = mask

    def history_opt(self, iters=4000, tol=1e-12):
        V = np.zeros(len(self.states))
        for _ in range(iters):
            Q = self.stepdmg + V[self.succ]
            Vn = 0.5 * Q.min(axis=1) + 0.5 * V
            Vd = Vn - Vn.mean()
            if np.abs(Vd - V).max() < tol:
                break
            V = Vd
        Q = self.stepdmg + V[self.succ]
        return float((Q.min(axis=1) - V).mean())

    def stationary(self, rule_mat, iters=600, damp=0.5, tol=1e-13):
        Sn = len(self.states)
        pi = np.full(Sn, 1.0 / Sn)
        for _ in range(iters):
            flow = pi[:, None] * rule_mat
            nxt = np.zeros(Sn)
            np.add.at(nxt, self.succ.ravel(), flow.ravel())
            nxt = damp * nxt + (1 - damp) * pi
            if np.abs(nxt - pi).max() < tol:
                pi = nxt
                break
            pi = nxt
        return float((pi[:, None] * rule_mat * self.stepdmg).sum())

    def value_static(self, d):
        return self.stationary(np.broadcast_to(d, (len(self.states), self.R)).copy())

    def supports(self):
        exp = 1.0 - self.S.min(axis=1)                           # per-route worst exposure
        out = {}
        u_l = np.zeros(self.R); u_l[self.lane_idx] = 1.0 / len(self.lane_idx)
        out["uniform_lanes"] = u_l
        iv = np.zeros(self.R)
        iv[self.lane_idx] = 1.0 / np.clip(exp[self.lane_idx], 1e-6, None)
        out["invrisk_lanes"] = iv / iv.sum()
        out["uniform_full"] = np.full(self.R, 1.0 / self.R)
        ivf = 1.0 / np.clip(exp, 1e-6, None)
        out["invrisk_full"] = ivf / ivf.sum()
        out["eq_support"] = self.d_eq
        return out


def rule_family(g: DynTheatre):
    rows = {"iid_eq": g.value_static(g.d_eq)}
    # local static optimum (CEM on the eq+low-exposure support)
    rng = np.random.default_rng(0)
    exp = 1.0 - g.S.min(axis=1)
    pool = sorted(set(np.where(g.d_eq > 1e-6)[0]) | set(np.argsort(exp)[:12]))
    mu = np.full(len(pool), 1.0 / len(pool)); best = np.inf
    for _ in range(25):
        smp = rng.dirichlet(mu * 24 + 0.3, size=24)
        vals = []
        for s in smp:
            d = np.zeros(g.R); d[pool] = s
            vals.append(g.value_static(d))
        vals = np.array(vals)
        mu = 0.6 * mu + 0.4 * smp[np.argsort(vals)[:6]].mean(axis=0)
        best = min(best, float(vals.min()))
    rows["static_localopt*fit"] = best
    # payoff-blind dynamic family: anti-repeat + rotation over each support
    for name, dd in g.supports().items():
        m = np.broadcast_to(dd, (len(g.states), g.R)).copy()
        m[g.in_window] = 0.0
        s = m.sum(axis=1, keepdims=True)
        m = np.where(s > 1e-12, m / np.where(s > 1e-12, s, 1.0), np.broadcast_to(dd, m.shape))
        rows[f"anti_{name}"] = g.stationary(m)
        sup = np.where(dd > 1e-9)[0]
        if len(sup) > g.w:
            rot = np.zeros((len(g.states), g.R))
            for si in range(len(g.states)):
                cand = [r for r in sup if not g.in_window[si, r]]
                rot[si, cand[0] if cand else sup[0]] = 1.0
            rows[f"rot_{name}"] = g.stationary(rot)
    # doctrine-informed fitted rules (information parity; disclosed caps)
    dodge = np.zeros((len(g.states), g.R))
    dodge[np.arange(len(g.states)), g.stepdmg.argmin(axis=1)] = 1.0
    rows["myopic_dodge"] = g.stationary(dodge)
    logeq = np.log(np.clip(g.d_eq, 1e-12, 1.0))[None, :]
    for tag, base in (("softdodge", 0.0), ("composed", 1.0)):
        bv = np.inf
        for beta in (0.5, 1, 2, 4, 8, 16, 32):
            L = base * logeq - beta * g.stepdmg
            L = L - L.max(axis=1, keepdims=True)
            m = np.exp(L); m /= m.sum(axis=1, keepdims=True)
            bv = min(bv, g.stationary(m))
        rows[f"{tag}*fit"] = bv
    return rows


def cell(base, seed, w, tau, q_rep, q_flee, q_ar=0.0):
    t0 = time.time()
    field = resample_field(base.coords, seed)
    g = DynTheatre(base, field, w, tau, q_rep, q_flee, q_ar)
    rows = rule_family(g)
    hopt = g.history_opt()
    blind = [k for k in rows if k.startswith(("anti_", "rot_"))]
    bb = min(rows[k] for k in blind); bbn = min(blind, key=lambda k: rows[k])
    fit = min(rows[k] for k in ("myopic_dodge", "softdodge*fit", "composed*fit"))
    cap = min(rows["iid_eq"], rows["static_localopt*fit"])
    g1, g2, g3 = cap / max(hopt, 1e-9), bb / max(hopt, 1e-9), fit / max(hopt, 1e-9)
    print(f"seed{seed} w{w} tau{tau} q({q_rep},{q_flee},{q_ar}): cap={cap:.3f} blind={bb:.3f}"
          f"({bbn[:16]}) dodge={rows['myopic_dodge']:.3f} fit={fit:.3f} hopt={hopt:.3f} | "
          f"G1={g1:.2f} G2={g2:.2f} G3={g3:.2f} eq={g.eq_static:.3f} [{time.time()-t0:.0f}s]",
          flush=True)
    return {"seed": seed, "w": w, "tau": tau, "q_rep": q_rep, "q_flee": q_flee, "q_ar": q_ar,
            "cap": cap, "best_blind": bb, "myopic_dodge": rows["myopic_dodge"], "fit": fit,
            "hist_opt": hopt, "G1": g1, "G2": g2, "G3": g3, "eq_static": g.eq_static}


def main():
    base = TheatreBase()
    print(f"[gen32] {base.th.name}: R={base.R} lanes={len(base.lane_idx)} H={base.H}", flush=True)
    out = []
    for q_rep, q_flee in ((1.0, 0.0), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.8, 0.2)):
        for tau in (0.10, 0.15):
            for seed in (5100, 5101, 5102):
                out.append(cell(base, seed, 2, tau, q_rep, q_flee))
        json.dump(out, open("models/runs/gen32_theatre_hunt.json", "w"), indent=1)
    print("[written] models/runs/gen32_theatre_hunt.json", flush=True)


if __name__ == "__main__":
    main()
