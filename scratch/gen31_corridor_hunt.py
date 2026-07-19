#!/usr/bin/env python3
"""gen31 Phase 0: the corridor hunt (ORACLE-ONLY, free; pre-registered gates G1-G5 in
experiments/gen31_aerial_dyn.md).

Enemy = ANTICIPATORY MIXED DOCTRINE: aim distribution a(window) = softmax_tau over per-position
scores  Z[h] = q_rep * E[dmg | window routes] + q_dodge * E[dmg | uniform over NON-window
routes] + q_eq * E[dmg | the static equilibrium stack mixture].  (Design decision recorded:
the q_eq component scores positions against the standard hedge rather than mixing in the LP
dual attacker; one softmax keeps the doctrine a single legible aim rule.)

Everything is EXACT at w=2 (and vectorised for w=3): the window MDP (S = R^w states) gives
history_opt by RVI; every rule's stationary damage comes from damped power iteration on the
window chain; static rules use the same table. Information parity: doctrine-informed rules
(myopic dodge, fitted soft dodge, fitted hedge-composed) are in the family, disclosed as
oracle-fitted where fitted.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from scripts.train_aerial_generalist import random_field
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.aerial_curves import (all_lane_sets, build_curve_menu, build_curved_game,
                                    dense_hazard_grid)
from src.envs.aerial_sector import SectorLattice

DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(4, j) for j in range(9) if j < 5} | {(8, j) for j in range(9) if j > 3}))
BASE = SectorLattice(ny=9, nx=13)
N = 3


class DynGame:
    """One (layout, doctrine, operating point) cell with exact machinery."""

    def __init__(self, lat, seed, w, tau, q_rep, q_dodge, q_eq, q_flee=0.0, r=1.2):
        menu, _ = build_curve_menu(lat, r, R=40, seed=0)
        centres = dense_hazard_grid(lat, step=0.5)
        game, S = build_curved_game(lat, menu, centres, 1, r=r,
                                    p_max=random_field(centres, seed))
        self.R = game.n_routes
        self.w = w
        self.dmg = 1.0 - S ** N                              # [R, H]
        sol = solve_multiconvoy(game, N, "mission")
        occs = list(itertools.combinations_with_replacement(range(self.R), N))
        d_eq = np.zeros(self.R)
        for i, o in enumerate(occs):
            if len(set(o)) == 1:
                d_eq[o[0]] += sol.defender_strategy[i]
        self.d_eq = d_eq / d_eq.sum() if d_eq.sum() > 0 else np.full(self.R, 1 / self.R)
        self.eq_static = float(sol.loss_mixed)
        self.game, self.S_surv, self.lat, self.menu = game, S, lat, menu
        R = self.R
        self.states = np.array(list(itertools.product(range(R), repeat=w)))   # [S, w]
        Sn = len(self.states)
        Vw = self.dmg[self.states].mean(axis=1)              # [S, H] damage vs window routes
        mask = np.zeros((Sn, R), bool)
        for k in range(w):
            mask[np.arange(Sn), self.states[:, k]] = True
        n_in = mask.sum(1)
        Vc = (self.dmg.sum(0)[None, :] - (mask @ self.dmg)) / (R - n_in)[:, None]
        Veq = (self.d_eq @ self.dmg)[None, :]                # [1, H]
        Z = q_rep * Vw + q_dodge * Vc + q_eq * Veq
        if q_flee > 0:
            # second-order 'flee' model: the enemy pre-aims at the route a pattern-punished
            # defender would obviously run to (argmin damage vs the pure-repeat aim)
            Zr = (Vw - Vw.max(axis=1, keepdims=True)) / tau
            Ar = np.exp(Zr); Ar /= Ar.sum(axis=1, keepdims=True)
            rflee = (Ar @ self.dmg.T).argmin(axis=1)
            Z = Z + q_flee * self.dmg[rflee]
        Zs = (Z - Z.max(axis=1, keepdims=True)) / tau
        A = np.exp(Zs)
        A /= A.sum(axis=1, keepdims=True)                    # aim distribution per state
        self.stepdmg = A @ self.dmg.T                        # [S, R]
        # successor index: state (r1..rw) --play r--> (r2..rw, r)
        pows = R ** np.arange(w - 1, -1, -1)
        self.sidx = self.states @ pows
        shifted = np.concatenate([self.states[:, 1:], np.zeros((Sn, 1), int)], axis=1)
        self.succ = (shifted @ pows)[:, None] + np.arange(R)[None, :]   # [S, R]
        self.in_window_mask = mask

    def history_opt(self, iters=2000, tol=1e-12):
        """RVI with the aperiodicity (lazy-chain) transform: T -> (T + I)/2, which preserves
        the average reward and kills periodic oscillation (the dbf385d lesson, reapplied)."""
        V = np.zeros(len(self.states))
        g = 0.0
        for _ in range(iters):
            Q = self.stepdmg + V[self.succ]
            Vn = 0.5 * Q.min(axis=1) + 0.5 * V
            g = 2.0 * float(Vn.mean() - V.mean())            # undo the transform's halving
            Vd = Vn - Vn.mean()
            if np.abs(Vd - V).max() < tol:
                return g
            V = Vd
        return g

    def stationary(self, rule_mat, iters=400, damp=0.5, tol=1e-12):
        """Exact stationary damage of a stationary rule given as D[s, r] (row-stochastic),
        by damped power iteration on the window chain."""
        Sn = len(self.states)
        pi = np.full(Sn, 1.0 / Sn)
        for _ in range(iters):
            flow = pi[:, None] * rule_mat                    # [S, R]
            nxt = np.zeros(Sn)
            np.add.at(nxt, self.succ.ravel(), flow.ravel())
            nxt = damp * nxt + (1 - damp) * pi
            if np.abs(nxt - pi).max() < tol:
                pi = nxt
                break
            pi = nxt
        return float((pi[:, None] * rule_mat * self.stepdmg).sum())

    def static_rule(self, d):
        return np.broadcast_to(np.asarray(d), (len(self.states), self.R)).copy()

    def value_static(self, d):
        return self.stationary(self.static_rule(d))


def rule_family(gme: DynGame):
    """The complete parity family. Returns {name: value}; fitted rules marked *fit."""
    rows = {}
    rows["iid_eq"] = gme.value_static(gme.d_eq)
    # multi-start local static optimum (CEM on the simplex; disclosed local)
    rng = np.random.default_rng(0)
    top = np.argsort(gme.dmg.max(axis=1))[:12]
    pool = sorted(set(np.where(gme.d_eq > 1e-6)[0]) | set(top))
    mu = np.full(len(pool), 1.0 / len(pool))
    best = np.inf
    for it in range(25):
        samples = rng.dirichlet(mu * 24 + 0.3, size=24)
        vals = []
        for smp in samples:
            d = np.zeros(gme.R)
            d[pool] = smp
            vals.append(gme.value_static(d))
        vals = np.array(vals)
        elite = samples[np.argsort(vals)[:6]]
        mu = 0.6 * mu + 0.4 * elite.mean(axis=0)
        best = min(best, float(vals.min()))
    rows["static_localopt*fit"] = best
    # payoff-blind dynamic rules: anti-repeat + rotation over every support family
    fams = {}
    lsets = all_lane_sets(gme.lat, gme.menu)
    for rc, li in (lsets.items() if lsets else [(0.0, [])]):
        for k, dd in lane_stack_distributions(gme.game, li, gme.S_surv).items():
            fams[f"{k}@{rc}"] = dd
    fams["eq_support"] = gme.d_eq
    blind = {}
    for name, dd in fams.items():
        m = np.broadcast_to(dd, (len(gme.states), gme.R)).copy()
        m[gme.in_window_mask] = 0.0
        s = m.sum(axis=1, keepdims=True)
        fallback = np.broadcast_to(dd, m.shape)
        m = np.where(s > 1e-12, m / np.where(s > 1e-12, s, 1.0), fallback)
        blind[f"anti_{name}"] = gme.stationary(m)
        sup = np.where(dd > 1e-9)[0]
        if len(sup) > gme.w:
            rot = np.zeros_like(m)
            for si in range(len(gme.states)):
                cand = [rr for rr in sup if not gme.in_window_mask[si, rr]]
                rot[si, cand[0] if cand else sup[0]] = 1.0
            blind[f"rot_{name}"] = gme.stationary(rot)
    rows.update(blind)
    # doctrine-informed rules (parity): myopic dodge; fitted soft dodge; fitted composed
    dodge = np.zeros((len(gme.states), gme.R))
    dodge[np.arange(len(gme.states)), gme.stepdmg.argmin(axis=1)] = 1.0
    rows["myopic_dodge"] = gme.stationary(dodge)
    logeq = np.log(np.clip(gme.d_eq, 1e-12, 1.0))[None, :]
    for tag, base in (("softdodge", 0.0), ("composed", 1.0)):
        bv = np.inf
        for beta in (0.5, 1, 2, 4, 8, 16, 32):
            L = base * logeq - beta * gme.stepdmg
            L = L - L.max(axis=1, keepdims=True)
            m = np.exp(L)
            m /= m.sum(axis=1, keepdims=True)
            bv = min(bv, gme.stationary(m))
        rows[f"{tag}*fit"] = bv
    return rows


def cell(tag, lat, seed, w, tau, q_rep, q_dodge, q_eq, q_flee=0.0):
    t0 = time.time()
    gme = DynGame(lat, seed, w, tau, q_rep, q_dodge, q_eq, q_flee)
    rows = rule_family(gme)
    hopt = gme.history_opt()
    blind_keys = [k for k in rows if k.startswith(("anti_", "rot_"))]
    best_blind = min(rows[k] for k in blind_keys)
    best_blind_name = min(blind_keys, key=lambda k: rows[k])
    fit_keys = ["myopic_dodge", "softdodge*fit", "composed*fit"]
    best_fit = min(rows[k] for k in fit_keys)
    cap = min(rows["iid_eq"], rows["static_localopt*fit"])
    g1 = cap / max(hopt, 1e-9)
    g2 = best_blind / max(hopt, 1e-9)
    g3 = best_fit / max(hopt, 1e-9)
    out = {"tag": tag, "seed": seed, "w": w, "tau": tau, "q": [q_rep, q_dodge, q_eq, q_flee],
           "eq_static": gme.eq_static, "iid_eq": rows["iid_eq"],
           "static_localopt": rows["static_localopt*fit"], "best_blind": best_blind,
           "best_blind_name": best_blind_name, "myopic_dodge": rows["myopic_dodge"],
           "softdodge_fit": rows["softdodge*fit"], "composed_fit": rows["composed*fit"],
           "history_opt": hopt, "G1_static_corridor": g1, "G2_blind_corridor": g2,
           "G3_fit_corridor": g3}
    print(f"{tag} s{seed} w={w} tau={tau} q=({q_rep},{q_dodge},{q_eq},{q_flee}): "
          f"cap={cap:.3f} blind={best_blind:.3f}({best_blind_name[:18]}) "
          f"dodge={rows['myopic_dodge']:.3f} fit={best_fit:.3f} hopt={hopt:.3f} | "
          f"G1={g1:.2f} G2={g2:.2f} G3={g3:.2f} [{time.time()-t0:.0f}s]", flush=True)
    return out


def main():
    out = []
    # slice 1: the doctrine grid at w=2 on three structured layouts (v4.0's probe layouts)
    for q_rep, q_dodge, q_eq, q_flee in (
            (1.0, 0.0, 0.0, 0.0),                       # v4.0 baseline doctrine
            (0.7, 0.3, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0),
            (0.6, 0.0, 0.0, 0.4), (0.4, 0.0, 0.0, 0.6),
            (0.5, 0.2, 0.0, 0.3), (0.4, 0.2, 0.2, 0.2), (0.3, 0.3, 0.0, 0.4)):
        for tau in (0.10, 0.15):
            for s in (2100, 2101, 2102):
                out.append(cell("dbl", DBL, s, 2, tau, q_rep, q_dodge, q_eq, q_flee))
        json.dump(out, open("models/runs/gen31_corridor_hunt.json", "w"), indent=1)
    print("[written] models/runs/gen31_corridor_hunt.json")


if __name__ == "__main__":
    main()
