"""Scorer for red forces under the pinned enemy semantics.

A force of K agents induces one dynamic enemy whose aim is the mean over agents of prior-shaped
softmax aims: each agent scores sites by q_rep * Vw + q_flee * dmg[rflee] + q_eq * Vstat, shaped by
an RBF prior centred on its resolved site. The value of a force is the best-responding
history-aware defender's stationary damage. Baseline force constructors (random floor, doctrine
heuristic, oracle search) live here so every ladder is built against the same machinery.
"""
from __future__ import annotations

import itertools

import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.aerial_theatre_vec import (build_theatre_game, lateral_width, load_vec_theatre,
                                         route_survival)

N = 3
GEN32_DOCTRINE = (0.7, 0.3, 0.0, 0.10, 2)      # (q_rep, q_flee, q_eq, tau, w)


def resample_field(coords, seed, length_scale=6.0, band=(0.30, 0.95)):
    """Draw the hidden effectiveness field: an RBF sample rank-mapped into the band."""
    rng = np.random.default_rng(seed)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2.0 * length_scale ** 2)) + 1e-8 * np.eye(len(coords))
    g = rng.multivariate_normal(np.zeros(len(coords)), cov)
    ranks = np.argsort(np.argsort(g))
    lo, hi = band
    return lo + (hi - lo) * ranks / (len(coords) - 1)


class ScoreBase:
    """Fixed theatre substrate (menu, sites, radii, topology) + per-field caches."""

    def __init__(self, path, lat_ref=None):
        self.th = load_vec_theatre(path)
        lw = lateral_width(self.th)
        self.scale = (lw / lat_ref) if lat_ref else 1.0
        s = self.scale
        game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(
            self.th, K=1, n_lanes=14, n_terrain=12, spacing_km=2.0 * s, standoff_km=4.0 * s,
            range_scale=s)
        self.menu, self.coords, self.rr, self.lane_idx = menu, coords, rr, lane_idx
        self.routes, self.route_edges = game.routes, game.route_edges
        self.isets, self.travel = game.interdiction_sets, game.travel_cost
        self.R, self.H = game.n_routes, len(coords)
        self.site_exposure = (1.0 - S).mean(axis=0)   # terrain-implied site value (field-blind)
        self.cls = [self.th.classify(c) for c in coords]
        self._fields: dict = {}

    def game_for(self, pp_field):
        S = np.stack([route_survival(self.th, self.menu[i], self.coords, self.rr, pp_field,
                                     los=True) for i in range(self.R)])
        logS = np.log(np.clip(S, 1e-300, 1.0))
        idx = np.asarray(self.isets, dtype=int)
        payoff = 1.0 - np.exp(logS[:, idx].sum(axis=2))
        game = InterdictionGame(self.routes, self.route_edges, self.isets, payoff,
                                self.travel, 1)
        return game, S

    def field(self, seed):
        """Per-field core: dmg, static equilibrium mix + value, hold-static site values."""
        if seed in self._fields:
            return self._fields[seed]
        pp = resample_field(self.coords, seed)
        game, S = self.game_for(pp)
        dmg = 1.0 - S ** N
        sol = solve_multiconvoy(game, N, "mission")
        occs = list(itertools.combinations_with_replacement(range(self.R), N))
        d_eq = np.zeros(self.R)
        for i, o in enumerate(occs):
            if len(set(o)) == 1:
                d_eq[o[0]] += sol.defender_strategy[i]
        d_eq = d_eq / d_eq.sum() if d_eq.sum() > 0 else np.full(self.R, 1.0 / self.R)
        fc = FieldCore(self, seed, dmg, d_eq, float(sol.loss_mixed))
        fc.game, fc.S = game, S                      # kept for env builds
        self._fields[seed] = fc
        return fc


class FieldCore:
    """One (theatre, hidden field): dmg + equilibrium + lazily-built window contexts per w_max."""

    def __init__(self, base: ScoreBase, seed, dmg, d_eq, eq_static):
        self.base, self.seed = base, seed
        self.dmg, self.d_eq, self.eq_static = dmg, d_eq, eq_static
        self.Vstat = d_eq @ dmg                        # [H] site value vs the equilibrium mix
        self.R, self.H = base.R, base.H
        self._wins: dict = {}

    def windows(self, w_max):
        if w_max in self._wins:
            return self._wins[w_max]
        R = self.R
        states = np.array(list(itertools.product(range(R), repeat=w_max)))
        Sn = len(states)
        pows = R ** np.arange(w_max - 1, -1, -1)
        shifted = np.concatenate([states[:, 1:], np.zeros((Sn, 1), int)], axis=1)
        succ = (shifted @ pows)[:, None] + np.arange(R)[None, :]
        Vw = {w: self.dmg[states[:, -w:]].mean(axis=1) for w in range(1, w_max + 1)}
        ctx = WindowCtx(states, succ, Vw)
        self._wins[w_max] = ctx
        return ctx


class WindowCtx:
    def __init__(self, states, succ, Vw):
        self.states, self.succ, self.Vw = states, succ, Vw


def _rows(Z, tau, prior):
    """Row-wise prior-shaped softmax: A ∝ prior * exp(Z/tau), normalised per row."""
    L = (Z - Z.max(axis=1, keepdims=True)) / tau
    A = prior[None, :] * np.exp(L)
    return A / A.sum(axis=1, keepdims=True)


def force_aim(fc: FieldCore, sites, doctrine, sigma_km):
    """The joint aim distribution A [Sn, H] of a resolved force (mixture over agents)."""
    w_max = max(int(d[4]) for d in doctrine)
    ctx = fc.windows(w_max)
    coords = fc.base.coords
    A = np.zeros((len(ctx.states), fc.H))
    for site, (q_rep, q_flee, q_eq, tau, w) in zip(sites, doctrine):
        if sigma_km:
            d2 = ((coords - coords[int(site)]) ** 2).sum(axis=1)
            prior = np.exp(-d2 / (2.0 * sigma_km ** 2))
        else:
            prior = np.ones(fc.H)
        prior = prior / prior.sum()
        Vw = ctx.Vw[min(int(w), w_max)]
        Ar = _rows(Vw, tau, prior)
        rflee = (Ar @ fc.dmg.T).argmin(axis=1)
        Z = q_rep * Vw + q_flee * fc.dmg[rflee] + q_eq * fc.Vstat[None, :]
        A += _rows(Z, tau, prior)
    return A / len(sites), ctx


def history_opt(stepdmg, succ, iters=4000, tol=1e-12):
    """Average damage to a best-responding history-aware defender, by relative value iteration."""
    V = np.zeros(stepdmg.shape[0])
    for _ in range(iters):
        Q = stepdmg + V[succ]
        Vn = 0.5 * Q.min(axis=1) + 0.5 * V
        Vd = Vn - Vn.mean()
        if np.abs(Vd - V).max() < tol:
            break
        V = Vd
    Q = stepdmg + V[succ]
    return float((Q.min(axis=1) - V).mean())


def force_value(fc: FieldCore, sites, doctrine, sigma_km, iters=4000, tol=1e-12):
    """The induced game value of a force = best-response damage (higher = harder world)."""
    A, ctx = force_aim(fc, sites, doctrine, sigma_km)
    stepdmg = A @ fc.dmg.T
    return history_opt(stepdmg, ctx.succ, iters=iters, tol=tol)


# ---------- baseline force constructors ----------

def random_force(base: ScoreBase, K, rng):
    """Know-nothing floor: iid uniform sites (stacking allowed), Dirichlet doctrine, uniform tau
    bin and memory."""
    sites = rng.choice(base.H, size=K, replace=True)
    doctrine = []
    for _ in range(K):
        q = rng.dirichlet((1.0, 1.0, 1.0))
        tau = float(rng.choice((0.05, 0.10, 0.20)))
        w = int(rng.integers(1, 4))
        doctrine.append((float(q[0]), float(q[1]), float(q[2]), tau, w))
    return [int(s) for s in sites], doctrine


def heuristic_force(base: ScoreBase, K):
    """Two-line doctrine heuristic: the K highest-exposure sites the rulebook implies, every agent
    on the screened anticipator operating point."""
    sites = list(np.argsort(-base.site_exposure)[:K])
    return [int(s) for s in sites], [GEN32_DOCTRINE] * K


def oracle_force(base: ScoreBase, K, sigma_km, seed=5100, n_random=48, n_ascent=12, rng_seed=0):
    """Oracle-optimised ceiling: random search over schema-legal forces then greedy per-agent site
    ascent, all scored on the search field seed. Callers re-score the winner on evaluation seeds."""
    rng = np.random.default_rng(rng_seed)
    fc = base.field(seed)
    docs = [(1.0, 0.0, 0.0), (0.7, 0.3, 0.0), (0.5, 0.5, 0.0), (0.34, 0.33, 0.33),
            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    p_site = base.site_exposure / base.site_exposure.sum()
    best_v, best = -np.inf, None
    for _ in range(n_random):
        sites = [int(s) for s in rng.choice(base.H, size=K, replace=True, p=p_site)]
        doctrine = []
        for _ in range(K):
            q = docs[int(rng.integers(len(docs)))]
            doctrine.append((q[0], q[1], q[2], float(rng.choice((0.05, 0.10))),
                             int(rng.choice((2, 3)))))
        v = force_value(fc, sites, doctrine, sigma_km)
        if v > best_v:
            best_v, best = v, (sites, doctrine)
    sites, doctrine = [list(best[0]), list(best[1])]
    pool = list(np.argsort(-base.site_exposure)[:24])
    for k in range(K):                                  # greedy per-agent site ascent
        cands = set(pool[:n_ascent]) | {int(s) for s in
                                        rng.choice(base.H, size=n_ascent, replace=False)}
        for s in cands:
            if s == sites[k]:
                continue
            trial = list(sites)
            trial[k] = int(s)
            v = force_value(fc, trial, doctrine, sigma_km)
            if v > best_v:
                best_v, sites = v, trial
    for k in range(K):                                  # one doctrine sweep on the winner
        for q in docs:
            trial = list(doctrine)
            trial[k] = (q[0], q[1], q[2], doctrine[k][3], doctrine[k][4])
            v = force_value(fc, sites, trial, sigma_km)
            if v > best_v:
                best_v, doctrine = v, trial
    return sites, doctrine, float(best_v)
