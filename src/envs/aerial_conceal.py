"""gen39: the concealment dynamic game (experiments/gen39_concealment.md).

Derives from the gen32 corridor-hunt machinery (`scratch/gen32_theatre_hunt.py`): the enemy is a
doctrine that aims by softmax against a model of the defender's recent track, the defender's state
is its own w-route window, and everything is exact on that window MDP (relative value iteration
for the optimum, stationary distributions for any state-conditioned rule).

What gen39 adds is the INFORMATION STRUCTURE. Under terrain table v2 the defender no longer knows
which emplacements are hot: it knows the public terrain, and it learns about a site only when that
site engages it AND the site sits on revealing ground (open, farmland). Sites in forest or town
engage without giving themselves away.

The reveal set is a deterministic function of the w-window, because which sites a route flies
through the rings of is fixed geometry, so the state space is UNCHANGED and every quantity stays
exactly computable. Memory is therefore the window, which is the bounded-memory form gen34 used.

Three rule classes, kept strictly apart because the whole point is what each is allowed to see:
  * BLIND      - terrain geometry only (lane structure), never the hot/cold field;
  * REVEALED   - blind, plus the sites its own recent track has exposed on revealing ground;
  * *fit       - field- or doctrine-informed (disclosed caps, never claimed as achievable).
"""
from __future__ import annotations

import itertools

import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.aerial_theatre_vec import (build_theatre_game, load_vec_theatre, reveal_flags,
                                         route_survival, containing_blockers, terrain_v2)

N_FLEET = 3


def _mm(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


def resample_field(coords, seed, length_scale=6.0, band=(0.55, 1.0)):
    """The hidden per-site effectiveness draw: a spatially correlated RBF over the real site
    coordinates, rank-mapped into a band.

    gen39 DESIGN CHANGE from gen32 (recorded): gen32 let this field REPLACE the terrain lethality,
    which would erase the whole point of table v2 (a concealed site is supposed to shoot weaker
    than an open one). Here the draw is a MULTIPLIER on the terrain's lethality, so the class
    structure survives and the field only says which positions are hot today. Band defaults
    accordingly to (0.55, 1.0) rather than gen32's absolute (0.30, 0.95)."""
    rng = np.random.default_rng(seed)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2.0 * length_scale ** 2)) + 1e-8 * np.eye(len(coords))
    g = rng.multivariate_normal(np.zeros(len(coords)), cov)
    ranks = np.argsort(np.argsort(g))
    lo, hi = band
    return lo + (hi - lo) * ranks / max(len(coords) - 1, 1)


class ConcealBase:
    """The fixed theatre: menu, sites, radii, terrain classes, reveal flags, exposure geometry.
    Built once per (theatre, terrain table, range scale); field-specific games are cheap."""

    def __init__(self, path, terrain=None, range_scale=1.0, spacing_km=2.0, standoff_km=4.0,
                 n_lanes=14, n_terrain=12, menu_step=None):
        self.path, self.terrain = path, (terrain if terrain is not None else terrain_v2())
        self.th = load_vec_theatre(path)
        # the cover-route seeding grid scales with the map so the big theatres stay affordable
        step = menu_step or max(1.0, float(np.sqrt(self.th.W * self.th.H / 1000.0)))
        self.menu_step = step
        game, menu, coords, rr, pp, S, lane_idx, cls = build_theatre_game(
            self.th, K=1, n_lanes=n_lanes, n_terrain=n_terrain, spacing_km=spacing_km,
            standoff_km=standoff_km, range_scale=range_scale, terrain=self.terrain,
            return_cls=True)
        self.menu, self.coords, self.rr, self.cls = menu, coords, rr, cls
        self.lane_idx, self.routes, self.route_edges = lane_idx, game.routes, game.route_edges
        self.isets, self.travel = game.interdiction_sets, game.travel_cost
        self.R, self.H = game.n_routes, len(coords)
        self.pp_base = np.asarray(pp, dtype=float)                       # terrain lethality [H]
        self.concealed = ~reveal_flags(cls, self.terrain)                # forest + urban
        self.own = containing_blockers(self.th, coords, self.terrain)
        self.reveal = reveal_flags(cls, self.terrain)                    # [H] revealing ground?
        # exposure geometry is field-INDEPENDENT (the taper is range-only), so compute it once:
        # expo[r, h] = route r flies inside site h's ring with line of sight.
        flat = np.ones(self.H)
        self.expo = np.stack([
            route_survival(self.th, m, coords, rr, flat * 0.5, los=True, terrain=self.terrain,
                           own_polys=self.own, return_exposed=True)[1] for m in menu])

    def lethality(self, field_mult, hidden_leth=1.0):
        """Per-site lethality = terrain class x the hidden hot/cold draw x the concealed-class
        knob. Keeping hidden_leth here (rather than in the terrain table) means the menu, the
        site set and the geometry are IDENTICAL across the sweep, so every cell is the same game
        with a different lethality vector: the standing same-menu comparison convention."""
        mult = np.where(self.concealed, hidden_leth, 1.0)
        return np.clip(self.pp_base * mult * np.asarray(field_mult), 1e-6, 0.999)

    def survival(self, pp):
        """S[r, h] for a lethality vector, cached: the screen asks for the same vector twice
        (once to rank ground for the laydown, once to build the game)."""
        key = np.asarray(pp, dtype=float).tobytes()
        if getattr(self, "_scache_key", None) != key:
            self._scache = np.stack([
                route_survival(self.th, m, self.coords, self.rr, pp, los=True,
                               terrain=self.terrain, own_polys=self.own) for m in self.menu])
            self._scache_key = key
        return self._scache

    def threat_rank(self, pp, n_fleet=N_FLEET):
        """Per site: the worst damage it could do to any single route. Ranks GROUND, which is what
        an enemy choosing where to emplace actually compares."""
        return (1.0 - self.survival(pp) ** n_fleet).max(axis=0)

    def game_for(self, pp_field):
        S = self.survival(pp_field)
        logS = np.log(np.clip(S, 1e-300, 1.0))
        idx = np.asarray(self.isets, dtype=int)
        payoff = 1.0 - np.exp(logS[:, idx].sum(axis=2))
        return InterdictionGame(self.routes, self.route_edges, self.isets, payoff,
                                self.travel, 1), S


class ConcealDyn:
    """One (field, doctrine, operating point) cell, exact on the window MDP."""

    def __init__(self, base: ConcealBase, pp_field, laydown, w=2, tau=0.10,
                 q_rep=0.6, q_flee=0.2, q_ar=0.3, sigma_r=1.5, same_class=True):
        """laydown: the site indices the enemy has EMPLACED on.

        Semantics are gen33's, which were regression-tested to reproduce the gen32 dynamic game
        exactly in the flat single-team case: each emplaced team contributes a spatial
        CONCENTRATION of engagement effort around its own position (an RBF bump of width
        sigma_km), and the enemy's aim each serial is its doctrine softmax WEIGHTED by that
        concentration. So the laydown decides where the enemy's weight sits without pretending it
        can only ever engage from K exact points.

        Two design facts, both measured in the 2026-07-25 smokes and recorded in the ledger:
        (i) an enemy free to re-aim over every site has nothing to reveal, so concealment is pure
        loss and the terrain classes are dead; (ii) an enemy pinned to K hard points is trivially
        evadable, because with 26 routes and 3 points a knowing defender simply flies a free route
        and the optimum collapses to zero. The concentration form avoids both."""
        game, S = base.game_for(pp_field)
        self.base, self.game, self.S = base, game, S
        self.R, self.w, self.H = base.R, w, base.H
        self.L = np.asarray(laydown, dtype=int)
        self.dmg = 1.0 - S ** N_FLEET                                    # [R, H]

        # per-team engagement concentration and its mixture (the enemy's spatial commitment).
        # The width is tied to the team's OWN weapon reach (sigma = sigma_r x r), not to a fixed
        # number of km: a short-range team in cover concentrates tightly, a long-range team on
        # open ground spreads. A map-sized sigma would smear every laydown over the whole theatre
        # and make the emplacement choice meaningless (measured, 2026-07-25).
        d2 = ((base.coords[None, :, :] - base.coords[self.L][:, None, :]) ** 2).sum(-1)
        sig = (sigma_r * np.asarray(base.rr)[self.L])[:, None]           # [k, 1]
        bump = np.exp(-d2 / (2.0 * np.clip(sig, 1e-6, None) ** 2))       # [k, H]
        if same_class:
            # A team manoeuvres within ITS OWN ground, not out of it. Without this the smear
            # leaks across terrain classes and the emplacement choice stops binding: measured on
            # kaliningrad 2026-07-25, a "forest" team delivered only 20% of its effect from
            # forest and 60% from OPEN ground, i.e. it drew open-ground reach and lethality while
            # keeping forest's invisibility (reveal is decided by the team's own site). That
            # diluted the price of concealment about fivefold and inflated every hide-vs-open
            # comparison. Masking to the own class removes the leak; an isolated patch collapses
            # the weight back onto the site itself, which is the right limit.
            cls = np.asarray(base.cls)
            bump = bump * (cls[None, :] == cls[self.L][:, None])
        self.prior_j = bump / np.clip(bump.sum(axis=1, keepdims=True), 1e-300, None)
        self.prior = self.prior_j.mean(axis=0)                           # [H]
        self.dmg_j = self.prior_j @ self.dmg.T                           # [k, R] threat per team

        # the static caps are computed against the EMPLACED enemy: one "interdiction option" per
        # team, whose payoff is that team's concentration-weighted damage, so the cap reflects the
        # enemy the defender actually faces rather than every site the terrain could host
        payoff_j = (self.prior_j @ game.payoff.T)                        # [k, R]
        game = InterdictionGame(game.routes, game.route_edges,
                                tuple((int(h),) for h in self.L),
                                payoff_j.T, game.travel_cost, 1)
        self.game = game
        sol = solve_multiconvoy(game, N_FLEET, "mission")
        occs = list(itertools.combinations_with_replacement(range(self.R), N_FLEET))
        d_eq = np.zeros(self.R)
        for i, o in enumerate(occs):
            if len(set(o)) == 1:
                d_eq[o[0]] += sol.defender_strategy[i]
        self.d_eq = d_eq / d_eq.sum() if d_eq.sum() > 0 else np.full(self.R, 1 / self.R)
        self.eq_static = float(sol.loss_mixed)

        R = self.R
        self.states = np.array(list(itertools.product(range(R), repeat=w)))
        Sn = len(self.states)
        self.in_window = np.zeros((Sn, R), bool)
        for k in range(w):
            self.in_window[np.arange(Sn), self.states[:, k]] = True

        # --- the enemy doctrine (gen31/gen32 form: punish the track, pre-aim at the obvious
        # escape, pre-aim at a naive spreader) ---------------------------------------------------
        Vw = self.dmg[self.states].mean(axis=1)                          # vs the recent track
        Zr = (Vw - Vw.max(axis=1, keepdims=True)) / tau
        Ar = np.exp(Zr); Ar /= Ar.sum(axis=1, keepdims=True)
        rflee = (Ar @ self.dmg.T).argmin(axis=1)                         # the obvious escape
        ar = (~self.in_window).astype(float)
        ar /= np.clip(ar.sum(axis=1, keepdims=True), 1e-12, None)
        Var = ar @ self.dmg                                              # vs a naive spreader
        Z = q_rep * Vw + q_flee * self.dmg[rflee] + q_ar * Var
        Zs = (Z - Z.max(axis=1, keepdims=True)) / tau
        A = np.exp(Zs) * self.prior[None, :]          # doctrine, weighted by where the enemy IS
        A /= np.clip(A.sum(axis=1, keepdims=True), 1e-300, None)
        self.aim = A                                                     # [Sn, H]
        self.stepdmg = A @ self.dmg.T                                    # [Sn, R]

        self.pows = R ** np.arange(w - 1, -1, -1)
        shifted = np.concatenate([self.states[:, 1:], np.zeros((Sn, 1), int)], axis=1)
        self.succ = (shifted @ self.pows)[:, None] + np.arange(R)[None, :]

        # --- gen39: what the defender has been shown --------------------------------------------
        # known[s, j]: some route in the window flew inside emplaced team j's ring, and that team
        # sits on revealing ground. A deterministic function of the state, so the MDP is unchanged.
        revealable = base.expo[:, self.L] & base.reveal[self.L][None, :]  # [R, k]
        self.known = np.zeros((Sn, len(self.L)), bool)
        for k in range(w):
            self.known |= revealable[self.states[:, k]]
        self.n_known = self.known.sum(axis=1)
        # perceived threat of each route given ONLY the teams whose position is known
        kf = self.known.astype(float)
        denom = np.clip(self.n_known, 1, None)[:, None]
        self.perceived = (kf @ self.dmg_j) / denom                       # [Sn, R]

    # --- persistent memory: the faithful form of "concealment buys persistence" -----------------
    #
    # With window memory a team that gives itself away is forgotten w serials later, so being
    # located costs an open-ground team almost nothing and hiding is under-rewarded. Here the
    # defender remembers every team it has seen for the whole mission, which is what the mechanic
    # is supposed to mean. The state gains the set of teams seen (2^k of them, k <= 6), and since
    # that set only ever GROWS, a long-run average would wash out exactly the phase of interest:
    # the measure becomes an EPISODIC one, expected damage over a T-serial mission starting from
    # complete ignorance, computed exactly by backward induction.

    def _memory_tables(self):
        """expose[r] = bitmask of revealable teams route r gives away; perceived_mask[m, r] = the
        route's threat as judged from the teams in mask m (uniform over what is known)."""
        if getattr(self, "_mem", None) is None:
            k = len(self.L)
            revealable = self.base.expo[:, self.L] & self.base.reveal[self.L][None, :]
            bits = (1 << np.arange(k))
            expose = (revealable * bits[None, :]).sum(axis=1).astype(int)      # [R]
            masks = np.arange(1 << k)
            memb = ((masks[:, None] & bits[None, :]) > 0).astype(float)        # [2^k, k]
            cnt = np.clip(memb.sum(axis=1, keepdims=True), 1.0, None)
            self._mem = (expose, (memb @ self.dmg_j) / cnt, memb.sum(axis=1))
        return self._mem

    def episodic(self, T=40, rule=None, start_mask=0, horizons=None):
        """Mean per-serial damage over a T-serial mission with PERSISTENT memory.

        rule=None returns the exact optimum for a defender that knows the laydown (backward
        induction). Otherwise rule(state_index, mask, perceived_row) -> [R] route distribution is
        evaluated. The start is averaged over track windows with nothing known, so no opening move
        is privileged.

        `horizons`: an iterable of mission lengths. The value after t backward steps IS the
        t-serial answer, so one sweep yields the whole mission-length curve and a dict {t: value}
        is returned instead of a scalar. Mission length is therefore free rather than a
        multiplier on the screen's cost."""
        expose, perceived, _ = self._memory_tables()
        Sn, R, M = len(self.states), self.R, 1 << len(self.L)
        nxt_m = np.bitwise_or(np.arange(M)[:, None], expose[None, :])          # [M, R]
        # the rule reads (state, mask) only, never V, so its matrix is built ONCE rather than once
        # per backward step: a T-fold saving on the term that otherwise dominates the screen
        W = None if rule is None else np.stack(
            [rule(np.arange(Sn), m, perceived[m]) for m in range(M)], axis=1)  # [Sn, M, R]
        want = sorted({int(t) for t in horizons}) if horizons else [int(T)]
        out, V = {}, np.zeros((Sn, M))
        for t in range(1, want[-1] + 1):
            Q = self.stepdmg[:, None, :] + V[self.succ[:, None, :], nxt_m[None, :, :]]  # [Sn,M,R]
            V = Q.min(axis=2) if W is None else (W * Q).sum(axis=2)
            if t in want:
                out[t] = float(V[:, start_mask].mean() / t)
        return out if horizons else out[want[-1]]

    def _topm_row(self, perc, m):
        """Uniform over the m routes that look safest given what is known: the rule a practitioner
        actually writes ("avoid the worst few, pick at random among the rest"). Unlike a softmax
        it keeps a flat, genuinely random spread, which is what a repetition-punishing enemy is
        hardest to exploit by."""
        row = np.zeros(self.R)
        row[np.argsort(perc)[:max(1, min(m, self.R))]] = 1.0
        return row / row.sum()

    def episodic_rule(self, fallback, anti_repeat=False, softness=0.0, T=40, topm=0,
                      horizons=None):
        """The avoid-revealed rule under persistent memory: fly the route that looks safest given
        every team seen so far this mission, falling back to the blind rule while nothing is
        known."""
        base_m = self._anti(fallback) if anti_repeat else np.broadcast_to(
            fallback, (len(self.states), self.R)).copy()
        base_m = np.array(base_m, dtype=float)

        def rule(idx, m, perc):
            if m == 0:
                return base_m
            if topm:
                out = np.broadcast_to(self._topm_row(perc, topm), (len(idx), self.R)).copy()
            elif softness > 0:
                L = -(perc - perc.min()) / softness
                E = np.exp(L - L.max())
                row = E / max(E.sum(), 1e-300)
                out = np.broadcast_to(row, (len(idx), self.R)).copy()
            else:
                out = np.zeros((len(idx), self.R))
                out[:, int(np.argmin(perc))] = 1.0
            if anti_repeat:
                out = np.where(self.in_window, 0.0, out)
                s = out.sum(axis=1, keepdims=True)
                out = np.where(s > 1e-12, out / np.where(s > 1e-12, s, 1.0), base_m)
            return out
        return self.episodic(T=T, rule=rule, horizons=horizons)

    # --- exact evaluators (gen32 verbatim) ------------------------------------------------------

    def history_opt(self, iters=6000, tol=1e-12):
        """The exact optimum for a defender that knows everything (damped relative value
        iteration; agrees with Karp minimum-mean-cycle, the 2026-07-23 repair)."""
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

    def stationary(self, rule_mat, iters=800, damp=0.5, tol=1e-13):
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

    # --- rule supports ---------------------------------------------------------------------------

    def blind_supports(self):
        """TERRAIN-ONLY supports: lane structure and the menu, nothing that reads the field."""
        out = {"uniform_lanes": np.zeros(self.R), "uniform_full": np.full(self.R, 1.0 / self.R)}
        out["uniform_lanes"][self.base.lane_idx] = 1.0 / max(len(self.base.lane_idx), 1)
        return out

    def fit_supports(self):
        """FIELD-INFORMED supports (disclosed caps): they read the hidden effectiveness draw."""
        exp = 1.0 - self.S[:, self.L].min(axis=1)
        iv = np.zeros(self.R)
        iv[self.base.lane_idx] = 1.0 / np.clip(exp[self.base.lane_idx], 1e-6, None)
        ivf = 1.0 / np.clip(exp, 1e-6, None)
        return {"invrisk_lanes*fit": iv / max(iv.sum(), 1e-12),
                "invrisk_full*fit": ivf / ivf.sum(), "eq_support*fit": self.d_eq}

    def _anti(self, d):
        m = np.broadcast_to(d, (len(self.states), self.R)).copy()
        m[self.in_window] = 0.0
        s = m.sum(axis=1, keepdims=True)
        return np.where(s > 1e-12, m / np.where(s > 1e-12, s, 1.0),
                        np.broadcast_to(d, m.shape))

    def avoid_revealed(self, fallback, anti_repeat=False, softness=0.0):
        """The gen39 two-line rule: fly the route that looks safest given ONLY the sites your own
        recent track has exposed on revealing ground; fall back to the blind rule when nothing is
        known yet. softness>0 turns the argmin into a softmax (a fairer, less brittle rule)."""
        Sn = len(self.states)
        base_m = self._anti(fallback) if anti_repeat else np.broadcast_to(
            fallback, (Sn, self.R)).copy()
        m = np.array(base_m, dtype=float)
        seen = self.n_known > 0
        if not seen.any():
            return m
        P = self.perceived[seen]
        if softness > 0:
            L = -(P - P.min(axis=1, keepdims=True)) / softness
            if anti_repeat:
                L = np.where(self.in_window[seen], -np.inf, L)
            E = np.exp(L - L.max(axis=1, keepdims=True))
            row = E / np.clip(E.sum(axis=1, keepdims=True), 1e-300, None)
        else:
            Pm = np.where(self.in_window[seen], np.inf, P) if anti_repeat else P
            Pm = np.where(np.isfinite(Pm).any(axis=1, keepdims=True), Pm, P)
            row = np.zeros_like(P)
            row[np.arange(len(P)), Pm.argmin(axis=1)] = 1.0
        m[seen] = row
        return m
