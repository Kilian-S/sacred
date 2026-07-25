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
                 n_lanes=14, n_terrain=12, menu_step=None, stratified=0, site_seed=0,
                 n_sites=0):
        self.path, self.terrain = path, (terrain if terrain is not None else terrain_v2())
        self.th = load_vec_theatre(path)
        # the cover-route seeding grid scales with the map so the big theatres stay affordable
        step = menu_step or max(1.0, float(np.sqrt(self.th.W * self.th.H / 1000.0)))
        self.menu_step = step
        game, menu, coords, rr, pp, S, lane_idx, cls = build_theatre_game(
            self.th, K=1, n_lanes=n_lanes, n_terrain=n_terrain, spacing_km=spacing_km,
            standoff_km=standoff_km, range_scale=range_scale, terrain=self.terrain,
            return_cls=True, stratified=stratified, site_seed=site_seed, n_sites=n_sites)
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

    def best_laydown(self, pp, K, pool=None, cand=60, restarts=3, iters=40, seed=0, n_out=0):
        """The best FORCE of K positions, not the K best positions.

        The screen's original picker took the K sites with the highest individual threat, which is
        not a force: with a dense candidate set it selects K nearly adjacent points in the same
        piece of ground and leaves the rest of the corridor open. Measured 2026-07-25: as the
        candidate count rose 563 -> 2360 the greedy force got monotonically WORSE against perfect
        play (ratio 0.28 -> 0.05), i.e. the numbers were reporting the picker rather than the
        terrain.

        Objective (cheap, and the one a planner actually has): maximise the damage on the SAFEST
        route, i.e. close every lane. Exact scoring still happens downstream on the chosen force;
        this only has to choose well, not to score. Greedy max-min seed + steepest-descent swaps,
        several restarts, restricted to the `cand` most threatening sites of the allowed classes.

        The surrogate is not the true objective (which is the defender's optimum under the enemy's
        doctrine), so it can lose to the old individual-threat picker on some maps: measured
        2026-07-25, +53% on ukraine but -9% on kaliningrad. `n_out>0` therefore returns the best
        n distinct candidate FORCES instead of one, so the caller can score them exactly and keep
        the winner. The force actually used is then never worse than the old picker's."""
        S = self.survival(pp)
        logS = np.log(np.clip(S, 1e-300, 1.0)) * N_FLEET                  # [R, H]
        idx = np.arange(self.H) if pool is None else np.asarray(pool, dtype=int)
        if len(idx) > cand:
            idx = idx[np.argsort(-self.threat_rank(pp)[idx])[:cand]]
        K = min(K, len(idx))

        def obj(L):
            return float((1.0 - np.exp(logS[:, list(L)].sum(axis=1))).min())

        best, best_v, out = None, -np.inf, []
        rng = np.random.default_rng(seed)
        for r in range(restarts):
            if r == 0:                                                    # greedy max-min seed
                L = []
                for _ in range(K):
                    L.append(int(max((c for c in idx if c not in L), key=lambda c: obj(L + [c]))))
            else:
                L = list(rng.choice(idx, size=K, replace=False))
            v = obj(L)
            for _ in range(iters):                                        # steepest-descent swaps
                moved = False
                for s in range(K):
                    for c in idx:
                        if c in L:
                            continue
                        alt = list(L); alt[s] = int(c)
                        av = obj(alt)
                        if av > v + 1e-12:
                            L, v, moved = alt, av, True
                if not moved:
                    break
            out.append((v, tuple(sorted(int(x) for x in L))))
            if v > best_v:
                best, best_v = list(L), v
        if not n_out:
            return np.array(sorted(best), dtype=int)
        seen, keep = set(), []
        for _, L in sorted(out, reverse=True):
            if L not in seen:
                seen.add(L); keep.append(np.array(L, dtype=int))
        return keep[:n_out]

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
        # known[s, j]: some route in the window was ENGAGEABLE by team j, and team j sits on
        # revealing ground. A deterministic function of the state, so the MDP is unchanged.
        #
        # SPOTTING FOLLOWS THE FIRE (Kilian 2026-07-25): a team relocates between serials within
        # its zone (the same concentration its damage is delivered from), so it is spotted when
        # the flight comes within range of ANY position it fights from, not only its nominal
        # site. The earlier own-site-only trigger let a team engage from the far side of its zone
        # while staying unspotted: free invisibility on open ground, biasing every hide-vs-open
        # number against concealment. "Fights from" = positions carrying at least 5% of the
        # team's peak concentration weight (the Gaussian tail cut, documented threshold); the
        # nominal site always carries the peak, so the new trigger is a superset of the old one.
        zone = self.prior_j >= 0.05 * self.prior_j.max(axis=1, keepdims=True)   # [k, H]
        engaged = (base.expo.astype(int) @ zone.T.astype(int)) > 0              # [R, k]
        self.revealable = engaged & base.reveal[self.L][None, :]
        self.known = np.zeros((Sn, len(self.L)), bool)
        for k in range(w):
            self.known |= self.revealable[self.states[:, k]]
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
        route's threat as judged from the teams in mask m (uniform over what is known). Uses the
        spot-where-it-fires trigger (self.revealable), same as the window form."""
        if getattr(self, "_mem", None) is None:
            k = len(self.L)
            bits = (1 << np.arange(k))
            expose = (self.revealable * bits[None, :]).sum(axis=1).astype(int)  # [R]
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


# --- force selection (gen39 finding 6) --------------------------------------------------------


def pick_laydown(base, pp, kind, K, rng):
    """Enemy laydown archetypes by top-K INDIVIDUAL threat (the original picker). Kept as one of
    the candidates `choose_force` scores, never trusted alone: with a dense candidate set it
    stacks K adjacent points and leaves the corridor open (finding 6)."""
    thr = base.threat_rank(pp)
    open_sites = np.where(~base.concealed)[0]
    hid_sites = np.where(base.concealed)[0]

    def top(idx, n):
        return idx[np.argsort(-thr[idx])[:n]]

    if kind == "open":
        return top(open_sites, K)
    if kind == "hidden":
        return top(hid_sites, K)
    if kind == "mixed":
        h = K // 2
        return np.concatenate([top(open_sites, K - h), top(hid_sites, h)])
    if kind == "random":
        return rng.choice(len(pp), size=K, replace=False)
    raise ValueError(kind)


def choose_force(base, pp, kind, K, rng, w=2, tau=0.10, doctrine=None, T=40):
    """The archetype's force from BOTH pickers (top-K individual threat AND the `best_laydown`
    combination search), every candidate scored EXACTLY (episodic optimum over a T-serial
    mission), keeping the winner and recording which picker produced it. The surrogate inside
    `best_laydown` can lose to the old picker on some maps, so scoring both exactly is what makes
    the chosen force never worse than either picker alone (finding 6).

    Returns (laydown, ConcealDyn, picker_name); the winner's game object is handed back so the
    caller never rebuilds it."""
    doctrine = doctrine or {}
    cands = [("topk", np.asarray(pick_laydown(base, pp, kind, K, rng)))]
    if kind == "mixed":
        h = K // 2
        parts = [base.best_laydown(pp, K - h, pool=np.where(~base.concealed)[0])]
        if h:
            parts.append(base.best_laydown(pp, h, pool=np.where(base.concealed)[0]))
        cands.append(("comb", np.concatenate(parts)))
    elif kind in ("open", "hidden"):
        pool = np.where(base.concealed if kind == "hidden" else ~base.concealed)[0]
        for i, L in enumerate(base.best_laydown(pp, K, pool=pool, n_out=3)):
            cands.append((f"comb{i}", L))
    best = None
    for name, L in cands:
        g = ConcealDyn(base, pp, L, w=w, tau=tau, **doctrine)
        v = g.episodic(T=T)
        if best is None or v > best[3]:
            best = (L, g, name, v)
    return best[0], best[1], best[2]
