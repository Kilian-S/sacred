"""Concealment dynamic game on the vector theatre, exact on the defender's window MDP.

The enemy aims by softmax against a model of the defender's recent track and the defender's state is
its own w-route window, so relative value iteration gives the optimum and stationary distributions
score any state-conditioned rule. What makes it a concealment game is the information structure: the
defender knows the public terrain, but learns about a site only once that site engages it from
revealing ground, while sites in forest or town engage without giving themselves away. Which sites a
route flies within reach of is fixed geometry, so the reveal set is a deterministic function of the
window and the state space is unchanged. Rules are kept in three classes by what each may see,
blind ones reading terrain geometry alone, revealed ones adding the sites their own track has
exposed, and ``*fit`` ones reading the hidden field as a disclosed cap.
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
    """The hidden per-site effectiveness draw, a spatially correlated RBF rank-mapped into a band.

    The draw multiplies the terrain's lethality rather than replacing it, so the terrain class
    structure survives and the field only says which positions are hot today.
    """
    rng = np.random.default_rng(seed)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2.0 * length_scale ** 2)) + 1e-8 * np.eye(len(coords))
    g = rng.multivariate_normal(np.zeros(len(coords)), cov)
    ranks = np.argsort(np.argsort(g))
    lo, hi = band
    return lo + (hi - lo) * ranks / max(len(coords) - 1, 1)


class ConcealBase:
    """The fixed theatre, holding the menu, sites, radii, terrain classes and exposure geometry.

    Built once per theatre, terrain table and range scale, after which field-specific games are
    cheap to derive.
    """

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
        self.concealed = ~reveal_flags(cls, self.terrain)                # forest and urban
        self.own = containing_blockers(self.th, coords, self.terrain)
        self.reveal = reveal_flags(cls, self.terrain)                    # [H] revealing ground
        # the taper is range-only, so exposure geometry is field-independent and computed once.
        # expo[r, h] = route r flies inside site h's ring with line of sight.
        flat = np.ones(self.H)
        self.expo = np.stack([
            route_survival(self.th, m, coords, rr, flat * 0.5, los=True, terrain=self.terrain,
                           own_polys=self.own, return_exposed=True)[1] for m in menu])

    def lethality(self, field_mult, hidden_leth=1.0):
        """Per-site lethality, the terrain class times the hidden draw times the concealment knob.

        Applying ``hidden_leth`` here rather than in the terrain table keeps the menu, the site set
        and the geometry identical across a sweep, so every cell is the same game under a different
        lethality vector.
        """
        mult = np.where(self.concealed, hidden_leth, 1.0)
        return np.clip(self.pp_base * mult * np.asarray(field_mult), 1e-6, 0.999)

    def survival(self, pp):
        """Per-route, per-site survival for a lethality vector, cached across repeated calls."""
        key = np.asarray(pp, dtype=float).tobytes()
        if getattr(self, "_scache_key", None) != key:
            self._scache = np.stack([
                route_survival(self.th, m, self.coords, self.rr, pp, los=True,
                               terrain=self.terrain, own_polys=self.own) for m in self.menu])
            self._scache_key = key
        return self._scache

    def threat_rank(self, pp, n_fleet=N_FLEET):
        """Per site, the worst damage it could do to any single route, which ranks the ground."""
        return (1.0 - self.survival(pp) ** n_fleet).max(axis=0)

    def best_laydown(self, pp, K, pool=None, cand=60, restarts=3, iters=40, seed=0, n_out=0):
        """The best force of K positions, as opposed to the K best individual positions.

        Taking the K sites of highest individual threat is not a force, because on a dense candidate
        set it stacks nearly adjacent points on one piece of ground and leaves the rest of the
        corridor open. The objective here is instead to maximise the damage on the safest route,
        that is to close every lane, searched by a greedy max-min seed and steepest-descent swaps
        over several restarts, restricted to the ``cand`` most threatening sites of the allowed
        classes.

        Args:
            n_out: above zero, return the best n distinct candidate forces rather than one, so the
                caller can score them exactly. The surrogate is not the true objective, so this is
                what keeps the force finally used from being worse than a simpler picker's.
        """
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
                 q_rep=0.6, q_flee=0.2, q_ar=0.3, sigma_r=1.5, same_class=True,
                 doctrines=None):
        """Build one cell of the game for a given enemy laydown.

        Each emplaced team contributes a spatial concentration of engagement effort around its own
        position, and the enemy's aim each serial is its doctrine softmax weighted by that
        concentration, so the laydown decides where the enemy's weight sits without pretending it
        can only ever engage from K exact points. Both limits of that choice are degenerate: an
        enemy free to re-aim over every site has nothing to reveal, which makes concealment pure
        loss, while an enemy pinned to K hard points is trivially evadable by flying a free route.

        Args:
            laydown: the site indices the enemy has emplaced on.
        """
        game, S = base.game_for(pp_field)
        self.base, self.game, self.S = base, game, S
        self.R, self.w, self.H = base.R, w, base.H
        self.L = np.asarray(laydown, dtype=int)
        self.dmg = 1.0 - S ** N_FLEET                                    # [R, H]

        # per-team engagement concentration and its mixture, the enemy's spatial commitment. The
        # width is tied to the team's own weapon reach (sigma = sigma_r x r) rather than a fixed
        # number of km, so a short-range team in cover concentrates tightly while a long-range team
        # on open ground spreads. A map-sized sigma would smear every laydown over the whole
        # theatre and make the emplacement choice meaningless.
        d2 = ((base.coords[None, :, :] - base.coords[self.L][:, None, :]) ** 2).sum(-1)
        sig = (sigma_r * np.asarray(base.rr)[self.L])[:, None]           # [k, 1]
        bump = np.exp(-d2 / (2.0 * np.clip(sig, 1e-6, None) ** 2))       # [k, H]
        if same_class:
            # a team manoeuvres within its own ground, not out of it. Without this mask the smear
            # leaks across terrain classes and the emplacement choice stops binding, since a team
            # would draw open-ground reach and lethality while keeping cover's invisibility, as
            # revealing is decided by its own site. An isolated patch collapses the weight back
            # onto the site itself, which is the right limit.
            cls = np.asarray(base.cls)
            bump = bump * (cls[None, :] == cls[self.L][:, None])
        self.prior_j = bump / np.clip(bump.sum(axis=1, keepdims=True), 1e-300, None)
        self.prior = self.prior_j.mean(axis=0)                           # [H]
        self.dmg_j = self.prior_j @ self.dmg.T                           # [k, R] threat per team

        # the static caps are computed against the emplaced enemy, one interdiction option per team
        # whose payoff is that team's concentration-weighted damage, so the cap reflects the enemy
        # the defender actually faces rather than every site the terrain could host
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

        # --- the enemy doctrine: punish the track, pre-aim at the obvious escape, and pre-aim at
        # a naive spreader ------------------------------------------------------------------------
        if doctrines is None:
            Vw = self.dmg[self.states].mean(axis=1)                      # vs the recent track
            Zr = (Vw - Vw.max(axis=1, keepdims=True)) / tau
            Ar = np.exp(Zr); Ar /= Ar.sum(axis=1, keepdims=True)
            rflee = (Ar @ self.dmg.T).argmin(axis=1)                     # the obvious escape
            ar = (~self.in_window).astype(float)
            ar /= np.clip(ar.sum(axis=1, keepdims=True), 1e-12, None)
            Var = ar @ self.dmg                                          # vs a naive spreader
            Z = q_rep * Vw + q_flee * self.dmg[rflee] + q_ar * Var
            Zs = (Z - Z.max(axis=1, keepdims=True)) / tau
            A = np.exp(Zs) * self.prior[None, :]      # doctrine, weighted by where the enemy is
            A /= np.clip(A.sum(axis=1, keepdims=True), 1e-300, None)
        else:
            # per-team doctrines. Each team contributes unit-peak eagerness over its own zone
            # shaped by its own doctrine, and one joint normalisation allocates the serial's
            # engagement across the force, so identical doctrines reproduce the single-doctrine
            # game exactly. The extra q_hold component sits on the zone's best ground regardless
            # of the track, and a team may carry a shorter memory w_j <= w, reading the most recent
            # w_j of the window; longer memories are clamped to w.
            assert len(doctrines) == len(self.L)
            ar = (~self.in_window).astype(float)
            ar /= np.clip(ar.sum(axis=1, keepdims=True), 1e-12, None)
            Var = ar @ self.dmg
            hold = self.dmg.max(axis=0)[None, :]                         # site value, track-free
            U = np.zeros((len(self.states), self.H))
            for j, d in enumerate(doctrines):
                wj = int(np.clip(d.get("w", w), 1, w))
                tj = float(d.get("tau", tau))
                Vw = self.dmg[self.states[:, self.w - wj:]].mean(axis=1)
                Zr = (Vw - Vw.max(axis=1, keepdims=True)) / tj
                Arj = np.exp(Zr); Arj /= Arj.sum(axis=1, keepdims=True)
                rflee = (Arj @ self.dmg.T).argmin(axis=1)
                Z = (d.get("q_rep", 0.0) * Vw + d.get("q_flee", 0.0) * self.dmg[rflee]
                     + d.get("q_ar", 0.0) * Var + d.get("q_hold", 0.0) * hold)
                Zs = (Z - Z.max(axis=1, keepdims=True)) / tj
                U += np.exp(Zs) * self.prior_j[j][None, :]
            A = U / np.clip(U.sum(axis=1, keepdims=True), 1e-300, None)
        self.aim = A                                                     # [Sn, H]
        self.stepdmg = A @ self.dmg.T                                    # [Sn, R]

        self.pows = R ** np.arange(w - 1, -1, -1)
        shifted = np.concatenate([self.states[:, 1:], np.zeros((Sn, 1), int)], axis=1)
        self.succ = (shifted @ self.pows)[:, None] + np.arange(R)[None, :]

        # --- what the defender has been shown -----------------------------------------------------
        # known[s, j] means some route in the window was engageable by team j and team j sits on
        # revealing ground. It is a deterministic function of the state, so the MDP is unchanged.
        #
        # Spotting follows the fire: a team relocates between serials within the same zone its
        # damage is delivered from, so it is spotted when the flight comes within range of any
        # position it fights from, not only its nominal site. Fighting positions are those carrying
        # at least 5% of the team's peak concentration weight, and the nominal site always carries
        # the peak.
        zone = self.prior_j >= 0.05 * self.prior_j.max(axis=1, keepdims=True)   # [k, H]
        engaged = (base.expo.astype(int) @ zone.T.astype(int)) > 0              # [R, k]
        self.revealable = engaged & base.reveal[self.L][None, :]
        self.known = np.zeros((Sn, len(self.L)), bool)
        for k in range(w):
            self.known |= self.revealable[self.states[:, k]]
        self.n_known = self.known.sum(axis=1)
        # perceived threat of each route given only the teams whose position is known
        kf = self.known.astype(float)
        denom = np.clip(self.n_known, 1, None)[:, None]
        self.perceived = (kf @ self.dmg_j) / denom                       # [Sn, R]

    # --- persistent memory ----------------------------------------------------------------------
    #
    # With window memory alone a team that gives itself away is forgotten w serials later, so being
    # located costs an open-ground team almost nothing and hiding is under-rewarded. Here the
    # defender instead remembers every team it has seen for the whole mission. The state gains the
    # set of teams seen, 2^k of them, and because that set only ever grows a long-run average would
    # wash out the phase of interest, so the measure is episodic: expected damage over a T-serial
    # mission starting from complete ignorance, computed exactly by backward induction.

    def _memory_tables(self):
        """Build the persistent-memory tables.

        Returns:
            ``expose[r]``, the bitmask of revealable teams route r gives away, the route threat as
            judged from the teams in each mask, uniform over what is known, and the mask sizes.
        """
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
        """Mean per-serial damage over a T-serial mission under persistent memory.

        The start is averaged over track windows with nothing known, so no opening move is
        privileged.

        Args:
            rule: None gives the exact optimum for a defender that knows the laydown. Otherwise
                ``rule(state_index, mask, perceived_row) -> [R]`` is evaluated as a route
                distribution.
            horizons: an iterable of mission lengths. The value after t backward steps is the
                t-serial answer, so one sweep yields the whole curve and a ``{t: value}`` dict is
                returned in place of a scalar.
        """
        expose, perceived, _ = self._memory_tables()
        Sn, R, M = len(self.states), self.R, 1 << len(self.L)
        nxt_m = np.bitwise_or(np.arange(M)[:, None], expose[None, :])          # [M, R]
        # the rule reads (state, mask) only and never V, so its matrix is built once rather than
        # once per backward step
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
        """Uniform over the m routes that look safest given what is known.

        This is the rule a practitioner writes, avoiding the worst few and picking at random among
        the rest. Unlike a softmax it keeps a flat spread, which a repetition-punishing enemy finds
        hardest to exploit.
        """
        row = np.zeros(self.R)
        row[np.argsort(perc)[:max(1, min(m, self.R))]] = 1.0
        return row / row.sum()

    def episodic_rule(self, fallback, anti_repeat=False, softness=0.0, T=40, topm=0,
                      horizons=None):
        """The avoid-revealed rule under persistent memory.

        Fly the route that looks safest given every team seen so far this mission, falling back to
        the blind rule while nothing is known.
        """
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

    # --- exact evaluators -----------------------------------------------------------------------

    def history_opt(self, iters=6000, tol=1e-12):
        """The exact optimum for a defender that knows everything, by relative value iteration."""
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
        """Terrain-only supports, built from lane structure and the menu alone."""
        out = {"uniform_lanes": np.zeros(self.R), "uniform_full": np.full(self.R, 1.0 / self.R)}
        out["uniform_lanes"][self.base.lane_idx] = 1.0 / max(len(self.base.lane_idx), 1)
        return out

    def fit_supports(self):
        """Field-informed supports, which read the hidden effectiveness draw and so are caps."""
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
        """Fly the route that looks safest given only the sites the recent track has exposed.

        Falls back to the blind rule while nothing is known yet.

        Args:
            softness: above zero, turns the argmin into a softmax, giving a less brittle rule.
        """
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


# --- force selection --------------------------------------------------------------------------


def pick_laydown(base, pp, kind, K, rng):
    """Enemy laydown archetypes taken by top-K individual threat.

    Kept as one of the candidates `choose_force` scores and never trusted alone, since on a dense
    candidate set it stacks K adjacent points and leaves the corridor open.
    """
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
    """Choose the archetype's force by scoring both pickers exactly.

    Candidates come from top-K individual threat and from the `best_laydown` combination search,
    and each is scored by its episodic optimum over a T-serial mission. Since the surrogate inside
    `best_laydown` can lose to the simpler picker on some maps, scoring both is what keeps the
    chosen force from being worse than either picker alone.

    Returns:
        ``(laydown, ConcealDyn, picker_name)``, handing back the winner's game object so the caller
        never rebuilds it.
    """
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
