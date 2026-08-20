#!/usr/bin/env python3
"""gen39 step 1: the dynamic screen (ORACLE-ONLY, FREE; experiments/gen39_concealment.md).

Finds the operating point where concealment makes the game interesting: where the simple rules a
practitioner would actually write leave real value on the table, and the game is not saturated.

Per cell we compute, exactly, on the window MDP:
  cap        = min(static equilibrium mixture, multi-start static local optimum)   [static ceiling]
  blind      = best TERRAIN-ONLY dynamic rule (rotation / anti-repeat over lanes and the menu)
  revealed   = best rule that also uses the sites its own recent track has exposed  [the gen39 rule]
  fit        = best field- or doctrine-informed rule                                [disclosed cap]
  opt        = the exact optimum for a defender that knows everything

Gates: G1 = cap/opt (is static play genuinely capped), G2 = min(blind, revealed)/opt (do the simple
rules leave value), G_conceal = blind/revealed (is the reveal channel worth anything at all).

    PYTHONPATH=. python analysis/gen39_screen.py --maps kgd_gvardeysk --quick
    PYTHONPATH=. python analysis/gen39_screen.py            # the full pre-registered sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from src.envs.aerial_conceal import (ConcealBase, ConcealDyn, choose_force, pick_laydown,  # noqa: F401
                                     resample_field)
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

MAPS = ["kgd_gvardeysk", "ukraine", "narva", "fulda"]
PATH = "data/maps/theatre_%s_vec.json"
OUT = "models/runs/gen39_screen.json"

# the gen32 pinned doctrine (q_rep/q_flee/q_ar). NOTE: the enemy habit window w=2 is a deliberate
# COST choice, not gen32's pinned w=3 (state space R^w). The defender's memory of DISCOVERED
# teams is whole-mission in the persistent arm (the arm the gates are read from), which is the
# quantity that must never be forgotten (Kilian 2026-07-25); w only bounds the enemy's memory of
# the defender's recent routes and the anti-repeat rules' window.
DOCTRINE = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3)
TAU, W = 0.10, 2
FIELDS = (5100, 5101, 5102)

# concealed-class lethality MULTIPLIER on the table's pinned 0.55. The two >1 points (added
# 2026-07-26 BEFORE the third run, disclosed) put effective forest lethality 0.70 and 0.90 in the
# grid, so the screen itself settles the raise-lethality question the v2 cost table opened; the
# four pre-registered points are unchanged.
HIDDEN_LETH = (0.4, 0.6, 0.8, 1.0, 1.27, 1.64)
RANGE_MULT = (0.7, 1.0, 1.3)            # on top of each map's comparability scale
TEAMS = (1, 2, 3, 4, 6)                 # emplaced teams (the enemy commits to ground)
CONCEAL_REACH = (0.43, 0.65, 0.85)      # forest reach as a fraction of open reach
KINDS = ("open", "hidden", "mixed", "random")   # laydown archetypes


def map_scale(name, ref_lat):
    return lateral_width(load_vec_theatre(PATH % name)) / ref_lat


def static_localopt(g, iters=20, pop=20, keep=5):
    rng = np.random.default_rng(0)
    exp = 1.0 - g.S[:, g.L].min(axis=1)
    pool = sorted(set(np.where(g.d_eq > 1e-6)[0]) | set(np.argsort(exp)[:12]))
    mu, best = np.full(len(pool), 1.0 / len(pool)), np.inf
    for _ in range(iters):
        smp = rng.dirichlet(mu * 24 + 0.3, size=pop)
        vals = []
        for s in smp:
            d = np.zeros(g.R); d[pool] = s
            vals.append(g.value_static(d))
        vals = np.array(vals)
        mu = 0.6 * mu + 0.4 * smp[np.argsort(vals)[:keep]].mean(axis=0)
        best = min(best, float(vals.min()))
    return best


def ladder(g: ConcealDyn):
    """Every arm, kept in its information class."""
    rows = {"iid_eq*fit": g.value_static(g.d_eq), "static_localopt*fit": static_localopt(g)}

    blind = g.blind_supports()
    for name, d in blind.items():                      # terrain-only, no field, no reveal
        rows[f"blind_static_{name}"] = g.value_static(d)
        rows[f"blind_anti_{name}"] = g.stationary(g._anti(d))
        sup = np.where(d > 1e-9)[0]
        if len(sup) > g.w:
            rot = np.zeros((len(g.states), g.R))
            for si in range(len(g.states)):
                cand = [r for r in sup if not g.in_window[si, r]]
                rot[si, cand[0] if cand else sup[0]] = 1.0
            rows[f"blind_rot_{name}"] = g.stationary(rot)

    for name, d in blind.items():                      # + what its own track has revealed
        for anti in (False, True):
            for soft in (0.0, 0.02, 0.05):
                tag = f"revealed_{name}{'_anti' if anti else ''}{'' if soft == 0 else f'_s{soft}'}"
                rows[tag] = g.stationary(g.avoid_revealed(d, anti_repeat=anti, softness=soft))

    for name, d in g.fit_supports().items():           # disclosed caps: they see the field
        rows[f"fit_anti_{name}"] = g.stationary(g._anti(d))
    dodge = np.zeros((len(g.states), g.R))
    dodge[np.arange(len(g.states)), g.stepdmg.argmin(axis=1)] = 1.0
    rows["myopic_dodge*fit"] = g.stationary(dodge)
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


def cell(base: ConcealBase, seed, hidden_leth, K, kind, tag):
    t0 = time.time()
    field = resample_field(base.coords, seed)
    pp = base.lethality(field, hidden_leth=hidden_leth)
    rng = np.random.default_rng(seed * 131 + K)
    L, g, picker = choose_force(base, pp, kind, K, rng, w=W, tau=TAU, doctrine=DOCTRINE)
    rows = ladder(g)
    opt = g.history_opt()

    def best(pref):
        ks = [k for k in rows if k.startswith(pref)]
        return (min(rows[k] for k in ks), min(ks, key=lambda k: rows[k])) if ks else (np.inf, "-")

    b_blind, n_blind = best("blind_")
    b_rev, n_rev = best("revealed_")
    b_fit = min(rows[k] for k in rows if k.endswith("*fit"))
    cap = min(rows["iid_eq*fit"], rows["static_localopt*fit"])
    simple = min(b_blind, b_rev)
    g1, g2 = cap / max(opt, 1e-9), simple / max(opt, 1e-9)
    gc = b_blind / max(b_rev, 1e-9)
    # a laydown a knowing defender can simply walk around is not a game: concealed teams are
    # short-ranged, so at low team counts they leave free routes and the optimum collapses to ~0.
    # Flagged rather than dropped, because WHERE that boundary sits is part of the result.
    degenerate = bool(opt < 5e-3 or cap > 0.90)
    phi = float(2.0 * np.asarray(base.rr)[L].sum() / lateral_width(base.th))
    rec = dict(tag=tag, seed=seed, hidden_leth=hidden_leth, K=K, kind=kind,
               degenerate=degenerate, phi=phi, conceal_reach=float(base.terrain["forest"]["r_km"]
                                                                  / base.terrain["open"]["r_km"]),
               n_conceal=int(base.concealed[L].sum()), opt=opt, cap=cap,
               blind=b_blind, blind_arm=n_blind, revealed=b_rev, revealed_arm=n_rev,
               fit=b_fit, eq_static=g.eq_static, G1=g1, G2=g2, G_conceal=gc, picker=picker,
               R=g.R, H=g.H, k_teams=len(L), mean_known=float(g.n_known.mean()),
               secs=time.time() - t0,
               rows={k: float(v) for k, v in rows.items()})
    print(f"  {tag} K{K}/{kind} hl{hidden_leth} s{seed}: opt={opt:.4f} cap={cap:.4f} blind={b_blind:.4f}"
          f"({n_blind[:22]}) rev={b_rev:.4f}({n_rev[:22]}) fit={b_fit:.4f} | G1={g1:.2f}"
          f" G2={g2:.2f} Gc={gc:.2f} known={rec['mean_known']:.1f}/{len(L)} [{rec['secs']:.0f}s]",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default=",".join(MAPS))
    ap.add_argument("--quick", action="store_true", help="one field, one range, coarse")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    maps = a.maps.split(",")
    fields = FIELDS[:1] if a.quick else FIELDS
    rmults = (1.0,) if a.quick else RANGE_MULT
    creach = (0.43,) if a.quick else CONCEAL_REACH
    hleth = (0.4, 1.0) if a.quick else HIDDEN_LETH
    teams = (3,) if a.quick else TEAMS
    kinds = KINDS

    ref_lat = lateral_width(load_vec_theatre(PATH % "kgd_gvardeysk"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = []
    for name in maps:
        sc = map_scale(name, ref_lat)
        for rm, cr in itertools.product(rmults, creach):
            t0 = time.time()
            # the site grid and the terminal standoff scale WITH the weapon ranges, so a candidate
            # position means the same thing on every map (2 km spacing against a 3.5 km reach on
            # Kaliningrad is 11.6 km against a 20 km reach on Fulda). Without this, the big maps
            # are both incomparable and needlessly expensive.
            # asymmetric-forest DEFAULT + the 200-site quota sampler, matching gen39_screen2
            # (this script's earlier runs used forest_los=True + the raster: DIFFERENT GAMES,
            # archived as *_symforest; rule 8)
            base = ConcealBase(PATH % name,
                               terrain=terrain_v2(hidden_leth=1.0, conceal_reach=cr),
                               range_scale=sc * rm, spacing_km=2.0 * sc, standoff_km=4.0 * sc,
                               n_sites=200)
            print(f"[{name} x{rm} cr{cr}] scale={sc * rm:.2f} R={base.R} H={base.H} "
                  f"lanes={len(base.lane_idx)} concealed={int(base.concealed.sum())} "
                  f"menu_step={base.menu_step:.1f}km build={time.time() - t0:.0f}s", flush=True)
            for K in teams:
                for kind in kinds:
                    for hl in hleth:
                        for seed in fields:
                            out.append(cell(base, seed, hl, K, kind, f"{name}x{rm}cr{cr}"))
                    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[written] {a.out} ({len(out)} cells)", flush=True)


if __name__ == "__main__":
    main()
