#!/usr/bin/env python3
"""gen39 step 1b: the PAIRED-MEMORY screen, run block-parallel (ORACLE-ONLY, FREE).

Same grid as `gen39_screen.py`, but every cell is scored under BOTH defender memories:

  forgetful   the w-serial track window (what step 1a ran): a located team is forgotten w serials
              later, so being seen is nearly free and the concealment trade is muted;
  persistent  the faithful form of "concealment buys persistence": the defender remembers every
              team it has seen for the whole mission. The set of seen teams only GROWS, so a
              long-run average washes out the phase of interest and the measure becomes EPISODIC,
              expected damage over a T-serial mission from complete ignorance, exact by backward
              induction. One sweep yields the whole mission-length curve (see `episodic`).

Pairing both memories in ONE cell is the point: the enemy, field, laydown and doctrine are shared,
so the difference between the two columns IS the price of the defender's memory, with nothing else
moving. Mission length becomes a real axis rather than a nuisance.

Parallelism is by BLOCK (map x range multiplier x concealed reach), one OS process each, every
maths thread pool capped to 1 so the workers do not fight (standing dogma). Blocks are resumable:
a block whose output file exists is skipped.

    PYTHONPATH=. python scratch/gen39_screen2.py --list
    PYTHONPATH=. python scratch/gen39_screen2.py --block 0 --quick     # one block, in this process
    PYTHONPATH=. python scratch/gen39_screen2.py --launch --workers 9  # the full sweep
    PYTHONPATH=. python scratch/gen39_screen2.py --merge
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from scratch.gen39_screen import (CONCEAL_REACH, DOCTRINE, FIELDS, HIDDEN_LETH, KINDS, MAPS, PATH,
                                  RANGE_MULT, TAU, TEAMS, W, map_scale, pick_laydown)
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

OUTDIR = "models/runs/gen39_screen2"
HORIZONS = (10, 20, 40, 80)     # mission lengths, free: one backward sweep gives them all
T_PIN = 40                      # the pinned mission length the gates are read at
SOFTNESS = (0.0, 0.05, 0.2, 0.5)
TOPM = (2, 3, 5)


def blocks():
    return [dict(i=i, map=m, rm=rm, cr=cr) for i, (m, rm, cr)
            in enumerate(itertools.product(MAPS, RANGE_MULT, CONCEAL_REACH))]


def block_path(b, outdir=OUTDIR):
    return f"{outdir}/b{b['i']:02d}_{b['map']}_r{b['rm']}_c{b['cr']}.json"


# --- the static cap, returning its MIXTURE too (needed to score the same static play episodically)
def static_localopt_d(g, iters=20, pop=20, keep=5):
    """Verbatim `gen39_screen.static_localopt`, but it also hands back the best mixture so the SAME
    static defender can be scored under both memories. Same rng and schedule, so the value is the
    one step 1a recorded."""
    rng = np.random.default_rng(0)
    exp = 1.0 - g.S[:, g.L].min(axis=1)
    pool = sorted(set(np.where(g.d_eq > 1e-6)[0]) | set(np.argsort(exp)[:12]))
    mu, best, best_d = np.full(len(pool), 1.0 / len(pool)), np.inf, None
    for _ in range(iters):
        smp = rng.dirichlet(mu * 24 + 0.3, size=pop)
        vals = []
        for s in smp:
            d = np.zeros(g.R)
            d[pool] = s
            vals.append(g.value_static(d))
        vals = np.array(vals)
        if float(vals.min()) < best:
            best = float(vals.min())
            best_d = np.zeros(g.R)
            best_d[pool] = smp[int(vals.argmin())]
        mu = 0.6 * mu + 0.4 * smp[np.argsort(vals)[:keep]].mean(axis=0)
    return best, best_d


def ladder_forgetful(g, sl_val):
    """`gen39_screen.ladder` with the static local optimum passed in rather than recomputed (it is
    the most expensive term in the cell and the persistent arm needs its mixture anyway).
    `--check` asserts this reproduces the step-1a ladder exactly."""
    rows = {"iid_eq*fit": g.value_static(g.d_eq), "static_localopt*fit": sl_val}
    blind = g.blind_supports()
    for name, d in blind.items():
        rows[f"blind_static_{name}"] = g.value_static(d)
        rows[f"blind_anti_{name}"] = g.stationary(g._anti(d))
        sup = np.where(d > 1e-9)[0]
        if len(sup) > g.w:
            rot = np.zeros((len(g.states), g.R))
            for si in range(len(g.states)):
                cand = [r for r in sup if not g.in_window[si, r]]
                rot[si, cand[0] if cand else sup[0]] = 1.0
            rows[f"blind_rot_{name}"] = g.stationary(rot)
    for name, d in blind.items():
        for anti in (False, True):
            for soft in (0.0, 0.02, 0.05):
                tag = f"revealed_{name}{'_anti' if anti else ''}{'' if soft == 0 else f'_s{soft}'}"
                rows[tag] = g.stationary(g.avoid_revealed(d, anti_repeat=anti, softness=soft))
    for name, d in g.fit_supports().items():
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
            m = np.exp(L)
            m /= m.sum(axis=1, keepdims=True)
            bv = min(bv, g.stationary(m))
        rows[f"{tag}*fit"] = bv
    return rows


def _episodic_fixed(g, mat, horizons=HORIZONS):
    """Score a memoryless (state-only) rule matrix episodically, so static and blind play sit on
    the SAME yardstick as the memory-using rules."""
    m = np.asarray(mat, dtype=float)
    return g.episodic(rule=lambda idx, mask, perc: m, horizons=horizons)


def ladder_persistent(g, sl_d, horizons=HORIZONS):
    """Every arm again, under whole-mission memory. Rows are {T: value} curves."""
    rows = {}
    Sn = len(g.states)
    for tag, d in (("iid_eq*fit", g.d_eq), ("static_localopt*fit", sl_d)):
        rows[tag] = _episodic_fixed(g, np.broadcast_to(d, (Sn, g.R)), horizons)
    blind = g.blind_supports()
    for name, d in blind.items():        # terrain only: these use no memory at all, by construction
        rows[f"blind_static_{name}"] = _episodic_fixed(g, np.broadcast_to(d, (Sn, g.R)), horizons)
        rows[f"blind_anti_{name}"] = _episodic_fixed(g, g._anti(d), horizons)
    for name, d in blind.items():        # + every team seen SO FAR THIS MISSION
        for anti in (False, True):
            a = "_anti" if anti else ""
            for soft in SOFTNESS:
                tag = f"revealed_{name}{a}{'' if soft == 0 else f'_s{soft}'}"
                rows[tag] = g.episodic_rule(d, anti_repeat=anti, softness=soft, horizons=horizons)
            for m in TOPM:               # "avoid the worst few, spread over the rest"
                rows[f"revealed_{name}{a}_top{m}"] = g.episodic_rule(
                    d, anti_repeat=anti, topm=m, horizons=horizons)
    for name, d in g.fit_supports().items():
        rows[f"fit_anti_{name}"] = _episodic_fixed(g, g._anti(d), horizons)
    return rows


def _best(rows, pref, pick=lambda v: v):
    ks = [k for k in rows if k.startswith(pref)]
    if not ks:
        return float("inf"), "-"
    k = min(ks, key=lambda k: pick(rows[k]))
    return float(pick(rows[k])), k


def cell(base, seed, hidden_leth, K, kind, tag, horizons=HORIZONS):
    t0 = time.time()
    pp = base.lethality(resample_field(base.coords, seed), hidden_leth=hidden_leth)
    rng = np.random.default_rng(seed * 131 + K)
    L = pick_laydown(base, pp, kind, K, rng)
    g = ConcealDyn(base, pp, L, w=W, tau=TAU, **DOCTRINE)

    sl_val, sl_d = static_localopt_d(g)
    fg = ladder_forgetful(g, sl_val)
    opt_f = g.history_opt()
    pe = ladder_persistent(g, sl_d, horizons)
    opt_p = g.episodic(horizons=horizons)

    at = lambda v: v[T_PIN]                                       # noqa: E731 (read the pinned T)
    out = {}
    for mem, rows, opt, pick in (("forgetful", fg, opt_f, lambda v: v),
                                 ("persistent", pe, at(opt_p), at)):
        b_blind, n_blind = _best(rows, "blind_", pick)
        b_rev, n_rev = _best(rows, "revealed_", pick)
        cap = min(pick(rows["iid_eq*fit"]), pick(rows["static_localopt*fit"]))
        simple = min(b_blind, b_rev)
        out[mem] = dict(
            opt=opt, cap=cap, blind=b_blind, blind_arm=n_blind, revealed=b_rev, revealed_arm=n_rev,
            G1=cap / max(opt, 1e-9), G2=simple / max(opt, 1e-9), G_conceal=b_blind / max(b_rev, 1e-9),
            degenerate=bool(opt < 5e-3 or cap > 0.90))

    rec = dict(tag=tag, seed=seed, hidden_leth=hidden_leth, K=K, kind=kind,
               phi=float(2.0 * np.asarray(base.rr)[L].sum() / lateral_width(base.th)),
               conceal_reach=float(base.terrain["forest"]["r_km"] / base.terrain["open"]["r_km"]),
               n_conceal=int(base.concealed[L].sum()), eq_static=g.eq_static,
               R=g.R, H=g.H, k_teams=int(len(L)), mean_known=float(g.n_known.mean()),
               opt_curve={str(t): v for t, v in opt_p.items()},
               forgetful=out["forgetful"], persistent=out["persistent"],
               rows_forgetful={k: float(v) for k, v in fg.items()},
               rows_persistent={k: {str(t): float(x) for t, x in v.items()} for k, v in pe.items()},
               secs=time.time() - t0)
    f, p = out["forgetful"], out["persistent"]
    print(f"  {tag} K{K}/{kind} hl{hidden_leth} s{seed}: "
          f"FORGET opt={f['opt']:.4f} G1={f['G1']:.2f} G2={f['G2']:.2f} Gc={f['G_conceal']:.2f} | "
          f"PERSIST(T{T_PIN}) opt={p['opt']:.4f} G1={p['G1']:.2f} G2={p['G2']:.2f} "
          f"Gc={p['G_conceal']:.2f} [{rec['secs']:.1f}s]", flush=True)
    return rec


def run_block(b, outdir=OUTDIR, quick=False, force=False, check=False):
    dst = block_path(b, outdir)
    if os.path.exists(dst) and not force:
        print(f"[skip] {dst} exists", flush=True)
        return dst
    ref_lat = lateral_width(load_vec_theatre(PATH % "kgd_gvardeysk"))
    sc = map_scale(b["map"], ref_lat)
    t0 = time.time()
    base = ConcealBase(PATH % b["map"],
                       # forest_los left at the v2 DEFAULT (False): woodland hides the team
                       # without blinding it. The symmetric variant is the disclosed sensitivity
                       # row and its numbers live in the *_symforest artefacts, never mixed.
                       terrain=terrain_v2(hidden_leth=1.0, conceal_reach=b["cr"]),
                       range_scale=sc * b["rm"], spacing_km=2.0 * sc, standoff_km=4.0 * sc)
    print(f"[b{b['i']:02d} {b['map']} x{b['rm']} cr{b['cr']}] scale={sc * b['rm']:.2f} R={base.R} "
          f"H={base.H} lanes={len(base.lane_idx)} concealed={int(base.concealed.sum())} "
          f"build={time.time() - t0:.0f}s", flush=True)

    if check:                                    # the forgetful arm must match step 1a exactly
        from scratch.gen39_screen import ladder as ladder_1a
        pp = base.lethality(resample_field(base.coords, FIELDS[0]), hidden_leth=0.6)
        L = pick_laydown(base, pp, "open", 3, np.random.default_rng(FIELDS[0] * 131 + 3))
        g = ConcealDyn(base, pp, L, w=W, tau=TAU, **DOCTRINE)
        a, b_ = ladder_1a(g), ladder_forgetful(g, static_localopt_d(g)[0])
        assert set(a) == set(b_) and all(a[k] == b_[k] for k in a), "forgetful ladder drifted"
        print("[check] forgetful ladder identical to step 1a", flush=True)

    teams = (3,) if quick else TEAMS
    hleth = (0.4, 1.0) if quick else HIDDEN_LETH
    fields = FIELDS[:1] if quick else FIELDS
    tag = f"{b['map']}x{b['rm']}cr{b['cr']}"
    out = []
    for K in teams:
        for kind in KINDS:
            for hl in hleth:
                for seed in fields:
                    out.append(cell(base, seed, hl, K, kind, tag))
        json.dump(out, open(dst, "w"))
    json.dump(out, open(dst, "w"))
    print(f"[written] {dst} ({len(out)} cells, {sum(r['secs'] for r in out) / 60:.1f} min)",
          flush=True)
    return dst


def launch(sel, workers, outdir, quick, force):
    os.makedirs(outdir, exist_ok=True)
    todo = [b for b in sel if force or not os.path.exists(block_path(b, outdir))]
    print(f"[launch] {len(todo)}/{len(sel)} blocks, {workers} workers -> {outdir}", flush=True)
    env = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    running, queue, t0 = [], list(todo), time.time()
    while queue or running:
        while queue and len(running) < workers:
            b = queue.pop(0)
            log = open(f"{outdir}/b{b['i']:02d}.log", "w")
            cmd = [sys.executable, __file__, "--block", str(b["i"]), "--out-dir", outdir]
            if quick:
                cmd.append("--quick")
            if force:
                cmd.append("--force")
            running.append((b, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT),
                            log))
        time.sleep(2.0)
        for it in list(running):
            b, pr, log = it
            if pr.poll() is not None:
                log.close()
                running.remove(it)
                ok = "ok" if pr.returncode == 0 else f"FAILED rc={pr.returncode}"
                print(f"[done b{b['i']:02d} {b['map']} x{b['rm']} cr{b['cr']}] {ok}  "
                      f"({len(queue)} queued, {len(running)} running, "
                      f"{(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
    print(f"[launch] all blocks finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def merge(outdir, dst):
    out = []
    for b in blocks():
        p = block_path(b, outdir)
        if os.path.exists(p):
            out.extend(json.load(open(p)))
    json.dump(out, open(dst, "w"))
    print(f"[merged] {dst} ({len(out)} cells from {outdir})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--maps", default=",".join(MAPS))
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="assert the forgetful arm matches step 1a")
    ap.add_argument("--out-dir", default=OUTDIR)
    ap.add_argument("--out", default="models/runs/gen39_screen2.json")
    a = ap.parse_args()

    allb = blocks()
    sel = [b for b in allb if b["map"] in a.maps.split(",")]
    if a.list:
        for b in sel:
            done = "done" if os.path.exists(block_path(b, a.out_dir)) else "-"
            print(f"  b{b['i']:02d}  {b['map']:14s} x{b['rm']} cr{b['cr']}  {done}")
        return
    if a.merge:
        return merge(a.out_dir, a.out)
    if a.block is not None:
        os.makedirs(a.out_dir, exist_ok=True)
        return run_block(allb[a.block], a.out_dir, a.quick, a.force, a.check)
    if a.launch:
        return launch(sel, a.workers, a.out_dir, a.quick, a.force)
    ap.error("give one of --list / --block N / --launch / --merge")


if __name__ == "__main__":
    main()
