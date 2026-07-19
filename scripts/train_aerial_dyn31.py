#!/usr/bin/env python3
"""gen31: the gen27 conversion trainer (ledger experiments/gen31_aerial_dyn.md; Phase 0
pinned the operating point 2026-07-19).

Enemy = the ANTICIPATORY MIXED DOCTRINE: softmax(tau=0.10) over Z[h] = 0.7 * E[dmg | window
routes] + 0.3 * dmg[r_flee], where r_flee = the route a pattern-punished defender would
obviously run to (argmin damage vs the pure-repeat aim). Fleet-route N=3 mission damage;
40-sortie episodes; head columns per route: [exposure, window-frequency, DOCTRINE
(this-window expected damage, min-max normalised)] - the information channel v4.0 lacked.
Yardsticks per instance, ALL EXACT over the window chain: iid_eq, local static optimum, the
payoff-blind dynamic family, the doctrine-informed fitted rules (dodge / softdodge /
composed, disclosed caps), history_opt (lazy RVI). PRIMARY bar object = the STATIC CAP
min(iid_eq, static_opt) (the gen27 shape); blind-family beats tracked as the aiming row.
Protocol: dev-test = D2100-2102 (burned by the corridor hunt, iteration diagnostics only);
GATED = D4100-4105 (pristine; evaluated ONLY at confirmation via --eval-gated).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path as _P

import numpy as np
import torch

from scripts.train_multiconvoy import route_one
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.agents.transition_builder import SMDPTransition
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.aerial_curves import all_lane_sets, build_curve_menu, build_curved_game, dense_hazard_grid
from src.envs.aerial_interdiction_env import AerialInterdictionEnv
from src.envs.aerial_sector import SectorLattice
from scripts.train_aerial_generalist import random_field

BASE = SectorLattice(ny=9, nx=13)
DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(4, j) for j in range(9) if j < 5} | {(8, j) for j in range(9) if j > 3}))
N, W, TAU, S_EP, GAMMA = 3, 2, 0.10, 40, 0.95
Q_REP, Q_FLEE = 0.7, 0.3


def _mm(x):
    rng_ = x.max() - x.min()
    return (x - x.min()) / rng_ if rng_ > 0 else np.zeros_like(x)


class DynInstance:
    def __init__(self, name: str, lat: SectorLattice, r: float, pmax_seed: int):
        self.name = name
        menu, _ = build_curve_menu(lat, r, R=40, seed=0)
        centres = dense_hazard_grid(lat, step=0.5)
        pm = random_field(centres, pmax_seed)
        self.env = AerialInterdictionEnv(lat, menu, centres, K=1, r=r, p_max=pm, N=N,
                                         head_feats=("exposure",))
        game, S = self.env.game, self.env.S
        self.R = game.n_routes
        self.dmg = 1.0 - S ** N                                   # [R, H] mission damage
        # per-window enemy BR probs and per-(window, route) expected damage - EXACT
        wins = list(itertools.product(range(self.R), repeat=W))
        self.wins = wins
        self.widx = {w: i for i, w in enumerate(wins)}
        V = self.dmg[np.array([w for w in wins])].mean(axis=1)    # [n_win, H]
        Zr = np.exp((V - V.max(axis=1, keepdims=True)) / TAU)
        Ar = Zr / Zr.sum(axis=1, keepdims=True)                   # pure-repeat aim
        rflee = (Ar @ self.dmg.T).argmin(axis=1)                  # the obvious escape route
        Zd = Q_REP * V + Q_FLEE * self.dmg[rflee]                 # anticipatory doctrine
        Ze = np.exp((Zd - Zd.max(axis=1, keepdims=True)) / TAU)
        brp = Ze / Ze.sum(axis=1, keepdims=True)
        self.stepdmg = brp @ self.dmg.T                           # [n_win, R]
        self.exposure = _mm(1.0 - S.min(axis=1))
        # ---- yardstick rows (exact chains) ----
        sol = solve_multiconvoy(game, N, "mission")
        occs = list(itertools.combinations_with_replacement(range(self.R), N))
        d_eq = np.zeros(self.R)
        for i, o in enumerate(occs):
            if len(set(o)) == 1:
                d_eq[o[0]] += sol.defender_strategy[i]
        d_eq = d_eq / d_eq.sum() if d_eq.sum() > 0 else np.full(self.R, 1 / self.R)
        self.d_eq_store = d_eq
        self.iid_eq = self.static_value(d_eq)
        self.static_opt = self._static_local_opt(d_eq)
        self.naive_dyn = self._naive_dyn_rows(lat, menu, game, S)
        self.hist_opt = self._rvi()
        self.fitted = self._fitted_rows()
        self.cap = min(self.iid_eq, self.static_opt)              # the PRIMARY bar object
        self.best_blind = min(self.naive_dyn.values())            # the aiming row
        self.bar = self.cap
        self.window = tuple(np.random.default_rng(0).integers(self.R, size=W))
        self.pol_hist: list = []

    def feats(self, window) -> torch.Tensor:
        wf = np.zeros(self.R)
        for r in window:
            wf[r] += 1.0 / W
        doc = _mm(self.stepdmg[self.widx[tuple(window)]])          # the doctrine column
        return torch.tensor(np.stack([self.exposure, wf, doc], axis=1), dtype=torch.float32)

    def _fitted_rows(self) -> dict[str, float]:
        """Doctrine-informed rules (information parity; fitted = disclosed caps)."""
        rows = {}
        dodge = {w: None for w in self.wins}
        for i, w in enumerate(self.wins):
            d = np.zeros(self.R)
            d[int(self.stepdmg[i].argmin())] = 1.0
            dodge[w] = d
        rows["myopic_dodge"] = self.chain_value(lambda w: dodge[w])
        logeq = None
        d_eq_s = np.clip(self.d_eq_store, 1e-12, 1.0)
        logeq = np.log(d_eq_s)
        for tag, base in (("softdodge", 0.0), ("composed", 1.0)):
            bv = np.inf
            for beta in (0.5, 1, 2, 4, 8, 16, 32):
                mats = {}
                for i, w in enumerate(self.wins):
                    L = base * logeq - beta * self.stepdmg[i]
                    L = L - L.max()
                    m = np.exp(L)
                    mats[w] = m / m.sum()
                bv = min(bv, self.chain_value(lambda w: mats[w]))
            rows[f"{tag}_fit"] = bv
        return rows

    def static_value(self, d: np.ndarray) -> float:
        """Exact stationary damage of i.i.d. static play d (window dist = product measure)."""
        pw = np.array([d[w[0]] * d[w[1]] for w in self.wins])
        return float(pw @ (self.stepdmg @ d))

    def _static_local_opt(self, d0: np.ndarray, starts: int = 4, iters: int = 120) -> float:
        rng = np.random.default_rng(1)
        best = self.static_value(d0)
        for s in range(starts):
            d = d0.copy() if s == 0 else rng.dirichlet(np.ones(self.R))
            v = self.static_value(d)
            for _ in range(iters):
                g = np.zeros(self.R)
                for r in range(self.R):                       # finite-diff simplex descent
                    e = d * 0.98
                    e[r] += 0.02
                    e /= e.sum()
                    g[r] = self.static_value(e) - v
                r = int(np.argmin(g))
                if g[r] >= -1e-6:
                    break
                d = d * 0.9
                d[r] += 0.1
                d /= d.sum()
                v = self.static_value(d)
            best = min(best, v)
        return float(best)

    def chain_value(self, dist_fn) -> float:
        """Exact stationary damage of a window-conditioned rule dist_fn(window)->route dist.
        Vectorised: windows are (a, b); the successor of ((a, b), r) is (b, r), so one power
        iteration is nu[b, r] = sum_a mu[a, b] * P[a, b, r] (an einsum, ~64k ops)."""
        R = self.R
        P = np.stack([dist_fn(w) for w in self.wins]).reshape(R, R, R)
        D = (P.reshape(-1, R) * self.stepdmg).sum(axis=1)
        mu = np.full((R, R), 1.0 / (R * R))
        for _ in range(1200):
            # LAZY chain (same stationary distribution, aperiodic by construction): plain power
            # iteration oscillates on the periodic chains deterministic rotation rules induce.
            nu = 0.5 * mu + 0.5 * np.einsum("ab,abr->br", mu, P)
            if np.abs(nu - mu).max() < 1e-13:
                mu = nu
                break
            mu = nu
        return float(mu.reshape(-1) @ D)

    def _naive_dyn_rows(self, lat, menu, game, S) -> dict[str, float]:
        rows = {}
        fam = {}
        lsets = all_lane_sets(lat, menu)
        for rc, li in (lsets.items() if lsets else [(0.0, [])]):
            for k, dd in lane_stack_distributions(game, li, S).items():
                fam[f"{k}@{rc}"] = dd
        for name, dd in fam.items():
            sup = list(np.where(dd > 1e-9)[0])

            def anti(w, dd=dd):
                m = dd.copy()
                for r in set(w):
                    m[r] = 0.0
                return m / m.sum() if m.sum() > 1e-12 else dd
            rows[f"anti_{name}"] = self.chain_value(anti)
            if len(sup) > W:
                def rot(w, sup=tuple(sup)):
                    out = np.zeros(self.R)
                    cand = [r for r in sup if r not in w]
                    out[cand[0] if cand else sup[0]] = 1.0
                    return out
                rows[f"rot_{name}"] = self.chain_value(rot)
        return rows

    def _rvi(self) -> float:
        n = len(self.wins)
        Vv = np.zeros(n)
        nxt = np.zeros((n, self.R), dtype=int)
        for i, w in enumerate(self.wins):
            for r in range(self.R):
                nxt[i, r] = self.widx[(w[1], r)]
        g = 0.0
        for _ in range(4000):
            Q = self.stepdmg + Vv[nxt]
            Vn = Q.min(axis=1)
            g = float(Vn.mean())
            # aperiodicity (lazy) transform: near-periodic optimal policies make plain RVI
            # oscillate above the true gain (caught by the rotation-beats-optimum test)
            Vd = 0.5 * Vv + 0.5 * (Vn - g)
            if np.abs(Vd - Vv).max() < 1e-11:
                Vv = Vd
                break
            Vv = Vd
        # the gain of the lazy iterate halves the step; report via one plain Bellman pass
        Q = self.stepdmg + Vv[nxt]
        return float((Q.min(axis=1) - Vv).mean())


def make_pool(eval_gated=False):
    train = [DynInstance(f"layoutB{1000+s}", BASE, 1.6, 1000 + s) for s in range(6)]
    train += [DynInstance(f"layoutD{1100+s}", DBL, 1.2, 1100 + s) for s in range(12)]
    val = [DynInstance(f"valD{3000+s}", DBL, 1.2, 3000 + s) for s in range(2)]
    val += [DynInstance(f"valB{3100+s}", BASE, 1.6, 3100 + s) for s in range(2)]
    # dev-test: burned by the Phase-0 corridor hunt; iteration diagnostics ONLY
    test = [DynInstance(f"devtestD{2100+s}", DBL, 1.2, 2100 + s) for s in range(3)]
    if eval_gated:  # the pristine gated set: confirmation runs only (the iteration protocol)
        test = [DynInstance(f"gatedD{4100+s}", DBL, 1.2, 4100 + s) for s in range(6)]
    ctx = [DynInstance(f"ctxB{4000+s}", BASE, 1.6, 4000 + s) for s in range(2)]
    return train, val, test, ctx


def policy_chain_value(prot, inst: DynInstance) -> float:
    """EXACT stationary damage of the current policy vs the adaptive enemy: encoder once,
    head per window (route_feats vary with the window), power-iterate the induced chain."""
    env = inst.env
    env.reset()
    obs = env.observe()
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    active = n2i[obs["trucks"][0]["current_node"]]
    prot.actor.menu_routes = obs["menu_route_node_idx"]
    prot.actor.eval()
    dists = np.zeros((len(inst.wins), inst.R))
    with torch.no_grad():
        h = prot.actor.encoder(pyg.x, pyg.edge_index, pyg.edge_attr)
        for i, w in enumerate(inst.wins):
            prot.actor.route_feats = inst.feats(w)
            probs, _ = prot.actor.head(h, active, list(range(inst.R)), torch.zeros(inst.R))
            dists[i] = probs.cpu().numpy()
    prot.actor.train()
    return inst.chain_value(lambda w: dists[inst.widx[w]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sorties", type=int, default=16000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--ent-frac", type=float, default=0.5)
    p.add_argument("--alpha-floor", type=float, default=0.20)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--json-out", default="")
    p.add_argument("--ckpt-dir", default="")
    p.add_argument("--eval-gated", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("[gen28d] building pool (exact chains per instance)...", flush=True)
    t0 = time.time()
    train, val, test, ctx = make_pool(eval_gated=args.eval_gated)
    print(f"[gen28d] pool in {time.time()-t0:.0f}s", flush=True)
    for it in val + test + ctx:
        print(f"    {it.name}: iid_eq={it.iid_eq:.3f} static_opt={it.static_opt:.3f} "
              f"CAP={it.cap:.3f} blind={it.best_blind:.3f} "
              f"dodge={it.fitted['myopic_dodge']:.3f} fit={min(it.fitted.values()):.3f} "
              f"hist_opt={it.hist_opt:.3f}", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3,
                          alpha_floor=args.alpha_floor)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))   # [exposure, winfreq, doctrine]
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w]})
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w],
                                          "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.follow_w, prot.q2.follow_w]})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.route_feat_w, prot.q2.route_feat_w],
                                           "lr": args.head_term_lr})

    if args.sorties == 0:          # refs-only probe + the untrained context row
        rows = {it.name: policy_chain_value(prot, it) for it in val + test + ctx}
        beats0 = sum(1 for i in test if rows[i.name] < i.cap)
        print(f"[gen31] UNTRAINED: test beats-CAP {beats0}/{len(test)}; "
              f"ratios-to-iid " + " ".join(f"{rows[i.name]/i.iid_eq:.2f}" for i in test))
        return

    hist = []
    sortie = 0
    t0 = time.time()
    while sortie < args.sorties:
        inst = train[int(rng.integers(len(train)))]
        env = inst.env
        window = tuple(rng.integers(inst.R, size=W))
        ep_steps = []                                  # per-sortie: (steps, reward)
        for s_i in range(S_EP):
            env.reset()
            obs0_feats = inst.feats(window)
            steps, occ, routes = [], None, None
            # fleet-route: leader decides; walk the env's 3 pushes with the window feats
            for _ in range(env.config.N):
                ci = env.current_convoy()
                obs = env.observe()
                obs["menu_route_feats"] = obs0_feats
                mask = env.defender_action_mask()
                act = (prot.select_action(obs, mask)[ci] if ci == 0 else leader_act)
                if ci == 0:
                    leader_act = act
                env.route_convoy_by_index(int(act))
                steps.append((obs, ci, act, mask))
            r_lead = int(leader_act)
            dmgv = float(inst.stepdmg[inst.widx[window], r_lead])
            ep_steps.append((steps, -args.interception_loss * dmgv))
            window = tuple(list(window[1:]) + [r_lead])
            sortie += 1
        # push the episode: within-sortie dt=0, across-sortie dt=1, done at the end
        flat = []
        for si, (steps, rew) in enumerate(ep_steps):
            for j, (obs, ci, act, mask) in enumerate(steps):
                last_push = j == len(steps) - 1
                flat.append((obs, ci, act, mask,
                             rew if last_push else 0.0,
                             1 if last_push else 0,
                             si == len(ep_steps) - 1 and last_push))
        for idx, (obs, ci, act, mask, rew, dt, done) in enumerate(flat):
            obs["target_entropy"] = (0.05 if ci != 0 and sortie > 250 else args.ent_frac) \
                * math.log(inst.R)
            obs["alpha_group"] = 1 if (ci != 0 and sortie > 250) else 0
            if done or idx == len(flat) - 1:
                nstate = {}
            else:
                nobs, nci, _, nmask = flat[idx + 1][0], flat[idx + 1][1], None, flat[idx + 1][3]
                nstate = dict(nobs)
                nstate["active_truck"] = nci
                nstate["allowed_destinations"] = {"protagonist": {nci: list(nmask[nci])}}
            prot.replay_buffer.push(SMDPTransition(
                agent="protagonist", state=obs, action={ci: act}, reward=rew,
                next_state=nstate, done=bool(done), elapsed_ticks=dt,
                action_mask={"protagonist": mask}, info={}))
        for _ in range(S_EP):
            prot.update(args.batch_size)

        if sortie % args.eval_every < S_EP:
            rows = {}
            for group in (train[:4], val, test, ctx):
                for it in group:
                    rows[it.name] = policy_chain_value(prot, it)
            va_m = float(np.mean([rows[i.name] / i.iid_eq for i in val]))
            te_beats = sum(1 for i in test if rows[i.name] < i.cap)
            te_blind = sum(1 for i in test if rows[i.name] < i.best_blind)
            te_m = float(np.mean([rows[i.name] / i.iid_eq for i in test]))
            fw = tuple(float(x) for x in prot.actor.route_feat_w.detach())
            hist.append((sortie, va_m, te_m, te_beats, te_blind, rows, fw, float(prot.alpha)))
            if args.ckpt_dir:
                _P(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(),
                           str(_P(args.ckpt_dir) / f"actor_ep{sortie}.pt"))
            print(f"  sortie {sortie:6d}: VAL {va_m:.2f} | TEST beats-CAP {te_beats}/"
                  f"{len(test)} beats-BLIND {te_blind}/{len(test)} ratio {te_m:.2f} | "
                  f"rw[{fw[0]:.2f},{fw[1]:.2f},{fw[2]:.2f}] a{prot.alpha:.2f} | "
                  f"{time.time()-t0:5.0f}s", flush=True)

    if args.json_out:
        refs = {it.name: {"iid_eq": it.iid_eq, "static_opt": it.static_opt,
                          "best_naive_dyn": it.best_blind, "cap": it.cap,
                          "fitted": it.fitted, "hist_opt": it.hist_opt}
                for it in train + val + test + ctx}
        _P(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _P(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "refs": refs, "history": hist}, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
