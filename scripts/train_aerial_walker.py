#!/usr/bin/env python3
"""gen28 v2.3: the WALKER generalist (the pre-registered structural re-aim; ledger
experiments/gen28_aerial.md v2.3 record).

Same pool, fields and bar structure as v2.2, but the policy is a next-waypoint walker in NODE
mode (12 position-conditioned decisions per sortie: replay-state diversity is structural; no
menu head, no route-level head terms to railroad). All reference rows (equilibrium, complete
lane-set family, menu stacks) are recomputed in the v2.3 geometry class (waypoint-leg
polylines, line-integral exposure), so every arm is same-geometry. Exploitability of the
walker is EXACT (backward DP over the legal DAG; no Monte Carlo).

Refs-only probe (prints anchors for the ledger pin + the untrained context row):
  --sorties 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path as _P

import numpy as np
import torch

from scripts.train_multiconvoy import _transition
from src.agents.sac import ProtagonistSAC
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.aerial_curves import dense_hazard_grid
from src.envs.aerial_sector import SectorLattice, banded_pmax
from src.envs.aerial_walker import (AerialWalkerEnv, build_dag_menu, build_polyline_game,
                                    path_survival, walker_exploitability)

BASE = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(6, j) for j in range(9) if j not in (3, 4, 5)}))
DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(4, j) for j in range(9) if j < 5} | {(8, j) for j in range(9) if j > 3}))

CELLS = [
    ("dblpinch_banded_K1_r1.2", DBL,  1, 1.2, "banded"),
    ("pinch_banded_K1_r1.6",   PINCH, 1, 1.6, "banded"),
    ("banded_K1_r1.6",         BASE,  1, 1.6, "banded"),
    ("base_K1_r0.8",           BASE,  1, 0.8, 0.9),
    ("base_K1_r1.2",           BASE,  1, 1.2, 0.9),
    ("base_K2_r1.2",           BASE,  2, 1.2, 0.9),
]


def random_field(centres: np.ndarray, seed: int, length_scale: float = 2.5,
                 band=(0.30, 0.95)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d2 = ((centres[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2.0 * length_scale ** 2)) + 1e-8 * np.eye(len(centres))
    g = rng.multivariate_normal(np.zeros(len(centres)), cov)
    ranks = np.argsort(np.argsort(g))
    lo, hi = band
    return lo + (hi - lo) * ranks / (len(centres) - 1)


class WalkerInstance:
    """v2.3 instance: walker env + same-geometry reference rows (polyline menu game)."""

    def __init__(self, name: str, lat: SectorLattice, K: int, r, pmax, menu_r: float = 1.6):
        self.name = name
        centres = dense_hazard_grid(lat, step=0.5)
        pm = pmax if not isinstance(pmax, str) else banded_pmax(centres, lat.ny)
        self.env = AerialWalkerEnv(lat, centres, K=K, r=r, p_max=pm)
        node_paths, lane_sets = build_dag_menu(self.env, R=40, seed=0)
        self.game = build_polyline_game(lat, node_paths, centres, K,
                                        self.env.arc_index, self.env.A)
        sol = solve(self.game)
        self.eq = float(sol.value)
        self.loss_det = float(sol.loss_det)
        S1 = np.stack([path_survival(p, self.env.arc_index, self.env.A) for p in node_paths])
        self.naive: dict[str, float] = {}
        for rc, li in (lane_sets.items() if lane_sets else [(0.0, [])]):
            for k, d in lane_stack_distributions(self.game, li, S1).items():
                self.naive[f"{k}@{rc}"] = float(best_response_attacker(self.game, d)[1])
        self.best_naive = min(self.naive.values())
        self.window: deque = deque(maxlen=250)      # realised per-hazard survival vectors
        self.win_sum = np.zeros(len(centres))


def make_layout_instance(name: str, seed: int, fam: str) -> WalkerInstance:
    lat = BASE if fam == "base" else DBL
    centres = dense_hazard_grid(lat, step=0.5)
    return WalkerInstance(name, lat, K=1, r=(1.6 if fam == "base" else 1.2),
                          pmax=random_field(centres, seed))


def sample_iset(inst: WalkerInstance, rng, tau: float) -> tuple[int, ...]:
    """Smooth-FP interdictor: softmax(tau) over expected interception vs the trailing window
    of realised flights (uniform when the window is empty); K=2 samples sequentially."""
    H = len(inst.env.centres)
    if not inst.window:
        return tuple(rng.choice(H, size=inst.env.K, replace=False))
    mean_s = inst.win_sum / len(inst.window)
    v1 = 1.0 - mean_s
    z = np.exp((v1 - v1.max()) / tau)
    h1 = int(rng.choice(H, p=z / z.sum()))
    if inst.env.K == 1:
        return (h1,)
    prod = np.stack(list(inst.window)) * np.stack(list(inst.window))[:, h1:h1 + 1]
    v2 = 1.0 - prod.mean(axis=0)
    v2[h1] = -1e9
    z2 = np.exp((v2 - v2.max()) / tau)
    h2 = int(rng.choice(H, p=z2 / z2.sum()))
    return tuple(sorted((h1, h2)))


def eval_all(prot, insts) -> dict[str, float]:
    out = {}
    for it in insts:
        val, _ = walker_exploitability(prot, it.env, isets=list(it.game.interdiction_sets))
        out[it.name] = val
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-layouts-train", type=int, default=18)
    p.add_argument("--n-layouts-test", type=int, default=6)
    p.add_argument("--sorties", type=int, default=12000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp-tau", type=float, default=0.05)
    p.add_argument("--ent-frac", type=float, default=0.5)
    p.add_argument("--alpha-floor", type=float, default=0.20)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--json-out", default="")
    p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("[gen28w] building pool...", flush=True)
    t0 = time.time()
    nb = args.n_layouts_train // 2
    train = [make_layout_instance(f"layoutB{1000 + s}", 1000 + s, "base") for s in range(nb)]
    train += [make_layout_instance(f"layoutD{1100 + s}", 1100 + s, "dbl")
              for s in range(args.n_layouts_train - nb)]
    train += [WalkerInstance(n, lat, K, r, pm) for n, lat, K, r, pm in CELLS]
    nt = args.n_layouts_test // 2
    test = [make_layout_instance(f"holdoutB{2000 + s}", 2000 + s, "base") for s in range(nt)]
    test += [make_layout_instance(f"holdoutD{2100 + s}", 2100 + s, "dbl")
             for s in range(args.n_layouts_test - nt)]
    print(f"[gen28w] pool built in {time.time() - t0:.1f}s", flush=True)
    for it in train + test:
        print(f"    {it.name}: eq={it.eq:.3f} best_naive={it.best_naive:.3f} "
              f"det={it.loss_det:.3f}", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3,
                          alpha_floor=args.alpha_floor)

    if args.sorties == 0:      # refs-only probe: anchors + the untrained context row
        tr0, te0 = eval_all(prot, train), eval_all(prot, test)
        trm0 = float(np.mean([tr0[i.name] / i.eq for i in train]))
        tem0 = float(np.mean([te0[i.name] / i.eq for i in test]))
        beats0 = sum(1 for i in test if te0[i.name] < i.best_naive)
        print(f"[gen28w] UNTRAINED context: train ratio {trm0:.2f} | held-out {tem0:.2f} "
              f"beats-best-naive {beats0}/6 | headline {tr0['dblpinch_banded_K1_r1.2']:.3f}")
        return

    hist = []
    t0 = time.time()
    for k in range(args.sorties):
        inst = train[int(rng.integers(len(train)))]
        env = inst.env
        iset = sample_iset(inst, rng, args.fp_tau)
        env.reset()
        steps = []
        done = False
        while not done:
            obs = env.observe()
            mask = env.action_mask()
            act = prot.select_action(obs, mask)
            steps.append((obs, act[0], mask))
            done = env.step(act[0])
        s_path = env.realised_survival()
        if len(inst.window) == inst.window.maxlen:
            inst.win_sum -= inst.window[0]
        inst.window.append(s_path)
        inst.win_sum += s_path
        p_int = 1.0 - float(np.prod([s_path[h] for h in iset]))
        reward = -args.interception_loss * p_int
        for i, (obs, a, mask) in enumerate(steps):
            last = i == len(steps) - 1
            obs["target_entropy"] = args.ent_frac * math.log(max(2, len(mask[0])))
            obs["alpha_group"] = 0
            nobs, nmask = (None, None) if last else (steps[i + 1][0], steps[i + 1][2])
            prot.replay_buffer.push(_transition(obs, 0, a, mask, reward if last else 0.0,
                                                nobs, 0, nmask, last))
        prot.update(args.batch_size)

        if (k + 1) % args.eval_every == 0:
            tr = eval_all(prot, train)
            te = eval_all(prot, test)
            tr_m = float(np.mean([tr[i.name] / i.eq for i in train]))
            te_m = float(np.mean([te[i.name] / i.eq for i in test]))
            te_beats = sum(1 for i in test if te[i.name] < i.best_naive)
            hist.append((k + 1, tr_m, te_m, te_beats, tr, te, float(prot.alpha)))
            if args.ckpt_dir:
                _P(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(_P(args.ckpt_dir) / f"actor_ep{k+1}.pt"))
            print(f"  sortie {k+1:6d}: TRAIN ratio {tr_m:.2f} | HELD-OUT {te_m:.2f} "
                  f"beats-BEST-naive {te_beats}/{len(test)} | headline "
                  f"{tr['dblpinch_banded_K1_r1.2']:.3f} | a{prot.alpha:.2f} | "
                  f"{time.time() - t0:5.0f}s", flush=True)

    if args.json_out:
        refs = {it.name: {"eq": it.eq, "loss_det": it.loss_det,
                          "best_naive": it.best_naive, **it.naive}
                for it in train + test}
        _P(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _P(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "refs": refs, "history": hist,
             "cells": [c[0] for c in CELLS]}, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
