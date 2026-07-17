#!/usr/bin/env python3
"""gen28: the aerial LAYOUT-GENERALIST (one trainer carries A1/A2/A3; ledger
experiments/gen28_aerial.md, bars pre-registered 2026-07-17 BEFORE this file existed).

ONE menu-select policy trained across aerial instances (random hidden-hazard effectiveness
LAYOUTS on the base sector + the screened A1/A2 cells: pinch/banded/base lattices, K/r cells),
conditioned on the instance through (a) the edge-vulnerability observation column (the layout's
per-arc threat projection) and (b) per-route transferable head features [cost, layout exposure]
(`route_feat_w`, dedicated lr; the gen11b/gen15 mechanism). NO route identity capacity.

Adversary: per-instance SMOOTH fictitious play (each instance keeps its own trailing play
window; softmax BR sampled fresh every sortie), exactly the proven gen15/16 recipe. Reward:
analytic -interception_loss * interception probability of the flown route under the committed
hazard set (N=1 mission == interception).

Evaluation: EXACT leader route distribution per instance (one forward pass) under that
instance's oracle BR, ratio to ITS equilibrium; TAP over the last 3 evals; select-on-train.
Zero-shot rows = the 6 held-out layouts (seeds 2000-2005), scored beside their pre-registered
naive rows (inv-risk-lane = the A3 comparator).

Run (per seed): PYTHONPATH=. python scripts/train_aerial_generalist.py --sorties 12000 --seed 0
Timing probe (the B9 gate, not a training run): --sorties 30 --eval-every 10
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path as _P

import numpy as np
import torch

from scripts.train_multiconvoy import _transition, route_one
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.fp_dynamics import sample_smooth_iset, smooth_fp_probs
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.baselines.multiconvoy_oracle import objective_value
from src.envs.aerial_curves import all_lane_sets, build_curve_menu, dense_hazard_grid
from src.envs.aerial_interdiction_env import AerialInterdictionEnv
from src.envs.aerial_sector import SectorLattice, banded_pmax

TAP_K = 3
BASE = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(6, j) for j in range(9) if j not in (3, 4, 5)}))
# staggered double pinch (v2.2): wall at x=4 open top, wall at x=8 open bottom -> forced S-turn;
# NO lane curve exists here (the lane rule structurally dies; naive rows = full-menu stacks).
DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(4, j) for j in range(9) if j < 5} | {(8, j) for j in range(9) if j > 3}))


def random_field(centres: np.ndarray, seed: int, length_scale: float = 2.5,
                 band=(0.30, 0.95)) -> np.ndarray:
    """The A3 layout sampler (identical to scratch/aerial_layout_probe.py: RBF Gaussian draw,
    rank-mapped into the band; independent of lattice geometry by construction)."""
    rng = np.random.default_rng(seed)
    d2 = ((centres[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2.0 * length_scale ** 2)) + 1e-8 * np.eye(len(centres))
    g = rng.multivariate_normal(np.zeros(len(centres)), cov)
    ranks = np.argsort(np.argsort(g))
    lo, hi = band
    return lo + (hi - lo) * ranks / (len(centres) - 1)


class AerialInstance:
    """Game v2.2 (2026-07-17): curved menu carrying ALL canonical lane spacings (the complete
    naive set, min over spacings = best_naive), dense 0.5-step hazard grid with standoff zones,
    line-integral exposure. pmax: scalar | "banded" | per-position array (layout); r scalar or
    per-position array (mixed threat types)."""
    def __init__(self, name: str, lat: SectorLattice, K: int, r, pmax, menu_r: float = 1.6):
        self.name = name
        menu, _ = build_curve_menu(lat, menu_r, R=40, seed=0)
        centres = dense_hazard_grid(lat, step=0.5)
        pm = pmax if not isinstance(pmax, str) else banded_pmax(centres, lat.ny)
        self.env = AerialInterdictionEnv(lat, menu, centres, K=K, r=r, p_max=pm)
        sol = solve(self.env.game)
        self.eq = float(sol.value)
        self.loss_det = float(sol.loss_det)
        self.naive: dict[str, float] = {}
        lsets = all_lane_sets(lat, menu)
        for rc, li in (lsets.items() if lsets else [(0.0, [])]):
            for k, d in lane_stack_distributions(self.env.game, li, self.env.S).items():
                self.naive[f"{k}@{rc}"] = float(best_response_attacker(self.env.game, d)[1])
        self.best_naive = min(self.naive.values())
        self.occ_seq: list[int] = []
        self.pol_hist: list[np.ndarray] = []

    @property
    def R(self):
        return self.env.game.n_routes


def make_layout_instance(name: str, seed: int, fam: str = "base") -> AerialInstance:
    """A3 layout families (v2.2 family probes, COMPLETE naive set): "base" = open sector r=1.6
    (best-naive 1.39 med / 1.29 min); "dbl" = staggered double pinch r=1.2 (1.55 med / 1.39
    min; no lane rule exists). One policy is trained and held out ACROSS BOTH."""
    lat = BASE if fam == "base" else DBL
    centres = dense_hazard_grid(lat, step=0.5)
    return AerialInstance(name, lat, K=1, r=(1.6 if fam == "base" else 1.2),
                          pmax=random_field(centres, seed))


# the screened A1/A2 cells (GAME V2.2: standoff + complete naive set, 2026-07-17): headline
# first; exact anchors pinned in the ledger from the pool-build printout BEFORE any training.
CELLS = [
    ("dblpinch_banded_K1_r1.2", DBL,  1, 1.2, "banded"),  # A1 headline (structure kills lanes)
    ("pinch_banded_K1_r1.6",   PINCH, 1, 1.6, "banded"),  # single-pinch point
    ("banded_K1_r1.6",         BASE,  1, 1.6, "banded"),  # open-sector banded point
    ("base_K1_r0.8",           BASE,  1, 0.8, 0.9),       # low-phi honest point
    ("base_K1_r1.2",           BASE,  1, 1.2, 0.9),       # low-gap honest point
    ("base_K2_r1.2",           BASE,  2, 1.2, 0.9),       # K axis honest point
]


def exact_ratio(prot: ProtagonistSAC, inst: AerialInstance) -> tuple[float, np.ndarray]:
    env = inst.env
    env.reset()
    obs = env.observe()
    prot.actor.menu_routes = obs["menu_route_node_idx"]
    if hasattr(prot.actor, "route_feat_w"):
        prot.actor.route_feats = obs["menu_route_feats"]
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    R = inst.R
    prot.actor.eval()
    with torch.no_grad():
        probs, _ = prot.actor(pyg, n2i[obs["trucks"][0]["current_node"]],
                              list(range(R)), torch.zeros(R))
    prot.actor.train()
    d = probs.numpy()                    # N=1: route distribution IS the strategy
    _, expl = best_response_attacker(inst.env.game, d)
    return float(expl) / inst.eq, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-layouts-train", type=int, default=18)
    p.add_argument("--n-layouts-test", type=int, default=6)
    p.add_argument("--sorties", type=int, default=12000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp-tau", type=float, default=0.05)
    p.add_argument("--smooth-window", type=int, default=250)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--ent-frac", type=float, default=0.5)
    p.add_argument("--alpha-floor", type=float, default=0.20)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--json-out", default="")
    p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    try:  # cap the INTER-op pool too (the 2026-07-16 system-time lesson: cap ALL pools)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("[gen28] building pool (layouts fixed across seeds: 1000+, test 2000+)...", flush=True)
    t0 = time.time()
    nb = args.n_layouts_train // 2
    train = [make_layout_instance(f"layoutB{1000 + s}", 1000 + s, "base") for s in range(nb)]
    train += [make_layout_instance(f"layoutD{1100 + s}", 1100 + s, "dbl")
              for s in range(args.n_layouts_train - nb)]
    train += [AerialInstance(n, lat, K, r, pm) for n, lat, K, r, pm in CELLS]
    nt = args.n_layouts_test // 2
    test = [make_layout_instance(f"holdoutB{2000 + s}", 2000 + s, "base") for s in range(nt)]
    test += [make_layout_instance(f"holdoutD{2100 + s}", 2100 + s, "dbl")
             for s in range(args.n_layouts_test - nt)]
    print(f"[gen28] pool built in {time.time() - t0:.1f}s: {len(train)} train "
          f"({args.n_layouts_train} layouts + {len(CELLS)} cells), {len(test)} held-out layouts")
    for it in train + test:
        print(f"    {it.name}: eq={it.eq:.3f} best_naive={it.best_naive:.3f} "
              f"det={it.loss_det:.3f}", flush=True)

    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3,
                          alpha_floor=args.alpha_floor)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(2))
        net.route_feats = None
    prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w]})
    prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w],
                                          "lr": args.head_term_lr})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.follow_w, prot.q2.follow_w]})
    prot.critic_optimizer.add_param_group({"params": [prot.q1.route_feat_w, prot.q2.route_feat_w],
                                           "lr": args.head_term_lr})

    hist = []
    t0 = time.time()
    for k in range(args.sorties):
        inst = train[int(rng.integers(len(train)))]
        env = inst.env
        probs = smooth_fp_probs(inst.occ_seq, len(env.occupancies), env.obj_matrix,
                                args.fp_tau, args.smooth_window)
        j = sample_smooth_iset(probs, rng)
        env.reset()
        env.commit(j)
        steps, occ, _ = route_one(prot, env, fleet_route=True)
        inst.occ_seq.append(env._occ_index[tuple(occ)])
        pay = env.game.payoff[:, j]
        reward = -args.interception_loss * objective_value(
            np.asarray(occ), pay, env.config.N, env.config.objective, env.config.threshold_m)
        te = args.ent_frac * math.log(inst.R)
        for obs_j, _, _, _ in steps:
            obs_j["target_entropy"] = te
            obs_j["alpha_group"] = 0
        N = env.config.N
        for i, (obs, ci, hop, mask) in enumerate(steps):
            last = i == N - 1
            nobs, nci, nmask = ((steps[i + 1][0], steps[i + 1][1], steps[i + 1][3])
                                if not last else (None, None, None))
            prot.replay_buffer.push(_transition(obs, ci, hop, mask, reward if last else 0.0,
                                                nobs, nci, nmask, last))
        prot.update(args.batch_size)

        if (k + 1) % args.eval_every == 0:
            for inst_set in (train, test):
                for it in inst_set:
                    _, d = exact_ratio(prot, it)
                    it.pol_hist.append(d)

            def tap_rows(insts):
                rows = {}
                for it in insts:
                    tap = np.mean(it.pol_hist[-TAP_K:], axis=0)
                    _, expl = best_response_attacker(it.env.game, tap)
                    rows[it.name] = float(expl)
                return rows

            tr = tap_rows(train); te_rows = tap_rows(test)
            tr_m = float(np.mean([tr[i.name] / i.eq for i in train]))
            te_m = float(np.mean([te_rows[i.name] / i.eq for i in test]))
            te_beats = sum(1 for i in test if te_rows[i.name] < i.best_naive)
            fw = tuple(float(x) for x in prot.actor.route_feat_w.detach())
            hist.append((k + 1, tr_m, te_m, te_beats, tr, te_rows, fw, float(prot.alpha)))
            if args.ckpt_dir:
                _P(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(), str(_P(args.ckpt_dir) / f"actor_ep{k+1}.pt"))
            print(f"  sortie {k+1:6d}: TRAIN ratio {tr_m:.2f} | HELD-OUT ratio {te_m:.2f} "
                  f"beats-BEST-naive {te_beats}/{len(test)} | headline "
                  f"{tr['dblpinch_banded_K1_r1.2']:.3f} | "
                  f"rw[{fw[0]:.2f},{fw[1]:.2f}] a{prot.alpha:.2f} | {time.time()-t0:5.0f}s",
                  flush=True)

    if args.json_out:
        refs = {it.name: {"eq": it.eq, "loss_det": it.loss_det, **it.naive}
                for it in train + test}
        _P(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _P(args.json_out).write_text(json.dumps(
            {"seed": args.seed, "refs": refs, "history": hist,
             "cells": [c[0] for c in CELLS]}, indent=2))
        print(f"  [written] {args.json_out}")


if __name__ == "__main__":
    main()
