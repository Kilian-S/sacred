#!/usr/bin/env python3
"""A2 + A3 (EVAL-ONLY): shuffled-map transfer row + intel-noise robustness curve.

A2: does the frozen gen16 generalist TRACK equilibria of threat maps decorrelated from geometry
    (candidate-edge value permutations = new realities), or was it reading geometry?
A3: reality fixed (true game scores everything); only the OBSERVED map is corrupted
    (shuffle-fraction / multiplicative noise): does the hedge survive intel error?

Pre-registration: experiments/zst_map_robustness.md (binding). No training; exact arithmetic.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import numpy as np
import torch

from scripts.train_generalist import sample_instances
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.baselines.interdiction_oracle import build_interdiction_game, survival_intercept_fn
from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, objective_matrix, solve_multiconvoy)

N, K, KX, BAND = 3, 1, 8, (0.15, 0.95)


def _mkprot(state=None):
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu", role_alpha=True)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2))
    prot.actor.route_feats = None
    if state is not None:
        prot.actor.load_state_dict(state)
    return prot


def _mm(x):
    rng_ = x.max() - x.min()
    return (x - x.min()) / rng_ if rng_ > 0 else np.zeros_like(x)


def cand_edges(inst):
    # canonical, PROCESS-INDEPENDENT order (sorting frozensets by repr depends on the per-process
    # string-hash seed, which made the seeded shuffles non-reproducible across invocations)
    return sorted(set().union(*inst.env.game.route_edges), key=lambda e: tuple(sorted(map(str, e))))


def true_cand_map(inst):
    """Candidate-edge vulnerabilities from the env's observable map, keyed by frozenset."""
    emap = inst.env.edge_vulnerability

    def key(e):
        u, v = tuple(e)
        return emap.get((u, v), emap.get((v, u), emap.get(tuple(sorted((u, v), key=repr)))))
    return {e: key(e) for e in cand_edges(inst)}


def policy_dist(states, inst, obs_map: dict, worst_per_route: np.ndarray) -> np.ndarray:
    """Exact stacked occupancy distribution of the TAP-of-checkpoints policy, with the OBSERVED
    edge-vulnerability map + per-route features overridden. obs_map keys: frozenset cand edges
    (non-candidate edges keep the env's true values)."""
    env = inst.env
    env.reset()
    obs = dict(env.observe())
    # observed edge map: env convention = tuple(sorted(e, key=repr)) keys over the full graph
    full = dict(env.edge_vulnerability)
    for e, p in obs_map.items():
        full[tuple(sorted(tuple(e), key=repr))] = float(p)
    obs["edge_vulnerability"] = full
    cost = np.asarray(env.game.travel_cost, dtype=float)
    feats = torch.tensor(np.stack([_mm(cost), _mm(worst_per_route)], axis=1), dtype=torch.float32)
    obs["menu_route_feats"] = feats
    R = env.game.n_routes
    ds = []
    for st in states:
        prot = _mkprot(st)
        prot.actor.menu_routes = obs["menu_route_node_idx"]
        prot.actor.route_feats = feats
        pyg = featurize_state(obs, 0)
        pyg.x = _clip_x(pyg.x, prot.node_in_dim)
        pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
        n2i = node_index_map(obs)
        prot.actor.eval()
        with torch.no_grad():
            lead, _ = prot.actor(pyg, n2i[obs["trucks"][0]["current_node"]],
                                 list(range(R)), torch.zeros(R))
        lead = lead.numpy()
        d = np.zeros(len(env.occupancies))
        for r in range(R):
            d[env._occ_index[tuple(N if i == r else 0 for i in range(R))]] = lead[r]
        ds.append(d)
    return np.mean(ds, axis=0)


def route_worst(inst, cmap: dict) -> np.ndarray:
    """Per-route worst vulnerability under an arbitrary candidate-edge map."""
    return np.array([max(cmap[e] for e in redges) for redges in inst.env.game.route_edges])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--json-out", default="models/runs/zst_map_robustness.json")
    ap.add_argument("--shuffles", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(4)

    ck_dir = a.actor.rsplit("/", 1)[0]
    cks = {int(re.search(r"ep(\d+)", c).group(1)): c for c in glob.glob(f"{ck_dir}/actor_ep*.pt")}
    best_at = int(re.search(r"ep(\d+)", a.actor).group(1))
    eps = sorted(cks)
    ci = eps.index(best_at)
    window = eps[max(0, ci - 1):ci + 2]
    states = [torch.load(cks[e], map_location="cpu") for e in window]
    rnd = [_mkprot().actor.state_dict()]
    print(f"[map-robustness] TAP window {window}; 6 held-out Gdansk ODs", flush=True)

    insts = sample_instances(6, N, K, BAND, KX, 0, city="gdansk")
    out = {"window": window, "A2": {}, "A3": {}}

    # ---------- sanity row (true map, true game) ----------
    def score(inst, dist, M, eq):
        _, expl = best_response_attacker_multi(M, dist)
        return float(expl) / eq
    sane_g, sane_r = [], []
    for it in insts:
        cmap = true_cand_map(it)
        wv = route_worst(it, cmap)
        sane_g.append(score(it, policy_dist(states, it, cmap, wv), it.env.obj_matrix, it.eq))
        sane_r.append(score(it, policy_dist(rnd, it, cmap, wv), it.env.obj_matrix, it.eq))
    out["sanity"] = {"gen": float(np.mean(sane_g)), "rand": float(np.mean(sane_r)),
                     "per_od": [round(x, 2) for x in sane_g]}
    print(f"SANITY true map: gen {np.mean(sane_g):.2f}x vs rand {np.mean(sane_r):.2f}x | "
          f"{[round(x,2) for x in sane_g]}", flush=True)

    # ---------- A2: shuffled REALITY ----------
    g_all, r_all, det_beats = [], [], 0
    cells = 0
    for it in insts:
        ce = cand_edges(it)
        cmap = true_cand_map(it)
        vals = [cmap[e] for e in ce]
        for s in range(a.shuffles):
            rng = random.Random(7000 + s)
            perm = vals[:]
            rng.shuffle(perm)
            smap = dict(zip(ce, perm))
            g2 = build_interdiction_game(it.env.graph, it.od[0], it.od[1], K, k_extra=KX,
                                         intercept_fn=survival_intercept_fn(smap))
            assert g2.routes == it.env.game.routes, "route set changed under shuffle"
            sol2 = solve_multiconvoy(g2, N, "mission")
            _, M2 = objective_matrix(g2, N, "mission")
            wv2 = route_worst(it, smap)
            gd = policy_dist(states, it, smap, wv2)
            rd = policy_dist(rnd, it, smap, wv2)
            gr = score(it, gd, M2, sol2.loss_mixed)
            rr = score(it, rd, M2, sol2.loss_mixed)
            g_all.append(gr); r_all.append(rr); cells += 1
            _, gexpl = best_response_attacker_multi(M2, gd)
            det_beats += int(gexpl < sol2.loss_det)
    out["A2"] = {"gen_mean": float(np.mean(g_all)), "rand_mean": float(np.mean(r_all)),
                 "beats_loss_det": det_beats, "cells": cells,
                 "gen": [round(x, 2) for x in g_all]}
    print(f"A2 SHUFFLED reality ({cells} cells): gen {np.mean(g_all):.2f}x vs rand "
          f"{np.mean(r_all):.2f}x | beats loss_det {det_beats}/{cells}", flush=True)

    # ---------- A3: intel error (true game scores; observed map corrupted) ----------
    for kind, levels in (("shuffle_frac", [0.25, 0.5, 1.0]), ("mult_sigma", [0.1, 0.25, 0.5])):
        for lv in levels:
            gs = []
            for it in insts:
                ce = cand_edges(it)
                cmap = true_cand_map(it)
                for s in range(3):
                    rng = random.Random(9000 + s)
                    obs_map = dict(cmap)
                    if kind == "shuffle_frac":
                        sub = rng.sample(ce, max(2, int(round(lv * len(ce)))))
                        sv = [obs_map[e] for e in sub]
                        rng.shuffle(sv)
                        for e, p in zip(sub, sv):
                            obs_map[e] = p
                    else:
                        nrng = np.random.default_rng(9000 + s)
                        for e in ce:
                            obs_map[e] = float(np.clip(obs_map[e] * (1 + nrng.normal(0, lv)),
                                                       0.05, 0.99))
                    wv = route_worst(it, obs_map)
                    gs.append(score(it, policy_dist(states, it, obs_map, wv),
                                    it.env.obj_matrix, it.eq))
            out["A3"][f"{kind}={lv}"] = {"gen_mean": float(np.mean(gs))}
            print(f"A3 {kind}={lv}: gen {np.mean(gs):.2f}x  (true-map anchor "
                  f"{out['sanity']['gen']:.2f}, rand {out['sanity']['rand']:.2f})", flush=True)

    # ---------- post-hoc DIAGNOSTIC (labelled, not gated): information-free observed map ----------
    # All candidate vulnerabilities observed as the band midpoint (0.55): if performance holds at
    # the true-map anchor, the map observation contributes ~nothing and the mechanism is the
    # geometry/cost pathway + multi-instance training, not per-edge map reading.
    gs = []
    for it in insts:
        obs_map = {e: 0.55 for e in cand_edges(it)}
        wv = route_worst(it, obs_map)
        gs.append(score(it, policy_dist(states, it, obs_map, wv), it.env.obj_matrix, it.eq))
    out["diagnostic_constant_map"] = {"gen_mean": float(np.mean(gs)),
                                      "per_od": [round(x, 2) for x in gs]}
    print(f"DIAGNOSTIC constant map (info-free obs): gen {np.mean(gs):.2f}x "
          f"(true-map anchor {out['sanity']['gen']:.2f})", flush=True)

    json.dump(out, open(a.json_out, "w"), indent=2)
    print(f"[written] {a.json_out}")


if __name__ == "__main__":
    main()
