#!/usr/bin/env python3
"""gen34 DESIGN PROBE (oracle-only, no training): the hidden-adversary-type family.

The gen27/31/32 positives share one legitimacy caveat: the adaptive enemy is a SINGLE
hand-chosen rule the defender can be tuned to. gen34's proposed repair: the enemy TYPE is drawn
per episode, hidden, from a small doctrine family; the defender must infer the type from what it
observes (its own window + the realised interdiction placements) and adapt. This probe computes
the exact landscape that decides whether that game is worth training on:

  - OMNI cap: expected exact optimum when the type is KNOWN each episode (mean over members of
    each member's exact dynamic optimum; Karp on each member's window MDP).
  - BLIND cap: the exact optimum over ALL type-blind window policies (window transitions are
    defender-controlled, so the optimum vs the uniform type mixture = Karp on the
    mixture-averaged cost). Every static object, composed rule and window-only policy is
    bounded below by the BLIND cap by construction.
  - The INFERENCE GAP = blind - omni: the value that only type-inference (from placement
    observations) can unlock. This is the act's PRIMARY headroom, the gen27 "beat the static
    cap" structure lifted one level.
  - Brittleness cross-table: each member's specialist optimal policy evaluated against every
    other member (off-diagonal blow-ups = why no fixed doctrine-counter works).
  - Naive rules (rotation / anti-repeat / iid_eq mixture / best fixed route) vs the mixture.
  - The FITTED "playbook" row: Bayes-MAP over members from observed placements, then play the
    MAP specialist (requires knowing the member set + functional forms = the disclosed
    oracle-fitted cap analogue, MC over finite episodes).

Members (all on the same L, K=1; window w=3 unless stated):
  M1 reactive      softmax-BR tau=0.15 to window counts (the gen19/27 incumbent)
  M2 sharp         softmax-BR tau=0.05 (near-argmax reactive)
  M3 anticipatory  predicts next route = uniform over routes NOT in window (anti-repeat
                   assumption), softmax-BRs (tau=0.15) to that prediction (the gen31 q_flee
                   analogue on roads)
  M4 doctrine      window-independent: softmax-BR (tau=0.15) to the defender's static
                   equilibrium mixture (aims at your long-run doctrine)
  M5 scattergun    uniform over interdiction sets (best answer is DETERMINISTIC: punishes
                   blanket randomisation)

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen34_family_probe.py
Writes models/runs/gen34_family_probe.json
"""
from __future__ import annotations

import json

import numpy as np
import torch

from scratch.critique_followup_probes import antirepeat_value, disjoint_subset, rotation_value
from scratch.dyn_exact import (
    build_window_mdp, damped_rvi, greedy_policy_from_rvi, karp_mmc, policy_value_exact)
from scripts.train_b1lite1 import softmax_br, stacked_L
from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(1)
N, K, BAND, KX, W, TAU = 3, 1, (0.15, 0.95), 8, 3, 0.15
EPISODE_LEN, MC_EPISODES, MAP_CONF = 40, 3000, 0.60
rng = np.random.default_rng(0)


def member_fns(L, eq):
    R = L.shape[0]

    def m1(counts):
        return softmax_br(counts, L, 0.15)

    def m2(counts):
        return softmax_br(counts, L, 0.05)

    def m3(counts):
        free = (counts == 0).astype(float)
        d = free / free.sum() if free.sum() > 0 else np.ones(R) / R
        e = d @ L
        z = np.exp((e - e.max()) / 0.15)
        return z / z.sum()

    q4 = None

    def m4(counts):
        nonlocal q4
        if q4 is None:
            e = eq @ L
            z = np.exp((e - e.max()) / 0.15)
            q4 = z / z.sum()
        return q4

    def m5(counts):
        return np.ones(L.shape[1]) / L.shape[1]

    return dict(reactive=m1, sharp=m2, anticipatory=m3, doctrine=m4, scattergun=m5)


def probe_instance(env, tag):
    L = stacked_L(env.game, N)
    R = L.shape[0]
    v_eq, eq = _row_minimiser(L)
    fns = member_fns(L, eq)
    names = list(fns)
    mdps = {}
    for nm in names:
        cost, n, R_, pw = build_window_mdp(L, TAU, W, member_fn=fns[nm])
        mdps[nm] = cost
    n, pw = R ** W, R ** (W - 1)

    omni, specialists = {}, {}
    for nm in names:
        omni[nm] = karp_mmc(mdps[nm], n, R, pw)
        specialists[nm] = greedy_policy_from_rvi(mdps[nm], n, R, pw)
    omni_cap = float(np.mean(list(omni.values())))

    mix_cost = np.mean([mdps[nm] for nm in names], axis=0)
    blind_cap = karp_mmc(mix_cost, n, R, pw)
    g_damp, conv = damped_rvi(mix_cost, n, R, pw)
    assert (not conv) or abs(g_damp - blind_cap) < 1e-6, (g_damp, blind_cap)
    blind_policy = greedy_policy_from_rvi(mix_cost, n, R, pw)

    cross = {}
    for pn in names:
        cross[pn] = {en: policy_value_exact(specialists[pn], mdps[en], n, R, pw)
                     for en in names}

    dis = disjoint_subset([set(e) for e in env.game.route_edges])
    d_eq = np.zeros(R); d_eq[:] = eq
    rules = {}
    rot_pol = None
    per_member_rules = {}
    for rn, pol in (("iid_eq_mixture", np.broadcast_to(eq, (n, R)).copy()),
                    ("blind_optimal", blind_policy)):
        per_member_rules[rn] = {en: policy_value_exact(pol, mdps[en], n, R, pw) for en in names}
        rules[rn] = float(np.mean(list(per_member_rules[rn].values())))
    for rn, fn in (("rotation", lambda c: rotation_value(list(dis), L, 0.15, W)),
                   ("anti_repeat", lambda c: antirepeat_value(dis, L, 0.15, W))):
        pass  # rotation/anti-repeat need per-member evaluation below

    def rule_policy_antirepeat():
        pol = np.zeros((n, R))
        dec = np.empty((n, W), dtype=np.int64)
        x = np.arange(n)
        for i in range(W):
            dec[:, W - 1 - i] = x % R
            x = x // R
        for s in range(n):
            free = [r for r in dis if r not in dec[s]]
            allowed = free or list(dis)
            pol[s, allowed] = 1.0 / len(allowed)
        return pol

    anti_pol = rule_policy_antirepeat()
    per_member_rules["anti_repeat"] = {en: policy_value_exact(anti_pol, mdps[en], n, R, pw)
                                       for en in names}
    rules["anti_repeat"] = float(np.mean(list(per_member_rules["anti_repeat"].values())))
    best_fixed = None
    fixed_vals = []
    for r in range(R):
        pol = np.zeros((n, R)); pol[:, r] = 1.0
        v = float(np.mean([policy_value_exact(pol, mdps[en], n, R, pw) for en in names]))
        fixed_vals.append(v)
    rules["best_fixed_route"] = float(min(fixed_vals))

    qtab = {nm: np.zeros((n, L.shape[1])) for nm in names}
    dec = np.empty((n, W), dtype=np.int64)
    x = np.arange(n)
    for i in range(W):
        dec[:, W - 1 - i] = x % R
        x = x // R
    for s in range(n):
        counts = np.bincount(dec[s], minlength=R).astype(float)
        for nm in names:
            qtab[nm][s] = fns[nm](counts)

    def mc_bayes_map():
        tot = 0.0
        for _ in range(MC_EPISODES):
            m_true = rng.integers(len(names))
            post = np.ones(len(names)) / len(names)
            win = []
            for _t in range(EPISODE_LEN):
                s = 0
                for v in ([0] * (W - len(win)) + win[-W:]):
                    s = s * R + v
                if post.max() >= MAP_CONF:
                    pol = specialists[names[int(post.argmax())]]
                else:
                    pol = blind_policy
                a = int(rng.choice(R, p=pol[s]))
                j = int(rng.choice(L.shape[1], p=qtab[names[m_true]][s]))
                tot += L[a, j]
                lik = np.array([qtab[nm][s][j] for nm in names])
                post = post * np.maximum(lik, 1e-12)
                post /= post.sum()
                win.append(a)
        return tot / (MC_EPISODES * EPISODE_LEN)

    fitted = mc_bayes_map()
    row = dict(tag=tag, R=R, m=len(dis), n_isets=int(L.shape[1]),
               omni_per_member=omni, omni_cap=omni_cap, blind_cap=blind_cap,
               inference_gap=blind_cap - omni_cap,
               inference_ratio=blind_cap / omni_cap,
               specialist_cross=cross, rules_vs_mixture=rules,
               per_member_rules=per_member_rules, fitted_bayes_map=fitted)
    print(f"\n=== {tag} (R={R}, m={len(dis)}) ===")
    print(f"  omni per member: " + " ".join(f"{k} {v:.4f}" for k, v in omni.items()))
    print(f"  OMNI cap {omni_cap:.4f} | BLIND cap {blind_cap:.4f} | inference gap "
          f"{row['inference_gap']:.4f} ({row['inference_ratio']:.2f}x)")
    print(f"  fitted Bayes-MAP (playbook) {fitted:.4f}")
    print(f"  rules vs mixture: " + " ".join(f"{k} {v:.4f}" for k, v in rules.items()))
    print("  specialist cross (rows=policy, cols=enemy):")
    for pn in names:
        print(f"    {pn:13s} " + " ".join(f"{cross[pn][en]:.4f}" for en in names))
    return row


def main():
    out = {"config": dict(N=N, K=K, band=BAND, kx=KX, w=W, tau=TAU,
                          episode_len=EPISODE_LEN, mc_episodes=MC_EPISODES,
                          map_conf=MAP_CONF)}
    rows = []
    for od in ["35-159", "62-97"]:
        s, t = od.split("-")
        env = make_multiconvoy_env(od=(s, t), N=N, K=K, k_extra_routes=KX, menu_select=True,
                                   edge_vuln_band=BAND, interception_loss=10.0)
        rows.append(probe_instance(env, f"kaliningrad {od}"))
    gd = sample_instances(6, N, K, BAND, KX, 0, city="gdansk")[0]
    rows.append(probe_instance(gd.env, f"gdansk {gd.od[0]}-{gd.od[1]}"))
    out["rows"] = rows
    with open("models/runs/gen34_family_probe.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote models/runs/gen34_family_probe.json")


if __name__ == "__main__":
    main()
