#!/usr/bin/env python3
"""Exact solvers for the pattern-of-life window MDP (shared by the 2026-07-23 probes).

The dynamic game of gen19/gen27 (fleet stacks; adversary softmax-BRs with temperature tau to
the trailing w-window of realised routes) is an average-cost MDP over the R^w window states
with DETERMINISTIC transitions. Its exact optimum is therefore the MINIMUM MEAN CYCLE of the
(state --action--> state) graph. Karp's algorithm computes it with no iteration/tolerance;
damped relative value iteration cross-checks it (the two agreed to 5 decimals on all 10 cells
of analysis/gen35_mmc_check.py).

Discovered 2026-07-23 (models/runs/gen35_mmc_check.json): the undamped RVI inside
scripts/train_b1lite1.py:oracle_refs OSCILLATES on this MDP class and its `history_opt` is
wrong on every cell tested (+18% on 35-159 K=1, -44%/-57% on 71-33 K=2/3). Use THIS module's
history_opt_exact for every future yardstick; ledger appendices restate the banked rows.

State encoding: window tuple s=(s_0..s_{w-1}) (s_0 oldest), integer id = sum s_i R^{w-1-i}
(lexicographic; identical to itertools.product order used by oracle_refs). Transition:
nxt(s, a) = (s mod R^{w-1}) * R + a.
"""
from __future__ import annotations

import numpy as np

from scripts.train_b1lite1 import softmax_br


def build_window_mdp(L, tau, w, member_fn=None):
    """cost[s, a] for the window MDP. member_fn(counts) -> iset distribution overrides the
    default softmax-BR adversary (used for the gen34 adversary-family members)."""
    R = L.shape[0]
    n = R ** w
    pw = R ** (w - 1)
    cost = np.zeros((n, R))
    dec = np.empty((n, w), dtype=np.int64)
    x = np.arange(n)
    for i in range(w):
        dec[:, w - 1 - i] = x % R
        x = x // R
    for s in range(n):
        counts = np.bincount(dec[s], minlength=R).astype(float)
        q = member_fn(counts) if member_fn is not None else softmax_br(counts, L, tau)
        cost[s] = L @ q
    return cost, n, R, pw


def karp_mmc(cost, n, R, pw):
    """Exact minimum mean cycle (= exact average-cost optimum of the deterministic MDP)."""
    v = np.arange(n)
    heads = v // R                       # s[1:] as integer prefix
    U = (np.arange(R)[:, None] * pw) + heads[None, :]   # U[x, v] = predecessor of v via x
    A = v % R                            # the action that leads into v
    Cin = cost[U, A[None, :]]            # cost of edge U[x,v] -> v
    d = np.full((n + 1, n), np.inf)
    d[0] = 0.0
    for k in range(1, n + 1):
        d[k] = (d[k - 1][U] + Cin).min(axis=0)
    ks = np.arange(n)[:, None]
    with np.errstate(invalid="ignore"):
        ratios = (d[n][None, :] - d[:n]) / (n - ks)
    ratios = np.where(np.isfinite(ratios), ratios, -np.inf)
    per_v = ratios.max(axis=0)
    per_v = np.where(np.isfinite(d[n]), per_v, np.inf)
    return float(per_v.min())


def damped_rvi(cost, n, R, pw, iters=60_000, damp=0.5, tol=1e-11):
    """Damped RVI cross-check; returns (gain, converged)."""
    heads = (np.arange(n) % pw) * R
    nxt = heads[:, None] + np.arange(R)[None, :]
    h = np.zeros(n)
    g = 0.0
    for _ in range(iters):
        q = (cost + h[nxt]).min(axis=1)
        g = q.mean() - h.mean()
        h_new = q - q[0]
        if np.max(np.abs(h_new - h)) < tol:
            return float(g), True
        h = damp * h_new + (1 - damp) * h
    return float(g), False


def history_opt_exact(L, tau, w, member_fn=None, check=True):
    """The corrected dynamic-optimum yardstick: Karp, cross-checked by damped RVI."""
    cost, n, R, pw = build_window_mdp(L, tau, w, member_fn=member_fn)
    mmc = karp_mmc(cost, n, R, pw)
    if check:
        g, conv = damped_rvi(cost, n, R, pw)
        if conv and abs(g - mmc) > 1e-6:
            raise AssertionError(f"Karp {mmc} vs damped RVI {g} disagree")
    return mmc


def greedy_policy_from_rvi(cost, n, R, pw, iters=60_000, damp=0.5, tol=1e-11):
    """Deterministic optimal policy (row-stochastic [n, R] one-hot) extracted from the damped
    RVI value function of THIS cost matrix."""
    heads = (np.arange(n) % pw) * R
    nxt = heads[:, None] + np.arange(R)[None, :]
    h = np.zeros(n)
    for _ in range(iters):
        q = (cost + h[nxt]).min(axis=1)
        h_new = q - q[0]
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new
            break
        h = damp * h_new + (1 - damp) * h
    a = (cost + h[nxt]).argmin(axis=1)
    pol = np.zeros((n, R))
    pol[np.arange(n), a] = 1.0
    return pol


def policy_value_exact(policy, cost, n, R, pw, iters=40_000, tol=1e-13, damp=0.5):
    """Exact average cost of a stationary (possibly stochastic) window policy.
    policy[s] = distribution over R actions. DAMPED power iteration on the induced chain
    (deterministic policies induce periodic cycles; undamped iteration oscillates on them -
    the dbf385d lesson a third time). Damping preserves the stationary distribution."""
    heads = (np.arange(n) % pw) * R
    pi = np.ones(n) / n
    c = (policy * cost).sum(axis=1)
    for _ in range(iters):
        nxt_pi = np.zeros(n)
        for a in range(R):
            np.add.at(nxt_pi, heads + a, pi * policy[:, a])
        nxt_pi = damp * nxt_pi + (1 - damp) * pi
        if np.max(np.abs(nxt_pi - pi)) < tol:
            pi = nxt_pi
            break
        pi = nxt_pi
    return float(pi @ c)
