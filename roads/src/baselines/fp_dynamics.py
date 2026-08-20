"""Smooth fictitious-play attacker dynamics.

Each block, the interdictor recomputes a softmax best response (temperature `tau`) to the
defender's trailing-window empirical play, then samples a fresh interdiction set from that
distribution every sortie. Sampling rather than committing to a single best response keeps the
dynamics from cycling. Shared by the single-convoy and multi-convoy trainers.
"""
from __future__ import annotations

import numpy as np


def empirical_play_hist(seq, n: int) -> np.ndarray:
    """Empirical distribution of a play sequence over `n` strategies (uniform if empty)."""
    h = np.zeros(n)
    for i in seq:
        h[i] += 1.0
    return h / h.sum() if h.sum() > 0 else np.ones(n) / n


def smooth_fp_probs(play_seq, n_strategies: int, obj_matrix: np.ndarray,
                    tau: float, window: int) -> np.ndarray:
    """Attacker distribution over interdiction sets: softmax (temperature `tau`) of the expected
    objective per iset against the defender's trailing-`window` empirical play. Recompute per
    block; sample the committed iset fresh each sortie (see `sample_smooth_iset`)."""
    d = empirical_play_hist(play_seq[-window:], n_strategies)
    e = d @ obj_matrix
    z = np.exp((e - e.max()) / tau)
    return z / z.sum()


def sample_smooth_iset(smooth_probs: np.ndarray, rng) -> int:
    """Sample one committed interdiction set from the smooth-FP distribution (fresh each sortie)."""
    return int(rng.choice(len(smooth_probs), p=smooth_probs))
