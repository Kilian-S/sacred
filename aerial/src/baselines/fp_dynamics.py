"""Smooth fictitious-play attacker discipline, shared by the single- and multi-convoy trainers.

Each block recomputes a softmax best response (temperature `tau`) to the defender's trailing-window
empirical play, and the committed interdiction set is then sampled fresh every sortie from those
probabilities. Holding one pure best response for a whole block over-disciplines the defender and
cycles, while a stale all-history mixture under-disciplines it and lets it park. The single-convoy
caller passes a route-play sequence with the route x iset payoff, the multi-convoy caller an
occupancy-play sequence with the occupancy x iset objective matrix.
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
    """Smooth-FP attacker distribution over interdiction sets.

    A softmax at temperature `tau` of the expected objective per set against the defender's
    trailing-`window` empirical play. Recompute it per block and sample the committed set fresh
    each sortie.
    """
    d = empirical_play_hist(play_seq[-window:], n_strategies)
    e = d @ obj_matrix
    z = np.exp((e - e.max()) / tau)
    return z / z.sum()


def sample_smooth_iset(smooth_probs: np.ndarray, rng) -> int:
    """Sample one committed interdiction set from the smooth-FP distribution (fresh each sortie)."""
    return int(rng.choice(len(smooth_probs), p=smooth_probs))
