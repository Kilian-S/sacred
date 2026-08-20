"""Smooth fictitious-play attacker discipline (single source of truth, gen08/gen09).

The SMOOTH-FP interdictor that stabilised the single-convoy B2-P3 result: each block it recomputes a
softmax best response (temperature `tau`) to the defender's TRAILING-WINDOW empirical play, then a
committed interdiction set is SAMPLED FRESH EVERY sortie from those probabilities. This is what keeps
the last iterate from cycling (block-holding one pure BR over-disciplines -> cycling; a stale
all-history mixture under-disciplines -> parking). Extracted verbatim from `scripts/train_interdiction.py`
so `scripts/train_multiconvoy.py` reuses the PROVEN-STABLE code rather than a second implementation:
single-convoy passes a route-play sequence with the route x iset payoff; multi-convoy passes an
occupancy-play sequence with the occupancy x iset objective matrix (the structure is analogous, K
committed edges either way).
"""
from __future__ import annotations

import numpy as np


def empirical_play_hist(seq, n: int) -> np.ndarray:
    """Empirical distribution of a play sequence over `n` strategies (uniform if empty).
    (= the single-convoy trainer's `_hist`.)"""
    h = np.zeros(n)
    for i in seq:
        h[i] += 1.0
    return h / h.sum() if h.sum() > 0 else np.ones(n) / n


def smooth_fp_probs(play_seq, n_strategies: int, obj_matrix: np.ndarray,
                    tau: float, window: int) -> np.ndarray:
    """Smooth-FP attacker distribution over interdiction sets: softmax(temperature `tau`) of the
    expected objective per iset against the defender's TRAILING-`window` empirical play. Recompute
    per block; SAMPLE the committed iset fresh each sortie (see `sample_smooth_iset`)."""
    d = empirical_play_hist(play_seq[-window:], n_strategies)
    e = d @ obj_matrix
    z = np.exp((e - e.max()) / tau)
    return z / z.sum()


def sample_smooth_iset(smooth_probs: np.ndarray, rng) -> int:
    """Sample one committed interdiction set from the smooth-FP distribution (fresh each sortie)."""
    return int(rng.choice(len(smooth_probs), p=smooth_probs))
