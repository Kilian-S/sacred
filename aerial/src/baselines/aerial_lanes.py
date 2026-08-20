"""Lane-heuristic baselines for the aerial sector.

These are the strongest naive strategies a practitioner would write for free-flight interdiction, a
uniform stack over floor(W / 2r) + 1 maximally separated lateral lanes, an inverse-risk lane stack,
the same two variants over the full route menu, and tabular smooth fictitious play against the same
best-response oracle. All are scored by `best_response_attacker` on the same game as every other
arm.
"""

from __future__ import annotations

import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame, best_response_attacker
from src.envs.aerial_sector import Path, SectorLattice, lane_path


def lane_rows(lat: SectorLattice, r: float) -> list[int]:
    """The maximally separated lane rows.

    n = floor(W / 2r) + 1 evenly spaced rows, clipped to the row count, rounded to lattice rows and
    deduplicated in order.
    """
    # A spacing of exactly 2r still separates two lanes, since the taper reaches 0 at exactly r, so
    # the boundary case counts; the epsilon guards float division (8/1.6 -> 4.999...).
    n = min(int(lat.W / (2.0 * r) + 1e-9) + 1, lat.ny)
    rows = np.linspace(0.0, lat.W, n)
    out: list[int] = []
    for v in rows:
        j = int(round(v))
        if j not in out:
            out.append(j)
    return out


def lane_menu_indices(lat: SectorLattice, menu: list[Path], r: float) -> list[int]:
    """Menu indices of the lane paths for radius r.

    The menu builder puts lanes first, so an index exists whenever the lane is constructible on the
    lattice.
    """
    idx: list[int] = []
    pos = {p: i for i, p in enumerate(menu)}
    for row in lane_rows(lat, r):
        p = lane_path(lat, row)
        if p is not None and p in pos:
            idx.append(pos[p])
    return idx


def _stack_distribution(n_routes: int, indices: list[int], weights: np.ndarray) -> np.ndarray:
    d = np.zeros(n_routes)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    for i, wi in zip(indices, w):
        d[i] += wi
    return d


def lane_stack_distributions(game: InterdictionGame, lane_idx: list[int],
                             S: np.ndarray) -> dict[str, np.ndarray]:
    """The four naive stacks as route distributions.

    ``S`` is the single-hazard survival matrix [R, H], from which exposure_i = 1 - min_h S[i, h].
    """
    exposure = 1.0 - S.min(axis=1)
    n = game.n_routes
    out: dict[str, np.ndarray] = {}
    if lane_idx:
        out["uniform_lane"] = _stack_distribution(n, lane_idx, np.ones(len(lane_idx)))
        out["invrisk_lane"] = _stack_distribution(
            n, lane_idx, 1.0 / np.clip(exposure[lane_idx], 1e-9, None))
    out["uniform_full"] = np.full(n, 1.0 / n)
    out["invrisk_full"] = (1.0 / np.clip(exposure, 1e-9, None))
    out["invrisk_full"] = out["invrisk_full"] / out["invrisk_full"].sum()
    return out


def tabular_smooth_fp(game: InterdictionGame, rounds: int = 4000, eta: float = 0.25,
                      ) -> tuple[float, np.ndarray]:
    """Tabular smooth fictitious play against the exact best-response oracle.

    A multiplicative-weights defender plays against the best response to its running average
    strategy, which is drift-free by construction. Returns the exploitability of the average
    strategy and the strategy itself.
    """
    n = game.n_routes
    x = np.full(n, 1.0 / n)
    total = np.zeros(n)
    for t in range(1, rounds + 1):
        total = total + x
        j, _ = best_response_attacker(game, total / t)
        x = x * np.exp(-eta * game.payoff[:, j])
        x = x / x.sum()
    avg = total / rounds
    _, value = best_response_attacker(game, avg)
    return value, avg
