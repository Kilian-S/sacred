"""Walker environment: leg-survival calibration, the polyline reference game, the node-mode
contract against the SAC stack, and exact-DP exploitability against brute-force enumeration."""

import itertools

import numpy as np
import pytest
import torch

from src.agents.sac import ProtagonistSAC
from src.envs.aerial_sector import SectorLattice
from src.envs.aerial_walker import (AerialWalkerEnv, _nid, build_arc_survival,
                                    build_polyline_game, path_survival,
                                    walker_exploitability, walker_policy_probs)

LAT = SectorLattice(ny=9, nx=13)
TINY = SectorLattice(ny=3, nx=5)


def _prot(hidden=32):
    return ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=hidden, num_layers=2,
                          heads=2, reward_scale=1.0, device="cpu", role_alpha=True)


def test_leg_survival_calibration():
    centres = np.array([[6.0, 4.0]])
    arcs, ai, A = build_arc_survival(LAT, centres, r=1.5, p_max=0.9)
    straight = tuple((i, 4) for i in range(LAT.nx))
    s = path_survival(straight, ai, A)
    assert 1.0 - s[0] == pytest.approx(0.9, rel=0.02)        # dead-centre transit = p_max
    far = np.array([[6.0, 8.0]])
    _, ai2, A2 = build_arc_survival(LAT, far, r=1.5, p_max=0.9)
    assert 1.0 - path_survival(straight, ai2, A2)[0] < 1e-9  # outside radius: safe


def test_polyline_game_payoff():
    centres = np.array([[4.0, 2.0], [8.0, 6.0]])
    arcs, ai, A = build_arc_survival(LAT, centres, r=1.5, p_max=0.8)
    paths = [tuple((i, 4) for i in range(LAT.nx)),
             tuple((i, min(8, 4 + i if i <= 6 else 16 - i)) for i in range(LAT.nx))]
    game = build_polyline_game(LAT, paths, centres, K=2, arc_index=ai, A=A)
    s0 = path_survival(paths[0], ai, A)
    assert game.payoff[0, 0] == pytest.approx(1.0 - s0[0] * s0[1], abs=1e-12)


def test_env_walk_and_node_mode_update():
    env = AerialWalkerEnv(LAT, np.array([[6.0, 4.0], [4.0, 6.0]]), K=1, r=1.2)
    prot = _prot()
    obs = env.reset()
    assert "menu_route_node_idx" not in obs                   # NODE mode, never menu
    assert obs["goal_dists"][0][_nid(LAT.target)] == 0.0
    steps = []
    done = False
    while not done:
        obs = env.observe()
        mask = env.action_mask()
        assert all(m in obs["nodes"] for m in mask[0])
        act = prot.select_action(obs, mask)
        assert act[0] in mask[0]
        steps.append((obs, act[0], mask))
        done = env.step(act[0])
    assert len(env.path) == LAT.nx and env.path[-1] == LAT.target
    s = env.realised_survival()
    assert s.shape == (2,) and np.all((0 < s) & (s <= 1))
    from scripts.train_multiconvoy import _transition
    for i, (obs, a, mask) in enumerate(steps):
        last = i == len(steps) - 1
        obs["target_entropy"] = 0.5 * np.log(max(2, len(mask[0])))
        obs["alpha_group"] = 0
        nobs, nmask = (None, None) if last else (steps[i + 1][0], steps[i + 1][2])
        prot.replay_buffer.push(_transition(obs, 0, a, mask, -1.0 if last else 0.0,
                                            nobs, 0, nmask, last))
    prot.update(8)


def test_dp_exploitability_matches_bruteforce():
    centres = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
    env = AerialWalkerEnv(TINY, centres, K=1, r=0.9)
    prot = _prot(hidden=16)
    torch.manual_seed(3)
    worst_dp, elen = walker_exploitability(prot, env)
    pi = walker_policy_probs(prot, env)
    allp = []
    def rec2(node, p, acc):
        if node[0] == TINY.nx - 1:
            allp.append((tuple(acc), p))
            return
        succ, pr = pi[node]
        for s, pj in zip(succ, pr):
            rec2(s, p * float(pj), acc + [s])
    rec2(TINY.base, 1.0, [TINY.base])
    assert sum(p for _, p in allp) == pytest.approx(1.0, abs=1e-6)
    worst_bf, elen_bf = 0.0, 0.0
    for h in range(len(centres)):
        e = sum(p * (1.0 - path_survival(path, env.arc_index, env.A)[h]) for path, p in allp)
        worst_bf = max(worst_bf, e)
    elen_bf = sum(p * sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:]))
                  for path, p in allp)
    assert worst_dp == pytest.approx(worst_bf, abs=1e-7)
    assert elen == pytest.approx(elen_bf, abs=1e-7)


def test_dp_k2_pairs():
    centres = np.array([[2.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    env = AerialWalkerEnv(TINY, centres, K=2, r=0.9)
    prot = _prot(hidden=16)
    worst, _ = walker_exploitability(env=env, prot=prot)
    pi = walker_policy_probs(prot, env)
    allp = []
    def rec(node, p, acc):
        if node[0] == TINY.nx - 1:
            allp.append((tuple(acc), p)); return
        succ, pr = pi[node]
        for s, pj in zip(succ, pr):
            rec(s, p * float(pj), acc + [s])
    rec(TINY.base, 1.0, [TINY.base])
    worst_bf = 0.0
    for i, j in itertools.combinations(range(3), 2):
        e = sum(p * (1.0 - path_survival(path, env.arc_index, env.A)[i]
                     * path_survival(path, env.arc_index, env.A)[j]) for path, p in allp)
        worst_bf = max(worst_bf, e)
    assert worst == pytest.approx(worst_bf, abs=1e-7)
