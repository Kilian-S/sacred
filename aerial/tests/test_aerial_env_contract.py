"""The aerial environment's observation and menu contract against the SAC stack: featuriser, node
index map, menu-select head, replay and update. Plumbing only, no training claims."""

import numpy as np
import pytest
import torch

from scripts.train_multiconvoy import _transition, route_one
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.envs.aerial_curves import build_curve_menu, dense_hazard_grid
from src.envs.aerial_interdiction_env import AerialInterdictionEnv, _nid
from src.envs.aerial_sector import SectorLattice

LAT = SectorLattice(ny=9, nx=13)


def _env():
    menu, _ = build_curve_menu(LAT, r=1.2, R=12, seed=0)
    centres = dense_hazard_grid(LAT, step=1.0)
    return AerialInterdictionEnv(LAT, menu, centres, K=1, r=1.2)


def _prot(feat_dim=2):
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=32, num_layers=2, heads=2,
                          reward_scale=1.0, device="cpu", role_alpha=True)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feat_w = torch.nn.Parameter(torch.zeros(feat_dim))
        net.route_feats = None
    return prot


def test_observation_featurizes_and_menu_indices_match_sorted_rows():
    env = _env()
    obs = env.reset()
    pyg = featurize_state(obs, 0)
    assert pyg.x.shape == (len(obs["nodes"]), 14)
    assert pyg.edge_attr.shape[1] == 5
    assert float(pyg.edge_attr[:, 4].max()) > 0.0          # the threat projection reaches col 4
    n2i = node_index_map(obs)
    for r, curve in enumerate(env.menu):                    # menu rows are sorted-order indices
        expect = [n2i[_nid(n)] for n in curve.node_seq]
        assert obs["menu_route_node_idx"][r].tolist() == expect
    assert obs["menu_route_feats"].shape == (env.game.n_routes, 2)


def test_obj_matrix_is_payoff_at_n1():
    env = _env()
    assert env.obj_matrix.shape == env.game.payoff.shape
    assert np.allclose(env.obj_matrix, env.game.payoff)     # at N=1 the mission is interception


def test_route_one_and_update_run_end_to_end():
    env = _env()
    prot = _prot()
    torch.manual_seed(0)
    for k in range(6):
        env.reset()
        env.commit(k % env.game.payoff.shape[1])
        steps, occ, routes = route_one(prot, env, fleet_route=True)
        assert len(steps) == 1 and sum(occ) == 1 and occ[routes[0]] == 1
        obs, ci, hop, mask = steps[0]
        obs["target_entropy"] = 0.5 * np.log(env.game.n_routes)
        obs["alpha_group"] = 0
        reward = -float(env.game.payoff[routes[0], k % env.game.payoff.shape[1]])
        prot.replay_buffer.push(_transition(obs, ci, hop, mask, reward, None, None, None, True))
    prot.update(4)


def test_exact_distribution_sums_to_one_and_scores():
    """``exact_ratio`` returns the stacked occupancy distribution and the mission best-response
    ratio to the fleet equilibrium."""
    from scripts.train_aerial_generalist import exact_ratio, make_layout_instance
    inst = make_layout_instance("t", 1234)
    assert inst.env.config.N == 3
    assert inst.env._obs_cache["menu_route_feats"].shape[1] == 1   # exposure only
    prot = _prot(feat_dim=1)
    ratio, d = exact_ratio(prot, inst)
    assert d.shape == (len(inst.env.occupancies),)
    assert d.sum() == pytest.approx(1.0, abs=1e-5)
    stacked = [i for i, o in enumerate(inst.env.occupancies) if max(o) == 3]
    assert d[stacked].sum() == pytest.approx(1.0, abs=1e-5)   # fleet-route: all mass stacked
    assert ratio >= 1.0 - 1e-9                                # nothing beats the equilibrium
