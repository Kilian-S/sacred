"""gen29 multi-OD env: joint-payoff cross-check, node-ordering contract, overlap/blind channel,
deterministic pool build, and exact-estimator vs Monte-Carlo agreement."""
import numpy as np
import pytest
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiod_interdiction import make_multiod_env

CELL = ("147", ("212", "188", "195"))


def _env():
    return make_multiod_env(*CELL, K=1)


def test_obj_matrix_matches_mission_failure_loop():
    env = _env()
    M = env.obj_matrix
    R = [len(rs) for rs in env.route_sets]
    rng = np.random.default_rng(0)
    for _ in range(200):
        routes = [int(rng.integers(R[f])) for f in range(env.F)]
        j = int(rng.integers(len(env.isets)))
        assert M[env.joint_index(routes), j] == pytest.approx(env.mission_failure(routes, j), abs=1e-12)


def test_eq_and_det_wellposed():
    env = _env()
    v_joint, _ = _row_minimiser(env.obj_matrix)
    v_det = float(env.obj_matrix.max(axis=1).min())
    assert 0.05 < v_joint < 0.9 and v_det > v_joint + 0.1     # non-degenerate, determinism exploitable
    assert v_joint == pytest.approx(0.205, abs=0.01)          # the pinned headline eq


def test_menu_node_order_contract():
    env = _env()
    obs = env.reset()
    n2i = node_index_map(obs)
    for r, route in enumerate(env.route_sets[0]):
        assert obs["menu_route_node_idx"][r].tolist() == [n2i[str(n)] for n in route]
    assert obs["menu_route_feats"].shape == (len(env.route_sets[0]), 2)


def test_overlap_channel_and_blind():
    env = _env()
    env.reset()
    env.route_stream_by_index(0)                              # commit stream 0
    o1 = env.observe()
    assert float(o1["menu_route_feats"][:, 1].max()) > 0.0    # overlap active for stream 1
    assert o1["taken_node_frac"]                              # earlier route recorded
    env.blind = True
    env.set_committed([0])
    ob = env.observe()
    assert float(ob["menu_route_feats"][:, 1].max()) == 0.0   # coordination channel zeroed
    assert ob["taken_node_frac"] == {}
    assert ob["menu_route_feats"][:, 0].tolist() == o1["menu_route_feats"][:, 0].tolist()  # rest identical


def test_deterministic_pool_build():
    e1, e2 = _env(), _env()
    assert [len(r) for r in e1.route_sets] == [len(r) for r in e2.route_sets]
    assert np.allclose(e1.obj_matrix, e2.obj_matrix)


def test_exact_estimator_vs_montecarlo():
    from scripts.train_multiod_generalist import joint_dist
    env = _env()
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=32, num_layers=2, heads=2,
                          reward_scale=1.0, device="cpu", role_alpha=True)
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.route_feat_w = torch.nn.Parameter(torch.zeros(2)); net.route_feats = None
    torch.manual_seed(0)
    d_exact = joint_dist(prot, env)
    rng = np.random.default_rng(1)
    N = 9000
    counts = np.zeros(env.n_joint)
    for _ in range(N):
        env.reset(); routes = []
        for f in range(env.F):
            obs = env.observe(); cur = env.current_stream(); mask = env.defender_action_mask()
            a = prot.select_action(obs, mask)[cur]
            env.route_stream_by_index(int(a)); routes.append(int(a))
        counts[env.joint_index(routes)] += 1
    d_mc = counts / N
    R = [len(rs) for rs in env.route_sets]
    # the headline quantity the estimator must get right: exploitability agrees tightly
    M = env.obj_matrix
    assert abs(float((d_exact @ M).max()) - float((d_mc @ M).max())) < 0.02
    # stream-0 marginal (11 bins) agrees; the raw 1210-bin joint L1 is pure sampling noise
    m0_exact = d_exact.reshape(R[0], -1).sum(axis=1)
    m0_mc = d_mc.reshape(R[0], -1).sum(axis=1)
    assert np.abs(m0_exact - m0_mc).sum() < 0.08
