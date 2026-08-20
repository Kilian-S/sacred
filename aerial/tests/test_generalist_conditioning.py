"""Pins per-transition instance conditioning for the generalist: the env attaches each instance's
menu, per-route features and edge vulnerability to every observation, and select_action honours the
menu carried on the observation rather than whichever instance the nets were last pointed at.
"""
from __future__ import annotations

import torch

from src.agents.networks import EDGE_FEATURE_DIM, featurize_state
from src.agents.sac import ProtagonistSAC
from src.envs.multiconvoy_interdiction import make_multiconvoy_env


def _env(od):
    return make_multiconvoy_env(od=od, N=3, K=1, k_extra_routes=8, menu_select=True,
                                edge_vuln_band=(0.15, 0.95), interception_loss=10.0, seed=0)


def test_env_attaches_instance_conditioning():
    env = _env(("62", "97"))
    env.reset()
    obs = env.observe()
    assert "menu_route_node_idx" in obs and len(obs["menu_route_node_idx"]) == env.game.n_routes
    assert obs["menu_route_feats"].shape == (env.game.n_routes, 2)
    assert (obs["menu_route_feats"] >= 0).all() and (obs["menu_route_feats"] <= 1).all()
    assert "edge_vulnerability" in obs and len(obs["edge_vulnerability"]) > 0
    # vulnerability rides in edge_attr column 4
    x = featurize_state(obs, 0)
    assert x.edge_attr.shape[1] == EDGE_FEATURE_DIM == 5
    assert float(x.edge_attr[:, 4].abs().max()) > 0.0


def test_two_instances_have_different_conditioning():
    o1 = _env(("62", "97")); o1.reset()
    o2 = _env(("35", "159")); o2.reset()
    f1 = o1.observe()["menu_route_feats"]
    f2 = o2.observe()["menu_route_feats"]
    assert f1.shape[0] == f2.shape[0]  # k_extra_routes=8 gives 12 routes on both
    assert not torch.allclose(f1, f2)


def test_select_action_uses_on_observation_menu_not_stale_attribute():
    env = _env(("62", "97")); env.reset()
    obs = env.observe()
    R = env.game.n_routes
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=32, num_layers=2, heads=2,
                          device="cpu")
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2))
    # a stale, reversed menu that select_action must overwrite from the observation
    prot.actor.menu_routes = list(reversed(obs["menu_route_node_idx"]))
    prot.actor.route_feats = torch.zeros(R, 2)
    mask = env.defender_action_mask()
    act = prot.select_action(obs, mask)
    ci = env.current_convoy()
    assert ci in act and act[ci] in mask[ci]
    assert all(torch.equal(a, b) for a, b in zip(prot.actor.menu_routes, obs["menu_route_node_idx"]))
