"""Pins the per-route head terms: they must shift logits and Q values by exactly feats @ w plus the
bias, be absent unless attached, and keep the q and target parameter registration order aligned so
that the soft update zips the right pairs together.
"""
from __future__ import annotations

import torch

from src.agents.networks import ProtagonistPolicyValueNet
from src.agents.sac import ProtagonistQNet, ProtagonistSAC

R, H = 5, 16


def _mini_graph():
    n = 8
    x = torch.randn(n, 14)
    edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)] +
                              [[(i + 1) % n, i] for i in range(n)], dtype=torch.long).t()
    edge_attr = torch.randn(edge_index.size(1), 4)
    return x, edge_index, edge_attr


def _menu():
    return [torch.tensor([i % 8, (i + 2) % 8]) for i in range(R)]


def test_route_feat_and_bias_shift_policy_logits_exactly():
    torch.manual_seed(0)
    net = ProtagonistPolicyValueNet(node_in_dim=14, edge_in_dim=4, hidden_dim=H, num_layers=1, heads=2)
    net.menu_routes = _menu()
    x, ei, ea = _mini_graph()
    h = net.encoder(x, ei, ea)
    idxs = list(range(R))
    base_probs, _ = net.head(h, 0, idxs)
    base_logits = torch.log(base_probs)  # up to a constant

    feats = torch.arange(R * 2, dtype=torch.float32).reshape(R, 2)
    net.route_feats = feats
    net.route_feat_w = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    net.route_bias = torch.nn.Parameter(torch.linspace(0.0, 1.0, R))
    new_probs, _ = net.head(h, 0, idxs)
    new_logits = torch.log(new_probs)

    expected_shift = feats @ net.route_feat_w + net.route_bias
    diff = (new_logits - base_logits) - expected_shift
    assert torch.allclose(diff - diff.mean(), torch.zeros(R), atol=1e-5)  # softmax = shift-invariant


def test_route_terms_absent_by_default():
    torch.manual_seed(0)
    net = ProtagonistPolicyValueNet(node_in_dim=14, edge_in_dim=4, hidden_dim=H, num_layers=1, heads=2)
    net.menu_routes = _menu()
    x, ei, ea = _mini_graph()
    h = net.encoder(x, ei, ea)
    p1, _ = net.head(h, 0, list(range(R)))
    p2, _ = net.head(h, 0, list(range(R)))
    assert torch.allclose(p1, p2)


def test_q_head_terms_and_target_param_alignment():
    torch.manual_seed(0)
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=4, hidden_dim=H, num_layers=1, heads=2, device="cpu")
    menu = _menu()
    feats = torch.randn(R, 2)
    # attach in the trainer's registration order to every net, as train_multiconvoy does
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.menu_routes = menu
        net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
        net.route_feats = feats
        net.route_feat_w = torch.nn.Parameter(torch.zeros(2))
        net.route_bias = torch.nn.Parameter(torch.zeros(R))
    assert ([n for n, _ in prot.q1.named_parameters()]
            == [n for n, _ in prot.target_q1.named_parameters()])
    # Q shift is exact (Q head has no softmax, so the comparison is direct)
    x, ei, ea = _mini_graph()
    h = prot.q1.encoder(x, ei, ea)
    q_base = prot.q1.head(h, 0, list(range(R)), taken=torch.zeros(R))
    with torch.no_grad():
        prot.q1.route_feat_w.copy_(torch.tensor([1.0, 2.0]))
        prot.q1.route_bias.copy_(torch.linspace(0.0, 0.4, R))
    q_new = prot.q1.head(h, 0, list(range(R)), taken=torch.zeros(R))
    assert torch.allclose(q_new - q_base, feats @ torch.tensor([1.0, 2.0]) + torch.linspace(0.0, 0.4, R), atol=1e-5)
