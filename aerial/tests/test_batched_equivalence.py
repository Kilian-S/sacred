"""Numerical-equivalence guard for the batch-encode path in ``SAC.update()``. The batched graph is
a disjoint union with no cross-graph edges, so encoding the whole minibatch in one pass and then
applying each network's head per sample must be identical, in eval mode, to calling ``forward`` on
each graph separately."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from src.agents.networks import ProtagonistPolicyValueNet, AntagonistPolicyValueNet
from src.agents.sac import ProtagonistQNet, AntagonistQNet

NODE_DIM = 9
EDGE_DIM = 2
ATOL = 1e-4
RTOL = 1e-3


def _make_graph(n_nodes: int, n_undirected_edges: int, seed: int) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, NODE_DIM, generator=g)
    src = torch.randint(0, n_nodes, (n_undirected_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_undirected_edges,), generator=g)
    # symmetric (undirected) directed edge index
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    edge_attr = torch.randn(edge_index.size(1), EDGE_DIM, generator=g)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _split(h_all: torch.Tensor, batch: Batch, i: int) -> torch.Tensor:
    return h_all[batch.ptr[i]: batch.ptr[i + 1]]


def test_protagonist_actor_batched_encode_matches_per_sample():
    torch.manual_seed(0)
    net = ProtagonistPolicyValueNet(NODE_DIM, EDGE_DIM, hidden_dim=16, num_layers=2, heads=2).eval()
    graphs = [_make_graph(n, n + 2, seed=i) for i, n in enumerate([5, 7, 4, 9])]
    actives = [0, 3, 1, 5]
    masks = [[1, 2, 4], [0, 1], [0, 2, 3], [1, 4, 7, 8]]

    with torch.no_grad():
        per = [net(g, a, m) for g, a, m in zip(graphs, actives, masks)]
        batch = Batch.from_data_list(graphs)
        h_all = net.encoder(batch.x, batch.edge_index, batch.edge_attr)
        for i, (a, m) in enumerate(zip(actives, masks)):
            probs_b, v_b = net.head(_split(h_all, batch, i), a, m)
            assert torch.allclose(probs_b, per[i][0], atol=ATOL, rtol=RTOL)
            assert torch.allclose(v_b, per[i][1], atol=ATOL, rtol=RTOL)


def test_protagonist_critic_batched_encode_matches_per_sample():
    torch.manual_seed(1)
    net = ProtagonistQNet(NODE_DIM, EDGE_DIM, hidden_dim=16, num_layers=2, heads=2).eval()
    graphs = [_make_graph(n, n + 2, seed=10 + i) for i, n in enumerate([6, 5, 8])]
    actives = [0, 2, 4]
    masks = [[1, 3], [0, 1, 4], [2, 6, 7]]

    with torch.no_grad():
        per = [net(g, a, m) for g, a, m in zip(graphs, actives, masks)]
        batch = Batch.from_data_list(graphs)
        h_all = net.encoder(batch.x, batch.edge_index, batch.edge_attr)
        for i, (a, m) in enumerate(zip(actives, masks)):
            q_b = net.head(_split(h_all, batch, i), a, m)
            assert torch.allclose(q_b, per[i], atol=ATOL, rtol=RTOL)


def test_antagonist_actor_batched_encode_matches_per_sample():
    torch.manual_seed(2)
    net = AntagonistPolicyValueNet(NODE_DIM, EDGE_DIM, hidden_dim=16, num_layers=2, heads=2,
                                   num_congestion_levels=4).eval()
    level_costs = [7.5, 15.0, 22.5, 30.0]
    budgets = [40.0, 100.0, 10.0]
    graphs = [_make_graph(n, n + 2, seed=20 + i) for i, n in enumerate([6, 7, 5])]
    node_to_idx = [{j: j for j in range(g.num_nodes)} for g in graphs]
    allowed = [[(0, 1), (2, 3)], [(1, 4), (0, 2), (3, 5)], [(0, 1)]]

    with torch.no_grad():
        per = [net(g, [], n2i, ae, b, level_costs)
               for g, n2i, ae, b in zip(graphs, node_to_idx, allowed, budgets)]
        batch = Batch.from_data_list(graphs)
        h_all = net.encoder(batch.x, batch.edge_index, batch.edge_attr)
        for i in range(len(graphs)):
            ep_b, lp_b, v_b = net.head(
                _split(h_all, batch, i), graphs[i].edge_index, graphs[i].edge_attr,
                node_to_idx[i], allowed[i], budgets[i], level_costs,
            )
            assert torch.allclose(ep_b, per[i][0], atol=ATOL, rtol=RTOL)
            assert torch.allclose(lp_b, per[i][1], atol=ATOL, rtol=RTOL)
            assert torch.allclose(v_b, per[i][2], atol=ATOL, rtol=RTOL)


def test_antagonist_critic_batched_encode_matches_per_sample():
    torch.manual_seed(3)
    net = AntagonistQNet(NODE_DIM, EDGE_DIM, hidden_dim=16, num_layers=2, heads=2,
                         num_congestion_levels=4).eval()
    level_costs = [7.5, 15.0, 22.5, 30.0]
    budgets = [40.0, 100.0, 10.0]
    graphs = [_make_graph(n, n + 2, seed=30 + i) for i, n in enumerate([6, 7, 5])]
    node_to_idx = [{j: j for j in range(g.num_nodes)} for g in graphs]
    allowed = [[(0, 1), (2, 3)], [(1, 4), (0, 2), (3, 5)], [(0, 1)]]

    with torch.no_grad():
        per = [net(g, [], n2i, ae, b, level_costs)
               for g, n2i, ae, b in zip(graphs, node_to_idx, allowed, budgets)]
        batch = Batch.from_data_list(graphs)
        h_all = net.encoder(batch.x, batch.edge_index, batch.edge_attr)
        for i in range(len(graphs)):
            eq_b, lq_b = net.head(
                _split(h_all, batch, i), graphs[i].edge_index, graphs[i].edge_attr,
                node_to_idx[i], allowed[i], budgets[i], level_costs,
            )
            assert torch.allclose(eq_b, per[i][0], atol=ATOL, rtol=RTOL)
            assert torch.allclose(lq_b, per[i][1], atol=ATOL, rtol=RTOL)
