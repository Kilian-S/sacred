"""Regression tests for the 2026-07-09 node-ordering fix.

`featurize_state` orders node rows by sorted(node ids); before the fix, every consumer built its
node->index map from dict INSERTION order, so on any observation whose insertion order differs from
sorted order the policy/critic heads indexed the wrong rows (demonstrated on the Kaliningrad graph:
a convoy at node 62 read node 167's embedding). These tests pin the single-source-of-truth helper
`node_index_map` to featurize_state's actual row order, on both a synthetic adversarial ordering
and the real Kaliningrad interdiction/multi-convoy envs.
"""
from __future__ import annotations

import torch

from src.agents.networks import featurize_state, node_index_map

_IS_ACTIVE_COL = 5      # is_active_truck
_IS_TARGET_COL = 11     # is_active_target


def _synthetic_obs():
    """Observation whose dict INSERTION order ('3', '1', '2') differs from sorted ('1', '2', '3')."""
    nodes = {
        "3": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
        "1": {"x": 1.0, "y": 0.0, "demand": 1.0, "has_depot": False},
        "2": {"x": 0.5, "y": 1.0, "demand": 0.0, "has_depot": False},
    }
    edges = {
        ("3", "2"): {"distance": 1.0, "congestion_level": 0.0},
        ("2", "1"): {"distance": 2.0, "congestion_level": 0.0},
    }
    trucks = {0: {"current_node": "3", "destination": None, "edge": None, "edge_progress": 0.0,
                  "capacity": 1.0, "load": 1.0, "path": ("3",), "path_index": 0,
                  "delivered_total": 0.0, "assigned_target": "1"}}
    return {"time": 0, "nodes": nodes, "edges": edges, "trucks": trucks}


def test_node_index_map_matches_featurize_rows_synthetic():
    obs = _synthetic_obs()
    n2i = node_index_map(obs)
    assert n2i == {"1": 0, "2": 1, "3": 2}  # sorted order, NOT insertion order ('3','1','2')
    x = featurize_state(obs, 0).x
    # the active-truck marker must sit on the row node_index_map assigns to the truck's node...
    assert x[n2i["3"], _IS_ACTIVE_COL] == 1.0
    # ...and NOT on the row insertion order would have assigned (index 0 = first-inserted '3').
    assert x[0, _IS_ACTIVE_COL] == 0.0
    # same for the assigned-target marker.
    assert x[n2i["1"], _IS_TARGET_COL] == 1.0
    assert x[int((x[:, _IS_TARGET_COL] == 1.0).nonzero()), _IS_TARGET_COL] == 1.0
    assert (x[:, _IS_ACTIVE_COL] == 1.0).sum() == 1
    assert (x[:, _IS_TARGET_COL] == 1.0).sum() == 1


def test_node_index_map_matches_featurize_rows_kaliningrad():
    from src.envs.interdiction import make_interdiction_env

    env = make_interdiction_env(od=("33", "71"), K=1, k_extra_routes=0)
    obs = env.reset()
    n2i = node_index_map(obs)
    x = featurize_state(obs, 0).x
    active_rows = (x[:, _IS_ACTIVE_COL] == 1.0).nonzero().flatten().tolist()
    assert active_rows == [n2i[obs["trucks"][0]["current_node"]]]
    target_rows = (x[:, _IS_TARGET_COL] == 1.0).nonzero().flatten().tolist()
    assert target_rows == [n2i["71"]]


def test_menu_route_node_idx_uses_featurize_row_order():
    from src.envs.multiconvoy_interdiction import make_multiconvoy_env

    env = make_multiconvoy_env(od=("62", "97"), N=3, K=1, k_extra_routes=8, menu_select=True)
    obs = env.observe()
    n2i = node_index_map(obs)
    menu = env.menu_route_node_idx()
    for route, idxs in zip(env.game.routes, menu):
        assert idxs == [n2i[str(n)] for n in route if str(n) in n2i]
    # the first node of every route is the base: its row must carry the active-truck marker.
    x = featurize_state(obs, 0).x
    assert all(x[idxs[0], _IS_ACTIVE_COL] == 1.0 for idxs in menu)
