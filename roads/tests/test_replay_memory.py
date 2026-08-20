"""Replay-memory contract tests.

A transition that holds its own deep snapshot of the instance's node and edge dicts, and its own
featurised graph tensors once sampled, makes replay memory scale with graph size times buffer
length, which exhausts memory on large city graphs. These tests pin the two fixes: the
dynamic-generalist observations share one base payload per instance, and the SAC featurisation
cache is shared per instance when the observation carries a ``_graph_key``.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_dyn_generalist import build_obs, prep_instance
from scripts.train_generalist import Instance


@pytest.fixture(scope="module")
def instance():
    it = Instance(("35", "159"), 3, 1, (0.15, 0.95), 8, 0, city="kaliningrad")
    return prep_instance(it, 0.15, 3)


def test_observations_share_one_base_payload(instance):
    """Two transitions of one instance share their node and edge dicts, not their route features."""
    o1 = build_obs(instance, np.zeros(instance.nR), 3)
    o2 = build_obs(instance, np.eye(instance.nR)[0], 3)
    assert o1 is not o2                          # distinct dicts per transition
    assert o1["nodes"] is o2["nodes"]            # ... sharing the heavy payload
    assert o1["edges"] is o2["edges"]
    assert o1["menu_route_feats"] is not o2["menu_route_feats"]
    assert not torch.equal(o1["menu_route_feats"], o2["menu_route_feats"])


def test_window_column_is_per_transition(instance):
    """The shared base must not leak one transition's window features into another."""
    counts = np.zeros(instance.nR)
    counts[0] = 3.0
    o_empty = build_obs(instance, np.zeros(instance.nR), 3)
    o_full = build_obs(instance, counts, 3)
    assert float(o_empty["menu_route_feats"][0, 2]) == 0.0
    assert float(o_full["menu_route_feats"][0, 2]) == 1.0
    # building the second observation must not have mutated the first
    assert float(o_empty["menu_route_feats"][0, 2]) == 0.0


def test_featurize_cache_is_shared_per_instance(instance):
    """With a _graph_key present, two transitions reuse a single featurised graph."""
    from src.agents import sac as sac_mod

    sac_mod._SHARED_GRAPH_CACHE.clear()
    calls = {"n": 0}

    class _T:
        def __init__(self, state):
            self.state = state
            self.feature_cache = {}

    def build():
        calls["n"] += 1
        return object()

    o1 = build_obs(instance, np.zeros(instance.nR), 3)
    o2 = build_obs(instance, np.eye(instance.nR)[0], 3)
    a = sac_mod._cached_featurize(_T(o1), "state", build)
    b = sac_mod._cached_featurize(_T(o2), "state", build)
    assert a is b and calls["n"] == 1
    sac_mod._SHARED_GRAPH_CACHE.clear()


def test_featurize_cache_unchanged_without_graph_key():
    """Absent the key, caching stays per transition."""
    from src.agents import sac as sac_mod

    calls = {"n": 0}

    class _T:
        def __init__(self):
            self.state = {"nodes": {}}
            self.feature_cache = {}

    def build():
        calls["n"] += 1
        return object()

    t1, t2 = _T(), _T()
    a = sac_mod._cached_featurize(t1, "state", build)
    b = sac_mod._cached_featurize(t2, "state", build)
    assert a is not b and calls["n"] == 2
    assert sac_mod._cached_featurize(t1, "state", build) is a and calls["n"] == 2
