"""A buffered GraphEnv observation must be insensitive to later env mutation, so observe() has to
snapshot the node and edge sub-dicts rather than share references."""
from __future__ import annotations

from src.envs.interdiction import make_interdiction_env


def test_buffered_observation_is_insensitive_to_later_mutation():
    env = make_interdiction_env(od=("33", "71"), K=1, k_extra_routes=0)
    ge = env.graph_env
    obs = ge.observe()
    a_node = next(iter(obs["nodes"]))
    an_edge = next(iter(obs["edges"]))
    d0 = obs["nodes"][a_node]["demand"]
    c0 = obs["edges"][an_edge]["congestion_level"]

    # mutate the LIVE env state after capturing obs
    ge._obs_nodes[a_node]["demand"] = d0 + 7.0
    ge._obs_edges[an_edge]["congestion_level"] = 1.0

    # the previously captured observation must be unchanged (snapshot, not shared reference)
    assert obs["nodes"][a_node]["demand"] == d0
    assert obs["edges"][an_edge]["congestion_level"] == c0

    # and a FRESH observe() does reflect the new state (the snapshot is per-call, not frozen)
    obs2 = ge.observe()
    assert obs2["nodes"][a_node]["demand"] == d0 + 7.0
    assert obs2["edges"][an_edge]["congestion_level"] == 1.0
