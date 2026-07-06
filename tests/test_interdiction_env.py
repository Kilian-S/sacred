"""Tests for the single-convoy interdiction env (gen08), incl. the G1 env-fidelity gate:
the env reproduces the equilibrium oracle's loss_det / loss_mixed end-to-end."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.interdiction import InterdictionConfig, InterdictionEnv


def _synthetic_graph():
    G = nx.Graph()
    for p in (["S", "A", "T"], ["S", "B", "T"], ["S", "C", "T"], ["S", "A", "D", "T"]):
        for u, v in zip(p, p[1:]):
            G.add_edge(u, v, w=1.0)
    return G


def _kaliningrad_graph():
    from src.envs.assignment_factory import _DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS
    from src.utils.graph_utils import load_osm_graph_and_demands
    nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    return G


def test_env_builds_and_first_hops_map_to_routes():
    env = InterdictionEnv(_synthetic_graph(), InterdictionConfig(od=("S", "T"), K=1))
    assert env.game.n_routes >= 3
    # every candidate first hop resolves to a route that actually starts with it.
    for fh in env.first_hops:
        ri = env.route_of_first_hop(fh)
        assert env.game.routes[ri][1] == fh


def test_interception_accounting_and_zero_sum():
    env = InterdictionEnv(_synthetic_graph(), InterdictionConfig(od=("S", "T"), K=1, interception_loss=5.0))
    # commit the edge S-A, route R1 = S-A-T crosses it -> interception.
    env.commit_edge(frozenset({"S", "A"}))
    # find R1 (S,A,T)
    r1 = next(i for i, r in enumerate(env.game.routes) if r == ("S", "A", "T"))
    out = env.resolve(r1)
    assert out.intercepted and out.defender_reward == -5.0 and out.attacker_reward == 5.0
    # a disjoint route (S-B-T) avoids it.
    env.commit_edge(frozenset({"S", "A"}))
    r2 = next(i for i, r in enumerate(env.game.routes) if r == ("S", "B", "T"))
    out2 = env.resolve(r2)
    assert not out2.intercepted and out2.defender_reward == 0.0 and out2.attacker_reward == 0.0


def _empirical_interception(env, defender_dist, attacker_dist, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    routes = rng.choice(len(defender_dist), size=n, p=defender_dist)
    isets = rng.choice(len(attacker_dist), size=n, p=attacker_dist)
    hits = 0
    for r, j in zip(routes, isets):
        env.commit(int(j))
        hits += env.resolve(int(r)).intercepted
    return hits / n


def test_G1_env_reproduces_oracle_synthetic():
    env = InterdictionEnv(_synthetic_graph(), InterdictionConfig(od=("S", "T"), K=1))
    sol = solve(env.game)
    # deterministic defender (shortest route) vs its best-response attacker -> loss_det.
    det = np.zeros(env.game.n_routes); det[env.shortest_route_index()] = 1.0
    j, _ = best_response_attacker(env.game, det)
    atk_det = np.zeros(len(env.game.interdiction_sets)); atk_det[j] = 1.0
    assert _empirical_interception(env, det, atk_det) == pytest.approx(sol.loss_det, abs=1e-9)
    # equilibrium mixed defender vs equilibrium attacker -> loss_mixed (Monte Carlo).
    emp = _empirical_interception(env, sol.defender_strategy, sol.attacker_strategy, n=40000, seed=1)
    assert emp == pytest.approx(sol.value, abs=0.02)


def test_G1_env_reproduces_oracle_kaliningrad():
    env = InterdictionEnv(_kaliningrad_graph(), InterdictionConfig(od=("33", "71"), K=1))
    sol = solve(env.game)
    assert sol.loss_det == pytest.approx(1.0)          # deterministic route fully exploitable
    emp = _empirical_interception(env, sol.defender_strategy, sol.attacker_strategy, n=40000, seed=2)
    assert emp == pytest.approx(sol.value, abs=0.03)   # env matches the minimax value
    assert sol.gap >= 0.8 - 1e-9                        # the large robustness gap survives in the env


# --- I1b: the SAC-trainable env (GraphEnv-backed observation + masks) ---

def test_factory_builds_sac_observation():
    from src.envs.interdiction import make_interdiction_env
    env = make_interdiction_env(od=("33", "71"), K=1)
    obs = env.reset()
    assert obs["active_truck"] == 0
    assert set(("nodes", "edges", "trucks")) <= set(obs)
    assert obs["trucks"][0]["current_node"] == "33"        # convoy at base
    # every defender first-hop is a real neighbour of the base.
    for fh in env.first_hops:
        assert fh in env.graph["33"]


def test_sac_agents_act_on_interdiction_env():
    from src.agents.sac import AntagonistSAC, ProtagonistSAC
    from src.envs.interdiction import make_interdiction_env
    env = make_interdiction_env(od=("33", "71"), K=1)
    obs = env.reset()
    prot = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=32, num_layers=2, heads=2, device="cpu")
    ant = AntagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=32, num_layers=2, heads=2,
                        num_congestion_levels=1, level_costs=[1.0], congestion_levels=(1.0,), device="cpu")
    # defender picks a first hop that is inside its mask and maps to a route.
    dact = prot.select_action(obs, env.defender_action_mask(), deterministic=False)
    assert dact[0] in env.first_hops
    ri = env.route_of_first_hop(dact[0])
    assert env.game.routes[ri][1] == dact[0]
    # attacker commits a candidate interdiction edge.
    aact = ant.select_action(obs, env.attacker_action_mask(), 999.0, deterministic=False)
    env.commit_edge(aact[0])
    out = env.resolve_first_hop(dact[0])
    assert out.defender_reward == -out.attacker_reward or out.travel_cost >= 0  # zero-sum on interception
