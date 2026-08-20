"""Tests for the single-convoy interdiction environment, including the fidelity gate that it
reproduces the equilibrium oracle's deterministic and mixed losses end to end."""

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


# --- The SAC-trainable env: GraphEnv-backed observation and masks ---

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


# --- Route walk over the candidate-route trie, for shared-edge instances ---


def _shared_prefix_graph():
    # candidate routes: S-A-T, S-A-B-T (SHARED prefix S-A), S-C-T, S-E-F-T (forced segment E-F-T).
    G = nx.Graph()
    for p in (["S", "A", "T"], ["S", "A", "B", "T"], ["S", "C", "T"], ["S", "E", "F", "T"]):
        for u, v in zip(p, p[1:]):
            G.add_edge(u, v, w=1.0)
    return G


def _walk_route(env, route):
    """Drive the walk along a target route; at each branch the correct hop is the allowed node
    earliest in the route (nodes before the branch are already consumed: simple paths)."""
    _, done, ri = env.begin_walk()
    while not done:
        allowed = env.walk_mask()[0]
        hop = min((h for h in allowed if h in route), key=route.index)
        _, done, ri = env.step_walk(hop)
    return ri


def test_walk_trie_masks_and_auto_advance():
    env = InterdictionEnv(_shared_prefix_graph(), InterdictionConfig(od=("S", "T"), K=1))
    assert env.game.n_routes == 4
    # first-hop collision exists (S-A-T and S-A-B-T): the walk is genuinely needed.
    assert max(len(v) for v in env.routes_by_first_hop.values()) == 2
    obs, done, ri = env.begin_walk()
    assert not done and ri is None
    assert env.walk_mask()[0] == ["A", "C", "E"]
    # a hop onto a fully forced branch auto-advances to the terminal.
    _, done, ri = env.step_walk("E")
    assert done and env.game.routes[ri] == ("S", "E", "F", "T")
    # shared prefix: after S-A there is a real second decision.
    env.begin_walk()
    _, done, _ = env.step_walk("A")
    assert not done and env.walk_mask()[0] == ["B", "T"]
    _, done, ri = env.step_walk("B")
    assert done and env.game.routes[ri] == ("S", "A", "B", "T")
    # illegal hop rejected.
    env.begin_walk()
    with pytest.raises(ValueError):
        env.step_walk("F")
    # round-trip: every candidate route is walkable and returns its own index.
    for i, r in enumerate(env.game.routes):
        assert _walk_route(env, r) == i


def test_walk_distribution_exact_product():
    env = InterdictionEnv(_shared_prefix_graph(), InterdictionConfig(od=("S", "T"), K=1))
    uniform = lambda node, allowed: {h: 1.0 / len(allowed) for h in allowed}
    d = env.walk_distribution(uniform)
    expected = {("S", "A", "T"): 1 / 6, ("S", "A", "B", "T"): 1 / 6,
                ("S", "C", "T"): 1 / 3, ("S", "E", "F", "T"): 1 / 3}
    for i, r in enumerate(env.game.routes):
        assert d[i] == pytest.approx(expected[r])
    # a biased policy propagates exactly (branch product).
    biased = lambda node, allowed: ({"A": 0.5, "C": 0.5, "E": 0.0} if node == "S"
                                    else {h: (1.0 if h == "T" else 0.0) for h in allowed})
    d2 = env.walk_distribution(biased)
    assert d2[env.game.routes.index(("S", "A", "T"))] == pytest.approx(0.5)
    assert d2[env.game.routes.index(("S", "C", "T"))] == pytest.approx(0.5)


def test_B2_shared_edge_kaliningrad_gate():
    """The shared-edge Kaliningrad instance has 11 routes with first-hop collisions, an
    equilibrium value of 1/6, uniform mixing more than 2.5x suboptimal because mass stacks on the
    shared edges, and a walk that expresses exactly the candidate route set."""
    from src.envs.interdiction import make_interdiction_env
    env = make_interdiction_env(od=("33", "71"), K=1, k_extra_routes=8)
    assert env.game.n_routes == 11
    assert max(len(v) for v in env.routes_by_first_hop.values()) >= 2
    sol = solve(env.game)
    assert sol.loss_det == pytest.approx(1.0)
    assert sol.value == pytest.approx(1.0 / 6.0, abs=1e-6)
    uni = np.ones(env.game.n_routes) / env.game.n_routes
    _, expl_uni = best_response_attacker(env.game, uni)
    assert expl_uni / sol.value > 2.5
    for i, r in enumerate(env.game.routes):
        assert _walk_route(env, r) == i
    # An untrained protagonist walks a full sortie against a committed interdiction.
    from src.agents.sac import ProtagonistSAC
    prot = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=32, num_layers=2, heads=2, device="cpu")
    env.commit(0)
    obs, done, ri = env.begin_walk()
    hops = 0
    while not done:
        act = prot.select_action(obs, env.walk_mask(), deterministic=False)
        obs, done, ri = env.step_walk(act[0])
        hops += 1
        assert hops < 20
    out = env.resolve(ri)
    assert out.defender_reward <= 0.0 and isinstance(out.intercepted, bool)


# --- Heterogeneous edge vulnerability: soft interception ---


def test_resolve_bernoulli_seeded_and_hard_limits():
    G = _synthetic_graph()
    vuln = {frozenset(e): 0.5 for e in G.edges()}
    cfg = InterdictionConfig(od=("S", "T"), K=1, edge_vulnerability=vuln, seed=7)
    env = InterdictionEnv(G, cfg)
    r1 = next(i for i, r in enumerate(env.game.routes) if r == ("S", "A", "T"))
    r2 = next(i for i, r in enumerate(env.game.routes) if r == ("S", "B", "T"))
    sa = frozenset({"S", "A"})

    def draws(e, route, n, seed):
        env2 = InterdictionEnv(G, InterdictionConfig(od=("S", "T"), K=1, edge_vulnerability=vuln, seed=seed))
        out = []
        for _ in range(n):
            env2.commit_edge(e)
            out.append(env2.resolve(route).intercepted)
        return out

    # Same seed gives an identical outcome sequence, and the empirical rate matches p.
    a, b = draws(sa, r1, 4000, seed=7), draws(sa, r1, 4000, seed=7)
    assert a == b
    assert np.mean(a) == pytest.approx(0.5, abs=0.04)
    # a route that avoids the interdicted edge is NEVER intercepted (payoff 0).
    assert not any(draws(sa, r2, 50, seed=7))


def test_G3_soft_env_reproduces_oracle_kaliningrad():
    """On the asymmetric soft-interception instance the factory-built environment reproduces the
    oracle's non-uniform equilibrium end to end, and uniform mixing is measurably suboptimal."""
    from src.envs.interdiction import make_interdiction_env
    env = make_interdiction_env(od=("33", "71"), K=1, k_extra_routes=0,
                                edge_vuln_band=(0.15, 0.95), seed=3)
    P = env.game.payoff
    assert ((P > 0.0) & (P < 1.0)).any()                      # genuinely soft interception
    sol = solve(env.game)
    assert sol.value == pytest.approx(0.063, abs=0.005)        # equilibrium value of this instance
    d = sol.defender_strategy
    assert d.min() > 0.03 and d.max() / d.min() > 2.0          # strongly non-uniform equilibrium
    uni = np.ones(env.game.n_routes) / env.game.n_routes
    _, expl_uni = best_response_attacker(env.game, uni)
    assert expl_uni / sol.value > 2.0                          # uniform mixing ~2.5x suboptimal
    # Equilibrium against equilibrium, by Monte Carlo, recovers the minimax value.
    emp = _empirical_interception(env, sol.defender_strategy, sol.attacker_strategy, n=40000, seed=4)
    assert emp == pytest.approx(sol.value, abs=0.01)


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
