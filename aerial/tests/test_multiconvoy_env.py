"""Tests for the multi-convoy interdiction env, including its fidelity against the oracle."""
import numpy as np

from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.multiconvoy_interdiction import make_multiconvoy_env


def _mc_average(env, occ_seq, iset_seq):
    """Drive the env over explicit (occupancy, interdiction-set) sequences; return mean objective."""
    total = 0.0
    for occ, j in zip(occ_seq, iset_seq):
        env.set_convoy_routes(env.occupancy_to_routes(occ))
        env.commit(int(j))
        total += env.resolve().objective_value
    return total / len(occ_seq)


def test_gm1_env_reproduces_loss_mixed():
    """Playing the oracle equilibrium, the env's sampled mission failure reproduces loss_mixed."""
    env = make_multiconvoy_env(od=("110", "135"), N=2, objective="mission", seed=1)
    sol = solve_multiconvoy(env.game, N=2, objective="mission")
    rng = np.random.default_rng(0)
    n = 15000
    occ = [sol.occupancies[i] for i in rng.choice(len(sol.occupancies), n, p=sol.defender_strategy)]
    iset = rng.choice(len(sol.attacker_strategy), n, p=sol.attacker_strategy)
    assert abs(_mc_average(env, occ, iset) - sol.loss_mixed) < 0.025


def test_gm1_env_reproduces_loss_det():
    """The deterministic optimum occupancy against its best-response set reproduces loss_det."""
    env = make_multiconvoy_env(od=("110", "135"), N=2, objective="mission", seed=2)
    sol = solve_multiconvoy(env.game, N=2, objective="mission")
    occs, M = objective_matrix(env.game, 2, "mission")
    det_i = int(np.argmin(M.max(axis=1)))
    j = int(M[det_i].argmax())
    n = 15000
    emp = _mc_average(env, [occs[det_i]] * n, [j] * n)
    assert abs(emp - sol.loss_det) < 0.025


def test_env_n1_matches_single_convoy_interception():
    """N=1 mission = P(caught) = the per-route interception probability of the chosen route/set."""
    env = make_multiconvoy_env(od=("110", "135"), N=1, objective="mission", seed=3)
    r, j = 0, 0
    n = 15000
    emp = _mc_average(env, [(0,) * 0 + tuple(1 if k == r else 0 for k in range(env.game.n_routes))] * n, [j] * n)
    assert abs(emp - float(env.game.payoff[r, j])) < 0.02


def test_sequential_routing_builds_occupancy():
    env = make_multiconvoy_env(od=("110", "135"), N=2, objective="mission")
    env.reset()
    fh = env.first_hops
    env.route_convoy_first_hop(fh[0])
    env.route_convoy_first_hop(fh[1])
    occ = env.defender_occupancy()
    assert sum(occ) == 2
    assert occ[env.route_of_first_hop(fh[0])] >= 1
    assert env.current_convoy() is None
    env.commit(0)
    out = env.resolve()
    assert 0 <= out.n_caught <= 2
    assert out.objective_value in (0.0, 1.0)          # mission is all-or-nothing


def test_env_determinism_same_seed():
    e1 = make_multiconvoy_env(od=("110", "135"), N=2, seed=7)
    e2 = make_multiconvoy_env(od=("110", "135"), N=2, seed=7)
    for _ in range(50):
        e1.set_convoy_routes([0, 1]); e1.commit(0); o1 = e1.resolve()
        e2.set_convoy_routes([0, 1]); e2.commit(0); o2 = e2.resolve()
        assert o1.caught == o2.caught


def test_exploitability_of_occupancy_dist():
    """A deterministic (single-occupancy) defender is exploited at least up to the equilibrium value."""
    env = make_multiconvoy_env(od=("110", "135"), N=2, objective="mission")
    sol = solve_multiconvoy(env.game, N=2, objective="mission")
    d = np.zeros(len(env.occupancies)); d[0] = 1.0
    assert env.exploitability_of_occupancy_dist(d) >= sol.loss_mixed - 1e-9
