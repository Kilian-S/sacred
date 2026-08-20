"""Exact-chain machinery: agreement between the chain value and a long rollout, the ordering of
the dynamic yardsticks, and the feature contract."""
import numpy as np
import pytest

from scripts.train_aerial_dyn import DynInstance, BASE


def test_dyn_instance_yardsticks_and_chain():
    inst = DynInstance("t", BASE, 1.6, 4242)
    assert 0.0 < inst.hist_opt < inst.iid_eq < 1.0
    assert inst.hist_opt <= inst.bar + 1e-9
    assert inst.bar <= max(inst.iid_eq, inst.static_opt) + 1e-9
    assert inst.static_opt <= inst.iid_eq + 1e-9              # local search never worse than eq-mix
    # a static rule's chain value equals its product-measure static value
    d = np.full(inst.R, 1.0 / inst.R)
    assert inst.chain_value(lambda w: d) == pytest.approx(inst.static_value(d), abs=1e-10)
    # chain vs stochastic rollout agreement on a window-dependent rule
    def anti(w):
        m = np.ones(inst.R)
        for r in set(w):
            m[r] = 0.0
        return m / m.sum()
    exact = inst.chain_value(anti)
    rng = np.random.default_rng(0)
    win = (0, 1)
    acc = 0.0
    T, burn = 30000, 2000
    n = 0
    for t in range(T):
        dd = anti(win)
        r = int(rng.choice(inst.R, p=dd))
        if t >= burn:
            acc += float(inst.stepdmg[inst.widx[win]] @ dd)
            n += 1
        win = (win[1], r)
    assert exact == pytest.approx(acc / n, abs=0.01)
    f = inst.feats((3, 7))
    assert f.shape == (inst.R, 2) and f[:, 1].sum() == pytest.approx(1.0)
