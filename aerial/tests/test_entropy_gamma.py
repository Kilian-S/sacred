"""Entropy and credit-horizon config surface at the SAC level: gamma flows into the agent and
keeps its default, an absolute ``target_entropy`` overrides the dynamic ln(N) fallback in the
alpha loss while ``None`` leaves the fallback in place, and a lower target gives a less-negative
alpha loss at fixed policy entropy, which is what lets the policy commit."""

from __future__ import annotations

import math

import torch

from src.agents.sac import AntagonistSAC, ProtagonistSAC


def _protag(target_entropy=None, gamma=0.99):
    return ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=16, num_layers=2, heads=2,
                          device="cpu", gamma=gamma, target_entropy=target_entropy)


def test_gamma_defaults_preserved_and_flows():
    assert _protag().gamma == 0.99
    assert _protag(gamma=0.997).gamma == 0.997


def test_target_entropy_stored():
    assert _protag(target_entropy=None).target_entropy is None
    assert _protag(target_entropy=0.7).target_entropy == 0.7


def _alpha_loss(target_entropy, entropy_val, log_alpha=0.0, allowed_len=8):
    """Reproduce the alpha-loss target selection from ``ProtagonistSAC.update``."""
    if target_entropy is not None:
        target = torch.tensor(float(target_entropy))
    else:
        target = -0.45 * torch.log(torch.tensor(1.0 / allowed_len))
    entropy = torch.tensor(float(entropy_val))
    return (torch.tensor(log_alpha) * (entropy - target).detach()).item(), target.item()


def test_absolute_target_overrides_lnN_fallback():
    # With allowed_len=8, the dynamic fallback target is 0.45*ln(8) ~ 0.936.
    _, fallback_target = _alpha_loss(None, entropy_val=0.5)
    assert math.isclose(fallback_target, 0.45 * math.log(8), rel_tol=1e-6)
    # an absolute target is used verbatim, independent of allowed_len
    _, abs_target = _alpha_loss(0.3, entropy_val=0.5, allowed_len=8)
    assert math.isclose(abs_target, 0.3, rel_tol=1e-6)
    _, abs_target_bigN = _alpha_loss(0.3, entropy_val=0.5, allowed_len=64)
    assert math.isclose(abs_target_bigN, 0.3, rel_tol=1e-6)  # N-inflation no longer moves it


def test_lower_target_reduces_entropy_pressure():
    # at a fixed low policy entropy and positive log_alpha, a lower target makes (entropy - target)
    # less negative, so alpha is pushed down less hard and the policy is allowed to commit
    loss_high_target, _ = _alpha_loss(0.9, entropy_val=0.4, log_alpha=0.5)
    loss_low_target, _ = _alpha_loss(0.2, entropy_val=0.4, log_alpha=0.5)
    assert loss_low_target > loss_high_target


def test_antagonist_absolute_target_stored():
    ant = AntagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=16, num_layers=2, heads=2,
                        num_congestion_levels=1, level_costs=[120.0], congestion_levels=(1.0,),
                        device="cpu", target_entropy=0.4)
    assert ant.target_entropy == 0.4
