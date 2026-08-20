"""Tests the contested ERB demo path: greedy no-attack transitions are collectable and load into
a protagonist replay buffer, as the ``--erb-path`` contract requires, without file I/O."""

from __future__ import annotations

from scripts.generate_erb_assign import greedy_choose_fn
from src.agents.sac import ProtagonistSAC
from src.agents.transition_builder import collect_protagonist_transitions
from src.env.smdp_wrapper import DecisionType, SMDPDecisionWrapper
from src.envs.contested import contested_config, make_contested_env


def test_contested_greedy_demos_collect_and_buffer():
    cfg = contested_config()
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=0.06, demand_seed=777), config=cfg)
    choose = greedy_choose_fn(smdp)

    transitions = []
    event = smdp.reset_decision_env()
    guard = 0
    while not event.done and guard < 5000:
        guard += 1
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            event, ts = collect_protagonist_transitions(smdp, event, choose)
            transitions.extend(ts)
        elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            event, _ = smdp.step_antagonist(None)
        else:
            event = smdp.advance_until_decision()

    assert transitions, "greedy produced no protagonist transitions"
    assert all(t.agent == "protagonist" for t in transitions)

    # They must load into a protagonist buffer exactly as --erb-path does.
    agent = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=16, num_layers=2, heads=2,
                           device="cpu")
    for t in transitions:
        agent.replay_buffer.push(t)
    assert len(agent.replay_buffer) == len(transitions)
