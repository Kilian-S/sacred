"""Tests for the exploitability BR-gate machinery: the hybrid greedy per-truck chooser and the
trainer's frozen-protagonist-chooser hook (antagonist trains vs a greedy victim, protagonist net
untouched)."""

from __future__ import annotations

from scripts.evaluate_hybrid import hybrid_config
from src.agents.sac import AntagonistSAC, ProtagonistSAC
from src.agents.sacred_atla import ATLACoevolutionTrainer
from src.baselines.greedy_dispatch import hybrid_greedy_chooser
from src.env.smdp_wrapper import DecisionType, SMDPDecisionWrapper
from src.envs.assignment_factory import make_hybrid_assign_env


def _hybrid_smdp():
    return SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=hybrid_config())


def test_hybrid_greedy_chooser_returns_valid_choices():
    smdp = _hybrid_smdp()
    choose = hybrid_greedy_chooser(smdp)
    event = smdp.reset_decision_env()
    made_a_choice = False
    guard = 0
    while not event.done and guard < 200:
        guard += 1
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            for tid, opts in event.protagonist_action_mask.items():
                if not opts:
                    continue
                out = choose(event.observation, {tid: opts}, tid)
                if out:  # every returned node must be inside the offered mask
                    assert out[tid] in opts
                    made_a_choice = True
            # advance greedily to keep the episode moving
            from src.baselines.greedy_dispatch import hybrid_greedy_policy
            event, _ = smdp.step_protagonist(hybrid_greedy_policy(smdp)(event))
        elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            event, _ = smdp.step_antagonist(None)
        else:
            event = smdp.advance_until_decision()
    assert made_a_choice


def test_frozen_chooser_trains_antagonist_without_touching_protag():
    smdp = _hybrid_smdp()
    cfg = hybrid_config()
    protag = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=16, num_layers=2, heads=2, device="cpu")
    antag = AntagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=16, num_layers=2, heads=2,
                          num_congestion_levels=len(cfg.congestion_levels),
                          level_costs=[l * cfg.congestion_duration for l in cfg.congestion_levels],
                          congestion_levels=cfg.congestion_levels, device="cpu")
    before = len(protag.replay_buffer)
    trainer = ATLACoevolutionTrainer(
        smdp=smdp, protag_agent=protag, antag_agent=antag, switch_every_episodes=1, batch_size=8,
        run_name="_test/br_gate", mode="antagonist_only",
        frozen_protagonist_chooser=hybrid_greedy_chooser(smdp))
    trainer.run_training(total_episodes=1)
    trainer.writer.close()
    # The greedy victim is frozen: the protagonist buffer must stay empty (only the antagonist learns).
    assert len(protag.replay_buffer) == before == 0
    import shutil
    shutil.rmtree("logs/tb_runs/_test", ignore_errors=True)
    shutil.rmtree("models/runs/_test", ignore_errors=True)
