"""Alternating Training with Learned Adversaries (ATLA) trainer for SACRED.

This class manages the coevolutionary minimax game, alternating training phases
between the fleet dispatcher (Protagonist) and the bottleneck congestion creator
(Antagonist) while writing real-time metrics to TensorBoard.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.env.smdp_wrapper import CongestionBudget, DecisionType, SMDPDecisionWrapper, SMDPTransition
from src.agents.sac import AntagonistSAC, ProtagonistSAC
from src.agents.transition_builder import collect_protagonist_transitions


class ATLACoevolutionTrainer:
    """Trainer orchestrating the alternating minimax training of both agents.

    Args:
        switch_every_episodes: episodes to train one agent for before freezing it and training
            the other.
        eval_fn: optional snapshot evaluation called as ``eval_fn(episode)``, its scalars logged
            under ``Eval/*``.
    """

    def __init__(
        self,
        smdp: SMDPDecisionWrapper,
        protag_agent: ProtagonistSAC,
        antag_agent: AntagonistSAC,
        log_dir: str = "logs/tb_runs",
        switch_every_episodes: int = 10,
        batch_size: int = 64,
        run_name: str | None = None,
        eval_fn: Callable[[int], dict] | None = None,
        eval_every: int = 0,
        mode: str = "atla",
        scripted_attacker: Callable | None = None,
        update_every: int = 1,
        attack_curriculum=None,
        scripted_attacker_pool=None,
        frozen_protagonist_chooser=None,
    ) -> None:
        if mode not in ("atla", "vanilla", "antagonist_only", "scripted_adversary"):
            raise ValueError(f"unknown trainer mode {mode!r}")
        if mode == "scripted_adversary" and scripted_attacker is None:
            raise ValueError("mode='scripted_adversary' requires a scripted_attacker callable")
        self.smdp = smdp
        self.protag = protag_agent
        self.antag = antag_agent
        self.switch_every_episodes = switch_every_episodes
        self.batch_size = batch_size
        # Training modes. "atla" plays the coevolutionary alternating game. "vanilla" is the
        # non-adversarial control: the protagonist trains every episode and the antagonist is
        # inert, with everything else identical. "antagonist_only" freezes the protagonist and
        # trains the antagonist alone, which builds the per-policy worst-case attack used for
        # evaluation. "scripted_adversary" trains the protagonist against a fixed scripted
        # attacker, leaving the antagonist nets unused.
        self.mode = mode
        self.scripted_attacker = scripted_attacker
        # Optional attack exposure/strength curriculum (scripted_adversary only). When set it
        # decides per episode whether the attack fires and at what budget, ramping difficulty only
        # while the defender stays competent.
        self.attack_curriculum = attack_curriculum
        self._episode_attacked = True  # overwritten each episode when a curriculum is set
        # Optional adversary population (scripted_adversary only): a ScriptedAttackerMixture whose
        # member is resampled each episode, overriding self.scripted_attacker for that episode.
        # Composes with the curriculum.
        self.scripted_attacker_pool = scripted_attacker_pool
        # antagonist_only: drive the frozen protagonist with this per-truck chooser instead of the
        # SAC net, so a best-response attacker can be trained against a competent deterministic
        # victim.
        self.frozen_protagonist_chooser = frozen_protagonist_chooser
        # Update-to-data ratio: one gradient update every N decision epochs per agent rather than
        # every epoch, trading updates-per-experience for wall-clock on registers that make many
        # cheap decisions. Applied identically to every arm of a generation for fairness.
        self.update_every = max(1, int(update_every))
        self._protag_decision_count = 0
        self._antag_decision_count = 0
        self.eval_fn = eval_fn
        self.eval_every = eval_every

        if run_name is None:
            run_name = f"sacred_atla_sw{switch_every_episodes}_b{batch_size}"
        self.run_name = run_name

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, self.run_name))

        # Non-ATLA modes pin the phase for the whole run, so the switch-every cadence then only
        # drives checkpoint and snapshot saving.
        self.current_phase: str = "antagonist" if mode == "antagonist_only" else "protagonist"
        self.episode_count = 0
        self.step_count = 0

    def run_training(self, total_episodes: int, start_episode: int = 0) -> None:
        """Run the coevolutionary training loop for total_episodes."""
        print(f"Starting SACRED training (mode={self.mode}) for {total_episodes} episodes...")
        print(f"Device: Protagonist={self.protag.device}, Antagonist={self.antag.device}")
        if self.mode == "atla":
            print(f"Alternating every {self.switch_every_episodes} episodes.")
        print(f"Initial Phase: Training {self.current_phase.upper()}\n")

        for ep in range(start_episode + 1, total_episodes + 1):
            self.episode_count = ep

            # 1. Check training phase switch
            if (ep - 1) > 0 and (ep - 1) % self.switch_every_episodes == 0:
                print("Saving checkpoints for both agents before phase switch...")
                # Run-specific directory, so other runs cannot clobber this run's resumable state.
                # Resume with: --resume-checkpoint models/runs/<run_name>
                ckpt_dir = os.path.join("models", "runs", self.run_name)
                self.protag.save_checkpoint(os.path.join(ckpt_dir, "protagonist", "checkpoint.pt"), ep - 1)
                self.antag.save_checkpoint(os.path.join(ckpt_dir, "antagonist", "checkpoint.pt"), ep - 1)
                # Per-phase actor snapshots, never overwritten, so best-checkpoint selection stays
                # possible post-hoc; the final checkpoint is misleading under co-evolution.
                snap_dir = os.path.join(ckpt_dir, "snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                torch.save(self.protag.actor.state_dict(), os.path.join(snap_dir, f"protagonist_ep{ep - 1}.pt"))
                torch.save(self.antag.actor.state_dict(), os.path.join(snap_dir, f"antagonist_ep{ep - 1}.pt"))
                if self.mode == "atla":
                    self.current_phase = "antagonist" if self.current_phase == "protagonist" else "protagonist"
                    print(f"\n--- Episode {ep}: Switching training phase to {self.current_phase.upper()} ---")

            # 2. Reset environment and tracking accumulators
            event = self.smdp.reset_decision_env()
            # The curriculum decides this episode's attack on/off and budget, overriding the config
            # budget the reset just installed. Clean episodes keep viable-state experience in the
            # replay, and the budget ramps only as competence is earned.
            if self.attack_curriculum is not None:
                self._episode_attacked, ep_budget = self.attack_curriculum.decide()
                self.smdp.budget = CongestionBudget(ep_budget)
            # Resample this episode's attacker from the population, if one is configured.
            self._episode_attacker_name = None
            if self.scripted_attacker_pool is not None:
                self._episode_attacker_name, self.scripted_attacker = self.scripted_attacker_pool.sample()
            ep_protag_reward = 0.0
            ep_antag_reward = 0.0
            ep_ticks = 0

            protag_losses = []
            antag_losses = []

            # 3. Episode loop
            while not event.done:
                self.step_count += 1
                dt = event.elapsed_ticks
                ep_ticks += dt

                # A. PROTAGONIST DECISION EPOCH
                if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                    # Sequential per-truck decisions with projection and claiming live in the
                    # shared transition builder, which the demo generator also uses so that demos
                    # are byte-identical to live transitions.
                    if self.frozen_protagonist_chooser is not None:
                        _choose = self.frozen_protagonist_chooser  # e.g. a greedy victim
                    else:
                        def _choose(projected_obs, truck_mask, truck_id):
                            return self.protag.select_action(projected_obs, truck_mask, deterministic=False)

                    next_event, t_transitions = collect_protagonist_transitions(self.smdp, event, _choose)
                    if self.mode != "antagonist_only":  # a frozen protagonist stores nothing
                        for t_trans in t_transitions:
                            self.protag.replay_buffer.push(t_trans)

                    # Reward bookkeeping (per-truck transitions all carry the same interval reward).
                    interval_reward = t_transitions[0].reward if t_transitions else 0.0
                    ep_protag_reward += interval_reward
                    ep_antag_reward += next_event.antagonist_reward

                    self._protag_decision_count += 1
                    if (self.current_phase == "protagonist"
                            and self._protag_decision_count % self.update_every == 0):
                        metrics = self.protag.update(self.batch_size)
                        if metrics:
                            protag_losses.append(metrics)

                    event = next_event

                # B. ANTAGONIST DECISION EPOCH
                elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                    if self.mode in ("vanilla", "scripted_adversary"):
                        # Under vanilla the adversary never acts; under scripted_adversary a fixed
                        # scripted attacker does. Either way there is no antagonist net forward
                        # pass, storage or update, so only the protagonist learns. On a curriculum
                        # clean episode the scripted attacker is held back for that episode.
                        attack_on = self.mode == "scripted_adversary" and self._episode_attacked
                        action = self.scripted_attacker(event) if attack_on else None
                        next_event, _ = self.smdp.step_antagonist(action)
                        ep_protag_reward += next_event.protagonist_reward
                        ep_antag_reward += next_event.antagonist_reward
                        event = next_event
                        continue
                    mask = event.antagonist_action_mask
                    remaining_budget_before = self.smdp.budget.remaining
                    action = self.antag.select_action(
                        event.observation, mask, remaining_budget_before, deterministic=False
                    )

                    next_event, transition = self.smdp.step_antagonist(action)
                    
                    # The antagonist update needs the budget either side of the step and the edge
                    # sets its mask allowed, neither of which the raw transition carries.
                    transition.info["antagonist_budget_remaining"] = remaining_budget_before
                    transition.info["next_antagonist_budget_remaining"] = self.smdp.budget.remaining
                    
                    state_copy = dict(event.observation)
                    state_copy["allowed_destinations"] = {
                        "antagonist": {
                            "allowed_edges": list(mask.get("levels_by_edge", {}).keys()),
                            "original_edges": list(event.observation["edges"].keys())
                        }
                    }
                    
                    next_state_copy = dict(next_event.observation)
                    next_state_copy["allowed_destinations"] = {
                        "antagonist": {
                            "allowed_edges": list(next_event.antagonist_action_mask.get("levels_by_edge", {}).keys()),
                            "original_edges": list(next_event.observation["edges"].keys())
                        }
                    }
                    
                    transition.state = state_copy
                    transition.next_state = next_state_copy
                    
                    self.antag.replay_buffer.push(transition)

                    ep_antag_reward += transition.reward
                    ep_protag_reward += next_event.protagonist_reward

                    self._antag_decision_count += 1
                    if (self.current_phase == "antagonist"
                            and self._antag_decision_count % self.update_every == 0):
                        metrics = self.antag.update(self.batch_size)
                        if metrics:
                            antag_losses.append(metrics)

                    event = next_event

                else:
                    event = self.smdp.advance_until_decision()

            # 4. Extract end-of-episode simulator stats
            env = self.smdp.env
            is_dynamic = getattr(env, "_dynamic_demand", False)
            if is_dynamic:
                # Demand arrives over time, so _initial_graph carries none. Count units that
                # actually entered the system = delivered + still-queued at the horizon.
                delivered = float(len(env._delivered_latencies))
                initial_demands = delivered + float(env.remaining_demand)
                delivery_rate = delivered / max(1e-6, initial_demands)
            else:
                env_state = env.observe()
                total_demands = sum(n["demand"] for n in env_state["nodes"].values())
                initial_demands = sum(
                    env._initial_graph.nodes[n].get("demand", 0.0)
                    for n in env_state["nodes"]
                )
                delivered = max(0.0, initial_demands - total_demands)
                delivery_rate = delivered / max(1e-6, initial_demands)
            budget_spent = self.smdp.budget.used

            # Feed the delivery rate back to the curriculum, where only attacked episodes drive the
            # ramp, and log its state so the ramp stays auditable post-hoc.
            if self.attack_curriculum is not None:
                self.attack_curriculum.record(delivery_rate)
                cstate = self.attack_curriculum.state()
                self.writer.add_scalar("Curriculum/Level", cstate["level"], ep)
                self.writer.add_scalar("Curriculum/Budget", cstate["budget"], ep)
                self.writer.add_scalar("Curriculum/Attacked", 1.0 if self._episode_attacked else 0.0, ep)
                self.writer.add_scalar("Curriculum/Window_Mean_Delivery", cstate["window_mean_delivery"], ep)

            # 5. Log episode results to console
            print(
                f"Ep {ep:4d} | Phase: {self.current_phase[:7].upper():7s} | "
                f"Ticks: {ep_ticks:3d} | Protag R: {ep_protag_reward:7.2f} | "
                f"Antag R: {ep_antag_reward:7.2f} | Delivery: {delivery_rate*100:5.1f}% | "
                f"Budget Used: {budget_spent:5.2f}/{self.smdp.config.congestion_budget}"
            )

            # 6. Log scalars to TensorBoard
            self.writer.add_scalar("Episode/Ticks", ep_ticks, ep)
            self.writer.add_scalar("Episode/Protagonist_Reward", ep_protag_reward, ep)
            self.writer.add_scalar("Episode/Antagonist_Reward", ep_antag_reward, ep)
            self.writer.add_scalar("Episode/Delivery_Rate", delivery_rate, ep)
            self.writer.add_scalar("Episode/Budget_Spent", budget_spent, ep)

            # In latency reward mode the protagonist reward telescopes to minus the total
            # outstanding wait, so total_wait = -ep_protag_reward and mean latency is that
            # normalised per request.
            if getattr(self.smdp.config, "reward_mode", "legacy") == "latency":
                total_wait = -ep_protag_reward
                num_requests = max(1.0, initial_demands)
                self.writer.add_scalar("Episode/Total_Wait", total_wait, ep)
                self.writer.add_scalar("Episode/Mean_Latency", total_wait / num_requests, ep)
            if is_dynamic:
                # Mean wait of completed requests only, which is untruncated, plus the residual
                # queue at the horizon.
                if env._delivered_latencies:
                    self.writer.add_scalar(
                        "Episode/Mean_Delivered_Latency",
                        sum(env._delivered_latencies) / len(env._delivered_latencies), ep)
                self.writer.add_scalar("Episode/Final_Queue", float(env.remaining_demand), ep)
                self.writer.add_scalar("Episode/Num_Arrivals", initial_demands, ep)
            self.writer.add_scalar("Phase/Training_Flag", 1.0 if self.current_phase == "protagonist" else 0.0, ep)

            if protag_losses and self.current_phase == "protagonist":
                avg_critic_loss = np.mean([m["protag_critic_loss"] for m in protag_losses])
                avg_actor_loss = np.mean([m["protag_actor_loss"] for m in protag_losses])
                avg_alpha_loss = np.mean([m["protag_alpha_loss"] for m in protag_losses])
                avg_q_val = np.mean([m.get("protag_q_val", 0) for m in protag_losses])
                avg_entropy = np.mean([m.get("protag_entropy", 0) for m in protag_losses])
                avg_q_spread = np.mean([m.get("protag_q_spread", 0) for m in protag_losses])
                avg_c_grad = np.mean([m.get("protag_critic_grad_norm", 0) for m in protag_losses])
                avg_a_grad = np.mean([m.get("protag_actor_grad_norm", 0) for m in protag_losses])
                
                self.writer.add_scalar("Loss/Protagonist_Critic", avg_critic_loss, ep)
                self.writer.add_scalar("Loss/Protagonist_Actor", avg_actor_loss, ep)
                self.writer.add_scalar("Loss/Protagonist_Alpha_Loss", avg_alpha_loss, ep)
                self.writer.add_scalar("Params/Protagonist_Alpha", self.protag.alpha, ep)
                self.writer.add_scalar("Value/Protagonist_Q", avg_q_val, ep)
                self.writer.add_scalar("Value/Protagonist_Q_Spread", avg_q_spread, ep)
                self.writer.add_scalar("Value/Protagonist_Entropy", avg_entropy, ep)
                self.writer.add_scalar("Gradients/Protagonist_Critic_Norm", avg_c_grad, ep)
                self.writer.add_scalar("Gradients/Protagonist_Actor_Norm", avg_a_grad, ep)

            if antag_losses and self.current_phase == "antagonist":
                avg_critic_loss = np.mean([m["antag_critic_loss"] for m in antag_losses])
                avg_actor_loss = np.mean([m["antag_actor_loss"] for m in antag_losses])
                avg_alpha_loss = np.mean([m["antag_alpha_loss"] for m in antag_losses])
                avg_q_val = np.mean([m.get("antag_q_val", 0) for m in antag_losses])
                avg_entropy = np.mean([m.get("antag_entropy", 0) for m in antag_losses])
                avg_c_grad = np.mean([m.get("antag_critic_grad_norm", 0) for m in antag_losses])
                avg_a_grad = np.mean([m.get("antag_actor_grad_norm", 0) for m in antag_losses])

                self.writer.add_scalar("Loss/Antagonist_Critic", avg_critic_loss, ep)
                self.writer.add_scalar("Loss/Antagonist_Actor", avg_actor_loss, ep)
                self.writer.add_scalar("Loss/Antagonist_Alpha_Loss", avg_alpha_loss, ep)
                self.writer.add_scalar("Params/Antagonist_Alpha", self.antag.alpha, ep)
                self.writer.add_scalar("Value/Antagonist_Q", avg_q_val, ep)
                self.writer.add_scalar("Value/Antagonist_Entropy", avg_entropy, ep)
                self.writer.add_scalar("Gradients/Antagonist_Critic_Norm", avg_c_grad, ep)
                self.writer.add_scalar("Gradients/Antagonist_Actor_Norm", avg_a_grad, ep)

            # 7. Periodic learned-vs-greedy eval snapshot (logged under Eval/*).
            if self.eval_fn is not None and self.eval_every > 0 and ep % self.eval_every == 0:
                em = self.eval_fn(ep)
                for key, val in em.items():
                    self.writer.add_scalar(f"Eval/{key}", val, ep)
                # Accepts both the single-cell and the multi-seed eval key sets.
                gat = em.get("greedy_atk", em.get("greedy_atk_mean"))
                lat = em.get("learned_atk", em.get("learned_atk_mean"))
                gap = em.get("gap_atk", em.get("gap_atk_mean"))
                if gap is not None:
                    std = em.get("gap_atk_std")
                    std_s = f" +/-{std:.0f}" if std is not None else ""
                    print(
                        f"  [EVAL ep {ep}] greedy_atk={gat:.0f} learned_atk={lat:.0f} "
                        f"gap={gap:+.0f}{std_s} (neg = learned beats greedy under attack)"
                    )

        model_save_dir = os.path.join("models", "runs", self.run_name)
        os.makedirs(os.path.join(model_save_dir, "protagonist"), exist_ok=True)
        os.makedirs(os.path.join(model_save_dir, "antagonist"), exist_ok=True)
        torch.save(self.protag.actor.state_dict(), os.path.join(model_save_dir, "protagonist", "actor.pt"))
        torch.save(self.antag.actor.state_dict(), os.path.join(model_save_dir, "antagonist", "actor.pt"))
        print(f"Saved coevolved protagonist actor to {os.path.join(model_save_dir, 'protagonist', 'actor.pt')}")
        print(f"Saved coevolved antagonist actor to {os.path.join(model_save_dir, 'antagonist', 'actor.pt')}")

        # Copies at the standard latest paths, which downstream scripts read.
        os.makedirs("models/protagonist", exist_ok=True)
        os.makedirs("models/antagonist", exist_ok=True)
        torch.save(self.protag.actor.state_dict(), "models/protagonist/actor.pt")
        torch.save(self.antag.actor.state_dict(), "models/antagonist/actor.pt")
        print("Saved copy of protagonist actor as latest to models/protagonist/actor.pt")
        print("Saved copy of antagonist actor as latest to models/antagonist/actor.pt")

        print("\nSACRED ATLA Training completed successfully! Logs saved to TensorBoard.")
        self.writer.close()
