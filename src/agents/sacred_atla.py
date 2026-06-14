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

from src.env.smdp_wrapper import DecisionType, SMDPDecisionWrapper, SMDPTransition
from src.agents.sac import AntagonistSAC, ProtagonistSAC


class ATLACoevolutionTrainer:
    """Trainer orchestrating alternating minimax GRL training of both agents.

    Parameters
    ----------
    smdp:
        SMDP Wrapper environment instance.
    protag_agent:
        Soft Actor-Critic agent for fleet routing.
    antag_agent:
        Soft Actor-Critic agent for edge congestion.
    log_dir:
        Directory to save TensorBoard training runs.
    switch_every_episodes:
        Number of episodes to train one agent before freezing it and training the other.
    batch_size:
        Mini-batch training size for SAC updates.
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
    ) -> None:
        self.smdp = smdp
        self.protag = protag_agent
        self.antag = antag_agent
        self.switch_every_episodes = switch_every_episodes
        self.batch_size = batch_size

        if run_name is None:
            run_name = f"sacred_atla_sw{switch_every_episodes}_b{batch_size}"
        self.run_name = run_name

        # Create TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, self.run_name))

        # Training phase tracking
        self.current_phase: str = "protagonist"  # starts with protagonist training
        self.episode_count = 0
        self.step_count = 0

    def run_training(self, total_episodes: int, start_episode: int = 0) -> None:
        """Run the coevolutionary training loop for total_episodes."""
        print(f"Starting SACRED ATLA training from scratch for {total_episodes} episodes...")
        print(f"Device: Protagonist={self.protag.device}, Antagonist={self.antag.device}")
        print(f"Alternating every {self.switch_every_episodes} episodes.")
        print(f"Initial Phase: Training {self.current_phase.upper()} (Adversary FROZEN)\n")

        for ep in range(start_episode + 1, total_episodes + 1):
            self.episode_count = ep

            # 1. Check training phase switch
            if (ep - 1) > 0 and (ep - 1) % self.switch_every_episodes == 0:
                print("Saving checkpoints for both agents before phase switch...")
                self.protag.save_checkpoint("models/protagonist/checkpoint.pt", ep - 1)
                self.antag.save_checkpoint("models/antagonist/checkpoint.pt", ep - 1)
                self.current_phase = "antagonist" if self.current_phase == "protagonist" else "protagonist"
                print(f"\n--- Episode {ep}: Switching training phase to {self.current_phase.upper()} ---")

            # 2. Reset environment and tracking accumulators
            event = self.smdp.reset_decision_env()
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
                    mask = event.protagonist_action_mask
                    
                    # For each waiting truck, we choose actions sequentially with state projection
                    actions = {}
                    projected_obs = dict(event.observation)
                    projected_obs["trucks"] = {tid: dict(t) for tid, t in event.observation["trucks"].items()}
                    truck_decision_states = {}
                    
                    for truck_id in event.waiting_trucks:
                        projected_obs["active_truck"] = truck_id
                        projected_obs["allowed_destinations"] = {
                            "protagonist": dict(mask)
                        }
                        
                        # Save the exact projected state this truck sees for replay buffer training compatibility
                        truck_decision_states[truck_id] = dict(projected_obs)
                        truck_decision_states[truck_id]["trucks"] = {tid: dict(t) for tid, t in projected_obs["trucks"].items()}
                        
                        # Select action for this truck
                        truck_action = self.protag.select_action(
                            projected_obs, mask, deterministic=False
                        )
                        actions.update(truck_action)
                        
                        # Project commitment: update destination and remove current node for this truck
                        chosen_node = truck_action.get(truck_id)
                        if chosen_node is not None:
                            projected_obs["trucks"][truck_id]["destination"] = chosen_node
                            projected_obs["trucks"][truck_id]["current_node"] = None

                    next_event, transition = self.smdp.step_protagonist(actions)
                    
                    # Push individual transitions to the replay buffer for each truck that was active
                    for truck_id in event.waiting_trucks:
                        state_used = truck_decision_states[truck_id]
                        
                        next_state_copy = dict(next_event.observation)
                        if next_event.waiting_trucks:
                            next_state_copy["active_truck"] = next_event.waiting_trucks[0]
                        else:
                            next_state_copy["active_truck"] = None
                            
                        next_state_copy["allowed_destinations"] = {
                            "protagonist": dict(next_event.protagonist_action_mask)
                        }
                        
                        t_trans = SMDPTransition(
                            agent="protagonist",
                            state=state_used,
                            action=dict(actions),  # Actions mapping contains the chosen node for truck_id
                            reward=transition.reward,
                            next_state=next_state_copy,
                            done=transition.done,
                            elapsed_ticks=transition.elapsed_ticks,
                            action_mask={"protagonist": dict(mask)},
                            info=dict(transition.info)
                        )
                        self.protag.replay_buffer.push(t_trans)

                    ep_protag_reward += transition.reward
                    ep_antag_reward += next_event.antagonist_reward

                    # Update protagonist parameters if in protagonist phase
                    if self.current_phase == "protagonist":
                        metrics = self.protag.update(self.batch_size)
                        if metrics:
                            protag_losses.append(metrics)

                    event = next_event

                # B. ANTAGONIST DECISION EPOCH
                elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                    mask = event.antagonist_action_mask
                    # Active congestion choice
                    remaining_budget_before = self.smdp.budget.remaining
                    action = self.antag.select_action(
                        event.observation, mask, remaining_budget_before, deterministic=False
                    )

                    next_event, transition = self.smdp.step_antagonist(action)
                    
                    # Enrich transition info and next_state allowed_destinations for Antagonist SAC update
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

                    # Update antagonist parameters if in antagonist phase
                    if self.current_phase == "antagonist":
                        metrics = self.antag.update(self.batch_size)
                        if metrics:
                            antag_losses.append(metrics)

                    event = next_event

                else:
                    # In case of event-only stepping (fallthrough)
                    event = self.smdp.advance_until_decision()

            # 4. Extract end-of-episode simulator stats
            env_state = self.smdp.env.observe()
            total_demands = sum(n["demand"] for n in env_state["nodes"].values())
            initial_demands = sum(
                self.smdp.env._initial_graph.nodes[n].get("demand", 0.0)
                for n in env_state["nodes"]
            )
            delivered = max(0.0, initial_demands - total_demands)
            delivery_rate = delivered / max(1e-6, initial_demands)
            budget_spent = self.smdp.budget.used

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
            self.writer.add_scalar("Phase/Training_Flag", 1.0 if self.current_phase == "protagonist" else 0.0, ep)

            # Log Protagonist training metrics
            if protag_losses and self.current_phase == "protagonist":
                avg_critic_loss = np.mean([m["protag_critic_loss"] for m in protag_losses])
                avg_actor_loss = np.mean([m["protag_actor_loss"] for m in protag_losses])
                avg_alpha_loss = np.mean([m["protag_alpha_loss"] for m in protag_losses])
                avg_q_val = np.mean([m.get("protag_q_val", 0) for m in protag_losses])
                avg_entropy = np.mean([m.get("protag_entropy", 0) for m in protag_losses])
                avg_c_grad = np.mean([m.get("protag_critic_grad_norm", 0) for m in protag_losses])
                avg_a_grad = np.mean([m.get("protag_actor_grad_norm", 0) for m in protag_losses])
                
                self.writer.add_scalar("Loss/Protagonist_Critic", avg_critic_loss, ep)
                self.writer.add_scalar("Loss/Protagonist_Actor", avg_actor_loss, ep)
                self.writer.add_scalar("Loss/Protagonist_Alpha_Loss", avg_alpha_loss, ep)
                self.writer.add_scalar("Params/Protagonist_Alpha", self.protag.alpha, ep)
                self.writer.add_scalar("Value/Protagonist_Q", avg_q_val, ep)
                self.writer.add_scalar("Value/Protagonist_Entropy", avg_entropy, ep)
                self.writer.add_scalar("Gradients/Protagonist_Critic_Norm", avg_c_grad, ep)
                self.writer.add_scalar("Gradients/Protagonist_Actor_Norm", avg_a_grad, ep)

            # Log Antagonist training metrics
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

        # Save trained actor models
        import os
        # 1. Save to unique run directory
        model_save_dir = os.path.join("models", "runs", self.run_name)
        os.makedirs(os.path.join(model_save_dir, "protagonist"), exist_ok=True)
        os.makedirs(os.path.join(model_save_dir, "antagonist"), exist_ok=True)
        torch.save(self.protag.actor.state_dict(), os.path.join(model_save_dir, "protagonist", "actor.pt"))
        torch.save(self.antag.actor.state_dict(), os.path.join(model_save_dir, "antagonist", "actor.pt"))
        print(f"Saved coevolved protagonist actor to {os.path.join(model_save_dir, 'protagonist', 'actor.pt')}")
        print(f"Saved coevolved antagonist actor to {os.path.join(model_save_dir, 'antagonist', 'actor.pt')}")

        # 2. Save copies to standard latest paths for downstream scripts (backward compatibility)
        os.makedirs("models/protagonist", exist_ok=True)
        os.makedirs("models/antagonist", exist_ok=True)
        torch.save(self.protag.actor.state_dict(), "models/protagonist/actor.pt")
        torch.save(self.antag.actor.state_dict(), "models/antagonist/actor.pt")
        print("Saved copy of protagonist actor as latest to models/protagonist/actor.pt")
        print("Saved copy of antagonist actor as latest to models/antagonist/actor.pt")

        print("\nSACRED ATLA Training completed successfully! Logs saved to TensorBoard.")
        self.writer.close()
