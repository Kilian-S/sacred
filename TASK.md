# TASK.md (The Direct Order)

## Objective
Implement a robust checkpointing and resume system for the ATLA training loop. This system must save the entire mathematical state of the training (neural weights, optimizer momentum, and the Replay Buffer) so that massive multi-day runs can be paused, safely resumed, or extended without suffering from Catastrophic Forgetting.

## Implementation Steps
- [x] **State Serialization**: In `src/agents/sac.py`, implement a `save_checkpoint(filepath)` method for both the `ProtagonistSAC` and `AntagonistSAC` agents. It must save a dictionary containing: actor weights, both critic weights, both target critic weights, all Adam optimizers, and the `replay_buffer` data.
- [x] **State Restoration**: In `src/agents/sac.py`, implement a corresponding `load_checkpoint(filepath)` method that properly restores the weights, optimizers, and replay buffer into the agent instances.
- [x] **Trainer Integration**: In `src/agents/sacred_atla.py`, add a parameter or logic to call `save_checkpoint()` on both agents at the end of every `switch_every` phase, saving the files as `models/protagonist/checkpoint.pt` and `models/antagonist/checkpoint.pt`.
- [x] **CLI Resume Flag**: In `scripts/train_sacred.py`, add an optional `--resume-checkpoint` string argument (e.g., path to a directory). If provided, the script should skip ALNS pre-seeding, load the checkpoints into the agents, and parse the start episode so the TensorBoard loop resumes correctly instead of starting at Episode 0.

## Definition of Done
*   All checklist items above are marked as `[x]`.
*   The `train_sacred.py` loop initiates properly without throwing dimensionality or masking errors.
*   The TensorBoard logs correctly register episodic progression.
