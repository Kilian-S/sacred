"""Contested-resupply arena (gen07 / Act IV): the single source of truth for its config.

The redirection's headline arena (DIRECTION.md, `experiments/gen07_contested_matrix.md`). It is
the dynassign dynamics (Poisson resupply demand, 2 depots, latency reward, destination-mode
assignment: the rung where protagonists demonstrably reach competence) with the antagonist's
reach set to **route** so both scripted and learned attacks aim along each vehicle's committed
delivery path -- the surface where interception is a real, learnable decision (the gen05 nugget:
learned best responses beat every scripted attack against a competent, predictable victim there).

Framing: contested autonomous resupply (depots = logistics hubs, Poisson stream = forward-unit
resupply demand, full-block antagonist = corridor denial / EW jamming under a sortie budget).
The physics are identical to dynassign; only `antag_reach` changes, so every existing attacker,
the greedy baseline and the whole evaluation harness carry over unchanged.

Both the trainer and the evaluation harness import `contested_config()` from here so the config
can never silently drift between train and eval (the dynassign/hybrid configs are duplicated by
hand across two files -- this arena does not repeat that).
"""

from __future__ import annotations

from src.env.smdp_wrapper import SMDPConfig
from src.envs.assignment_factory import make_dynamic_assign_env

# Budget is TO-FINALISE by the B9 recoverability probe (gen07 ledger): the target is a
# fitted-scripted attack that costs greedy ~+30-60% with attacked delivery in a trainable band,
# NOT the gen06 collapse regime. 4000 is the dynassign lineage default carried over as a
# placeholder until that probe pins it; do not treat it as final.
DEFAULT_CONTESTED_BUDGET = 4000.0


def contested_config(congestion_budget: float = DEFAULT_CONTESTED_BUDGET) -> SMDPConfig:
    """The contested-resupply SMDPConfig. dynassign lineage + route reach.

    Kept in lockstep with the dynassign config (`scripts/evaluate_dynamic_assign.dynassign_config`)
    except for `antag_reach="route"`; if that config changes, change this deliberately.
    """
    return SMDPConfig(
        max_ticks=800,
        antagonist_interval=25,           # ~32 antagonist decision events / episode
        congestion_duration=120,          # each full roadblock persists ~5 events (sustained)
        congestion_budget=congestion_budget,
        congestion_cooldown=0,
        congestion_cost=0.1,
        reward_mode="latency",
        routing_mode="destination",       # assignment only: env auto-routes (routing deferred)
        congestion_levels=(1.0,),         # full blockage only
        max_antag_actions_per_event=1,    # one strategic roadblock per event
        antag_reach="route",              # aim along each vehicle's committed delivery path
    )


# The env itself is the dynassign env (same geometry/demand). Re-exported under a
# contested-framing name so call sites read in the arena's own language.
make_contested_env = make_dynamic_assign_env
