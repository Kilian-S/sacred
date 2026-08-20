"""Shared protagonist transition construction, the single source of truth for
``SMDPTransition`` records.

Trucks decide sequentially with state projection (each truck's choice is committed into the
projected observation before the next decides) and sequential claiming (a claimed demand node is
removed from later trucks' masks, so two trucks never take the same request; depots are never
claimed). A ``choose_fn`` parameter is the only difference between training and demo collection.
"""

from __future__ import annotations

from typing import Any, Callable

from src.env.smdp_wrapper import DecisionEvent, SMDPDecisionWrapper, SMDPTransition

# choose_fn(projected_obs, truck_mask, truck_id) -> {truck_id: chosen_node}
ChooseFn = Callable[[dict, dict, int], "dict[int, Any]"]


def collect_protagonist_transitions(
    smdp: SMDPDecisionWrapper,
    event: DecisionEvent,
    choose_fn: ChooseFn,
) -> tuple[DecisionEvent, list[SMDPTransition]]:
    """Run one protagonist decision epoch and return (next_event, per-truck transitions).

    Args:
        choose_fn: Selects the action for the active truck, given its projected observation and
            its claim-reduced mask.
    """
    mask = event.protagonist_action_mask

    actions: dict[int, Any] = {}
    projected_obs = dict(event.observation)
    projected_obs["trucks"] = {tid: dict(t) for tid, t in event.observation["trucks"].items()}
    truck_decision_states: dict[int, dict] = {}
    truck_masks: dict[int, dict] = {}
    claimed: set = set()

    def is_demand(n: Any) -> bool:
        return smdp.env.graph.nodes[n].get("demand", 0.0) > 0.0

    for truck_id in event.waiting_trucks:
        truck_mask = {tid: [n for n in opts if n not in claimed] for tid, opts in mask.items()}
        truck_masks[truck_id] = truck_mask
        projected_obs["active_truck"] = truck_id
        projected_obs["allowed_destinations"] = {"protagonist": dict(truck_mask)}

        # Replay needs the exact projected state this truck saw.
        state_used = dict(projected_obs)
        state_used["trucks"] = {tid: dict(t) for tid, t in projected_obs["trucks"].items()}
        truck_decision_states[truck_id] = state_used

        truck_action = choose_fn(projected_obs, truck_mask, truck_id)
        actions.update(truck_action)

        chosen_node = truck_action.get(truck_id)
        if chosen_node is not None:
            projected_obs["trucks"][truck_id]["destination"] = chosen_node
            projected_obs["trucks"][truck_id]["current_node"] = None
            if is_demand(chosen_node):
                claimed.add(chosen_node)

    next_event, transition = smdp.step_protagonist(actions)

    transitions: list[SMDPTransition] = []
    for truck_id in event.waiting_trucks:
        next_state_copy = dict(next_event.observation)
        next_state_copy["active_truck"] = next_event.waiting_trucks[0] if next_event.waiting_trucks else None
        next_state_copy["allowed_destinations"] = {
            "protagonist": dict(next_event.protagonist_action_mask)
        }
        transitions.append(
            SMDPTransition(
                agent="protagonist",
                state=truck_decision_states[truck_id],
                action=dict(actions),
                reward=transition.reward,
                next_state=next_state_copy,
                done=transition.done,
                elapsed_ticks=transition.elapsed_ticks,
                action_mask={"protagonist": dict(truck_masks[truck_id])},
                info=dict(transition.info),
            )
        )

    return next_event, transitions
