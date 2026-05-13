"""Reactive behavior tree for the stochastic Household task.

The tree ticks once per primitive step. Composition (top to bottom):

    Fallback
    ├── Sequence: handle locked door
    │   ├── locked_door?           (condition)
    │   ├── ensure_key             (pickup key if not holding it)
    │   ├── ensure_at("door")      (charge if needed → go to door)
    │   └── unlock_door            (action)
    └── go_to_goal                 (default branch)

`ensure_at(target)` first checks `need_charge` (whether we can safely reach
the target with a small buffer); if not, it diverts to the nearest charger
and tops off, then navigates to the actual target. Earlier children win,
so the locked-door branch always pre-empts go_to_goal when a door blocks
the path. The tree is essentially stateless aside from a `recovering`
latch reset on each episode.

Run with: python -m a4 eval q2 student
"""

from household_core.behavior_tree import (
    ActionNode,
    BehaviorNode,
    BehaviorTreeController,
    ConditionNode,
    FallbackNode,
    SequenceNode,
)
from household_core.household_env import ControllerObservation
from household_core.options import OptionCall


class StudentBehaviorTreeController(BehaviorTreeController):

    def __init__(self) -> None:
        self.recovering = False
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self.recovering = False

    def build_tree(self) -> BehaviorNode:
        """Return the root BehaviorNode for the policy."""
        # Resolve a symbolic target ("charger", "door", "key") to a concrete
        # id from the current observation.
        get_id = lambda o, id: {
            "charger": o.nearest_charger_id,
            "door": o.current_door_target,
            "key": o.doors[o.current_door_target].matching_key_id,
        }[id]

        need_unlock = ConditionNode(
            "locked_door", lambda o: o.current_door_target is not None
        )
        need_key = ConditionNode(
            "have_key", lambda o: o.can_unlock(o.current_door_target)
        )
        need_charge = lambda target: ConditionNode(
            "need_charge",
            lambda o: o.can_safely_reach(get_id(o, target), extra_buffer=2),
        )
        need_navigation = lambda target: ConditionNode(
            "is_at", lambda o: o.at(get_id(o, target))
        )

        navigate_to = lambda target: ActionNode(
            "navigate", lambda o: OptionCall("navigate_to", get_id(o, target))
        )
        go_to_goal = ActionNode("go_to_goal", lambda o: OptionCall("go_to_goal", "goal"))
        charge = ActionNode(
            "charge_at", lambda o: OptionCall("charge_at", get_id(o, "charger"))
        )
        unlock = ActionNode(
            "unlock", lambda o: OptionCall("unlock_door", get_id(o, "door"))
        )
        pickup_key = ActionNode(
            "pickup_key", lambda o: OptionCall("pickup_key", get_id(o, "key"))
        )

        go_to = lambda target: FallbackNode(need_navigation(target), navigate_to(target))
        charged_enough = lambda target: FallbackNode(
            need_charge(target), SequenceNode(go_to("charger"), charge)
        )
        ensure_at = lambda target: SequenceNode(charged_enough(target), go_to(target))
        ensure_key = FallbackNode(need_key, SequenceNode(ensure_at("key"), pickup_key))
        doors_unlocked = SequenceNode(need_unlock, ensure_key, ensure_at("door"), unlock)

        return FallbackNode(doors_unlocked, go_to_goal)
