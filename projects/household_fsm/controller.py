"""Finite-state controller for the deterministic Household task.

Run-to-completion: `select_option` only fires when the previous option
terminates, so each branch picks the next macro action and returns. The
controller plans against `ControllerObservation` symbolic state (door
targets, key inventory, battery, reachability).

Logic:
  - PLAN: look at the current door target. If we already hold the matching
    key (or it's unlocked) go drive at the door; otherwise navigate to the
    key first. With no door pending, head to the goal.
  - Pre-empt any movement when can_safely_reach() says we can't afford to
    reach the current target — divert to the nearest charger first.
  - NAVIGATE_TO_*: once at the target, transition to the corresponding
    action state (PICKUP_KEY / UNLOCK_DOOR / CHARGE).
  - CHARGE: top off and replan.

Run with: python -m a4 eval q1 student
"""

from enum import Enum, auto

from household_core.fsm import FiniteStateController
from household_core.household_env import ControllerObservation
from household_core.options import OptionCall


class FSMState(Enum):
    PLAN = auto()
    PICKUP_KEY = auto()
    UNLOCK_DOOR = auto()
    NAVIGATE_TO_KEY = auto()
    NAVIGATE_TO_DOOR = auto()
    NAVIGATE_TO_CHARGER = auto()
    CHARGE = auto()
    GO_TO_GOAL = auto()


class StudentFSMController(FiniteStateController):

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset per-episode FSM memory before each seed."""
        self.state = FSMState.PLAN
        self.target: str | None = None

    def select_option(self, observation: ControllerObservation) -> OptionCall:
        """Choose the next option from the current FSM state.

        Loops internally because most transitions don't emit an action —
        e.g. NAVIGATE_TO_KEY → PICKUP_KEY → PLAN can all happen in a single
        select_option call once the robot is already on the key.
        """
        o = observation
        count = 0
        while count < 50:
            count += 1

            if self.state == FSMState.PLAN:
                next_door = o.current_door_target
                if next_door is not None:
                    if o.can_unlock(next_door):
                        self.target = next_door
                        self.state = FSMState.NAVIGATE_TO_DOOR
                    else:
                        next_key = o.doors[next_door].matching_key_id
                        self.target = next_key
                        self.state = FSMState.NAVIGATE_TO_KEY
                else:
                    self.state = FSMState.GO_TO_GOAL

            # Pre-empt anything if we can't safely reach the current target.
            if not o.can_safely_reach(self.target):
                self.target = o.nearest_charger_id
                self.state = FSMState.NAVIGATE_TO_CHARGER

            if self.state == FSMState.GO_TO_GOAL:
                if not o.success:
                    return OptionCall("go_to_goal", "goal")

            if self.state in (FSMState.NAVIGATE_TO_KEY, FSMState.NAVIGATE_TO_DOOR, FSMState.NAVIGATE_TO_CHARGER):
                if o.at(self.target):
                    if self.state == FSMState.NAVIGATE_TO_KEY:
                        self.state = FSMState.PICKUP_KEY
                    if self.state == FSMState.NAVIGATE_TO_DOOR:
                        self.state = FSMState.UNLOCK_DOOR
                    if self.state == FSMState.NAVIGATE_TO_CHARGER:
                        self.state = FSMState.CHARGE
                else:
                    return OptionCall("navigate_to", self.target)

            if self.state == FSMState.CHARGE:
                if o.battery_level == o.battery_capacity:
                    self.state = FSMState.PLAN
                else:
                    return OptionCall("charge_at", self.target)

            if self.state == FSMState.PICKUP_KEY:
                if o.at(self.target) and not (o.carried_key_id == self.target):
                    return OptionCall("pickup_key", self.target)
                else:
                    self.state = FSMState.PLAN

            if self.state == FSMState.UNLOCK_DOOR:
                if o.at(self.target) and not o.doors[self.target].is_open:
                    return OptionCall("unlock_door", self.target)
                else:
                    self.state = FSMState.PLAN

        raise TimeoutError("ERROR: Infinite loop in the select option method.")
