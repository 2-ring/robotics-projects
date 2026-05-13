"""Small reactive behavior tree framework for Q2."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from household_core.household_env import ControllerObservation, HouseholdEnv
from household_core.options import BaseOption, OptionCall, OptionLibrary, OptionStatus


class BTStatus:
    """Behavior tree node statuses."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


@dataclass
class BTContext:
    """Execution context shared across one behavior tree."""

    env: HouseholdEnv
    option_library: OptionLibrary
    active_action_id: str | None = None
    active_option_call: OptionCall | None = None
    active_option: BaseOption | None = None
    ticked_action_id: str | None = None

    def start_tick(self) -> None:
        """Prepare context state for a new root tick."""
        self.ticked_action_id = None

    def finish_tick(self) -> None:
        """Halt the active action if it was not selected this tick."""
        if self.active_action_id is not None and self.active_action_id != self.ticked_action_id:
            self.active_action_id = None
            self.active_option_call = None
            self.active_option = None

    def step_option(self, action_id: str, option_call: OptionCall) -> str:
        """Step the option owned by one action node."""
        self.ticked_action_id = action_id
        if self.active_action_id != action_id or self.active_option_call != option_call:
            self.active_action_id = action_id
            self.active_option_call = option_call
            self.active_option = self.option_library.create(option_call)

        if self.active_option is None:
            raise RuntimeError("BT context has no active option to step.")
        status = self.active_option.step(self.env)
        if status != OptionStatus.RUNNING:
            self.active_action_id = None
            self.active_option_call = None
            self.active_option = None
        return status


class BehaviorNode(ABC):
    """Abstract behavior tree node."""

    @abstractmethod
    def tick(self, observation: ControllerObservation, context: BTContext) -> str:
        """Tick this node once and return a BT status."""


class ConditionNode(BehaviorNode):
    """Condition node that never returns RUNNING."""

    def __init__(
        self,
        name: str,
        predicate: Callable[[ControllerObservation], bool],
    ) -> None:
        """Store the user-facing name and predicate."""
        self.name = name
        self.predicate = predicate

    def tick(self, observation: ControllerObservation, context: BTContext) -> str:
        """Evaluate the predicate and map it to BT success or failure."""
        _ = context
        return BTStatus.SUCCESS if self.predicate(observation) else BTStatus.FAILURE


class ActionNode(BehaviorNode):
    """Action node that wraps one parameterized option."""

    def __init__(
        self,
        name: str,
        option_factory: Callable[[ControllerObservation], OptionCall | None],
    ) -> None:
        """Store the action name and factory for option calls."""
        self.name = name
        self.option_factory = option_factory
        self.action_id = f"action:{name}"

    def tick(self, observation: ControllerObservation, context: BTContext) -> str:
        """Tick the wrapped option-producing action node once."""
        option_call = self.option_factory(observation)
        if option_call is None:
            return BTStatus.FAILURE
        status = context.step_option(self.action_id, option_call)
        if status == OptionStatus.SUCCESS:
            return BTStatus.SUCCESS
        if status == OptionStatus.FAILURE:
            return BTStatus.FAILURE
        return BTStatus.RUNNING


class SequenceNode(BehaviorNode):
    """Reactive sequence node."""

    def __init__(self, *children: BehaviorNode) -> None:
        """Store child nodes ticked from left to right."""
        self.children = children

    def tick(self, observation: ControllerObservation, context: BTContext) -> str:
        """Tick children until one fails or runs."""
        for child in self.children:
            status = child.tick(observation, context)
            if status != BTStatus.SUCCESS:
                return status
        return BTStatus.SUCCESS


class FallbackNode(BehaviorNode):
    """Reactive fallback node."""

    def __init__(self, *children: BehaviorNode) -> None:
        """Store child nodes tried from left to right."""
        self.children = children

    def tick(self, observation: ControllerObservation, context: BTContext) -> str:
        """Tick children until one succeeds or runs."""
        for child in self.children:
            status = child.tick(observation, context)
            if status == BTStatus.SUCCESS:
                return BTStatus.SUCCESS
            if status == BTStatus.RUNNING:
                return BTStatus.RUNNING
        return BTStatus.FAILURE


class BehaviorTreeController(ABC):
    """Base class for BT controllers."""

    def __init__(self) -> None:
        """Create the option library and build the root tree."""
        self.option_library = OptionLibrary()
        self.root = self.build_tree()
        self._last_context: BTContext | None = None

    @abstractmethod
    def build_tree(self) -> BehaviorNode:
        """Construct and return the BT root node."""

    def reset(self) -> None:
        """Reset per-episode BT state."""
        self._last_context = None

    def tick(self, env: HouseholdEnv, context: BTContext | None = None) -> str:
        """Tick the root node once."""
        active_context = (
            context
            or self._last_context
            or BTContext(
                env=env,
                option_library=self.option_library,
            )
        )
        active_context.env = env
        active_context.start_tick()
        observation = env.controller_observation()
        status = self.root.tick(observation, active_context)
        active_context.finish_tick()
        self._last_context = active_context
        return status
