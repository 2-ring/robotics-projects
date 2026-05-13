"""Define an interface for options and implement concrete skills for the Household domain."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from household_core.household_env import ControllerObservation, HouseholdEnv, PrimitiveAction


class OptionStatus:
    """Status constants for option execution."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass(frozen=True)
class OptionCall:
    """A high-level option invocation."""

    option_name: str
    target_id: str | None = None


def NavigateTo(target_id: str) -> OptionCall:
    """Construct a navigation option call."""
    return OptionCall("navigate_to", target_id)


def PickupKey(key_id: str) -> OptionCall:
    """Construct a pickup option call."""
    return OptionCall("pickup_key", key_id)


def UnlockDoor(door_id: str) -> OptionCall:
    """Construct a door-unlock option call."""
    return OptionCall("unlock_door", door_id)


def ChargeAt(charger_id: str) -> OptionCall:
    """Construct a charge option call."""
    return OptionCall("charge_at", charger_id)


def GoToGoal() -> OptionCall:
    """Construct a goal-navigation option call."""
    return OptionCall("go_to_goal", "goal")


class BaseOption(ABC):
    """Standard option interface with stepwise execution."""

    def __init__(self, call: OptionCall) -> None:
        """Store the option call and initialize per-execution state."""
        self.call = call
        self.started = False
        self.last_action = PrimitiveAction.WAIT

    def start(self, env: HouseholdEnv) -> None:
        """Initialize option-local state before the first primitive step."""
        _ = env
        self.started = True

    @abstractmethod
    def status(self, observation: ControllerObservation) -> str:
        """Return the current option status."""

    @abstractmethod
    def choose_action(
        self,
        env: HouseholdEnv,
        observation: ControllerObservation,
    ) -> PrimitiveAction | None:
        """Return the next primitive action, or `None` if impossible."""

    def after_step(
        self,
        observation: ControllerObservation,
        info: dict[str, object],
    ) -> str:
        """Return the post-step option status."""
        _ = info
        return self.status(observation)

    def step(self, env: HouseholdEnv) -> str:
        """Advance the option by one primitive environment step."""
        if not self.started:
            self.start(env)
        observation = env.controller_observation()
        status = self.status(observation)
        if status != OptionStatus.RUNNING:
            return status
        action = self.choose_action(env, observation)
        if action is None:
            return OptionStatus.FAILURE
        self.last_action = action
        _, _, _, _, info = env.step(int(action))
        return self.after_step(env.controller_observation(), info)


class NavigateOption(BaseOption):
    """Navigate to a key, door, charger, room anchor, or goal."""

    def __init__(self, call: OptionCall) -> None:
        """Initialize navigation state for one option execution."""
        super().__init__(call)
        self.failed_this_step = False

    def status(self, observation: ControllerObservation) -> str:
        """Return whether navigation should run, succeed, or fail."""
        if self.failed_this_step:
            return OptionStatus.FAILURE
        if observation.terminated and not observation.success:
            return OptionStatus.FAILURE
        target_id = self.call.target_id
        if target_id is None:
            return OptionStatus.FAILURE
        if observation.at(target_id):
            return OptionStatus.SUCCESS
        if observation.primitive_distance_to(target_id) is None:
            return OptionStatus.FAILURE
        return OptionStatus.RUNNING

    def choose_action(
        self,
        env: HouseholdEnv,
        observation: ControllerObservation,
    ) -> PrimitiveAction | None:
        """Choose the next navigation action or emit a stochastic failure."""
        _ = observation
        if self.call.target_id is None:
            return None
        self.failed_this_step = bool(
            env.stochasticity.navigation_failure_prob > 0
            and env.np_random.random() < env.stochasticity.navigation_failure_prob,
        )
        if self.failed_this_step:
            return PrimitiveAction.WAIT
        return env.next_action_toward(self.call.target_id)

    def after_step(
        self,
        observation: ControllerObservation,
        info: dict[str, object],
    ) -> str:
        """Convert stochastic navigation failures into option-level failures."""
        _ = info
        if self.failed_this_step:
            self.failed_this_step = False
            return OptionStatus.FAILURE
        return self.status(observation)


class PickupKeyOption(BaseOption):
    """Pick up a key at the current cell."""

    def status(self, observation: ControllerObservation) -> str:
        """Return whether pickup should run, succeed, or fail."""
        if observation.terminated and not observation.success:
            return OptionStatus.FAILURE
        if observation.carried_key_id == self.call.target_id:
            return OptionStatus.SUCCESS
        target_id = self.call.target_id
        if target_id is None or observation.primitive_distance_to(target_id) != 0:
            return OptionStatus.FAILURE
        return OptionStatus.RUNNING

    def choose_action(
        self,
        env: HouseholdEnv,
        observation: ControllerObservation,
    ) -> PrimitiveAction | None:
        """Issue the pickup interaction action."""
        _ = env, observation
        return PrimitiveAction.INTERACT

    def after_step(
        self,
        observation: ControllerObservation,
        info: dict[str, object],
    ) -> str:
        """Treat failed interactions as option failure for key pickup."""
        if not bool(info["interaction_succeeded"]):
            return OptionStatus.FAILURE
        return self.status(observation)


class UnlockDoorOption(BaseOption):
    """Unlock the specified door."""

    def status(self, observation: ControllerObservation) -> str:
        """Return whether unlock should run, succeed, or fail."""
        target_id = self.call.target_id
        if target_id is None:
            return OptionStatus.FAILURE
        if observation.terminated and not observation.success:
            return OptionStatus.FAILURE
        if observation.door_is_open(target_id):
            return OptionStatus.SUCCESS
        if not observation.can_unlock(target_id):
            return OptionStatus.FAILURE
        if observation.primitive_distance_to(target_id) != 0:
            return OptionStatus.FAILURE
        return OptionStatus.RUNNING

    def choose_action(
        self,
        env: HouseholdEnv,
        observation: ControllerObservation,
    ) -> PrimitiveAction | None:
        """Issue the unlock interaction action."""
        _ = env, observation
        return PrimitiveAction.INTERACT

    def after_step(
        self,
        observation: ControllerObservation,
        info: dict[str, object],
    ) -> str:
        """Treat failed interactions as option failure for door unlock."""
        if not bool(info["interaction_succeeded"]):
            return OptionStatus.FAILURE
        return self.status(observation)


class ChargeOption(BaseOption):
    """Recharge at the given charger."""

    def status(self, observation: ControllerObservation) -> str:
        """Return whether charging should run, succeed, or fail."""
        target_id = self.call.target_id
        if target_id is None:
            return OptionStatus.FAILURE
        if observation.terminated and not observation.success:
            return OptionStatus.FAILURE
        if observation.at(target_id) and observation.battery_level >= observation.battery_capacity:
            return OptionStatus.SUCCESS
        if observation.primitive_distance_to(target_id) != 0:
            return OptionStatus.FAILURE
        return OptionStatus.RUNNING

    def choose_action(
        self,
        env: HouseholdEnv,
        observation: ControllerObservation,
    ) -> PrimitiveAction | None:
        """Issue the charging interaction action."""
        _ = env, observation
        return PrimitiveAction.INTERACT

    def after_step(
        self,
        observation: ControllerObservation,
        info: dict[str, object],
    ) -> str:
        """Treat failed interactions as option failure for charging."""
        if not bool(info["interaction_succeeded"]):
            return OptionStatus.FAILURE
        return self.status(observation)


class GoToGoalOption(NavigateOption):
    """Navigate to the final goal cell."""


class OptionLibrary:
    """Instantiate concrete options from controller-level calls."""

    def create(self, call: OptionCall) -> BaseOption:
        """Instantiate a concrete option from a call description."""
        if call.option_name == "navigate_to":
            return NavigateOption(call)
        if call.option_name == "pickup_key":
            return PickupKeyOption(call)
        if call.option_name == "unlock_door":
            return UnlockDoorOption(call)
        if call.option_name == "charge_at":
            return ChargeOption(call)
        if call.option_name == "go_to_goal":
            return GoToGoalOption(call)
        raise KeyError(f"Unknown option: {call.option_name}")
