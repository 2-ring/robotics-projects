"""Define an abstract base class as an interface for Q1 finite-state controllers."""

from abc import ABC, abstractmethod

from household_core.household_env import ControllerObservation
from household_core.options import OptionCall


class FiniteStateController(ABC):
    """Run-to-completion high-level controller interface."""

    @abstractmethod
    def reset(self) -> None:
        """Reset controller-internal state before a new episode."""

    @abstractmethod
    def select_option(self, observation: ControllerObservation) -> OptionCall:
        """Choose the next option when the current option has terminated."""

    def on_option_terminated(
        self,
        observation: ControllerObservation,
        option_call: OptionCall,
        status: str,
    ) -> None:
        """Observe the outcome of a terminated option."""
        _ = observation, option_call, status
