"""Evaluation helpers and seeded suites for the household-planning domain."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean

from household_core.behavior_tree import BehaviorTreeController, BTContext
from household_core.fsm import FiniteStateController
from household_core.household_env import HouseholdConfig, HouseholdEnv, StochasticityConfig
from household_core.options import OptionLibrary, OptionStatus

DEFAULT_Q1_SEEDS = (5, 11, 17, 23, 29, 31, 37, 41)
DEFAULT_Q2_SEEDS = (7, 13, 19, 27, 33, 39, 45, 51)

Q1_CONFIG = HouseholdConfig()
Q2_CONFIG = HouseholdConfig(battery_capacity_bonus=10, max_episode_steps_scale=80)
Q2_STOCHASTICITY = StochasticityConfig(navigation_failure_prob=0.07, interaction_failure_prob=0.05)


@dataclass(frozen=True)
class EpisodeResult:
    """Evaluation result for a single episode."""

    success: bool
    seed: int
    primitive_steps: int
    option_history: tuple[str, ...]
    failure_reason: str | None


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate result across a seed suite."""

    successes: int
    total: int
    average_steps: float
    seeds: tuple[int, ...]
    failures: tuple[str | None, ...]


def exception_failure_reason(error: Exception) -> str:
    """Format a controller exception as an episode failure reason."""
    message = str(error)
    if message:
        return f"exception: {type(error).__name__}: {message}"
    return f"exception: {type(error).__name__}"


def failed_episode_from_exception(seed: int, error: Exception) -> EpisodeResult:
    """Create a failed episode result from a controller-construction exception."""
    failure_reason = exception_failure_reason(error)
    return EpisodeResult(
        success=False,
        seed=seed,
        primitive_steps=0,
        option_history=(failure_reason,),
        failure_reason=failure_reason,
    )


def make_env(question: str, *, seed: int, render: bool = False) -> HouseholdEnv:
    """Create and reset the environment variant for one assignment question."""
    render_mode = "human" if render else None
    if question == "q1":
        env = HouseholdEnv(
            config=Q1_CONFIG,
            stochasticity=StochasticityConfig(),
            render_mode=render_mode,
        )
    elif question == "q2":
        env = HouseholdEnv(
            config=Q2_CONFIG,
            stochasticity=Q2_STOCHASTICITY,
            render_mode=render_mode,
        )
    else:
        raise ValueError(f"Unknown question: {question}")
    env.reset(seed=seed)
    return env


def run_fsm_episode(
    controller: FiniteStateController,
    *,
    seed: int,
    render: bool = False,
) -> EpisodeResult:
    """Run one deterministic Q1 episode with run-to-completion options."""
    env = make_env("q1", seed=seed, render=render)
    option_library = OptionLibrary()
    option_history: list[str] = []
    stalled_options = 0

    try:
        controller.reset()
        while not (env.terminated or env.truncated):
            previous_steps = env.step_count
            observation = env.controller_observation()
            option_call = controller.select_option(observation)
            option = option_library.create(option_call)
            status = option.step(env)
            while status == OptionStatus.RUNNING and not (env.terminated or env.truncated):
                status = option.step(env)
            option_history.append(f"{option_call.option_name}({option_call.target_id}) -> {status}")
            controller.on_option_terminated(env.controller_observation(), option_call, status)
            progressed = env.step_count > previous_steps
            stalled_options = 0 if progressed else stalled_options + 1
            if stalled_options >= 3 and not env.success:
                env.terminated = True
                env.success = False
                env.failure_reason = "controller_stalled"
                break
    except Exception as error:
        env.terminated = True
        env.success = False
        env.failure_reason = exception_failure_reason(error)
        option_history.append(env.failure_reason)

    result = EpisodeResult(
        success=env.success,
        seed=seed,
        primitive_steps=env.step_count,
        option_history=tuple(option_history),
        failure_reason=env.failure_reason,
    )
    env.close()
    return result


def run_behavior_tree_episode(
    controller: BehaviorTreeController,
    *,
    seed: int,
    render: bool = False,
) -> EpisodeResult:
    """Run one stochastic Q2 episode with reactive BT ticking."""
    env = make_env("q2", seed=seed, render=render)
    option_history: list[str] = []
    stalled_ticks = 0

    try:
        controller.reset()
        context = BTContext(env=env, option_library=controller.option_library)
        while not (env.terminated or env.truncated):
            previous_steps = env.step_count
            status = controller.tick(env, context)
            progressed = env.step_count > previous_steps
            if context.active_option_call is not None:
                call = context.active_option_call
                option_history.append(f"{call.option_name}({call.target_id}) -> {status}")
            else:
                option_history.append(f"idle -> {status}")
            stalled_ticks = 0 if progressed else stalled_ticks + 1
            if stalled_ticks >= 2 and not env.success:
                env.terminated = True
                env.success = False
                env.failure_reason = "tree_stalled"
                break
    except Exception as error:
        env.terminated = True
        env.success = False
        env.failure_reason = exception_failure_reason(error)
        option_history.append(env.failure_reason)

    result = EpisodeResult(
        success=env.success,
        seed=seed,
        primitive_steps=env.step_count,
        option_history=tuple(option_history),
        failure_reason=env.failure_reason,
    )
    env.close()
    return result


def evaluate_fsm(
    controller_factory: Callable[[], FiniteStateController],
    *,
    seeds: tuple[int, ...] = DEFAULT_Q1_SEEDS,
    render: bool = False,
) -> EvaluationSummary:
    """Evaluate an FSM controller over the public deterministic suite."""
    results: list[EpisodeResult] = []
    for seed in seeds:
        try:
            controller = controller_factory()
        except Exception as error:
            results.append(failed_episode_from_exception(seed, error))
            continue
        results.append(run_fsm_episode(controller, seed=seed, render=render))
    return EvaluationSummary(
        successes=sum(result.success for result in results),
        total=len(results),
        average_steps=mean(result.primitive_steps for result in results),
        seeds=seeds,
        failures=tuple(result.failure_reason for result in results),
    )


def evaluate_behavior_tree(
    controller_factory: Callable[[], BehaviorTreeController],
    *,
    seeds: tuple[int, ...] = DEFAULT_Q2_SEEDS,
    render: bool = False,
) -> EvaluationSummary:
    """Evaluate a BT controller over the public stochastic suite."""
    results: list[EpisodeResult] = []
    for seed in seeds:
        try:
            controller = controller_factory()
        except Exception as error:
            results.append(failed_episode_from_exception(seed, error))
            continue
        results.append(run_behavior_tree_episode(controller, seed=seed, render=render))
    return EvaluationSummary(
        successes=sum(result.success for result in results),
        total=len(results),
        average_steps=mean(result.primitive_steps for result in results),
        seeds=seeds,
        failures=tuple(result.failure_reason for result in results),
    )
