"""Run the FSM controller on the deterministic household suite.

    python evaluate.py                  # full suite
    python evaluate.py --seed 17        # one episode
    python evaluate.py --render         # open the visual debugger
"""

import argparse

from controller import StudentFSMController
from household_core import (
    DEFAULT_Q1_SEEDS,
    evaluate_fsm,
    failed_episode_from_exception,
    run_fsm_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Run a single episode at this seed.")
    parser.add_argument("--render", action="store_true", help="Open the visual debugger.")
    args = parser.parse_args()

    if args.seed is not None:
        try:
            controller = StudentFSMController()
        except Exception as error:  # noqa: BLE001
            result = failed_episode_from_exception(args.seed, error)
        else:
            result = run_fsm_episode(controller, seed=args.seed, render=args.render)

        print(f"success: {result.success}")
        print(f"primitive steps: {result.primitive_steps}")
        print(f"failure reason: {result.failure_reason}")
        print("option history:")
        for entry in result.option_history:
            print(f"  {entry}")
        return

    summary = evaluate_fsm(
        StudentFSMController,
        seeds=DEFAULT_Q1_SEEDS,
        render=args.render,
    )
    print(f"successes: {summary.successes}/{summary.total}")
    print(f"average primitive steps: {summary.average_steps:.2f}")
    failures = [(s, f) for s, f in zip(summary.seeds, summary.failures, strict=True) if f]
    if failures:
        print("failures:")
        for seed, failure in failures:
            print(f"  seed {seed}: {failure}")
    else:
        print("failures: none")


if __name__ == "__main__":
    main()
