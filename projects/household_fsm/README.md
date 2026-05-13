# household_fsm

A robot in a grid-world household needs to reach a goal cell while
navigating locked doors, distractor keys (wrong color), charging pads,
and a battery that drains every step it takes — including failed
actions. To get to the goal, the robot has to plan which door to
unlock, which key to grab, when to detour to a charger, and the order
of all of that. The environment is fully observable and deterministic:
every action does exactly what it says.

The controller is a **finite-state machine with run-to-completion
semantics**. Eight states (PLAN, NAVIGATE_TO_KEY, PICKUP_KEY,
NAVIGATE_TO_DOOR, UNLOCK_DOOR, NAVIGATE_TO_CHARGER, CHARGE,
GO_TO_GOAL); `select_option` is only called when the previous macro
option terminates, so each call picks one option, returns, and waits
for that option to finish. Two structural things are worth noting:
the **pre-empt check** at every entry that diverts to a charger if
the current target can't be safely reached, and an **internal
multi-transition loop** that lets several state transitions happen in
a single call when no movement is needed.

## Run-to-completion vs. reactive

In the deterministic version of this domain, options always succeed,
so re-evaluating the world on every tick is wasted work. A
run-to-completion FSM commits to a macro action (e.g. "navigate to
door K1"), returns, and only thinks again when that action finishes.
The state of the world will have advanced exactly as expected. This
makes the controller's logic very clean: each state is just a small
decision about what to do next given the current symbolic snapshot,
not a constant battle to be reactive to a stochastic environment.
(The reactive variant lives next door in [household_bt](../household_bt).)

## Pre-emptive charging

Every entry to `select_option` checks `can_safely_reach(target)` —
which estimates whether the robot's current battery is enough to
complete the trip and have some margin left. If not, the target gets
overridden to the nearest charger and the state changes to
NAVIGATE_TO_CHARGER. This is the only "non-greedy" piece of the
policy: it makes the controller willing to walk away from the goal
temporarily to ensure it can actually reach it. Without this check
the robot will deterministically fail any scenario where the optimal
route is too long for one charge.

## The multi-transition loop

Several state transitions in this FSM don't require any new option
to execute — they're pure bookkeeping. For example, the moment the
robot arrives at a key, NAVIGATE_TO_KEY should transition to
PICKUP_KEY, and then PICKUP_KEY can return the actual pickup option.
Without an internal loop, this would require two `select_option`
calls to resolve, with a wasted "no-op tick" in between. The
implementation wraps the whole body in a `while count < 50` loop so
the FSM can chain through any number of trivial transitions and only
return when it has an actual option to dispatch. The `count` bound
is a guard against an infinite-loop bug in the state graph itself.

## Running it

```bash
# In the repo root:
pip install -e shared/household_core
cd projects/household_fsm
python evaluate.py            # full suite
python evaluate.py --seed 17  # one episode
python evaluate.py --render   # open the visual debugger
```
