# household_bt

Same household task as [household_fsm](../household_fsm) — robot,
locked doors, keys, chargers, finite battery, goal cell — but now the
macro options can *fail*. An `unlock_door` might slip; a `pickup_key`
might miss. The robot needs to recover gracefully without explicit
"I'm in recovery mode" bookkeeping.

The controller is a **reactive behavior tree** that ticks once per
primitive timestep. Two compositional primitives do all the work:
**Sequence** (run children in order until one fails) and **Fallback**
(run children in order until one succeeds), with **ConditionNode** and
**ActionNode** leaves. Earlier children win in a Fallback, so priority
is structural — the locked-door branch sits above `go_to_goal` and
always pre-empts it whenever a door blocks the path. The whole tree is
essentially stateless aside from a one-bit `recovering` latch that
resets each episode, which is exactly why it handles option failures
gracefully: every tick re-evaluates the world from scratch, so a
failure on tick N just means the next tick falls through to the right
branch automatically.

## Why a BT for the stochastic version

In a deterministic world an FSM is fine — commit to a macro action,
let it finish, plan the next one (see [household_fsm](../household_fsm)).
Once macros can fail, an FSM needs explicit recovery states: "I was
trying to unlock door D1 and it failed; now I'm in
HANDLE_UNLOCK_FAILURE state; what do I do?" Multiply by every option
type and you have an N² state explosion.

A behavior tree dodges this by making the *whole policy* implicitly
reactive. There's no "I'm currently in state X" stored anywhere —
every tick walks the tree from the root and decides based on the
current observation. An unlock that failed leaves the door still
locked, so on the next tick the `locked_door?` condition succeeds
again, `ensure_key` succeeds (we still have the key), and `unlock_door`
runs again. The recovery is implicit in the tree's structure.

## The tree

```
Fallback
├── Sequence "handle locked door"
│   ├── locked_door?           condition
│   ├── ensure_key             pickup if not holding
│   ├── ensure_at("door")      charge if needed → navigate
│   └── unlock_door            action
└── go_to_goal                 default branch
```

Earlier children win, so the locked-door handler always runs first if
a door is in the way; the `go_to_goal` fallback only fires when there's
no pending door. `ensure_at(target)` is itself a small subtree:

```
Sequence
├── Fallback                            # charge if needed
│   ├── need_charge?  (condition)
│   └── Sequence: go to charger; charge
└── Fallback                            # then navigate
    ├── already_at?  (condition)
    └── navigate_to(target)
```

The condition gating each branch makes the whole thing self-correcting
under failure: if a navigation gets interrupted, the next tick
re-checks `already_at?` and either continues navigation or falls
through to the next sequence step.

## Why the `recovering` latch exists

The tree is essentially stateless — but one bit of memory across
ticks turns out to be useful when an option failure could otherwise
make the tree oscillate between two branches. The `recovering` latch
holds a recovery decision in place for the duration of the option it
chose, then resets. It's the single "remember this for now" piece of
the policy.

## Running it

```bash
# In the repo root:
pip install -e shared/household_core
cd projects/household_bt
python evaluate.py            # full suite
python evaluate.py --seed 27  # one episode
python evaluate.py --render   # open the visual debugger
```
