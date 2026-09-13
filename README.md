# AI Travel Agent RL Environment

A Gymnasium environment for training and evaluating agents that plan trips,
negotiate partially observed client preferences, operate in an evolving market,
and recover from mid-trip disruptions.

The environment rewards satisfying traveler intent under changing constraints,
not merely selecting cheap or highly rated inventory.

## Episode lifecycle

```text
RESET
  |
  v
SEARCH -> BUILD -> PROPOSE
                    |
              +-----+------+
              |            |
          rejected      accepted
              |            |
              v            v
           REVISE     ADVANCE_TRIP
              |            |
              +------> DISRUPTION?
                           |
                    +------+------+
                    |             |
                   no            yes
                    |             |
                    v             v
             TRIP COMPLETE     REBOOK
                    ^             |
                    |         RE-PROPOSE
                    +-------------+
                    |
                    v
                  FINISH
```

## Design principles

1. **Constraints are transitions, not prices.** Budget overruns, unavailable
   inventory, overlapping activities, and invalid replacements are rejected
   rather than allowed for a penalty.
2. **Client intent is partially observed.** The simulator knows latent utility
   weights, quality floors, and preferred pace. The agent sees an
   under-specified request and learns more through finite feedback turns.
3. **The world changes causally.** Unbooked prices drift, visible inventory can
   disappear, booked quotes remain immutable, and disruptions invalidate
   dependent reservations.
4. **Reward quality is tested empirically.** Exploit policies, shaping
   invariance, held-out outcomes, paired experiments, and a privileged oracle
   test the environment rather than assuming the reward is correct.

## Headline evidence

| Question | Result | Interpretation |
|---|---:|---|
| Does competent behavior beat random? | 100% vs 0% success | The environment separates planning competence from random interaction |
| Does difficulty scale? | 99% to 65% heuristic success from level 1 to 4 | Harder levels require more information and recovery |
| Are masks sound? | 0 invalid heuristic actions across curriculum evaluation | Policy-visible validity matches environment transitions |
| Is shaping policy invariant? | Discounted-return spread below `1e-9` | Terminal potential and discounting are handled correctly |
| Does flexibility improve outcomes? | `+0.0291` mean satisfaction across 10 seed blocks | Refundability has consistent downstream value |
| Does reward fully price flexibility? | `-0.4371` mean reward difference | The training reward underprices that value |
| How close is the heuristic to a privileged oracle? | Mean utility regret `0.0436` | The heuristic is strong but leaves measurable headroom |
| How fast is the environment? | About `3.3k` steps/s at inventory size 32 | The Python implementation is suitable for local rollout experiments |

Results are deterministic for their documented seed ranges but are not claims
about every possible task distribution. See
[`docs/evaluation.md`](docs/evaluation.md) for commands, complete measurements,
confidence information, profiling, and oracle scope.

## Installation

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Quick start

```python
from travel_env import TravelAgentEnv
from travel_env.actions import ActionType

env = TravelAgentEnv()
observation, info = env.reset(
    seed=42,
    options={"difficulty": 2},
)

action = {
    "action_type": int(ActionType.SEARCH_FLIGHTS),
    "source_index": 0,
    "target_index": 0,
}

observation, reward, terminated, truncated, info = env.step(action)
```

The API follows Gymnasium:

```text
reset(seed, options) -> observation, info
step(action) -> observation, reward, terminated, truncated, info
```

`terminated` represents either successful completion or explicit client
rejection. `truncated` represents the curriculum-specific step horizon.

## Action and observation spaces

The canonical action is:

```python
spaces.Dict({
    "action_type": spaces.Discrete(13),
    "source_index": spaces.Discrete(max_inventory),
    "target_index": spaces.Discrete(max_inventory),
})
```

Actions cover search, selection, removal, swapping, rebooking, client
interaction, trip advancement, and completion.

Observations include:

- normalized request and trip state
- stated and revealed preferences
- a padded inventory tensor
- booking and disruption masks
- an action-type mask
- per-action target masks
- exact pairwise `(source, target)` masks for swaps and rebooking

The pairwise mask preserves a fixed Gymnasium interface while representing
state-dependent replacement validity.

## Structured and text interaction

Structured actions remain canonical, but a thin adapter lets an LLM use the
same environment:

```python
text = env.observation_to_text(observation)
action = env.action_from_text(
    "REBOOK source=12 target=18"
)
```

## Synthetic world

Each seeded episode generates:

- a continuous latent client persona
- a partially specified request
- correlated flights, hotels, and activities
- price, quality, location, convenience, schedule, and refundability tradeoffs
- enough feasible inventory for at least one valid itinerary

Refundable inventory carries a premium. Cheap inventory has useful residual
variation rather than being deterministically low quality. Unbooked offers
reprice and can disappear during deliberation, while booked quote prices remain
fixed for cancellation and refund accounting.

## Reward and outcome evaluation

The transition reward combines action cost, invalid-action cost, a small useful
booking signal, and potential-based shaping:

$$
r_t =
-c_{\mathrm{step}}
-c_{\mathrm{invalid}}\mathbf{1}[\mathrm{invalid}]
+b_{\mathrm{booking}}\mathbf{1}[\mathrm{useful}]
+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

True terminal states use $\Phi(s_T)=0$. The test suite verifies that discounted
return is invariant across shaping scales `0.0`, `0.25`, and `1.0`.

Terminal reward combines preference fit, coherence, budget fit, quality,
convenience, recovery, and unresolved violations. Separately implemented
realized satisfaction evaluates pace, quality-floor failures, robustness,
sunk cost, and post-disruption completion. Tests verify that neither
implementation calls into the other.

The full equations, assumptions, exploit table, and derivation are in
[`docs/reward-design.md`](docs/reward-design.md).

## Curriculum

| Level | Required activities | Visible preference dimensions | Maximum disruptions | Step limit |
|---|---:|---:|---:|---:|
| 1 | 1 | 3 | 0 | 35 |
| 2 | 2 | 2 | 1 | 45 |
| 3 | 3 | 2 | 2 | 60 |
| 4 | 4 | 1 | 3 | 75 |

Higher levels also tighten budgets, increase scarcity, reduce client patience,
and expose more recovery work.

## Commands

```bash
# Random and heuristic baselines
python -m evaluation.evaluate --policy both --episodes 200

# Reward/profile behavior sweep
python -m evaluation.reward_sweep --episodes 200

# Difficulty progression
python -m evaluation.curriculum --episodes 200

# Exploit policies and paired fragility probe
python -m evaluation.reward_hacking \
  --difficulty 3 \
  --episodes 100 \
  --probe-episodes 200

# Robustness across independent seed blocks
python -m evaluation.multiseed \
  --blocks 10 \
  --episodes-per-block 50

# Reset, rollout, memory, and profile measurements
python -m evaluation.benchmark \
  --steps 10000 \
  --resets 500 \
  --profile

# Exact fixed-slot oracle regret
python -m evaluation.oracle_regret \
  --difficulty 2 \
  --episodes 50
```

## Configuration

[`configs/default.yaml`](configs/default.yaml) controls inventory size, horizon,
acceptance, reward coefficients, shaping scale, price drift, depletion,
refundability premiums, cancellation costs, disruption probability, and
difficulty. Unknown keys raise an error rather than being ignored.

## Tests and CI

The suite covers deterministic resets, Gymnasium compatibility, mask
soundness, randomized transition invariants, budget accounting, quote
stability, schedule feasibility, terminal semantics, disruption cascades,
potential-shaping invariance, held-out metric isolation, exploit-policy
behavior, and text parsing.

GitHub Actions installs the package and runs the complete suite on Python 3.9
and 3.11.

## Repository map

```text
travel_env/    environment, state, generation, rewards, market, disruptions
policies/      random, heuristic, and exploit-specific policies
evaluation/    baseline, curriculum, reward, robustness, benchmark, and oracle tools
tests/         unit, invariant, randomized, and behavioral tests
configs/       editable environment and reward configuration
docs/          assessment brief and detailed technical evidence
```

## Scope

Implemented:

- Gymnasium lifecycle and deterministic seeding
- parameterized actions and relational masks
- partial observability and client feedback
- correlated synthetic inventory and evolving markets
- immutable booked quotes and sunk-cost accounting
- structural constraints and causal disruptions
- four-level curriculum
- potential-based shaping and independent outcome evaluation
- adversarial policies, multi-seed analysis, throughput profiling, and oracle regret
- YAML configuration, text adapter, tests, and CI

Deliberately excluded are RL training, multi-city routing, live booking APIs,
and visualization. Those would substantially increase risk without
strengthening the core environment-design argument.

The original prompt is preserved in
[`docs/assessment-brief.md`](docs/assessment-brief.md).
