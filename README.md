# RL Travel Agent

I built a Gymnasium environment for training and evaluating AI travel agents.
Agents plan complete trips, uncover partially observed client preferences,
operate in an evolving market, and recover from mid-trip disruptions.

The project includes the environment, synthetic travel market, client personas,
disruption engine, baseline policies, curriculum, evaluation harnesses, tests,
and CI. Its reward system measures how well an itinerary satisfies traveler
intent under changing constraints rather than rewarding agents for merely
selecting cheap or highly rated inventory.

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
   invariance, held-out outcomes, paired experiments, and an exact fixed-slot
   oracle test the environment rather than assuming the reward is correct.

## Headline evidence

| Question | Result | Interpretation |
|---|---:|---|
| Does competent behavior beat random? | 98.5% vs 0% success | The environment separates planning competence from random interaction |
| Does difficulty scale? | 98.0% to 56.5% heuristic success from level 1 to 4 | Harder levels require more information and recovery |
| Are masks sound? | 0 invalid heuristic actions across curriculum evaluation | Policy-visible validity matches environment transitions |
| Is shaping policy invariant? | Discounted-return spread below `1e-9` | Terminal potential and discounting are handled correctly |
| Does flexibility improve outcomes? | `+0.0341` mean satisfaction across 10 seed blocks | Refundability has consistent downstream value |
| Does reward fully price flexibility? | Non-refundable exceeds flexibility by `0.2305`, or 4.66% of mean successful terminal utility | The training objective underprices insurance value |
| Do hidden preferences materially change the optimum? | Using another persona's optimum loses 14.39% utility on average | Partial observability affects selection, not only acceptance |
| How close is the heuristic to an exact fixed-slot oracle? | Mean regret `0.0342` at level 2 and `0.0492` at level 4 | Selection is strong; interaction success still falls at higher difficulty |
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
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
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

The complete transition reward combines action cost, invalid-action cost,
potential-based shaping, and finish-dependent reward:

```math
r_t =
-c_{\mathrm{step}}
-c_{\mathrm{invalid}}\mathbf{1}[\mathrm{invalid}]
+\beta\left(\gamma\Phi(s_{t+1})-\Phi(s_t)\right)
+\mathbf{1}[\mathrm{success}]R_T
-\mathbf{1}[\mathrm{completion\ failure}]c_{\mathrm{failure}}.
```

The shaping term is applied on every transition, including the terminal
transition, where $\Phi(s_T)=0$. The test suite verifies that discounted return
is invariant across shaping scales `0.0`, `0.25`, and `1.0`. There is no
separate booking bonus, so book/remove churn cannot create positive return.

Successful terminal reward is a single weighted persona-dependent utility.
That utility combines theme fit, quality-floor fit, location, convenience,
one-sided budget fit, and preferred pace. Coherence and unresolved failures are
structural validity checks rather than successful-terminal reward terms.
Separately implemented realized satisfaction evaluates pace, quality-floor
failures, robustness, sunk cost, and post-disruption completion. It is exposed
only through `info`, not through policy observations or reward.

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

# Heuristic strategy-profile sweep under one shared reward
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

# Does the utility-maximizing itinerary change with hidden preferences?
python -m evaluation.persona_sensitivity \
  --worlds 25 \
  --personas-per-world 4

# Compare successful level-4 trips with and without recovery
python -m evaluation.recovery_analysis \
  --episodes 500 \
  --difficulty 4
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
docs/          project overview and detailed technical evidence
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
- potential-based shaping and separately implemented outcome evaluation
- adversarial policies, multi-seed analysis, throughput profiling, and oracle regret
- YAML configuration, text adapter, tests, and CI

The current scope deliberately excludes RL training, multi-city routing, live
booking APIs, and visualization so the repository stays focused on a reliable,
measurable environment. See [`docs/project-overview.md`](docs/project-overview.md)
for the system goals, capabilities, and extension path.
