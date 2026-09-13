# AI Travel Agent RL Environment

A Gymnasium environment for training and evaluating agents that plan trips,
respond to client feedback, and recover from travel disruptions.

The environment rewards satisfying the traveler's intent under changing
constraints, rather than merely selecting cheap or highly rated inventory. It
models the parts of travel planning that create meaningful decisions:

- correlated price, quality, location, convenience, and scarcity
- partially observed client preferences
- hard budget, availability, and scheduling constraints
- irreversible choices, cancellation costs, and finite client patience
- cancellations that can invalidate dependent itinerary items
- terminal quality measured separately from held-out realized satisfaction

The original assessment prompt is preserved in
[`docs/assessment-brief.md`](docs/assessment-brief.md).

## Installation

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
from travel_env import TravelAgentEnv
from travel_env.actions import ActionType

env = TravelAgentEnv()
observation, info = env.reset(seed=42)

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

`terminated` means the client accepted a complete itinerary and the agent
finished. `truncated` means the episode reached its configured step horizon.

## Environment model

Each episode samples:

1. A continuous latent client persona.
2. An under-specified observable request.
3. A correlated market of flights, hotels, and activities.
4. Availability and disruption risk.

Price is derived from latent quality, location, convenience, season, scarcity,
party size, and noise. This creates realistic but imperfect correlations:
expensive inventory is often better, but price is not a sufficient ranking
signal.

The simulator knows the client's complete utility weights and future stochastic
events. The agent sees only stated preferences, information revealed by client
feedback, searched inventory, current bookings, and active disruptions.

## Action space

The canonical action is a parameterized Gymnasium `Dict`:

```python
spaces.Dict({
    "action_type": spaces.Discrete(12),
    "source_index": spaces.Discrete(max_inventory),
    "target_index": spaces.Discrete(max_inventory),
})
```

Supported verbs:

- `SEARCH_FLIGHTS`, `SEARCH_HOTELS`, `SEARCH_ACTIVITIES`
- `SELECT_FLIGHT`, `SELECT_HOTEL`, `SELECT_ACTIVITY`
- `REMOVE_BOOKING`, `SWAP_BOOKING`, `REBOOK`
- `PROPOSE_ITINERARY`, `MESSAGE_CLIENT`, `FINISH`

`source_index` identifies the booking being replaced. `target_index` identifies
the selected replacement or new inventory item. Unused parameters are set to
zero.

The observation includes both an action-type mask and per-action target masks.
The masks make a variable inventory usable through fixed Gymnasium spaces
without hiding validity rules inside a particular policy implementation.

## Observation space

Observations contain fixed-size numeric tensors:

- normalized request features
- stated and revealed preference signals
- budget, booking, patience, acceptance, and episode progress
- a padded inventory slate
- visibility, booking, disruption, action, and target masks

Inventory features include category, relative price, quality, location,
convenience, thematic attributes, schedule, refundability, and availability.
Unsearched inventory is masked and represented by zero vectors.

## Structural constraints

The environment rejects impossible state transitions rather than making every
violation purchasable through a reward penalty:

- unavailable or unsearched inventory cannot be booked
- flight and hotel slots cannot be duplicated
- overlapping activities cannot be booked
- hard budget overruns are rejected
- swaps require a booked source and same-category replacement
- incomplete or unaccepted itineraries cannot terminate successfully
- disrupted reservations no longer count toward itinerary completeness

These rules keep the environment causal and reduce dependence on fragile reward
weights.

## Reward

Step reward combines:

- a small action cost
- potential-based progress shaping
- a small valid-booking signal
- invalid-action penalties
- a decomposed terminal objective

The potential is the fraction of required flight, hotel, and activity slots
filled. Repeated searching does not increase it, so search loops accumulate cost
instead of reward.

Terminal reward includes:

- latent preference match
- temporal coherence
- distance from persona-specific target spend
- quality and convenience
- disruption recovery
- unresolved constraint violations

Every step exposes `info["reward_components"]`. The environment also reports
`realized_satisfaction`, a held-out metric that includes robustness and
post-disruption outcomes. It is not identical to training reward.

## Reward-hacking defenses

| Potential exploit | Structural or reward defense |
|---|---|
| Book nothing to maximize money left | Budget score targets persona-specific expected spend; incomplete plans are penalized |
| Always choose the cheapest option | Preference, quality, location, and convenience contribute to utility |
| Always choose the highest-rated option | Hard budget and schedule constraints still apply; quality utility is bounded |
| Finish immediately | Completion requires a feasible itinerary and client acceptance |
| Search forever | Searches have a step cost and do not change the shaping potential |
| Farm repeated feedback | Client patience and preference revelation are finite and monotonic |
| Rebook repeatedly | Non-refundable inventory incurs cancellation costs |
| Ignore a cancellation | Unavailable reservations stop satisfying completeness and reduce realized satisfaction |
| Stack activities | Overlapping reservations are rejected structurally |

## Disruptions

The disruption engine currently models:

- flight cancellation
- hotel overbooking
- weather closure

A flight cancellation can invalidate already-booked activities that become
unreachable after the changed arrival. The environment records the causal set
on the disruption and requires the affected reservations to be rebooked.

## Baselines and evaluation

Run random and heuristic policies over the same seeded task distribution:

```bash
python -m evaluation.evaluate --policy both --episodes 200
```

Reported metrics include:

- episode success and itinerary completion
- reward and held-out realized satisfaction
- preference match, quality, convenience, and spend
- budget compliance and unresolved violations
- invalid actions, steps, rebookings, and recovery

The heuristic is intentionally simple. It searches all categories, scores
visible feasible inventory, builds a complete plan, revises activities after
feedback, rebooks disrupted reservations, obtains acceptance, and finishes.
Its purpose is to verify that competent behavior materially outperforms random
interaction for the intended reasons.

## Reward-profile experiment

```bash
python -m evaluation.reward_sweep --episodes 200
```

The sweep compares budget, balanced, and experience-oriented reward and policy
profiles. The output shows whether weighting changes produce measurable
differences in spend, quality, preference fit, convenience, satisfaction,
success, and reward.

## Configuration

The default configuration is in [`configs/default.yaml`](configs/default.yaml).
It controls episode length, inventory size, acceptance, disruption probability,
difficulty, cancellation costs, and reward weights.

```python
from travel_env import TravelAgentEnv
from travel_env.config import EnvironmentConfig

config = EnvironmentConfig.from_yaml("configs/default.yaml")
env = TravelAgentEnv(config)
```

Unknown YAML keys raise an error instead of being silently ignored.

## Testing

```bash
pytest
```

Tests cover deterministic resets, Gymnasium-compatible observations, inventory
visibility, terminal preconditions, search-spam shaping, cascading disruption
effects, and baseline separation.

## Connecting real APIs

The environment boundary should remain stable while the synthetic generator is
replaced by provider adapters. A production integration would additionally
need:

- immutable offer snapshots and provider-specific offer IDs
- quote expiration and explicit repricing transitions
- idempotency keys for booking and cancellation
- asynchronous confirmation and partial-failure states
- rate limits, timeouts, retries, and provider error taxonomies
- currency, tax, fee, timezone, and localization handling
- credential isolation and audit logging
- replayable recorded fixtures for deterministic training and evaluation

Training should continue against snapshots or simulators. Live APIs are useful
for data refresh and final evaluation, but their non-determinism and side effects
make them unsuitable as the only environment backend.

## Current scope

Implemented:

- Gymnasium environment and deterministic seeding
- correlated inventory and continuous client personas
- partial observability and deterministic preference feedback
- structural booking constraints
- decomposed reward and held-out satisfaction
- basic causal disruption propagation
- random and heuristic baselines
- fixed-seed evaluation and reward-profile comparison

Next priorities are richer dependency graphs, multi-city curricula, explicit
client communication intents, stronger adversarial reward tests, and broader
task-distribution coverage.
