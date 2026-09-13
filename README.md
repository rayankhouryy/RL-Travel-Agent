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

After client acceptance, `ADVANCE_TRIP` moves through the trip one simulated
day at a time and can expose hidden disruption events. `terminated` means the
episode reached either successful completion or explicit client rejection.
`info["client_accepted"]` and `info["trip_completed"]` distinguish success from
failure. `truncated` means the episode reached its configured step horizon.

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
    "action_type": spaces.Discrete(13),
    "source_index": spaces.Discrete(max_inventory),
    "target_index": spaces.Discrete(max_inventory),
})
```

Supported verbs:

- `SEARCH_FLIGHTS`, `SEARCH_HOTELS`, `SEARCH_ACTIVITIES`
- `SELECT_FLIGHT`, `SELECT_HOTEL`, `SELECT_ACTIVITY`
- `REMOVE_BOOKING`, `SWAP_BOOKING`, `REBOOK`
- `PROPOSE_ITINERARY`, `MESSAGE_CLIENT`
- `ADVANCE_TRIP`, `FINISH`

`source_index` identifies the booking being replaced. `target_index` identifies
the selected replacement or new inventory item. Unused parameters are set to
zero.

The observation includes an action-type mask, per-action target masks, and an
explicit pairwise `(source, target)` mask for swaps and disruption recovery.
These masks make variable inventory usable through fixed Gymnasium spaces
without hiding validity rules inside a particular policy implementation.

## Observation space

Observations contain fixed-size numeric tensors:

- normalized request features
- stated and revealed preference signals
- budget, booking, patience, acceptance, and episode progress
- curriculum level, required activities, trip day, and trip phase
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

### Reward equations

For a nonterminal transition, the environment returns:

$$
r_t =
-c_{\text{step}}
-c_{\text{invalid}}\mathbf{1}[\text{invalid}]
+b_{\text{booking}}\mathbf{1}[\text{useful booking}]
+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

The potential is scaled feasible itinerary coverage:

$$
\Phi(s)=
\alpha\frac{
\mathbf{1}[\text{flight}]
+\mathbf{1}[\text{hotel}]
+\min(n_{\text{activities}}/n_{\text{required}},1)
}{3}.
$$

At a true terminal state, the implementation sets
$\Phi(s_{\mathrm{terminal}})=0$. Therefore, for a fixed trajectory and a
learner using the same discount factor $\gamma$, the shaping terms telescope
in discounted return and do not change the optimal policy. Time-limit
truncations are not treated as artificial terminal states; learners should
bootstrap through them.

Successful terminal reward is:

$$
R_T =
w_pP+w_cC+w_bB+w_qQ+w_vV+w_rR-w_xX,
$$

where $P$ is latent preference match, $C$ is schedule coherence, $B$ is
budget fit, $Q$ is quality, $V$ is convenience, $R$ is recovery, and $X$ is
the number of unresolved disruption violations.

Budget fit is centered on the client's expected spend rather than on zero:

$$
B =
\exp\left(
-\frac{|\text{spend}-\text{target spend}|}
{\max(\text{target spend},1)}
\right).
$$

The latent acceptance utility is a normalized weighted combination:

$$
U_{\text{latent}} =
\frac{\sum_i \eta_i u_i}{\sum_i \eta_i},
$$

where the utilities cover theme fit, quality, location, convenience, and
budget fit, and the weights $\eta_i$ come from the hidden persona.

### Held-out realized satisfaction

Evaluation additionally computes:

$$
U_{\text{realized}} =
\operatorname{clip}
\left(
0.85U_{\text{latent}}+
0.15\rho_{\text{robustness}}d_{\text{tolerance}},
0,1
\right)
\left(1-0.3n_{\text{unresolved}}\right)_+
\left(1-1.5\frac{\text{sunk cost}}{\text{hard budget}}\right)_+.
$$

This quantity is reported in `info` but never enters the reward calculation.
An invariant test replaces `realized_satisfaction` with a constant and verifies
that the complete reward trace remains unchanged.

The metric is held out from optimization, but it is not fully independent:
it deliberately reuses latent client utility before applying robustness,
unresolved-disruption, and sunk-cost adjustments. A shared bug in the utility
components could therefore affect both measures. A production evaluator should
add an independently implemented outcome metric.

### Shaping invariance check

The test suite runs an identical deterministic policy and episode with shaping
scales `0.0`, `0.25`, and `1.0`, computes discounted return, and requires the
spread to remain below `1e-9`. It also verifies that $\Phi(s_0)=0$ after reset.

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
| Avoid paying for flexibility | Paired refundable/non-refundable policies are compared on held-out satisfaction and sunk cost |

## Disruptions

The disruption engine currently models:

- flight cancellation
- hotel overbooking
- weather closure

A flight cancellation can invalidate already-booked activities that become
unreachable after the changed arrival. The environment records the causal set
on the disruption and requires the affected reservations to be rebooked.
Disruptions are exposed during the simulated trip phase rather than while the
agent is still searching inventory.

## Curriculum

Reset with a difficulty from 1 through 4:

```python
observation, info = env.reset(seed=42, options={"difficulty": 3})
```

Higher levels increase required activity coverage, hide more latent preference
dimensions, tighten budgets, reduce client patience, increase scarcity, allow
more disruptions, and provide a longer horizon for recovery.

Compare the heuristic policy across all levels:

```bash
python -m evaluation.curriculum --episodes 200
```

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

## Reward-hacking diagnostics

Run adversarial policies and a paired fragility probe:

```bash
python -m evaluation.reward_hacking \
  --difficulty 3 \
  --episodes 100 \
  --probe-episodes 200
```

The policies intentionally optimize narrow proxies:

- `cheapest`
- `quality_maximizer`
- `nonrefundable`
- `flexibility_buyer`
- `heuristic`
- `random`

The harness ranks policies by reward and held-out satisfaction, reports their
Spearman rank correlation, and compares flexibility and non-refundable policies
on identical generated worlds.

One fixed-seed run produced:

| Policy | Reward | Success | Realized satisfaction | Spend | Quality | Refundable share | Sunk-cost fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | -0.604 | 0.00 | 0.165 | 2847 | 0.490 | 0.413 | 0.108 |
| Cheapest | 8.612 | 0.70 | 0.546 | 2469 | 0.435 | 0.340 | 0.010 |
| Quality maximizer | 12.060 | 0.86 | 0.548 | 3508 | 0.695 | 0.419 | 0.013 |
| Non-refundable | 11.293 | 0.82 | 0.517 | 3190 | 0.617 | 0.058 | 0.018 |
| Flexibility buyer | 11.562 | 0.85 | 0.589 | 3373 | 0.591 | 0.826 | 0.001 |
| Heuristic | 12.055 | 0.86 | 0.562 | 3475 | 0.619 | 0.394 | 0.010 |

The reward-versus-realized rank correlation was `0.714`, showing that reward is
directionally useful but does not perfectly order downstream outcomes.

In the paired 200-episode probe, flexibility minus non-refundable produced:

| Metric | Mean difference | Approximate 95% CI half-width |
|---|---:|---:|
| Reward | -0.6761 | 0.7716 |
| Realized satisfaction | +0.0302 | 0.0217 |
| Sunk-cost fraction | -0.0162 | 0.0026 |
| Refundable share | +0.7625 | 0.0342 |
| Spend | +158.65 | 67.26 |

The reward difference is inconclusive, while realized satisfaction improves
and sunk loss falls. This is evidence of a remaining reward gap: flexibility
has measurable downstream value that the training signal does not reliably
price.

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
visibility, terminal preconditions, exact discounted shaping invariance,
held-out metric isolation, search-spam behavior, cascading disruption effects,
pairwise masks, exploit-policy separation, fragility detection, and baseline
separation.

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
