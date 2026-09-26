# RL Travel Agent: Project Overview

This project is a complete reinforcement-learning environment for an AI travel
agent. It turns trip planning into a structured, reproducible decision problem
with realistic tradeoffs, hidden client preferences, changing inventory, and
operational disruptions.

## What the agent does

During an episode, the agent:

1. searches flights, hotels, and activities
2. builds an itinerary within budget and schedule constraints
3. presents the plan and incorporates client feedback
4. advances the trip through a changing market
5. recovers from cancellations, overbookings, and closures
6. completes the trip only when all structural requirements are satisfied

The challenge is multi-constraint optimization under uncertainty. Availability
changes, preferences are only partially visible, budgets have hard and soft
limits, and recovery decisions trade cost against traveler satisfaction.

## What I built

- **Gymnasium environment:** deterministic `reset` and `step` behavior with
  fixed observation and action spaces.
- **Synthetic travel market:** correlated pricing, quality, convenience,
  geography, availability, and refundability.
- **Client personas:** latent utility weights, quality floors, preferred pace,
  flexibility, and finite feedback.
- **Causal disruption engine:** cancellations and dependent booking failures
  that require valid rebooking sequences.
- **Reward system:** terminal utility, transition costs, potential-based
  shaping, and structural guards against common reward exploits.
- **Baseline policies:** random, heuristic, and exploit-specific strategies.
- **Evaluation suite:** curriculum runs, multi-seed robustness, reward probes,
  persona sensitivity, recovery analysis, throughput benchmarks, and exact
  fixed-slot oracle regret.
- **Configuration and quality controls:** YAML configuration, comprehensive
  tests, deterministic seeds, and GitHub Actions across supported Python
  versions.

## Why the environment is structured this way

Constraints are enforced in transitions rather than left for reward penalties
to clean up. Invalid swaps, overlapping activities, unavailable inventory, and
hard-budget violations do not mutate state. This keeps the learned task aligned
with the actual travel workflow.

Client intent remains partially observed so interaction has value. The agent
starts from an under-specified request, reveals preferences through feedback,
and must decide when it has enough information to commit.

The market evolves causally. Unbooked offers can reprice or disappear, while
accepted booking prices remain fixed for cancellation and refund accounting.
Disruptions invalidate reservations and their dependencies instead of simply
subtracting reward.

## Evidence that it works

The included heuristic reaches 98.5% success in the baseline comparison while
the random policy reaches 0%. Success falls from 98.0% at curriculum level 1 to
56.5% at level 4 as hidden information, scarcity, and recovery requirements
increase. Across curriculum evaluation, the heuristic produces no invalid
masked actions.

Reward and outcome quality are checked with exploit policies, shaping
invariance tests, held-out satisfaction metrics, paired flexibility
experiments, persona-sensitivity analysis, and comparison with an exact
fixed-slot oracle. Full commands, seed ranges, measurements, and limitations
are documented in [`evaluation.md`](evaluation.md) and
[`reward-design.md`](reward-design.md).

## Extension path

The next major additions would be a training pipeline, multi-city routing, live
travel-provider adapters, richer natural-language interaction, parallel vector
environments, and visualization. The current repository keeps those concerns
outside the core so the environment remains deterministic, testable, and easy
to evaluate.
