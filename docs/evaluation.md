# Evaluation and Performance Evidence

All results below are deterministic for the documented seed ranges. They are
diagnostics for this generated task distribution, not universal benchmarks.

## Baseline separation

The default fixed-seed evaluation shows that the heuristic completes the task
while random interaction does not:

| Metric | Random | Heuristic |
|---|---:|---:|
| Success | 0.00 | 1.00 |
| Reward | -0.392 | 11.878 |
| Realized satisfaction | 0.287 | 0.634 |
| Preference match | 0.218 | 0.353 |
| Quality | 0.507 | 0.727 |
| Mean steps | 38.30 | 14.21 |
| Invalid actions | 1.83 | 0.00 |

Command:

```bash
python -m evaluation.evaluate --policy both --episodes 100
```

## Reward hacking

At difficulty 3:

| Policy | Reward | Success | Realized | Spend | Quality | Refundable share | Sunk fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | -0.560 | 0.00 | 0.144 | 2738 | 0.498 | 0.501 | 0.093 |
| Cheapest | 9.089 | 0.73 | 0.507 | 2196 | 0.436 | 0.368 | 0.010 |
| Quality maximizer | 13.039 | 0.94 | 0.559 | 3179 | 0.719 | 0.404 | 0.012 |
| Non-refundable | 13.164 | 0.95 | 0.541 | 2967 | 0.645 | 0.074 | 0.017 |
| Flexibility buyer | 12.751 | 0.94 | 0.575 | 3042 | 0.603 | 0.858 | 0.000 |
| Heuristic | 13.104 | 0.93 | 0.561 | 3150 | 0.643 | 0.422 | 0.009 |

Spearman correlation between reward rank and realized-outcome rank was `0.486`.

The paired 200-episode flexibility-minus-non-refundable probe found:

| Metric | Mean difference | Approximate 95% CI half-width |
|---|---:|---:|
| Reward | -0.0304 | 0.7403 |
| Realized satisfaction | +0.0375 | 0.0219 |
| Sunk-cost fraction | -0.0184 | 0.0026 |
| Refundable share | +0.7618 | 0.0324 |
| Spend | +177.73 | 68.95 |

Reward does not clearly prefer flexibility, while realized satisfaction and
sunk loss do.

## Multi-seed robustness

Ten independently spaced blocks of 50 paired episodes produced:

| Metric | Mean block delta | Between-block standard deviation | Positive blocks |
|---|---:|---:|---:|
| Reward | -0.4371 | 0.4610 | 30% |
| Realized satisfaction | +0.0291 | 0.0139 | 100% |
| Sunk-cost fraction | -0.0157 | 0.0036 | 0% |
| Success | -0.0200 | 0.0298 | 20% |
| Spend | +125.61 | 74.20 | 90% |

For sunk-cost fraction, zero positive blocks means all ten blocks favored
flexibility through lower sunk loss.

Command:

```bash
python -m evaluation.multiseed \
  --blocks 10 \
  --episodes-per-block 50 \
  --difficulty 3
```

## Curriculum

The curriculum increases hidden information, required activity coverage,
scarcity, disruption count, and recovery horizon. Earlier fixed-seed evaluation
showed success decreasing from 99% at level 1 to 65% at level 4,
while rebooking requirements rose and mask-following invalid actions remained
zero.

## Throughput

Measured locally with Python 3.9 on the assessment machine:

### Inventory scaling

| Inventory size | Reset latency | Steps/s | Approx. KiB/env |
|---:|---:|---:|---:|
| 24 | 0.343 ms | 4104 | 27.7 |
| 32 | 0.440 ms | 3331 | 33.3 |
| 64 | 0.892 ms | 1870 | 57.7 |

### Environment-count scaling

| Environments | Steps/s | Episodes/s | Approx. KiB/env |
|---:|---:|---:|---:|
| 1 | 3340 | 206.43 | 35.3 |
| 8 | 3268 | 200.68 | 33.4 |
| 32 | 3279 | 198.05 | 33.3 |

These are sequential round-robin environments, not parallel workers. The
stable aggregate throughput shows low per-environment memory overhead but does
not claim CPU scaling.

Command:

```bash
python -m evaluation.benchmark \
  --steps 10000 \
  --resets 500 \
  --env-counts 1 8 32 \
  --inventory-sizes 24 32 64 \
  --profile
```

## Profile-guided optimization

Initial profiling showed that observation construction recomputed the pairwise
swap mask three times per step. Reusing one mask calculation per observation:

- increased one-environment throughput from about `2227` to `3340` steps/s
- reduced the 10,000-step profiled runtime from `6.20` to `4.33` seconds
- reduced swap-mask calls from `31,857` to `10,619`

The remaining dominant costs are target-mask construction, inventory feature
encoding, pairwise swap validation, and synthetic world generation. No further
optimization was applied because these costs are acceptable at the current
inventory size and additional caching would increase invalidation complexity.

## Exact fixed-slot oracle

The oracle enumerates:

- one available flight
- one available hotel
- exactly the curriculum-required number of non-overlapping activities

It has access to latent client utility and evaluates a static, fully observed,
no-disruption world. It is exact only within those explicitly stated
constraints.

Across 50 difficulty-2 episodes:

| Metric | Value |
|---|---:|
| Oracle utility | 0.7474 |
| Heuristic utility | 0.7038 |
| Mean regret | 0.0436 |
| Median regret | 0.0324 |
| 90th-percentile regret | 0.1093 |
| Heuristic success | 1.0000 |

Command:

```bash
python -m evaluation.oracle_regret \
  --episodes 50 \
  --difficulty 2
```

The result shows measurable optimization headroom without describing a greedy
or approximate solver as an oracle.
