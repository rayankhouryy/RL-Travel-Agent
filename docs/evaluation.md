# Evaluation and Performance Evidence

All results below are deterministic for the documented seed ranges. They are
diagnostics for this generated task distribution, not universal benchmarks.

## Baseline separation

The default fixed-seed evaluation shows that the heuristic completes the task
while random interaction does not:

| Metric | Random | Heuristic |
|---|---:|---:|
| Success | 0.000 | 0.985 |
| Reward | -0.876 | 4.921 |
| Realized satisfaction | 0.282 | 0.653 |
| Preference match | 0.217 | 0.358 |
| Quality | 0.495 | 0.726 |
| Mean steps | 38.565 | 14.175 |
| Invalid actions | 1.805 | 0.000 |

Command:

```bash
python -m evaluation.evaluate --policy both --episodes 100
```

## Reward hacking

At difficulty 3:

| Policy | Reward | Success | Realized | Spend | Quality | Refundable share | Sunk fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | -1.431 | 0.00 | 0.141 | 2744 | 0.499 | 0.500 | 0.095 |
| Cheapest | 3.452 | 0.84 | 0.510 | 2119 | 0.394 | 0.368 | 0.007 |
| Quality maximizer | 4.228 | 0.92 | 0.566 | 3206 | 0.722 | 0.416 | 0.012 |
| Non-refundable | 4.386 | 0.94 | 0.548 | 2975 | 0.641 | 0.065 | 0.018 |
| Flexibility buyer | 4.148 | 0.92 | 0.576 | 3035 | 0.599 | 0.894 | 0.001 |
| Heuristic | 4.467 | 0.93 | 0.574 | 3150 | 0.646 | 0.410 | 0.009 |

Spearman correlation between reward rank and realized-outcome rank was `0.600`.

The paired 200-episode flexibility-minus-non-refundable probe found:

| Metric | Mean difference | Approximate 95% CI half-width |
|---|---:|---:|
| Reward | -0.1172 | 0.2650 |
| Realized satisfaction | +0.0361 | 0.0215 |
| Sunk-cost fraction | -0.0203 | 0.0030 |
| Refundable share | +0.7617 | 0.0332 |
| Spend | +186.22 | 69.04 |

Reward does not clearly prefer flexibility, while realized satisfaction and
sunk loss do.

The paired realized-outcome delta decomposes as:

| Outcome contribution | Mean difference |
|---|---:|
| Base itinerary outcome | -0.0160 |
| Explicit robustness | +0.0353 |
| Unresolved disruptions | +0.0000 |
| Sunk-cost effect | +0.0169 |

The explicit robustness term is the largest contributor by construction.
Lower realized sunk loss supplies a separate record-derived benefit.
Rounded components may differ from the reported aggregate by `0.0001`.

## Multi-seed robustness

Ten independently spaced blocks of 50 paired episodes produced:

| Metric | Mean block delta | Between-block standard deviation | Positive blocks |
|---|---:|---:|---:|
| Reward | -0.2305 | 0.1846 | 10% |
| Realized satisfaction | +0.0341 | 0.0140 | 100% |
| Sunk-cost fraction | -0.0150 | 0.0032 | 0% |
| Success | -0.0220 | 0.0319 | 10% |
| Spend | +142.60 | 63.08 | 100% |

For sunk-cost fraction, zero positive blocks means all ten blocks favored
flexibility through lower sunk loss.

Reward-component deltas explain the aggregate difference:

| Component | Flexibility minus non-refundable |
|---|---:|
| Client utility | -0.1916 |
| Failed-finish penalty | -0.0360 |
| Step cost and shaping | -0.0029 |

The flexibility policy pays an ex-ante premium and accepts somewhat weaker
inventory on the training objective. The separately implemented outcome score
values its lower sunk loss and stronger post-disruption robustness. The
non-refundable policy exceeds the flexibility policy's mean reward by `0.2305`,
which is `4.66%` of the non-refundable policy's mean successful terminal
utility (`4.9420`).

Across the same ten seed blocks, the `+0.0341` realized-outcome difference
decomposes into:

| Outcome contribution | Flexibility minus non-refundable |
|---|---:|
| Base itinerary outcome | -0.0145 |
| Explicit robustness | +0.0364 |
| Unresolved disruptions | +0.0000 |
| Sunk-cost effect | +0.0122 |

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
showed success decreasing from 98.0% at level 1 to 56.5% at level 4,
while rebooking requirements rose and mask-following invalid actions remained
zero:

| Metric | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---:|---:|---:|---:|
| Success | 0.980 | 0.935 | 0.855 | 0.565 |
| Reward | 4.861 | 4.531 | 3.947 | 2.067 |
| Realized satisfaction | 0.652 | 0.623 | 0.563 | 0.435 |
| Mean steps | 14.225 | 15.890 | 19.135 | 28.920 |
| Mean rebookings | 0.145 | 0.610 | 1.060 | 1.525 |

## Throughput

Measured locally with Python 3.9 on the assessment machine:

### Inventory scaling

| Inventory size | Reset latency | Steps/s | Approx. KiB/env |
|---:|---:|---:|---:|
| 24 | 0.320 ms | 4113 | 27.7 |
| 32 | 0.437 ms | 3346 | 33.3 |
| 64 | 0.877 ms | 1862 | 57.7 |

### Environment-count scaling

| Environments | Steps/s | Episodes/s | Approx. KiB/env |
|---:|---:|---:|---:|
| 1 | 3291 | 203.74 | 35.3 |
| 8 | 3302 | 203.42 | 33.4 |
| 32 | 3250 | 196.64 | 33.3 |

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

- increased one-environment throughput from about `2227` to roughly `3300` steps/s
- reduced the 10,000-step profiled runtime from `6.20` to about `4.41` seconds
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
| Oracle utility | 0.7136 |
| Heuristic utility | 0.6794 |
| Mean regret, all episodes | 0.0342 |
| Mean regret, successful episodes | 0.0325 |
| Mean regret, unsuccessful episodes | 0.0611 |
| Median regret | 0.0254 |
| 90th-percentile regret | 0.0737 |
| Heuristic success | 0.9400 |

Command:

```bash
python -m evaluation.oracle_regret \
  --episodes 50 \
  --difficulty 2

python -m evaluation.oracle_regret \
  --episodes 20 \
  --difficulty 4
```

At difficulty 4, a 20-episode run reports `0.0492` mean regret over all
episodes, `0.0448` on successes, `0.0558` on failures, and `0.60` heuristic
success. No failed episodes are dropped from the headline regret.

The oracle measures item selection under a fixed slot structure. It does not
optimize when to ask questions, propose, or sequence recovery. The modest
selection regret therefore says that the heuristic is strong at item choice;
the falling success rate identifies interaction policy as the larger source of
learning headroom.

## Persona-dependent optimal choices

For 25 fixed difficulty-2 worlds, four independently sampled latent personas
were evaluated against the same inventory, request, and hard-budget
flexibility. For each target persona, the evaluator measures the utility lost
when it receives another persona's exact optimal itinerary instead of its own.

| Metric | Value |
|---|---:|
| Mean cross-persona utility loss | 0.1034 |
| Median cross-persona utility loss | 0.0899 |
| 90th-percentile loss | 0.2171 |
| Mean loss relative to target optimum | 14.39% |
| Worlds with more than one optimum | 25/25 |
| Mean distinct optima across four personas | 3.72 |

```bash
python -m evaluation.persona_sensitivity \
  --worlds 25 \
  --personas-per-world 4 \
  --difficulty 2
```

The cross-persona loss, rather than the near-guaranteed argmax-change count, is
the evidence that hidden preferences materially change the selection objective
instead of only changing the acceptance gate.

## Successful recovery outcomes

Across 500 difficulty-4 heuristic episodes, successful trips separate as:

| Metric | No disruption | Recovered disruption |
|---|---:|---:|
| Successful episodes | 43 | 217 |
| Mean terminal utility | 4.9905 | 4.9115 |
| Terminal utility standard deviation | 0.4264 | 0.3606 |
| Mean realized outcome | 0.5863 | 0.5620 |
| Mean steps | 16.56 | 21.58 |
| Mean sunk-cost fraction | 0.0004 | 0.0229 |

```bash
python -m evaluation.recovery_analysis \
  --episodes 500 \
  --difficulty 4
```

Recovery is therefore more than a binary feasibility check: successful
disrupted episodes still show lower final utility, roughly five additional
steps, and greater sunk cost. The environment does not add a separate recovery
bonus; those consequences arise from the recovered itinerary and trajectory.
