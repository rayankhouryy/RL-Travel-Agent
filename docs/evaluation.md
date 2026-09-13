# Evaluation and Performance Evidence

All results below are deterministic for the documented seed ranges. They are
diagnostics for this generated task distribution, not universal benchmarks.

## Baseline separation

The default fixed-seed evaluation shows that the heuristic completes the task
while random interaction does not:

| Metric | Random | Heuristic |
|---|---:|---:|
| Success | 0.00 | 0.99 |
| Reward | -0.880 | 4.924 |
| Realized satisfaction | 0.284 | 0.654 |
| Preference match | 0.223 | 0.355 |
| Quality | 0.476 | 0.724 |
| Mean steps | 38.70 | 14.24 |
| Invalid actions | 1.70 | 0.00 |

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
`0.2305` reward difference is `4.66%` of the non-refundable policy's mean
successful terminal utility (`4.9420`).

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
showed success decreasing from 98% at level 1 to 60% at level 4,
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
flexibility. The exact utility-maximizing itinerary changed in all 25 worlds,
with `3.72` distinct optima per four personas on average.

```bash
python -m evaluation.persona_sensitivity \
  --worlds 25 \
  --personas-per-world 4 \
  --difficulty 2
```

This demonstrates that hidden client preferences change the selection
objective itself rather than only changing the acceptance gate.
