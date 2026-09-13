# Reward and Verification Design

## Transition reward

The complete transition reward is:

```math
r_t =
-c_{\mathrm{step}}
-c_{\mathrm{invalid}}\mathbf{1}[\mathrm{invalid}]
+\beta\left(\gamma\Phi(s_{t+1})-\Phi(s_t)\right)
+\mathbf{1}[\mathrm{success}]R_T
-\mathbf{1}[\mathrm{completion\ failure}]c_{\mathrm{failure}}.
```

The potential is feasible itinerary coverage:

```math
\Phi(s)=
\frac{
\mathbf{1}[\mathrm{flight}]
+\mathbf{1}[\mathrm{hotel}]
+\min(n_{\mathrm{activities}}/n_{\mathrm{required}},1)
}{3}.
```

The shaping scale $\beta$ is configured separately from the potential. The
shaping term is applied on every transition, including the terminal transition,
where $\Phi(s_T)=0$. For a fixed trajectory:

```math
\sum_{t=0}^{T-1}\gamma^t
\beta\left(
\gamma\Phi(s_{t+1})-\Phi(s_t)
\right)
=
\beta\left(-\Phi(s_0)+\gamma^T\Phi(s_T)\right).
```

The initial itinerary is empty, so $\Phi(s_0)=0$. With terminal potential also
zero, changing the shaping scale does not change the discounted ordering of
policies. The learner must use the same $\gamma$. Time-limit truncations retain
their nonzero potential and should bootstrap rather than being treated as
artificial terminals.

## Terminal reward

Successful completion receives:

```math
R_T =
w_pU_{\mathrm{latent}}+w_bB+w_qQ+w_vV+w_rR,
```

where:

- $U_{\mathrm{latent}}$: persona-dependent latent client utility
- $B$: budget fit
- $Q$: quality
- $V$: convenience
- $R$: disruption recovery

Coherence and unresolved disruption failures are enforced structurally, so they
cannot vary on a valid successful finish and do not appear in $R_T$.

Budget fit penalizes only spending above the persona's expected target. Finding
an equally good itinerary below the target is not penalized:

```math
B =
\exp\left(
-\frac{\max(\mathrm{spend}-\mathrm{target\ spend},0)}
{\max(\mathrm{target\ spend},1)}
\right).
```

Latent acceptance utility is:

```math
U_{\mathrm{latent}} =
\frac{\sum_i\eta_i u_i}{\sum_i\eta_i},
```

where the hidden persona weights theme fit, quality, location, convenience,
and budget fit.

## Separately implemented realized outcome

The outcome evaluator first calculates:

```math
O =
0.30T+
0.22Q_f+
0.13L+
0.10C+
0.15P+
0.10B,
```

where:

- $T$: thematic outcome
- $Q_f$: quality after the client-specific quality-floor penalty
- $L$: location quality
- $C$: convenience
- $P$: pace fit
- $B$: budget fit

Then:

```math
U_{\mathrm{realized}} =
\mathrm{clip}
\left(
0.85O+
0.15\rho_{\mathrm{robustness}}d_{\mathrm{sensitivity}},
0,1
\right)
\left(1-0.3n_{\mathrm{unresolved}}\right)_+
\left(
1-1.5\frac{\mathrm{sunk\ cost}}{\mathrm{hard\ budget}}
\right)_+.
```

Higher $d_{\mathrm{sensitivity}}$ means the client values disruption protection
more strongly. Realized satisfaction is reported through `info` for evaluation;
it is not included in the policy observation or reward. Tests replace each
implementation separately and verify that reward and realized outcome remain
isolated. The two measures are not claimed to be statistically independent:
they evaluate the same simulated trip and intentionally share some primitives.

## Reward-hacking defenses

| Exploit | Defense |
|---|---|
| Book nothing | Successful completion structurally requires a complete itinerary |
| Always cheapest | Preference, quality floor, location, pace, and convenience |
| Always highest-rated | Hard budget and scheduling constraints |
| Finish immediately | Client acceptance and trip completion are required |
| Search forever | No potential increase; prices drift and inventory depletes |
| Farm feedback | Patience and preference revelation are finite |
| Book/remove churn | No booking bonus; shaping telescopes and step costs make the cycle negative |
| Rebook repeatedly | Cancellation fees and sunk-cost accounting |
| Ignore disruptions | Broken reservations stop satisfying completeness |
| Stack activities | Overlapping reservations are rejected |
| Avoid flexibility | Paired refundable/non-refundable evaluation |

## Verification

The suite checks:

- $\Phi(s_0)=0$
- terminal potential is zero
- terminal transitions include the final negative shaping term
- discounted return is invariant across shaping scales
- reward equals the sum of its reported components on every tested transition
- book/remove churn has negative discounted return
- every unaccepted terminal path has negative reward
- changing realized satisfaction cannot change reward
- changing latent utility cannot change realized satisfaction
- hidden persona weights change successful terminal reward
- exploit policies produce measurably different behavior
- flexibility changes held-out outcomes and sunk losses on paired worlds
- heuristic solutions are compared with an exact fixed-slot oracle

## Remaining reward gap

Across ten seed blocks, the flexibility policy produces higher realized
satisfaction and lower sunk loss but lower training reward. Component
decomposition shows why: it pays more and accepts modestly lower ex-ante
quality, budget, convenience, and latent-intent scores in exchange for
post-disruption robustness. This is a concrete objective tradeoff rather than
an unexplained aggregate mismatch.

One possible future correction is a small proposal-time robustness component.
That change should only be accepted after rerunning the paired probe to verify
that it closes the gap without making refundability universally optimal.
Any such component must use ex-ante information such as refundability,
cancellation penalties, and estimated risk rather than privileged knowledge of
future disruptions.
