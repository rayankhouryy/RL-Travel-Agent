# Reward and Verification Design

## Transition reward

For a nonterminal transition:

$$
r_t =
-c_{\mathrm{step}}
-c_{\mathrm{invalid}}\mathbf{1}[\mathrm{invalid}]
+b_{\mathrm{booking}}\mathbf{1}[\mathrm{useful\ booking}]
+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

The potential is scaled feasible itinerary coverage:

$$
\Phi(s)=
\alpha\frac{
\mathbf{1}[\mathrm{flight}]
+\mathbf{1}[\mathrm{hotel}]
+\min(n_{\mathrm{activities}}/n_{\mathrm{required}},1)
}{3}.
$$

At a true terminal state, $\Phi(s_T)=0$. For a fixed trajectory:

$$
\sum_{t=0}^{T-1}\gamma^t
\left(
\gamma\Phi(s_{t+1})-\Phi(s_t)
\right)
=
-\Phi(s_0)+\gamma^T\Phi(s_T).
$$

The initial itinerary is empty, so $\Phi(s_0)=0$. With terminal potential also
zero, changing the shaping scale does not change the discounted ordering of
policies. The learner must use the same $\gamma$. Time-limit truncations retain
their nonzero potential and should bootstrap rather than being treated as
artificial terminals.

## Terminal reward

Successful completion receives:

$$
R_T =
w_pP+w_cC+w_bB+w_qQ+w_vV+w_rR-w_xX,
$$

where:

- $P$: latent preference match
- $C$: schedule coherence
- $B$: budget fit
- $Q$: quality
- $V$: convenience
- $R$: disruption recovery
- $X$: unresolved disruption violations

Budget fit targets expected spending rather than maximizing money left:

$$
B =
\exp\left(
-\frac{|\mathrm{spend}-\mathrm{target\ spend}|}
{\max(\mathrm{target\ spend},1)}
\right).
$$

Latent acceptance utility is:

$$
U_{\mathrm{latent}} =
\frac{\sum_i\eta_i u_i}{\sum_i\eta_i},
$$

where the hidden persona weights theme fit, quality, location, convenience,
and budget fit.

## Independently implemented realized outcome

The outcome evaluator first calculates:

$$
O =
0.30T+
0.22Q_f+
0.13L+
0.10C+
0.15P+
0.10B,
$$

where:

- $T$: thematic outcome
- $Q_f$: quality after the client-specific quality-floor penalty
- $L$: location quality
- $C$: convenience
- $P$: pace fit
- $B$: budget fit

Then:

$$
U_{\mathrm{realized}} =
\mathrm{clip}
\left(
0.85O+
0.15\rho_{\mathrm{robustness}}d_{\mathrm{tolerance}},
0,1
\right)
\left(1-0.3n_{\mathrm{unresolved}}\right)_+
\left(
1-1.5\frac{\mathrm{sunk\ cost}}{\mathrm{hard\ budget}}
\right)_+.
$$

Realized satisfaction is reported through `info` and does not enter the reward.
Tests replace each implementation independently and verify that reward and
realized outcome remain isolated.

## Reward-hacking defenses

| Exploit | Defense |
|---|---|
| Book nothing | Incomplete itinerary penalty and target-spend budget score |
| Always cheapest | Preference, quality floor, location, pace, and convenience |
| Always highest-rated | Hard budget and scheduling constraints |
| Finish immediately | Client acceptance and trip completion are required |
| Search forever | No potential increase; prices drift and inventory depletes |
| Farm feedback | Patience and preference revelation are finite |
| Rebook repeatedly | Cancellation fees and sunk-cost accounting |
| Ignore disruptions | Broken reservations stop satisfying completeness |
| Stack activities | Overlapping reservations are rejected |
| Avoid flexibility | Paired refundable/non-refundable evaluation |

## Verification

The suite checks:

- $\Phi(s_0)=0$
- terminal potential is zero
- discounted return is invariant across shaping scales
- every unaccepted terminal path has negative reward
- changing realized satisfaction cannot change reward
- changing latent utility cannot change realized satisfaction
- exploit policies produce measurably different behavior
- flexibility changes held-out outcomes and sunk losses on paired worlds

## Remaining reward gap

The non-refundable policy can receive slightly higher training reward while
producing lower realized satisfaction and greater sunk loss than the
flexibility policy. This is intentional evidence that the evaluator can expose
a weakness in the training objective rather than merely confirm it.

One possible future correction is a small proposal-time robustness component.
That change should only be accepted after rerunning the paired probe to verify
that it closes the gap without making refundability universally optimal.
