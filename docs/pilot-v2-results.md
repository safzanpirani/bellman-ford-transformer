# Pilot v2 diagnosis-gate report

## Bottom line

**STOPPED before Phase 1 because the supplied v1 diagnosis is materially wrong.**
V1 finite-distance exact-match already used the attainable half-unit rounding bin.
The constructed model obtains zero loss and exact outputs under that metric.
The step-0 selection artifact and repeated fixed corpus are confirmed.
The claimed 55.7% unreachable rate is false; it combines unreachable nodes and sources.
Sentinel dominance is not established by the loss or saved v1 measurements.
The LayerNorm optimisation explanation was not tested in v1.
No v2 training, pre-registration, grid, or mechanistic analysis was run.
The recurrence question remains **inconclusive**; venue estimate: **not publishable**.

## Stop rule applied

The v2 handoff required all five causal claims about v1 to be verified before
changing the experiment, and required a stop if any claim was wrong. The audit
used the checkpointed repository at commit `32a1805`, the immutable v1
documents, the v1 implementation, and the archived v1 corpus and run records.

No file from v1 was edited. In particular,
`docs/pilot-results.md` and `docs/pilot-preregistration.md` remain unchanged.

## Diagnosis audit

| # | Supplied diagnosis | Verdict | Evidence |
|---|---|---|---|
| 1 | G2 was unreachable because a continuous readout essentially never equals an integer. | **CONTRADICTED** | `train.pilot.behaviour_metrics` accepts every finite prediction with `abs(prediction - target) < 0.5`; exact integer equality is not required. `tests/test_train_pilot.py` explicitly verifies that predictions such as 0.49 and 2.49 pass. This is the same open half-unit bin that rounding to an integer would create, apart from exact half-unit tie handling. |
| 2 | Small-init residual identity produced the initial DIST state and made step 0 win for free. | **CONFIRMED IN MECHANISM; NUMERIC DESCRIPTION WRONG** | All 12 selected checkpoints were step 0. Their exact predictions were confined to sources, unreachable nodes, and graphs with maximum finite hop 0. On validation, however, unreachable nodes were 880/2,498 = 35.23%, not 55.7%. The reported 55.72% node exact is 880 unreachable nodes plus 512 source nodes. |
| 3 | The `INF_B=161` sentinel dominated the loss because it was much larger than real distances. | **NOT ESTABLISHED** | The training split's largest finite target was 33, so the scale disparity is real. But the implementation divides both predictions and targets by 161 before Smooth-L1. Above 1.61 distance units, the per-node gradient magnitude is saturated and independent of error magnitude. Unreachable nodes were 7,487/20,487 = 36.55% of training nodes, not a majority. V1 did not save loss or gradient contributions split by reachability, so dominance cannot be inferred from target scale alone. |
| 4 | A 4,096-graph fixed corpus was cycled for about 31 epochs. | **CONFIRMED** | `FixedBatchStream` repeatedly permutes the same 4,096 examples. With batch size 32 and 4,000 updates, exposure is `4000 × 32 / 4096 = 31.25` corpus passes. |
| 5 | Lack of LayerNorm over 15 rounds caused a hard optimisation problem. | **UNTESTED** | The code and G1 tests confirm that v1 had no LayerNorm and that adding ordinary LayerNorm would break exact construction parity. V1 did not run a normalised comparator or any other intervention capable of identifying LayerNorm absence as the cause of its training trajectory. |

## Why claim 1 forces the stop

For integer target \(y\), v1 accepts a finite prediction \(p\) whenever

\[
|p-y| < 0.5.
\]

Thus `p = y + 0.49` is already an exact node prediction under v1. A direct
check through the public v1 metric gives 100% node and graph exact-match for
continuous predictions `[0.49, 2.49, 6.51]` against integer targets
`[0, 2, 7]`. Rounding those predictions also yields the targets.

More decisively, G1 loads the constructed weights into the same tied trainable
module and obtains exact outputs and zero final and per-round loss. Therefore
the G2 metric is attainable in the hypothesis space. V1 supplied no empirical
evidence that random-initialised gradient training could reach the 90%/98%
thresholds, but that is different from the metric being unreachable by
construction.

Changing the metric from strict half-unit error to `round(prediction) ==
target` would not repair the diagnosed problem. Except at exact half-unit
ties, it assigns the same finite predictions to the same integer target bins.
Proceeding on the premise that it makes regression exact-match attainable
would repeat the threshold-design error rather than correct it.

## Confirmed v1 artifacts

### Step-0 checkpoint selection

The checkpoint key was lexicographic on graph exact, node exact, negative MAE,
and earliest step. The small random residual updates left the input DIST
register nearly unchanged: zero at the source and `INF_B` elsewhere. On the
validation set:

- 880 of 2,498 nodes were unreachable;
- 512 were source nodes;
- their sum, 1,392/2,498, is 55.7246%, exactly the tied step-0 node score; and
- every selected v1 checkpoint was step 0.

This confirms that checkpoint selection rewarded the identity-like
initialisation. It does not show that the finite-distance exact metric itself
was unattainable.

### Fixed training corpus

The fixed and archived corpus, fixed batch stream, and 31.25 effective passes
are exactly as described. Fresh graph sampling could remove repeated-example
exposure in a future design, but it does not rescue the false diagnosis gate
for this requested v2.

## Sentinel-loss assessment

`INF_B` is 161. The theoretical maximum simple-path distance at `N_MAX=16`,
`W_MAX=10` is 150; the fixed v1 training split happened to have a maximum
finite target of 33, mean 3.98, and median 2.

The v1 elementary loss is

\[
\operatorname{SmoothL1}(p/161, y/161; \beta=0.01).
\]

The physical-error transition to the linear branch is 1.61. Beyond it,
errors of 10 have the same per-node gradient magnitude whether the target is
10 or 161. The target magnitude can dominate a scalar loss value when a
sentinel prediction is badly wrong, but it does not automatically dominate
the optimisation gradient. V1 lacks the reachability-split training records
needed to make the stronger causal claim.

Separating reachability from finite-distance regression is a coherent
alternative objective. It is not justified by the claim that sentinel scale
was already demonstrated to be the cause of v1's failure.

## Work not performed

The diagnosis stop occurred before architecture or objective changes.
Accordingly:

- no reachability loss or rounded metric was implemented;
- G1 was not modified or weakened;
- no feasibility-ladder rung was run;
- F1 and F2 were not reached;
- no v2 numeric threshold was chosen;
- `docs/pilot-v2-preregistration.md` was not created;
- none of the 12 grid runs was started; and
- no probe, attention, ablation, or causal analysis was run.

## Go/no-go

**NO-GO on the requested v2 as specified.** The binding diagnosis contains a
direct contradiction of the code and tests, and two additional causal claims
that v1 did not measure. The v2 recipe may contain individually reasonable
ideas, but executing it would violate the explicit instruction to stop when
the diagnosis is wrong.

The recurrence question remains unanswered. This diagnosis audit is not a
research result and has no publication venue.
