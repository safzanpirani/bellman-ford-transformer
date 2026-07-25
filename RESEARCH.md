# Research plan: does recurrence select relaxation?

## Binding bottom line

The construction is not the research contribution. RASP, Tracr, and looped
transformer expressivity already establish that transformers can be programmed
to execute algorithms, and graph-algorithm constructions already exist.

The original precision-theory Track B+C is closed. The useful softmax sharpness
bound is published in [*Softmax is not Enough*
(arXiv:2410.01104)](https://arxiv.org/abs/2410.01104), while the proposed
finite-logit *exact* theorem mixed incompatible arithmetic models. It will not
be pursued, experimentally or theoretically.

The broad Track A headline is also occupied. [*Spectral Journey*
(arXiv:2502.08794)](https://arxiv.org/abs/2502.08794) trains shallow
decoder-only transformers on shortest paths and finds a spectral mechanism
rather than dynamic programming. Conversely, trained looped transformers
([arXiv:2409.15647](https://arxiv.org/abs/2409.15647)) and
sparsity-regularised shortest-path GNNs
([arXiv:2503.19173](https://arxiv.org/abs/2503.19173)) can recover iterative
algorithmic behaviour.

The sole live question is:

> **Does weight-tied recurrence decide whether a trained model learns
> relaxation rather than a shortcut?**

This is a pilot to answer that question, not a paper-production programme.

## Why this artifact makes the question identifiable

The repository supplies:

- a compact, hand-constructed Bellman–Ford circuit;
- exact state after every relaxation round from `verify.reference_states`;
- explicit register locations for distances and scratch state;
- known attention routing for edge reads and candidate gathers; and
- a complete tracer for interventions and ablations.

The trained comparator must use the same tokenisation, residual layout, register
map, and block operations. Loading the constructed weights into it must reproduce
the constructed model exactly. Without that result, failure to learn Bellman–Ford
would be indistinguishable from inability to represent Bellman–Ford.

## Validity gates

### G1 — hypothesis-space parity

Before any training:

1. construct the trainable tied model at `N_MAX = 16`;
2. load weights produced by the hand construction into it;
3. run the public graph interface and per-round state path; and
4. assert exact decoded outputs and zero training loss on valid graphs.

The primary architecture has no LayerNorm. LayerNorm rescales literal distance
registers and therefore breaks construction parity. A normalised model, if ever
tested, is a separate secondary architecture and cannot substitute for G1.

**Stop:** if G1 fails, do not train. Report that the comparison is invalid.

### G2 — competent task learning

Every architecture × supervision condition must reach the pre-registered
in-distribution exact-match threshold. A condition below the threshold is
reported as untrained and receives no mechanistic interpretation.

**Stop:** if no valid comparison remains after applying G2, report training
failure rather than analysing mechanisms.

### G3 — causal follow-up only

Causal intervention is permitted only when the pre-registered probe criterion
detects a diagonal round structure. Linear decodability is not evidence of use;
[*Represented Is Not Computed* (arXiv:2605.22488)](https://arxiv.org/abs/2605.22488)
demonstrates that failure mode directly.

**Stop:** if no probe diagonal appears, report the absence of round structure.
Do not search post hoc for a more favourable intervention target.

## Phase 1 — documentation corrections

- State the construction's prior-art boundary in `README.md`.
- Separate real softmax, approximate sharpness, machine underflow, and ReLU
  reduction.
- Record that float32 bit-exact attention is an implementation property.
- Preserve the prior-art verdicts in `docs/prior-art.md` and correct identifiers.

This phase is committed separately before implementation begins.

## Phase 2 — trainable twin

Implement `train/model.py` around the existing register map and token stream:

- sink token, then node tokens, then edge tokens;
- the same residual width and named register slices;
- the same attention and MLP parameter shapes;
- no positional embedding, dropout, or LayerNorm;
- **tied:** one trainable block applied for \(R\) rounds;
- **untied:** \(R\) distinct blocks with otherwise identical shapes; and
- `N_MAX = 16` in the training configuration so encoding can represent the OOD
  range even though training graphs use at most eight nodes.

Development follows G1 as a red-first integration test through the public model
interface. No looser “similar transformer” is an acceptable substitute.

## Phase 3 — immutable pre-registration

Before the first optimiser step, commit `docs/pilot-preregistration.md` with:

- exact architecture × supervision × seed conditions;
- graph distributions and fixed dataset generation;
- optimiser, schedule, batch size, step budget, selection rule, and checkpoint
  rule;
- metric formulas and aggregation;
- exact G2 and G3 thresholds;
- definitions of relaxation, shortcut, and ambiguous;
- mandatory random-initialisation and shuffled-label probe baselines;
- predictions for each condition; and
- the stopping/deviation policy.

The file is not edited after its commit. Deviations are logged in
`docs/pilot-results.md`.

## Phase 4 — pilot

### Conditions

The pilot is the fixed \(2 \times 2 \times 3\) grid:

| Axis | Values |
|---|---|
| Architecture | tied · untied |
| Supervision | final distances only · exact per-round targets |
| Seeds | three fixed seeds chosen in the pre-registration |

This is 12 runs. There is no capacity sweep and no normalised primary variant.

### Data

- Generate graphs through `verify.random_graph`.
- Train and validate on \(n \in [2,8]\).
- Hold out \(n \in [9,16]\) for size extrapolation.
- Stratify behavioural results by graph size and shortest-path hop count.
- Keep dataset seeds and graph identities fixed across model conditions.

### Measurements, in order

1. **Behaviour.** Exact distance-vector match by \(n\) and hop count, for
   in-distribution and OOD data, aggregated across seeds with uncertainty.
2. **Round-state probes.** At loop iteration \(k\), fit a linear probe for every
   exact reference state \(j\), producing the complete \(k \times j\) matrix.
   Run the identical procedure on random-initialisation and shuffled-label
   baselines.
3. **Attention agreement.** Compare learned attention against the constructed
   edge-to-source and node-to-candidate routing, allowing head permutation.
4. **Iteration ablation.** Run fewer rounds than trained. Relaxation-like
   behaviour should improve monotonically with added rounds; a fixed-depth
   shortcut need not.
5. **Causal intervention.** Only after G3, intervene on the probe-identified
   distance state and test the algorithmically predicted downstream change.

### Outcome classes

The pre-registration supplies numeric thresholds. Conceptually:

- **learned relaxation:** passes G2, exhibits a baseline-separated diagonal
  round-state structure, and passes the corresponding causal test;
- **shortcut:** passes G2 but lacks the round-state criterion, or exhibits state
  information that fails the causal test, with OOD/ablation behaviour reported
  descriptively;
- **ambiguous:** evidence conflicts, seed variation crosses a threshold, or the
  required causal test is underpowered;
- **untrained:** fails G2 and is not interpreted mechanistically.

If tied and untied models both lack round structure, the result is a controlled
replication of the shortcut result in a recurrent-compatible architecture, not
evidence for a hidden algorithm.

## Phase 5 — report

`docs/pilot-results.md` begins with a ten-line-or-shorter answer:
**yes**, **no**, or **inconclusive**.

It must contain:

- all 12 conditions and all registered seeds, including failures;
- seed variance and confidence intervals rather than selected runs;
- behavioural results by size and hop count;
- full \(k \times j\) probe matrices with both mandatory baselines;
- routing and iteration-ablation results;
- causal results only if G3 permitted them;
- every deviation from the pre-registration and its reason;
- a direct go/no-go recommendation; and
- an honest venue ceiling, including “workshop” or “not publishable” if that is
  what the evidence supports.

## Contribution boundary and stop rule

The maximum possible contribution is a controlled, causally tested comparison
of hand-constructed and trained tied/untied Bellman–Ford transformers in the
same hypothesis space. The project does not claim novelty for:

- compiling graph algorithms into transformers;
- showing that shortest-path transformers can learn shortcuts;
- using intermediate-state probes;
- softmax size dispersion; or
- exact ReLU min reductions.

Do not build a general benchmark or training framework, add algorithms, revive
precision theory, or tune past the registered pilot. A failed gate or a negative
result ends the corresponding work cleanly.
