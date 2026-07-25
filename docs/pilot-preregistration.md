# Pilot pre-registration: tied recurrence versus shortcut learning

Status: **frozen before the first optimiser step**

This document is immutable after its commit. Any implementation or execution
deviation will be recorded in `docs/pilot-results.md`; it will not be repaired
here retroactively.

## Question

> Does weight-tied recurrence decide whether a trained transformer learns
> Bellman–Ford relaxation rather than a shortcut?

The pilot compares tied and untied models in the exact residual architecture
that contains the hand-constructed solution. G1 has already passed at
`N_MAX=16`: commit `c124716` loads the constructed block into the tied
trainable module and obtains an identical full residual stream and zero
distance loss.

This is not a test of whether generic transformers can solve shortest paths,
whether intermediate states are linearly decodable in general, or whether
softmax can implement exact reduction.

## Registered conditions

The full grid is fixed at 12 runs:

| Condition ID | Architecture | Supervision | Model seeds |
|---|---|---|---|
| `tied-final` | one block reused for 15 rounds | final distances only | 17, 29, 43 |
| `tied-round` | one block reused for 15 rounds | every round boundary | 17, 29, 43 |
| `untied-final` | 15 distinct same-shaped blocks | final distances only | 17, 29, 43 |
| `untied-round` | 15 distinct same-shaped blocks | every round boundary | 17, 29, 43 |

All conditions use:

- `train.config.TRAIN_CONFIG`, with `N_MAX=16`, `D_MAX=8`,
  `W_MAX=10`, `R=15`, and the resulting register map;
- sink, node, and edge tokens from `bf_transformer.encode_graph`;
- the constructed sublayer order and parameter shapes;
- no LayerNorm, BatchNorm, dropout, positional embeddings, or AMP; and
- random initialisation, never constructed-weight initialisation.

The initialisation is the implementation committed before this document:
Xavier uniform with gain 0.1 for matrices and zero for vector parameters.

## Fixed data

### Generation

All graphs come from `bf_transformer.verify.random_graph`, with non-negative
integer weights and `allow_negative=False`. To include both dense and
longer-hop graphs without introducing a second generator, samples alternate
between:

- `D_MAX=8`, the primary configuration; and
- `D_MAX=2`, passed only to `random_graph` during generation and then encoded
  under the primary `D_MAX=8` register layout.

Graph sizes are uniform over the range assigned to each split. Sources remain
uniform as implemented by `random_graph`. Self-loops, parallel edges, isolated
nodes, zero-weight edges, and unreachable nodes are retained.

The data seed is `20260726`. One fixed generated corpus is shared by every
condition and seed:

| Split | Size | Node range | Use |
|---|---:|---:|---|
| train | 4,096 | 2–8 | optimiser updates |
| validation | 512 | 2–8 | checkpoint selection and G2 |
| ID test | 1,024 | 2–8 | final behavioural report |
| OOD behavioural test | 1,024 | 9–16, exactly 128 per size | size/hop report |
| OOD probe fit | 1,024 | 9–16, exactly 128 per size | fit linear probes |
| OOD probe test | 1,024 | 9–16, exactly 128 per size | evaluate probes/interventions |

The five non-training splits are disjoint from each other and from training.
The generated corpus and a manifest containing graph/source identities and a
SHA-256 digest will be saved with the results.

### Targets

`verify.reference_states(graph, source, 15)` supplies states 0 through 15.
Infinity is represented in losses and probes by `TRAIN_CONFIG.inf_b = 161`.
Only real node rows participate in losses and metrics; padding rows do not.

For hop stratification, each reachable node is assigned the minimum number of
edges among paths attaining its shortest distance. Reports contain node-level
bins `0`, `1`, `2`, `3`, `4`, and `5+`, plus `unreachable`. Graph-level hop
count is the maximum finite node hop count. Empty bins are reported as empty,
not merged after seeing results.

## Optimisation

Every run receives the same budget and data order:

- AdamW, learning rate `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`,
  weight decay `0`;
- 200-step linear warm-up, then cosine decay to `3e-5`;
- batch size 32;
- 4,000 optimiser steps;
- global gradient-norm clipping at 1.0;
- validation every 200 steps, including step 0;
- float32 throughout; and
- no early stopping or budget extension.

The train corpus is shuffled with data-order seed `9001` and cycled as needed.
Model seeds affect parameter initialisation only, so matched conditions see the
same batches in the same order.

The registered execution device is the RTX 3090 on `maints`. CUDA TF32 and AMP
are disabled. Deterministic PyTorch algorithms are requested. A device change,
unsupported deterministic operation, crash, non-finite loss, or restart is a
reportable deviation. A failed run is not replaced with a new seed.

### Loss

For a predicted distance \(p\) and bounded target \(y\), the elementary loss is

\[
\operatorname{SmoothL1}(p/161,\ y/161;\ \beta=0.01).
\]

- `final` conditions average this loss over real nodes after round 15.
- `round` conditions average it over real nodes and all post-block boundaries
  1 through 15. Targets naturally remain constant after a small graph has
  converged.

No loss is applied to scratch, structural, or attention registers.

Checkpoint selection is lexicographic:

1. highest validation graph exact-match;
2. then highest validation node exact-match;
3. then lowest validation mean absolute distance error; and
4. then earliest step.

The test sets are evaluated only after checkpoint selection.

## Behavioural metrics and G2

A node prediction is exact when:

- both prediction and target represent unreachable (`prediction >= 160.5`), or
- the target is finite and `abs(prediction - target) < 0.5`.

A graph is exact only when every real node is exact. No rounding is performed
before applying this rule.

For each checkpoint, report:

- graph exact-match and node exact-match;
- mean and median absolute distance error, with infinity represented by 161;
- each metric by `n`;
- node metrics by shortest-path hop bin; and
- graph metrics by maximum shortest-path hop bin.

### G2 threshold

A seed passes G2 only if its selected checkpoint obtains, on the fixed
validation set:

- graph exact-match at least 90%; and
- node exact-match at least 98%.

A condition is G2-valid only if all three registered seeds pass. Runs and
conditions that fail remain in the behavioural tables but receive no probes,
routing interpretation, ablation interpretation, or causal analysis.

If no tied/untied pair under the same supervision is G2-valid, the recurrence
question is **inconclusive due to training failure** and mechanism work stops.

## Probe analysis

Probe work is run only for G2-valid conditions.

### Features and targets

For every model seed, collect each real node token's full residual vector at
round boundaries \(k=0,\ldots,15\). For every boundary \(k\), fit a separate
linear ridge probe to every exact reference state \(j=0,\ldots,15\). This
produces a complete \(16\times16\) matrix; no cells are omitted.

Probes are fit on the fixed OOD probe-fit split and evaluated on the disjoint
OOD probe-test split. Features are standardised using fit-split mean and
standard deviation. Targets are bounded distances divided by 161. The probe
includes an intercept and uses fixed ridge coefficient `1e-3`; no coefficient
search is performed.

Each cell reports:

- \(R^2\);
- node exact-match after converting the prediction back to distance units; and
- mean absolute distance error.

The same real-node mask is used for every cell. Sources and unreachable nodes
are included, with separate subgroup metrics.

### Mandatory baselines

For each architecture and model seed:

1. **Random-init:** run the identical probe pipeline on the saved step-0 model.
2. **Shuffled-label:** fit to targets permuted across graph/node rows
   independently for each \(j\), using shuffle seed `100000 + model_seed`.

Neither baseline may be dropped because of poor conditioning or runtime.

### Registered diagonal criterion

Let active target rounds \(J\) be rounds \(j=1,\ldots,15\) for which at least
100 probe-test node targets change between reference states \(j-1\) and \(j\).
This definition is fixed before observing activations.

A model seed has a probe diagonal only if:

1. \(|J|\ge3\);
2. for at least 80% of \(j\in J\), the diagonal \(R^2[j,j]\) exceeds every
   cell \(R^2[k,j]\) with \(|k-j|\ge2\) by at least 0.05;
3. mean diagonal \(R^2\) over \(J\) exceeds the larger of the random-init and
   shuffled-label mean diagonals by at least 0.20; and
4. mean diagonal node exact-match over \(J\) is at least 90%.

A condition satisfies G3 only if at least two of its three seeds meet all four
criteria. A visible-looking heatmap that misses these thresholds is not called
a diagonal.

## Attention agreement

For G2-valid conditions, compute top-1 routing agreement on the OOD behavioural
test:

- edge-read agreement: an edge query selects its source node;
- gather agreement: a node query selects its edge in slot \(k\), or itself
  when the slot is empty.

Gather heads are matched to constructed slots by the permutation maximising
agreement; the permutation is selected on OOD probe-fit and frozen for OOD
test. Report per-head and mean agreement. Report the same metrics for the
saved random-init model.

Attention agreement is descriptive. It cannot by itself classify a model as
using Bellman–Ford because an alternative mechanism can share routing.

## Iteration ablation

For each G2-valid seed, evaluate prefixes of 0 through 15 rounds on ID and OOD
behavioural tests. The untied model uses the corresponding block prefix; the
tied model reuses its block.

Report graph exact-match and normalised distance MAE at every prefix. A curve
is labelled monotone only if normalised MAE has at most one adjacent increase
larger than 0.01. This label is descriptive and does not override probe or
causal criteria.

## G3 and causal intervention

Causal work is run only for conditions satisfying the registered diagonal
criterion.

For each qualifying seed:

1. use the fitted diagonal probe at active round \(k\);
2. select the first 256 probe-test graph/node/round cases, in manifest order,
   for which the node is non-source, has a finite positive distance, and
   decreasing that round's reference distance by 1 changes at least one final
   reference distance;
3. apply the minimum-L2 residual perturbation along the probe weight that
   lowers its decoded distance by exactly 1;
4. continue the remaining model rounds; and
5. compare the model's final distance-change vector with synchronous
   Bellman–Ford continued from the correspondingly modified state.

Report median cosine similarity of non-zero change vectors and sign agreement
on nodes whose reference output changes. The baseline applies equal-norm random
directions, ten per case, with random seed `200000 + model_seed`.

A seed passes the causal criterion only if:

- median cosine similarity is at least 0.80;
- sign agreement is at least 90%; and
- each exceeds its random-direction baseline by at least 0.20.

There is no causal claim if fewer than 128 eligible cases exist; that seed is
classified ambiguous rather than changing case selection.

## Model-level and condition-level classifications

A seed is:

- **learned relaxation** if it passes G2, the probe-diagonal criterion, and the
  causal criterion;
- **shortcut** if it passes G2 but fails the probe-diagonal criterion, or if it
  passes the probe criterion and fails the causal criterion;
- **ambiguous** if it passes G2 but a required threshold cannot be evaluated,
  fewer than 128 causal cases exist, or numerical evidence lands exactly on a
  threshold; and
- **untrained** if it fails G2.

A condition receives a label only when at least two seeds share it and the
third is not a contradictory non-ambiguous label. Otherwise the condition is
ambiguous.

## Answering the recurrence question

The pilot answer is:

- **yes** only if a G2-valid tied condition is labelled learned relaxation,
  its supervision-matched untied condition is G2-valid and labelled shortcut,
  and the 95% paired interval for their registered diagonal-score difference
  excludes zero; the tied condition must separately pass its registered causal
  criterion, while G3 forbids causal testing of an untied condition without a
  probe diagonal;
- **no** if at least one supervision-matched tied/untied pair is G2-valid and
  both conditions receive the same non-ambiguous label, with no other valid
  pair showing the registered “yes” pattern; and
- **inconclusive** otherwise, including training failure, conflicting seeds,
  insufficient active rounds, or underpowered intervention.

These labels will not be softened based on attention plots, individual graphs,
or single seeds.

## Uncertainty

Every result is shown per seed. Condition summaries include mean, sample
standard deviation, and a 95% hierarchical-bootstrap interval using 10,000
draws and seed `20260727`: resample the three model seeds with replacement,
then resample graphs with replacement within each selected seed. No single-run
number is used as the condition result.

## Registered predictions

These predictions are not gate thresholds:

| Condition | G2 prediction | Mechanism prediction |
|---|---|---|
| `tied-final` | pass | shortcut or ambiguous; no registered diagonal |
| `tied-round` | pass | learned relaxation |
| `untied-final` | pass | shortcut; no registered diagonal |
| `untied-round` | pass | diagonal decodability but weaker or failed causal criterion, hence ambiguous or shortcut |

The primary predicted contrast is `tied-round` versus `untied-round`. A failure
of these predictions is reported without replacement hypotheses.

## Stopping and deviations

- G1 has passed and will not be weakened.
- G2 is applied before any mechanistic analysis.
- G3 is applied before any causal intervention.
- No condition, seed, baseline, test graph, or metric may be dropped.
- No optimiser, learning rate, budget, threshold, initialisation, loss,
  checkpoint rule, or data distribution may be changed after results are seen.
- Implementation bugs are fixed only when demonstrated independently of model
  performance. The affected run is rerun with the same seed and both the bug
  and rerun are logged.
- Infrastructure failures are logged. A same-seed restart is allowed only from
  the last deterministic checkpoint and is reported.
- If the registered setup fails, the report says it failed. There is no tuning
  sweep after the pilot.
