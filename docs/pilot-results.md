# Pilot results: tied recurrence versus shortcut learning

## Bottom line

**Answer: INCONCLUSIVE due to training failure.**
G1 passed: the tied trainable twin exactly contains the constructed circuit.
All 12 registered runs completed, but all 12 failed G2.
Every selected checkpoint was step 0; trained checkpoints lowered MAE but worsened exact-match.
No tied/untied pair was task-competent, so the recurrence question was not tested.
The probe, attention, ablation, and causal stages were not run, as registered.
**Go/no-go: NO-GO on further Track A claims from this pilot.**
**Venue estimate: not publishable.**

## Scope and execution

The frozen protocol is [pilot-preregistration.md](pilot-preregistration.md),
commit `7e9ca30`. The 12 runs used commit `0736737` on the RTX 3090 host
`maints`, with PyTorch `2.11.0+cu126`, CUDA, float32, TF32 disabled, AMP
disabled, and deterministic algorithms requested. Every run reached 4,000
steps without a crash, restart, non-finite loss, seed replacement, or budget
change.

The fixed corpus had SHA-256
`1f586fa74ae874d01b0e9632944f622c162a64f4101cc2fdea49c5518dc309d8`.
It contained the registered 4,096 training, 512 validation, 1,024 ID test,
1,024 OOD behavioural, 1,024 OOD probe-fit, and 1,024 OOD probe-test graphs.

The reported intervals are the registered 10,000-draw hierarchical bootstrap
with seed `20260727`: model seeds were resampled first, then graphs within each
selected seed. Table entries of the form `mean ± SD [low, high]` contain the
condition mean, sample standard deviation across the three model seeds, and
the 95% bootstrap interval. Exact-match values are percentages.

## Validity gates

| Gate | Result | Consequence |
|---|---|---|
| G1: constructed weights load into the tied twin with zero loss and identical full residual stream | **PASS** at commit `c124716` | A trained failure is falsifiable: the exact circuit is in the tied hypothesis space. |
| G2: every seed in a condition reaches at least 90% validation graph exact and 98% node exact | **FAIL** for all four conditions and all 12 seeds | Every seed is classified `untrained`; no mechanism label is permitted. |
| G3: a registered probe diagonal appears in at least two seeds | **NOT REACHED** | Causal intervention is prohibited. |

The primary model remained LayerNorm-free. No normalised secondary variant was
run.

### G2 by seed

Each validation cell is `selected step; graph exact / node exact`, in percent.

| Condition | Seed 17 | Seed 29 | Seed 43 | Seed passes | Condition |
|---|---:|---:|---:|---:|---|
| `tied-final` | `0; 25.20 / 55.72` | `0; 25.20 / 55.72` | `0; 25.20 / 55.72` | 0/3 | G2 fail |
| `tied-round` | `0; 25.20 / 55.72` | `0; 25.20 / 55.72` | `0; 25.20 / 55.72` | 0/3 | G2 fail |
| `untied-final` | `0; 25.20 / 55.72` | `0; 25.20 / 55.72` | `0; 23.44 / 55.20` | 0/3 | G2 fail |
| `untied-round` | `0; 25.20 / 55.72` | `0; 25.20 / 55.72` | `0; 23.44 / 55.20` | 0/3 | G2 fail |

This was not a near miss: the best seed was 64.80 percentage points below the
graph threshold and 42.28 points below the node threshold.

## Behavioural results

Because checkpoint selection was lexicographic on validation exact-match, all
selected checkpoints were the random initialisations at step 0. For a fixed
architecture and seed, `final` and `round` therefore have byte-identical
validation, ID, OOD, and per-graph records. Their independently trained
trajectories differ, but their selected behavioural results do not.

The non-zero step-0 scores are not task competence. The initial models
preserved the source value and predicted the unreachable sentinel for almost
every other node. That exactly solves graphs whose only reachable node is the
source while failing every graph with a reachable non-source node.

### Overall metrics

| Condition | Split | Graph exact | Node exact | MAE | Median AE |
|---|---|---:|---:|---:|---:|
| `tied-final` | validation | 25.20 ± 0.00 [23.05, 27.34] | 55.72 ± 0.00 [53.99, 57.48] | 68.60 ± 0.00 [65.89, 71.29] | 0.06 ± 0.04 [0.02, 0.09] |
| `tied-final` | ID test | 23.44 ± 0.00 [21.94, 24.93] | 55.45 ± 0.00 [54.27, 56.66] | 69.03 ± 0.00 [67.18, 70.87] | 0.06 ± 0.04 [0.02, 0.09] |
| `tied-final` | OOD test | 21.68 ± 0.00 [20.21, 23.14] | 53.70 ± 0.00 [52.41, 54.98] | 70.93 ± 0.01 [68.96, 72.90] | 0.14 ± 0.09 [0.05, 0.23] |
| `tied-round` | validation | 25.20 ± 0.00 [23.05, 27.34] | 55.72 ± 0.00 [53.99, 57.48] | 68.60 ± 0.00 [65.89, 71.29] | 0.06 ± 0.04 [0.02, 0.09] |
| `tied-round` | ID test | 23.44 ± 0.00 [21.94, 24.93] | 55.45 ± 0.00 [54.27, 56.66] | 69.03 ± 0.00 [67.18, 70.87] | 0.06 ± 0.04 [0.02, 0.09] |
| `tied-round` | OOD test | 21.68 ± 0.00 [20.21, 23.14] | 53.70 ± 0.00 [52.41, 54.98] | 70.93 ± 0.01 [68.96, 72.90] | 0.14 ± 0.09 [0.05, 0.23] |
| `untied-final` | validation | 24.61 ± 1.01 [22.27, 26.89] | 55.55 ± 0.30 [53.76, 57.31] | 68.60 ± 0.00 [65.92, 71.35] | 0.02 ± 0.01 [0.01, 0.02] |
| `untied-final` | ID test | 22.85 ± 1.01 [21.06, 24.61] | 55.31 ± 0.26 [54.04, 56.51] | 69.03 ± 0.00 [67.18, 70.94] | 0.02 ± 0.01 [0.01, 0.03] |
| `untied-final` | OOD test | 21.68 ± 0.00 [20.21, 23.14] | 53.70 ± 0.00 [52.42, 54.95] | 70.92 ± 0.00 [68.99, 72.89] | 0.04 ± 0.02 [0.01, 0.05] |
| `untied-round` | validation | 24.61 ± 1.01 [22.27, 26.89] | 55.55 ± 0.30 [53.76, 57.31] | 68.60 ± 0.00 [65.92, 71.35] | 0.02 ± 0.01 [0.01, 0.02] |
| `untied-round` | ID test | 22.85 ± 1.01 [21.06, 24.61] | 55.31 ± 0.26 [54.04, 56.51] | 69.03 ± 0.00 [67.18, 70.94] | 0.02 ± 0.01 [0.01, 0.03] |
| `untied-round` | OOD test | 21.68 ± 0.00 [20.21, 23.14] | 53.70 ± 0.00 [52.42, 54.95] | 70.92 ± 0.00 [68.99, 72.89] | 0.04 ± 0.02 [0.01, 0.05] |

### What optimisation did

This table is descriptive and does not replace the registered checkpoint
rule. Each entry aggregates the corresponding per-seed post-step-0 extremum;
the extrema need not occur at the same step.

| Condition | Maximum validation graph exact | Maximum node exact | Minimum MAE | Step-4000 graph exact | Step-4000 node exact | Step-4000 MAE |
|---|---:|---:|---:|---:|---:|---:|
| `tied-final` | 18.10 ± 3.47 | 45.29 ± 5.98 | 30.59 ± 0.06 | 2.73 ± 2.84 | 28.50 ± 1.92 | 30.65 ± 0.10 |
| `tied-round` | 7.62 ± 5.87 | 31.12 ± 7.51 | 30.90 ± 0.24 | 1.89 ± 3.27 | 18.19 ± 10.48 | 30.99 ± 0.28 |
| `untied-final` | 12.63 ± 5.81 | 43.50 ± 4.52 | 30.56 ± 0.02 | 2.54 ± 1.22 | 27.62 ± 0.94 | 30.64 ± 0.04 |
| `untied-round` | 11.65 ± 5.54 | 43.09 ± 4.59 | 30.23 ± 0.72 | 2.41 ± 1.08 | 27.78 ± 2.64 | 30.34 ± 0.78 |

Optimisation roughly halved MAE while making exact-match worse than the
sentinel-heavy initialisation. Under the registered checkpoint rule, that
selects step 0. It does not support a claim that training found either a
shortest-path shortcut or iterative relaxation.

### Metrics by graph size

These tables report `graph exact / node exact / MAE / median AE`, each as
`mean ± seed SD`. Because the selected `final` and `round` records are
identical, columns are collapsed by architecture; each value applies to both
supervision conditions.

#### ID test, n = 2–8

| n | Tied | Untied |
|---:|---:|---:|
| 2 | 31.25 ± 0.00 / 65.62 ± 0.00 / 53.99 ± 0.01 / 0.02 ± 0.02 | 31.25 ± 0.00 / 65.62 ± 0.00 / 53.99 ± 0.00 / 0.00 ± 0.00 |
| 3 | 22.38 ± 0.00 / 58.04 ± 0.00 / 65.80 ± 0.00 / 0.03 ± 0.00 | 22.38 ± 0.00 / 57.96 ± 0.13 / 65.80 ± 0.00 / 0.01 ± 0.00 |
| 4 | 25.17 ± 0.00 / 56.29 ± 0.00 / 68.19 ± 0.00 / 0.03 ± 0.02 | 22.45 ± 4.71 / 55.56 ± 1.28 / 68.18 ± 0.00 / 0.01 ± 0.01 |
| 5 | 20.57 ± 0.00 / 55.74 ± 0.00 / 68.78 ± 0.00 / 0.04 ± 0.03 | 20.57 ± 0.00 / 55.74 ± 0.00 / 68.78 ± 0.00 / 0.01 ± 0.01 |
| 6 | 21.05 ± 0.00 / 54.50 ± 0.00 / 70.60 ± 0.01 / 0.05 ± 0.04 | 19.74 ± 2.28 / 54.17 ± 0.57 / 70.60 ± 0.00 / 0.02 ± 0.01 |
| 7 | 23.29 ± 0.00 / 54.11 ± 0.00 / 70.69 ± 0.01 / 0.07 ± 0.05 | 23.29 ± 0.00 / 54.11 ± 0.00 / 70.69 ± 0.00 / 0.02 ± 0.01 |
| 8 | 20.53 ± 0.00 / 53.39 ± 0.00 / 71.75 ± 0.01 / 0.08 ± 0.06 | 20.53 ± 0.00 / 53.39 ± 0.00 / 71.74 ± 0.00 / 0.03 ± 0.01 |

#### OOD test, n = 9–16

| n | Tied | Untied |
|---:|---:|---:|
| 9 | 22.66 ± 0.00 / 56.86 ± 0.00 / 66.75 ± 0.01 / 0.08 ± 0.06 | 22.66 ± 0.00 / 56.86 ± 0.00 / 66.74 ± 0.00 / 0.02 ± 0.01 |
| 10 | 26.56 ± 0.00 / 56.72 ± 0.00 / 66.62 ± 0.01 / 0.10 ± 0.07 | 26.56 ± 0.00 / 56.72 ± 0.00 / 66.61 ± 0.00 / 0.02 ± 0.02 |
| 11 | 20.31 ± 0.00 / 54.90 ± 0.00 / 69.11 ± 0.01 / 0.11 ± 0.07 | 20.31 ± 0.00 / 54.90 ± 0.00 / 69.10 ± 0.00 / 0.03 ± 0.02 |
| 12 | 25.00 ± 0.00 / 55.34 ± 0.00 / 68.64 ± 0.01 / 0.12 ± 0.08 | 25.00 ± 0.00 / 55.34 ± 0.00 / 68.63 ± 0.00 / 0.03 ± 0.02 |
| 13 | 20.31 ± 0.00 / 53.12 ± 0.00 / 71.66 ± 0.01 / 0.15 ± 0.10 | 20.31 ± 0.00 / 53.12 ± 0.00 / 71.65 ± 0.00 / 0.04 ± 0.02 |
| 14 | 12.50 ± 0.00 / 48.27 ± 0.00 / 79.11 ± 0.01 / 139.00 ± 0.00 | 12.50 ± 0.00 / 48.27 ± 0.00 / 79.10 ± 0.00 / 139.00 ± 0.00 |
| 15 | 24.22 ± 0.00 / 53.80 ± 0.00 / 70.52 ± 0.01 / 0.17 ± 0.11 | 24.22 ± 0.00 / 53.80 ± 0.00 / 70.51 ± 0.00 / 0.04 ± 0.03 |
| 16 | 21.88 ± 0.00 / 53.08 ± 0.00 / 71.60 ± 0.01 / 0.19 ± 0.12 | 21.88 ± 0.00 / 53.08 ± 0.00 / 71.59 ± 0.00 / 0.05 ± 0.03 |

### Metrics by shortest-path hop count

Node-hop cells are `node exact / MAE / median AE`, as `mean ± seed SD`.
Again, each architecture column applies to both supervision conditions.

#### ID node-hop strata

| Target hop | Tied | Untied |
|---|---:|---:|
| unreachable | 100.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 99.58 ± 0.72 / 0.00 ± 0.00 / 0.00 ± 0.00 |
| 0 | 100.00 ± 0.00 / 0.04 ± 0.02 / 0.04 ± 0.02 | 100.00 ± 0.00 / 0.01 ± 0.01 / 0.01 ± 0.01 |
| 1 | 0.00 ± 0.00 / 157.12 ± 0.00 / 158.00 ± 0.00 | 0.00 ± 0.00 / 157.12 ± 0.00 / 158.00 ± 0.00 |
| 2 | 0.00 ± 0.00 / 153.92 ± 0.00 / 154.00 ± 0.00 | 0.00 ± 0.00 / 153.92 ± 0.00 / 154.00 ± 0.00 |
| 3 | 0.00 ± 0.00 / 151.46 ± 0.00 / 153.00 ± 0.00 | 0.00 ± 0.00 / 151.46 ± 0.00 / 153.00 ± 0.00 |
| 4 | 0.00 ± 0.00 / 146.56 ± 0.00 / 146.00 ± 0.00 | 0.00 ± 0.00 / 146.56 ± 0.00 / 146.00 ± 0.00 |
| 5+ | 0.00 ± 0.00 / 143.07 ± 0.00 / 142.00 ± 0.00 | 0.00 ± 0.00 / 143.07 ± 0.00 / 142.00 ± 0.00 |

#### OOD node-hop strata

| Target hop | Tied | Untied |
|---|---:|---:|
| unreachable | 100.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 100.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 |
| 0 | 100.00 ± 0.00 / 0.14 ± 0.09 / 0.14 ± 0.09 | 100.00 ± 0.00 / 0.04 ± 0.02 / 0.04 ± 0.02 |
| 1 | 0.00 ± 0.00 / 156.83 ± 0.00 / 157.00 ± 0.00 | 0.00 ± 0.00 / 156.83 ± 0.00 / 157.00 ± 0.00 |
| 2 | 0.00 ± 0.00 / 153.85 ± 0.00 / 154.00 ± 0.00 | 0.00 ± 0.00 / 153.85 ± 0.00 / 154.00 ± 0.00 |
| 3 | 0.00 ± 0.00 / 151.40 ± 0.00 / 152.00 ± 0.00 | 0.00 ± 0.00 / 151.40 ± 0.00 / 152.00 ± 0.00 |
| 4 | 0.00 ± 0.00 / 149.13 ± 0.00 / 150.00 ± 0.00 | 0.00 ± 0.00 / 149.13 ± 0.00 / 150.00 ± 0.00 |
| 5+ | 0.00 ± 0.00 / 147.72 ± 0.00 / 150.00 ± 0.00 | 0.00 ± 0.00 / 147.72 ± 0.00 / 150.00 ± 0.00 |

Graph exact-match by maximum finite shortest-path hop is:

| Split | Maximum hop | Tied | Untied |
|---|---:|---:|---:|
| ID | 0 | 100.00 ± 0.00 | 97.50 ± 4.33 |
| ID | 1 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ID | 2 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ID | 3 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ID | 4 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ID | 5+ | 0.00 ± 0.00 | 0.00 ± 0.00 |
| OOD | 0 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| OOD | 1 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| OOD | 2 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| OOD | 3 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| OOD | 4 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| OOD | 5+ | 0.00 ± 0.00 | 0.00 ± 0.00 |

The hop tables show the initialisation artifact directly: exactness is
confined to unreachable nodes and sources (hop 0), not relaxation over one or
more edges.

## Mechanistic measurements

No mechanistic result exists for this pilot.

| Registered measurement | Status | Reason |
|---|---|---|
| Full 16×16 \(k \times j\) probe matrices | **NOT RUN** | No condition passed G2. |
| Random-init probe baseline | **NOT RUN** | The probe pipeline is prohibited for G2-invalid conditions. |
| Shuffled-label probe baseline | **NOT RUN** | The probe pipeline is prohibited for G2-invalid conditions. |
| Attention routing agreement | **NOT RUN** | No condition passed G2. |
| Iteration ablation | **NOT RUN** | No condition passed G2. |
| Probe-diagonal test / G3 | **NOT REACHED** | No probe matrices were fit. |
| Causal state intervention | **NOT RUN** | G3 was not reached. |

There are therefore no \(k \times j\) matrices to display and no valid
comparison with their mandatory baselines. Producing them after G2 failed
would violate the pre-registration and invite interpretation of an untrained
model.

## Pre-registration deviations and operational log

There were **no deviations from a registered condition, seed, data split,
model initialisation, optimiser, learning-rate schedule, batch size, loss,
budget, checkpoint rule, metric, threshold, dtype, or device**.

Operational events that did not change a registered run:

- After the pre-registration commit and before the registered runs, a
  one-step CPU smoke test used two hand-specified graphs to verify that the
  runner executed. It was not drawn from the fixed corpus, was not retained as
  a result, and did not change the configuration.
- The Windows host lacked the test-only `hypothesis` dependency. Version
  `6.161.5` was installed before the full pre-run suite.
- The source-lint test relied on the platform default text encoding. It failed
  under Windows `cp1252`; commit `0736737` fixed the test to read source as
  UTF-8. This was independently reproducible, touched no model or pilot code,
  and the full suite passed before training.
- An initial staging-copy path typo created a malformed remote archive. It was
  removed before extraction or execution. The exact `0736737` archive was
  then staged in a new directory and verified with the full suite.

No registered run was rerun or replaced.

## Artifacts and reproducibility

- Phase 1 corrections: commit `27f4ec9`
- Trainable twin and G1: commit `c124716`
- Frozen pre-registration: commit `7e9ca30`
- Registered runner: commit `6ad0e69`
- Exact code executed on `maints`: commit `0736737`
- Remote run directory:
  `C:\Users\Admin\bellmanford-pilot-0736737\results\pilot`
- Local JSON/corpus archive:
  `results/pilot-records-0736737.tar` (gitignored, 21 MiB)
- Archive SHA-256:
  `91eb3107a237abcef852676594fb25b1259623e03ca546d2605932df06498024`

The remote directory retains model checkpoints. The local archive contains
the fixed corpus, manifest, all 12 `run.json` files with per-seed histories
and per-graph predictions, `gates.json`, and `training-summary.json`; it
excludes checkpoint tensors.

## Go/no-go

**NO-GO.** This pilot does not answer whether weight tying causes learned
relaxation. It establishes only that the registered loss and optimisation
setup failed to train either architecture to high in-distribution exact-match.
Calling the result a replication of shortcut learning would be false because
G2 excluded every model from mechanism analysis.

The result does not support a paper or workshop submission. Any attempt to
change the loss, checkpoint criterion, initialisation, schedule, or training
budget after seeing this outcome would be a new pilot requiring a new frozen
registration; it is not part of this project run.

## Notes

The main diagnostic is the conflict between the continuous objective and the
exact task metric: post-update MAE fell from about 68.6 to about 30.2–30.9,
while graph exact-match never recovered the step-0 sentinel baseline. This
explains the G2 failure but does not identify a repair or a learned mechanism.
