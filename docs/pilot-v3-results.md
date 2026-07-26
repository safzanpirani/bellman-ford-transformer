# Pilot v3: learnability diagnosis

## Bottom line

- From-scratch learnability was **not established**, even on \(n \leq 4\).
- Two maximally favourable, random-initialised diagnostic runs (wide, pre-LayerNorm, per-round supervision, fresh graphs, 12,000 steps) missed the 90% graph / 98% node competence gate.
- Across the two runs, the maxima were 74.2% graph-exact and 87.0% node-exact; changing only the learning rate from \(3\times10^{-4}\) to \(10^{-3}\) did not close the gap.
- A G1-compatible model perturbed slightly from the constructed circuit *was* locally recoverable: it reached 100% validation exactness by step 100 and 99.41% graph / 99.80% node exactness on test at the selected checkpoint.
- That local result does not show learnability from random initialisation, and later optimisation drifted away from the competent solution.
- Because no from-scratch D1 recipe reached competence, D2, D3, the pre-registration, and the 2×2×3 grid were not run.
- V3 therefore does not establish why v1 failed. Every proposed cause remains untested.
- **Go/no-go: no-go.** There is no validated training recipe for the comparator experiment. The present result is not publishable.

## Scope and decision rule

V3 kept the v1 task interface, loss, and metrics. In particular, finite predictions
were already counted exact within the half-unit integer bin; no v2 metric or
reachability-loss changes were made. Competence used the inherited gate of at least
90% graph exactness and 98% node exactness on the fixed validation set.

All runs in this report are diagnostics, not experimental conditions. The two
random-initialised runs include LayerNorm and therefore violate G1. The perturbation
runs use a G1-compatible architecture, but start from deliberately perturbed
constructed weights and therefore do not constitute a from-scratch experimental
condition.

Runs used seed 17, data seed 20260730, a fixed 512-graph validation set, and a fixed
1,024-graph test set. Training ran with Python 3.11.9 and PyTorch 2.11.0+cu126 on an
RTX 3090. The diagnostic source was commit `25c10e2`.

## D1 — Existence from random initialisation

The maximally favourable configuration used:

- \(n_{\max}=4\);
- fresh graphs every batch;
- width 128 in every attention and MLP subspace, beyond the constructed minimum;
- bias-free pre-LayerNorm;
- tied recurrence;
- per-round supervision;
- batch size 128;
- 200 warmup steps and cosine decay over 12,000 steps; and
- random initialisation with gain 0.1.

Only the learning rate differed between the two runs.

| LR | G1 status | Best step | Validation graph | Validation node | Validation reachable non-source | Validation MAE | Test graph | Test node | Competent |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| \(3\times10^{-4}\) | diagnostic, non-G1 | 12,000 | 73.05% | 87.04% | 68.48% | 0.4543 | 69.34% | 84.37% | No |
| \(10^{-3}\) | diagnostic, non-G1 | 11,250 | 74.22% | 86.84% | 64.86% | 0.4528 | 70.70% | 84.83% | No |

Both runs started at 32.03% graph exactness, 63.32% node exactness, 0% reachable
non-source exactness, and MAE 13.40. Thus checkpoint selection was not receiving
free credit from the identity solution: the selector prioritised reachable
non-source exactness and excluded step 0.

The \(3\times10^{-4}\) curve improved from 50.59% / 73.16% graph/node exactness at
step 6,000 to 73.05% / 87.04% at step 12,000. A longer run might perform differently;
it was not run and is untested. The \(10^{-3}\) run had a lower final smooth loss but
did not improve exactness, so lower regression loss alone did not satisfy the task
gate in this comparison.

**D1 from-scratch verdict: not demonstrated within the time box.** This is bounded
evidence for the two recipes above, not a claim that no optimiser or schedule can
learn the task.

### Perturbation calibration

The first perturbation attempt used \(\sigma=10^{-3}\). Although its parameter
distance was small, its step-0 function was only 32.23% graph-exact and 60.40%
node-exact. It was therefore too functionally disrupted to support a strong
near-circuit conclusion. It peaked at 85.74% / 94.42% validation exactness at step
100 and remained below the gate. This attempt is retained but is not the primary
local-recovery diagnostic.

Runner-identical, zero-update calibration then located a functional boundary:

| Gaussian noise \(\sigma\) | Parameter distance | Step-0 graph | Step-0 node | Step-0 reachable non-source | Calibration interpretation |
|---:|---:|---:|---:|---:|---|
| \(3\times10^{-4}\) | 0.0266 | 100.00% | 100.00% | 100.00% | still competent |
| \(4\times10^{-4}\) | 0.0355 | 93.95% | 97.94% | 99.82% | near-circuit, just below gate |
| \(6\times10^{-4}\) | 0.0532 | 41.99% | 68.77% | 94.93% | functionally disrupted |
| \(10^{-3}\) | 0.0886 | 32.23% | 60.40% | 68.48% | functionally disrupted |

The abrupt functional change across these scales is observed, not explained. No
claim is made about its cause.

### Valid local-recovery diagnostic

The replacement perturbation diagnostic used the constructed-minimal, tied,
LayerNorm-free model; \(\sigma=4\times10^{-4}\); fresh graphs; per-round supervision;
batch size 128; \(3\times10^{-5}\) learning rate; 200 warmup steps; and a 4,000-step
budget.

| Step | Graph exact | Node exact | Reachable non-source | MAE | Distance from constructed |
|---:|---:|---:|---:|---:|---:|
| 0 | 93.95% | 97.94% | 99.82% | 0.7322 | 0.0355 |
| 100 | 100.00% | 100.00% | 100.00% | 0.0112 | 0.0363 |
| 400 | 100.00% | 100.00% | 100.00% | 0.00134 | 0.0391 |
| 4,000 | 88.87% | 96.08% | 93.48% | 0.3126 | 0.6384 |

The selected step-400 checkpoint scored 99.41% graph exactness, 99.80% node
exactness, 99.48% reachable non-source exactness, and MAE 0.0197 on test. The task
is therefore locally recoverable under this exact condition with early stopping.
The parameter distance increased even while functional accuracy recovered, and the
run later left the competent region. The evidence does not show convergence back to
the constructed parameters or stable training for 4,000 steps.

This fallback does not answer whether ordinary random initialisation can reach the
solution. Starting near a known solution is not a usable recipe for the comparator
experiment.

### \(n \leq 8\) climb

Not run. The climb was conditional on a competent from-scratch \(n \leq 4\) recipe.
Using constructed-weight perturbations for the climb would replace the
learnability question with a local-recovery question.

## D2 — Backward ablations

D2 required a working from-scratch D1 recipe. None was found. Ablating from the
constructed-perturbation fallback would not identify why v1 training from random
initialisation failed, so the following causal hypotheses remain untested:

| Proposed ingredient | “With” condition | “Without” condition | Result |
|---|---|---|---|
| Fresh data | fresh graph each batch | fixed 4,096-graph corpus | Not run; no competent from-scratch baseline |
| Wider capacity | width 128 subspaces | constructed-minimal shapes | Not run; no competent from-scratch baseline |
| LayerNorm | pre-LayerNorm | LayerNorm-free | Not run; no competent from-scratch baseline |
| Per-round supervision | every round supervised | final distance only | Not run; no competent from-scratch baseline |
| Generous budget | 12,000-step diagnostic | v1's 4,000 steps | Not run; 12,000-step baseline was not competent |

Accordingly, v3 does **not** establish that v1 failed because of corpus reuse,
minimal capacity, missing LayerNorm, final-only supervision, or insufficient
steps. Data reuse remains a confirmed property of v1, not a demonstrated cause.

## D3 — G1-satisfying learnability

Not reached. The additive diagnostic model verified that an unnormalised widened
model can embed the constructed weights in its leading coordinates with zero
extra coordinates and reproduce the constructed outputs bit-exactly for
\(n_{\max}\in\{4,8,16\}\). This establishes representability of that widened,
LayerNorm-free family.

It does not establish learnability from random initialisation. The only competent
G1-compatible diagnostic began at a small perturbation of the constructed
solution. No random-initialised G1-satisfying recipe was tested because D1 did not
produce a from-scratch recipe to ablate toward G1.

## Stopped work and untested alternatives

The following were deliberately not run:

- additional seeds or confidence intervals;
- lower learning rates or more than 12,000 steps from random initialisation;
- alternative optimisers, curricula, batch sizes, widths, or initialisation gains;
- the \(n \leq 8\) climb;
- all D2 one-factor ablations;
- D3 recipe search;
- a v3 pre-registration;
- the tied/untied × final/per-round grid; and
- probes, attention comparisons, ablations, or causal interventions.

These are untested, not negative results. Continuing to tune them would exceed the
D1 time box without a principled stopping point.

## Go/no-go and venue estimate

**No-go for the comparator experiment.** A constructed solution is representable,
and a small perturbation can be repaired briefly, but no from-scratch competent
model exists under the tested recipes. Without that, recurrence versus shortcut
learning cannot be measured.

The v3 result is not publishable by itself. It is an optimisation diagnostic with
one seed and a bounded negative from-scratch search; it does not answer the
research question owned in broader form by *Spectral Journey*.
