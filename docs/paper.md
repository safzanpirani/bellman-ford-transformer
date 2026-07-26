# An Exact Bellman–Ford Transformer as Optimization Ground Truth

## Representability, learnability, and parameter non-recovery in a small-graph diagnostic

*Anonymous draft*

### Abstract

Hand-constructed transformers provide unusually strong ground truth: their
forward pass, intermediate state, and intended circuit are known exactly. That
does not imply that gradient descent can recover the same computation, even
when the constructed solution belongs to the trainable model's hypothesis
class. We study this prerequisite using a weight-tied transformer whose forward
pass executes Bellman–Ford relaxation. The construction is not our novelty; it
builds on program-compiled and looped-transformer work. Its useful properties
are stricter: integer-weight outputs match a conventional shortest-path
implementation bit-for-bit, every relaxation state is exposed, and the
constructed parameters can be embedded exactly into a wider trainable model.

On directed graphs with at most four nodes, a widened, LayerNorm-free,
weight-tied model trained from random initialization reached 96.97% held-out
graph exactness, 98.85% node exactness, and 96.97% exactness on reachable
non-source nodes. The corresponding constructed embedding attained 100%
exactness, zero training objective, and zero gradient. The trained model
nevertheless remained approximately as far from the constructed parameters as
it was at initialization. A matched 4,000-step run was still improving at
50.20% validation graph exactness, whereas an 80,000-step run satisfied a
predefined saturation criterion at 97.46% in a LayerNorm variant. Finally,
perturbation experiments around the exact circuit support an optimizer
step-size explanation for previously observed drift: with a smaller peak
learning rate and decay to zero, exact behavior was retained even though the
parameters did not return to the construction.

These are single-seed, small-graph diagnostics. We did not run the tied-versus-
untied comparison required to determine whether recurrence causes trained
models to learn iterative relaxation. Our result is therefore a validity and
optimization result, not an answer to that mechanistic question.

## 1. Introduction

Neural algorithmic reasoning asks whether a neural network has merely fit the
input-output relation of an algorithm or has learned a procedure that will
continue to work outside the training distribution. Shortest paths make the
distinction concrete. A model can implement repeated local relaxation, exploit
a correlated graph statistic, memorize a bounded-size mapping, or compute a
different but behaviorally adequate algorithm. In-distribution output accuracy
does not identify which explanation is correct.

Prior work already establishes both sides of this problem. Trained
transformers can use shortcuts on formal tasks [1] and a trained shortest-path
transformer has been reported to use spectral navigation rather than textbook
dynamic programming [2]. Conversely, trained recurrent models can learn
iterative procedures with length generalization [3], and sparsity-regularized
graph neural networks can recover Bellman–Ford-like relaxation [4]. The broad
claim that trained models always fail to learn algorithms is therefore both
occupied and false.

A controlled comparison needs a stronger prerequisite than ordinary task
accuracy: the intended solution must actually be present in the trainable
hypothesis class. Otherwise a failure to learn the intended circuit is
uninterpretable. We call this prerequisite **G1 parity**. A trainable model has
G1 parity when the hand-constructed parameters can be loaded into it and
reproduce the exact outputs and objective. This criterion rules out seemingly
minor architectural changes—most notably LayerNorm here—that alter the literal
arithmetic performed by the construction.

This paper reports the diagnostic work required to establish G1 parity and
learnability in the same model family. It makes three bounded contributions:

1. We describe an exact Bellman–Ford transformer as executable optimization
   ground truth. Its construction follows existing expressivity work; the
   contribution is the exact verifier, complete round state, and direct
   embedding into a trainable twin.
2. We show, for one seed and graphs with at most four nodes, that a
   G1-compatible model can be trained from random initialization to near-exact
   behavior. This separates representability from the optimization failure
   observed in shorter runs.
3. We characterize local drift around the exact circuit. The construction is a
   zero-loss, zero-gradient global minimum, and learning-rate-controlled runs
   recover exact behavior without recovering the original parameters.

The claim boundary is important. We do not show that weight tying makes a
trained transformer use Bellman–Ford relaxation. We did not run an untied
comparator, size extrapolation, state probes, attention comparison, iteration
ablation, or causal interventions. Those are requirements for a later
mechanistic experiment, not results of this paper.

## 2. Related work

### 2.1 Compiled and looped transformers

RASP introduced a programming abstraction for reasoning about the computations
available to transformers [5]. Tracr subsequently compiled restricted programs
into transformer weights, making known-circuit transformers available as
laboratory objects [6]. Looped-transformer constructions further showed that a
reused block can implement iterative programs [7]. Most directly, *Simulation
of Graph Algorithms with Looped Transformers* constructs looped transformers
for graph algorithms and discusses finite-precision and softmax limitations
[8]. These papers preclude a construction-novelty claim for the present model.

Our distinction is engineering and measurement discipline. The model exposes
all residual registers after every relaxation round, has an executable
Bellman–Ford reference trajectory, and is tested for exact agreement rather
than only approximate task accuracy. The same parameter tensors can also be
embedded into a widened trainable module, with all added coordinates set to
zero.

The implementation uses two different exactness mechanisms that should not be
conflated with general theorems. Pairwise minima use the exact ReLU identity

\[
\min(a,b)=a-\operatorname{ReLU}(a-b).
\]

Routing uses a fixed float32 score gap large enough that non-target
exponentials underflow to zero. The latter is an implementation property
dependent on dtype and exponential-kernel behavior, not exact hard selection by
finite-logit real-arithmetic softmax. Approximate sharpness of ordinary softmax
requires score spread growing with set size and desired error [9].

### 2.2 Training algorithms and shortcuts

Liu et al. show that shallow transformers can fit automata tasks through
shortcuts that fail outside the training regime [1]. On graphs, *Spectral
Journey* analyzes a trained shortest-path transformer and reports a spectral
mechanism rather than Bellman–Ford or Dijkstra-style dynamic programming [2].
These results own the broad shortcut-learning headline.

The negative conclusion is not universal. Looped transformers have been
trained to learn iterative solutions with length generalization [3].
Sparsity-regularized graph neural networks have also been shown to recover a
fixed number of Bellman–Ford iterations and extrapolate to larger graphs [4].
MINAR recovers compact Bellman–Ford-like circuitry in a trained graph neural
network and also demonstrates that correct outputs can conceal an unintended
internal shortcut [10].

The remaining narrow question is whether weight-tied recurrence changes what a
trained transformer learns when an exact relaxation circuit exists in the same
architecture. The present study validates one prerequisite for asking that
question, but does not answer it.

### 2.3 Known-circuit interpretability

InterpBench argues that mechanistic-interpretability methods need models with
known ground-truth circuits beyond raw Tracr programs [11]. A hand-constructed
graph algorithm offers exact labels at the level of registers and iterations,
but a single model is not itself a benchmark. Establishing benchmark value
would require comparing interpretability methods or using the ground truth to
expose a specific failure.

Intermediate-state probes would also be insufficient on their own. Exact
algorithmic state can be decodable without being causally used [12].
Consequently, any future claim that a trained model executes relaxation must
combine state alignment with causal interventions.

## 3. Exact Bellman–Ford transformer

### 3.1 Algorithm and tokenization

For source \(s\), Bellman–Ford initializes

\[
d_s^{(0)}=0,\qquad d_v^{(0)}=\mathrm{INF}_B \quad (v\ne s),
\]

where \(\mathrm{INF}_B=N_{\max}W_{\max}+1\) exceeds every representable simple
path length. Each round computes

\[
d_v^{(t+1)}
=
\min\left(
d_v^{(t)},
\min_{(u,v,w)\in E}\left(d_u^{(t)}+w\right)
\right).
\]

The sequence contains one sink token, a fixed allocation of node tokens, and a
fixed allocation of directed-edge tokens. Residual coordinates encode node
identities, the distance register, edge endpoints and weights, candidate
slots, and temporary routing state.

A single block performs one relaxation round:

1. edge tokens read the previous distance of their source node;
2. fixed gather heads route incoming candidates to destination-node slots;
3. a ReLU network folds those candidates with the previous node distance; and
4. the resulting delta is written back while temporary registers are cleared.

The same block is reused \(N_{\max}-1\) times. An optional additional round
detects a reachable negative cycle by testing whether any distance decreases.
The construction contains no LayerNorm because normalizing the residual stream
would rescale the literal distance values.

### 3.2 Verification and tracing

The artifact is checked against an independent Bellman–Ford reference after
every round. Integer-weight outputs are bit-exact against
`networkx.single_source_bellman_ford_path_length`; a seeded 1,000-graph suite
spans graph sizes through the configured maximum. The tracer records the full
residual state, named sublayer deltas, and attention patterns.

This verification does not make the construction novel. It makes the
construction useful as an exact reference point for optimization and
mechanistic comparison.

## 4. Trainable twin and G1 parity

The trainable twin preserves the constructed tokenization, residual-register
layout, sublayer order, number of heads, and tied recurrent application. Each
attention projection and MLP hidden space may be widened. To load the
construction, the original matrix is copied into the leading block of the
wider tensor and every additional entry is set to zero. Thus widening is
compatible with the constructed function.

LayerNorm is optional in diagnostic variants but incompatible with G1 parity.
For a G1 audit, the model must be LayerNorm-free, the constructed weights must
load without approximation, and the loaded model must attain exact outputs and
zero objective.

The widened LayerNorm-free model passed this audit:

- graph, node, reachability, and reachable-non-source exactness were all 100%
  on validation and test;
- supervised loss and mean absolute error were zero;
- the full-objective gradient had both zero L2 norm and zero maximum absolute
  component; and
- Euclidean parameter distance from the constructed reference was zero.

Because the objective is non-negative, zero objective also certifies that the
constructed parameters are a global minimum of the measured training loss.

## 5. Diagnostic protocol

All results in this paper are diagnostic and use a single model seed, 17, and
data seed, 20260730. The fixed validation and test sets contain 512 and 1,024
graphs, respectively. Training graphs are freshly sampled on every batch.
Graphs are directed, have integer non-negative edge weights, and contain
between two and four nodes. The model therefore executes three tied relaxation
rounds.

The principal random-initialization recipe uses:

- width 128 for every attention and MLP subspace;
- no LayerNorm;
- per-round supervision;
- batch size 128;
- Xavier initialization with gain 0.1;
- AdamW with \(\beta=(0.9,0.999)\), no weight decay, and gradient norm clipped
  to 1;
- peak learning rate \(3\times10^{-4}\);
- 200 warmup steps followed by cosine decay to zero; and
- 80,000 optimization steps.

The loss is the mean Smooth-L1 loss over valid nodes and supervised rounds:

\[
\mathcal{L}
=
\operatorname{mean}
\operatorname{SmoothL1}_{\beta=0.01}
\left(
\frac{\hat d}{\mathrm{INF}_B},
\frac{d}{\mathrm{INF}_B}
\right).
\]

For finite targets, a node is exact when
\(\lvert\hat d-d\rvert<0.5\). For unreachable targets, it is exact when
\(\hat d\ge \mathrm{INF}_B-0.5\). Graph exactness requires every real node in
the graph to be exact. We additionally report node exactness, reachability
accuracy, and reachable non-source exactness (RNSE). RNSE removes source nodes
and unreachable nodes, the cases already correct under an identity-like
initialization.

### 5.1 Saturation

We do not define competence by an inherited accuracy threshold. A run is
classified as saturated only if its terminal learning rate is zero and each of
graph exactness, node exactness, and RNSE satisfies all of the following over
the final two 10,000-step windows:

- the best value in the final window improves by less than one percentage point
  over the preceding window;
- the absolute least-squares slope in the final window is less than one
  percentage point per 10,000 steps; and
- the range in the final window is less than two percentage points.

This criterion establishes a measured plateau; it does not claim global
optimality.

### 5.2 Local perturbation diagnostic

To distinguish loss/metric conflict from ordinary optimizer drift, we also
start from the exact constructed parameters plus independent Gaussian noise
with standard deviation \(4\times10^{-4}\). These runs use the
constructed-minimal, tied, LayerNorm-free model and are therefore
G1-compatible. We measure the objective and gradient at the exact
construction, then log validation loss, exactness, and Euclidean parameter
distance throughout training. Peak learning rates of \(3\times10^{-5}\) and
\(10^{-5}\) both decay to zero.

## 6. Results

### 6.1 A G1-compatible model is trainable on \(n\le4\)

Table 1 reports the saturated LayerNorm and LayerNorm-free results. The
LayerNorm-free model reaches 96.97% held-out graph exactness and 96.97% RNSE.
This establishes existence of a trainable, G1-compatible recipe in the tested
small-graph regime.

**Table 1. Saturated random-initialization runs. All values are percentages.**

| Architecture | G1-compatible | Best validation graph / node / RNSE | Test graph / node / RNSE | Reachability | Verdict |
|---|:---:|---:|---:|---:|---|
| width 128, pre-LayerNorm | No | 97.46 / 99.14 / 97.64 | 96.97 / 98.92 / 97.14 | 100 | Saturated |
| width 128, no LayerNorm | Yes | 96.68 / 98.74 / 96.56 | 96.97 / 98.85 / 96.97 | 100 | Saturated |

LayerNorm is therefore not necessary under this widened, long-horizon recipe.
The 0.78 percentage-point validation graph difference is a single-seed
observation, not an estimate of a systematic normalization effect.

The trained no-LayerNorm endpoint is functionally successful but does not
approach the constructed parameter vector. Its Euclidean distance from the
construction is approximately 76,648, compared with approximately 76,649 at
initialization. The evidence establishes functional learnability, not recovery
of the known parameterization.

### 6.2 The short run is budget-limited

Table 2 compares two otherwise matched LayerNorm runs. The 4,000-step run ends
at its best checkpoint and therefore cannot be interpreted as a performance
ceiling. Extending the schedule to 80,000 steps changes validation graph
exactness by 47.27 percentage points.

**Table 2. Matched training-horizon comparison.**

| Steps | Validation graph | Validation node | Validation RNSE | Curve verdict |
|---:|---:|---:|---:|---|
| 4,000 | 50.20 | 65.25 | 27.72 | Budget-limited; best at final step |
| 80,000 | 97.46 | 99.14 | 97.64 | Saturated |

This intervention could have falsified a budget effect under the selected
recipe and instead found a large one. It does not establish that training
duration was the sole cause of earlier failed runs. In particular, the
4,000-step diagnostic already exceeds the earlier approximately 18% post-
initialization graph exactness, so at least one other difference mattered.
Fresh versus fixed data, width 128 versus the constructed-minimal width,
per-round versus final-only supervision, and batch 128 versus batch 32 remain
untested as isolated causes.

The no-LayerNorm learning curve also contradicts an absolute claim that
normalization is required:

**Table 3. Selected no-LayerNorm validation trajectory.**

| Step | Graph exactness | Node exactness | RNSE |
|---:|---:|---:|---:|
| 0 | 32.03 | 63.32 | 0.00 |
| 10,000 | 75.00 | 89.10 | 72.28 |
| 20,000 | 79.88 | 91.36 | 79.17 |
| 40,000 | 89.26 | 95.42 | 88.22 |
| 60,000 | 95.31 | 98.27 | 95.29 |
| 80,000 | 96.68 | 98.74 | 96.56 |

### 6.3 Drift around the exact circuit is consistent with optimizer step size

At the exact construction, the objective is zero and both the L2 and maximum
absolute gradient are zero. The measured loss therefore does not actively push
the model away from the exact circuit.

After perturbation, both zero-floor schedules recover exact or nearly exact
behavior. The smaller peak learning rate ends at 100% validation and test graph
exactness. Parameter distance nevertheless grows slightly instead of returning
to zero.

**Table 4. Local perturbation results.**

| Peak learning rate | Final validation graph | Selected test graph | Final loss | Parameter distance, initial → final |
|---:|---:|---:|---:|---:|
| \(3\times10^{-5}\) | 100.00 | 99.80 | \(7.06\times10^{-7}\) | 0.0355 → 0.1257 |
| \(10^{-5}\) | 100.00 | 100.00 | \(7.58\times10^{-8}\) | 0.0355 → 0.0433 |

Across both logged trajectories, there are zero evaluation intervals in which
validation graph exactness falls while validation loss also falls. Exactness
declines coincide with higher validation loss, not an objective improvement.
Both endpoints have positive loss, whereas the construction has exactly zero
loss.

These measurements support an ordinary optimizer-step explanation for the
previous drift: an update can leave a narrow exact region, and lowering the
peak learning rate limits that movement. They do not support the competing
claim that the regression objective rewards sacrificing half-unit-bin
exactness. The parameters need not return to the hand construction to recover
the same measured function.

## 7. Discussion

### 7.1 Representability is a testable experimental condition

It is easy to describe a trained model as having “failed to learn the known
algorithm” when the intended circuit cannot be represented by the exact
training architecture. G1 parity turns that ambiguity into an executable test.
The exact construction must load, its outputs must remain exact, and its
objective must be zero. A normalization layer that seems harmless from an
optimization perspective fails this test because it changes the arithmetic
state.

Widening, by contrast, can preserve G1 parity when the original matrices are
embedded and every new coordinate is zeroed. This provides a useful compromise:
optimization is not restricted to the constructed-minimal width, but the exact
circuit remains inside the hypothesis class.

### 7.2 Functional success does not imply circuit recovery

The randomly initialized model learns a near-exact function while remaining
almost exactly as far from the constructed parameter vector as at
initialization. Likewise, local perturbation runs recover exact outputs without
returning to the reference parameters. Raw parameter distance is therefore not
a valid circuit-recovery criterion in this setting.

This observation alone does not establish that the learned mechanism differs
from Bellman–Ford. Neural parameterizations have symmetries and redundant
representations, and widening adds unused or alternative subspaces. Determining
mechanistic correspondence requires comparisons at the level of round state,
routing, intervention response, and executed function.

### 7.3 What the diagnostics say about failed short runs

The supported causal statement is narrow: under the selected recipe, 4,000
steps are insufficient and 80,000 steps reach saturation. The evidence does not
show that data reuse, minimal capacity, missing LayerNorm, supervision, or batch
size caused any prior failure. Those explanations remain hypotheses because
the corresponding one-factor interventions were not completed.

This distinction matters methodologically. A checkpoint at the end of an
improving curve supports “budget-limited,” not “unlearnable.” Conversely,
reaching a high score under a more favorable recipe does not identify which
ingredient fixed an earlier run.

## 8. Limitations

The limitations determine the paper's scope:

- All completed training results use one model seed. There are no seed
  variances, confidence intervals, or population-level comparisons.
- Training and evaluation are limited to graphs with at most four nodes.
  An \(n\le8\) saturation run was not completed, and no size extrapolation was
  evaluated.
- The principal recipe uses per-round supervision, fresh graphs, width 128,
  batch 128, and 80,000 steps. Most ingredients were not independently ablated.
- The tied-versus-untied and final-only-versus-per-round grid was not run.
  Nothing here determines whether recurrence changes the learned mechanism.
- No linear-probe matrix, shuffled-label baseline, attention agreement,
  iteration ablation, or causal state intervention was run.
- Parameter distance is sensitive to parameterization, symmetries, and
  widening. We use it only to reject literal recovery of the reference vector.
- Exact attention routing depends on the tested float32 underflow behavior and
  is not a real-arithmetic softmax theorem.
- The saturation criterion describes the measured final windows; it does not
  prove that no better optimizer or longer schedule could improve the result.
- These diagnostics were developed after earlier failed pilots. They are not a
  preregistered confirmatory experiment.

Accordingly, this is workshop-scale evidence for experimental validity and
optimization behavior. It is not a shortest-path generalization result and not
a mechanistic claim about learned relaxation.

## 9. Deferred mechanistic experiment

The exact artifact makes a stronger study possible, but that study remains
future work. A valid test would first reproduce G1-compatible learnability at
the intended \(N_{\max}=16\) scale, train on \(n\le8\), and evaluate on held-out
graphs through \(n=16\). It would then compare tied and untied models under
final-only and per-round supervision across multiple seeds.

The decisive measurements would be:

1. behavioral exactness stratified by graph size and shortest-path hop count;
2. a full \(k\times j\) matrix probing loop state at iteration \(k\) for the
   exact Bellman–Ford state after round \(j\), with random-initialization and
   shuffled-label baselines;
3. attention-routing agreement with the constructed circuit;
4. iteration ablation, testing whether additional or removed loops change the
   result monotonically; and
5. causal interventions on decoded round state, performed only if a probe
   diagonal appears.

Without those conditions, the question “does weight-tied recurrence make a
trained model learn relaxation?” remains unanswered.

## 10. Conclusion

An exact algorithmic transformer is useful for more than an expressivity
demonstration. It supplies an executable global minimum, exact intermediate
state, and a direct test of whether a training architecture contains the
intended solution.

In the small-graph regime studied here, the answer to the prerequisite
learnability question is yes: a widened, LayerNorm-free, weight-tied model that
contains the exact Bellman–Ford circuit reaches 96.97% held-out graph exactness
from random initialization. The trained parameters do not recover the
construction, and local perturbation experiments show that exact functional
recovery depends on optimizer step size while remaining compatible with
parameter drift.

The result resolves an experimental-validity objection, not the motivating
mechanistic question. Whether recurrence changes a trained transformer's
algorithm remains open because the required comparison was not run.

## Reproducibility statement

The complete evaluation histories used in this draft are stored in
[`pilot-v4-curves.json`](pilot-v4-curves.json), and the audit trail is in
[`pilot-v4-results.md`](pilot-v4-results.md). The diagnostic implementation is
in `train/v3_model.py`, `train/v3_diagnostic.py`, and
`train/v4_diagnostic.py`. The v4 diagnostic source was committed as
`6489e31`; the frozen report and curve artifact were committed as `ccb199b`.
Runs used Python 3.11.9, PyTorch 2.11.0+cu126, and an NVIDIA RTX 3090.

## References

[1] Liu et al. “Transformers Learn Shortcuts to Automata.”
[arXiv:2210.10749](https://arxiv.org/abs/2210.10749), 2022.

[2] Cohen et al. “Spectral Journey: How Transformers Predict the Shortest
Path.”
[arXiv:2502.08794](https://arxiv.org/abs/2502.08794);
[OpenReview:mAEsGkITgG](https://openreview.net/forum?id=mAEsGkITgG), 2025.

[3] Fan et al. “Looped Transformers for Length Generalization.”
[arXiv:2409.15647](https://arxiv.org/abs/2409.15647), 2024.

[4] Nerem et al. “Graph Neural Networks Extrapolate Out-of-Distribution
for Shortest Paths.”
[arXiv:2503.19173](https://arxiv.org/abs/2503.19173);
[PMLR 336](https://proceedings.mlr.press/v336/nerem26a.html), 2026.

[5] Weiss, Goldberg, and Yahav. “Thinking Like Transformers.”
[arXiv:2106.06981](https://arxiv.org/abs/2106.06981), 2021.

[6] Lindner et al. “Tracr: Compiled Transformers as a Laboratory for
Interpretability.”
[arXiv:2301.05062](https://arxiv.org/abs/2301.05062), 2023.

[7] Giannou et al. “Looped Transformers as Programmable Computers.”
[arXiv:2301.13196](https://arxiv.org/abs/2301.13196), 2023.

[8] de Luca and Fountoulakis. “Simulation of Graph Algorithms with Looped
Transformers.”
[arXiv:2402.01107](https://arxiv.org/abs/2402.01107), 2024.

[9] Veličković et al. “Softmax Is Not Enough (for Sharp Size
Generalisation).”
[arXiv:2410.01104](https://arxiv.org/abs/2410.01104);
[PMLR 267](https://proceedings.mlr.press/v267/velickovic25a.html), 2025.

[10] He et al. “MINAR: Mechanistic Interpretability for Neural Algorithmic
Reasoning.”
[OpenReview:3PbQUU9rgu](https://openreview.net/forum?id=3PbQUU9rgu);
[code](https://github.com/pnnl/MINAR).

[11] Gupta et al. “InterpBench: Semi-Synthetic Transformers for Evaluating
Mechanistic Interpretability Techniques.”
[arXiv:2407.14494](https://arxiv.org/abs/2407.14494), 2024.

[12] Darade and Thorat. “Represented Is Not Computed: A Causal Test of
Candidate Algorithmic Intermediates in a Transformer.”
[arXiv:2605.22488](https://arxiv.org/abs/2605.22488), 2026.
