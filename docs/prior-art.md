# Prior-art triage

## Bottom line

- **Track A — NO-GO as currently claimed.** Shortcut learning, graph OOD failure, and mechanistic analysis of trained shortest-path transformers are already published.
- A much narrower contribution may remain: a controlled comparison of a hand-constructed and a trained **weight-tied Bellman–Ford transformer in the same architecture**, scored against exact round-by-round state and tested causally.
- **Track B+C — NO-GO.** The useful softmax result—an \(\Omega(\log(n/\varepsilon))\) score spread for an \(\varepsilon\)-sharp selection—is already in the literature.
- The proposed *exact* softmax claim is not a stronger version of that theorem: over real arithmetic finite-temperature softmax is never exactly hard selection, while this repository's float32-underflow construction can make a fixed score gap exact independently of \(n\).
- No paper-level contribution remains from saying that pairwise ReLU min avoids softmax dispersion; exact pairwise ReLU min and logarithmic-depth reduction trees are standard constructions.
- Recommendation: stop Track B+C in its present form. Continue Track A only after replacing its broad motivation with the narrow comparison above.

## Verdicts

| Claim | Verdict | Consequence |
|---|---|---|
| C1. Trained transformers do not learn genuine iterative relaxation on shortest-path/graph tasks; they learn OOD-failing shortcuts | **PARTIAL** | Direct graph-shortcut and shortest-path-mechanism evidence already exists, but I found no exact trained, tied Bellman–Ford transformer study with round-state and size-OOD tests. The universal negative is false and cannot be the contribution. |
| C2. Per-iteration probing against exact intermediate algorithmic state is an established test of whether a model learned an algorithm | **SETTLED** | This methodology is already explicit in latent-step mechanistic work. Probes alone are also already known to be insufficient without causal tests. |
| C3. Hand-constructed graph-algorithm models are useful mechanistic-interpretability ground truth beyond Tracr | **PARTIAL** | Known-circuit semi-synthetic transformer benchmarks beyond Tracr exist, and graph-algorithm circuit recovery exists, but I found no benchmark joining both in a hand-constructed graph transformer with full round/register labels. |
| C4. Exact min over \(n\) candidates via a softmax head requires logit scale \(\Omega(\log n)\); ReLU folds do not | **SETTLED** | The approximate softmax statement is published. The exact statement is malformed or false depending on arithmetic semantics, and the ReLU construction is standard. This track dies as framed. |
| C5. There is a known depth-vs-precision trade-off for exact transformer min/max reductions | **PARTIAL** | Relevant upper bounds and transformer precision/complexity characterizations exist, but I found no matching lower-bound theorem. Without restrictions on width, heads, attention, and arithmetic, the proposed separation is unlikely even to be true. |

Unless marked **[ABSTRACT ONLY]**, the evidence below was checked in full paper text. No finding below is inferred from a title.

## C1 — trained graph transformers learn shortcuts rather than iterative relaxation

### Verdict: PARTIAL

The broad empirical story is occupied.

1. Liu et al., [*Transformers Learn Shortcuts to Automata* (arXiv:2210.10749)](https://arxiv.org/abs/2210.10749), show that shallow non-recurrent transformers trained on automata tasks can fit in-distribution data through shortcut solutions yet fail on unseen lengths and other OOD regimes. This is strong prior art for the general “training finds a shortcut, not the intended recurrence” claim, although it is not a graph or shortest-path study.

2. Abbe et al., [*How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad* (NeurIPS 2024 paper URL)](https://proceedings.neurips.cc/paper_files/paper/2024/file/3107e4bdb658c79053d7ef59cbc804dd-Paper-Conference.pdf), study graph reachability/cycle prediction directly. Their regular transformer obtains good in-distribution accuracy using a degree-based shortcut and falls to chance under the relevant OOD shift. Their educated scratchpad can also fail when test instances require more reasoning steps than training, while an inductive scratchpad improves length extrapolation. This already establishes graph-task shortcut learning and OOD failure.

3. Cohen et al., [*Spectral Journey: How Transformers Predict the Shortest Path* (OpenReview paper URL)](https://openreview.net/pdf?id=mAEsGkITgG), train decoder-only transformers from scratch on shortest paths in small undirected graphs and mechanistically analyze their solution. They report no evidence for Bellman–Ford/Dijkstra-style dynamic programming; instead, the model builds a spectral representation of the line graph and follows an approximate spectral-navigation procedure. That directly pre-empts a generic “trained shortest-path transformers use an unexpected shortcut rather than the textbook algorithm” contribution. Its evaluation is on unseen graph structures within a small, fixed size range, not convincing graph-size extrapolation.

4. Shin et al., [*Discovering Mechanisms in Tokenized Graph Transformers* (OpenReview paper URL)](https://openreview.net/pdf?id=0WEuCiokZq) **[ABSTRACT ONLY]**, use activation patching and linear probes on tokenized graph transformers, including shortest-path distance. The abstract reports a serial mechanism progressing from local topology through an adjacency-copying head to distance-like refinement. This is close enough that any claim of being the first mechanistic study of a trained shortest-path transformer is unsafe.

5. The negative result is not universal. Fan et al., [*Looped Transformers for Length Generalization* (arXiv:2409.15647)](https://arxiv.org/abs/2409.15647), train looped transformers that learn iterative solutions and generalize to substantially longer algorithmic inputs. More directly on graphs, Nerem et al., [*Graph neural networks extrapolate out-of-distribution for shortest paths* (arXiv:2503.19173; PMLR 336)](https://proceedings.mlr.press/v336/nerem26a.html), prove that a sparsity-regularized trained GNN exactly implements a fixed number of Bellman–Ford iterations and extrapolates to larger graphs. The latter is not a transformer, but it defeats any architecture-agnostic assertion that training cannot recover relaxation.

6. He et al., [*MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning* (OpenReview:3PbQUU9rgu)](https://openreview.net/forum?id=3PbQUU9rgu), with [code at `pnnl/MINAR`](https://github.com/pnnl/MINAR), recover a minimal circuit from a trained min-aggregating GNN and show that it implements two Bellman–Ford steps. They also exhibit a multitask model that reaches the right outputs through an unintended internal shortcut that ordinary OOD evaluation misses. This already combines shortest paths, recovered algorithmic circuitry, and the warning that behavior alone does not identify the learned algorithm.

**Exact remaining gap:** No retrieved work compares a hand-constructed and a trained weight-tied transformer in the same Bellman–Ford architecture using exact state alignment at every relaxation round, causal interventions on those states, and extrapolation to graph sizes or iteration counts beyond training.

That sentence is the maximum defensible Track A contribution statement. “Transformers learn shortcuts on graph tasks,” “trained shortest-path transformers do not implement textbook shortest path,” and “OOD accuracy reveals whether the algorithm was learned” are not novel claims.

## C2 — probing exact intermediate algorithmic state

### Verdict: SETTLED

Liang and Pan, [*Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks* (arXiv:2602.00449)](https://arxiv.org/abs/2602.00449), define tasks with exact iterative state sequences and train linear probes at each layer/latent position to decode the state \(s_t\). They combine the probes with logit-lens, attention, and activation-patching evidence to distinguish full step-by-step rollout from compressed shortcuts. That is the claimed method, not merely adjacent use of intermediate supervision.

Cabannes et al., [*Iteration Head: A Mechanistic Study of Chain-of-Thought* (arXiv:2406.02128; DOI:10.52202/079017-3463)](https://doi.org/10.52202/079017-3463), use controlled iterative tasks with known state transitions, identify an “iteration head,” and validate the proposed mechanism with interventions. This is additional prior art for testing learned iterative computation against known algorithmic trajectories.

The stronger warning is also published. Darade and Thorat, [*Represented Is Not Computed: A Causal Test of Candidate Algorithmic Intermediates in a Transformer* (arXiv:2605.22488)](https://arxiv.org/abs/2605.22488), show that candidate intermediate values can be linearly decodable while causal tests reject the hypothesized staged computation. Therefore:

- alignment of a probe with \(d^{(t)}\) establishes **decodability**, not use;
- a successful Track A result needs causal interchange/patching or an equivalent intervention;
- presenting round-state probes themselves as a methodological contribution would be indefensible.

The neural-algorithmic-reasoning literature also makes exact step-level Bellman–Ford targets routine rather than exotic: Veličković et al., [*Neural Execution of Graph Algorithms* (ICLR 2020 OpenReview URL)](https://openreview.net/forum?id=SkgKO0EtvS), train models to imitate individual execution steps of graph algorithms including Bellman–Ford. That is supervision rather than post-hoc probing, but it removes novelty from the intermediate-state target.

## C3 — constructed graph algorithms as interpretability ground truth

### Verdict: PARTIAL

The motivation is established; only the exact benchmark instance may be new.

Gupta et al., [*InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques* (arXiv:2407.14494)](https://arxiv.org/abs/2407.14494), provide 86 semi-synthetic transformers with known ground-truth circuits and use them to evaluate circuit-discovery methods. The paper explicitly argues that raw Tracr programs are distributionally unrepresentative: their weights and activations are unusually sparse, they lack superposition, and they use only a small fraction of the activation space. InterpBench's entire premise is that ground-truth transformer circuits beyond unmodified Tracr are useful.

InterpBench also limits how much can be claimed for the present artifact. Its contribution is not merely “a model whose circuit is known”; it supplies a dataset of models, preserved realistic behavior, known circuits, tasks, and quantitative evaluation of interpretability methods. One hand-constructed Bellman–Ford model is a case study unless it is used to discriminate between methods or falsify an interpretability claim.

On the graph side, MINAR ([OpenReview:3PbQUU9rgu](https://openreview.net/forum?id=3PbQUU9rgu)) already treats neural graph algorithms as mechanistic-interpretability objects and recovers compact Bellman–Ford circuitry from trained GNNs. Shin et al. ([OpenReview URL](https://openreview.net/pdf?id=0WEuCiokZq)) **[ABSTRACT ONLY]** do mechanistic interpretation of tokenized graph transformers with probing and patching on shortest-path distance. These papers make “graph algorithms are useful interpretability testbeds” prior art.

The construction baseline is also crowded. de Luca and Fountoulakis, [*Simulation of Graph Algorithms with Looped Transformers* (arXiv:2402.01107)](https://arxiv.org/abs/2402.01107), construct looped transformers for graph algorithms including Dijkstra, BFS, DFS, and strongly connected components, and explicitly discuss finite-precision/softmax limitations. The present Bellman–Ford implementation may still have useful engineering differences, but hand-compiling graph algorithms into looped transformers is not new.

**Exact remaining gap:** I found no benchmark that uses a hand-constructed, weight-tied graph transformer with complete per-register and per-round ground truth to compare interpretability methods or to measure circuit correspondence against a trained model of the same architecture.

This gap supports a benchmark/evaluation contribution only if the artifact is actually used to test interpretability methods. “Useful ground truth beyond Tracr” by itself is settled.

## C4 — softmax logit scale versus ReLU folds

### Verdict: SETTLED

The proposed claim combines three different mathematical models. They have different answers.

#### 1. Real-arithmetic exact softmax

At every finite temperature and with finite logits, every softmax coefficient is strictly positive. A single softmax-weighted average of unequal scalar candidates therefore cannot equal the strict minimum exactly: non-minimal candidates retain positive weight. Exact hard selection requires a limiting operation, an explicit hardmax, infinite score separation, or some additional discrete/cancellation mechanism.

Thus “exact selection requires \(\Omega(\log n)\)” is not the correct real-arithmetic theorem. A finite \(\Theta(\log n)\) gap gives controlled approximation, not exactness.

#### 2. Approximate sharp selection

Veličković et al., [*Softmax is not Enough (for Sharp Size Generalisation)* (PMLR 267, ICML 2025)](https://proceedings.mlr.press/v267/velickovic25a.html), prove the relevant statement. With logits bounded between \(m\) and \(M\), each softmax coefficient is at most

\[
\frac{\exp((M-m)/\theta)}{n}.
\]

Keeping a selected coefficient bounded below as \(n\) grows therefore requires score spread divided by temperature to grow as \(\Omega(\log n)\). Their paper uses this to show dispersion of fixed-vocabulary softmax attention under size extrapolation and relates sharpness to large query/key norms.

Hu et al., [*Transformer Approximations from ReLUs* (arXiv:2604.24878)](https://arxiv.org/abs/2604.24878), give a quantitative sufficient condition for hardmax approximation: with a unique maximum separated by gap \(\Delta\), inverse temperature on the order of

\[
\frac{\log(n-1)+\log(1/\varepsilon)}{\Delta}.
\]

The \(\Omega(\log(n/\varepsilon))\) phenomenon is therefore published prior art.

#### 3. Finite-precision implementation

This repository obtains exact one-hot attention by choosing a very large fixed score gap so that non-selected float32 exponentials underflow to zero. Once they are exactly zero, adding more candidates does not change the selected result; a fixed underflow-inducing gap can remain exact independently of \(n\), until another implementation limit is hit.

That behavior is neither the real-arithmetic theorem nor the ordinary approximation theorem. It depends on the exact exponential kernel, dtype, subnormal/flush-to-zero behavior, score shifting, and device. Under this repository's stated arithmetic model, the proposed \(\Omega(\log n)\) lower bound is contradicted by its own construction.

#### 4. ReLU min

The identity

\[
\min(a,b)=a-\operatorname{ReLU}(a-b)
\]

is exact with constant-sized weights. Liu et al. ([arXiv:2210.10749](https://arxiv.org/abs/2210.10749), Appendix C.4) explicitly note that a pairwise ReLU reduction computes a finite-set minimum in logarithmic depth with constant weight norm. The implementation can instead fold sequentially, trading depth or recurrent steps for fixed width. Neither identity is new.

The viable statement is only:

> Ordinary finite-temperature softmax needs \(\Theta(\log(n/\varepsilon))\) score scale for \(\varepsilon\)-sharp selection, whereas an exact pairwise ReLU min uses bounded coefficients and a reduction tree or repeated fold.

That statement is already known. The stronger word “exact” does not create a new result; it makes the proposed result wrong or model-dependent.

## C5 — depth versus precision for exact reductions

### Verdict: PARTIAL

There is a collection of neighboring results, not the proposed theorem.

- Liu et al. ([arXiv:2210.10749](https://arxiv.org/abs/2210.10749), Appendix C.4) give the constructive upper bound: pairwise ReLU reduction computes min with \(O(\log n)\) depth and constant weight norm. They describe logarithmic depth as conjecturally unimprovable in that construction, not as a proved transformer lower bound.
- Merrill and Sabharwal, [*The Parallelism Tradeoff: Limitations of Log-Precision Transformers* (arXiv:2207.00729; DOI:10.1162/tacl_a_00562)](https://doi.org/10.1162/tacl_a_00562), characterize constant-depth log-precision transformers through uniform threshold-circuit complexity. This is a general depth/precision result, but it does not prove the desired min/max separation. In particular, min/max of polynomially many log-bit integers is compatible with threshold-circuit computation.
- Sanford, Hsu, and Telgarsky, [*Transformers, Parallel Computation, and Logarithmic Depth* (arXiv:2402.09268; PMLR 235)](https://proceedings.mlr.press/v235/sanford24a.html), relate logarithmic transformer depth to parallel algorithms and establish conditional depth lower bounds for harder problems. It does not establish that exact scalar min/max itself needs logarithmic transformer depth.
- de Luca and Fountoulakis ([arXiv:2402.01107](https://arxiv.org/abs/2402.01107)) discuss how graph-algorithm simulations trade loop iterations, attention sharpness, and finite precision, but again do not supply the claimed exact min/max lower bound.

An unrestricted claim is implausible. Width may grow with \(n\); multiple heads can parallelize comparisons; hard attention changes the problem; threshold-like nonlinearities can encode wide circuits; and arbitrary real weights can hide unbounded information. “Precision” might refer to parameter bit length, activation bit length, logit magnitude, temperature, or machine dtype. Those are not interchangeable resources.

**Exact remaining gap:** Under a fully specified transformer model with explicit bounds on width, heads, parameter norms, attention type, number representation, and activation precision, prove a lower bound excluding constant-depth, constant-precision exact min/max while retaining a logarithmic-depth bounded-precision construction.

I found no paper proving that sentence. It is not ready to serve as a research claim because the model restrictions needed to make it true are currently unspecified, and existing threshold-circuit characterizations give reasons to expect counterexamples.

## Closest prior work by track

### Track A: trained model versus constructed Bellman–Ford circuit

**Most threatening paper:** Cohen et al., [*Spectral Journey: How Transformers Predict the Shortest Path*](https://openreview.net/pdf?id=mAEsGkITgG).

It already trains a transformer on shortest paths and mechanistically concludes that the learned solution is not Dijkstra/Bellman–Ford but a spectral shortcut. That is the headline result Track A currently appears to seek.

The honest distinction is narrow: their model is a shallow decoder-only transformer rather than a weight-tied recurrent Bellman–Ford architecture; they do not have an exact constructed comparator in the same hypothesis class; they do not align every recurrent round to exact Bellman–Ford state; and their structural generalization does not establish extrapolation to larger graph sizes or more relaxation rounds.

MINAR ([OpenReview:3PbQUU9rgu](https://openreview.net/forum?id=3PbQUU9rgu)) is nearly as threatening because it recovers trained Bellman–Ford circuitry and exposes hidden shortcuts, but it studies message-passing GNNs rather than transformers.

### Track B+C: exact reduction and precision

**Most threatening paper:** Veličković et al., [*Softmax is not Enough (for Sharp Size Generalisation)*](https://proceedings.mlr.press/v267/velickovic25a.html).

It already proves the useful logit-spread-versus-\(n\) statement for sharp softmax. What distinguishes this repository is not a stronger theorem: it uses machine underflow for bit-exact selection and a ReLU fold for exact min. Those are different implementation semantics, and they invalidate rather than strengthen the proposed \(\Omega(\log n)\) exactness claim.

## Reframing options

### Track A

Only one defensible reframe survived:

> In a controlled weight-tied transformer architecture capable of exact Bellman–Ford, compare a hand-constructed circuit with circuits found by training, using exact round-state labels plus causal interventions to determine whether training recovers relaxation, a compressed algorithm, or a shortcut under size extrapolation.

This is narrower than the current Track A and should not claim that probing, shortcut learning, graph OOD failure, trained shortest-path mechanistic analysis, or graph-algorithm ground truth is new.

A second possible output is an interpretability benchmark, but only if it evaluates multiple methods against the known round/register circuit and demonstrates failures not exposed by existing InterpBench tasks. Releasing one known circuit without such evaluation is not enough.

### Track B+C

There is no credible theory reframe on the present evidence. The clean replacement is a technical note separating:

1. impossibility of exact hard selection by finite-logit real softmax;
2. the published \(\Theta(\log(n/\varepsilon))\) scale for approximate sharpness;
3. implementation-specific exactness from float underflow; and
4. exact ReLU reduction with depth/recurrent-step costs.

That clarification could improve the artifact's documentation. It is not a novel research contribution.

## Coverage gaps

### Access limitations

- I could retrieve only the abstract/search text for Shin et al., *Discovering Mechanisms in Tokenized Graph Transformers*; its evidence is therefore marked **[ABSTRACT ONLY]** and was not used to claim details beyond the abstract.
- Some OpenReview-hosted PDFs returned an access challenge through direct download. I used only text actually returned by the fetched paper endpoint and did not infer results from inaccessible titles.
- I did not have subscription citation-index access to Scopus or Web of Science. The search used arXiv, PMLR, NeurIPS proceedings, ACL/DOI pages, OpenReview, and ordinary web search, followed by backward/forward citation leads visible there.
- I did not exhaust non-English literature, theses, patents, or unpublished workshop posters without indexed text.
- Confidence is lowest on C5. Terminology varies among “precision,” “temperature,” “weight magnitude,” “bit complexity,” “circuit depth,” and “sharp attention,” making a directly relevant lower bound easy to miss. That uncertainty is why C5 is **PARTIAL**, not **OPEN** or **SETTLED**.
- Confidence is next-lowest on the exact benchmark gap in C3. Several 2025–2026 interpretability benchmarks are recent, and a graph-program instance may exist in supplementary material that search indexing does not expose.

### Actual search queries

The following are the actual query strings used, including overlap:

```text
site:arxiv.org transformer shortest path intermediate states probing algorithm iterations Bellman Ford trained
site:openreview.net transformer graph algorithm per-step probing intermediate state shortest path
"Transformers Learn Shortcuts to Automata"
Bachmann Nagarajan path star transformer shortest path next token prediction
"per-iteration" linear probe transformer algorithmic state
mechanistic interpretability algorithmic transformer probe each layer ground truth intermediate state
"Do Latent-CoT Models Think Step-by-Step?"
InterpBench mechanistic interpretability benchmark ground truth circuits paper arxiv
"ground truth" mechanistic interpretability graph algorithm transformer benchmark
hand-coded transformer graph algorithm interpretability ground truth
"MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning"
trained transformer shortest path size generalization out-of-distribution graph algorithm
transformer shortest path OOD graph size fails learned shortcut paper
site:arxiv.org transformer softmax maximum approximation log n temperature precision exact max min theorem
transformer theorem depth precision tradeoff exact max min reduction constant depth log precision
site:arxiv.org "depth-precision tradeoff" transformer
site:arxiv.org transformer log precision max min TC0 exact reduction
site:arxiv.org attention precision depth tradeoff max min transformer reduction
```

I also followed citation trails from arXiv:2210.10749, arXiv:2407.14494, arXiv:2402.01107, OpenReview:3PbQUU9rgu, the PMLR page for *Softmax is not Enough*, and the NeurIPS paper *How Far Can Transformers Reason?*

## Notes

- Linear probe success must not be reported as evidence that the trained model *uses* Bellman–Ford state. The minimum credible standard is exact state alignment plus a causal intervention.
- Any future precision statement must specify arithmetic semantics before stating a bound. Real softmax, float32 softmax with underflow, hard attention, and an abstract bounded-bit machine are different models.
- These verdicts address novelty of the proposed follow-on research, not correctness or pedagogical value of the existing constructed artifact.
