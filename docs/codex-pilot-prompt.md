# Handoff prompt — corrections + recurrence pilot

Paste everything below the line into Codex, running in the repo root.

---

You are taking this project through its remaining work. Read `README.md`,
`RESEARCH.md`, and `docs/prior-art.md` first — the last of those is a prior-art
triage that already killed most of the original plan, and its conclusions are
binding. Do not re-litigate them.

## Where things stand

We hand-constructed a weight-tied transformer that executes Bellman-Ford
exactly (42 passing tests, bit-exact vs networkx). Prior-art triage then found:

- **Track B+C (precision theory) is dead.** The useful softmax result is
  published as *softmax is not enough* (arXiv:2410.01104). Our proposed "exact
  min needs Ω(log n) logit scale" claim is malformed: under real arithmetic
  finite-logit softmax is never exactly hard selection, and under float32
  underflow a *fixed* gap stays exact independently of n — our own construction
  contradicts the claim.
- **Track A's headline is taken.** *Spectral Journey* (arXiv:2502.08794) trains
  transformers on shortest paths and finds no dynamic programming — a spectral
  shortcut instead.
- But *Spectral Journey* used a shallow 2-layer decoder-only model, while
  looped transformers (arXiv:2409.15647) and sparsity-regularised GNNs
  (arXiv:2503.19173) *do* recover iterative/relaxation behaviour.

That tension is the one live question:

> **Does weight-tied recurrence decide whether a trained model learns
> relaxation rather than a shortcut?**

We have the exact constructed circuit in the same architecture as the
comparator, which is what makes this answerable.

## Definition of done

1. Documentation corrected (Phase 1).
2. A trainable twin that provably *can* represent the constructed solution
   (Phase 2 — hard gate).
3. A pre-registered, executed pilot answering the question above (Phases 3–4).
4. An honest report with a go/no-go for further work (Phase 5).

This is a **pilot**, not a paper. Target scope is small. If you find yourself
building infrastructure for experiments not listed here, stop.

---

## Phase 1 — Corrections (do first, it is cheap and certain)

1. In `README.md`, replace any claim that our exactness argument is a novel or
   general result. Add a short subsection distinguishing four genuinely
   different things, since we previously conflated them:
   - real-arithmetic softmax: never exactly hard selection at finite logits;
   - approximate sharpness: needs Θ(log(n/ε)) score spread (cite
     arXiv:2410.01104);
   - our float32 construction: exact via underflow at a *fixed* gap,
     independent of n, but dependent on dtype and flush-to-zero behaviour;
   - ReLU fold: exact at constant precision, costs depth.
   State plainly that (3) is an implementation property, not a theorem.
2. Cite the closest prior art for the construction itself:
   *Simulation of Graph Algorithms with Looped Transformers* (arXiv:2402.01107),
   plus Tracr and RASP. Our differentiators are the exactness discipline, the
   executable round-by-round verifier, and the tracer — not the construction.
3. Rewrite `RESEARCH.md` around the recurrence question and this pilot. Delete
   the Track B+C theory target; it is wrong.
4. Fix the MINAR citation in `docs/prior-art.md`: arXiv:2602.21442 does not
   resolve. Use OpenReview `3PbQUU9rgu` / github.com/pnnl/MINAR.

Commit Phase 1 separately before starting Phase 2.

---

## Phase 2 — Trainable twin and the validity gate

Build `train/model.py`: a trainable transformer using **the same tokenisation
and residual layout** as `bf_transformer` (sink token, node tokens, edge
tokens, the register map from `registers.py`).

Two architecture variants, identical parameter shapes:
- **tied** — one block applied R times (matches the construction);
- **untied** — R distinct blocks.

Set `N_MAX = 16` in the config used for training so the encoding has room for
size-extrapolation tests, even though training graphs will be small. Note this
changes `d_model` and `rounds`; that is expected and handled by the register map.

### Hard gate G1

Write a test that loads the **constructed** weights into the tied trainable
module and asserts it achieves zero loss / exact outputs on random graphs.

**If G1 fails, STOP and report.** Without it, "the trained model didn't learn
the circuit" is unfalsifiable — the circuit would not be in the hypothesis
space, and the entire pilot is vacuous. Do not proceed with a "close enough"
architecture.

Note the LayerNorm tension explicitly: the construction requires no LayerNorm
(it would rescale real distances). If you add normalisation for trainability,
G1 breaks. Resolve by keeping the primary variant LayerNorm-free; if you also
train a normalised variant, treat it as secondary and say so.

---

## Phase 3 — Pre-registration (before any results exist)

Write `docs/pilot-preregistration.md` **before running a single training job**,
containing:

- the exact conditions to be run (architecture × supervision × seeds);
- the metrics and how each will be computed;
- what result would count as "learned relaxation", what would count as
  "shortcut", and what would count as ambiguous — with thresholds, chosen now;
- your predictions for each condition.

Commit it before Phase 4. Do not edit it afterwards; record deviations in the
final report instead.

---

## Phase 4 — Run the pilot

**Data.** Reuse `verify.random_graph`. Train on n ∈ [2, 8]. Held-out test on
n ∈ [9, 16]. Stratify the test set by shortest-path hop count — that is the
axis on which a shortcut should fail.

**Conditions.** tied × untied, supervision ∈ {final distances only, per-round
targets}, ≥3 seeds each. That is 12 runs; they are small and cheap. The 3090 on
`maints` is far more than sufficient; CPU is likely fine.

### Gate G2

Every condition must reach high in-distribution exact-match before any
mechanistic claim is made about it. A model that cannot do the task has no
mechanism worth interrogating. If a condition fails, fix training or report it
as untrained — do not analyse it.

### Measurements, in this order

1. **Behavioural:** exact-match accuracy by n and by hop count, in-distribution
   and OOD.
2. **Per-round probes (the core result).** Train a linear probe at loop
   iteration k to predict exact reference state after j rounds
   (`verify.reference_states` gives these targets for free). **Report the full
   k×j matrix, not just the diagonal** — an off-diagonal or flat matrix is
   itself the finding. Mandatory baselines: random-init model, and
   shuffled-label probes. Without them, probe accuracy is uninterpretable.
3. **Attention-pattern agreement.** We know the exact ground-truth routing
   pattern. Measure per-head agreement with it.
4. **Iteration ablation.** Run the trained loop for fewer/more iterations than
   trained. Algorithm-like → graceful, monotone; shortcut-like → collapse.

### Gate G3

Only if a probe diagonal appears, proceed to causal intervention (patch along
the probe-identified direction; check outputs move as the algorithm predicts).
Probes alone establish *decodability, not use* — arXiv:2605.22488 shows
precisely this failure mode. Never report probe success as evidence the model
*uses* the state.

If neither tied nor untied shows round structure, that is a clean replication
of *Spectral Journey* in a new architecture. Stop, write it up as that, and do
not manufacture a more exciting story.

---

## Phase 5 — Report

Write `docs/pilot-results.md`:

1. **Bottom line first**, ≤10 lines: does recurrence change the answer? Yes / no
   / inconclusive.
2. Results per condition, with seed variance and CIs. Never a single run.
3. The full k×j probe matrices, with baselines.
4. Where you deviated from the pre-registration, and why.
5. **Go/no-go recommendation** for further work, with your honest estimate of
   the strongest venue this could reach. "Workshop paper" and "not publishable"
   are acceptable and expected answers.

---

## Rules

- **Do not break the existing artifact.** `bf_transformer/` and `tests/` are
  verified; all 42 tests must stay green. Additive changes only. Run the suite
  before every commit.
- **A negative result is a result.** If training finds shortcuts everywhere,
  say so in the bottom line. Do not bury it, and do not go looking for a
  configuration that produces a nicer story.
- **No motivated language.** Avoid "promising", "encouraging", "as expected".
  State what happened.
- **No silent scope cuts.** If you drop a condition, a seed, or a baseline, log
  it explicitly in the report. Silent truncation reads as full coverage.
- **Stop at the gates.** G1, G2, G3 are stop conditions, not suggestions.
- If you are blocked or a gate fails, stop and report rather than working
  around it. A clean stop is worth more than a result we cannot trust.

## Out of scope

Do not: write the blog post (that needs the author's voice — a bullet outline
in the final report is enough); pursue any Track B+C theory; build a general
benchmark or training framework; add new algorithms; touch `viz/` beyond
regenerating `trace.json` if a config change requires it.
