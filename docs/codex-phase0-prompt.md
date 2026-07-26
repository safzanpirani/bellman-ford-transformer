# Handoff prompt — Phase 0 novelty triage

Paste everything below the line into Codex, running in the repo root.

---

You are doing a **prior-art triage** for a research project. This is a
literature task, not a coding task. Do not write, refactor, or run any
model/training code. The only file you produce is `docs/prior-art.md`.

## Context

Read `README.md` and `RESEARCH.md` in this repo first — they describe the
artifact and the two research tracks. In short: we hand-constructed (not
trained) a weight-tied transformer whose forward pass exactly executes the
Bellman-Ford shortest-path algorithm, verified bit-exact against networkx, with
exact per-round intermediate state available as ground truth.

We already know the *construction itself* is not novel — it follows from
RASP/Tracr and looped-transformer expressivity work. Your job is to determine
whether the two proposed follow-on contributions are novel, and to kill them if
they are not.

## Your stance

**Adversarial. Your default assumption is that each claim below is already
published.** Your job is to find the paper that kills it. Only if a genuine
search fails to find one should you mark a claim open.

A wrong "this is novel" verdict costs us eight weeks of wasted experiments. A
wrong "this is done" verdict costs us one follow-up question. The asymmetry
should shape how you write every verdict.

Do not be encouraging. Do not soften findings. If everything here is already
done, say so plainly and recommend we stop.

## Claims to adjudicate

| # | Claim |
|---|---|
| C1 | Trained transformers do **not** learn genuine iterative relaxation on shortest-path / graph tasks — they learn shortcuts that fail out-of-distribution |
| C2 | Probing a trained model per-iteration against **exact intermediate algorithmic state** is an established method for testing "did it learn the algorithm" |
| C3 | Hand-constructed graph-algorithm models are useful mech-interp ground truth **beyond what Tracr already provides** |
| C4 | Exact min over n candidates via a softmax attention head requires logit scale Ω(log n); ReLU-based folds do not |
| C5 | There is a known depth-vs-precision trade-off for **exact** reductions (min/max) in transformers |

C1–C3 gate Track A (empirical: does a trained model learn our circuit).
C4–C5 gate Track B+C (theory: exactness requires bounded degree or unbounded
precision).

## Where to look

Search these clusters. For each, I have named papers I believe are relevant —
**verify each one actually exists and says what I claim before citing it**, and
treat my descriptions as unverified hearsay:

1. **Expressivity / construction** (establishes the baseline of what is not
   novel): RASP (Weiss, Goldberg, Yahav ~2021); Tracr (Lindner et al. ~2023)
   plus any critiques of it; Looped Transformers as Programmable Computers
   (Giannou et al. ~2023); Pérez et al. on Turing-completeness; looped
   transformers for length generalization (~2024).

2. **Does training find algorithms** (decides C1, C2): the CLRS benchmark and
   neural-algorithmic-reasoning line (Veličković and collaborators);
   Transformers Learn Shortcuts to Automata (Liu et al. ~2023); algorithmic
   alignment (Xu et al. ~2020); GNNs are Dynamic Programmers (Dudzik &
   Veličković ~2022); Transformers meet NAR (Bounsi et al. ~2024); path-finding
   failure work (Bachmann & Nagarajan on next-token-prediction pitfalls; maze
   navigation papers). **This cluster is the most likely to kill Track A.**

3. **Ground-truth interpretability** (decides C3): InterpBench and SIIT;
   critiques showing Tracr-compiled models are distributionally unrepresentative;
   circuit-discovery evaluation (ACDC and successors); SAE evaluations that
   needed ground truth.

4. **Precision theory** (decides C4, C5): Merrill & Sabharwal on log-precision
   transformers and TC⁰; Hahn's impossibility results; saturated/hard attention
   analyses; Chiang & Cholak; Strobl et al.'s survey on transformer formal
   language results.

Go beyond this list. It is a starting point, not a boundary. Follow citation
chains — check who cites the central papers, since the killing paper is often a
2024–2026 follow-up rather than the original.

## Verification rules — non-negotiable

Fabricated citations make this deliverable worse than useless, because we would
act on them.

- Every cited paper needs a **verifiable identifier**: arXiv ID, DOI, or a URL
  you actually fetched. No identifier → do not cite it.
- If you cannot access a paper's contents and are going on title/abstract alone,
  mark it **[ABSTRACT ONLY]**. If you are recalling it from memory without
  fetching anything, mark it **[UNVERIFIED — FROM MEMORY]** and treat it as a
  lead, not evidence.
- Never infer a paper's findings from its title. If the abstract does not state
  the finding, say the finding is unconfirmed.
- If web access is unavailable in your environment, **stop and say so
  explicitly** at the top of your output rather than producing a
  memory-only review presented as a literature search. A clearly-labelled
  memory-based lead list is acceptable; a memory-based review dressed up as
  verified is not.

## Verdict taxonomy

For each claim, exactly one:

- **SETTLED** — a specific paper already establishes this. Cite it. The
  corresponding track dies or must be substantially reframed.
- **PARTIAL** — adjacent work exists but leaves a specific gap. State the gap in
  one sentence, precisely enough that we could write the contribution statement
  from it.
- **OPEN** — genuine search found nothing. Say what you searched so we can judge
  how much to trust this.

Be sparing with OPEN. It is the verdict most likely to be wrong.

## Deliverable: `docs/prior-art.md`

Structure it as:

1. **Bottom line** (≤10 lines). Per-track go/no-go: does Track A survive? Does
   Track B+C survive? If both die, say so first, before any detail.
2. **Verdict table** — claim, verdict, killing/nearest paper, one-line reason.
3. **Per-claim sections** — the evidence, with citations and identifiers, and
   for PARTIAL verdicts the exact gap.
4. **Closest prior work** — the single most threatening paper to each track, and
   an honest paragraph on what distinguishes our proposal from it. If nothing
   meaningful distinguishes it, say that.
5. **Reframing options** — if a track is SETTLED, is there a nearby question
   that is still open? Concrete suggestions, or state that there is none.
6. **Coverage gaps** — what you could not access, what you did not search, and
   where you are least confident. Include your actual search queries.

## Explicitly out of scope

Do not design experiments, write training code, modify anything in
`bf_transformer/`, `tests/`, or `viz/`, or start implementing any track. If you
find yourself with an opinion about the experimental design, put it in a short
final "notes" section and leave it there.

Do not update `RESEARCH.md`. We will revise it after reading your findings.
