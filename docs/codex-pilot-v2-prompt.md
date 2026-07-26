# Handoff prompt — pilot v2 (retry after mis-specified gate)

Paste everything below the line into Codex, running in the repo root.

---

Pilot v1 is complete and its conclusion stands: it answered nothing. You are
running v2 to actually test the question. Read `docs/pilot-results.md`,
`docs/pilot-preregistration.md`, `RESEARCH.md`, and `train/` first.

**Commit all current uncommitted work before starting.** v1 is a legitimate
negative-result trail and must be preserved verbatim — do not edit
`docs/pilot-results.md` or `docs/pilot-preregistration.md`. v2 gets new files.

## What went wrong in v1 (diagnosis — verify it before acting on it)

v1 was not a training collapse. It was a mis-specified gate plus a weak setup.

1. **G2 was unreachable by construction.** The loss is `smooth_l1` regression
   over continuous outputs, but G2 demanded 90% graph-exact / 98% node-exact.
   A continuous readout essentially never lands on exact integers, so no amount
   of successful training could have passed. The threshold was frozen before any
   evidence that it was attainable.
2. **Step-0 selection was an artifact.** A residual network at small init is
   approximately the identity, and in this encoding the identity emits the
   *initial* `DIST` — 0 at source, `INF_B` elsewhere — which is exactly correct
   for every unreachable node (55.7% of them). So step 0 scored 25.2%
   graph-exact for free, training smeared INF into finite values, and
   lexicographic selection on exact-match picked step 0 in all 12 runs.
3. **The sentinel dominated the loss.** `INF_B = 161` at `N_MAX=16` is roughly
   4× any real distance, so Huber loss is driven by unreachable nodes.
4. **The corpus was tiny.** 4,096 fixed graphs, batch 32, 4,000 steps ≈ 31
   epochs — when `verify.random_graph` makes data free and unlimited.
5. **No LayerNorm at 15 rounds** is a hard optimisation problem. G1 parity
   requires LN-free; that tension is real and must be handled, not ignored.

Confirm each of these against the code and v1 artifacts before proceeding. If
you find the diagnosis wrong, say so and stop rather than building on it.

## The binding process lesson

**No threshold may be frozen until you have empirical evidence it is
attainable.** v1's pre-registration protected against motivated reasoning but
not against mis-specification, and that cost the whole run. v2 therefore runs a
feasibility ladder *before* the pre-registration is written.

---

## Phase 1 — Fix loss, metric, and data (no architecture change)

**Preserve G1.** Do not add a classification head or otherwise change the model
interface — that would break constructed-weight parity, which is the asset that
makes this experiment identifiable at all. Keep the continuous `DIST` readout.
Re-run the G1 test after any change and confirm it still passes exactly.

1. **Metric.** Define exact-match as `round(prediction) == target`, on the
   grounds that integer edge weights give integer distances. Justify this in
   writing. This makes exact-match attainable by a regression readout.
2. **Reachability.** Add a separate binary reachability output and remove the
   `INF_B` sentinel from the regression target, so the distance loss is computed
   only over reachable nodes. Report reachability accuracy separately.
   Verify the constructed model still satisfies G1 under this split — the
   constructed circuit already encodes reachability as `DIST ≥ INF_B − 0.5`, so
   a fixed thresholding readout should reproduce it exactly.
3. **Data.** Sample fresh graphs every batch from `verify.random_graph` rather
   than cycling a fixed corpus. Keep evaluation sets fixed and seeded.
4. **Checkpoint selection.** Select on a metric that is not degenerate at init.
   Report explicitly what step-0 scores under the new metric — if the identity
   map still scores well, the metric is still wrong.

## Phase 2 — Feasibility ladder (HARD GATE, before any pre-registration)

Establish that *some* configuration learns the task at all. Run in order,
smallest first, and stop climbing as soon as a rung fails:

1. `n ≤ 4`, per-round supervision, tied, one seed. Easiest possible case.
2. `n ≤ 8`, per-round supervision, tied, one seed.
3. `n ≤ 8`, final-distance supervision, tied, one seed.

Tune freely at this stage — learning rate, steps, warmup, batch size, width.
Tuning here is legitimate because nothing is registered yet; record what you
tried.

**Gate F1.** If no rung reaches task competence, run one diagnostic before
giving up: initialise from the constructed weights perturbed by small noise and
train. If the model *diverges away* from the constructed circuit, then the
solution is representable but not reachable by gradient descent from a nearby
point — report that. **"Representable but not learnable" is a legitimate and
interesting finding**, and is the honest fallback if the ladder fails.

**Gate F2.** If the LN-free model cannot be trained but an LN variant can, do
not silently switch. Report both. The gap between what the architecture can
*represent* (LN-free, G1-verified) and what it can *learn* (LN required, G1
broken) is the most interesting result available from this pilot — treat it as
a finding, not an obstacle.

**Time box:** if the ladder has not produced a competent model within a
reasonable budget, stop and report. Do not grind. A clean "not learnable under
these conditions, here is what was tried" is worth more than an exhausted
search.

## Phase 3 — Pre-registration v2

Only after the ladder succeeds. Write `docs/pilot-v2-preregistration.md`
containing everything v1's did, plus, for **every** numeric threshold, a
one-line citation of the ladder run demonstrating it is attainable. A threshold
with no feasibility evidence may not be registered.

Commit before Phase 4. Do not edit afterwards.

## Phase 4 — The grid

Same 2×2×3 as v1: tied × untied, final × per-round supervision, ≥3 seeds, using
the ladder-validated recipe. Then measurements in the registered order —
behaviour by n and hop count, the full k×j probe matrix with random-init and
shuffled-label baselines, attention agreement, iteration ablation.

Gates G2 and G3 carry over unchanged in spirit: no mechanistic claim about a
model that fails competence, and no causal claim without a probe diagonal.
Probes show decodability, not use (arXiv:2605.22488).

## Phase 5 — Report

Write `docs/pilot-v2-results.md`: bottom line first (does recurrence change the
answer — yes / no / inconclusive); results with seed variance and CIs; full
probe matrices with baselines; a short section comparing v1 and v2 and what the
v1 mis-specification cost; deviations from the v2 pre-registration; and a
go/no-go with an honest venue estimate.

Note for calibration: even a clean positive result here is workshop-tier, since
*Spectral Journey* (arXiv:2502.08794) owns the headline. Do not inflate the
recommendation.

---

## Rules

- `bf_transformer/` and `tests/` stay green — all existing tests must pass. Run
  the suite before every commit.
- **G1 is non-negotiable.** Any change that breaks constructed-weight parity
  invalidates the experiment. Re-verify it after every architecture-adjacent
  change.
- Do not edit v1 artifacts. v2 is additive.
- A negative result is a result. If nothing learns, say so in the bottom line.
- No motivated language. No silent scope cuts — log every dropped condition,
  seed, or baseline.
- Stop at the gates. F1, F2, G2, G3 are stop conditions.
- If the diagnosis above turns out to be wrong, stop and report rather than
  forcing the plan onto contrary evidence.

## Out of scope

The blog post; Track B+C theory; new algorithms; general benchmark or training
frameworks; `viz/` beyond regenerating `trace.json` if a config change requires
it; any architecture change that breaks G1.
