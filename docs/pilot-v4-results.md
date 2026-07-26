# Pilot v4 results

## Bottom line

V4 establishes that the task is trainable in a G1-compatible architecture on \(n \le 4\).
The widened, LayerNorm-free model saturated at 96.68% validation graph exactness and 96.97% test graph exactness.
The exact constructed weights embed in that model with 100% exact outputs, zero objective, and zero gradient.
The best favourable LayerNorm run saturated at 97.46% validation graph exactness; LayerNorm is therefore not necessary under this recipe.
An otherwise matched 4,000-step run reached only 50.20% graph exactness and ended at its best checkpoint: budget causally matters under the selected recipe, but its role in exact v1 remains unisolated.
V4 does **not** establish that budget was the only cause; fixed data, constructed-minimal width, final-only supervision, and batch size were not ablated under the four-hour cut.
The perturbation study supports ordinary optimizer-step drift, not an objective that rewards sacrificing exactness.
**Go at \(n \le 4\):** a validated G1-compatible training recipe exists; scaling it to the original \(N_{\max}=16\) experiment remains unvalidated.

## Status and scope

Every run in this report is diagnostic, not part of the deferred 2×2×3 experiment. All completed training runs use seed 17 and data seed 20260730. There is no seed variance or confidence interval: v4 tested existence and failure modes, not a population claim.

The user imposed a final four-hour cap after the initial T1 work. We jointly retained one full G1-relevant 80,000-step run plus analysis and reporting. The following planned conditions were explicitly dropped: the \(n \le 8\) climb, fixed-4,096 data, constructed-minimal width, final-only supervision, and batch size 32. They are marked untested below and are not approximated with shorter runs.

The complete per-evaluation histories for every completed run are in [`pilot-v4-curves.json`](pilot-v4-curves.json). Exactness values there are fractions; the tables below use percentages.

## Saturation criterion

A run counts as saturated only when its terminal learning-rate floor is zero and, for graph exactness, node exactness, and reachable non-source exactness, the final two 10,000-step windows satisfy all three conditions:

- final-window best improves by less than 1 percentage point over the preceding window;
- the absolute least-squares slope is less than 1 percentage point per 10,000 steps;
- the final-window range is less than 2 percentage points.

This is a curve criterion, not a competence threshold. The old 90%/98% v1 gate is reported nowhere as a stopping rule.

## T1 — existence and saturation

### Completed search

| Diagnostic recipe | G1 status | Horizon | Best step | Validation graph / node / RNSE | Test graph / node / RNSE | Saturation verdict |
|---|---|---:|---:|---:|---:|---|
| width 128, LayerNorm, LR \(3\times10^{-4}\) | non-G1 | 60k | 55k | 96.48 / 98.60 / 96.20 | 95.31 / 98.26 / 95.41 | **Budget-limited**: graph and RNSE late-window gains were 1.17 and 1.27 pp |
| width 128, LayerNorm, LR \(3\times10^{-4}\) | non-G1 | 80k | 70k | **97.46 / 99.14 / 97.64** | **96.97 / 98.92 / 97.14** | **Saturated** |
| width 256, LayerNorm, LR \(3\times10^{-4}\) | non-G1 | 60k | 50k | 94.92 / 98.07 / 94.75 | 94.73 / 98.03 / 94.80 | **Saturated** |
| width 128, LayerNorm, LR \(10^{-3}\) | non-G1 | 60k | 55k | 88.09 / 95.02 / 86.41 | 86.33 / 94.55 / 85.62 | **Budget-limited**: graph and RNSE were still rising |

RNSE means reachable non-source exactness, the subset that cannot be obtained for free from the identity-like initialization.

The best observed saturated \(n \le 4\) ceiling was the width-128 LayerNorm run: 97.46% validation graph exactness and 96.97% test graph exactness. This is a ceiling for the tested seed and recipes, not a claim of global optimality.

For this seed, the width-256 condition falsifies the simple claim that more width monotonically fixes optimization: it saturated 2.54 pp below width 128 on validation graph exactness. It does not test whether width 128 is easier than the constructed-minimal model.

The clean LR \(10^{-3}\) rerun was worse but still improving at 60k. It is budget-limited, not evidence for a lower asymptotic ceiling.

### Selected curves

| Step | LayerNorm graph | LayerNorm node | LayerNorm RNSE | No-LN graph | No-LN node | No-LN RNSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32.03 | 63.32 | 0.00 | 32.03 | 63.32 | 0.00 |
| 10k | 79.10 | 90.43 | 74.64 | 75.00 | 89.10 | 72.28 |
| 20k | 86.72 | 94.82 | 86.05 | 79.88 | 91.36 | 79.17 |
| 30k | 92.77 | 97.14 | 92.21 | 81.25 | 91.63 | 79.53 |
| 40k | 94.34 | 98.01 | 94.57 | 89.26 | 95.42 | 88.22 |
| 50k | 96.29 | 98.74 | 96.56 | 94.14 | 97.61 | 93.48 |
| 60k | 96.48 | 98.80 | 96.92 | 95.31 | 98.27 | 95.29 |
| 70k | 97.46 | 99.14 | 97.64 | 96.48 | 98.60 | 96.20 |
| 80k | 97.27 | 99.07 | 97.46 | 96.68 | 98.74 | 96.56 |

The complete 1,000-step curves, including loss and MAE, are in the companion JSON.

### \(n \le 8\)

No \(n \le 8\) result is reported. A blocking remote run lost its SSH session at step 49k and died before writing `run.json`; it was excluded. Its detached replacement was intentionally stopped under the user-approved four-hour scope cut. Neither partial trajectory reached the terminal schedule, so neither can establish saturation or a ceiling.

## T2 — backward ablations

The ablation table is deliberately incomplete. “Untested” means no causal conclusion is drawn.

| Ingredient removed from saturated LayerNorm recipe | With ingredient | Without ingredient | Result | G1-compatible without? |
|---|---:|---:|---|---|
| Extended budget: 80k → 4k | 97.46 graph / 97.64 RNSE, saturated | 50.20 graph / 27.72 RNSE, best at final step | **Budget materially matters under this recipe.** The 4k run is budget-limited, not a ceiling. | No; both use LayerNorm |
| LayerNorm → no LayerNorm | 97.46 graph / 97.64 RNSE, saturated | 96.68 graph / 96.56 RNSE, saturated | **LayerNorm is not required.** Difference is 0.78 pp graph and 1.09 pp RNSE on validation; test graph exactness is identical at 96.97%. | **Yes** |
| Fresh graphs → fixed 4,096 | completed fresh run | untested | One fixed-data run was operationally interrupted at 27k during an aliasing error and excluded. | Would remain no |
| Width 128 → constructed-minimal | completed width-128 run | untested | No causal claim about capacity. | Depends on LayerNorm; minimal alone would still be non-G1 here |
| Per-round → final-only | completed per-round run | untested | No causal claim about supervision. | Architecture would remain G1-compatible if no-LN |
| Batch 128 → batch 32 | completed batch-128 run | untested | No causal claim about batch size. | Unchanged |
| LR → v1 value \(3\times10^{-4}\) | selected recipe already uses \(3\times10^{-4}\) | not applicable | LR is already matched to v1; no separate ablation exists. | Unchanged |

### What this says about v1

The supported statement is narrow: **4,000 steps were insufficient for the selected recipe, while 80,000 steps saturated near 97% graph exactness.** This experiment could have falsified a budget effect and instead found a 47.27 pp validation graph-exactness gap.

It does not prove that the v1 step budget was the sole cause of v1's roughly 18% post-step-0 graph exactness. The 4k selected-recipe run already reached 50.20%, so at least one other difference between v1 and the selected recipe also mattered. Which of fresh data, width, supervision, or batch size mattered remains untested.

LayerNorm absence is not an absolute optimization blocker: the widened no-LN model saturates at 96.68% graph exactness. Its interaction with constructed-minimal width remains untested.

## T3 — constructed-basin drift

### Measurements

| Measurement | Result |
|---|---|
| Objective at exact constructed weights | 0 |
| Gradient L2 / max absolute gradient at construction | 0 / 0 |
| Peak LR \(3\times10^{-5}\), zero floor | Validation returned to 100% graph exactness by the end; selected test graph exactness 99.80%; final loss \(7.06\times10^{-7}\) |
| Peak LR \(10^{-5}\), zero floor | Validation and test ended at 100% graph exactness; final loss \(7.58\times10^{-8}\) |
| Loss–exactness conflict | Across both logged trajectories, there were zero evaluation intervals where validation graph exactness fell while validation loss also fell |
| Parameter distance, LR \(3\times10^{-5}\) | 0.0355 initially after perturbation → 0.1257 finally |
| Parameter distance, LR \(10^{-5}\) | 0.0355 initially after perturbation → 0.0433 finally |
| Endpoint loss versus construction | Both endpoints have positive loss; construction has exactly zero |

### Verdict

The evidence supports the mundane optimizer-step explanation. The exact construction is a zero-loss, zero-gradient global minimum. With a zero-floor schedule, the original peak LR recovers after transient exactness dips, and the smaller peak LR stays exact after its early transient. Exactness declines were accompanied by higher validation loss, not lower loss.

The learned parameters do not converge back to the hand construction: both perturbation runs finish farther away in parameter space while recovering functionally exact behaviour. Parameter recovery is therefore the wrong success criterion.

The tested trajectories provide no evidence for the substantive alternative that descent reduces the measured objective while pushing predictions outside the half-unit exactness bin.

## D3 / G1 verdict

The widened no-LN model was loaded with the exact constructed weights before random-init training:

- validation and test graph, node, reachability, and RNSE exactness: 100%;
- supervised loss and MAE: 0;
- objective and gradient: 0;
- distance from constructed reference: 0.

The random-init run in the same architecture then saturated at:

- validation: 96.68% graph, 98.74% node, 96.56% RNSE;
- test: 96.97% graph, 98.85% node, 96.97% RNSE;
- 100% reachability accuracy.

Its final parameter distance from the construction is approximately 76,648, versus approximately 76,649 at initialization. It learned a functionally successful solution without moving materially toward the hand-built parameter vector.

**D3 verdict: yes, a G1-satisfying configuration is learnable.** The validated recipe is: tied recurrence, \(n_{\max}=4\), width 128, no LayerNorm, per-round supervision, fresh graphs, batch 128, LR \(3\times10^{-4}\) with 200-step warmup and cosine decay to zero, 80,000 steps, seed 17.

## Operational deviations and exclusions

- A three-job same-GPU launch caused an OOM in the first LR \(10^{-3}\) attempt. That trace was excluded and rerun alone.
- `main-win` and `maints` appeared as distinct fleet names but resolve to the same physical machine and RTX 3090. A fixed-data run accidentally overlapped the \(n \le 8\) run until host UUIDs exposed the alias; the fixed run was stopped at 27k and excluded.
- The first \(n \le 8\) run used blocking remote execution and died at step 49k when SSH disconnected. It wrote no `run.json` and was excluded.
- The detached \(n \le 8\) rerun was stopped under the user-approved final four-hour cut.
- A zero-step G1 precheck produced 100% exactness and zero loss but exited nonzero because the runner requires a post-update checkpoint. A clean one-step, zero-LR precheck reproduced the result and is the reported check.
- No completed result, seed, or baseline was silently removed. Interrupted runs are not included in the companion curve file.

## Go / no-go and venue

**Go for pre-registering a narrow \(n \le 4\) recurrence comparison:** the primary no-LN architecture both contains the exact constructed circuit and is trainable from random initialization in that regime. The corresponding validity objection that killed v1–v3 is resolved. The original \(N_{\max}=16\), train-\(n \le 8\) experiment still needs a successful scale-validation run before it can be called valid.

**No claim yet about recurrence:** v4 did not run tied versus untied models, final-only supervision, probes, attention agreement, iteration ablation, or causal interventions. It therefore does not answer whether weight tying changes learned mechanism.

Even a clean future positive result remains workshop-tier because *Spectral Journey* owns the broader shortest-path shortcut headline. V4 alone is not publishable as a recurrence result. Its independent observation is the optimizer-drift characterization around an exact circuit, but that is currently one architecture, one graph regime, and one seed.

## Notes

If the project continues at \(n \le 4\), the next action is a new pre-registration using the validated no-LN recipe and empirically attainable thresholds derived from these curves. If the original \(N_{\max}=16\) grid remains the target, validate this recipe at that scale before pre-registering it. The dropped ablations should not be backfilled unless they change a concrete decision.
