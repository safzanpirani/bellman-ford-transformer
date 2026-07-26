# T2 matched-budget ablations

Status: needs-triage
Lifecycle: completed

From the saturated T1 recipe, independently ablate data freshness, width,
LayerNorm, supervision, budget, batch size, and learning rate where it differs
from v1. Record G1 compatibility and do not assign causality without a matched
comparison.

## Resolution

- Completed the matched budget and LayerNorm ablations.
- Data freshness, constructed-minimal width, final-only supervision, and batch
  size were explicitly left untested under the user-approved time cut.
- The LayerNorm-free width-128 condition passed G1 and saturated near 97%
  graph exactness.
