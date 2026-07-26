# Pilot v4 diagnostic cycle

V4 corrects the premature v3 stop without revising any v1–v3 artifact. It has
three bounded tasks:

1. obtain an evidence-backed saturation curve on \(n \leq 4\), then climb to
   \(n \leq 8\);
2. run matched-budget, one-factor ablations from the saturating recipe; and
3. distinguish objective/metric drift from non-zero-step-size escape around
   the exact constructed circuit.

The cycle ends with `docs/pilot-v4-results.md`. It does not run the registered
2×2×3 grid.

Final scope: completed under a user-approved four-hour cut. The \(n \leq 8\)
climb and four secondary ablations were explicitly dropped; the retained
G1-compatible run, saturation analysis, drift study, and report were completed.
